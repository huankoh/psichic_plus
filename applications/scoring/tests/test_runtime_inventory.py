import subprocess
import sys


def test_installed_inventory_covers_late_packages_namespaces_and_exact_paths(tmp_path):
    script = r'''
import importlib
import importlib.metadata as metadata
import importlib.util
import json
import platform
import sys
from pathlib import Path
from applications.scoring import runtime

site = Path(sys.argv[1])
sys.path.insert(0, str(site))
sources = {
    'late_package/__init__.py': '',
    'late_package/unused.py': 'value = 1\n',
    'late_package/tests/helper.py': 'value = 2\n',
    'shared_namespace/first.py': 'value = 3\n',
    'shared_namespace/second.py': 'value = 4\n',
}
for name, content in sources.items():
    path = site / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
for name, paths in (
    ('late-package', list(sources)[:-1]),
    ('namespace-peer', ['shared_namespace/first.py', 'shared_namespace/second.py']),
):
    directory = site / (name.replace('-', '_') + '-1.0.dist-info')
    directory.mkdir()
    (directory / 'METADATA').write_text(f'Name: {name}\nVersion: 1.0\n')
    paths += [str((directory / file).relative_to(site)) for file in ('METADATA', 'RECORD')]
    (directory / 'RECORD').write_text(''.join(f'{path},,\n' for path in paths))
distributions = list(metadata.distributions(path=[str(site)]))
metadata.distributions = lambda: iter(distributions)
roots = {'runtime': runtime.__file__}
profile = {'source': runtime.source_inventory(roots), 'weights': [],
           'environment': {'python': platform.python_version(), 'packages': runtime.package_versions()}}
known = {item['path']: item for item in profile['source']}
assert all(str((site / name).resolve()) in known for name in sources)
assert len(known) == len(profile['source']), 'duplicate resolved inventory paths'
assert 'late_package' not in sys.modules and 'shared_namespace' not in sys.modules
assert all('module' not in known[str((site / name).resolve())] for name in sources)
distributions.reverse()
assert runtime.source_inventory(roots) == profile['source'], 'metadata order changed inventory'
frozen = json.dumps(profile, sort_keys=True)
for name in ('late_package.unused', 'late_package.tests.helper',
             'shared_namespace.first', 'shared_namespace.second'):
    importlib.import_module(name)
sys.modules['dependency_alias'] = sys.modules['late_package.unused']
loaded = runtime.verify_loaded(profile)
assert loaded['dependency_alias'] == loaded['late_package.unused']
for name in ('late_package.unused', 'shared_namespace.first', 'shared_namespace.second'):
    assert loaded[name]['sha256'] == known[loaded[name]['path']]['sha256']
assert json.dumps(profile, sort_keys=True) == frozen, 'verification extended the frozen profile'

def rejected(message):
    try:
        runtime.verify_loaded(profile)
    except runtime.RuntimeMismatch as error:
        assert message in str(error), str(error)
    else:
        raise AssertionError('Unrecorded or changed source accepted')

for name, path in (('external_late', site / 'external.py'),
                   ('late_package.new', site / 'late_package/new.py'),
                   ('unknown_propagate', site / 'generated.py')):
    path.write_text('value = 5\n')
    spec = importlib.util.spec_from_file_location(name, path)
    sys.modules[name] = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sys.modules[name])
    rejected('Loaded source was not fingerprinted: ' + name)
    del sys.modules[name]
(site / 'late_package/unused.py').write_text('value = 6\n')
rejected('changed or disappeared')
print('late installed packages and namespaces accepted; exact frozen paths and bytes enforced')
'''
    result = subprocess.run([sys.executable, '-B', '-c', script, str(tmp_path)],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


def test_inventory_deduplicates_roots_and_records_external_installed_files(tmp_path, monkeypatch):
    from importlib import metadata
    import pytest
    from applications.scoring import runtime

    site = tmp_path / 'site'
    info = site / 'editable-1.0.dist-info'
    info.mkdir(parents=True)
    external = tmp_path / 'external.py'
    external.write_text('value = 1\n')
    (info / 'METADATA').write_text('Name: editable\nVersion: 1.0\n')
    record = info / 'RECORD'
    record.write_text('../external.py,,\neditable-1.0.dist-info/METADATA,,\neditable-1.0.dist-info/RECORD,,\n')
    monkeypatch.setattr(runtime.metadata, 'distributions', lambda: [metadata.PathDistribution(info)])
    roots = {'second': external, 'first': external}
    assert len(runtime.source_files(roots)) == 1
    source = runtime.source_inventory(roots)
    assert len([item for item in source if item['path'] == str(external)]) == 1
    recorded = runtime.source_inventory({'runtime': runtime.__file__})
    item = next(item for item in recorded if item['path'] == str(external))
    assert 'module' not in item and item['sha256'] == runtime.checksum(external)
    record.write_text(record.read_text() + 'new.py,,\n')
    with pytest.raises(runtime.RuntimeMismatch, match='changed or disappeared'):
        runtime.verify_files(recorded)
    source = runtime.source_inventory({'runtime': runtime.__file__})
    missing = site / 'new.py'
    assert str(missing) not in {item['path'] for item in source}
    missing.write_text('value = 2\n')
    import importlib.util
    import platform

    spec = importlib.util.spec_from_file_location('newly_created_dependency', missing)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    profile = {'source': source, 'weights': [],
               'environment': {'python': platform.python_version(), 'packages': runtime.package_versions()}}
    with pytest.raises(runtime.RuntimeMismatch, match='Loaded source was not fingerprinted: newly_created_dependency'):
        runtime.verify_loaded(profile)


def test_standard_library_alias_and_external_source_have_consistent_inventory_rules(tmp_path):
    external = tmp_path / "external.py"
    external.write_text("value = 1\n")
    script = r'''
import enum
import importlib.util
import platform
import sys
from applications.scoring import runtime

sys.modules['numba.core.types.scalars.enum'] = enum
profile = {
    'source': runtime.source_inventory({'runtime': runtime.__file__}),
    'weights': [], 'modules': ['numba'],
    'environment': {'python': platform.python_version(), 'packages': runtime.package_versions()},
}
assert runtime.standard_library(__import__('pathlib').Path(enum.__spec__.origin))
assert enum.__spec__.origin not in {item['path'] for item in profile['source']}
runtime.verify_loaded(profile)

spec = importlib.util.spec_from_file_location('external_probe', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
sys.modules['external_probe'] = module
try:
    runtime.verify_loaded(profile)
except runtime.RuntimeMismatch as error:
    assert 'external_probe' in str(error)
else:
    raise AssertionError('Unrecorded external source was accepted')
profile['source'] = runtime.source_inventory({'runtime': runtime.__file__})
runtime.verify_loaded(profile)
with open(sys.argv[1], 'a') as handle:
    handle.write('changed = True\n')
try:
    runtime.verify_loaded(profile)
except runtime.RuntimeMismatch:
    pass
else:
    raise AssertionError('Changed external source was accepted')
print('stdlib alias accepted; unrecorded and changed external sources rejected')
'''
    result = subprocess.run([sys.executable, "-B", "-c", script, str(external)],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr


def test_generated_templates_are_part_of_source_inventory(tmp_path):
    from applications.scoring import runtime
    import pytest

    template = tmp_path / 'propagate.jinja'
    template.write_text('original template')
    source = runtime.source_files({'generator': tmp_path})
    assert [item['path'] for item in source] == [str(template.resolve())]
    template.write_text('changed template')
    with pytest.raises(runtime.RuntimeMismatch, match='changed or disappeared'):
        runtime.verify_files(source)


def test_prerecorded_temporary_source_requires_module_identity_and_exact_bytes(tmp_path):
    first, second = tmp_path / 'first', tmp_path / 'second'
    first.mkdir()
    second.mkdir()
    for directory in (first, second):
        (directory / 'generated.py').write_text('value = 1\n')
    script = r'''
import importlib.util
import json
import platform
import sys
from pathlib import Path
from applications.scoring import runtime

path, profile_path = map(Path, sys.argv[1:3])
name = sys.argv[3]
spec = importlib.util.spec_from_file_location(name, path)
sys.modules[name] = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sys.modules[name])
if not profile_path.exists():
    profile = {'source': runtime.source_inventory({'runtime': runtime.__file__}), 'weights': [],
               'environment': {'python': platform.python_version(), 'packages': runtime.package_versions()}}
    profile_path.write_text(json.dumps(profile))
else:
    profile = json.loads(profile_path.read_text())
runtime.verify_loaded(profile)
'''
    profile = tmp_path / 'profile.json'
    for path, name, accepted in (
        (first / 'generated.py', 'recorded_generator', True),
        (second / 'generated.py', 'recorded_generator', True),
        (first / 'generated.py', 'unexpected_generator', False),
    ):
        result = subprocess.run([sys.executable, '-B', '-c', script, str(path), str(profile), name],
                                capture_output=True, text=True, timeout=120)
        assert (result.returncode == 0) == accepted, result.stdout + result.stderr
        if not accepted:
            assert 'Loaded source was not fingerprinted' in result.stderr
    (second / 'generated.py').write_text('value = 2\n')
    result = subprocess.run([sys.executable, '-B', '-c', script, str(second / 'generated.py'),
                             str(profile), 'recorded_generator'], capture_output=True, text=True, timeout=120)
    assert result.returncode != 0 and 'Loaded source changed' in result.stderr
