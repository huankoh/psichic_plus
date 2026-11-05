import subprocess
import sys


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
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
