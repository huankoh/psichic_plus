"""The real Linux compiler lane is also runnable without pytest in the pinned image."""

from copy import deepcopy
from importlib import metadata
import json
from pathlib import Path

import pytest

from applications.scoring import native_build, runtime
from applications.scoring.executor import _profile_assets


def test_native_resources_are_frozen_as_bytes_not_just_record(tmp_path, monkeypatch):
    package = tmp_path / 'triton'
    package.mkdir()
    info = tmp_path / 'triton-3.3.1.dist-info'
    info.mkdir()
    (info / 'METADATA').write_text('Name: triton\nVersion: 3.3.1\n')
    resources = ('driver.c', 'cuda.h', 'ptxas', 'libdevice.bc', 'driver.py')
    for name in resources:
        (package / name).write_bytes(b'original input\n')
    (info / 'RECORD').write_text(''.join('triton/' + n + ',,\n' for n in resources))
    monkeypatch.setattr(runtime.metadata, 'distributions', lambda: [metadata.PathDistribution(info)])
    frozen = runtime.source_inventory({'runtime': runtime.__file__})
    assert {str(package / n) for n in resources} <= {f['path'] for f in frozen}
    before = (info / 'RECORD').read_bytes()
    (package / 'driver.c').write_bytes(b'changed C input\n')
    with pytest.raises(runtime.RuntimeMismatch, match='changed or disappeared'):
        runtime.verify_files(frozen)
    assert runtime.source_inventory({'runtime': runtime.__file__}) != frozen
    assert (info / 'RECORD').read_bytes() == before


def test_native_policy_binds_inputs_and_controls_without_binding_build_paths(tmp_path):
    path = tmp_path / 'compiler'
    path.write_bytes(b'compiler bytes')
    policy = {'contract': native_build.CONTRACT, 'files': [runtime.file_evidence('compiler', path)],
              'environment': {'CC': '/qualified/compiler'}}
    profile = {'source': [], 'native_generation': policy}
    expected = _profile_assets(profile, 'source')
    moved = deepcopy(profile)
    moved['native_generation']['files'][0].update(path='/another/temp/path', stamp=[0] * 5)
    assert _profile_assets(moved, 'source') == expected
    for key, value in [('sha256', '0' * 64), ('name', 'another input')]:
        changed = deepcopy(profile)
        changed['native_generation']['files'][0][key] = value
        assert _profile_assets(changed, 'source') != expected
    changed = deepcopy(profile)
    changed['native_generation']['environment']['CC'] = '/other/compiler'
    assert _profile_assets(changed, 'source') != expected
    assert native_build.policy_identity(json.loads(json.dumps(policy))) == native_build.policy_identity(policy)


def test_external_native_controls_are_rejected(monkeypatch):
    for key in native_build.FORBIDDEN:
        with monkeypatch.context() as patch:
            patch.setenv(key, 'unrecorded')
            with pytest.raises(runtime.RuntimeMismatch, match=key):
                native_build.controls()


def test_real_native_compiler_contract(tmp_path):
    pytest.importorskip('triton')
    import subprocess
    import sys

    if sys.platform != 'linux':
        pytest.skip('The qualified native compiler lane is Linux only')
    script = Path(__file__).with_name('native_build_probe.py')
    for name in ('first', 'second'):
        result = subprocess.run([sys.executable, '-B', str(script), str(tmp_path / name),
                                 str(tmp_path / 'profile.json')], capture_output=True, text=True, timeout=600)
        assert result.returncode == 0, result.stdout + result.stderr
    first = json.loads((tmp_path / 'first/result.json').read_text())
    second = json.loads((tmp_path / 'second/result.json').read_text())
    assert first['profile_digest'] == second['profile_digest']
    assert first['checks'] == second['checks']


def test_native_inputs_require_frozen_bytes_when_container_stamps_change(tmp_path):
    path = tmp_path / 'loader-cache'
    path.write_bytes(b'frozen loader bytes')
    policy = {'files': [runtime.file_evidence('loader-cache', path)]}
    path.unlink()
    path.write_bytes(b'frozen loader bytes')
    native_build.verify_inputs(policy)
    assert runtime.stamp(path) != policy['files'][0]['stamp']
    path.write_bytes(b'changed loader bytes')
    with pytest.raises(runtime.RuntimeMismatch, match='changed or disappeared'):
        native_build.verify_inputs(policy)


def test_backend_code_alias_requires_the_same_code_and_source_path(tmp_path, monkeypatch):
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location('applications.scoring.native_build_alias', native_build.__file__)
    alias = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, alias)
    spec.loader.exec_module(alias)
    actual, expected = alias.controls.__code__, native_build.controls.__code__
    assert actual is not expected and native_build.same_code(actual, expected)
    assert not native_build.same_code(actual.replace(co_filename=str(tmp_path / 'foreign.py')), expected)
    assert not native_build.same_code(alias.freeze.__code__, expected)
