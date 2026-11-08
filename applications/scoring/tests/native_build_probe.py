"""CPU-only real Triton probe. Run twice with one profile and distinct output paths."""

from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch
import triton
from triton.backends import backends
from triton.backends.nvidia import driver as canonical_driver
from triton.compiler import ASTSource

from applications.scoring import native_build, runtime

driver = SimpleNamespace(**backends['nvidia'].driver.__init__.__globals__)


@triton.jit
def representative(x, n):
    pass


def rejected(checks, name, action, message):
    try:
        action()
    except runtime.RuntimeMismatch as error:
        chain = []
        cause = error
        while cause is not None:
            chain.append(str(cause))
            cause = cause.__context__
        detail = '\n'.join(chain)
        assert message in detail, detail
        checks[name] = detail
    else:
        raise AssertionError('Accepted ' + name)


def main():
    out, profile_path = map(Path, sys.argv[1:])
    out.mkdir()
    runtime.offline_only()
    assert not torch.cuda.is_initialized()
    with patch.object(torch.nn.Module, '__init__', side_effect=AssertionError('model construction')), \
            patch.object(torch, 'load', side_effect=AssertionError('checkpoint load')), \
            patch.object(torch.cuda, '_lazy_init', side_effect=AssertionError('CUDA initialization')):
        if profile_path.exists():
            profile = json.loads(profile_path.read_text())
            assert native_build.policy_identity(native_build.freeze()) == native_build.policy_identity(profile['native_generation'])
        else:
            policy = native_build.freeze()
            profile = {'native_generation': policy,
                       'source': runtime.source_inventory({'scoring': Path(runtime.__file__).parent, 'probe': __file__}),
                       'weights': [], 'environment': {'python': platform.python_version(), 'packages': runtime.package_versions()}}
            profile_path.write_text(json.dumps(profile, sort_keys=True))
        frozen = json.dumps(profile, sort_keys=True)
        os.environ['TRITON_CACHE_DIR'] = str(out / 'builds/cache')
        source = ASTSource(representative, signature={'x': '*fp32', 'n': 'i32'})
        metadata = SimpleNamespace(global_scratch_size=0, global_scratch_align=1, launch_cooperative_grid=False)
        with native_build.NativeBuilds(profile, out / 'builds') as builds:
            driver.CudaUtils()
            driver.CudaLauncher(source, metadata)
            driver.CudaLauncher(ASTSource(representative, signature={'x': '*fp64', 'n': 'i64'}), metadata)
            canonical_driver.CudaUtils()
        assert sys.getprofile() is None
        loaded = runtime.verify_loaded(profile, builds)
        evidence = builds.evidence()
        (out / 'builds.json').write_text(json.dumps(evidence, indent=2))
        (out / 'loaded.json').write_text(json.dumps(loaded, indent=2))
        checks = {'registered_backend_alias': 'accepted', 'utility': 'accepted', 'launcher_runtime_variants': 'accepted', 'in_process_cache_reuse': 'accepted'}
        original = Path(builds.records[0].output)
        original_bytes = original.read_bytes()
        replacement = out / 'replacement.so'
        replacement.write_bytes(original_bytes + b'corruption')
        os.replace(replacement, original)
        rejected(checks, 'corrupted_binary', lambda: runtime.verify_loaded(profile, builds), 'changed after loading')
        replacement.write_bytes(original_bytes)
        os.replace(replacement, original)
        for name, path in (('unrelated_module', original), ('cuda_utils', out / 'unrecorded.so')):
            if path != original:
                shutil.copyfile(original, path)
            module = ModuleType(name)
            module.__spec__ = importlib.util.spec_from_file_location(name, path)
            with patch.dict(sys.modules, {name: module}):
                rejected(checks, 'unknown_' + name, lambda: runtime.verify_loaded(profile, builds), 'not fingerprinted')
        for name in ('unknown_external', 'unknown_installed'):
            path = out / (name + '.py')
            path.write_text('value = 1\n')
            module = ModuleType(name)
            module.__spec__ = importlib.util.spec_from_file_location(name, path)
            with patch.dict(sys.modules, {name: module}):
                rejected(checks, name, lambda: runtime.verify_loaded(profile, builds), 'not fingerprinted')
        private = out / 'private-inputs'
        private.mkdir()
        candidates = {
            'C': Path(driver.dirname) / 'driver.c', 'header': Path(driver.include_dir[0]) / 'cuda.h',
            'generator': Path(driver.__file__), 'compiler': Path(profile['native_generation']['compiler']),
            'compiler_resource': next(Path(f['path']) for f in profile['native_generation']['files'] if f['path'].endswith('/specs')),
        }
        for name, original_input in candidates.items():
            copied = private / name
            shutil.copyfile(original_input, copied)
            changed = deepcopy(profile)
            item = runtime.file_evidence('private/' + name, copied)
            changed['native_generation']['files'].append(item)
            copied.write_bytes(copied.read_bytes() + b'\nchanged\n')
            checker = native_build.NativeBuilds(changed, out / ('changed-' + name))
            rejected(checks, 'changed_' + name, checker.check_inputs, 'changed or disappeared')
        with patch.dict(os.environ, {'CC': '/unrecorded/compiler'}):
            rejected(checks, 'changed_toolchain_selection', builds.check_inputs, 'controls changed')
        with patch.dict(os.environ, {'TRITON_CUDART_PATH': str(private)}):
            rejected(checks, 'unqualified_options', builds.check_inputs, 'Unqualified native build control')
        for mode in ('stale-cache', 'corrupt-cache', 'unauthorized-generator', 'bad-signature'):
            directory = out / mode
            os.environ['TRITON_CACHE_DIR'] = str(directory / 'cache')
            def attempt():
                with native_build.NativeBuilds(profile, directory):
                    if mode in ('stale-cache', 'corrupt-cache'):
                        destination = directory / 'cache' / original.parent.name / original.name
                        destination.parent.mkdir()
                        destination.write_bytes(original_bytes + (b'bad' if mode == 'corrupt-cache' else b''))
                        driver.CudaUtils()
                    elif mode == 'unauthorized-generator':
                        driver.compile_module_from_src(Path(driver.dirname, 'driver.c').read_text(), 'cuda_utils')
                    else:
                        driver.CudaLauncher(ASTSource(representative, signature={'x': '*fp32', 'n': 'i32; injected'}), metadata)
            message = ('cache binary' if mode.endswith('cache') else
                       'Unauthorized native generator' if mode == 'unauthorized-generator' else 'signature')
            rejected(checks, mode, attempt, message)
            assert sys.getprofile() is None
        assert json.dumps(profile, sort_keys=True) == frozen
        assert not torch.cuda.is_initialized()
        digest = hashlib.sha256(frozen.encode()).hexdigest()
        result = {'result': 'PASS', 'profile_digest': digest, 'checks': sorted(checks),
                  'details': checks, 'builds': len(builds.records), 'loaded': len(loaded),
                  'cuda_initialized': False, 'source_inventory_extended': False}
        (out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps({k: v for k, v in result.items() if k != 'details'}))


if __name__ == '__main__':
    main()
