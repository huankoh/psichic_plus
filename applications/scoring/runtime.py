"""Read-only runtime inspection, also executable in the isolated Boltz Python."""

import base64
from dataclasses import asdict
import hashlib
import importlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import sys
import sysconfig
import tarfile

STANDARD_LIBRARY = tuple(Path(sysconfig.get_path(name)).resolve() for name in ('stdlib', 'platstdlib'))


class RuntimeMismatch(ValueError):
    pass


def checksum(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def stamp(path):
    info = Path(path).stat()
    return [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns]


def file_evidence(name, path, *, allow_empty=False):
    path = Path(path).resolve(strict=True)
    before = stamp(path)
    if not path.is_file() or not before[2] and not allow_empty:
        raise ValueError(f'Required cached asset is empty or not a file: {path}')
    sha = checksum(path)
    if before != stamp(path):
        raise RuntimeMismatch(f'File changed while hashing: {path}')
    return {'name': name, 'path': str(path), 'sha256': sha, 'stamp': before}


def package_versions():
    return dict(sorted((d.metadata['Name'], d.version) for d in metadata.distributions() if d.metadata['Name']))


def source_path(path):
    return ('__pycache__' not in path.parts
            and (path.suffix in ('.py', '.so', '.pyd', '.dylib', '.smi', '.jsonl', '.jinja') or '.so.' in path.name))


def source_files(roots):
    files = {}
    for name, root in sorted(roots.items()):
        root = Path(root).resolve()
        paths = [root] if root.is_file() else sorted(p for p in root.rglob('*') if p.is_file()
            and 'tests' not in p.relative_to(root).parts and source_path(p))
        if not paths:
            raise ValueError(f'No source found for {name} at {root}')
        for path in paths:
            resolved = str(path.resolve())
            if resolved not in files:
                files[resolved] = file_evidence(name + '/' + (path.name if root.is_file() else path.relative_to(root).as_posix()),
                                                path, allow_empty=True)
    return list(files.values())


def standard_library(path):
    return (any(path.is_relative_to(root) for root in STANDARD_LIBRARY)
            and not {'site-packages', 'dist-packages'}.intersection(path.parts))


def source_inventory(roots):
    """Freeze explicit roots and installed distribution files without importing dependencies."""
    files = {item['path']: item for item in source_files({**roots, 'python-executable': sys.executable})}
    distributions = sorted(metadata.distributions(),
                           key=lambda d: (d.metadata['Name'] or '', str(d.locate_file(''))))
    for distribution in distributions:
        distribution_name = distribution.metadata['Name']
        for item in sorted(distribution.files or ()):
            native_resource = (distribution_name or '').lower() == 'triton' and '__pycache__' not in item.parts
            if not native_resource and not source_path(item) and item.name not in ('RECORD', 'METADATA', 'PKG-INFO', 'SOURCES.txt', 'direct_url.json'):
                continue
            path = Path(item.locate()).resolve(strict=True)
            if str(path) not in files:
                name = f'installed/{distribution_name}/{item}'
                files[str(path)] = file_evidence(name, path, allow_empty=True)
    known = set(files)
    files = list(files.values())
    for name, module in tuple(sys.modules.items()):
        origin = getattr(getattr(module, '__spec__', None), 'origin', None)
        if not origin or origin in ('built-in', 'frozen'):
            continue
        path = Path(origin).resolve()
        if str(path) in known or standard_library(path):
            continue
        # Some libraries generate import-time source in a new temporary directory
        # per process. Bind its real bytes by module identity, retaining both paths.
        files.append({**file_evidence('import/' + name, path, allow_empty=True), 'module': name})
    return sorted(files, key=lambda item: (item['name'], item['path']))


def verify_files(files):
    for item in files:
        if 'module' in item:
            continue  # Each process verifies these imports by exact bytes below.
        try:
            unchanged = stamp(item['path']) == item['stamp']
        except OSError:
            unchanged = False
        if not unchanged:
            raise RuntimeMismatch(f"Verified file changed or disappeared: {item['path']}. Create a new run.")


def native_support():
    return importlib.import_module((__package__ + '.' if __package__ else '') + 'native_build')


def identity(profile):
    return {**{key: ([{'name': f['name'], 'sha256': f['sha256']} for f in profile[key]]
                    if key in ('weights', 'source') else profile[key])
              for key in ('engine_version', 'weights', 'source', 'environment', 'device', 'precision')},
            'generated_source': profile.get('generated_source', []),
            'native_generation': (native_support().policy_identity(profile['native_generation'])
                                  if profile.get('native_generation') else None)}


def verify_loaded(profile, native=None):
    """Check actual imported bytes against the preflight inventory before accepting output."""
    verify_files(profile['weights'] + profile['source'])
    if package_versions() != profile['environment']['packages'] or platform.python_version() != profile['environment']['python']:
        raise RuntimeMismatch('Loaded packages or Python changed. Create a new run.')
    known = {f['path']: f for f in profile['source'] if 'module' not in f}
    imported = {f['module']: f for f in profile['source'] if 'module' in f}
    generated = {f['module']: f for f in profile.get('generated_source', [])}
    if native is not None and native.profile != profile:
        raise RuntimeMismatch('Native build evidence belongs to a different frozen profile')
    native_files = native.verified() if native is not None else {}
    loaded = {}
    for name, module in tuple(sys.modules.items()):
        spec = getattr(module, '__spec__', None)
        path = getattr(spec, 'origin', None)
        if not path or path in ('built-in', 'frozen'):
            continue
        path = str(Path(path).resolve())
        expected = generated.get(name) or known.get(path) or imported.get(name)
        if expected is None and (name, path) in native_files:
            expected = {'sha256': native_files[(name, path)]}
        if expected is not None or not standard_library(Path(path)):
            if expected is None:
                raise RuntimeMismatch(f'Loaded source was not fingerprinted: {name} at {path}')
            if checksum(path) != expected['sha256']:
                raise RuntimeMismatch(f'Loaded source changed: {path}')
            loaded[name] = {'path': path, 'sha256': expected['sha256']}
    return loaded


def environment(device):
    import torch

    gpu = None
    if device in ('cuda:0', 'gpu'):
        import subprocess
        allocation = os.environ.get('CUDA_VISIBLE_DEVICES')
        if not allocation or ',' in allocation or allocation in ('-1', 'all'):
            raise ValueError('Set CUDA_VISIBLE_DEVICES to one assigned GPU before validation')
        query = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,name,driver_version,compute_cap',
                                '--format=csv,noheader'], capture_output=True, text=True, check=True, timeout=15)
        matches = [line for line in query.stdout.splitlines() if allocation in [v.strip() for v in line.split(',')[:2]]]
        if len(matches) != 1:
            raise ValueError('Cannot identify the assigned GPU without initializing it; use its full UUID or physical index')
        gpu = matches[0]
    return {'python': platform.python_version(), 'packages': package_versions(),
            'runtime': json.dumps({'platform': platform.platform(), 'machine': platform.machine(),
                'gpu': gpu, 'torch_cuda': torch.version.cuda, 'torch_hip': torch.version.hip,
                'matmul_precision': torch.get_float32_matmul_precision(),
                'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                'controls': {k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                    'OPENBLAS_NUM_THREADS', 'CUBLAS_WORKSPACE_CONFIG', 'CUEQ_DEFAULT_CONFIG',
                    'CUEQ_DISABLE_AOT_TUNING', 'PYTHONHASHSEED')}}, sort_keys=True)}


def inspect_psichic(options):
    from applications.scorer.psichic_plus.predictor import load_helpers
    from applications.scoring.pyg_source import expected_source
    import applications.scoring

    helpers = load_helpers()
    generated = expected_source()
    roots = {'screening': Path(helpers.__file__).parent,
             'scoring': Path(applications.scoring.__file__).parent,
             'scorer': Path(applications.scoring.__file__).parent.parent / 'scorer'}
    modules = ['torch', 'numpy', 'pandas', 'rdkit', 'torch_geometric', 'torch_scatter', 'esm', 'psichic_ligand']
    for name in modules:
        module = importlib.import_module(name)
        roots[name] = Path(module.__file__).parent
    model = Path(options['model_dir'])
    esm = Path(options['esm_checkpoint'])
    files = [(f'psichic/{name}', model / name) for name in ('config.json', 'degree.pt', 'model.pt')]
    files += [('esm/model', esm), ('esm/contact-regression', esm.with_name(esm.stem + '-contact-regression.pt'))]
    profile = {'engine_version': 'psichic-plus-local-v1', 'weights': [file_evidence(n, p) for n, p in files],
               'source': source_inventory(roots), 'environment': environment(options['device']), 'device': options['device'],
               'generated_source': generated,
               'precision': 'fp32; ESM token storage fp16', 'modules': modules,
               'executable': sys.executable, 'screening': str(roots['screening'])}
    profile['loaded'] = verify_loaded(profile)
    return profile


def verify_boltz_record():
    distribution = metadata.distribution('boltz')
    if distribution.version != '2.2.1' or sys.version_info[:2] != (3, 12):
        raise ValueError('Use unmodified Boltz 2.2.1 in a separate Python 3.12 environment')
    source = []
    for item in distribution.files or ():
        if str(item).startswith('boltz/') and item.suffix == '.py':
            if not item.hash or item.hash.mode != 'sha256':
                raise ValueError(f'Missing Boltz RECORD hash: {item}')
            path = item.locate().resolve()
            actual = checksum(path)
            if base64.urlsafe_b64encode(bytes.fromhex(actual)).rstrip(b'=').decode() != item.hash.value:
                raise ValueError(f'Boltz source differs from installed RECORD: {path}')
            source.append(file_evidence(str(item), path, allow_empty=True))
    if not source:
        raise ValueError('Boltz has no verifiable installed source RECORD')
    package_root = Path(distribution.locate_file('boltz')).resolve()
    if {f['path'] for f in source} != {str(p.resolve()) for p in package_root.rglob('*.py')}:
        raise ValueError('Installed Boltz contains source absent from its RECORD')
    return source


def inspect_boltz(options):
    verified = verify_boltz_record()
    from boltz.main import predict, Boltz2DiffusionParams, PairformerArgsV2, MSAModuleArgs, BoltzSteeringParams
    import boltz
    import click

    expected = {f['path'] for f in verified}
    if str(Path(boltz.__file__).resolve()) not in expected:
        raise ValueError('Imported Boltz differs from the installed distribution')
    params = {p.name: p for p in predict.params if isinstance(p, click.Option)}
    supplied = options['native']
    resolved = {}
    with click.Context(predict) as ctx:
        for name, param in params.items():
            value = supplied.get(name, param.get_default(ctx))
            resolved[name] = param.type_cast_value(ctx, value)
        native_help = predict.get_help(ctx)
    if set(supplied) - set(params):
        raise ValueError(f'Unsupported native options: {sorted(set(supplied) - set(params))}')
    resolved['cache'] = str(Path(resolved['cache']).expanduser().absolute())
    cache = Path(resolved['cache'])
    # Native download_boltz2 checks even the default files when custom checkpoints are selected.
    files = [(f'cache/{n}', cache / n) for n in ('mols.tar', 'boltz2_conf.ckpt', 'boltz2_aff.ckpt')]
    mols = molecule_files(cache)
    files += [(f'mols/{p.relative_to(cache / "mols").as_posix()}', p) for p in mols]
    for name, default in (('checkpoint', 'boltz2_conf.ckpt'), ('affinity_checkpoint', 'boltz2_aff.ckpt')):
        resolved[name] = str(Path(resolved[name] or cache / default).expanduser().absolute())
        files.append((name, Path(resolved[name])))
    resolved['step_scale'] = 1.5 if resolved['step_scale'] is None else resolved['step_scale']
    modules = ['boltz', 'torch', 'numpy', 'rdkit', 'pytorch_lightning', 'psutil']
    roots = {name: Path(importlib.import_module(name).__file__).parent for name in modules}
    roots['executor-runtime'] = Path(__file__)
    roots['executor-guardian'] = Path(__file__).with_name('guardian.py')
    roots['executor-native-build'] = Path(__file__).with_name('native_build.py')
    native_generation = native_support().freeze()
    profile = {'engine_version': '2.2.1', 'native_generation': native_generation, 'weights': [file_evidence(n, p) for n, p in files],
               'source': source_inventory(roots), 'environment': environment(resolved['accelerator']), 'device': resolved['accelerator'],
               'precision': 'bf16-mixed; float32 matmul highest', 'modules': modules,
               'executable': sys.executable, 'native_help': native_help, 'resolved': resolved,
               'option_flags': {name: p.opts[0] for name, p in params.items()},
               'flag_options': [name for name, p in params.items() if p.is_flag],
               'internal_defaults': {'diffusion': {**asdict(Boltz2DiffusionParams()), 'step_scale': resolved['step_scale']},
                   'pairformer': asdict(PairformerArgsV2()),
                   'msa': {**asdict(MSAModuleArgs()), 'subsample_msa': resolved['subsample_msa'],
                           'num_subsampled_msa': resolved['num_subsampled_msa'], 'use_paired_feature': True},
                   'steering': {**asdict(BoltzSteeringParams()), 'fk_steering': resolved['use_potentials'],
                                'physical_guidance_update': resolved['use_potentials']},
                   'affinity_steering': {**asdict(BoltzSteeringParams()), 'fk_steering': False,
                                        'physical_guidance_update': False, 'contact_guidance_update': False},
                   'affinity_recycling_steps': 5,
                   'affinity_max_parallel_samples': 1}}
    profile['loaded'] = verify_loaded(profile)
    return profile


def molecule_files(cache):
    """Reject a partially extracted offline cache before native model initialization."""
    expected = set()
    with tarfile.open(cache / 'mols.tar', 'r') as archive:
        for member in archive:
            path = Path(member.name)
            if path.suffix != '.pkl':
                continue
            if not member.isfile() or path.is_absolute() or '..' in path.parts or path.parts[0] != 'mols' or path in expected:
                raise ValueError(f'Invalid molecule archive member: {member.name}')
            expected.add(path)
            destination = cache / path
            if not destination.is_file() or destination.stat().st_size != member.size:
                raise ValueError(f'Incomplete Boltz molecule cache: {destination}. Restore the complete offline extraction of mols.tar.')
    if not expected:
        raise ValueError(f'No molecule records in {cache / "mols.tar"}. Prepare the complete Boltz cache offline.')
    return sorted((cache / 'mols').rglob('*.pkl'))


def offline_only():
    def audit(event, args):
        if event == 'urllib.Request' or event == 'socket.connect' and isinstance(args[1], tuple):
            raise RuntimeError('Local scoring forbids network access. Prepare all cached assets before running.')
    sys.addaudithook(audit)


def verify_inputs(directory, request):
    for item in request['target_assets']:
        if checksum(item['path']) != item['sha256']:
            raise RuntimeMismatch('Retained target asset changed before inference')
    tasks = {t['entry_key']: t for t in request['tasks']}
    ordered = [t['entry_key'] for t in request['tasks']]
    if ordered != request['tasks'][0]['fingerprint']['batch']['row_keys'] or ordered != [m['entry_key'] for m in request['mapping']]:
        raise RuntimeMismatch('Native input mapping changed the exact prepared batch')
    names = set()
    for index, item in enumerate(request['mapping'], 1):
        task = tasks.pop(item['entry_key'])
        native_key = f'row_{index:08d}'
        expected_input = native_key + '.yaml' if request['engine'] == 'boltz' else task['entry_key'] + '.json'
        if (item['attempt_id'] != task['attempt_id'] or item['input'] in names or item['native_row'] != index
                or item['native_key'] != native_key or item['input'] != expected_input):
            raise RuntimeMismatch('Native input mapping changed')
        names.add(item['input'])
        if checksum(directory / 'inputs' / item['input']) != task['fingerprint']['native_input_sha256']:
            raise RuntimeMismatch('Prepared native input bytes changed')
    if tasks or names != {p.name for p in (directory / 'inputs').iterdir()}:
        raise RuntimeMismatch('Native input batch differs from prepared tasks')


def execute_boltz(directory):
    import signal
    import threading

    if sys.stdin.buffer.readline() != b'GO\n':
        return 125

    def watch_owner():
        os.read(sys.stdin.fileno(), 1)
        os.killpg(os.getpgrp(), signal.SIGKILL)

    threading.Thread(target=watch_owner, daemon=True).start()
    import importlib.util
    spec = importlib.util.spec_from_file_location('local_group_guardian', Path(__file__).with_name('guardian.py'))
    guardian = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guardian)
    guardian.start_guardian(directory)
    directory = Path(directory)
    request = json.loads((directory / 'request.json').read_text())
    profile = request['profile']
    verify_inputs(directory, request)
    verify_files(profile['source'] + profile['weights'])
    verify_boltz_record()
    offline_only()
    from boltz.main import predict

    command = request['native_command']
    with predict.make_context('predict', command[2:]) as ctx:
        for name, value in profile['resolved'].items():
            if name != 'out_dir' and ctx.params[name] != value:
                raise RuntimeMismatch(f'Native option changed after preparation: {name}')
        verify_loaded(profile)
        from contextlib import nullcontext
        builds = (native_support().NativeBuilds(profile, directory / 'native-builds')
                  if profile.get('native_generation') else nullcontext())
        native = None
        try:
            with builds as native:
                predict.invoke(ctx)
        finally:
            observed = []
            for name, module in tuple(sys.modules.items()):
                origin = getattr(getattr(module, '__spec__', None), 'origin', None)
                if origin and origin not in ('built-in', 'frozen'):
                    path = Path(origin).resolve()
                    observed.append({'module': name, 'path': str(path),
                                     'sha256': checksum(path) if path.is_file() else None})
            (directory / 'loaded-observed.json').write_text(json.dumps(observed, sort_keys=True))
            loaded = verify_loaded(profile, native)
            if native is not None:
                (directory / 'native-builds.json').write_text(json.dumps(native.evidence(), sort_keys=True, allow_nan=False))
            content = {'profile_digest': hashlib.sha256(json.dumps(identity(profile), sort_keys=True,
                       separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode()).hexdigest(),
                       'loaded': loaded, 'native_command': command}
            temporary = directory / 'loaded.tmp'
            with temporary.open('x') as handle:
                json.dump(content, handle, allow_nan=False, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, directory / 'loaded.json')
    return 0


def main():
    # fd redirection also contains native C-extension prints during imports.
    response_fd = os.dup(1)
    os.dup2(2, 1)
    request = json.load(sys.stdin)
    try:
        profile = inspect_psichic(request['options']) if request['engine'] == 'psichic' else inspect_boltz(request['options'])
        result = {'profile': profile}
    except Exception as error:
        result = {'error': f'{type(error).__name__}: {error}'}
    with os.fdopen(response_fd, 'w') as output:
        json.dump(result, output, allow_nan=False, sort_keys=True)
        output.write('\n')
    return int('error' in result)


if __name__ == '__main__':
    sys.modules['runtime'] = sys.modules[__name__]
    sys.exit(execute_boltz(sys.argv[2]) if sys.argv[1:2] == ['--execute'] else main())
