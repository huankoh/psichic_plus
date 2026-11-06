"""Resolve external inputs once, then retain the exact local execution contract."""

import copy
import csv
from dataclasses import asdict, dataclass
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import yaml

from applications.scorer import Output
from applications.scorer.boltz2.models import EnsembleMethod, Metric
from applications.scorer.boltz2.target import Target as NativeTarget
from .contracts import Asset, Engine, Entry, Settings, Target, canonical, digest, loads
from .evidence import fsync_directory, write_file
from .runtime import checksum, identity

QUICK = {'recycling_steps': 1, 'sampling_steps': 2, 'diffusion_samples': 1,
         'sampling_steps_affinity': 2, 'diffusion_samples_affinity': 1, 'num_workers': 0,
         'preprocessing_threads': 1}
NATIVE_VALUES = {'checkpoint', 'affinity_checkpoint', 'devices', 'accelerator', 'recycling_steps',
                 'sampling_steps', 'diffusion_samples', 'max_parallel_samples', 'step_scale',
                 'output_format', 'num_workers', 'method', 'preprocessing_threads',
                 'sampling_steps_affinity', 'diffusion_samples_affinity', 'max_msa_seqs', 'num_subsampled_msa'}
NATIVE_FLAGS = {'write_full_pae', 'write_full_pde', 'use_potentials', 'affinity_mw_correction',
                'subsample_msa', 'no_kernels', 'write_embeddings'}
CONTROLS = {'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
            'CUEQ_DEFAULT_CONFIG': '1', 'CUEQ_DISABLE_AOT_TUNING': '1',
            'CUBLAS_WORKSPACE_CONFIG': None, 'CUDA_VISIBLE_DEVICES': None}


class ConfigurationError(ValueError):
    def __init__(self, code, message, next_action='Correct the configuration and run rank --validate-only again.'):
        super().__init__(message)
        self.code, self.next_action = code, next_action


def candidates(path, id_column, smiles_column, *, source=None):
    with Path(path).open(newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle, strict=True)
        names = reader.fieldnames or []
        if not names or any(not name for name in names) or len(set(names)) != len(names):
            raise ConfigurationError('INVALID_CSV', 'Candidate CSV needs unique, nonempty headers')
        if id_column == smiles_column or id_column not in names or smiles_column not in names:
            raise ConfigurationError('INVALID_CSV', 'Select distinct existing ID and SMILES columns')
        count = 0
        for count, row in enumerate(reader, 1):
            if None in row or any(value is None for value in row.values()):
                raise ConfigurationError('INVALID_CSV', f'Malformed candidate CSV row {count}')
            yield Entry(row[id_column], row[smiles_column], f'{source or Path(path).absolute()}#row={count}',
                        canonical({k: v for k, v in row.items() if k not in (id_column, smiles_column)}))
        if count == 0:
            raise ConfigurationError('EMPTY_POPULATION', 'Candidate CSV has no entries')


def native_overrides(tokens):
    tokens = list(tokens)
    if tokens[:1] == ['--']:
        tokens.pop(0)
    result = {}
    while tokens:
        token = tokens.pop(0)
        name, equal, value = token.partition('=')
        if not name.startswith('--'):
            raise ConfigurationError('NATIVE_OPTION', f'Expected a complete native option, got {name!r}')
        native = name[2:].replace('-', '_')
        canonical_flag = '--preprocessing-threads' if native == 'preprocessing_threads' else '--' + native
        if name != canonical_flag or native not in NATIVE_VALUES | NATIVE_FLAGS:
            raise ConfigurationError('NATIVE_OPTION', f'Unsupported or workflow-owned native option {name!r}')
        if native in result:
            raise ConfigurationError('NATIVE_OPTION', f'Duplicate native option {name!r}')
        if native in NATIVE_FLAGS:
            if equal:
                raise ConfigurationError('NATIVE_OPTION', f'{name} is a flag and takes no value')
            value = True
        elif not equal:
            if not tokens or tokens[0].startswith('--'):
                raise ConfigurationError('NATIVE_OPTION', f'{name} requires a value')
            value = tokens.pop(0)
        if value == '':
            raise ConfigurationError('NATIVE_OPTION', f'{name} requires a nonempty value')
        result[native] = value
    return result


def child_environment(recorded, *, boltz=False):
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')
    for key, value in recorded.items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = str(value)
    for key in ('BOLTZ_MSA_USERNAME', 'BOLTZ_MSA_PASSWORD', 'MSA_API_KEY_VALUE'):
        env.pop(key, None)
    if boltz:
        for key in ('PYTHONPATH', 'PYTHONHOME', 'PSICHIC_SCREENING_DIR'):
            env.pop(key, None)
        env['LD_LIBRARY_PATH'] = os.pathsep.join(p for p in env.get('LD_LIBRARY_PATH', '').split(os.pathsep)
                                               if p and Path(p) != Path(sys.prefix) / 'lib')
    else:
        # The imported installation, never the caller's cwd, supplies applications.
        env['PYTHONPATH'] = str(Path(__file__).resolve().parents[2])
        env.pop('PYTHONHOME', None)
    return env


def probe(engine, python, options, controls):
    command = [python, '-B', str(Path(__file__).with_name('runtime.py'))]
    result = subprocess.run(command, input=canonical({'engine': engine.value, 'options': options}),
                            capture_output=True, text=True, timeout=600,
                            env=child_environment(controls, boltz=engine is Engine.BOLTZ), cwd=Path.home())
    try:
        response = loads(result.stdout)
    except ValueError as error:
        raise ConfigurationError('RUNTIME_PROBE', f'{engine.value} runtime probe did not return valid JSON; {result.stderr[-2000:]}') from error
    if result.returncode or 'error' in response:
        raise ConfigurationError('RUNTIME_UNAVAILABLE', response.get('error', result.stderr[-2000:]),
            'Prepare the documented dependencies and cached assets offline, then rerun validation. No model was started.')
    return response['profile']


@dataclass(frozen=True)
class LocalPlan:
    output: Path
    source: Path
    id_column: str
    smiles_column: str
    row_count: int | None
    target: Target
    settings: Settings
    document: dict
    ligand_id: str
    assets: tuple[dict, ...]
    runtime: dict

    def summary(self):
        return {'entries': self.row_count, 'evaluations': {
            'psichic': self.row_count if Engine.PSICHIC in self.settings.engines else 0,
            'boltz': (self.settings.shortlist_size if self.settings.pyramid else self.row_count)
                     if Engine.BOLTZ in self.settings.engines else 0},
            'shortlist_condition': 'Boltz starts only after K eligible PSICHIC entries' if self.settings.pyramid else None,
            'target': asdict(self.target), 'settings': asdict(self.settings),
            'runtime_paths': {e: p['executable'] for e, p in self.runtime['profiles'].items()},
            'preset': self.runtime['preset'], 'overrides': self.runtime['overrides'],
            'quality': 'reduced sampling integration check, not production quality' if self.runtime['preset'] == 'quick-check'
                       else 'pinned native sampling defaults unless explicitly overridden; scientific quality unqualified'}


def _path(value):
    return str(Path(value).expanduser().absolute())


def resolve(args, *, future_population=False):
    uses_p, uses_b = args.mode != 'boltz-only', args.mode != 'psichic-only'
    unused = (('model_dir', 'psichic_device', 'psichic_cache', 'esm_checkpoint', 'batch_size', 'psichic_output', 'psichic_chain')
              if not uses_p else ('boltz_python', 'boltz_cache', 'boltz_timeout') if not uses_b else ())
    for name in unused:
        if getattr(args, name) is not None:
            raise ConfigurationError('UNUSED_OPTION', f'--{name.replace("_", "-")} is unused in {args.mode}')
    overrides = native_overrides(args.boltz_options)
    if not uses_b and (overrides or args.preset == 'quick-check'):
        raise ConfigurationError('UNUSED_OPTION', 'Boltz options and quick-check require a Boltz stage')
    if args.mode not in ('full-ensemble', 'pyramid-ensemble') and any(v is not None for v in (args.psichic_weight, args.boltz_weight)):
        raise ConfigurationError('UNUSED_OPTION', 'Manual ensemble weights require an ensemble mode')
    # Validate mode semantics before checking any external runtime.
    common = dict(shortlist_size=args.shortlist_size, ensemble=EnsembleMethod(args.ensemble) if args.ensemble else None,
                  metric=Metric(args.rank_by) if args.rank_by else None, psichic_weight=args.psichic_weight,
                  boltz_weight=args.boltz_weight, psichic_output=Output(args.psichic_output or 'affinity'))
    Settings(args.mode, **common, psichic_options='{"checking":true}' if uses_p else None,
             boltz_options='{"checking":true}' if uses_b else None)
    if not 1 <= args.chunk_size <= 4096 or args.batch_size is not None and args.batch_size < 1:
        raise ConfigurationError('INVALID_BATCH', 'Use chunk size 1..4096 and a positive PSICHIC batch size')
    if args.seed < 0 or args.seed > 2**32 - 1:
        raise ConfigurationError('INVALID_SEED', 'Seed must be in 0..4294967295')
    if args.boltz_timeout is not None and (not 0 < args.boltz_timeout < float('inf')):
        raise ConfigurationError('INVALID_TIMEOUT', 'Boltz timeout must be positive and finite')
    output, source = Path(_path(args.output)), Path(_path(args.candidates))
    if output.exists():
        raise ConfigurationError('OUTPUT_EXISTS', f'Output already exists: {output}', 'Use resume for that run, or choose a fresh output path.')
    count = None if future_population else sum(1 for _ in candidates(source, args.id_column, args.smiles_column))
    target = NativeTarget.load(fasta=args.fasta, native=args.target_yaml, ligand_id=args.ligand_id, msa_mode='single')
    if not uses_b and args.target_yaml:
        raise ConfigurationError('UNUSED_TARGET', 'PSICHIC-only takes one FASTA; native complex settings require Boltz')
    if args.fasta and args.psichic_chain:
        raise ConfigurationError('UNUSED_OPTION', '--psichic-chain applies only to native YAML')
    if args.fasta and args.ligand_id and not uses_b:
        raise ConfigurationError('UNUSED_OPTION', '--ligand-id requires Boltz')
    protein = target.protein_sequence(args.psichic_chain) if uses_p else target.protein_sequence(next(
        (p['protein']['id'][0] if isinstance(p['protein']['id'], list) else p['protein']['id'])
        for p in target.document['sequences'] if 'protein' in p))
    sequence = ''.join(protein[1].split()).upper()
    if not re.fullmatch('[A-Z]+', sequence):
        raise ConfigurationError('INVALID_TARGET', 'Selected protein sequence must contain amino-acid letters')
    document = copy.deepcopy(target.document)
    assets = [{'name': 'source_target', 'path': str(target.source), 'sha256': checksum(target.source),
               'copy': 'source_target' + target.source.suffix}]
    for index, template in enumerate(document.get('templates', [])):
        kind = 'cif' if 'cif' in template else 'pdb'
        path = target._asset(template[kind])
        name = f'template_{index:04d}{path.suffix}'
        assets.append({'name': name, 'path': str(path), 'sha256': checksum(path), 'copy': name})
        template[kind] = str(output / 'inputs' / name)
    for item in document['sequences']:
        if 'protein' in item:
            item['protein']['msa'] = 'empty'
    if not any('affinity' in prop for prop in document.get('properties', [])):
        document.setdefault('properties', []).append({'affinity': {'binder': target.ligand_id}})
    retained_target = Target(digest({'document': target.document, 'chain': protein[0]}), sequence,
                             tuple(Asset(a['name'], a['sha256']) for a in assets))
    controls = {k: os.environ.get(k, v) for k, v in CONTROLS.items()}
    controls['PYTHONHASHSEED'] = str(args.seed)
    profiles, p_options, b_options = {}, None, None
    if uses_p:
        model = args.model_dir or os.environ.get('PSICHIC_MODEL_DIR')
        if not model:
            raise ConfigurationError('MODEL_REQUIRED', 'Provide --model-dir or PSICHIC_MODEL_DIR')
        esm = args.esm_checkpoint or os.environ.get('PSICHIC_ESM_CHECKPOINT')
        if not esm:
            torch_home = Path(os.environ.get('TORCH_HOME', Path(os.environ.get('XDG_CACHE_HOME', Path.home() / '.cache')) / 'torch'))
            esm = torch_home / 'hub/checkpoints/esm2_t33_650M_UR50D.pt'
        if Path(esm).name != 'esm2_t33_650M_UR50D.pt':
            raise ConfigurationError('ESM_MODEL', 'Use esm2_t33_650M_UR50D.pt and its adjacent contact-regression checkpoint')
        device = args.psichic_device or os.environ.get('PSICHIC_DEVICE', 'cpu')
        if device not in ('cpu', 'cuda:0'):
            raise ConfigurationError('DEVICE', 'Use cpu or cuda:0 within one explicitly assigned CUDA_VISIBLE_DEVICES allocation')
        screening = os.environ.get('PSICHIC_SCREENING_DIR', str(Path(__file__).resolve().parents[2] / 'psichic_plus_screening'))
        controls['PSICHIC_SCREENING_DIR'] = _path(screening)
        controls['PSICHIC_ESM_CHECKPOINT'] = _path(esm)
        p_options = dict(model_dir=_path(model), esm_checkpoint=_path(esm), device=device,
                         cache=_path(args.psichic_cache or os.environ.get('PSICHIC_CACHE', output / 'cache')),
                         batch_size=args.batch_size or 32, chunk_size=args.chunk_size, seed=args.seed,
                         call_profile='predictor-evidence-v1', preset='fast_screening', precision='fp32',
                         motifpool_dense_strategy='pyg_to_dense', protein_mincut_strategy='pyg_dense_mincut',
                         protein_dense_cache_strategy='none', loader_workers=0, ligand_disk_cache=False,
                         batching_strategy='input_order', controls=controls)
        profiles['psichic'] = probe(Engine.PSICHIC, sys.executable, p_options, controls)
    if uses_b:
        python = args.boltz_python or os.environ.get('BOLTZ_PYTHON')
        if not python:
            raise ConfigurationError('BOLTZ_REQUIRED', 'Provide --boltz-python or BOLTZ_PYTHON for isolated Python 3.12 with Boltz 2.2.1')
        native = {**(QUICK if args.preset == 'quick-check' else {}), **overrides, 'seed': args.seed,
                  'cache': _path(args.boltz_cache or os.environ.get('BOLTZ_CACHE', Path.home() / '.boltz'))}
        profile = probe(Engine.BOLTZ, _path(python), {'native': native}, controls)
        values = profile['resolved']
        if values['devices'] != 1 or values['accelerator'] not in ('cpu', 'gpu'):
            raise ConfigurationError('DEVICE', 'Local execution supports one assigned GPU or CPU, with --devices 1')
        for name in ('sampling_steps', 'diffusion_samples', 'sampling_steps_affinity', 'diffusion_samples_affinity',
                     'max_parallel_samples', 'preprocessing_threads', 'max_msa_seqs', 'num_subsampled_msa'):
            if values[name] is None or values[name] < 1:
                raise ConfigurationError('NATIVE_OPTION', f'{name} must be positive')
        if values['num_workers'] < 0 or values['recycling_steps'] < 0:
            raise ConfigurationError('NATIVE_OPTION', 'Worker and recycling counts cannot be negative')
        b_options = dict(native=values, internal_defaults=profile['internal_defaults'], preset=args.preset,
                         chunk_size=args.chunk_size, timeout=args.boltz_timeout, controls=controls)
        profiles['boltz'] = profile
    if (uses_p and p_options['device'].startswith('cuda')) or (uses_b and b_options['native']['accelerator'] == 'gpu'):
        allocation = controls['CUDA_VISIBLE_DEVICES']
        if not allocation or ',' in allocation or allocation in ('-1', 'all'):
            raise ConfigurationError('GPU_ALLOCATION', 'Set CUDA_VISIBLE_DEVICES to your one assigned GPU. No GPU is selected automatically.')
    settings = Settings(args.mode, **common, psichic_options=canonical(p_options) if p_options else None,
                        boltz_options=canonical(b_options) if b_options else None)
    runtime = {'schema_version': 1, 'preset': args.preset, 'profiles': profiles, 'controls': controls,
               'seed': args.seed, 'chunk_size': args.chunk_size, 'overrides': overrides,
               'candidate_sha256': None if future_population else checksum(source), 'source': str(source), 'id_column': args.id_column,
               'smiles_column': args.smiles_column, 'target_document': document, 'ligand_id': target.ligand_id,
               'target_assets': assets}
    return LocalPlan(output, source, args.id_column, args.smiles_column, count, retained_target, settings,
                     document, target.ligand_id, tuple(assets), runtime)


def retain_inputs(plan, run):
    directory = run.path / 'inputs'
    directory.mkdir()
    for item in (*plan.assets, {'path': str(plan.source), 'copy': 'candidates.csv', 'sha256': plan.runtime['candidate_sha256']}):
        destination = directory / item['copy']
        shutil.copyfile(item['path'], destination)
        if checksum(destination) != item['sha256']:
            raise ConfigurationError('INPUT_CHANGED', 'Input changed during run creation. Preserve this directory and create a new run.')
        with destination.open('rb') as handle:
            os.fsync(handle.fileno())
    write_file(directory / 'target.fasta', f'>target\n{plan.target.sequence}\n')
    write_file(directory / 'target.yaml', yaml.safe_dump(plan.document, sort_keys=False))
    write_file(run.path / 'local.json', canonical(plan.runtime) + '\n')
    write_file(run.path / 'local.sha256', checksum(run.path / 'local.json') + '\n')
    fsync_directory(directory)
    fsync_directory(run.path)


def verify_runtime(runtime, settings):
    for engine in settings.engines:
        previous = runtime['profiles'][engine.value]
        options = loads(getattr(settings, engine.value + '_options'))
        current = probe(engine, previous['executable'], options, runtime['controls'])
        if identity(current) != identity(previous) or (engine is Engine.BOLTZ and current['resolved'] != previous['resolved']):
            raise ConfigurationError('CREATE_NEW_RUN', f'{engine.value} source, weights, or environment changed.',
                                     'Create a new run. Saved observations cannot be mixed with this runtime.')
