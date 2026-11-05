"""Controlled subprocess observations. These fixtures do not run trained models."""

import csv
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tarfile
import time

import pytest

from applications.scoring import ScoringRun

ROOT = Path(__file__).resolve().parents[3]


def read(path):
    return json.loads(Path(path).read_text())


@pytest.fixture
def runtime(tmp_path):
    screening = tmp_path / 'controlled-screening'
    screening.mkdir()
    for name in ('featurizers', 'psichic_model', 'inference'):
        shutil.copyfile(ROOT / f'psichic_plus_screening/{name}.py', screening / f'native_{name}.py')
    (screening / 'featurizers.py').write_text('''from native_featurizers import *
from types import SimpleNamespace
import torch
import numpy as np
class Alphabet:
    def get_batch_converter(self):
        return None
esm = SimpleNamespace(pretrained=SimpleNamespace(load_model_and_alphabet=lambda path: (torch.nn.Linear(1, 1), Alphabet())))
def esm_extract(*args, **kwargs):
    return torch.zeros(2, 1280), np.zeros((2, 2)), None
def contact_map(contacts):
    return torch.zeros(2, 0, dtype=torch.long), torch.zeros(0)
''')
    (screening / 'psichic_model.py').write_text('''from native_psichic_model import *
class PSICHICPlusModel:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()
''')
    (screening / 'inference.py').write_text('''from native_inference import *
from types import SimpleNamespace
import os
import json
import time
from pathlib import Path

def build_loader(*args):
    return (SimpleNamespace(dataset=SimpleNamespace(ligands={})),)

def run_screening(frame, *args):
    if os.environ.get('CONTROLLED_LATE_SOURCE'):
        import importlib.util
        import sys
        spec = importlib.util.spec_from_file_location('unfingerprinted_runtime_extension', os.environ['CONTROLLED_LATE_SOURCE'])
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    if os.environ.get('CONTROLLED_DESCENDANT'):
        import subprocess
        import sys
        child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])
        Path(os.environ['CONTROLLED_DESCENDANT']).write_text(str(child.pid))
    with open(os.environ['CONTROLLED_CALLS'], 'a') as log:
        log.write(json.dumps({'engine': 'psichic', 'pid': os.getpid(), 'rows': frame.ID.tolist()}) + '\\n')
    os.write(1, b'controlled native fd stdout progress\\n')
    block = Path(os.environ['CONTROLLED_BLOCK'])
    while block.exists():
        time.sleep(.02)
    rows = []
    for index, row in frame.iterrows():
        native = dict(row)
        if row.Ligand == 'FAIL' and not Path(os.environ['CONTROLLED_RETRY']).exists():
            native.update(inference_status='failed', failure_stage='controlled', failure_reason='controlled failure')
        else:
            native.update(predicted_binding_affinity=9. if row.Ligand == 'CCN' else 6.,
                          reg_alpha=3. if row.Ligand == 'CCN' else 1., reg_beta=1.,
                          inference_status='success', diagnostic=float('nan'))
        if row.Ligand == 'DIST':
            native.update(predicted_binding_affinity=9., reg_alpha=float('inf'), reg_beta=1.)
        if row.Ligand == 'NUMERIC':
            native.update(predicted_binding_affinity=9., reg_alpha='3', reg_beta=1.)
        if row.Ligand == 'MISSING_DIST':
            native.update(predicted_binding_affinity=9., reg_alpha=pd.NA, reg_beta=1.)
        rows.append(native)
    return pd.DataFrame(rows, dtype=object), None
''')
    weights = tmp_path / 'weights'
    weights.mkdir()
    for name in ('config.json', 'degree.pt', 'model.pt', 'esm2_t33_650M_UR50D.pt', 'esm2_t33_650M_UR50D-contact-regression.pt'):
        (weights / name).write_text('controlled fixture, not trained weights')
    fasta = tmp_path / 'target.fasta'
    fasta.write_text('>test\nAC\n')
    candidates = tmp_path / 'molecules.csv'
    candidates.write_text('ID,SMILES,route\nsame,CCO,one\nsame,CCN,two\n,CCO,three\n')
    env = {**os.environ, 'PYTHONPATH': str(ROOT), 'PYTHONDONTWRITEBYTECODE': '1',
           'PSICHIC_SCREENING_DIR': str(screening), 'CONTROLLED_CALLS': str(tmp_path / 'calls.jsonl'),
           'CONTROLLED_BLOCK': str(tmp_path / 'block'), 'CONTROLLED_RETRY': str(tmp_path / 'retry')}
    args = ['rank', '--mode', 'psichic-only', '--candidates', str(candidates), '--fasta', str(fasta),
            '--msa-mode', 'single', '--model-dir', str(weights), '--esm-checkpoint', str(weights / 'esm2_t33_650M_UR50D.pt'),
            '--chunk-size', '2', '--batch-size', '2', '--output', str(tmp_path / 'run'), '--json']
    return {'env': env, 'args': args, 'root': tmp_path, 'run': tmp_path / 'run', 'csv': candidates, 'weights': weights}


def cli(runtime, args=None, timeout=120):
    result = subprocess.run([sys.executable, '-B', '-m', 'applications.scoring', *(args or runtime['args'])],
                            env=runtime['env'], cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    record = runtime['root'] / 'commands.jsonl'
    with record.open('a') as log:
        log.write(json.dumps({'argv': result.args, 'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}) + '\n')
    return result, json.loads(result.stdout)


def calls(runtime):
    path = runtime['root'] / 'calls.jsonl'
    return path.read_text().splitlines() if path.exists() else []


def test_real_module_validation_has_no_inference_or_run_creation(runtime):
    result, value = cli(runtime, [*runtime['args'], '--validate-only'])
    assert result.returncode == 0, value.get("error", (result.stdout, result.stderr))
    assert value['status'] == 'validated'
    assert value['resolved']['evaluations'] == {'psichic': 3, 'boltz': 0}
    assert not runtime['run'].exists()
    assert calls(runtime) == []


def test_success_freezes_population_and_resume_does_not_execute_again(runtime):
    result, value = cli(runtime)
    assert result.returncode == 0, value.get("error", (result.stdout, result.stderr))
    assert value['status'] == 'complete' and value['counts']['ranked'] == 3
    assert len(calls(runtime)) == 2
    rows = read(value['reports']['results.json'])['rows']
    assert [r['entry']['id'] for r in rows] == ['same', 'same', '']
    assert [r['entry']['metadata']['route'] for r in rows] == ['one', 'two', 'three']
    assert value['top'][0]['entry']['smiles'] == 'CCN'
    before = Path(value['reports']['results.json']).read_bytes()
    runtime['csv'].write_text('malformed original changed after freezing')
    result, after = cli(runtime, ['resume', '--run', str(runtime['run']), '--json'])
    assert result.returncode == 0, (result.stdout, result.stderr)
    assert after['report_version'] == 1 and len(calls(runtime)) == 2
    assert Path(after['reports']['results.json']).read_bytes() == before
    for row in rows:
        raw = next(a for a in row['psichic']['artifacts'] if a['role'] == 'raw_psichic')
        native = read(Path(value['reports']['results.json']).parent / raw['link'])['native']
        assert native['diagnostic'] == {'nonfinite_float': 'nan'}


def test_partial_explicit_retry_keeps_prior_report(runtime):
    runtime['csv'].write_text('ID,SMILES\none,CCO\ntwo,FAIL\n')
    result, first = cli(runtime)
    assert result.returncode == 1 and first['status'] == 'partial', (result.stdout, result.stderr)
    before = Path(first['reports']['results.json']).read_bytes()
    result, same = cli(runtime, ['resume', '--run', str(runtime['run']), '--json'])
    assert result.returncode == 1 and len(calls(runtime)) == 1
    Path(runtime['env']['CONTROLLED_RETRY']).touch()
    result, final = cli(runtime, ['retry', '--run', str(runtime['run']), '--failed', '--json'])
    assert result.returncode == 0, (result.stdout, result.stderr)
    assert final['report_version'] == 2 and len(calls(runtime)) == 2
    assert Path(first['reports']['results.json']).read_bytes() == before


def test_changed_weight_rejected_before_inference(runtime):
    result, _ = cli(runtime)
    assert result.returncode == 0
    (runtime['weights'] / 'model.pt').write_text('changed bytes')
    result, value = cli(runtime, ['resume', '--run', str(runtime['run']), '--json'])
    assert result.returncode == 2 and value['error']['code'] == 'CREATE_NEW_RUN'
    assert len(calls(runtime)) == 2


@pytest.mark.parametrize('extra', [['--msa-mode', 'server'], ['--typo'], ['--boltz-python', '/unused'],
                                  ['--psichic-weight', '1'], ['--', '--use_msa_server'],
                                  ['--', '--api_key_value=do-not-record']])
def test_json_configuration_errors_before_output(runtime, extra):
    result, value = cli(runtime, [*runtime['args'], *extra])
    assert result.returncode == 2 and value['status'] == 'configuration-error'
    assert not runtime['run'].exists()
    assert 'do-not-record' not in result.stdout + result.stderr


@pytest.mark.parametrize('content', ['ID,ID,SMILES\na,b,CCO\n', 'ID,SMILES\n', 'ID,SMILES,x\na,CCO\n'])
def test_rejects_malformed_stream_without_creating_output(runtime, content):
    runtime['csv'].write_text(content)
    result, value = cli(runtime)
    assert result.returncode == 2, result.stdout
    assert not runtime['run'].exists()


def wait_for(predicate, timeout=60):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(.03)
    raise AssertionError('Owned process did not reach the observable boundary')


def test_killed_coordinator_does_not_redispatch_live_owned_worker(runtime):
    block = Path(runtime['env']['CONTROLLED_BLOCK'])
    block.touch()
    process = subprocess.Popen([sys.executable, '-B', '-m', 'applications.scoring', *runtime['args']],
                               env=runtime['env'], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    owner = None
    try:
        wait_for(lambda: calls(runtime))
        invocation = next((runtime['run'] / 'local/invocations').iterdir())
        owner = read(invocation / 'owner.json')
        process.kill()
        process.communicate(timeout=10)
        result, value = cli(runtime, ['resume', '--run', str(runtime['run']), '--json'])
        assert result.returncode == 1 and value['status'] == 'waiting'
        assert len(calls(runtime)) == 1
        block.unlink()
        wait_for(lambda: (invocation / 'completion.json').exists())
        result, value = cli(runtime, ['resume', '--run', str(runtime['run']), '--json'])
        assert result.returncode == 0, (result.stdout, result.stderr)
        assert len(calls(runtime)) == 2
    finally:
        block.unlink(missing_ok=True)
        if process.poll() is None:
            process.kill()
            process.communicate()
        if owner:
            from applications.scoring.processes import alive, signal_owned
            if alive(owner):
                signal_owned(owner, signal.SIGTERM)
                wait_for(lambda: not alive(owner))


@pytest.fixture
def boltz_runtime(runtime):
    import base64
    import hashlib

    root = runtime['root'] / 'controlled-boltz'
    packages = root / 'packages'
    (packages / 'boltz').mkdir(parents=True)
    (packages / 'pytorch_lightning').mkdir()
    (packages / 'boltz/__init__.py').write_text('__version__ = "2.2.1"\n')
    (packages / 'pytorch_lightning/__init__.py').write_text('controlled_fixture = True\n')
    source = '''import json
import os
from pathlib import Path
import time
from dataclasses import dataclass
import click
import yaml
@dataclass
class Boltz2DiffusionParams:
    step_scale: float = 1.5
PairformerArgsV2 = MSAModuleArgs = BoltzSteeringParams = Boltz2DiffusionParams

def callback(**params):
    import psutil
    prior = Path(os.environ['CONTROLLED_CALLS'])
    if prior.exists():
        for line in prior.read_text().splitlines():
            observation = json.loads(line)
            if observation['engine'] == 'psichic':
                assert not psutil.pid_exists(observation['pid']), 'PSICHIC remained resident at Boltz start'
    with open(os.environ['CONTROLLED_CALLS'], 'a') as log:
        log.write(json.dumps({'engine': 'boltz', 'pid': os.getpid(), 'allocation': os.environ.get('CUDA_VISIBLE_DEVICES')}) + '\\n')
    os.write(1, b'controlled native Boltz fd output\\n')
    while Path(os.environ['CONTROLLED_BLOCK']).exists():
        time.sleep(.02)
    data = Path(params['data'])
    destination = Path(params['out_dir']) / ('boltz_results_' + data.stem) / 'predictions'
    for index, path in enumerate(sorted(data.glob('*.yaml'))):
        doc = yaml.safe_load(path.read_text())
        smiles = next(s['ligand']['smiles'] for s in doc['sequences'] if 'ligand' in s)
        if smiles == 'MISSING':
            continue
        output = destination / path.stem
        output.mkdir(parents=True)
        (output / ('affinity_' + path.stem + '.json')).write_text(json.dumps({
            'affinity_pred_value': -float(index), 'affinity_probability_binary': .8,
            'diagnostic': 'controlled native fixture'}))
        if smiles != 'NO_STRUCTURE':
            (output / (path.stem + '_model_0.cif')).write_text('controlled structure fixture\\n')
    raise SystemExit(int(os.environ.get('CONTROLLED_BOLTZ_EXIT', '0')))

values = {
    'out_dir': (str, './'), 'cache': (str, '~/.boltz'), 'checkpoint': (str, None),
    'affinity_checkpoint': (str, None), 'devices': (int, 1), 'accelerator': (str, 'gpu'),
    'recycling_steps': (int, 3), 'sampling_steps': (int, 200), 'diffusion_samples': (int, 1),
    'max_parallel_samples': (int, 5), 'step_scale': (float, None), 'output_format': (str, 'mmcif'),
    'num_workers': (int, 2), 'seed': (int, None), 'model': (str, 'boltz2'), 'method': (str, None),
    'preprocessing_threads': (int, 8), 'sampling_steps_affinity': (int, 200),
    'diffusion_samples_affinity': (int, 5), 'max_msa_seqs': (int, 8192), 'num_subsampled_msa': (int, 1024),
    'msa_server_url': (str, 'https://unused.invalid'), 'msa_server_username': (str, None),
    'msa_server_password': (str, None), 'msa_pairing_strategy': (str, 'greedy'),
    'api_key_header': (str, None), 'api_key_value': (str, None),
}
params = [click.Argument(['data'])]
for name, (kind, default) in values.items():
    flag = '--preprocessing-threads' if name == 'preprocessing_threads' else '--' + name
    params.append(click.Option([flag], type=kind, default=default))
for name in ('write_full_pae', 'write_full_pde', 'override', 'use_msa_server', 'use_potentials',
             'affinity_mw_correction', 'subsample_msa', 'no_kernels', 'write_embeddings'):
    params.append(click.Option(['--' + name], is_flag=True))
predict = click.Command('predict', params=params, callback=callback)
'''
    (packages / 'boltz/main.py').write_text(source)
    dist = packages / 'boltz-2.2.1.dist-info'
    dist.mkdir()
    (dist / 'METADATA').write_text('Metadata-Version: 2.1\nName: boltz\nVersion: 2.2.1\n')
    with (dist / 'RECORD').open('w', newline='') as handle:
        writer = csv.writer(handle)
        for path in (packages / 'boltz').glob('*.py'):
            blob = path.read_bytes()
            digest = base64.urlsafe_b64encode(hashlib.sha256(blob).digest()).rstrip(b'=').decode()
            writer.writerow([str(path.relative_to(packages)), 'sha256=' + digest, len(blob)])
    binary = root / 'bin'
    binary.mkdir()
    python = binary / 'python'
    python.write_text(f'#!{sys.executable}\nimport sys, runpy\nsys.executable = {str(python)!r}\nsys.path.insert(0, {str(packages)!r})\nsys.argv.pop(0)\nif sys.argv[0] == "-B": sys.argv.pop(0)\nrunpy.run_path(sys.argv[0], run_name="__main__")\n')
    python.chmod(0o755)
    (binary / 'boltz').write_text('# controlled native executable marker\n')
    cache = root / 'cache'
    (cache / 'mols').mkdir(parents=True)
    for name in ('boltz2_conf.ckpt', 'boltz2_aff.ckpt', 'mols/ALA.pkl', 'mols/CYS.pkl'):
        (cache / name).write_text('controlled cached asset, not model data')
    with tarfile.open(cache / 'mols.tar', 'w') as archive:
        archive.add(cache / 'mols', arcname='mols')
    runtime['boltz_python'] = str(python)
    runtime['boltz_cache'] = str(cache)
    return runtime


def mode_args(runtime, mode, extra=()):
    args = runtime['args'][:]
    args[args.index('--mode') + 1] = mode
    if mode == 'boltz-only':
        for name in ('--model-dir', '--esm-checkpoint', '--batch-size'):
            start = args.index(name)
            del args[start:start + 2]
    if 'pyramid' in mode:
        args += ['--shortlist-size', '2']
    return [*args, '--boltz-python', runtime['boltz_python'], '--boltz-cache', runtime['boltz_cache'],
            *extra, '--', '--accelerator', 'cpu']


@pytest.mark.parametrize('mode', ['boltz-only', 'full-ensemble', 'pyramid-ensemble', 'pyramid-refinement'])
def test_all_boltz_modes_validate_without_inference(boltz_runtime, mode):
    result, value = cli(boltz_runtime, mode_args(boltz_runtime, mode, ['--validate-only']))
    assert result.returncode == 0, value.get('error')
    assert value['resolved']['evaluations']['boltz'] == (2 if 'pyramid' in mode else 3)
    assert not boltz_runtime['run'].exists() and not calls(boltz_runtime)
    options = json.loads(value['resolved']['settings']['boltz_options'])['native']
    assert options['sampling_steps'] == 200 and options['diffusion_samples_affinity'] == 5
    assert options['preprocessing_threads'] == 8


def test_incomplete_extracted_molecule_cache_rejected_before_model(boltz_runtime):
    (Path(boltz_runtime['boltz_cache']) / 'mols/CYS.pkl').unlink()
    result, value = cli(boltz_runtime, mode_args(boltz_runtime, 'boltz-only', ['--validate-only']))
    assert result.returncode == 2 and value['error']['code'] == 'RUNTIME_UNAVAILABLE', value
    assert 'CYS.pkl' in value['error']['message']
    assert not boltz_runtime['run'].exists() and not calls(boltz_runtime)


def test_human_validation_displays_resolved_ranking_contract(boltz_runtime):
    args = mode_args(boltz_runtime, 'pyramid-ensemble', ['--validate-only'])
    args.remove('--json')
    result = subprocess.run([sys.executable, '-B', '-m', 'applications.scoring', *args],
                            env=boltz_runtime['env'], cwd=ROOT, capture_output=True, text=True, timeout=120)
    (boltz_runtime['root'] / 'human-validation.txt').write_text(result.stdout)
    assert result.returncode == 0, result.stderr
    for expected in ('concentration-v1', 'affinity', 'K=2', 'distribution', 'psichic=3', 'boltz=2'):
        assert expected in result.stdout
    assert not boltz_runtime['run'].exists() and not calls(boltz_runtime)


def test_concentration_exact_native_artifacts_and_sequential_residency(boltz_runtime):
    result, value = cli(boltz_runtime, mode_args(boltz_runtime, 'full-ensemble', ['--preset', 'quick-check']))
    assert result.returncode == 0, (value.get('error'), result.stderr)
    assert value['ranking']['method'] == 'concentration-v1'
    assert value['counts']['ranked'] == 3
    observations = [json.loads(line) for line in calls(boltz_runtime)]
    assert [v['engine'] for v in observations] == ['psichic', 'psichic', 'boltz', 'boltz']
    import psutil
    for observation in observations:
        assert not psutil.pid_exists(observation['pid'])
    rows = read(value['reports']['results.json'])['rows']
    assert all(r['psichic']['raw']['distribution']['state'] == 'valid' for r in rows)
    for row in rows:
        artifacts = {a['role']: Path(value['reports']['results.json']).parent / a['link'] for a in row['boltz']['artifacts']}
        assert artifacts['structure'].read_text() == 'controlled structure fixture\n'
        assert read(artifacts['native_affinity']) == row['boltz']['raw']['native']
        assert read(artifacts['execution'])['returncode'] == 0
        mapping = read(artifacts['native_mapping'])
        assert row['entry']['key'] in {m['entry_key'] for m in mapping}
        assert all(len(m['native_key']) == 12 and len(m['entry_key']) == 16 for m in mapping)


def test_shortage_starts_no_boltz(boltz_runtime):
    boltz_runtime['csv'].write_text('ID,SMILES\none,FAIL\ntwo,CCO\n')
    result, value = cli(boltz_runtime, mode_args(boltz_runtime, 'pyramid-ensemble'))
    assert result.returncode == 1 and value['status'] == 'shortage', (value, result.stderr)
    assert value['shortage'] == {'eligible': 1, 'required': 2}
    assert [json.loads(line)['engine'] for line in calls(boltz_runtime)] == ['psichic']


def test_native_partial_crash_salvages_only_complete_outputs(boltz_runtime):
    boltz_runtime['csv'].write_text('ID,SMILES\none,CCO\ntwo,NO_STRUCTURE\n')
    boltz_runtime['env']['CONTROLLED_BOLTZ_EXIT'] = '7'
    result, value = cli(boltz_runtime, mode_args(boltz_runtime, 'boltz-only'))
    assert result.returncode == 1 and value['status'] == 'waiting', (value, result.stderr)
    run = ScoringRun.open(boltz_runtime['run'])
    states = [v.state.value for v in run.diagnostics()]
    assert states == ['succeeded', 'submitted']
    receipt = next((run.path / 'receipts').glob('*.json'))
    saved = read(receipt)
    execution = next(a for a in saved['artifacts'] if a['role'] == 'execution')
    assert read(run.path / 'blobs' / execution['sha256'])['returncode'] == 7


def test_failed_structure_is_not_success(boltz_runtime):
    boltz_runtime['csv'].write_text('ID,SMILES\none,NO_STRUCTURE\n')
    result, value = cli(boltz_runtime, mode_args(boltz_runtime, 'boltz-only'))
    assert result.returncode == 1 and value['counts']['boltz_failed'] == 1, (value, result.stderr)
    assert value['counts']['ranked'] == 0


def test_receipt_recovery_rejects_changed_native_name_mapping(boltz_runtime):
    result, value = cli(boltz_runtime, mode_args(boltz_runtime, 'boltz-only'))
    assert result.returncode == 0, value
    invocation = next(p for p in (boltz_runtime['run'] / 'local/invocations').iterdir()
                      if len(read(p / 'request.json')['tasks']) == 2)
    recovery = boltz_runtime['root'] / 'changed-mapping'
    shutil.copytree(invocation, recovery)
    (recovery / 'published.json').unlink()
    request = read(recovery / 'request.json')
    request['mapping'][0]['native_key'] = request['mapping'][1]['native_key']
    (recovery / 'request.json').write_text(json.dumps(request))
    process = subprocess.run([sys.executable, '-B', '-m', 'applications.scoring.worker', 'publish', str(recovery)],
                             env=boltz_runtime['env'], cwd=ROOT, capture_output=True, text=True, timeout=30)
    (boltz_runtime['root'] / 'recovery.stderr.log').write_text(process.stderr)
    assert process.returncode == 1 and 'Native input mapping changed' in process.stderr
    assert len(calls(boltz_runtime)) == 2


@pytest.mark.parametrize('native', [False, True], ids=['psichic', 'boltz'])
def test_killed_owner_stops_model_group_and_resume_creates_new_attempt(runtime, boltz_runtime, native):
    from applications.scoring.processes import alive

    block = Path(runtime['env']['CONTROLLED_BLOCK'])
    block.touch()
    args = mode_args(boltz_runtime, 'boltz-only') if native else runtime['args']
    process = subprocess.Popen([sys.executable, '-B', '-m', 'applications.scoring', *args],
                               env=runtime['env'], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        wait_for(lambda: calls(runtime))
        invocation = next((runtime['run'] / 'local/invocations').iterdir())
        owner, child = read(invocation / 'owner.json'), read(invocation / 'child.json')
        os.kill(owner['pid'], signal.SIGKILL)
        process.communicate(timeout=20)
        wait_for(lambda: alive(child) is False)
        assert not (invocation / 'completion.json').exists()
        block.unlink()
        result, value = cli(runtime, ['resume', '--run', str(runtime['run']), '--json'])
        assert result.returncode == 0, (value, result.stderr)
        assert len(calls(runtime)) == 3
        assert len(list((runtime['run'] / 'local/invocations').iterdir())) == 3
        assert alive(owner) is False and alive(child) is False
    finally:
        block.unlink(missing_ok=True)
        if process.poll() is None:
            process.terminate()
            process.communicate(timeout=20)


@pytest.mark.parametrize('sig', [signal.SIGINT, signal.SIGTERM])
def test_interruption_reaps_owned_processes_and_resumes(runtime, sig):
    from applications.scoring.processes import alive

    block = Path(runtime['env']['CONTROLLED_BLOCK'])
    block.touch()
    process = subprocess.Popen([sys.executable, '-B', '-m', 'applications.scoring', *runtime['args']],
                               env=runtime['env'], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        wait_for(lambda: calls(runtime))
        invocation = next((runtime['run'] / 'local/invocations').iterdir())
        owner, child = read(invocation / 'owner.json'), read(invocation / 'child.json')
        process.send_signal(sig)
        stdout, stderr = process.communicate(timeout=20)
        assert process.returncode == 1 and json.loads(stdout)['status'] == 'waiting', stderr
        assert alive(owner) is False and alive(child) is False
        assert read(invocation / 'completion.json')['returncode'] < 0
        block.unlink()
        result, value = cli(runtime, ['resume', '--run', str(runtime['run']), '--json'])
        assert result.returncode == 0, (value, result.stderr)
    finally:
        block.unlink(missing_ok=True)
        if process.poll() is None:
            process.terminate()
            process.communicate(timeout=20)


def test_unrelated_pid_start_identity_is_not_signaled():
    from applications.scoring.processes import alive, process_identity, signal_owned

    process = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], start_new_session=True)
    try:
        identity = process_identity(process.pid)
        wrong = {**identity, 'started': identity['started'] - 1}
        assert alive(wrong) is False
        assert signal_owned(wrong, signal.SIGKILL) is False
        assert process.poll() is None
    finally:
        process.terminate()
        process.wait(timeout=10)


def test_retry_transaction_does_not_reset_boltz_before_prohibited_psichic(make_run, tmp_path):
    from .conftest import prepare, transfer
    from applications.scoring import Engine, StateError

    run = make_run('pyramid-refinement', n=2, shortlist_size=1)
    one, two = prepare(run)
    run.accept(transfer(tmp_path / 'one', one, affinity=9))
    run.accept(transfer(tmp_path / 'two', two, failure='controlled failure'))
    run.select()
    task, = prepare(run, Engine.BOLTZ)
    run.accept(transfer(tmp_path / 'boltz', task, failure='controlled failure'))
    before = tuple(run.diagnostics())
    with pytest.raises(StateError, match='frozen'):
        run.retry_many([(task.entry_key, Engine.BOLTZ), (two.entry_key, Engine.PSICHIC)])
    assert tuple(run.diagnostics()) == before
    with pytest.raises(StateError, match='frozen'):
        run.retry_failed()
    assert tuple(run.diagnostics()) == before
    assert run.retry_failed(engine=Engine.BOLTZ) == 1


def test_installed_console_runs_outside_checkout(runtime, tmp_path):
    install = tmp_path / 'installed'
    source = tmp_path / 'build-source'
    shutil.copytree(ROOT / 'applications', source / 'applications', ignore=shutil.ignore_patterns('build', '*.egg-info', '__pycache__'))
    shutil.copytree(ROOT / 'psichic_plus_screening', source / 'psichic_plus_screening',
                    ignore=shutil.ignore_patterns('pretrained_weights', '__pycache__', 'tests'))
    build = subprocess.run([sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check', '--no-deps',
                            '--no-build-isolation', '--no-compile', '--target', str(install), str(source / 'applications')],
                           capture_output=True, text=True, timeout=120)
    (tmp_path / 'install.stdout.log').write_text(build.stdout)
    (tmp_path / 'install.stderr.log').write_text(build.stderr)
    assert build.returncode == 0, build.stdout + build.stderr
    outside = tmp_path / 'outside'
    outside.mkdir()
    env = {**runtime['env'], 'PYTHONPATH': str(install)}
    result = subprocess.run([str(install / 'bin/psichic'), '--help'], env=env, cwd=outside,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0 and 'resume' in result.stdout
    origins = subprocess.run([sys.executable, '-c', 'import applications.scoring, importlib.util; '
        'print(applications.scoring.__file__); print(importlib.util.find_spec("psichic_plus_screening").submodule_search_locations[0])'],
        env=env, cwd=outside, capture_output=True, text=True, timeout=30)
    assert str(ROOT) not in origins.stdout
    assert str(install) in origins.stdout
    assert not list(install.rglob('model.pt'))
    (tmp_path / 'installed-imports.log').write_text(origins.stdout)
    result = subprocess.run([str(install / 'bin/psichic'), *runtime['args']], env=env, cwd=outside,
                            capture_output=True, text=True, timeout=180)
    (tmp_path / 'installed-cli.stdout.json').write_text(result.stdout)
    (tmp_path / 'installed-cli.stderr.log').write_text(result.stderr)
    value = json.loads(result.stdout)
    assert result.returncode == 0, (value, result.stderr)
    assert value['counts']['ranked'] == 3
    installed_source = next((runtime['run'] / 'local/invocations').iterdir()) / 'profile.json'
    assert all(not f['path'].startswith(str(ROOT)) for f in read(installed_source)['source'])


@pytest.mark.parametrize('extra', [
    ['--psichic-weight', '1'], ['--boltz-weight', '1'], ['--rank-by', 'affinity_probability_binary'],
])
def test_concentration_rejects_explicit_incompatible_settings(boltz_runtime, extra):
    args = mode_args(boltz_runtime, 'full-ensemble', extra)
    result, value = cli(boltz_runtime, args)
    assert result.returncode == 2 and value['status'] == 'configuration-error'
    assert not boltz_runtime['run'].exists() and not calls(boltz_runtime)


@pytest.mark.parametrize('native', [['--sampling_steps', '2', '--sampling_steps=3'],
                                    ['--sampl', '2'], ['--cache', '/unused'], ['--seed', '99'],
                                    ['--write_full_pae=false'], ['--use_msa_server=true']])
def test_native_overrides_reject_collision_and_unsupported_syntax(boltz_runtime, native):
    args = mode_args(boltz_runtime, 'boltz-only')
    args += native
    result, value = cli(boltz_runtime, args)
    assert result.returncode == 2 and value['error']['code'] == 'NATIVE_OPTION'
    assert not boltz_runtime['run'].exists() and not calls(boltz_runtime)


def test_quick_preset_expansion_retains_explicit_overrides(boltz_runtime):
    args = mode_args(boltz_runtime, 'boltz-only', ['--preset', 'quick-check', '--validate-only'])
    result, value = cli(boltz_runtime, [*args, '--sampling_steps', '3'])
    assert result.returncode == 0, value
    options = json.loads(value['resolved']['settings']['boltz_options'])['native']
    assert {k: options[k] for k in ('recycling_steps', 'sampling_steps', 'diffusion_samples',
            'sampling_steps_affinity', 'diffusion_samples_affinity', 'num_workers', 'preprocessing_threads')} == {
        'recycling_steps': 1, 'sampling_steps': 3, 'diffusion_samples': 1, 'sampling_steps_affinity': 2,
        'diffusion_samples_affinity': 1, 'num_workers': 0, 'preprocessing_threads': 1}
    assert value['resolved']['overrides']['sampling_steps'] == '3'
    assert not boltz_runtime['run'].exists() and not calls(boltz_runtime)


def test_installed_boltz_record_tampering_rejected_before_model(boltz_runtime):
    package = Path(boltz_runtime['boltz_python']).parent.parent / 'packages/boltz/main.py'
    with package.open('a') as handle:
        handle.write('\nchanged = True\n')
    result, value = cli(boltz_runtime, mode_args(boltz_runtime, 'boltz-only', ['--validate-only']))
    assert result.returncode == 2 and 'RECORD' in value['error']['message']
    assert not calls(boltz_runtime) and not boltz_runtime['run'].exists()


def test_offline_boundary_denies_download_before_connection():
    code = '''from applications.scoring.runtime import offline_only
import urllib.request
offline_only()
try:
    urllib.request.urlopen('https://example.invalid/model')
except RuntimeError as error:
    assert 'forbids network access' in str(error)
else:
    raise AssertionError('network request was allowed')
'''
    result = subprocess.run([sys.executable, '-B', '-c', code], cwd=ROOT, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr


def test_distribution_diagnostics_round_trip_from_public_native_seam(runtime):
    runtime['csv'].write_text('ID,SMILES\nretained,DIST\n')
    result, value = cli(runtime, [*runtime['args'], '--psichic-output', 'distribution'])
    assert result.returncode == 1 and value['status'] == 'partial', value
    row, = read(value['reports']['results.json'])['rows']
    assert row['psichic']['raw']['raw_affinity'] == 9.
    assert row['psichic']['raw']['distribution']['raw_alpha'] == 'inf'
    artifact = next(a for a in row['psichic']['artifacts'] if a['role'] == 'raw_psichic')
    native = read(Path(value['reports']['results.json']).parent / artifact['link'])['native']
    assert native['reg_alpha'] == {'nonfinite_float': 'inf'}
    assert 'Alpha' in Path(value['reports']['summary.md']).read_text()


def test_changed_profile_source_rejected_before_expensive_work(runtime):
    result, value = cli(runtime)
    assert result.returncode == 0, value
    source = Path(runtime['env']['PSICHIC_SCREENING_DIR']) / 'inference.py'
    with source.open('a') as handle:
        handle.write('\nchanged_call_profile = True\n')
    result, value = cli(runtime, ['resume', '--run', str(runtime['run']), '--json'])
    assert result.returncode == 2 and value['error']['code'] == 'CREATE_NEW_RUN'
    assert len(calls(runtime)) == 2


def test_late_external_source_without_preflight_hash_cannot_be_accepted(runtime):
    external = runtime['root'] / 'external-runtime.py'
    external.write_text('loaded = True\n')
    runtime['env']['CONTROLLED_LATE_SOURCE'] = str(external)
    result, value = cli(runtime)
    assert result.returncode == 1 and value['status'] == 'waiting', value
    invocation, = (runtime['run'] / 'local/invocations').iterdir()
    assert 'Loaded source was not fingerprinted' in (invocation / 'stderr.log').read_text()
    assert not list((invocation / 'transfers').iterdir())


def test_numeric_string_distribution_is_diagnostic_not_a_numeric_parameter(runtime):
    runtime['csv'].write_text('ID,SMILES\nretained,NUMERIC\n')
    result, value = cli(runtime, [*runtime['args'], '--psichic-output', 'distribution'])
    assert result.returncode == 1 and value['status'] == 'partial', (value, result.stderr)
    row, = read(value['reports']['results.json'])['rows']
    assert row['psichic']['raw']['raw_affinity'] == 9.
    assert row['psichic']['raw']['distribution']['raw_alpha'] == "'3'"


def test_pandas_missing_distribution_preserves_finite_affinity(runtime):
    runtime['csv'].write_text('ID,SMILES\nretained,MISSING_DIST\n')
    result, value = cli(runtime, [*runtime['args'], '--psichic-output', 'distribution'])
    assert result.returncode == 1 and value['status'] == 'partial', value
    row, = read(value['reports']['results.json'])['rows']
    assert row['psichic']['raw']['raw_affinity'] == 9.
    assert row['psichic']['raw']['distribution']['raw_alpha'] == 'None'
    raw = next(a for a in row['psichic']['artifacts'] if a['role'] == 'raw_psichic')
    assert read(Path(value['reports']['results.json']).parent / raw['link'])['native']['reg_alpha'] is None


def test_owner_and_model_hard_kill_leave_no_descendant(runtime):
    import psutil
    from applications.scoring.processes import alive, process_identity, signal_owned

    block = Path(runtime['env']['CONTROLLED_BLOCK'])
    block.touch()
    marker = runtime['root'] / 'descendant.pid'
    runtime['env']['CONTROLLED_DESCENDANT'] = str(marker)
    process = subprocess.Popen([sys.executable, '-B', '-m', 'applications.scoring', *runtime['args']],
                               env=runtime['env'], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    descendant = owner = model = None
    try:
        wait_for(lambda: marker.exists())
        invocation = next((runtime['run'] / 'local/invocations').iterdir())
        owner, model = read(invocation / 'owner.json'), read(invocation / 'child.json')
        descendant = process_identity(int(marker.read_text()))
        signal_owned(owner, signal.SIGSTOP)
        wait_for(lambda: psutil.Process(owner['pid']).status() == psutil.STATUS_STOPPED)
        signal_owned(model, signal.SIGKILL)
        signal_owned(owner, signal.SIGKILL)
        process.communicate(timeout=20)
        wait_for(lambda: alive(descendant) is False, timeout=5)
    finally:
        for item in (descendant, model, owner):
            if item:
                signal_owned(item, signal.SIGKILL)
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=10)
        block.unlink(missing_ok=True)


def test_modified_frozen_options_require_new_run_before_inference(runtime):
    result, value = cli(runtime)
    assert result.returncode == 0, value
    path = runtime['run'] / 'local.json'
    config = read(path)
    config['seed'] = 111
    path.write_text(json.dumps(config))
    result, value = cli(runtime, ['resume', '--run', str(runtime['run']), '--json'])
    assert result.returncode == 2 and value['error']['code'] == 'CREATE_NEW_RUN'
    assert len(calls(runtime)) == 2


def test_gpu_requires_explicit_allocation_without_selecting_a_device(runtime):
    runtime['env'].pop('CUDA_VISIBLE_DEVICES', None)
    result, value = cli(runtime, [*runtime['args'], '--psichic-device', 'cuda:0', '--validate-only'])
    assert result.returncode == 2 and 'CUDA_VISIBLE_DEVICES' in value['error']['message']
    assert not runtime['run'].exists() and not calls(runtime)


def test_single_native_target_retains_template_bytes_and_original_identity(boltz_runtime):
    import yaml

    template = boltz_runtime['root'] / 'template.cif'
    template.write_text('controlled original template\n')
    native = boltz_runtime['root'] / 'target.yaml'
    native.write_text(yaml.safe_dump({'version': 1, 'sequences': [
        {'protein': {'id': 'A', 'sequence': 'AC'}}, {'ligand': {'id': 'B', 'smiles': 'C'}}],
        'templates': [{'cif': 'template.cif', 'chain_id': 'A'}]}))
    args = mode_args(boltz_runtime, 'boltz-only')
    index = args.index('--fasta')
    args[index:index + 2] = ['--target-yaml', str(native), '--ligand-id', 'B']
    result, first = cli(boltz_runtime, args)
    assert result.returncode == 0, (first, result.stderr)
    assert (boltz_runtime['run'] / 'inputs/template_0000.cif').read_text() == 'controlled original template\n'
    template.write_text('changed original')
    native.write_text('changed original target')
    before = len(calls(boltz_runtime))
    result, value = cli(boltz_runtime, ['resume', '--run', str(boltz_runtime['run']), '--json'])
    assert result.returncode == 0, (value, result.stderr)
    assert len(calls(boltz_runtime)) == before
    assert value['report_version'] == 1
