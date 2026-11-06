"""User-facing subprocess checks with controlled predictions and native search."""

import csv
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

import pytest

from applications.scorer import BetaDistribution
from applications.scorer.boltz2.models import BoltzScore, Candidate, EngineRun, EnsembleMethod, Metric, Mode, PSICHICScore, Plan
from applications.scorer.boltz2.workflow import run_workflow
from applications.scoring.processes import alive, process_identity
from applications.scoring.tests.test_executor_cli import calls, cli as shared_cli, mode_args, read, wait_for

pytest_plugins = ['applications.scoring.tests.test_executor_cli']

ROOT = Path(__file__).resolve().parents[3]
EXAMPLE = ROOT / 'applications/generator/example'


def products(runtime):
    source = runtime['root'] / 'native'
    source.mkdir()
    (source / 'molecules.csv').write_text(
        'node_id,smiles,score,num_reactions,reaction_1_id,building_block_1_1_id,extra\n'
        '../../unsafe,CCO,99.000,1,7.0,0001,one\n'
        '../../unsafe,CCN,88,2,22,-1,two\n'
        ',CCO,77,1,7.0,0002,three\n')
    return source


def rank_args(runtime, source, *, mode=None, extra=()):
    args = mode_args(runtime, mode, list(extra)) if mode else [*runtime['args'], *extra]
    index = args.index('--candidates')
    args[index:index + 2] = ['--source-run', str(source)]
    return args


def cli(runtime, args, timeout=180):
    result = subprocess.run([sys.executable, '-B', '-m', 'applications.generator', *args],
                            env=runtime['env'], cwd=runtime['root'], capture_output=True, text=True, timeout=timeout)
    with (runtime['root'] / 'generator-commands.jsonl').open('a') as log:
        log.write(json.dumps({'argv': result.args, 'returncode': result.returncode,
                             'stdout': result.stdout, 'stderr': result.stderr}) + '\n')
    return result, json.loads(result.stdout)


def native_args(runtime):
    native = os.environ.get('SYNTHEMOL_RUNTIME')
    if native:
        runtime['env']['PYTHONPATH'] += os.pathsep + native
    else:
        pytest.skip('Set SYNTHEMOL_RUNTIME to a private installed or read-only patched native package')
    blocks = runtime['root'] / 'blocks.csv'
    with (EXAMPLE / 'building_blocks.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    with blocks.open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=['reagent_id', 'smiles', 'score'])
        writer.writeheader()
        writer.writerows({**row, 'score': '6.0'} for row in rows)
    return ['--search_type', 'mcts', '--save_dir', str(runtime['root'] / 'native'),
            '--score_types', 'psichic_plus', '--score_names', 'psichic_affinity',
            '--score_model_paths', str(runtime['weights']), '--psichic_target_fasta', str(runtime['root'] / 'target.fasta'),
            '--psichic_device', 'cpu', '--building_blocks_paths', str(blocks),
            '--reaction_to_building_blocks_paths', str(EXAMPLE / 'real.pkl'), '--chemical_spaces', 'real',
            '--n_rollout', '2', '--rng_seed', '0', '--save_frequency', '1', '--store_nodes']


def auto_args(runtime, native, *, validate=False, mode=None):
    args = mode_args(runtime, mode) if mode else runtime['args'].copy()
    index = args.index('--candidates')
    del args[index:index + 2]
    args[0] = 'generate-and-rank'
    return [*args, *(['--validate-only'] if validate else []), '--native-args', *native]


def test_import_preserves_every_native_string_duplicate_row_and_safe_path(runtime):
    source = products(runtime)
    before = (source / 'molecules.csv').read_bytes()
    result, value = cli(runtime, rank_args(runtime, source))
    assert result.returncode == 0, (value, result.stderr)
    rows = read(value['reports']['results.json'])['rows']
    originals = list(csv.DictReader(before.decode().splitlines()))
    assert [row['entry']['metadata']['native'] for row in rows] == originals
    assert [row['entry']['source'] for row in rows] == [f'{source}/molecules.csv#row={i}' for i in (1, 2, 3)]
    assert [row['entry']['key'] for row in rows] == [f'row_{i:012d}' for i in (1, 2, 3)]
    assert value['counts']['ranked'] == 3 and value['top'][0]['entry']['smiles'] == 'CCN'
    assert all(row['psichic']['raw']['raw_affinity'] != float(row['entry']['metadata']['native']['score']) for row in rows)
    assert (runtime['run'] / 'inputs/molecules.csv').read_bytes() == before == (source / 'molecules.csv').read_bytes()
    assert not (runtime['root'] / 'unsafe').exists()
    assert value['generation']['completion']['process_completion_verified'] is False
    before_calls = calls(runtime)
    (source / 'molecules.csv').write_text('original no longer available')
    result, resumed = cli(runtime, ['resume', '--run', str(runtime['run']), '--json'])
    assert result.returncode == 0 and calls(runtime) == before_calls
    assert resumed['report_version'] == value['report_version']


@pytest.mark.parametrize('content', [
    'node_id,smiles,score\na,CCO,1\n', 'node_id,smiles,score,num_reactions\n',
    'node_id,smiles,score,num_reactions\na,CCO,1\n',
    'node_id,node_id,smiles,score,num_reactions\na,a,CCO,1,1\n',
])
def test_native_schema_errors_start_no_inference(runtime, content):
    source = products(runtime)
    (source / 'molecules.csv').write_text(content)
    result, value = cli(runtime, rank_args(runtime, source))
    assert result.returncode == 2 and value['error']['code'] in ('INVALID_PRODUCTS', 'EMPTY_POPULATION')
    assert not runtime['run'].exists() and not calls(runtime)


def test_source_mutation_during_real_preflight_prevents_import(runtime):
    source = products(runtime)
    process = subprocess.Popen([sys.executable, '-B', '-m', 'applications.generator', *rank_args(runtime, source)],
                               env=runtime['env'], cwd=runtime['root'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    import psutil

    try:
        wait_for(lambda: any('runtime.py' in ' '.join(child.cmdline()) for child in psutil.Process(process.pid).children()))
        with (source / 'molecules.csv').open('a') as handle:
            handle.write('changed,CCO,1,1,22,0001,four\n')
        stdout, stderr = process.communicate(timeout=120)
        value = json.loads(stdout)
        assert process.returncode == 2 and value['error']['code'] == 'SOURCE_CHANGED', (value, stderr)
        assert not runtime['run'].exists() and not calls(runtime)
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=20)


@pytest.mark.parametrize('mode,method', [('boltz-only', None), ('full-ensemble', None), ('pyramid-ensemble', None),
                                      ('pyramid-refinement', None), ('full-ensemble', 'rank-v1')])
def test_optional_modes_match_standalone_with_identical_saved_predictions(boltz_runtime, mode, method):
    source = products(boltz_runtime)
    result, value = cli(boltz_runtime, rank_args(boltz_runtime, source, mode=mode, extra=['--ensemble', method] if method else []))
    assert result.returncode == 0, (value, result.stderr)
    saved = read(value['reports']['results.json'])['rows']
    candidates = [Candidate(i, row['entry']['id'], row['entry']['smiles']) for i, row in enumerate(saved, 1)]
    p, b = {}, {}
    for candidate, row in zip(candidates, saved, strict=True):
        raw = row['psichic']['raw']
        if raw:
            distribution = raw['distribution']
            kwargs = {'distribution': BetaDistribution(distribution['alpha'], distribution['beta'])} if distribution['state'] == 'valid' else {}
            p[candidate.key] = PSICHICScore(raw['raw_affinity'], raw['adaptive_score'], candidate.smiles, **kwargs)
        raw = row['boltz']['raw']
        if raw:
            b[candidate.key] = BoltzScore(raw['affinity_probability_binary'], raw['affinity_pred_value'], raw['native'], '', candidate.smiles)
    settings = read(value['reports']['results.json'])['settings']
    options = {'ensemble': EnsembleMethod(settings['ensemble'])} if settings['ensemble'] else {}
    plan = Plan(Mode(mode), shortlist_size=settings['shortlist_size'], metric=Metric(settings['metric']), **options)
    standalone = run_workflow(candidates, plan,
        psichic=lambda selected: EngineRun({row.key: p[row.key] for row in selected}),
        boltz=lambda selected: EngineRun({row.key: b[row.key] for row in selected}))
    for durable, reference in zip(saved, standalone.rows, strict=True):
        score = (reference.normalized_ensemble_score if settings['ensemble'] == 'concentration-v1' else
                 reference.experimental_combined_value if settings['ensemble'] == 'rank-v1' else
                 getattr(reference.boltz, settings['metric'], None))
        assert durable['score'] == pytest.approx(score, abs=1e-12) if score is not None else durable['score'] is None
        assert durable['final_rank'] == reference.final_rank
    assert value['settings']['metric'] == ('affinity_probability_binary' if method == 'rank-v1' or mode in ('boltz-only', 'pyramid-refinement') else 'affinity_pred_value')


def test_existing_and_future_validation_counts_and_no_model_start(runtime):
    native = native_args(runtime)
    result, future = cli(runtime, auto_args(runtime, native, validate=True))
    assert result.returncode == 0, (future, result.stderr)
    assert future['population_state'] == 'future-generation' and future['resolved']['entries'] is None
    assert future['resolved']['evaluations'] == {'psichic': None, 'boltz': 0}
    assert not calls(runtime) and not runtime['run'].exists() and not (runtime['root'] / 'native').exists()
    source = products(runtime)
    result, existing = cli(runtime, rank_args(runtime, source, extra=['--validate-only']))
    assert result.returncode == 0 and existing['resolved']['entries'] == 3
    assert existing['population_state'] == 'existing-products'
    assert not runtime['run'].exists() and not calls(runtime)


def test_pinned_native_generation_then_scoring_preserves_all_columns_and_process_barrier(runtime):
    result, value = cli(runtime, auto_args(runtime, native_args(runtime)))
    assert result.returncode == 0, (value, result.stderr)
    native = runtime['root'] / 'native/molecules.csv'
    rows = read(value['reports']['results.json'])['rows']
    assert [row['entry']['metadata']['native'] for row in rows] == list(csv.DictReader(native.open()))
    assert rows[0]['entry']['metadata']['native']['building_block_1_1_id'] in ('0001', '0002')
    completion = read(runtime['run'] / 'generation/completion.json')
    assert completion['returncode'] == 0 and completion['group_retired'] and not alive(completion['child'])
    invocations = runtime['run'] / 'scoring/local/invocations'
    assert all(read(path / 'child.json')['started'] >= completion['observed_exit_ns'] / 1e9 for path in invocations.iterdir())
    assert 'Running rollouts' not in result.stdout and 'Running rollouts' in (runtime['run'] / 'generation/stdout.log').read_text()
    before = calls(runtime)
    result, after = cli(runtime, ['resume', '--run', str(runtime['run']), '--json'])
    assert result.returncode == 0 and calls(runtime) == before and after['report_version'] == 1


def test_automatic_boltz_options_keep_shared_defaults_and_delimiters(boltz_runtime):
    result, value = cli(boltz_runtime, auto_args(boltz_runtime, native_args(boltz_runtime), mode='full-ensemble'))
    assert result.returncode == 0, (value, result.stderr)
    assert value['settings']['ensemble'] == 'concentration-v1'
    assert value['settings']['metric'] == 'affinity_pred_value'
    assert value['resolved']['preset'] == 'normal'
    assert value['counts']['ranked'] == value['generation']['rows']


def test_generator_retry_is_explicit_and_preserves_report_history(runtime):
    source = products(runtime)
    (source / 'molecules.csv').write_text('node_id,smiles,score,num_reactions\nfailed,FAIL,99,1\n')
    result, first = cli(runtime, rank_args(runtime, source))
    assert result.returncode == 1 and first['status'] == 'partial'
    original_report = Path(first['reports']['results.json']).read_bytes()
    result, resumed = cli(runtime, ['resume', '--run', str(runtime['run']), '--json'])
    assert result.returncode == 1 and resumed['report_version'] == 1 and len(calls(runtime)) == 1
    Path(runtime['env']['CONTROLLED_RETRY']).touch()
    result, retried = cli(runtime, ['retry', '--run', str(runtime['run']), '--failed', '--json'])
    assert result.returncode == 0 and retried['report_version'] == 2 and len(calls(runtime)) == 2
    assert Path(first['reports']['results.json']).read_bytes() == original_report


def test_native_failure_prevents_final_scoring_and_resume_never_regenerates(runtime):
    native = native_args(runtime)
    (runtime['root'] / 'blocks.csv').write_text((EXAMPLE / 'building_blocks.csv').read_text())
    result, value = cli(runtime, auto_args(runtime, native))
    assert result.returncode == 1 and value['status'] == 'generation-failed', (value, result.stderr)
    assert not (runtime['run'] / 'scoring').exists() and not calls(runtime)
    completion = (runtime['run'] / 'generation/completion.json').read_bytes()
    result, value = cli(runtime, ['resume', '--run', str(runtime['run']), '--json'])
    assert result.returncode == 2 and value['error']['code'] == 'GENERATION_FAILED'
    assert (runtime['run'] / 'generation/completion.json').read_bytes() == completion


@pytest.mark.parametrize('hard_kill', [False, True])
def test_cancel_or_kill_generation_retires_descendants_and_stops_scoring(runtime, hard_kill):
    native = native_args(runtime)
    marker = runtime['root'] / 'descendant.pid'
    runtime['env']['CONTROLLED_DESCENDANT'] = str(marker)
    Path(runtime['env']['CONTROLLED_BLOCK']).touch()
    process = subprocess.Popen([sys.executable, '-B', '-m', 'applications.generator', *auto_args(runtime, native)],
                               env=runtime['env'], cwd=runtime['root'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        wait_for(lambda: (runtime['run'] / 'generation/child.json').exists(), timeout=120)
        wait_for(lambda: marker.exists())
        descendant = process_identity(int(marker.read_text()))
        child = read(runtime['run'] / 'generation/child.json')
        process.send_signal(signal.SIGKILL if hard_kill else signal.SIGTERM)
        stdout, stderr = process.communicate(timeout=20)
        if not hard_kill:
            assert process.returncode == 1 and json.loads(stdout)['status'] == 'generation-failed', stderr
        wait_for(lambda: alive(child) is False and alive(descendant) is False, timeout=10)
        assert not (runtime['run'] / 'scoring').exists()
        assert len(calls(runtime)) == 1
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=20)


@pytest.mark.parametrize('mutate', [False, True])
def test_interrupted_handoff_resumes_scoring_without_regenerating(runtime, mutate):
    code = '''import os, signal, sys
from applications.generator import workflow
from applications.generator.cli import main
original = workflow.atomic_pointer
def cancel_after_completion(path, value):
    original(path, value)
    if path.name == 'completion.json':
        os.kill(os.getpid(), signal.SIGINT)
workflow.atomic_pointer = cancel_after_completion
sys.exit(main(sys.argv[1:]))
'''
    process = subprocess.Popen([sys.executable, '-B', '-c', code, *auto_args(runtime, native_args(runtime))],
                               env=runtime['env'], cwd=runtime['root'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        stdout, stderr = process.communicate(timeout=180)
        assert process.returncode == 1 and json.loads(stdout)['status'] == 'interrupted', stderr
        assert not (runtime['run'] / 'scoring').exists()
        native_calls = calls(runtime)
        assert len(native_calls) == 1
        if mutate:
            path = runtime['root'] / 'native/molecules.csv'
            path.write_text(path.read_text().replace('psichic_affinity', 'changed_affinity'))
        result, value = cli(runtime, ['resume', '--run', str(runtime['run']), '--json'])
        if mutate:
            assert result.returncode == 2 and value['error']['code'] == 'SOURCE_CHANGED', value
            assert not (runtime['run'] / 'scoring').exists() and len(calls(runtime)) == 1
        else:
            assert result.returncode == 0, (value, result.stderr)
            assert len(calls(runtime)) == 2
        assert calls(runtime)[0] == native_calls[0]
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=20)


def test_existing_run_import_needs_no_native_dependency_and_missing_native_is_actionable(runtime):
    unavailable = runtime['root'] / 'unavailable'
    (unavailable / 'synthemol').mkdir(parents=True)
    (unavailable / 'synthemol/__init__.py').write_text('raise ImportError("controlled missing native dependency")\n')
    runtime['env']['PYTHONPATH'] += os.pathsep + str(unavailable)
    source = products(runtime)
    result, value = cli(runtime, rank_args(runtime, source, extra=['--validate-only']))
    assert result.returncode == 0, value
    native = ['--save_dir', str(runtime['root'] / 'new-native')]
    result, value = cli(runtime, auto_args(runtime, native, validate=True))
    assert result.returncode == 2 and value['error']['code'] == 'SYNTHEMOL_UNAVAILABLE'
    assert 'applications.generator.install' in value['next_action'] and not calls(runtime)


@pytest.mark.parametrize('option', ['--api_key_value=do-not-record', '--msa_server_password=do-not-record'])
def test_generator_rejects_msa_authentication_without_reflecting_values(runtime, option):
    result, value = cli(runtime, [*rank_args(runtime, products(runtime)), option])
    assert result.returncode == 2 and value['error']['code'] == 'NATIVE_OPTION'
    assert 'do-not-record' not in result.stdout + result.stderr
    assert not calls(runtime) and not runtime['run'].exists()


def test_generator_human_status_accepts_an_ordinary_shared_scoring_run(runtime):
    result, value = shared_cli(runtime)
    assert result.returncode == 0, value
    status = subprocess.run([sys.executable, '-B', '-m', 'applications.generator', 'status', '--run', str(runtime['run'])],
                            env=runtime['env'], cwd=runtime['root'], capture_output=True, text=True, timeout=30)
    assert status.returncode == 0 and 'COMPLETE' in status.stdout, status.stderr
    assert 'entries=3' in status.stdout and len(calls(runtime)) == 2
