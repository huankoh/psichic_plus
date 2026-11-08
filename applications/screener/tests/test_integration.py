"""Native preparation and Linux workers with controlled, provenance-checked observations."""

import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import gzip
import threading
import signal
from concurrent.futures import ThreadPoolExecutor

import psutil

import pytest

from applications.scoring.contracts import Ownership
from applications.scoring.run import ScoringRun
from applications.scoring.processes import alive, process_identity, signal_owned, wait_owned
from applications.scoring.tests.test_executor_cli import calls, mode_args, wait_for
from applications.screener.population import snapshot

pytest_plugins = ['applications.scoring.tests.test_executor_cli']
ROOT = Path(__file__).resolve().parents[3]


def archive(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, 'w:gz') as handle:
        for name, value in records:
            member = tarfile.TarInfo('0000001/' + name + '.smi')
            data = (value + '\n').encode()
            member.size = len(data)
            handle.addfile(member, io.BytesIO(data))


def native_source(root, *, reference=None):
    source = root / 'afvs'
    if reference:
        subprocess.run(['git', 'clone', '-q', '--no-hardlinks', reference, str(source)], check=True, capture_output=True)
    else:
        (source / 'tools/templates').mkdir(parents=True)
    archive(source / 'library/A/0000001.tar.gz', [('same', 'CCO'), ('other', 'CCN')])
    archive(source / 'library/B/0000001.tar.gz', [('same', 'CCC')])
    (source / 'tools/templates/selected.csv').write_text(
        'collection_name,collection_number,ligand,original_metadata\n'
        'A,0000001,same,one\nB,0000001,same,two\nA,0000001,same,three\nA,0000001,other,four\n')
    (root / 'tmp').mkdir(exist_ok=True)
    (source / 'tools/templates/all.ctrl').write_text(f'''job_name=controlled
threads_per_docking=1
threads_to_use=1
program_timeout=300
batchsystem=bash
bash_template=./templates/template1.bash
bash_array_job_size=1
aws_region=us-east-2
data_storage_mode=sharedfs
job_storage_mode=sharedfs
data_collection_addressing_mode=metatranche
data_collection_identifier=
job_addressing_mode=metatranche
object_store_job_bucket=
object_store_job_prefix=controlled
collection_folder=../library
summary_formats=csv.gz
print_attrs_in_summary=smi
collection_list_type=csv_collection_key_ligand
collection_list_file=templates/selected.csv
dockings_per_subjob=2
ligand_library_format=smi
dynamic_tranche_filtering=0
dynamic_tranche_filtering_regex=.*
tempdir_default={root / 'tmp'}
docking_scenario_names=target
docking_scenario_methods=psichic_plus
docking_scenario_replicas=1
docking_scenario_batchsizes=1
docking_scenario_basefolder=../input-files
prescreen_mode=0
run_atom_check=0
''')
    return source


def test_population_keeps_native_order_and_repeated_identical_pairs(tmp_path):
    source = native_source(tmp_path)
    with snapshot(source, 'target') as population:
        entries = list(population.entries())
        assert [e.id for e in entries] == ['same', 'same', 'same', 'other']
        assert [e.smiles for e in entries] == ['CCO', 'CCC', 'CCO', 'CCN']
        assert [json.loads(e.metadata_json)['source_occurrence'] for e in entries] == [1, 2, 3, 4]
        assert json.loads(entries[2].metadata_json)['selection']['original_metadata'] == 'three'
        assert entries[0].source != entries[2].source


def sparse_source(source):
    from applications.screener.tests.run_native import write_archive
    write_archive(source / 'library/sparse/0000001.tar.gz', {'a': 'CCO', 'b': 'CCC', 'c': 'CCN'},
                  'B_0000001,b,0\nA_0000001,a,0\nA_0000001,c,1\n')
    (source / 'tools/templates/selected.csv').write_text('sparse_0000001 3\nsparse_0000001 3\n')
    path = source / 'tools/templates/all.ctrl'
    path.write_text(path.read_text().replace('collection_list_type=csv_collection_key_ligand', 'collection_list_type=standard')
                    .replace('prescreen_mode=0', 'prescreen_mode=1\nprescreen_ligands_per_tranche=1'))


def test_sparse_listing_preserves_order_original_collections_and_repeated_occurrences(tmp_path):
    source = native_source(tmp_path)
    sparse_source(source)
    with snapshot(source, 'target') as population:
        entries = list(population.entries())
        assert [e.id for e in entries] == ['b', 'a', 'b', 'a']
        assert [json.loads(e.metadata_json)['collection'] for e in entries] == ['B_0000001', 'A_0000001'] * 2
        assert json.loads(entries[2].metadata_json)['source_occurrence'] == 2
        assert json.loads(entries[0].metadata_json)['prescreen_listing'] == 'B_0000001,b,0'


@pytest.mark.parametrize('problem', ['missing', 'duplicate', 'traversal', 'replicas'])
def test_invalid_native_population_rejected_before_preparation(tmp_path, problem):
    source = native_source(tmp_path)
    if problem == 'missing':
        with (source / 'tools/templates/selected.csv').open('a') as handle:
            handle.write('A,0000001,missing,no\n')
    elif problem in ('duplicate', 'traversal'):
        archive(source / 'library/A/0000001.tar.gz', [('same', 'CCO'), ('same' if problem == 'duplicate' else '../../escape', 'N')])
    else:
        path = source / 'tools/templates/all.ctrl'
        path.write_text(path.read_text().replace('docking_scenario_replicas=1', 'docking_scenario_replicas=2'))
    with pytest.raises(ValueError):
        with snapshot(source, 'target'):
            pytest.fail('Invalid population imported')
    assert not (source / 'workflow').exists()


def test_completed_screen_retains_summary_as_metadata_and_requires_original_selection(tmp_path):
    from applications.screener.population import read_control
    source = native_source(tmp_path)
    (source / 'workflow').mkdir()
    (source / 'workflow/config.json').write_text(json.dumps(read_control(source / 'tools/templates/all.ctrl')))
    summary = source / 'output-files/target/csv/1/0.csv.gz'
    summary.parent.mkdir(parents=True)
    with gzip.open(summary, 'wt') as handle:
        handle.write('collection_key,ligand,score_0,attr_psichic_affinity\nA_0000001,same,-99,99\n')
    with snapshot(source, 'target', completed=True) as population:
        entries = list(population.entries())
        metadata = json.loads(entries[0].metadata_json)
        assert metadata['native_summaries'][0]['row']['attr_psichic_affinity'] == '99'
        assert population.evidence['legacy_scores_reused'] is False
    (source / 'tools/templates/selected.csv').unlink()
    with pytest.raises(ValueError, match='original ordered'):
        with snapshot(source, 'target', completed=True):
            pass


def setup_runtime(runtime):
    reference = os.environ.get('AFVS_TEST_UPSTREAM')
    if not reference:
        pytest.skip('Set AFVS_TEST_UPSTREAM to pinned native source')
    source = native_source(runtime['root'], reference=reference)
    runtime['env'].update(PYTHONPATH=str(ROOT) + os.pathsep + str(ROOT / 'applications/screener'),
                          AWS_EC2_METADATA_DISABLED='true', AWS_ACCESS_KEY_ID='fixture',
                          AWS_SECRET_ACCESS_KEY='fixture', AWS_DEFAULT_REGION='us-east-2', CUDA_VISIBLE_DEVICES='')
    return source


def arguments(runtime, source, mode='psichic-only', extra=()):
    args = mode_args(runtime, mode, extra) if mode != 'psichic-only' else [*runtime['args'], *extra]
    index = args.index('--candidates')
    del args[index:index + 2]
    args[0] = 'prepare'
    insert = args.index('--') if '--' in args else len(args)
    args[insert:insert] = ['--afvs', str(source), '--scenario', 'target']
    return args


def cli(runtime, args, *, timeout=300):
    result = subprocess.run([sys.executable, '-B', '-m', 'applications.screener', *args], cwd=runtime['root'],
                            env=runtime['env'], capture_output=True, text=True, timeout=timeout)
    with (runtime['root'] / 'screen-commands.jsonl').open('a') as handle:
        handle.write(json.dumps({'args': result.args, 'exit': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}) + '\n')
    return result, json.loads(result.stdout)


def execute_native(runtime, stage):
    status = json.loads((stage / 'workflow/status.json').read_text())
    for unit, work in status['workunits'].items():
        for subjob in work['subjobs']:
            env = dict(runtime['env'], AFVS_WORKUNIT=unit, AFVS_WORKUNIT_SUBJOB=subjob,
                       AFVS_JOB_STORAGE_MODE='sharedfs', AFVS_CONFIG_JOB_TGZ=str(stage / 'workflow/workunits' / (unit + '.tar.gz')),
                       AFVS_TMP_PATH=str(runtime['root'] / 'tmp'), AFVS_TOOLS_PATH=str(stage / 'tools/bin'), AFVS_VCPUS='1')
            if os.environ.get('AFVS_WORKER_PATH'):
                env['PYTHONPATH'] += os.pathsep + os.environ['AFVS_WORKER_PATH']
            with (stage / f'worker-{unit}-{subjob}.log').open('w') as log:
                process = subprocess.Popen([os.environ.get('AFVS_WORKER_PYTHON', sys.executable), str(stage / 'tools/templates/afvs_run.py')],
                                           cwd=stage / 'tools', env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                identity = process_identity(process.pid)
                finished = threading.Event()

                def record_processes():
                    seen = set()
                    with (stage / f'processes-{unit}-{subjob}.jsonl').open('w') as handle:
                        while not finished.is_set():
                            try:
                                parent = psutil.Process(process.pid)
                                for child in [parent, *parent.children(recursive=True)]:
                                    value = process_identity(child.pid)
                                    key = (child.pid, value['started'])
                                    if key not in seen:
                                        handle.write(json.dumps(dict(value, command=child.cmdline())) + '\n')
                                        handle.flush()
                                        seen.add(key)
                            except (psutil.Error, ProcessLookupError):
                                pass
                            finished.wait(.1)

                recorder = threading.Thread(target=record_processes)
                recorder.start()
                try:
                    code, timed_out = wait_owned(process, identity, 300, lambda: False)
                finally:
                    finished.set()
                    recorder.join(timeout=5)
            assert not timed_out and code == 0, (stage / f'worker-{unit}-{subjob}.log').read_text()[-4000:]


@pytest.mark.skipif(sys.platform != 'linux', reason='Real native multiprocessing worker requires Linux')
def test_native_killed_owner_requires_explicit_reconciliation(runtime):
    source = setup_runtime(runtime)
    block = Path(runtime['env']['CONTROLLED_BLOCK'])
    block.touch()
    result, value = cli(runtime, arguments(runtime, source))
    stage = Path(value['prepared_stage'])
    with ThreadPoolExecutor(max_workers=1) as workers:
        job = workers.submit(execute_native, runtime, stage)
        try:
            wait_for(lambda: calls(runtime))
            invocations = runtime['run'] / 'local/invocations'
            invocation = next(p for p in invocations.iterdir() if (p / 'child.json').exists())
            owner = json.loads((invocation / 'owner.json').read_text())
            child = json.loads((invocation / 'child.json').read_text())
            result, rejected = cli(runtime, ['reconcile', '--run', str(runtime['run']), '--attempt',
                json.loads((invocation / 'request.json').read_text())['tasks'][0]['attempt_id'],
                '--resolution', 'confirmed-stopped', '--job-id', 'controlled', '--evidence', 'Fault-injection test of live guard', '--json'])
            assert result.returncode == 2 and 'live or uninspectable' in rejected['error']['message']
            assert signal_owned(owner, signal.SIGKILL)
            wait_for(lambda: alive(child) is False)
        finally:
            block.unlink(missing_ok=True)
        job.result(timeout=300)
    run = ScoringRun.open(runtime['run'])
    result, waiting = cli(runtime, ['resume', '--run', str(run.path), '--json'])
    assert result.returncode == 1 and waiting['prepared_stage'] is None
    for diagnostic in tuple(run.diagnostics()):
        if diagnostic.state.value == 'submitted':
            result, reconciled = cli(runtime, ['reconcile', '--run', str(run.path), '--attempt', diagnostic.attempt_id,
                '--resolution', 'confirmed-stopped', '--job-id', 'controlled', '--evidence', 'Native subprocess exited and owner/model identities are stopped', '--json'])
            assert result.returncode == 1, reconciled
    result, replacement = cli(runtime, ['resume', '--run', str(run.path), '--json'])
    execute_native(runtime, Path(replacement['prepared_stage']))
    result, finished = cli(runtime, ['resume', '--run', str(run.path), '--json'])
    assert result.returncode == 0, finished
    assert alive(owner) is False and alive(child) is False


@pytest.mark.skipif(sys.platform != 'linux', reason='Real native multiprocessing worker requires Linux')
def test_native_sparse_selection_across_bounded_preparation_windows(runtime):
    source = setup_runtime(runtime)
    sparse_source(source)
    result, value = cli(runtime, arguments(runtime, source, extra=['--stage-size', '2']))
    seen = set()
    while value['status'] == 'waiting':
        stage = Path(value['prepared_stage'])
        assert str(stage) not in seen and len(seen) < 2, value
        seen.add(str(stage))
        execute_native(runtime, stage)
        result, value = cli(runtime, ['resume', '--run', str(runtime['run']), '--json'])
    assert result.returncode == 0 and len(seen) == 2, value
    run = ScoringRun.open(runtime['run'])
    assert [row.entry.id for row in run.population()] == ['b', 'a', 'b', 'a']


@pytest.mark.skipif(sys.platform != 'linux', reason='Real native multiprocessing worker requires Linux')
def test_native_failure_retry_history_and_unknown_liveness(boltz_runtime):
    runtime = boltz_runtime
    source = setup_runtime(runtime)
    archive(source / 'library/A/0000001.tar.gz', [('same', 'CCO'), ('other', 'FAIL')])
    result, value = cli(runtime, arguments(runtime, source))
    assert result.returncode == 1, value
    initial = {s['directory'] for s in value['native_stages']}
    run = ScoringRun.open(runtime['run'])
    original = [d.attempt_id for d in run.diagnostics()]
    result, waiting = cli(runtime, ['resume', '--run', str(run.path), '--json'])
    assert result.returncode == 1 and [d.attempt_id for d in run.diagnostics()] == original
    assert waiting['prepared_stage'] is None
    assert not calls(runtime)
    for stage in initial:
        execute_native(runtime, Path(stage))
    result, failed = cli(runtime, ['resume', '--run', str(run.path), '--json'])
    assert result.returncode == 1 and failed['status'] == 'partial', failed
    first_report = Path(failed['reports']['results.json'])
    saved = first_report.read_bytes()
    prior_calls = calls(runtime)
    result, unchanged = cli(runtime, ['resume', '--run', str(run.path), '--json'])
    assert calls(runtime) == prior_calls and unchanged['report_version'] == failed['report_version']
    Path(runtime['env']['CONTROLLED_RETRY']).touch()
    result, retrying = cli(runtime, ['retry', '--run', str(run.path), '--failed', '--json'])
    assert result.returncode == 1, retrying
    added = {s['directory'] for s in retrying['native_stages']} - initial
    assert len(added) == 1
    for stage in added:
        execute_native(runtime, Path(stage))
    result, finished = cli(runtime, ['resume', '--run', str(run.path), '--json'])
    assert result.returncode == 0 and finished['report_version'] == failed['report_version'] + 1, finished
    assert first_report.read_bytes() == saved


@pytest.mark.skipif(sys.platform != 'linux', reason='Real native multiprocessing worker requires Linux')
def test_native_global_shortage_creates_zero_boltz_tasks(boltz_runtime):
    runtime = boltz_runtime
    source = setup_runtime(runtime)
    archive(source / 'library/A/0000001.tar.gz', [('same', 'FAIL'), ('other', 'DIST')])
    result, value = cli(runtime, arguments(runtime, source, 'pyramid-ensemble'))
    assert result.returncode == 1, value
    for stage in value['native_stages']:
        execute_native(runtime, Path(stage['directory']))
    result, shortage = cli(runtime, ['resume', '--run', str(runtime['run']), '--json'])
    assert result.returncode == 1 and shortage['status'] == 'shortage', shortage
    assert {s['engine'] for s in shortage['native_stages']} == {'psichic'}
    assert not any(json.loads(line)['engine'] == 'boltz' for line in calls(runtime))


def test_public_validation_has_no_run_or_inference(runtime):
    source = setup_runtime(runtime)
    tools = source / 'tools'
    modes = {}
    for path in sorted((p for p in tools.rglob('*') if not p.is_symlink()), key=lambda p: len(p.parts), reverse=True):
        modes[path] = path.stat().st_mode
        path.chmod(path.stat().st_mode & ~0o222)
    try:
        result, value = cli(runtime, arguments(runtime, source, extra=['--validate-only']))
    finally:
        for path, mode in sorted(modes.items(), key=lambda item: len(item[0].parts)):
            path.chmod(mode)
    assert result.returncode == 0, value
    assert value['resolved']['entries'] == 4
    assert not runtime['run'].exists() and not calls(runtime)


def test_public_malformed_archive_is_a_json_configuration_error(runtime):
    source = setup_runtime(runtime)
    (source / 'library/A/0000001.tar.gz').write_bytes(b'broken gzip archive')
    result, value = cli(runtime, arguments(runtime, source, extra=['--validate-only']))
    assert result.returncode == 2 and value['status'] == 'configuration-error'
    assert not runtime['run'].exists() and not calls(runtime)


def test_actual_child_rich_same_pass_preserves_affinity_on_distribution_failure(runtime, monkeypatch):
    from psichic_adaptiveflow.adapter import Affinity, LigandKey, ScoreInput, Scorer, Target
    with (runtime['root'] / 'controlled-screening/inference.py').open('a') as handle:
        handle.write('\ndef build_protein_state(*args):\n    return {}, None\n')
    for name, value in runtime['env'].items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv('TORCH_HOME', str(runtime['root'] / 'controlled-torch'))
    monkeypatch.setenv('PYTHONPATH', os.pathsep.join([str(runtime['root'] / 'controlled-screening'), str(ROOT), str(ROOT / 'applications/screener')]))
    monkeypatch.chdir(runtime['root'])
    scorer = Scorer(Target(runtime['root'] / 'target.fasta', runtime['weights'], 'cpu'), 1)
    entries = [ScoreInput(LigandKey('A_1', 'same'), 0, Path('unused'), smiles) for smiles in ('CCN', 'DIST', 'MISSING_DIST')]
    try:
        result = scorer.score(entries, 60, output='distribution')
        assert all(isinstance(r, Affinity) and r.value == 9 for r in result), result
        assert result[0].distribution == {'state': 'valid', 'alpha': 3., 'beta': 1., 'concentration': 4.}
        assert result[1].distribution['state'] == 'distribution-failure'
        assert result[2].distribution['state'] == 'distribution-failure'
        assert len(calls(runtime)) == 1
        assert scorer.score(entries, 60) == [Affinity(9), Affinity(9), Affinity(9)]
        assert len(calls(runtime)) == 2
    finally:
        scorer.close()


@pytest.mark.skipif(sys.platform != 'linux', reason='Real native multiprocessing worker requires Linux')
@pytest.mark.parametrize('mode,method', [('psichic-only', None), ('boltz-only', None), ('full-ensemble', None),
                                      ('pyramid-ensemble', None), ('pyramid-refinement', None), ('full-ensemble', 'rank-v1')])
def test_native_workunits_receipts_and_saved_observation_parity(boltz_runtime, mode, method):
    runtime = boltz_runtime
    source = setup_runtime(runtime)
    command = arguments(runtime, source, mode, ['--ensemble', method] if method else [])
    if mode == 'psichic-only':
        scenario = source / 'input-files/target'
        scenario.mkdir(parents=True)
        (scenario / 'config.txt').write_text(f'protein_fasta={runtime["root"] / "target.fasta"}\nmodel_dir={runtime["weights"]}\ndevice=cpu\n')
        with (source / 'legacy-preparation.log').open('w') as log:
            prepared = subprocess.run([sys.executable, 'afvs_prepare_folders.py'], cwd=source / 'tools', env=runtime['env'],
                                      stdout=log, stderr=subprocess.STDOUT, timeout=60)
        assert prepared.returncode == 0, (source / 'legacy-preparation.log').read_text()
        summary = source / 'output-files/target/csv/1/0.csv.gz'
        summary.parent.mkdir(parents=True)
        with gzip.open(summary, 'wt') as handle:
            handle.write('collection_key,ligand,score_0,attr_psichic_affinity\nA_0000001,same,-99,99\n')
        command[0] = 'rank'
        command += ['--source-run', str(source)]
    result, value = cli(runtime, command)
    assert result.returncode == 1, value
    assert (runtime['run'] / 'screening/source/selection').read_bytes() == (source / 'tools/templates/selected.csv').read_bytes()
    stages = set()
    while value['status'] == 'waiting':
        pending = [Path(s['directory']) for s in value['native_stages'] if s['directory'] not in stages]
        assert pending, value
        for stage in pending:
            stages.add(str(stage))
            execute_native(runtime, stage)
        result, value = cli(runtime, ['resume', '--run', str(runtime['run']), '--json'])
        assert result.returncode in (0, 1), value
    assert result.returncode == 0, value
    run = ScoringRun.open(runtime['run'])
    assert all(run.task(d.attempt_id).ownership is Ownership.EXTERNAL for d in tuple(run.diagnostics()))
    assert [r.entry.id for r in run.population()] == ['same', 'same', 'same', 'other']
    if mode == 'psichic-only':
        rows = list(run.results())
        assert rows[0]['entry']['metadata']['native_summaries'][0]['row']['attr_psichic_affinity'] == '99'
        assert rows[0]['psichic']['raw']['raw_affinity'] == 6
    saved_calls = calls(runtime)
    report = Path(value['reports']['results.json']).read_bytes()
    result, resumed = cli(runtime, ['resume', '--run', str(runtime['run']), '--json'])
    assert result.returncode == 0 and resumed['report_version'] == value['report_version']
    assert report == Path(value['reports']['results.json']).read_bytes() and calls(runtime) == saved_calls
    from applications.screener.tests.compare_saved import compare
    compare(Path(value['reports']['results.json']))
