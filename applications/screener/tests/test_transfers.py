"""Native delivery faults leave the durable completion barrier intact."""

import json
import tarfile

import pytest

from applications.scoring import Acceptance, Engine, Ownership, Resolution
from applications.scoring.tests.conftest import prepare, transfer as core_transfer
from applications.scoring.contracts import canonical, digest, loads
from applications.scoring.evidence import Artifact, Receipt
from applications.scoring.runtime import checksum, identity
from applications.scoring.worker import publish
from applications.screener.native import ingest
from applications.screener.worker import execute

pytest_plugins = ['applications.scoring.tests.conftest']


def transfer(root, task, **kwargs):
    root = core_transfer(root, task, **kwargs)
    receipt = Receipt.from_json((root / 'receipt.json').read_bytes())
    profile = {'engine_version': 'controlled', 'source': [], 'weights': [], 'environment': {}, 'device': 'cpu', 'precision': 'controlled'}
    extra = {'loaded_runtime': {'profile_digest': digest(identity(profile)), 'loaded': {'controlled': {'sha256': 'controlled'}}},
             'runtime_inventory': profile, 'native_mapping': []}
    for name, value in extra.items():
        (root / name).write_text(canonical(value))
    receipt = Receipt(task, receipt.record_json, receipt.artifacts + tuple(Artifact.from_file(root, name, name) for name in extra))
    (root / 'receipt.json').write_text(receipt.to_json())
    return root


def delivery(run, source, *, name='1'):
    root = run.path / 'screening/stages/stage/output-files/score/logs' / name
    root.mkdir(parents=True, exist_ok=True)
    (run.path / 'screening/imports').mkdir(exist_ok=True)
    if (source / 'receipt.json').exists():
        receipt = loads((source / 'receipt.json').read_bytes())
        request = run.path / 'local/invocations' / receipt['task']['invocation_id'] / 'request.json'
        request.parent.mkdir(parents=True, exist_ok=True)
        request.write_text(canonical({'profile': loads((source / 'runtime_inventory').read_bytes()), 'mapping': []}))
    path = root / '0.tar.gz'
    with tarfile.open(path, 'w:gz') as archive:
        archive.add(source, arcname='0/collection/collection/row/0/receipt')
    return path


def invocation(root, task):
    source = core_transfer(root / 'observed', task)
    receipt = Receipt.from_json((source / 'receipt.json').read_bytes())
    directory = root / 'invocation'
    for name in ('inputs', 'raw', 'transfers'):
        (directory / name).mkdir(parents=True)
    name = task.entry_key + '.json'
    (directory / 'inputs' / name).write_bytes((source / 'input.txt').read_bytes())
    mapping = [{'entry_key': task.entry_key, 'attempt_id': task.attempt_id,
                'native_row': 1, 'native_key': 'row_00000001', 'input': name}]
    profile = {'engine_version': 'controlled', 'source': [], 'weights': [], 'environment': {}, 'device': 'cpu', 'precision': 'controlled'}
    request = {'engine': 'psichic', 'tasks': [task.to_dict()], 'mapping': mapping, 'target_assets': [], 'profile': profile}
    for name, value in [('request.json', request), ('profile.json', profile), ('mapping.json', mapping),
                        ('completion.json', loads((source / 'execution.json').read_bytes())),
                        ('loaded.json', {'profile_digest': digest(identity(profile)), 'loaded': {'controlled': {}}})]:
        (directory / name).write_text(canonical(value))
    for name in ('stdout.log', 'stderr.log'):
        (directory / name).write_bytes((source / name).read_bytes())
    (directory / 'raw' / f'{task.attempt_id}.json').write_text(canonical({
        'record': loads(receipt.record_json), 'raw': loads((source / 'raw.json').read_bytes())}))
    manifest = {'schema_version': 1, 'invocation': str(directory), 'request_sha256': checksum(directory / 'request.json'),
                'run_id': task.run_id, 'population_digest': task.population_digest, 'engine': 'psichic', 'rows': [{'key': task.entry_key}]}
    map_path = root / f'{digest(manifest)}.json'
    map_path.write_text(canonical(manifest))
    return map_path, directory


@pytest.mark.parametrize('interruptions', [1, 3])
def test_export_recovers_repeated_publication_before_marker(make_run, tmp_path, interruptions):
    run = make_run(n=1)
    task, = prepare(run, ownership=Ownership.EXTERNAL)
    map_path, directory = invocation(tmp_path, task)
    for _ in range(interruptions):
        publish(directory)
        (directory / 'published.json').unlink()
    output = tmp_path / 'export'
    execute(map_path, task.entry_key, output)
    assert len(list((directory / 'transfers').glob('*/receipt.json'))) == interruptions + 1
    assert run.accept(output / 'receipt') is Acceptance.ACCEPTED
    execute(map_path, task.entry_key, tmp_path / 'redelivery')
    assert run.accept(tmp_path / 'redelivery/receipt') is Acceptance.IDENTICAL
    assert run.finalize().status.value == 'complete'
    assert list(run.results())[0]['psichic']['raw']['raw_affinity'] == 6


def test_export_rejects_contradictory_receipts_after_republication(make_run, tmp_path):
    run = make_run(n=1)
    task, = prepare(run, ownership=Ownership.EXTERNAL)
    map_path, directory = invocation(tmp_path, task)
    publish(directory)
    raw_path = directory / 'raw' / f'{task.attempt_id}.json'
    row = loads(raw_path.read_bytes())
    row['record'].update(raw_affinity=7, adaptive_score=-7)
    row['raw']['native']['predicted_binding_affinity'] = 7
    raw_path.write_text(canonical(row))
    (directory / 'published.json').unlink()
    output = tmp_path / 'export'
    with pytest.raises(ValueError):
        execute(map_path, task.entry_key, output)
    assert not (output / 'receipt').exists()
    assert next(run.diagnostics()).state.value == 'submitted'


@pytest.mark.parametrize('copies', [1, 2])
@pytest.mark.parametrize('damage', ['missing', 'truncated', 'tampered'])
def test_export_rejects_missing_or_corrupt_artifacts(make_run, tmp_path, copies, damage):
    run = make_run(n=1)
    task, = prepare(run, ownership=Ownership.EXTERNAL)
    map_path, directory = invocation(tmp_path, task)
    for index in range(copies):
        if index:
            (directory / 'published.json').unlink()
        publish(directory)
    artifact = next((directory / 'transfers').iterdir()) / 'raw_psichic'
    original = artifact.read_bytes()
    if damage == 'missing':
        artifact.unlink()
    else:
        artifact.write_bytes(original[:20] if damage == 'truncated' else b'x' * len(original))
    output = tmp_path / 'export'
    with pytest.raises((OSError, ValueError)):
        execute(map_path, task.entry_key, output)
    assert not (output / 'receipt').exists()
    assert next(run.diagnostics()).state.value == 'submitted'


def test_short_native_copy_never_accepts_then_complete_redelivery_is_idempotent(make_run, tmp_path):
    run = make_run(n=1)
    task, = prepare(run, ownership=Ownership.EXTERNAL)
    source = transfer(tmp_path / 'worker', task)
    path = delivery(run, source)
    original = path.read_bytes()
    path.write_bytes(original[:len(original) // 2])
    assert ingest(run)[Acceptance.INCOMPLETE.value] == 1
    assert next(run.diagnostics()).state.value == 'submitted'
    path.write_bytes(original)
    assert ingest(run)[Acceptance.ACCEPTED.value] == 1
    imports = set((run.path / 'screening/imports').iterdir())
    assert ingest(run)[Acceptance.IDENTICAL.value] == 1
    assert set((run.path / 'screening/imports').iterdir()) == imports
    assert run.finalize().status.value == 'complete'


def test_conflicting_native_delivery_rejected_and_old_attempt_never_replaces_current(make_run, tmp_path):
    run = make_run(n=1)
    task, = prepare(run, ownership=Ownership.EXTERNAL)
    first = transfer(tmp_path / 'first', task)
    path = delivery(run, first)
    assert ingest(run)[Acceptance.ACCEPTED.value] == 1
    second = transfer(tmp_path / 'second', task, affinity=7)
    delivery(run, second, name='2')
    with pytest.raises(ValueError, match='Contradictory'):
        ingest(run)
    assert list(run.results(run.finalize().version))[0]['psichic']['raw']['raw_affinity'] == 6
    assert path.exists()


def test_superseded_native_receipt_does_not_complete_replacement(make_run, tmp_path):
    run = make_run(n=1)
    old, = prepare(run, ownership=Ownership.EXTERNAL)
    run.reconcile(old.attempt_id, Resolution.STOPPED, evidence='controlled scheduler stopped')
    new, = prepare(run, ownership=Ownership.EXTERNAL)
    delivery(run, transfer(tmp_path / 'old', old))
    assert ingest(run)[Acceptance.STALE.value] == 1
    assert next(run.diagnostics()).attempt_id == new.attempt_id
    assert run.pending(Engine.PSICHIC) == ()


def test_overview_zero_exit_and_missing_receipt_have_no_completion_authority(make_run, tmp_path):
    run = make_run(n=1)
    task, = prepare(run, ownership=Ownership.EXTERNAL)
    root = tmp_path / 'native'
    root.mkdir()
    (root / 'overview.json').write_text(json.dumps({'success': 1, 'returncode': 0}))
    delivery(run, root)
    assert not any(ingest(run).values())
    assert next(run.diagnostics()).attempt_id == task.attempt_id
    assert run.pending(Engine.PSICHIC) == ()


def test_wrong_run_identity_is_rejected(make_run, tmp_path):
    run, wrong = make_run(n=1), make_run(n=1)
    prepare(run, ownership=Ownership.EXTERNAL)
    task, = prepare(wrong, ownership=Ownership.EXTERNAL)
    delivery(run, transfer(tmp_path / 'wrong', task))
    with pytest.raises(ValueError, match='Unexpected attempt'):
        ingest(run)


def test_missing_artifact_remains_pending_and_tamper_is_rejected(make_run, tmp_path):
    run = make_run(n=1)
    task, = prepare(run, ownership=Ownership.EXTERNAL)
    source = transfer(tmp_path / 'worker', task)
    raw = source / 'raw.json'
    original = raw.read_bytes()
    raw.write_bytes(original[:20])
    delivery(run, source)
    assert ingest(run)[Acceptance.INCOMPLETE.value] == 1
    raw.write_bytes(b'x' * len(original))
    delivery(run, source)
    with pytest.raises(ValueError, match='digest/size mismatch'):
        ingest(run)


def test_native_receipt_without_loaded_runtime_evidence_never_completes(make_run, tmp_path):
    run = make_run(n=1)
    task, = prepare(run, ownership=Ownership.EXTERNAL)
    source = transfer(tmp_path / 'worker', task)
    receipt = Receipt.from_json((source / 'receipt.json').read_bytes())
    (source / 'receipt.json').write_text(Receipt(task, receipt.record_json, tuple(
        artifact for artifact in receipt.artifacts if artifact.role != 'loaded_runtime')).to_json())
    delivery(run, source)
    with pytest.raises(ValueError, match='missing required qualified runtime evidence'):
        ingest(run)
    assert next(run.diagnostics()).state.value == 'submitted'
