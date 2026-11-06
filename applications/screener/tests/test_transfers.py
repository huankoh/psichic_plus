"""Native delivery faults leave the durable completion barrier intact."""

import json
import tarfile

import pytest

from applications.scoring import Acceptance, Engine, Ownership, Resolution
from applications.scoring.tests.conftest import prepare, transfer as core_transfer
from applications.scoring.contracts import canonical, digest, loads
from applications.scoring.evidence import Artifact, Receipt
from applications.scoring.runtime import identity
from applications.screener.native import ingest

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
