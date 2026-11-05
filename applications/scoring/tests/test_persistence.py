from dataclasses import replace
import json
import subprocess
import sys

import pytest

from applications.scorer import Output
from applications.scoring import (
    Acceptance, ContractError, Engine, Entry, Fingerprint, IncompleteRun, Ownership, Receipt,
    Resolution, RunBusy, ScoringRun, SelectionState, Settings, State, StateError, Task,
)
from conftest import declarations, prepare, transfer


def test_streamed_population_identity_and_creation(make_run, tmp_path, target):
    run = make_run(n=4)
    rows = list(run.population())
    assert len({r.key for r in rows}) == 4
    assert [r.entry.id for r in rows] == ["duplicate"] * 4
    assert [r.entry.source for r in rows] == [f"fixture.csv:{i}" for i in range(4)]
    assert rows[0].entry.metadata_json == '{"route":"kept"}'
    same = ScoringRun.create(tmp_path / "same", (r.entry for r in rows), target=target, settings=run.settings)
    other = ScoringRun.create(tmp_path / "other", (r.entry for r in reversed(rows)), target=target, settings=run.settings)
    changed_target = ScoringRun.create(tmp_path / "target", (r.entry for r in rows), target=replace(target, identity="other"), settings=run.settings)
    assert same.population_digest == run.population_digest
    assert other.population_digest != run.population_digest != changed_target.population_digest
    with pytest.raises(FileExistsError):
        ScoringRun.create(run.path, [], target=target, settings=run.settings)
    assert len(list(run.population())) == 4


def test_failed_create_is_explicit_and_preserves_evidence(tmp_path, target):
    path = tmp_path / "failed"
    def broken():
        yield Entry("id", "CC", "fixture:1")
        raise RuntimeError("interrupted importer")
    with pytest.raises(RuntimeError):
        ScoringRun.create(path, broken(), target=target, settings=Settings("psichic-only", psichic_options='{"seed":1}'))
    with pytest.raises(IncompleteRun):
        ScoringRun.open(path)
    assert (path / "coordinator.lock").exists()


def test_serialized_roundtrip_and_accepted_not_rescheduled(make_run, tmp_path):
    run = make_run(n=2)
    tasks = prepare(run)
    assert Task.from_json(tasks[0].to_json()) == tasks[0]
    directory = transfer(tmp_path / "transfer", tasks[0])
    receipt = Receipt.from_json((directory / "receipt.json").read_text())
    assert Receipt.from_json(receipt.to_json()) == receipt
    assert run.accept(directory) is Acceptance.ACCEPTED
    assert run.accept(directory) is Acceptance.IDENTICAL
    run = ScoringRun.open(run.path)
    assert run.pending(Engine.PSICHIC) == ()
    assert [d.state for d in run.diagnostics()] == [State.SUCCEEDED, State.SUBMITTED]
    (directory / "input.txt").write_text("changed after import")
    stored = run.path / "blobs" / tasks[0].fingerprint.native_input_sha256
    assert stored.read_text() == "CCO"


def test_external_unknown_needs_positive_stopped_reconciliation(make_run):
    run = make_run(n=1)
    task, = prepare(run, ownership=Ownership.EXTERNAL)
    run = ScoringRun.open(run.path)
    assert not run.pending(Engine.PSICHIC)
    run.reconcile(task.attempt_id, Resolution.UNKNOWN, evidence="scheduler unavailable")
    assert not run.pending(Engine.PSICHIC)
    run.reconcile(task.attempt_id, Resolution.RUNNING, evidence="job 123 still running")
    assert not run.pending(Engine.PSICHIC)
    run.reconcile(task.attempt_id, Resolution.STOPPED, evidence="scheduler job 123 terminal and workers reaped")
    assert len(run.pending(Engine.PSICHIC)) == 1
    new, = prepare(run)
    assert task.attempt_id != new.attempt_id


def test_failure_requires_retry_old_receipt_stale_new_observation_allowed(make_run, tmp_path):
    run = make_run(n=1)
    first, = prepare(run)
    failed = transfer(tmp_path / "failed", first, failure="explicit failure")
    assert run.accept(failed) is Acceptance.ACCEPTED
    assert not run.pending(Engine.PSICHIC)
    run.retry(first.entry_key, Engine.PSICHIC)
    second, = prepare(run)
    assert run.accept(failed) is Acceptance.STALE
    assert run.accept(transfer(tmp_path / "success", second, affinity=9)) is Acceptance.ACCEPTED
    assert len(list((run.path / "receipts").glob("*.json"))) == 2


def test_late_superseded_success_retained_conflicting_same_attempt_rejected(make_run, tmp_path):
    run = make_run(n=1)
    first, = prepare(run)
    run.reconcile(first.attempt_id, Resolution.STOPPED, evidence="owned process waited and reaped")
    second, = prepare(run)
    assert run.accept(transfer(tmp_path / "old", first, affinity=3)) is Acceptance.STALE
    assert run.accept(transfer(tmp_path / "new", second, affinity=8)) is Acceptance.ACCEPTED
    with pytest.raises(ContractError, match="Contradictory"):
        run.accept(transfer(tmp_path / "conflict", second, affinity=9))
    assert len(list((run.path / "receipts").glob("*.json"))) == 3


def test_missing_artifacts_and_temporary_receipts_remain_unaccepted(make_run, tmp_path):
    run = make_run(n=1)
    task, = prepare(run)
    directory = transfer(tmp_path / "partial", task)
    (directory / "receipt.json").rename(directory / "receipt.tmp")
    assert run.accept(directory) is Acceptance.INCOMPLETE
    (directory / "receipt.tmp").rename(directory / "receipt.json")
    (directory / "raw.json").rename(directory / "raw.tmp")
    assert run.accept(directory) is Acceptance.INCOMPLETE
    assert next(run.diagnostics()).state is State.SUBMITTED
    (directory / "raw.tmp").rename(directory / "raw.json")
    assert run.accept(directory) is Acceptance.ACCEPTED
    assert list((run.path / "staging").iterdir())


@pytest.mark.parametrize("attack", ["digest", "traversal", "symlink", "native", "empty-artifacts"])
def test_untrusted_artifact_boundary(make_run, tmp_path, attack):
    run = make_run("boltz-only", n=1)
    task, = prepare(run, Engine.BOLTZ)
    directory = transfer(tmp_path / "attack", task)
    receipt = json.loads((directory / "receipt.json").read_text())
    if attack == "digest":
        (directory / "structure.cif").write_text("tampered")
    elif attack == "traversal":
        receipt["artifacts"][0]["path"] = "../outside"
    elif attack == "symlink":
        (directory / "structure.cif").unlink()
        (directory / "structure.cif").symlink_to(directory / "input.txt")
    elif attack == "native":
        receipt["record"]["affinity_pred_value"] = -99
    else:
        receipt["artifacts"] = []
    (directory / "receipt.json").write_text(json.dumps(receipt))
    with pytest.raises((ContractError, OSError)):
        run.accept(directory)
    assert next(run.diagnostics()).state is State.SUBMITTED


def test_exact_coverage_duplicate_for_missing_even_equal_counts(make_run):
    run = make_run(n=2)
    tasks = prepare(run)
    identities = [(t.entry_key, t.engine, t.attempt_id) for t in tasks]
    run.check_coverage(identities)
    with pytest.raises(ContractError, match="Duplicate"):
        run.check_coverage([identities[0], identities[0]])
    with pytest.raises(ContractError, match="Missing or unexpected"):
        run.check_coverage(identities[:1])
    with pytest.raises(ContractError):
        run.check_coverage([(tasks[0].entry_key, Engine.BOLTZ, tasks[0].attempt_id), identities[1]])


@pytest.mark.parametrize("field", list(Fingerprint.__dataclass_fields__))
def test_every_fingerprint_field_required_and_hash_bound(make_run, field):
    run = make_run(n=1)
    task, = prepare(run)
    value = task.to_dict()
    del value["fingerprint"][field]
    with pytest.raises((ContractError, ValueError)):
        Task.from_dict(value)
    value = task.to_dict()
    value["fingerprint"][field] = "mutated"
    with pytest.raises((ContractError, ValueError, TypeError, AttributeError)):
        Task.from_dict(value)


def test_batch_shape_and_exact_expected_fingerprint(make_run, tmp_path):
    run = make_run(n=2)
    rows = run.pending(Engine.PSICHIC)
    declared = declarations(run, rows, Engine.PSICHIC)
    with pytest.raises(ContractError, match="batch"):
        run.prepare(Engine.PSICHIC, declared[:1], ownership=Ownership.LOCAL)
    tasks = run.prepare(Engine.PSICHIC, declared, ownership=Ownership.LOCAL)
    altered = replace(tasks[0], fingerprint=replace(tasks[0].fingerprint, precision="float64"))
    with pytest.raises(ContractError, match="exact expected"):
        run.accept(transfer(tmp_path / "wrong-fingerprint", altered))


def test_distribution_required_affinity_only_rejected_and_failure_retained(make_run, tmp_path):
    run = make_run(n=1, psichic_output=Output.DISTRIBUTION)
    task, = prepare(run)
    directory = transfer(tmp_path / "valid", task)
    receipt = json.loads((directory / "receipt.json").read_text())
    receipt["record"]["distribution"] = {"state": "not-requested"}
    (directory / "receipt.json").write_text(json.dumps(receipt))
    with pytest.raises(ContractError, match="affinity-only"):
        run.accept(directory)
    bad = transfer(tmp_path / "invalid-distribution", task, alpha=None, beta=None, distribution_failure=True)
    assert run.accept(bad) is Acceptance.ACCEPTED
    assert next(run.diagnostics()).state is State.SUCCEEDED
    run.retry(task.entry_key, Engine.PSICHIC)
    assert len(run.pending(Engine.PSICHIC)) == 1


def test_global_barrier_and_shortage_zero_boltz(make_run, tmp_path):
    run = make_run("pyramid-ensemble", n=2, shortlist_size=2)
    tasks = prepare(run)
    run.accept(transfer(tmp_path / "one", tasks[0]))
    assert run.select().state is SelectionState.WAITING
    assert not run.pending(Engine.BOLTZ)
    run.accept(transfer(tmp_path / "two", tasks[1], failure="failed"))
    assert run.select().state is SelectionState.SHORTAGE
    assert not run.pending(Engine.BOLTZ)
    run.retry(tasks[1].entry_key, Engine.PSICHIC)
    replacement, = prepare(run)
    run.accept(transfer(tmp_path / "retry", replacement))
    assert run.select().state is SelectionState.COMMITTED
    assert len(run.pending(Engine.BOLTZ)) == 2
    with pytest.raises(StateError, match="frozen"):
        run.retry(tasks[1].entry_key, Engine.PSICHIC)


def test_concurrent_coordinator_lock_and_sigkill_release(make_run):
    run = make_run(n=1)
    code = """import fcntl,sys,time
f=open(sys.argv[1], 'r+')
fcntl.flock(f, fcntl.LOCK_EX)
print('locked', flush=True)
time.sleep(30)
"""
    child = subprocess.Popen([sys.executable, "-B", "-c", code, str(run.path / "coordinator.lock")], stdout=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == "locked"
        with pytest.raises(RunBusy):
            ScoringRun.open(run.path)
    finally:
        child.kill()
        child.wait(timeout=10)
        child.stdout.close()
    assert ScoringRun.open(run.path).row_count == 1


def test_subprocess_accept_then_abrupt_exit_survives(make_run, tmp_path):
    run = make_run(n=2)
    tasks = prepare(run)
    directory = transfer(tmp_path / "success", tasks[0])
    code = """import os,sys
from applications.scoring import ScoringRun
ScoringRun.open(sys.argv[1]).accept(sys.argv[2])
os._exit(73)
"""
    result = subprocess.run([sys.executable, "-B", "-c", code, str(run.path), str(directory)], timeout=30)
    assert result.returncode == 73
    reopened = ScoringRun.open(run.path)
    assert [d.state for d in reopened.diagnostics()] == [State.SUCCEEDED, State.SUBMITTED]
    assert not reopened.pending(Engine.PSICHIC)
