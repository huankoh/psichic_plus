import json
import select
import subprocess
import sys

import pytest

from applications.scoring import Acceptance, Engine, IncompleteRun, RunBusy, ScoringRun, State
from .conftest import prepare, transfer
from .test_reports import file_hashes

CHILD = r'''
import os, sys
from pathlib import Path
from applications.scoring import ScoringRun
run = ScoringRun.open(sys.argv[1])
boundary = sys.argv[2]
def audit(event, args):
    stop = False
    if event == "os.rename":
        destination = str(args[1])
        stop = ((boundary == "report-directory" and "/reports/v" in destination)
                or (boundary == "latest-pointer" and destination.endswith("/latest.json")))
    elif event == "open" and boundary == "snapshot":
        name = Path(str(args[0])).name
        stop = len(name) == 69 and name.endswith(".json")
    elif event == "os.link" and boundary == "acceptance":
        stop = "/receipts/" in str(args[1])
    if stop:
        print("boundary", flush=True)
        sys.stdin.read(1)
sys.addaudithook(audit)
if boundary == "acceptance":
    run.accept(sys.argv[3])
else:
    run.finalize()
'''


def kill_at_boundary(run, boundary, directory=None):
    command = [sys.executable, "-B", "-c", CHILD, str(run.path), boundary]
    if directory:
        command.append(str(directory))
    child = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        readable, _, _ = select.select([child.stdout], [], [], 30)
        assert readable, "Child did not reach the actual filesystem publication boundary"
        line = child.stdout.readline()
        if line != "boundary\n":
            raise AssertionError(f"Child exited without boundary: {line!r} {child.stderr.read()}")
        child.kill()
        assert child.wait(timeout=10) < 0
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=10)
        for stream in (child.stdin, child.stdout, child.stderr):
            stream.close()


@pytest.mark.parametrize("boundary", ["snapshot", "report-directory", "latest-pointer"])
def test_sigkill_during_report_publication_preserves_previous_version(make_run, tmp_path, boundary):
    run = make_run(n=2)
    first, second = prepare(run)
    run.accept(transfer(tmp_path / "success", first))
    run.accept(transfer(tmp_path / "failure", second, failure="controlled failure"))
    original = run.finalize()
    original_hashes = file_hashes(original.directory)
    original_rows = list(run.results())
    run.retry(second.entry_key, Engine.PSICHIC)
    retry, = prepare(run)
    run.accept(transfer(tmp_path / "retry", retry, affinity=9))
    kill_at_boundary(run, boundary)
    reopened = ScoringRun.open(run.path)
    assert reopened.latest() == original
    assert list(reopened.results()) == original_rows
    assert file_hashes(original.directory) == original_hashes
    final = reopened.finalize()
    assert final.version == 2
    assert [r["score"] for r in reopened.results()] == [6, 9]
    assert file_hashes(original.directory) == original_hashes
    assert json.loads((run.path / "latest.json").read_text())["version"] == 2
    assert reopened.finalize() == final
    if boundary == "report-directory":
        assert list((run.path / "reports").glob(".*.tmp"))


def test_sigkill_after_copy_before_receipt_acceptance_can_reimport(make_run, tmp_path):
    run = make_run(n=1)
    task, = prepare(run)
    directory = transfer(tmp_path / "transfer", task)
    kill_at_boundary(run, "acceptance", directory)
    reopened = ScoringRun.open(run.path)
    assert next(reopened.diagnostics()).state is State.SUBMITTED
    assert reopened.accept(directory) is Acceptance.ACCEPTED
    assert reopened.accept(directory) is Acceptance.IDENTICAL
    assert list((run.path / "staging").iterdir())


@pytest.mark.parametrize("complete_retry", [False, True])
def test_interrupted_snapshot_publishes_after_observations_change(make_run, tmp_path, complete_retry):
    run = make_run(n=3)
    tasks = prepare(run)
    for index, task in enumerate(tasks):
        run.accept(transfer(tmp_path / f"initial-{index}", task, failure="failed" if index else None))
    first = run.finalize()
    original = file_hashes(first.directory)
    run.retry(tasks[1].entry_key, Engine.PSICHIC)
    retry, = prepare(run)
    run.accept(transfer(tmp_path / "second", retry, affinity=8))
    kill_at_boundary(run, "report-directory")
    run = ScoringRun.open(run.path)
    run.retry(tasks[2].entry_key, Engine.PSICHIC)
    if complete_retry:
        retry, = prepare(run)
        run.accept(transfer(tmp_path / "third", retry, affinity=9))
    run.finalize()
    assert [row["score"] for row in run.results(2)] == [6, 8, None]
    assert run.latest().version == (3 if complete_retry else 2)
    assert file_hashes(first.directory) == original


def test_sigkill_during_create_rejects_incomplete_manifest(tmp_path, target):
    path = tmp_path / "interrupted-create"
    script = r'''
import sys
from applications.scoring import Asset, Entry, ScoringRun, Settings, Target

def population():
    yield Entry("one", "CCO", "source:1")
    print("row inserted", flush=True)
    sys.stdin.read(1)
    yield Entry("two", "CCN", "source:2")

ScoringRun.create(sys.argv[1], population(), target=Target("target", "ACDE", (Asset("fasta", "0"*64),)),
                  settings=Settings("psichic-only", psichic_options='{"controlled":true}'))
'''
    child = subprocess.Popen([sys.executable, "-B", "-c", script, str(path)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        assert select.select([child.stdout], [], [], 30)[0]
        assert child.stdout.readline() == "row inserted\n"
        child.kill()
        child.wait(timeout=10)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=10)
        child.stdin.close()
        child.stdout.close()
    with pytest.raises(IncompleteRun):
        ScoringRun.open(path)
    assert (path / "state.sqlite3").is_file()


def test_two_public_coordinators_contend_on_same_stable_lock(make_run):
    run = make_run(n=2)
    inode = (run.path / "coordinator.lock").stat().st_ino
    script = r'''
import sys
from applications.scoring import ScoringRun
run = ScoringRun.open(sys.argv[1])
rows = run.population()
next(rows)
print("reading", flush=True)
sys.stdin.read(1)
rows.close()
'''
    child = subprocess.Popen([sys.executable, "-B", "-c", script, str(run.path)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        assert select.select([child.stdout], [], [], 30)[0]
        assert child.stdout.readline() == "reading\n"
        with pytest.raises(RunBusy):
            prepare(run)
        with pytest.raises(RunBusy):
            run.finalize()
        child.stdin.write("x")
        child.stdin.flush()
        assert child.wait(timeout=10) == 0
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=10)
        child.stdin.close()
        child.stdout.close()
    assert (run.path / "coordinator.lock").stat().st_ino == inode
    assert len(prepare(run)) == 2
