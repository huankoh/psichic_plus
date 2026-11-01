"""Invocation isolation, including workers completing after a timeout."""
import threading
from types import SimpleNamespace

import pytest


from psichic_ligand.normalization import pipeline


def wait(event):
    assert event.wait(5), "Test synchronization did not complete"


def start_call(handler, function):
    outcome = {}

    def call():
        try:
            outcome["result"] = handler.run_with_timeout(function)
        except Exception as error:
            outcome["exception"] = error

    caller = threading.Thread(target=call, daemon=True)
    caller.start()
    return caller, outcome


def finish(thread):
    thread.join(5)
    assert not thread.is_alive(), "Test worker did not finish"


def pause_completed_join(monkeypatch, selected_worker):
    """Expose the interval between join completion and reading a call's output."""
    joined, resume = threading.Event(), threading.Event()
    workers = []

    class ControlledThread(threading.Thread):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.index = len(workers)
            workers.append(self)

        def join(self, timeout=None):
            super().join(timeout)
            if self.index == selected_worker and not self.is_alive():
                joined.set()
                wait(resume)

    monkeypatch.setattr(pipeline, "threading", SimpleNamespace(Thread=ControlledThread))
    return joined, resume, workers


def test_failure_does_not_poison_next_success():
    handler = pipeline.TimeoutHandler(5)
    error = ValueError("first molecule failed")

    def fail():
        raise error

    with pytest.raises(ValueError) as caught:
        handler.run_with_timeout(fail)
    assert caught.value is error
    assert handler.run_with_timeout(lambda: "second molecule") == "second molecule"


def test_normalizer_recovers_after_standardization_failure(monkeypatch):
    from rdkit import Chem

    normalizer = pipeline.MoleculeNormalizer(5)
    smiles = "CC(=O)Oc1ccccc1C(=O)O"

    def fail(mol):
        raise ValueError("first molecule failed")

    monkeypatch.setattr(normalizer, "_standardize_molecule", fail)
    failed = normalizer.standardize_normalize(smiles)
    assert (failed.smiles, failed.status, failed.error_message) == (
        None, "GENERAL_ERROR", "first molecule failed")
    monkeypatch.setattr(normalizer, "_standardize_molecule",
                        lambda mol: (mol, SimpleNamespace(value=1)))
    actual = normalizer.standardize_normalize(smiles)
    assert (actual.smiles, actual.status, actual.error_message) == (
        Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=False),
        "SUCCESS", None)


def test_late_exception_cannot_poison_an_active_call():
    handler = pipeline.TimeoutHandler(0)
    release_late, late_started = threading.Event(), threading.Event()
    release_active, active_started = threading.Event(), threading.Event()
    late_worker = []

    def late_failure():
        late_worker.append(threading.current_thread())
        late_started.set()
        wait(release_late)
        raise ValueError("late failure")

    def active_success():
        active_started.set()
        wait(release_active)
        return "active result"

    try:
        assert handler.run_with_timeout(late_failure) is None
        wait(late_started)
        handler.timeout_seconds = 5
        caller, outcome = start_call(handler, active_success)
        wait(active_started)
        release_late.set()
        finish(late_worker[0])
        release_active.set()
        finish(caller)
        assert outcome == {"result": "active result"}
        assert handler.run_with_timeout(lambda: "next result") == "next result"
    finally:
        release_late.set()
        release_active.set()


def test_late_result_cannot_replace_an_active_calls_result(monkeypatch):
    joined, resume, workers = pause_completed_join(monkeypatch, selected_worker=1)
    handler = pipeline.TimeoutHandler(0)
    release_late = threading.Event()

    def late_success():
        wait(release_late)
        return "late result"

    try:
        assert handler.run_with_timeout(late_success) is None
        handler.timeout_seconds = 5
        caller, outcome = start_call(handler, lambda: "active result")
        wait(joined)
        release_late.set()
        finish(workers[0])
        resume.set()
        finish(caller)
        assert outcome == {"result": "active result"}
    finally:
        release_late.set()
        resume.set()


@pytest.mark.parametrize("first_fails", [False, True])
def test_overlapping_calls_keep_results_and_errors_separate(monkeypatch, first_fails):
    joined, resume, _ = pause_completed_join(monkeypatch, selected_worker=0)
    handler = pipeline.TimeoutHandler(5)
    error = ValueError("first failure")

    def first():
        if first_fails:
            raise error
        return "first result"

    try:
        caller, outcome = start_call(handler, first)
        wait(joined)
        assert handler.run_with_timeout(lambda: "second result") == "second result"
        resume.set()
        finish(caller)
        assert outcome == ({"exception": error} if first_fails else {"result": "first result"})
    finally:
        resume.set()
