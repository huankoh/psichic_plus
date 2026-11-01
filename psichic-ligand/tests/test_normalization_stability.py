"""Stable normalization preserves chemistry and bounds retries and worker lifetime."""

import dataclasses
import threading

import pytest

from helpers import graph_digest

from psichic_ligand.graph import smiles_to_graph
from psichic_ligand.normalization import pipeline


ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"


def install_passes(monkeypatch, normalizer, transitions):
    calls = []

    def normalize(current):
        calls.append(current)
        return pipeline.NormalizationResult(transitions.get(current, current), "SUCCESS")

    monkeypatch.setattr(normalizer, "_normalize_once", normalize)
    return calls


def test_stable_input_needs_only_one_complete_pass(monkeypatch):
    normalizer = pipeline.MoleculeNormalizer()
    calls = install_passes(monkeypatch, normalizer, {})
    assert normalizer.standardize_normalize(ASPIRIN) == pipeline.NormalizationResult(ASPIRIN, "SUCCESS")
    assert calls == [ASPIRIN]


@pytest.mark.parametrize("changes", [1, 2, 5, 19])
def test_changes_are_confirmed_before_success(monkeypatch, changes):
    normalizer = pipeline.MoleculeNormalizer()
    forms = ["C" * (8 + i) for i in range(changes + 1)]
    calls = install_passes(monkeypatch, normalizer, dict(zip(forms, forms[1:])))
    assert normalizer.standardize_normalize(forms[0]) == pipeline.NormalizationResult(forms[-1], "SUCCESS")
    assert calls == forms
    calls.clear()
    assert normalizer.standardize_normalize(forms[-1]) == pipeline.NormalizationResult(forms[-1], "SUCCESS")
    assert calls == [forms[-1]]


@pytest.mark.parametrize("forms", [["CCCCCCCC", "CCCCCCCCC"], ["CCCCCCCC", "CCCCCCCCC", "CCCCCCCCCC"]])
def test_cycles_are_explicit_failures(monkeypatch, forms):
    normalizer = pipeline.MoleculeNormalizer()
    calls = install_passes(monkeypatch, normalizer, dict(zip(forms, forms[1:] + forms[:1])))
    assert normalizer.standardize_normalize(forms[0]) == pipeline.NormalizationResult(
        None, "NORMALIZATION_CYCLE", "Normalization entered a cycle",
    )
    assert calls == forms


def test_limit_does_not_return_the_last_unconfirmed_molecule(monkeypatch):
    normalizer = pipeline.MoleculeNormalizer()
    forms = ["C" * (8 + i) for i in range(21)]
    calls = install_passes(monkeypatch, normalizer, dict(zip(forms, forms[1:])))
    assert normalizer.standardize_normalize(forms[0]) == pipeline.NormalizationResult(
        None, "NORMALIZATION_LIMIT", "Normalization did not stabilize within 20 passes",
    )
    assert calls == forms[:20]


@pytest.mark.parametrize("smiles", ["not-a-smiles", "C("])
def test_invalid_smiles_returns_a_failure(smiles):
    result = pipeline.MoleculeNormalizer().standardize_normalize(smiles)
    assert result.status == "INVALID_SMILES"
    assert result.smiles is None


@pytest.mark.parametrize("status,message", [
    ("NON_SMALL_MOLECULE", "Molecular weight outside allowed range"),
    ("MIXTURE", "Multiple fragments detected"),
    ("GENERAL_ERROR", "General standardization error"),
])
def test_intermediate_failure_keeps_original_status(monkeypatch, status, message):
    normalizer = pipeline.MoleculeNormalizer()
    calls = []

    def normalize(smiles):
        calls.append(smiles)
        if len(calls) == 1:
            return pipeline.NormalizationResult("CCCCCCCCC", "SUCCESS")
        return pipeline.NormalizationResult(None, status, message)

    monkeypatch.setattr(normalizer, "_normalize_once", normalize)
    assert normalizer.standardize_normalize("CCCCCCCC") == pipeline.NormalizationResult(None, status, message)
    assert calls == ["CCCCCCCC", "CCCCCCCCC"]


def test_intermediate_exception_does_not_return_previous_success(monkeypatch):
    normalizer = pipeline.MoleculeNormalizer()
    calls = []

    def standardize(smiles):
        calls.append(smiles)
        if len(calls) == 1:
            return pipeline.NormalizationResult("CCCCCCCCC", "SUCCESS")
        raise ValueError("second pass failed")

    monkeypatch.setattr(normalizer, "_normalize_once", standardize)
    assert normalizer.standardize_normalize("CCCCCCCC") == pipeline.NormalizationResult(
        None, "GENERAL_ERROR", "second pass failed",
    )
    assert calls == ["CCCCCCCC", "CCCCCCCCC"]


def test_timeout_budget_covers_all_passes_in_one_worker(monkeypatch):
    normalizer = pipeline.MoleculeNormalizer(10)
    clock = [0.0]
    threads = []
    calls = []
    monkeypatch.setattr(pipeline.time, "monotonic", lambda: clock[0])

    def standardize(current):
        threads.append(threading.current_thread())
        calls.append(current)
        clock[0] += 4
        return pipeline.NormalizationResult(current + "C", "SUCCESS")

    monkeypatch.setattr(normalizer, "_normalize_once", standardize)
    assert normalizer.standardize_normalize("CCCCCCCC") == pipeline.NormalizationResult(
        None, "TIMEOUT", "Standardization timeout",
    )
    assert calls == ["CCCCCCCC", "CCCCCCCCC", "CCCCCCCCCC"]
    assert len(set(threads)) == 1


@pytest.mark.parametrize("late_failure", [False, True])
def test_late_result_or_exception_cannot_override_deadline(monkeypatch, late_failure):
    normalizer = pipeline.MoleculeNormalizer(10)
    clock = [0.0]
    monkeypatch.setattr(pipeline.time, "monotonic", lambda: clock[0])

    def standardize(smiles):
        clock[0] = 11
        if late_failure:
            raise ValueError("failed after the deadline")
        return pipeline.NormalizationResult(smiles, "SUCCESS")

    monkeypatch.setattr(normalizer, "_normalize_once", standardize)
    assert normalizer.standardize_normalize(ASPIRIN).status == "TIMEOUT"


@pytest.mark.parametrize("late_failure", [False, True])
def test_timed_out_worker_stops_before_another_pass_and_cannot_poison_reuse(monkeypatch, late_failure):
    normalizer = pipeline.MoleculeNormalizer(0.1)
    started, release = threading.Event(), threading.Event()
    old_workers, calls = [], []

    def standardize(current):
        calls.append(current)
        if current == "CCCCCCCC":
            old_workers.append(threading.current_thread())
            started.set()
            assert release.wait(5)
            if late_failure:
                raise ValueError("old worker failed")
            return pipeline.NormalizationResult("CCCCCCCCC", "SUCCESS")
        return pipeline.NormalizationResult(current, "SUCCESS")

    monkeypatch.setattr(normalizer, "_normalize_once", standardize)
    try:
        assert normalizer.standardize_normalize("CCCCCCCC").status == "TIMEOUT"
        assert started.wait(5)
        normalizer.timeout_handler.timeout_seconds = 5
        assert normalizer.standardize_normalize(ASPIRIN).smiles == ASPIRIN
        release.set()
        old_workers[0].join(5)
        assert not old_workers[0].is_alive()
        assert normalizer.standardize_normalize(ASPIRIN).smiles == ASPIRIN
        assert calls == ["CCCCCCCC", ASPIRIN, ASPIRIN]
    finally:
        release.set()
        for worker in old_workers:
            worker.join(5)


# Recorded regressions from the old one-pass pipeline, including the case that
# requires five changes followed by a sixth pass confirming the final form.
CHEMISTRY_CASES = [
    (
        "Cc1cccc(-c2cc(Cl)cc(-c3cc4c(c(=O)[nH]3)CON=C4NC=[NH+][O-])n2)c1",
        "Cc1cccc(-c2cc(Cl)cc(-c3cc4c(c(=O)[nH]3)CON=C4NC=NO)n2)c1",
    ),
    (
        "Cc1cccc2cc3c([N+]([O-])=C([O-])[NH2+][O-])nn(C)c3nc12",
        "Cc1cccc2cc3c(N(O)C(=O)NO)nn(C)c3nc12",
    ),
    (
        "CNC(=O)CC1=C(C(N)=O)C(C(N)=O)NCC(N)=C(N)C=C(C=C(N=O)N=C(N)N=C2N(CC(=O)S)CCSC23CC3)NC(=O)NN1",
        "CNC(=O)Cc1n[nH]c(=O)[nH]c(CC(=NO)NC(=N)N=C2N(CC(=O)S)CCSC23CC3)cc(N)c(N)cnc(=C(N)O)c1=C(N)O",
    ),
]



@pytest.mark.parametrize("raw,expected", CHEMISTRY_CASES)
def test_recorded_chemistry_reaches_stable_molecule_and_graph(raw, expected):
    normalizer = pipeline.MoleculeNormalizer()
    actual = normalizer.standardize_normalize(raw)
    assert actual.status == "SUCCESS" and actual.smiles is not None
    repeat = normalizer.standardize_normalize(actual.smiles)
    assert repeat == actual
    for one_hot in [False, True]:
        assert graph_digest(smiles_to_graph(actual.smiles, one_hot)) == graph_digest(
            smiles_to_graph(repeat.smiles, one_hot),
        )
