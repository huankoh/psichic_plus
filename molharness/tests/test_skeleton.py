"""Contract tests for the skeleton. Fake blocks only. No models, no GPU."""

import csv
import json

import pytest

from molharness import (
    Blocks,
    CandidateTable,
    ExecutionEngine,
    JobArgs,
    Target,
    expand_pairs,
)


@pytest.fixture()
def candidates(tmp_path):
    path = tmp_path / "molecules.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ID", "SMILES"])
        writer.writeheader()
        writer.writerows(
            [
                {"ID": "aspirin", "SMILES": "CC(=O)Oc1ccccc1C(=O)O"},
                {"ID": "caffeine", "SMILES": "Cn1c(=O)c2c(ncn2C)n(C)c1=O"},
                {"ID": "broken", "SMILES": "INVALID"},
                {"ID": "ethanol", "SMILES": "CCO"},
            ]
        )
    return path


def test_brute_force(tmp_path, candidates):
    engine = ExecutionEngine(tmp_path / "run")
    h = engine.run("csv_source", path=str(candidates))
    h = engine.run("fake", data=h, name="psichic")
    h = engine.run("rank_by", data=h, by="psichic.affinity")
    by_id = {r["candidate_id"]: r for r in h.rows}
    assert by_id["broken"]["psichic.status"] == "failure"
    assert by_id["broken"]["rank_by.final_rank"] is None
    assert by_id["aspirin"]["psichic.status"] == "success"
    ranks = sorted(r["rank_by.final_rank"] for r in h.rows if r["rank_by.final_rank"])
    assert ranks == [1.0, 2.0, 3.0]
    manifest = json.loads((tmp_path / "run" / "manifest.json").read_text())
    assert [s["block"] for s in manifest["steps"]] == ["csv_source", "fake", "rank_by"]


def test_pyramid_gating_and_rescore_skip(tmp_path, candidates):
    engine = ExecutionEngine(tmp_path / "run")
    h = engine.run("csv_source", path=str(candidates))
    h = engine.run("fake", data=h, name="psichic")
    h = engine.run("topk", data=h, by="psichic.affinity", k=2)
    h = engine.run("fake", data=h, name="boltz", only_selected="shortlist", scale=1.0)
    selected = [r for r in h.rows if r.get("shortlist.selected")]
    assert len(selected) == 2
    assert all(r.get("boltz.status") == "success" for r in selected)
    assert all("boltz.status" not in r for r in h.rows if r not in selected)
    again = engine.run("fake", data=h, name="psichic")
    assert {r["candidate_id"]: r.get("psichic.affinity") for r in again.rows} == {
        r["candidate_id"]: r.get("psichic.affinity") for r in h.rows
    }


def test_shortage_marks_nothing_selected(tmp_path, candidates):
    engine = ExecutionEngine(tmp_path / "run")
    h = engine.run("csv_source", path=str(candidates))
    h = engine.run("fake", data=h, name="psichic")
    h = engine.run("topk", data=h, by="psichic.affinity", k=99)
    assert not any(r.get("shortlist.selected") for r in h.rows)
    assert all(r["shortlist.reason"] == "insufficient_successes" for r in h.rows)


def test_cycle_with_proposer(tmp_path):
    engine = ExecutionEngine(tmp_path / "run")
    h = CandidateTable()
    for i, sub in engine.cycles("opt", 2):
        h = sub.run(
            "list_proposer", data=h, smiles=("CCO", "CCN"), round_index=i, name="gen"
        )
        h = sub.run("fake", data=h, name="psichic")
    assert len(h) == 4
    assert all(r["psichic.status"] == "success" for r in h.rows)
    assert {r["source"] for r in h.rows} == {
        "generator:gen:round=0",
        "generator:gen:round=1",
    }


def test_multi_target_pairs(tmp_path, candidates):
    table = CandidateTable.from_csv(candidates)
    targets = [Target("kinaseA", "MKTAYIAKQR"), Target("kinaseB", "MEEPQSDPSV")]
    pairs = expand_pairs(
        table, targets, pairs=[("aspirin", "kinaseA"), ("aspirin", "kinaseB")]
    )
    assert len(pairs) == 2
    engine = ExecutionEngine(tmp_path / "run")
    scored = engine.run("fake", data=pairs, name="psichic")
    assert {r["target_id"] for r in scored.rows} == {"kinaseA", "kinaseB"}
    assert all(r["psichic.status"] == "success" for r in scored.rows)


def test_chunking_respects_rows_per_job(tmp_path, candidates):
    engine = ExecutionEngine(tmp_path / "run")
    h = engine.run("csv_source", path=str(candidates))
    h = engine.run("fake", data=h, name="psichic", job_args=JobArgs(rows_per_job=1))
    assert sum(1 for r in h.rows if r.get("psichic.status") == "success") == 3


def test_registry_and_manifests():
    assert "fake" in Blocks
    manifest = Blocks.get("fake").manifest()
    assert manifest["role"] == "scorer"
    assert manifest["chunked"] is True
    with pytest.raises(KeyError):
        Blocks.get("does-not-exist")


def test_slurm_mode_is_explicit(tmp_path):
    from molharness import RuntimeArgs

    engine = ExecutionEngine(tmp_path / "run")
    with pytest.raises(NotImplementedError):
        engine.run("fake", data=CandidateTable(), run_args=RuntimeArgs(mode="slurm"))
