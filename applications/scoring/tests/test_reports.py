import csv
import hashlib
import json
from pathlib import Path

import pytest

from applications.scorer import BetaDistribution, Failure
from applications.scorer.boltz2.models import BoltzScore, Candidate, EngineRun, EnsembleMethod, Metric, Mode, PSICHICScore, Plan
from applications.scorer.boltz2.workflow import run_workflow
from applications.scoring import Completion, Engine, Report, ScoringRun, StateError, Waiting
from .conftest import prepare, transfer

GOLDEN = json.loads((Path(__file__).parents[2] / "scorer/boltz2/tests/fixtures/concentration-v1.json").read_text())


def file_hashes(directory):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir() if p.is_file()}


@pytest.mark.parametrize("case", GOLDEN["cases"], ids=lambda c: c["id"])
def test_sql_finalization_matches_every_golden_case(make_run, tmp_path, case):
    selected = [r for r in case["rows"] if r["selected_for_ensemble"]]
    run = make_run("full-ensemble", n=len(selected))
    for engine in Engine:
        for task, row in zip(prepare(run, engine), selected):
            kwargs = {"affinity": row["psichic_affinity"], "alpha": row["psichic_alpha"], "beta": row["psichic_beta"]}
            if engine is Engine.BOLTZ:
                if row["boltz_affinity_status"] != "valid":
                    kwargs["failure"] = "controlled failure"
                else:
                    kwargs.update(boltz=row["boltz_affinity_raw"], probability=row["boltz_binder_probability_not_a_formula_input"] or .5)
            run.accept(transfer(tmp_path / f"{engine.value}-{task.entry_key}", task, **kwargs))
    report = run.finalize()
    assert isinstance(report, Report)
    result = list(run.results())
    policy = {r["row_key"]: r for r in case["policy"]["manifest_results"]}
    for record, source in zip(result, selected):
        expected = policy[source["row_key"]]
        if expected["score"] is None:
            assert record["score"] is None and record["final_rank"] is None
        else:
            assert record["score"] == pytest.approx(expected["score"], abs=1e-12, rel=1e-12)
            assert record["final_rank"] == expected["average_rank_one_based"]
    envelope = json.loads((report.directory / "results.json").read_text())
    assert envelope["ranking"]["discrimination"] == case["policy"]["discrimination"]
    assert envelope["rows"] == result
    assert envelope["ranking"]["reference_commit"] == GOLDEN["provenance"]["reference_integrity"]["commit"]


@pytest.mark.parametrize("mode,method,metric", [
    ("full-ensemble", EnsembleMethod.RANK, Metric.PROBABILITY),
    ("full-ensemble", EnsembleMethod.RANK, Metric.AFFINITY),
    ("pyramid-ensemble", EnsembleMethod.RANK, Metric.PROBABILITY),
    ("pyramid-refinement", None, Metric.AFFINITY),
    ("boltz-only", None, Metric.PROBABILITY),
    ("boltz-only", None, Metric.AFFINITY),
    ("full-ensemble", EnsembleMethod.CONCENTRATION, Metric.AFFINITY),
])
def test_saved_observation_equivalence_to_in_memory_coordinator(make_run, tmp_path, mode, method, metric):
    kwargs = {"ensemble": method, "metric": metric}
    shortlist = 3 if "pyramid" in mode else None
    if shortlist:
        kwargs["shortlist_size"] = shortlist
    if method is EnsembleMethod.RANK:
        kwargs.update(psichic_weight=2., boltz_weight=.5)
    run = make_run(mode, n=4, **kwargs)
    rows = list(run.population())
    affinities, shapes, boltz, probabilities = [3, 9, 6, 6], [(1, 3), (6, 2), (8, 8), (4, 4)], [-8, -4, -1, -4], [.9, .1, .5, .1]
    p, b = {}, {}
    if Engine.PSICHIC in run.settings.engines:
        for task in prepare(run):
            i = int(task.entry_key[4:]) - 1
            run.accept(transfer(tmp_path / f"p{i}", task, affinity=affinities[i], alpha=shapes[i][0], beta=shapes[i][1]))
            p[task.entry_key] = PSICHICScore(affinities[i], -affinities[i], "CCO", distribution=BetaDistribution(*shapes[i]))
    run.select()
    for task in prepare(run, Engine.BOLTZ):
        i = int(task.entry_key[4:]) - 1
        failure = "controlled failure" if i == 2 else None
        run.accept(transfer(tmp_path / f"b{i}", task, boltz=boltz[i], probability=probabilities[i], failure=failure))
        b[task.entry_key] = Failure(task.entry_key, "controlled", failure) if failure else BoltzScore(probabilities[i], boltz[i], {}, "", "CCO")
    run.finalize()
    legacy_args = {"ensemble": method} if method else {}
    if method is EnsembleMethod.RANK:
        legacy_args.update(psichic_weight=2., boltz_weight=.5)
    candidates = [Candidate(r.ordinal, r.entry.id, r.entry.smiles) for r in rows]
    mapping = {candidate.key: row.key for candidate, row in zip(candidates, rows)}
    expected = run_workflow(candidates, Plan(Mode(mode), shortlist, metric, **legacy_args),
                            psichic=lambda selected: EngineRun({r.key: p[mapping[r.key]] for r in selected}),
                            boltz=lambda selected: EngineRun({r.key: b[mapping[r.key]] for r in selected}))
    actual = list(run.results())
    for a, e in zip(actual, expected.rows):
        assert a["final_rank"] == e.final_rank
        score = e.normalized_ensemble_score if method is EnsembleMethod.CONCENTRATION else e.experimental_combined_value
        if method:
            assert a["score"] == (pytest.approx(score, abs=1e-12, rel=1e-12) if score is not None else None)
        assert a["ensemble_psichic_rank"] == e.ensemble_psichic_rank
        assert a["ensemble_boltz_rank"] == e.ensemble_boltz_rank


def test_psichic_only_reports_rank_ties_and_stable_display(make_run, tmp_path):
    run = make_run(n=4)
    for task, affinity in zip(prepare(run), [3, 9, 9, 6]):
        run.accept(transfer(tmp_path / task.entry_key, task, affinity=affinity))
    report = run.finalize(top_n=2)
    rows = list(run.results())
    assert [r["final_rank"] for r in rows] == [4, 1.5, 1.5, 3]
    assert [r["position"] for r in rows] == [4, 1, 2, 3]
    assert [r["entry"]["ordinal"] for r in rows] == [1, 2, 3, 4]
    assert all(r["boltz"]["state"] == "not-run" for r in rows)
    envelope = json.loads((report.directory / "results.json").read_text())
    assert envelope["rows"] == rows
    with (report.directory / "results.csv").open() as handle:
        csv_rows = list(csv.DictReader(handle))
    assert [float(r["final_rank"]) for r in csv_rows] == [r["final_rank"] for r in rows]
    assert [json.loads(r["psichic"]) for r in csv_rows] == [r["psichic"] for r in rows]
    summary = (report.directory / "summary.md").read_text()
    assert rows[1]["entry"]["key"] in summary and rows[0]["entry"]["key"] not in summary
    before = file_hashes(report.directory)
    again = ScoringRun.open(run.path).finalize()
    assert again == report and file_hashes(report.directory) == before
    assert len(list((run.path / "reports").glob("v*"))) == 1


def test_unfinished_population_has_diagnostics_and_no_report(make_run):
    run = make_run()
    assert run.finalize() == Waiting(3, 0, 0)
    prepare(run, limit=2)
    assert run.finalize() == Waiting(1, 2, 0)
    assert run.latest() is None
    assert list(run.results()) == []
    assert not list((run.path / "reports").iterdir())


def test_shortage_report_and_immutable_retry_versions(make_run, tmp_path):
    run = make_run("pyramid-ensemble", n=2, shortlist_size=2)
    first, second = prepare(run)
    run.accept(transfer(tmp_path / "p1", first))
    run.accept(transfer(tmp_path / "p2", second, failure="failed"))
    shortage = run.finalize()
    assert shortage.status is Completion.SHORTAGE
    assert not run.pending(Engine.BOLTZ)
    initial = file_hashes(shortage.directory)
    run.retry(second.entry_key, Engine.PSICHIC)
    retry, = prepare(run)
    run.accept(transfer(tmp_path / "retry", retry))
    assert isinstance(run.finalize(), Waiting)
    for task in prepare(run, Engine.BOLTZ):
        run.accept(transfer(tmp_path / f"b{task.entry_key}", task))
    final = run.finalize()
    assert final.version == 2 and final.status is Completion.COMPLETE
    assert file_hashes(shortage.directory) == initial
    assert list(run.results(shortage.version))[0]["score"] is None


@pytest.mark.parametrize("method", [EnsembleMethod.RANK, EnsembleMethod.CONCENTRATION])
def test_failure_retry_changes_common_cohort_and_preserves_reports(make_run, tmp_path, method):
    run = make_run("pyramid-ensemble", n=3, shortlist_size=3, ensemble=method)
    for task, values in zip(prepare(run), [(3, 1, 3), (6, 4, 4), (9, 12, 4)]):
        run.accept(transfer(tmp_path / f"p{task.entry_key}", task, affinity=values[0], alpha=values[1], beta=values[2]))
    run.select()
    tasks = prepare(run, Engine.BOLTZ)
    for i, task in enumerate(tasks):
        run.accept(transfer(tmp_path / f"b{i}", task, boltz=[-8, -4, -1][i], probability=[.9, .5, .1][i], failure="failed" if i == 2 else None))
    first = run.finalize()
    before = list(run.results())
    hashes = file_hashes(first.directory)
    assert first.status is Completion.PARTIAL
    run.retry(tasks[2].entry_key, Engine.BOLTZ)
    assert isinstance(run.finalize(), Waiting)
    retry, = prepare(run, Engine.BOLTZ)
    run.accept(transfer(tmp_path / "retry", retry, boltz=-1, probability=.1))
    second = run.finalize()
    after = list(run.results())
    assert first.version == 1 and second.version == 2
    assert any(before[i]["score"] != after[i]["score"] for i in (0, 1))
    assert before[2]["score"] is None and after[2]["score"] is not None
    assert file_hashes(first.directory) == hashes
    assert list(run.results(1)) == before
    assert json.loads((second.directory / "results.json").read_text())["warnings"]


def test_shortlist_ties_no_backfill_after_failure(make_run, tmp_path):
    run = make_run("pyramid-refinement", n=4, shortlist_size=2)
    for task in reversed(prepare(run)):
        run.accept(transfer(tmp_path / f"p{task.entry_key}", task))
    selected = run.select()
    assert selected.eligible == 4 and run.select() == selected
    tasks = prepare(run, Engine.BOLTZ)
    assert [t.entry_key for t in tasks] == ["row_000000000001", "row_000000000002"]
    for i, task in enumerate(tasks):
        run.accept(transfer(tmp_path / f"b{i}", task, failure="failed" if i else None))
    run.finalize()
    assert not run.pending(Engine.BOLTZ)
    assert [r["selected"] for r in run.results()] == [True, True, False, False]
    with pytest.raises(StateError):
        run.retry(tasks[1].entry_key, Engine.PSICHIC)


def test_extreme_finite_boltz_affinities_normalize_without_overflow(make_run, tmp_path):
    run = make_run("full-ensemble", n=2)
    for task in prepare(run):
        run.accept(transfer(tmp_path / f"p{task.entry_key}", task))
    for task, value in zip(prepare(run, Engine.BOLTZ), [-1e308, 1e308]):
        run.accept(transfer(tmp_path / f"b{task.entry_key}", task, boltz=value))
    run.finalize()
    assert [r["score"] for r in run.results()] == [.8125, .1875]
