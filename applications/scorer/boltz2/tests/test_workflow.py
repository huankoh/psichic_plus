import math

import pytest

from applications.scorer import Failure, Success
from applications.scorer.boltz2.models import BoltzScore, Candidate, EngineRun, Metric, Mode, PSICHICScore, Plan
from applications.scorer.boltz2.workflow import average_ranks, row_record, run_workflow, score_psichic


def engines(a_values, b_values):
    calls = {"a": [], "b": []}

    def a(rows):
        calls["a"].extend(row.row for row in rows)
        return EngineRun({row.key: Failure(row.key, "a", "bad") if a_values[row.row - 1] is None else
                          PSICHICScore(a_values[row.row - 1], -a_values[row.row - 1], row.smiles) for row in rows})

    def b(rows):
        calls["b"].extend(row.row for row in rows)
        return EngineRun({row.key: Failure(row.key, "b", "bad") if b_values[row.row - 1] is None else
                          BoltzScore(b_values[row.row - 1], b_values[row.row - 1] * 10, {}, "native", row.smiles) for row in rows})

    return a, b, calls


ROWS = [Candidate(1, "same", "CCO"), Candidate(2, "same", "CCO"), Candidate(3, "", "CCC"), Candidate(4, "../id", "CCN")]


@pytest.mark.parametrize("mode,expected", [
    (Mode.BOLTZ, [1, 2, 3, 4]), (Mode.ENSEMBLE, [1, 2, 3, 4]),
    (Mode.PYRAMID_ENSEMBLE, [2, 3]), (Mode.REFINEMENT, [2, 3]),
])
def test_all_modes_preserve_rows_and_select_the_right_inputs(mode, expected):
    a, b, calls = engines([1, 4, 4, 2], [.8, .3, .9, .4])
    report = run_workflow(ROWS, Plan(mode, 2 if mode.pyramid else None), psichic=a, boltz=b)
    assert calls["a"] == ([] if mode is Mode.BOLTZ else [1, 2, 3, 4])
    assert calls["b"] == expected
    assert [r.candidate for r in report.rows] == ROWS
    assert not report.failed
    if mode.pyramid:
        assert [r.selection for r in report.rows] == ["not_selected", "selected", "selected", "not_selected"]
        assert report.rows[0].final_rank is None
    assert report.rows[2].final_rank == 1


def test_shortlist_tie_uses_row_order_and_does_not_replenish_after_failure():
    a, b, calls = engines([4, 4, 4, 1], [.8, None, .9, .4])
    report = run_workflow(ROWS, Plan(Mode.REFINEMENT, 2), psichic=a, boltz=b)
    assert calls["b"] == [1, 2]
    assert [r.psichic_screen_rank for r in report.rows] == [2, 2, 2, 4]
    assert [r.selection_reason for r in report.rows[:3]] == ["shortlist_tie_input_order"] * 3
    assert report.rows[1].final_rank is None
    assert report.rows[0].final_rank == 1
    assert report.failed


def test_ensemble_uses_only_common_successes_and_preserves_partial_scores():
    a, b, calls = engines([9, 8, 7, None], [None, .2, .9, .99])
    report = run_workflow(ROWS, Plan(Mode.ENSEMBLE), psichic=a, boltz=b)
    assert calls["b"] == [1, 2, 3, 4]
    assert [r.final_rank for r in report.rows] == [None, 1.5, 1.5, None]
    assert report.rows[1].ensemble_psichic_rank == 1
    assert report.rows[1].psichic_screen_rank == 2
    assert report.rows[2].ensemble_boltz_rank == 1
    assert report.rows[3].boltz.affinity_probability_binary == .99
    assert row_record(report.rows[0])["unranked_reason"] == "engine_failure"


@pytest.mark.parametrize("metric,ranks", [(Metric.PROBABILITY, [4, 3, 2, 1]), (Metric.AFFINITY, [1, 2, 3, 4])])
def test_native_ranking_direction(metric, ranks):
    _, b, _ = engines([], [.1, .2, .3, .4])
    report = run_workflow(ROWS, Plan(Mode.BOLTZ, metric=metric), boltz=b)
    assert [r.final_rank for r in report.rows] == ranks


def test_weights_change_rank_without_averaging_raw_scores():
    a, b, _ = engines([1000, 5, 2, 1], [.1, .2, .3, .4])
    report = run_workflow(ROWS, Plan(Mode.ENSEMBLE, psichic_weight=0, boltz_weight=2), psichic=a, boltz=b)
    assert [r.final_rank for r in report.rows] == [4, 3, 2, 1]


def test_large_finite_weights_do_not_overflow_rank_combination():
    a, b, _ = engines([1, 2, 3, 4], [.4, .3, .2, .1])
    report = run_workflow(ROWS, Plan(Mode.ENSEMBLE, psichic_weight=1e308, boltz_weight=1), psichic=a, boltz=b)
    assert [r.final_rank for r in report.rows] == [4, 3, 2, 1]
    assert all(math.isfinite(r.experimental_combined_value) for r in report.rows)


def test_insufficient_successes_is_explicit_and_starts_no_boltz():
    a, b, calls = engines([None, 4, None, None], [.1] * 4)
    report = run_workflow(ROWS, Plan(Mode.REFINEMENT, 2), psichic=a, boltz=b)
    assert not calls["b"]
    assert report.failed and "got 1" in report.errors[0]
    assert len(report.rows) == 4


@pytest.mark.parametrize("kwargs", [
    {"mode": Mode.REFINEMENT}, {"mode": Mode.REFINEMENT, "shortlist_size": 0},
    {"mode": Mode.BOLTZ, "shortlist_size": 1},
    {"mode": Mode.ENSEMBLE, "psichic_weight": math.nan},
    {"mode": Mode.ENSEMBLE, "psichic_weight": 0, "boltz_weight": 0},
    {"mode": Mode.ENSEMBLE, "boltz_weight": -1},
])
def test_invalid_plans_are_rejected(kwargs):
    with pytest.raises(ValueError):
        Plan(**kwargs)


def test_average_ranks_and_nonfinite_values():
    assert average_ranks({"a": 3, "b": 3, "c": 1}, higher_is_better=True) == {"a": 1.5, "b": 1.5, "c": 3}
    with pytest.raises(ValueError):
        average_ranks({"a": math.inf}, higher_is_better=True)


@pytest.mark.parametrize("results", [[], [Success("unknown", 3, -3)], [Success(ROWS[0].key, 3, -3)] * 2])
def test_psichic_mapping_failures_never_merge_duplicate_user_ids(results):
    class Predictor:
        def score(self, inputs):
            assert inputs == [(r.key, r.smiles) for r in ROWS]
            return results

    run = score_psichic(ROWS, Predictor())
    assert set(run.outcomes) == {r.key for r in ROWS}
    assert all(isinstance(value, Failure) for value in run.outcomes.values())


def test_engine_exception_retains_every_row_and_runs_full_boltz():
    _, b, calls = engines([], [.1, .2, .3, .4])

    def failed(_):
        raise RuntimeError("weights missing")

    report = run_workflow(ROWS, Plan(Mode.ENSEMBLE), psichic=failed, boltz=b)
    assert report.failed and len(report.rows) == 4
    assert calls["b"] == [1, 2, 3, 4]
    assert all(row.final_rank is None for row in report.rows)
