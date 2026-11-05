import json
import math
from pathlib import Path

import pytest

from applications.scorer import (
    BetaDistribution, DistributionFailure, DistributionNotRequested, Failure, Output, Success,
)
from applications.scorer.boltz2.ensemble import combine_ensemble, concentration_scores
from applications.scorer.boltz2.models import (
    BoltzScore, Candidate, EngineRun, EnsembleMethod, Metric, Mode, PSICHICScore, Plan,
)
from applications.scorer.boltz2.workflow import row_record, run_workflow, score_psichic


GOLDEN = json.loads((Path(__file__).parent / "fixtures/concentration-v1.json").read_text())
TOLERANCE = 1e-12  # Cross-platform math.log rounding may differ in the last bits.


def concentration_plan(mode=Mode.ENSEMBLE, shortlist_size=None, **kwargs):
    return Plan(mode, shortlist_size, ensemble=EnsembleMethod.CONCENTRATION, **kwargs)


def rich(affinity=6, alpha=4, beta=4, **kwargs):
    return PSICHICScore(affinity, -affinity, "CCO", distribution=BetaDistribution(alpha, beta), **kwargs)


def native(affinity=-3, probability=.5):
    return BoltzScore(probability, affinity, {}, "native", "CCO")


@pytest.mark.parametrize("case", GOLDEN["cases"], ids=lambda case: case["id"])
def test_all_recorded_source_outputs_and_integration_policy(case):
    call = case["call"]
    actual = concentration_scores(call["psichic_predictions"], call["psichic_concentrations"],
                                  [-value for value in call["boltz2_predictions"]])
    assert actual == pytest.approx(case["scores"], rel=TOLERANCE, abs=TOLERANCE)
    psichic, boltz = {}, {}
    for row in case["rows"]:
        if not row["selected_for_ensemble"]:
            continue
        key = row["row_key"]
        psichic[key] = rich(row["psichic_affinity"], row["psichic_alpha"], row["psichic_beta"])
        boltz[key] = (native(row["boltz_affinity_raw"], row["boltz_binder_probability_not_a_formula_input"] or .5)
                      if row["boltz_affinity_status"] == "valid" else Failure(key, "boltz", "failed"))
    combined = combine_ensemble(psichic, boltz, concentration_plan())
    assert combined.cohort == call["row_keys_in_call_order"]
    assert combined.discrimination == case["policy"]["discrimination"]
    for row in case["policy"]["manifest_results"]:
        key = row["row_key"]
        if row["score"] is None:
            assert key not in combined.scores and key not in combined.ranks
        else:
            assert combined.scores[key] == pytest.approx(row["score"], rel=TOLERANCE, abs=TOLERANCE)
            assert combined.ranks[key] == row["average_rank_one_based"]
    assert sorted(combined.ranks, key=combined.ranks.get) == case["policy"]["display_order_row_keys"]
    assert bool(combined.warning) == ("lacks_discrimination" in combined.discrimination)
    assert combined.metadata()["reference_commit"] == GOLDEN["provenance"]["reference_integrity"]["commit"]


@pytest.mark.parametrize("case", GOLDEN["errors"], ids=lambda case: case["case_id"])
def test_all_recorded_source_errors(case):
    call = case["actual_upstream_call"]
    assert case["actual_upstream_exception"]["exception_type"] == "ValueError"
    with pytest.raises(ValueError, match="positive|same length"):
        concentration_scores(call["psichic_predictions"], call["psichic_concentrations"],
                             [-value for value in call["boltz2_predictions"]])


@pytest.mark.parametrize("bad", [True, "1", None, math.nan, math.inf, -math.inf, 10**1000])
@pytest.mark.parametrize("column", [0, 1, 2])
def test_mathematical_boundary_rejects_invalid_types_and_nonfinite_numbers(bad, column):
    columns = [[3.0], [4.0], [-8.0]]
    columns[column] = [bad]
    with pytest.raises(ValueError, match="finite numbers"):
        concentration_scores(*columns)


def test_finite_extreme_values_normalize_without_overflow():
    assert concentration_scores([0, 12], [1, 2], [-1e308, 1e308]) == [.75, .5]


def test_boltz_direction_ignores_probability():
    a = {"a": rich(), "b": rich()}
    b = {"a": native(-10, .01), "b": native(-1, .99)}
    combined = combine_ensemble(a, b, concentration_plan())
    assert combined.ranks == {"a": 1, "b": 2}
    assert combined.scores == {"a": .8125, "b": .1875}


def test_failure_recomputes_all_three_normalizations_from_exact_common_cohort():
    before, after = GOLDEN["cases"][-2:]
    assert before["id"] == "cohort_before" and after["id"] == "cohort_after"
    rows = [Candidate(i, "duplicate", "CCO") for i in range(1, 5)]
    a = {r.key: rich(*values) for r, values in zip(rows, [(3, 1, 3), (6, 4, 4), (9, 12, 4), (10.5, 28, 4)])}
    b = {r.key: native(v) for r, v in zip(rows, [-8, -4, -1, -20])}
    b[rows[2].key] = Failure(rows[2].key, "boltz", "failed")
    a[rows[3].key] = PSICHICScore(10.5, -10.5, "CCO", distribution=DistributionFailure("invalid", "bad", "bad"))
    report = run_workflow(rows, concentration_plan(), psichic=lambda _: EngineRun(a), boltz=lambda _: EngineRun(b))
    assert report.ensemble.cohort == [r.key for r in rows[:2]]
    assert [r.normalized_ensemble_score for r in report.rows] == [.75, .5, None, None]
    assert report.rows[3].psichic.raw_affinity == 10.5
    assert row_record(report.rows[3])["unranked_reason"] == "psichic_distribution_failed"
    assert report.failed


@pytest.mark.parametrize("distribution", [DistributionNotRequested(), DistributionFailure("bad", "bad", "bad"),
                                          {"alpha": 4, "beta": 4}, BetaDistribution(1, 3),
                                          BetaDistribution(True, 1)])
def test_external_distribution_boundary_keeps_affinity_but_excludes_invalid_distribution(distribution):
    row = Candidate(1, "id", "CCO")
    score = PSICHICScore(6, -6, "CCO", distribution=distribution)
    report = run_workflow([row], concentration_plan(), psichic=lambda _: EngineRun({row.key: score}),
                          boltz=lambda _: EngineRun({row.key: native()}))
    assert report.rows[0].psichic.raw_affinity == 6
    assert isinstance(report.rows[0].psichic.distribution, DistributionFailure)
    assert not report.rows[0].ensemble_eligibility.eligible
    assert report.rows[0].final_rank is None and report.failed
    json.dumps(row_record(report.rows[0]), allow_nan=False)


@pytest.mark.parametrize("bad", [True, "6", None, math.inf, math.nan])
def test_external_affinity_and_output_boundaries_are_failures(bad):
    row = Candidate(1, "id", "CCO")
    report = run_workflow([row], concentration_plan(),
                          psichic=lambda _: EngineRun({row.key: PSICHICScore(bad, -6, "CCO")}),
                          boltz=lambda _: EngineRun({row.key: native()}))
    assert isinstance(report.rows[0].psichic, Failure)
    assert report.failed


def test_external_distribution_affinity_tolerance_and_original_point_are_preserved():
    assert isinstance(rich(6.000001).validated().distribution, BetaDistribution)
    assert rich(6.000001).validated().raw_affinity == 6.000001
    assert isinstance(rich(6.001).validated().distribution, DistributionFailure)


@pytest.mark.parametrize("mode,concentration,expected_calls", [
    (Mode.PYRAMID_ENSEMBLE, True, []), (Mode.PYRAMID_ENSEMBLE, False, [1, 2]),
    (Mode.REFINEMENT, False, [1, 2]), (Mode.ENSEMBLE, True, [1, 2, 3]),
])
def test_mode_specific_distribution_eligibility_and_zero_launch_shortage(mode, concentration, expected_calls):
    rows = [Candidate(i, "same", "CCO") for i in range(1, 4)]
    scores = [PSICHICScore(10, -10, "CCO"), rich(9, 3, 1), PSICHICScore(8, -8, "CCO")]
    calls = []

    def boltz(selected):
        calls.extend(row.row for row in selected)
        return EngineRun({row.key: native() for row in selected})

    plan = (concentration_plan(mode, 2 if mode.pyramid else None) if concentration
            else Plan(mode, 2, psichic_output=Output.DISTRIBUTION))
    report = run_workflow(rows, plan, psichic=lambda _: EngineRun(dict(zip([r.key for r in rows], scores))), boltz=boltz)
    assert calls == expected_calls
    assert report.failed
    if not expected_calls:
        assert "valid affinity and distribution" in report.errors[0]
        assert not report.ensemble.cohort


def test_concentration_shortlist_ties_and_failure_do_not_replenish():
    rows = [Candidate(i, "same", "CCO") for i in range(1, 5)]
    calls = []

    def boltz(selected):
        calls.extend(row.row for row in selected)
        return EngineRun({row.key: native() if row.row == 1 else Failure(row.key, "boltz", "bad") for row in selected})

    report = run_workflow(rows, concentration_plan(Mode.PYRAMID_ENSEMBLE, 2),
                          psichic=lambda _: EngineRun({r.key: rich() for r in reversed(rows)}), boltz=boltz)
    assert calls == [1, 2]
    assert [r.selection_reason for r in report.rows] == ["shortlist_tie_input_order"] * 4
    assert [r.normalized_ensemble_score for r in report.rows] == [.5, None, None, None]
    assert report.rows[0].final_rank == 1 and report.ensemble.warning
    assert report.failed


def test_output_meaning_and_legacy_positional_constructor():
    score = PSICHICScore(6, -6, "CCO", "normalized")
    assert score.normalized_smiles == "normalized"
    assert isinstance(score.distribution, DistributionNotRequested)
    row = Candidate(1, "id", "CCO")
    report = run_workflow([row], concentration_plan(), psichic=lambda _: EngineRun({row.key: rich()}),
                          boltz=lambda _: EngineRun({row.key: native()}))
    record = row_record(report.rows[0])
    assert record["normalized_ensemble_score"] == .5 and record["experimental_combined_value"] is None
    assert record["ensemble_psichic_rank"] is None and record["ensemble_boltz_rank"] is None
    assert record["ensemble_score_direction"] == "descending"
    assert record["psichic"]["distribution"]["concentration"] == 8
    assert not report.failed


def test_predictor_output_selector_keeps_legacy_default_and_passes_rich_output():
    row = Candidate(1, "id", "CCO")

    class LegacyPredictor:
        def score(self, pairs):
            return [Success(row.key, 6, -6)]

    class RichPredictor:
        def score(self, pairs, *, output):
            assert output is Output.DISTRIBUTION
            return [Success(row.key, 6, -6, BetaDistribution(4, 4))]

    assert isinstance(score_psichic([row], LegacyPredictor()).outcomes[row.key].distribution, DistributionNotRequested)
    assert score_psichic([row], RichPredictor(), output="distribution").outcomes[row.key].distribution == BetaDistribution(4, 4)
    assert isinstance(score_psichic([row], LegacyPredictor(), output="distribution").outcomes[row.key], Failure)


@pytest.mark.parametrize("override", [{"metric": Metric.PROBABILITY}, {"psichic_weight": 1.0},
                                      {"boltz_weight": 1.0}, {"psichic_weight": 0}, {"boltz_weight": 2}])
def test_concentration_rejects_explicit_incompatible_options(override):
    with pytest.raises(ValueError, match="requires native|manual fixed"):
        concentration_plan(**override)


def test_plan_omission_defaults_and_explicit_rank_compatibility():
    old = Plan(Mode.ENSEMBLE, None, Metric.PROBABILITY, 1.0, 1.0)
    assert old.ensemble is EnsembleMethod.RANK
    assert old.required_psichic_output is Output.AFFINITY
    concentration = concentration_plan(metric=Metric.AFFINITY)
    assert concentration.metric is Metric.AFFINITY
    assert concentration.psichic_weight is None and concentration.boltz_weight is None
    assert concentration.psichic_output is Output.AFFINITY
    assert concentration.required_psichic_output is Output.DISTRIBUTION


@pytest.mark.parametrize("mode", [Mode.BOLTZ, Mode.REFINEMENT])
def test_explicit_legacy_neutral_weights_preserve_nonensemble_results(mode):
    rows = [Candidate(1, "first", "CCO"), Candidate(2, "second", "CCN")]
    shortlist_size = 1 if mode.pyramid else None
    default = Plan(mode, shortlist_size)
    explicit = Plan(mode, shortlist_size, Metric.PROBABILITY, 1.0, 1.0)

    def psichic(candidates):
        return EngineRun({row.key: rich() for row in candidates})

    def boltz(candidates):
        return EngineRun({row.key: native(probability=row.row / 3) for row in candidates})

    expected = run_workflow(rows, default, psichic=psichic, boltz=boltz)
    actual = run_workflow(rows, explicit, psichic=psichic, boltz=boltz)
    assert [row_record(row) for row in actual.rows] == [row_record(row) for row in expected.rows]


@pytest.mark.parametrize("mode", [Mode.BOLTZ, Mode.REFINEMENT])
@pytest.mark.parametrize("override", [{"ensemble": EnsembleMethod.RANK}, {"ensemble": EnsembleMethod.CONCENTRATION},
                                      {"psichic_weight": 0.5}, {"boltz_weight": 2.0}])
def test_meaningless_nonensemble_options_are_rejected(mode, override):
    with pytest.raises(ValueError):
        Plan(mode, 2 if mode.pyramid else None, **override)


@pytest.mark.parametrize("override", [{"psichic_output": Output.AFFINITY}, {"psichic_output": Output.DISTRIBUTION},
                                      {"metric": "affinity_pred_value"}, {"ensemble": "concentration-v1"}])
def test_invalid_plan_boundaries(override):
    with pytest.raises(ValueError):
        Plan(Mode.BOLTZ, **override)


@pytest.mark.parametrize("result", [None, {}, "invalid"])
def test_external_output_type_failure_retains_rows_and_full_boltz_population(result):
    row = Candidate(1, "id", "CCO")
    calls = []

    def boltz(rows):
        calls.extend(rows)
        return EngineRun({row.key: native()})

    report = run_workflow([row], concentration_plan(), psichic=lambda _: EngineRun({row.key: result}), boltz=boltz)
    assert calls == [row]
    assert isinstance(report.rows[0].psichic, Failure)
    assert not report.ensemble.cohort and report.failed


def test_invalid_predictor_output_type_is_reported_as_failure():
    row = Candidate(1, "id", "CCO")

    class InvalidPredictor:
        def score(self, pairs, *, output):
            return [object()]

    run = score_psichic([row], InvalidPredictor(), output=Output.DISTRIBUTION)
    assert isinstance(run.outcomes[row.key], Failure)
    assert "invalid output type" in run.error
