from dataclasses import asdict

from applications.scorer import Failure, Output, Success
from .ensemble import average_ranks, combine_ensemble, shortlist_order
from .models import (
    BoltzScore, Eligibility, EngineRun, EnsembleMethod, Mode, NotRun,
    PSICHICScore, Report, RowResult, finite_number, psichic_eligibility,
)


def score_psichic(candidates, predictor, *, output=Output.AFFINITY):
    output = Output(output)
    inputs = {row.key: row for row in candidates}
    outcomes = {}
    try:
        pairs = [(row.key, row.smiles) for row in candidates]
        results = predictor.score(pairs, output=output) if output is Output.DISTRIBUTION else predictor.score(pairs)
        for result in results:
            if not isinstance(result, (Success, Failure)):
                raise ValueError("PSICHIC returned an invalid output type")
            if result.token not in inputs or result.token in outcomes:
                reason = f"PSICHIC returned an unexpected or duplicate row key: {result.token!r}"
                return EngineRun({key: Failure(key, "psichic_mapping", reason) for key in inputs}, error=reason)
            if isinstance(result, Success):
                try:
                    outcomes[result.token] = PSICHICScore(
                        result.raw_affinity, result.adaptive_score, inputs[result.token].smiles,
                        distribution=result.distribution,
                    ).validated(require_distribution=output is Output.DISTRIBUTION)
                except ValueError as error:
                    outcomes[result.token] = Failure(result.token, "psichic_output", str(error))
            else:
                outcomes[result.token] = result
    except Exception as error:
        return EngineRun({key: Failure(key, "psichic_prediction", str(error)) for key in inputs}, error=str(error))
    for key in inputs.keys() - outcomes.keys():
        outcomes[key] = Failure(key, "psichic_mapping", "Missing prediction")
    return EngineRun(outcomes)


def _score(candidates, scorer, engine, report):
    try:
        run = scorer(candidates)
        if not isinstance(run, EngineRun) or not isinstance(run.outcomes, dict):
            raise ValueError("Scorer must return EngineRun with keyed outcomes")
    except Exception as error:
        run = EngineRun({}, error=f"{engine}: {error}")
    report.engines[engine] = run.metadata
    if run.error:
        report.errors.append(run.error)
    expected = {row.key for row in candidates}
    if run.outcomes.keys() - expected:
        reason = f"{engine} returned unexpected row keys"
        report.errors.append(reason)
        return {key: Failure(key, f"{engine}_mapping", reason) for key in expected}
    result_type = PSICHICScore if engine == "psichic" else BoltzScore
    for key in expected:
        outcome = run.outcomes.get(key)
        if not isinstance(outcome, (result_type, Failure)):
            run.outcomes[key] = Failure(key, f"{engine}_mapping", "Missing or invalid result")
        elif isinstance(outcome, PSICHICScore):
            try:
                run.outcomes[key] = outcome.validated(require_distribution=report.distribution_requested)
            except ValueError as error:
                run.outcomes[key] = Failure(key, "psichic_output", str(error))
        elif isinstance(outcome, BoltzScore):
            values = (outcome.affinity_pred_value, outcome.affinity_probability_binary)
            if any(not finite_number(v) for v in values) or not 0 <= outcome.affinity_probability_binary <= 1:
                run.outcomes[key] = Failure(key, "boltz_output", "Invalid numeric score")
    return run.outcomes


def _combine(report, plan):
    if not plan.mode.ensemble:
        return
    for row in report.rows:
        row.ensemble_method = plan.ensemble
        eligibility = psichic_eligibility(row.psichic, require_distribution=plan.ensemble is EnsembleMethod.CONCENTRATION)
        if eligibility.eligible and row.selection != "selected":
            eligibility = Eligibility(False, row.boltz.reason if isinstance(row.boltz, NotRun) else "not_selected")
        if eligibility.eligible and not isinstance(row.boltz, BoltzScore):
            eligibility = Eligibility(False, "boltz_failed" if isinstance(row.boltz, Failure) else "boltz_not_run")
        row.ensemble_eligibility = eligibility
    cohort = [row for row in report.rows if row.ensemble_eligibility.eligible]
    report.ensemble = combine_ensemble({row.candidate.key: row.psichic for row in cohort},
                                       {row.candidate.key: row.boltz for row in cohort}, plan)
    for row in cohort:
        key = row.candidate.key
        row.final_rank = report.ensemble.ranks[key]
        if plan.ensemble is EnsembleMethod.CONCENTRATION:
            row.normalized_ensemble_score = report.ensemble.scores[key]
        else:
            row.experimental_combined_value = report.ensemble.scores[key]
            row.ensemble_psichic_rank = report.ensemble.psichic_ranks[key]
            row.ensemble_boltz_rank = report.ensemble.boltz_ranks[key]


def run_workflow(candidates, plan, *, boltz, psichic=None):
    if len({row.key for row in candidates}) != len(candidates):
        raise ValueError("Input row numbers must be unique")
    if plan.mode is not Mode.BOLTZ and psichic is None:
        raise ValueError("This workflow requires a PSICHIC scorer")
    report = Report([RowResult(row) for row in candidates],
                    distribution_requested=plan.required_psichic_output is Output.DISTRIBUTION)
    by_key = {row.candidate.key: row for row in report.rows}
    if plan.mode is not Mode.BOLTZ:
        for key, outcome in _score(candidates, psichic, "psichic", report).items():
            by_key[key].psichic = outcome
        scores = {key: row.psichic.raw_affinity for key, row in by_key.items() if isinstance(row.psichic, PSICHICScore)}
        for key, rank in average_ranks(scores, higher_is_better=True).items():
            by_key[key].psichic_screen_rank = rank
    selected = candidates
    if plan.mode.pyramid:
        require_distribution = plan.ensemble is EnsembleMethod.CONCENTRATION
        for row in report.rows:
            eligibility = psichic_eligibility(row.psichic, require_distribution=require_distribution)
            row.selection_reason = "below_shortlist" if eligibility.eligible else eligibility.reason
        ordered = [row.key for row in shortlist_order(candidates, {key: row.psichic for key, row in by_key.items()},
                                                     require_distribution=require_distribution)]
        if len(ordered) < plan.shortlist_size:
            requirement = "valid affinity and distribution" if require_distribution else "successful PSICHIC"
            report.errors.append(f"Shortlist requires {plan.shortlist_size} {requirement} rows; got {len(ordered)}")
            for row in report.rows:
                row.boltz = NotRun("insufficient_psichic_successes")
                if row.selection_reason == "below_shortlist":
                    row.selection_reason = "insufficient_psichic_successes"
            _combine(report, plan)
            return report
        selected_keys = ordered[:plan.shortlist_size]
        cutoff = scores[selected_keys[-1]]
        split_tie = len(ordered) > plan.shortlist_size and scores[ordered[plan.shortlist_size]] == cutoff
        for key in ordered:
            row = by_key[key]
            if key in selected_keys:
                row.selection = "selected"
                row.selection_reason = "shortlist_tie_input_order" if split_tie and scores[key] == cutoff else "psichic_top_k"
            elif split_tie and scores[key] == cutoff:
                row.selection_reason = "shortlist_tie_input_order"
        selected = [by_key[key].candidate for key in selected_keys]
    else:
        for row in report.rows:
            row.selection, row.selection_reason = "selected", "all_inputs"
    if selected:
        for key, outcome in _score(selected, boltz, "boltz", report).items():
            by_key[key].boltz = outcome
    b_values = {key: getattr(row.boltz, plan.metric.value) for key, row in by_key.items() if isinstance(row.boltz, BoltzScore)}
    for key, rank in average_ranks(b_values, higher_is_better=plan.metric.higher_is_better).items():
        by_key[key].boltz_rank = rank
        if not plan.mode.ensemble:
            by_key[key].final_rank = rank
    _combine(report, plan)
    return report


def row_record(row):
    record = asdict(row)
    for engine in ("psichic", "boltz"):
        outcome = getattr(row, engine)
        record[engine]["status"] = "failure" if isinstance(outcome, Failure) else (
            "not_run" if isinstance(outcome, NotRun) else "success"
        )
    record["candidate"]["key"] = row.candidate.key
    if row.ensemble_method:
        record["ensemble_reference_commit"] = row.ensemble_method.reference_commit
        record["ensemble_reference_source"] = row.ensemble_method.reference_source
        record["ensemble_version"] = 1
        record["ensemble_score_direction"] = row.ensemble_method.direction
    if row.final_rank is None:
        reason = row.ensemble_eligibility.reason if row.ensemble_method else row.selection_reason
        if reason == "psichic_distribution_failed":
            record["unranked_reason"] = reason
        else:
            record["unranked_reason"] = "engine_failure" if any(
                isinstance(value, Failure) for value in (row.psichic, row.boltz)
            ) else reason
    else:
        record["unranked_reason"] = None
    return record
