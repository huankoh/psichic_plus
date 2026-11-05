from dataclasses import asdict
import math

from applications.scorer import Failure, Success
from .models import BoltzScore, EngineRun, Mode, NotRun, PSICHICScore, Report, RowResult


def average_ranks(values, *, higher_is_better):
    if any(not math.isfinite(value) for value in values.values()):
        raise ValueError("Cannot rank non-finite scores")
    ordered = sorted(values, key=values.get, reverse=higher_is_better)
    ranks = {}
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[start]]:
            end += 1
        for key in ordered[start:end]:
            ranks[key] = (start + 1 + end) / 2
        start = end
    return ranks


def score_psichic(candidates, predictor):
    inputs = {row.key: row for row in candidates}
    outcomes = {}
    try:
        results = predictor.score([(row.key, row.smiles) for row in candidates])
    except Exception as error:
        return EngineRun({key: Failure(key, "psichic_prediction", str(error)) for key in inputs}, error=str(error))
    for result in results:
        if result.token not in inputs or result.token in outcomes:
            reason = f"PSICHIC returned an unexpected or duplicate row key: {result.token!r}"
            return EngineRun({key: Failure(key, "psichic_mapping", reason) for key in inputs}, error=reason)
        if isinstance(result, Success) and all(math.isfinite(v) for v in (result.raw_affinity, result.adaptive_score)):
            outcomes[result.token] = PSICHICScore(result.raw_affinity, result.adaptive_score, inputs[result.token].smiles)
        elif isinstance(result, Failure):
            outcomes[result.token] = result
        else:
            outcomes[result.token] = Failure(result.token, "psichic_prediction", "Invalid or non-finite score")
    for key in inputs.keys() - outcomes.keys():
        outcomes[key] = Failure(key, "psichic_mapping", "Missing prediction")
    return EngineRun(outcomes)


def _score(candidates, scorer, engine, report):
    try:
        run = scorer(candidates)
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
        elif isinstance(outcome, result_type):
            values = (outcome.raw_affinity, outcome.adaptive_score) if engine == "psichic" else (
                outcome.affinity_pred_value, outcome.affinity_probability_binary,
            )
            if any(type(v) not in (int, float) or not math.isfinite(v) for v in values) or (
                engine == "boltz" and not 0 <= outcome.affinity_probability_binary <= 1
            ):
                run.outcomes[key] = Failure(key, f"{engine}_output", "Invalid numeric score")
    return run.outcomes


def run_workflow(candidates, plan, *, boltz, psichic=None):
    if len({row.key for row in candidates}) != len(candidates):
        raise ValueError("Input row numbers must be unique")
    if plan.mode is not Mode.BOLTZ and psichic is None:
        raise ValueError("This workflow requires a PSICHIC scorer")
    report = Report([RowResult(row) for row in candidates])
    by_key = {row.candidate.key: row for row in report.rows}
    if plan.mode is not Mode.BOLTZ:
        for key, outcome in _score(candidates, psichic, "psichic", report).items():
            by_key[key].psichic = outcome
        scores = {key: row.psichic.raw_affinity for key, row in by_key.items() if isinstance(row.psichic, PSICHICScore)}
        for key, rank in average_ranks(scores, higher_is_better=True).items():
            by_key[key].psichic_screen_rank = rank
    selected = candidates
    if plan.mode.pyramid:
        for row in report.rows:
            row.selection_reason = "psichic_failed" if isinstance(row.psichic, Failure) else "below_shortlist"
        ordered = sorted(scores, key=lambda key: (-scores[key], by_key[key].candidate.row))
        if len(ordered) < plan.shortlist_size:
            report.errors.append(f"Shortlist requires {plan.shortlist_size} successful PSICHIC rows; got {len(ordered)}")
            for row in report.rows:
                row.boltz = NotRun("insufficient_psichic_successes")
                if not isinstance(row.psichic, Failure):
                    row.selection_reason = "insufficient_psichic_successes"
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
    if plan.mode.ensemble:
        cohort = {key for key in b_values if isinstance(by_key[key].psichic, PSICHICScore)}
        a_ranks = average_ranks({key: by_key[key].psichic.raw_affinity for key in cohort}, higher_is_better=True)
        b_ranks = average_ranks({key: b_values[key] for key in cohort}, higher_is_better=plan.metric.higher_is_better)
        combined = {}
        total_weight = plan.psichic_weight + plan.boltz_weight
        a_weight, b_weight = plan.psichic_weight / total_weight, plan.boltz_weight / total_weight
        for key in cohort:
            row = by_key[key]
            row.ensemble_psichic_rank, row.ensemble_boltz_rank = a_ranks[key], b_ranks[key]
            combined[key] = a_weight * a_ranks[key] + b_weight * b_ranks[key]
            row.experimental_combined_value = combined[key]
        for key, rank in average_ranks(combined, higher_is_better=False).items():
            by_key[key].final_rank = rank
    return report


def row_record(row):
    record = asdict(row)
    for engine in ("psichic", "boltz"):
        outcome = getattr(row, engine)
        record[engine]["status"] = "failure" if isinstance(outcome, Failure) else (
            "not_run" if isinstance(outcome, NotRun) else "success"
        )
    record["candidate"]["key"] = row.candidate.key
    if row.final_rank is None:
        record["unranked_reason"] = "engine_failure" if any(
            isinstance(value, Failure) for value in (row.psichic, row.boltz)
        ) else row.selection_reason
    else:
        record["unranked_reason"] = None
    return record
