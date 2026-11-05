"""Pure selection and combination rules shared with future finalizers."""

import math

from .models import BoltzScore, EnsembleMethod, EnsembleResult, finite_number, psichic_eligibility


def average_ranks(values, *, higher_is_better):
    if any(not finite_number(value) for value in values.values()):
        raise ValueError("Cannot rank invalid or non-finite scores")
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


def shortlist_order(candidates, outcomes, *, require_distribution=False):
    eligible = [row for row in candidates if psichic_eligibility(
        outcomes[row.key], require_distribution=require_distribution,
    ).eligible]
    return sorted(eligible, key=lambda row: (-outcomes[row.key].raw_affinity, row.row))


def _minmax(values):
    lo, hi = min(values), max(values)
    if hi == lo:
        return [0.5] * len(values)
    span = hi - lo
    if not math.isfinite(span):
        scale = max(abs(lo), abs(hi))
        return _minmax([value / scale for value in values])
    return [(value - lo) / span for value in values]


def concentration_scores(psichic_affinities, concentrations, boltz_affinities):
    """Combine one common cohort using native, lower-better Boltz affinity."""
    if not len(psichic_affinities) == len(concentrations) == len(boltz_affinities):
        raise ValueError("Prediction and concentration series must be the same length")
    if any(not finite_number(value) for values in (psichic_affinities, concentrations, boltz_affinities)
           for value in values):
        raise ValueError("Ensemble inputs must be finite numbers")
    if any(value <= 0 for value in concentrations):
        raise ValueError("Concentration must be positive")
    if not psichic_affinities:
        return []
    scaled_p = _minmax([float(value) for value in psichic_affinities])
    scaled_b = _minmax([-float(value) for value in boltz_affinities])
    scaled_c = _minmax([math.log(value) for value in concentrations])
    result = []
    for p, b, s in zip(scaled_p, scaled_b, scaled_c):
        weight = (0.5 + 0.5 * s) / 2.0
        result.append(weight * p + (1.0 - weight) * b)
    return result


def combine_ensemble(psichic, boltz, plan):
    """Combine validated outcomes without invoking either engine."""
    method = plan.ensemble
    if not isinstance(method, EnsembleMethod):
        raise ValueError("Combination requires an ensemble plan")
    concentration = method is EnsembleMethod.CONCENTRATION
    cohort = [key for key, outcome in psichic.items() if isinstance(boltz.get(key), BoltzScore) and psichic_eligibility(
        outcome, require_distribution=concentration,
    ).eligible]
    p_values = {key: psichic[key].raw_affinity for key in cohort}
    b_values = {key: getattr(boltz[key], plan.metric.value) for key in cohort}
    result = EnsembleResult(method, cohort, {}, {})
    if concentration:
        concentrations = [psichic[key].distribution.concentration for key in cohort]
        result.scores = dict(zip(cohort, concentration_scores(list(p_values.values()), concentrations,
                                                            list(b_values.values()))))
        for name, values in (("psichic_affinity", list(p_values.values())),
                             ("boltz_affinity", list(b_values.values())),
                             ("log_concentration", [math.log(v) for v in concentrations])):
            if values and min(values) == max(values):
                result.constant_series.append(name)
    else:
        result.psichic_ranks = average_ranks(p_values, higher_is_better=True)
        result.boltz_ranks = average_ranks(b_values, higher_is_better=plan.metric.higher_is_better)
        total = plan.psichic_weight + plan.boltz_weight
        a_weight, b_weight = plan.psichic_weight / total, plan.boltz_weight / total
        result.scores = {key: a_weight * result.psichic_ranks[key] + b_weight * result.boltz_ranks[key]
                         for key in cohort}
    result.ranks = average_ranks(result.scores, higher_is_better=concentration)
    if len(cohort) == 1:
        result.discrimination = "singleton_lacks_discrimination"
    elif cohort and len(set(result.scores.values())) == 1:
        result.discrimination = "all_tied_lacks_discrimination"
    elif cohort:
        result.discrimination = "multiple_distinct_scores"
    if "lacks_discrimination" in result.discrimination:
        result.warning = "No discrimination in this cohort; computed scores and average ranks are retained."
    return result
