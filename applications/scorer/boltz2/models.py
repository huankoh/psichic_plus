from dataclasses import dataclass, field, replace
from enum import Enum
import math

from applications.scorer import (
    BetaDistribution, DistributionFailure, DistributionNotRequested,
    DistributionOutcome, Failure, Output,
)


class Mode(str, Enum):
    BOLTZ = "boltz-only"
    ENSEMBLE = "full-ensemble"
    PYRAMID_ENSEMBLE = "pyramid-ensemble"
    REFINEMENT = "pyramid-refinement"

    @property
    def pyramid(self):
        return self in (self.PYRAMID_ENSEMBLE, self.REFINEMENT)

    @property
    def ensemble(self):
        return self in (self.ENSEMBLE, self.PYRAMID_ENSEMBLE)


class Metric(str, Enum):
    PROBABILITY = "affinity_probability_binary"
    AFFINITY = "affinity_pred_value"

    @property
    def higher_is_better(self):
        return self is self.PROBABILITY


class EnsembleMethod(str, Enum):
    RANK = "rank-v1"
    CONCENTRATION = "concentration-v1"

    @property
    def reference_commit(self):
        return ("00a44d9a99cac5dc35913b0f9e40ed9d82169896" if self is self.CONCENTRATION
                else "f414ee6223b29ea182ba1b0ae44c2c5b236a2094")

    @property
    def reference_source(self):
        return ("huankoh/psichic-plus/codebase/reproducibility/ensemble.py" if self is self.CONCENTRATION
                else "psichic_plus/applications/scorer/boltz2/workflow.py")

    @property
    def direction(self):
        return "descending" if self is self.CONCENTRATION else "ascending"


class _Omitted(Enum):
    VALUE = "omitted"


OMITTED = _Omitted.VALUE


def finite_number(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


@dataclass(frozen=True)
class Plan:
    mode: Mode
    shortlist_size: int | None = None
    metric: Metric | _Omitted = OMITTED
    psichic_weight: float | None | _Omitted = OMITTED
    boltz_weight: float | None | _Omitted = OMITTED
    ensemble: EnsembleMethod | None | _Omitted = OMITTED
    psichic_output: Output | _Omitted = OMITTED

    def __post_init__(self):
        if not isinstance(self.mode, Mode):
            raise ValueError("Use a supported workflow mode")
        if self.mode.pyramid:
            if type(self.shortlist_size) is not int or self.shortlist_size < 1:
                raise ValueError("Pyramid modes require a positive integer shortlist size")
        elif self.shortlist_size is not None:
            raise ValueError("Shortlist size applies only to pyramid modes")
        method = self.ensemble
        if method is not OMITTED and not isinstance(method, EnsembleMethod):
            raise ValueError("Use a supported ensemble method")
        if self.mode.ensemble:
            method = EnsembleMethod.RANK if method is OMITTED else method
        else:
            if method is not OMITTED:
                raise ValueError("Ensemble method applies only to ensemble modes")
            method = None
        concentration = method is EnsembleMethod.CONCENTRATION
        if self.metric is not OMITTED and not isinstance(self.metric, Metric):
            raise ValueError("Use a supported Boltz metric")
        if concentration and self.metric not in (OMITTED, Metric.AFFINITY):
            raise ValueError("concentration-v1 requires native affinity_pred_value, not binder probability")
        weights = (self.psichic_weight, self.boltz_weight)
        if any(weight is not OMITTED for weight in weights):
            if not self.mode.ensemble:
                raise ValueError("Weights apply only to ensemble modes")
            if concentration:
                raise ValueError("concentration-v1 does not accept manual fixed weights")
        weights = tuple(1.0 if weight is OMITTED else weight for weight in weights)
        if any(not finite_number(w) or w < 0 for w in weights) or not 0 < sum(weights) < math.inf:
            raise ValueError("Weights must be finite, nonnegative, and have a positive finite sum")
        if self.mode is Mode.BOLTZ and self.psichic_output is not OMITTED:
            raise ValueError("PSICHIC output applies only to modes that use PSICHIC")
        output = Output.AFFINITY if self.psichic_output is OMITTED else Output(self.psichic_output)
        object.__setattr__(self, "ensemble", method)
        object.__setattr__(self, "metric", (Metric.AFFINITY if concentration else Metric.PROBABILITY)
                           if self.metric is OMITTED else self.metric)
        object.__setattr__(self, "psichic_weight", None if concentration else weights[0])
        object.__setattr__(self, "boltz_weight", None if concentration else weights[1])
        object.__setattr__(self, "psichic_output", output)

    @property
    def required_psichic_output(self):
        return Output.DISTRIBUTION if self.ensemble is EnsembleMethod.CONCENTRATION else self.psichic_output


@dataclass(frozen=True)
class Candidate:
    row: int
    id: str
    smiles: str

    @property
    def key(self):
        return f"row_{self.row:08d}"


@dataclass(frozen=True)
class NotRun:
    reason: str


@dataclass(frozen=True)
class PSICHICScore:
    raw_affinity: float
    adaptive_score: float
    submitted_smiles: str
    normalized_smiles: str | None = None
    distribution: DistributionOutcome = DistributionNotRequested()

    def validated(self, *, require_distribution=False):
        if not all(finite_number(v) for v in (self.raw_affinity, self.adaptive_score)):
            raise ValueError("Invalid numeric PSICHIC score")
        distribution = self.distribution
        reason = None
        if isinstance(distribution, BetaDistribution):
            try:
                if not all(finite_number(v) and v > 0 for v in (distribution.alpha, distribution.beta)):
                    raise ValueError("Alpha and beta must be positive finite numbers")
                checked = BetaDistribution(distribution.alpha, distribution.beta)
                if not finite_number(distribution.concentration) or checked.concentration != distribution.concentration:
                    raise ValueError("Concentration must equal alpha plus beta")
                expected = 12.0 * (checked.alpha / checked.concentration)
                if not math.isclose(self.raw_affinity, expected, rel_tol=1e-6, abs_tol=1e-6):
                    raise ValueError("Affinity is inconsistent with 12 * alpha / (alpha + beta)")
            except ValueError as error:
                reason = str(error)
        elif isinstance(distribution, DistributionNotRequested):
            if require_distribution:
                reason = "Requested distribution is missing"
        elif not isinstance(distribution, DistributionFailure):
            reason = "Invalid distribution outcome type"
        if reason:
            distribution = DistributionFailure(reason, repr(getattr(distribution, "alpha", None)),
                                               repr(getattr(distribution, "beta", None)))
        return replace(self, distribution=distribution)


@dataclass(frozen=True)
class Eligibility:
    eligible: bool
    reason: str | None = None


def psichic_eligibility(outcome, *, require_distribution=False):
    if not isinstance(outcome, PSICHICScore):
        return Eligibility(False, "psichic_failed" if isinstance(outcome, Failure) else "psichic_not_run")
    if require_distribution and not isinstance(outcome.distribution, BetaDistribution):
        return Eligibility(False, "psichic_distribution_failed")
    return Eligibility(True)


@dataclass(frozen=True)
class BoltzScore:
    affinity_probability_binary: float
    affinity_pred_value: float
    native: dict
    output_directory: str
    submitted_smiles: str


@dataclass
class EngineRun:
    outcomes: dict[str, PSICHICScore | BoltzScore | Failure]
    metadata: dict = field(default_factory=dict)
    error: str | None = None


@dataclass
class EnsembleResult:
    method: EnsembleMethod
    cohort: list[str]
    scores: dict[str, float]
    ranks: dict[str, float]
    psichic_ranks: dict[str, float] = field(default_factory=dict)
    boltz_ranks: dict[str, float] = field(default_factory=dict)
    constant_series: list[str] = field(default_factory=list)
    discrimination: str = "no_scored_entries"
    warning: str | None = None

    def metadata(self):
        return {"method": self.method.value, "version": 1, "reference_commit": self.method.reference_commit,
                "reference_source": self.method.reference_source, "direction": self.method.direction,
                "score_field": "normalized_ensemble_score" if self.method is EnsembleMethod.CONCENTRATION
                else "experimental_combined_value", "cohort": self.cohort, "constant_series": self.constant_series,
                "discrimination": self.discrimination, "warning": self.warning}


@dataclass
class RowResult:
    candidate: Candidate
    psichic: PSICHICScore | Failure | NotRun = field(default_factory=lambda: NotRun("mode"))
    boltz: BoltzScore | Failure | NotRun = field(default_factory=lambda: NotRun("not_selected"))
    selection: str = "not_selected"
    selection_reason: str = ""
    psichic_screen_rank: float | None = None
    boltz_rank: float | None = None
    ensemble_psichic_rank: float | None = None
    ensemble_boltz_rank: float | None = None
    experimental_combined_value: float | None = None
    final_rank: float | None = None
    normalized_ensemble_score: float | None = None
    ensemble_method: EnsembleMethod | None = None
    ensemble_eligibility: Eligibility = field(default_factory=lambda: Eligibility(False, "not_an_ensemble"))


@dataclass
class Report:
    rows: list[RowResult]
    engines: dict = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    ensemble: EnsembleResult | None = None
    distribution_requested: bool = False

    @property
    def failed(self):
        return bool(self.errors) or any(
            isinstance(outcome, Failure) for row in self.rows for outcome in (row.psichic, row.boltz)
        ) or (self.distribution_requested and any(
            isinstance(row.psichic, PSICHICScore) and isinstance(row.psichic.distribution, DistributionFailure)
            for row in self.rows
        ))
