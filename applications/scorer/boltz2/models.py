from dataclasses import dataclass, field
from enum import Enum
import math

from applications.scorer import Failure


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


@dataclass(frozen=True)
class Plan:
    mode: Mode
    shortlist_size: int | None = None
    metric: Metric = Metric.PROBABILITY
    psichic_weight: float = 1.0
    boltz_weight: float = 1.0

    def __post_init__(self):
        if not isinstance(self.mode, Mode) or not isinstance(self.metric, Metric):
            raise ValueError("Use a supported workflow mode and Boltz metric")
        if self.mode.pyramid:
            if type(self.shortlist_size) is not int or self.shortlist_size < 1:
                raise ValueError("Pyramid modes require a positive integer shortlist size")
        elif self.shortlist_size is not None:
            raise ValueError("Shortlist size applies only to pyramid modes")
        weights = (self.psichic_weight, self.boltz_weight)
        if any(not math.isfinite(w) or w < 0 for w in weights) or not 0 < sum(weights) < math.inf:
            raise ValueError("Weights must be finite, nonnegative, and have a positive finite sum")
        if not self.mode.ensemble and weights != (1.0, 1.0):
            raise ValueError("Weights apply only to ensemble modes")


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


@dataclass
class Report:
    rows: list[RowResult]
    engines: dict = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    @property
    def failed(self):
        return bool(self.errors) or any(
            isinstance(outcome, Failure) for row in self.rows for outcome in (row.psichic, row.boltz)
        )
