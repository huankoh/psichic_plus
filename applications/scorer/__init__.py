import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal


class Output(str, Enum):
    AFFINITY = "affinity"
    DISTRIBUTION = "distribution"


@dataclass(frozen=True)
class DistributionNotRequested:
    state: Literal["not-requested"] = field(default="not-requested", init=False)


@dataclass(frozen=True)
class BetaDistribution:
    alpha: float
    beta: float
    concentration: float = field(init=False)
    state: Literal["valid"] = field(default="valid", init=False)

    def __post_init__(self):
        if not all(math.isfinite(value) and value > 0 for value in (self.alpha, self.beta)):
            raise ValueError("Alpha and beta must be positive and finite")
        concentration = self.alpha + self.beta
        if not math.isfinite(concentration):
            raise ValueError("Alpha plus beta must be finite")
        object.__setattr__(self, "concentration", concentration)


@dataclass(frozen=True)
class DistributionFailure:
    reason: str
    raw_alpha: str
    raw_beta: str
    state: Literal["distribution-failure"] = field(default="distribution-failure", init=False)


DistributionOutcome = DistributionNotRequested | BetaDistribution | DistributionFailure


@dataclass(frozen=True)
class Success:
    token: str
    raw_affinity: float
    adaptive_score: float
    distribution: DistributionOutcome = DistributionNotRequested()


@dataclass(frozen=True)
class Failure:
    token: str
    stage: str
    reason: str
