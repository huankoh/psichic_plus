from dataclasses import dataclass

@dataclass(frozen=True)
class Success:
    token: str
    raw_affinity: float
    adaptive_score: float


@dataclass(frozen=True)
class Failure:
    token: str
    stage: str
    reason: str
