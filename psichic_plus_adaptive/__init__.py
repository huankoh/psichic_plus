"""External PSICHIC+ adapter for the pinned AdaptiveFlow-VS workflow."""

from dataclasses import dataclass

UPSTREAM_REVISION = "df74e005e456e94654fa93a40d748de399450b81"


@dataclass(frozen=True)
class Molecule:
    token: str
    collection_id: str
    molecule_id: str
    tranche: str
    smiles: str


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
