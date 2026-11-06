"""Candidate and pair tables. Stdlib only.

Mirrors Chaperone's PipelineDataHandler role with a small-molecule schema.
Rows are plain dicts. Core fields use fixed names. Block outputs use
namespaced keys such as ``psichic.affinity``.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

CORE_FIELDS = ("row_key", "candidate_id", "smiles_submitted", "source")


def row_key(ordinal: int) -> str:
    if ordinal < 1:
        raise ValueError("ordinal must be positive")
    return f"row_{ordinal:012d}"


@dataclass(frozen=True)
class Target:
    """Resolved target artifact. Featurized state is cached elsewhere."""

    target_id: str
    sequence: str
    assets: tuple[tuple[str, str], ...] = ()

    @property
    def digest(self) -> str:
        payload = {"id": self.target_id, "seq": self.sequence, "assets": self.assets}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    @classmethod
    def from_fasta(cls, path: str | Path, target_id: str | None = None) -> "Target":
        path = Path(path)
        lines = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
        headers = [ln for ln in lines if ln.startswith(">")]
        if len(headers) != 1:
            raise ValueError(f"Expected exactly one FASTA record in {path}")
        sequence = "".join(ln for ln in lines if not ln.startswith(">")).upper()
        if not sequence.isalpha():
            raise ValueError("FASTA sequence must contain amino-acid letters")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return cls(
            target_id or headers[0][1:].split()[0], sequence, (("fasta", digest),)
        )


@dataclass
class CandidateTable:
    """Ordered candidate rows. Blocks append namespaced columns."""

    rows: list[dict] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self) -> Iterator[dict]:
        return iter(self.rows)

    @classmethod
    def from_records(cls, records: Iterable[dict]) -> "CandidateTable":
        rows = []
        for i, rec in enumerate(records, 1):
            row = {
                "row_key": rec.get("row_key", row_key(i)),
                "candidate_id": str(rec.get("candidate_id", rec.get("ID", i))),
                "smiles_submitted": str(
                    rec.get("smiles_submitted", rec.get("SMILES", ""))
                ),
                "source": str(rec.get("source", "records")),
            }
            for key, value in rec.items():
                row.setdefault(key, value)
            rows.append(row)
        return cls(rows)

    @classmethod
    def from_csv(
        cls, path: str | Path, id_column: str = "ID", smiles_column: str = "SMILES"
    ) -> "CandidateTable":
        path = Path(path)
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            names = reader.fieldnames or []
            if id_column not in names or smiles_column not in names:
                raise ValueError(
                    f"CSV needs {id_column!r} and {smiles_column!r} columns"
                )
            records = [
                {
                    "candidate_id": row[id_column],
                    "smiles_submitted": row[smiles_column],
                    "source": f"{path.name}#{i}",
                }
                for i, row in enumerate(reader, 1)
            ]
        if not records:
            raise ValueError("Candidate CSV is empty")
        return cls.from_records(records)

    def columns(self) -> list[str]:
        seen: list[str] = []
        for row in self.rows:
            for key in row:
                if key not in seen:
                    seen.append(key)
        return seen

    def pending(self, instance: str, only_selected: str | None = None) -> list[dict]:
        """Rows still needing ``instance``, optionally gated by a selection."""
        status = f"{instance}.status"
        out = []
        for row in self.rows:
            if row.get(status) in ("success", "failure"):
                continue
            if (
                only_selected is not None
                and row.get(f"{only_selected}.selected") is not True
            ):
                continue
            out.append(row)
        return out

    def copy(self) -> "CandidateTable":
        return CandidateTable([dict(r) for r in self.rows])

    def write_csv(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        cols = self.columns()
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=cols, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self.rows)

    def write_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.rows, indent=1, allow_nan=False) + "\n")


def expand_pairs(
    table: CandidateTable,
    targets: list[Target],
    pairs: Iterable[tuple[str, str]] | None = None,
) -> CandidateTable:
    """Build the pair table. Default is the full molecule by target grid.

    ``pairs`` optionally gives explicit (candidate_id, target_id) tuples for
    sparse panels. Storage stays in long pair form. Wide views are derived.
    """
    by_id = {r["candidate_id"]: r for r in table.rows}
    wanted = (
        list(pairs)
        if pairs is not None
        else [(r["candidate_id"], t.target_id) for r in table.rows for t in targets]
    )
    target_ids = {t.target_id for t in targets}
    rows = []
    for i, (cid, tid) in enumerate(wanted, 1):
        if cid not in by_id or tid not in target_ids:
            raise ValueError(f"Unknown pair ({cid!r}, {tid!r})")
        base = by_id[cid]
        rows.append(
            {
                **base,
                "row_key": row_key(i),
                "pair_key": f"pair_{i:012d}",
                "molecule_key": base["row_key"],
                "target_id": tid,
            }
        )
    return CandidateTable(rows)
