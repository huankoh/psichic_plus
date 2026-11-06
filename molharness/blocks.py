"""Builtin skeleton blocks. Real model blocks land in later phases."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

from .block_config import BlockFunction, JobArgs
from .data import CandidateTable
from .registry import register


@register
@dataclass
class CsvSource(BlockFunction):
    _name = "csv_source"
    _role = "source"
    _requires_input = False
    _description = "Read candidates from a CSV file."

    path: str = ""
    id_column: str = "ID"
    smiles_column: str = "SMILES"
    name: str = ""

    def execute(self, table, ctx):
        return CandidateTable.from_csv(self.path, self.id_column, self.smiles_column)


@register
@dataclass
class FakeScorer(BlockFunction):
    """Deterministic stand-in scorer for contract tests. No model."""

    _name = "fake"
    _role = "scorer"
    _chunked = True
    _resources = JobArgs(rows_per_job=512)
    _produces = ("affinity",)
    _description = "Hash-based deterministic fake affinity in [0, 12)."

    scale: float = 12.0
    only_selected: str | None = None
    name: str = ""

    def score_rows(self, rows, ctx):
        for row in rows:
            smiles = row.get("smiles_submitted", "")
            if not smiles.strip() or smiles.strip().upper() == "INVALID":
                row[f"{ctx.instance}.status"] = "failure"
                row[f"{ctx.instance}.stage"] = "ligand"
                row[f"{ctx.instance}.reason"] = "empty or invalid SMILES"
                continue
            digest = hashlib.sha256(smiles.encode()).digest()
            value = (int.from_bytes(digest[:4]) / 2**32) * self.scale
            row[f"{ctx.instance}.affinity"] = round(value, 6)
            row[f"{ctx.instance}.status"] = "success"


@register
@dataclass
class TopKSelector(BlockFunction):
    _name = "topk"
    _role = "selector"
    _description = "Mark the best K rows by one metric column."

    by: str = ""
    k: int = 0
    higher_is_better: bool = True
    name: str = "shortlist"

    def execute(self, table, ctx):
        eligible = [
            r
            for r in table.rows
            if isinstance(r.get(self.by), (int, float)) and math.isfinite(r[self.by])
        ]
        ordered = sorted(
            eligible, key=lambda r: r[self.by], reverse=self.higher_is_better
        )
        chosen = {r["row_key"] for r in ordered[: self.k]}
        shortage = len(eligible) < self.k
        for row in table.rows:
            row[f"{ctx.instance}.selected"] = row["row_key"] in chosen and not shortage
            row[f"{ctx.instance}.reason"] = (
                "insufficient_successes"
                if shortage
                else "top_k"
                if row["row_key"] in chosen
                else "below_shortlist"
            )
        return table


@register
@dataclass
class SingleMetricRank(BlockFunction):
    _name = "rank_by"
    _role = "combiner"
    _description = "Average-tie rank by one metric. Larger is better by default."

    by: str = ""
    higher_is_better: bool = True
    name: str = ""

    def execute(self, table, ctx):
        scored = [
            r
            for r in table.rows
            if isinstance(r.get(self.by), (int, float)) and math.isfinite(r[self.by])
        ]
        ordered = sorted(
            scored, key=lambda r: r[self.by], reverse=self.higher_is_better
        )
        ranks: dict[str, float] = {}
        i = 0
        while i < len(ordered):
            j = i + 1
            while j < len(ordered) and ordered[j][self.by] == ordered[i][self.by]:
                j += 1
            for r in ordered[i:j]:
                ranks[r["row_key"]] = (i + 1 + j) / 2.0
            i = j
        for row in table.rows:
            row[f"{ctx.instance}.final_rank"] = ranks.get(row["row_key"])
        return table


@register
@dataclass
class ListProposer(BlockFunction):
    """Toy proposer. Emits a fixed list of SMILES per round."""

    _name = "list_proposer"
    _role = "generator"
    _requires_input = False
    _description = "Emit a fixed SMILES list. Stands in for proposer generators."

    smiles: tuple[str, ...] = ()
    round_index: int = 0
    name: str = ""

    def execute(self, table, ctx):
        start = len(table.rows)
        new = CandidateTable.from_records(
            [
                {
                    "candidate_id": f"{ctx.instance}-r{self.round_index}-{i}",
                    "smiles_submitted": smi,
                    "source": f"generator:{ctx.instance}:round={self.round_index}",
                }
                for i, smi in enumerate(self.smiles)
            ]
        )
        for offset, row in enumerate(new.rows, start + 1):
            row["row_key"] = f"row_{offset:012d}"
        return CandidateTable(table.rows + new.rows)


@register
@dataclass
class ReportSink(BlockFunction):
    _name = "report"
    _role = "sink"
    _description = "Write the table as CSV and JSON under the run directory."

    top_n: int = 20
    name: str = ""

    def execute(self, table, ctx):
        return table
