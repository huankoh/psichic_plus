"""Load and verify the independently recorded ligand contract."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import dataclass, field
from importlib.resources import files
from typing import Literal, Mapping

import numpy as np
import rdkit
import torch

from ._ligands import FEATURIZATION_VERSION, LigandFailure, featurize_ligand
from .normalization.tautomer_contract import RDKIT_RELEASE


@dataclass(frozen=True)
class ContractHeader:
    contract: str
    rdkit: str
    digest: str
    source: str
    recorded_on: str
    cross_checked_on: str
    panel_sha256: str
    selection: str
    rows: int
    input_sha256: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ContractRow:
    raw: str
    stage: Literal["ligand_standardization", "ligand_graph"] | None
    status: str
    message: str | None
    smiles: str | None
    digest: str | None


@dataclass(frozen=True)
class Mismatch:
    index: int
    row: ContractRow
    field: Literal["stage", "status", "message", "smiles", "digest", "clique_pe"]
    actual: str | None


@dataclass(frozen=True)
class VerifyReport:
    contract: str
    checked: int
    mismatches: tuple[Mismatch, ...]
    platform: str
    python: str
    rdkit: str
    rdkit_path: str
    torch: str

    @property
    def ok(self) -> bool:
        return not self.mismatches


DIGEST_EXCLUDED_KEYS = frozenset({"clique_pe"})
DIGEST_ALGORITHM = "sha256-bench-v2-no-clique-pe"


def graph_digest(graph: Mapping[str, object]) -> str:
    h = hashlib.sha256()
    for key in sorted(graph, key=str):
        if key in DIGEST_EXCLUDED_KEYS:
            continue
        h.update(f"k:{key};".encode())
        value = graph[key]
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu().contiguous()
            h.update(f"t:{tensor.dtype}:{tuple(tensor.shape)};".encode())
            h.update(tensor.numpy().tobytes())
        else:
            h.update(f"v:{type(value).__name__}:{value!r};".encode())
    return h.hexdigest()


def valid_clique_pe(graph: Mapping[str, object]) -> bool:
    values = graph.get("clique_pe")
    n = graph["clique_num_nodes"]
    if not isinstance(values, torch.Tensor) or values.dtype != torch.float32 or values.shape != (n, 20):
        return False
    if not torch.isfinite(values).all():
        return False
    expected = np.zeros((n, 20), dtype=np.float64)
    if n > 1:
        row, col = graph["tree_edge_index"].cpu().numpy()
        degree = np.bincount(row, minlength=n).clip(min=1)
        adjacency = np.zeros((n, n), dtype=np.float64)
        adjacency[row, col] = 1.0 / degree[row]
        power = adjacency.copy()
        for step in range(20):
            expected[:, step] = np.diag(power)
            power = power @ adjacency
    return np.allclose(values.detach().cpu().numpy(), expected, rtol=0, atol=1e-6)


def load_contract(version: str) -> tuple[ContractHeader, list[ContractRow]]:
    path = files("psichic_ligand").joinpath("contract", version + ".jsonl")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Missing ligand contract {version}. Record a new contract before changing the version."
        ) from exc
    header = ContractHeader(**json.loads(lines[0]))
    rows = [ContractRow(**json.loads(line)) for line in lines[1:]]
    if (
        header.contract != version
        or header.rdkit != RDKIT_RELEASE
        or header.digest != DIGEST_ALGORITHM
        or header.rows != len(rows)
    ):
        raise ValueError(f"Invalid ligand contract header for {version}")
    return header, rows


def conda_subdir() -> str:
    machine = platform.machine().lower()
    if sys.platform == "darwin":
        return "osx-arm64" if machine == "arm64" else "osx-64"
    if sys.platform == "win32":
        return "win-64" if machine in ("amd64", "x86_64") else f"win-{machine}"
    return (
        "linux-64"
        if sys.platform.startswith("linux") and machine == "x86_64"
        else f"{sys.platform}-{machine}"
    )


def verify() -> VerifyReport:
    header, rows = load_contract(FEATURIZATION_VERSION)
    mismatches = []
    for index, row in enumerate(rows):
        result = featurize_ligand(row.raw, timeout_seconds=10)
        if isinstance(result, LigandFailure):
            actual = dict(
                stage=result.stage,
                status=result.status,
                message=result.message,
                smiles=None,
                digest=None,
            )
        else:
            actual = dict(
                stage=None,
                status="SUCCESS",
                message=None,
                smiles=result["smiles"],
                digest=graph_digest(result),
            )
        if not isinstance(result, LigandFailure) and not valid_clique_pe(result):
            mismatches.append(Mismatch(index, row, "clique_pe", "invalid shape or values"))
        for key, value in actual.items():
            if value != getattr(row, key):
                mismatches.append(Mismatch(index, row, key, value))
    return VerifyReport(
        header.contract,
        len(rows),
        tuple(mismatches),
        conda_subdir(),
        platform.python_version(),
        rdkit.__version__,
        rdkit.__file__,
        torch.__version__,
    )
