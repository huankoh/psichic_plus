"""The versioned ligand graph contract and its persistent cache."""

from __future__ import annotations

import hashlib
import logging
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict

import torch
from torch import Tensor

from .normalization import NORMALIZATION_VERSION, MoleculeNormalizer
from .graph import GRAPH_VERSION, smiles_to_graph

FEATURIZATION_VERSION = f"{NORMALIZATION_VERSION}+{GRAPH_VERSION}"
FailureStage = Literal["ligand_standardization", "ligand_graph"]


class LigandGraph(TypedDict):
    smiles: str
    atom_idx: Tensor
    atom_types: Tensor
    atom_feature: Tensor
    atom_edge_index: Tensor
    atom_edge_attr: Tensor
    atom_num_nodes: int
    tree_edge_index: Tensor
    atom2clique_index: Tensor
    x_clique: Tensor
    clique_num_nodes: int
    clique_pe: Tensor


@dataclass(frozen=True)
class LigandFailure:
    stage: FailureStage
    status: str
    message: str | None

    @property
    def reason(self) -> str:
        default = (
            "Ligand could not be standardized"
            if self.stage == "ligand_standardization"
            else "Molecule graph is None"
        )
        return f"{self.status}: {self.message if self.message is not None else default}"


class LigandFeaturizationError(ValueError):
    def __init__(self, smiles: str, failure: LigandFailure) -> None:
        self.smiles = smiles
        self.failure = failure
        super().__init__(failure.reason)


def featurize_ligand(
    smiles: str, *, timeout_seconds: int = 10
) -> LigandGraph | LigandFailure:
    if not isinstance(smiles, str):
        raise TypeError("smiles must be a str")
    result = MoleculeNormalizer(timeout_seconds).standardize_normalize(smiles)
    if result.smiles is None:
        return LigandFailure(
            "ligand_standardization", result.status, result.error_message
        )
    try:
        graph = smiles_to_graph(result.smiles)
    except Exception as exc:
        return LigandFailure("ligand_graph", "GRAPH_CONSTRUCTION_ERROR", str(exc))
    if graph is None:
        return LigandFailure("ligand_graph", "GRAPH_CONSTRUCTION_ERROR", None)
    return graph


class LigandStore:
    def __init__(
        self,
        cache_dir: str | os.PathLike | None,
        *,
        error_log: str | os.PathLike | None = None,
        timeout_seconds: int = 10,
    ) -> None:
        self._cache_dir = Path(cache_dir) if cache_dir is not None else None
        self._error_log = Path(error_log) if error_log is not None else None
        self._timeout_seconds = timeout_seconds
        self._failures: dict[str, LigandFailure] = {}
        if self._cache_dir is not None and self._cache_dir.is_dir():
            with os.scandir(self._cache_dir) as entries:
                if any(
                    re.fullmatch(r"[0-9a-f]{32}\.pt", entry.name) for entry in entries
                ):
                    logging.warning(
                        "Ignoring legacy unversioned ligand cache files in %s. They can be deleted.",
                        self._cache_dir,
                    )

    @property
    def directory(self) -> Path | None:
        return (
            self._cache_dir / FEATURIZATION_VERSION
            if self._cache_dir is not None
            else None
        )

    def get(self, smiles: str) -> LigandGraph:
        if not isinstance(smiles, str):
            raise TypeError("smiles must be a str")
        if smiles in self._failures:
            raise LigandFeaturizationError(smiles, self._failures[smiles])
        path = self._path(smiles) if self.directory is not None else None
        if path is not None:
            graph = self._read(path)
            if graph is not None:
                return graph
        result = featurize_ligand(smiles, timeout_seconds=self._timeout_seconds)
        if isinstance(result, LigandFailure):
            self._failures[smiles] = result
            self._record_failure(smiles, result)
            raise LigandFeaturizationError(smiles, result)
        if path is not None:
            self._publish(path, result)
        return result

    def _path(self, smiles: str) -> Path:
        return self.directory / (
            hashlib.sha256(smiles.encode("utf-8")).hexdigest()[:32] + ".pt"
        )

    def _read(self, path: Path) -> LigandGraph | None:
        if not path.exists():
            return None
        try:
            graph = torch.load(path, map_location="cpu", weights_only=True)
            if (
                not isinstance(graph, dict)
                or graph.keys() != LigandGraph.__annotations__.keys()
            ):
                raise ValueError("not a ligand graph dictionary")
            if (
                not isinstance(graph["smiles"], str)
                or any(
                    not isinstance(graph[k], Tensor)
                    for k in graph
                    if k not in ("smiles", "atom_num_nodes", "clique_num_nodes")
                )
                or any(
                    type(graph[k]) is not int
                    for k in ("atom_num_nodes", "clique_num_nodes")
                )
            ):
                raise ValueError("invalid ligand graph values")
            return graph
        except Exception as exc:
            logging.warning("Unreadable ligand cache %s, recomputing: %s", path, exc)
            return None

    def _publish(self, path: Path, graph: LigandGraph) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=path.parent, suffix=".tmp", delete=False
            ) as handle:
                temporary = Path(handle.name)
                torch.save(graph, handle)
            try:
                os.replace(temporary, path)
            except PermissionError:
                if os.name != "nt" or self._read(path) is None:
                    raise
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def _record_failure(self, smiles: str, failure: LigandFailure) -> None:
        if self._error_log is None:
            return
        self._error_log.parent.mkdir(parents=True, exist_ok=True)
        if failure.stage == "ligand_standardization":
            status = f"failed to normalize because [{failure.status}]"
            message = failure.message
        else:
            status = "successful normalization but fail at graph construction"
            message = None
        with self._error_log.open("a", encoding="utf-8") as handle:
            handle.write(
                f"||||| {smiles} ||||| with status ({status}); and error message - {message}\n"
            )
