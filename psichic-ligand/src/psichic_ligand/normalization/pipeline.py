"""Normalize raw SMILES until the complete PSICHIC+ cleanup is stable."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional, TypeVar

from rdkit import Chem

from .native import normalize_once as _native_normalize_once
from .native import papyrus_rest as _native_papyrus_rest
from .papyrus_structure_pipeline import standardizer as Papyrus_standardizer

T = TypeVar("T")
# Salts runtime.safe_smiles_filename so @100 graphs do not mix with legacy @1000.
NORMALIZATION_VERSION = "rdkit-2024.03.3-enumpick-max100-v1"
MAX_NORMALIZATION_PASSES = 20


class TimeoutHandler:
    def __init__(self, timeout_seconds: int = 10):
        self.timeout_seconds = timeout_seconds

    def run_with_timeout(self, func: Callable[..., T], *args, **kwargs) -> Optional[T]:
        # A timed-out worker can finish after another invocation has started.
        # Keep its result and exception private to this invocation.
        result: Optional[T] = None
        exception: Optional[Exception] = None

        def worker():
            nonlocal result, exception
            try:
                result = func(*args, **kwargs)
            except Exception as exc:  # pragma: no cover - propagated below
                exception = exc

        thread = threading.Thread(target=worker)
        thread.daemon = True
        thread.start()
        thread.join(timeout=self.timeout_seconds)

        if thread.is_alive():
            return None
        if exception is not None:
            raise exception
        return result


@dataclass
class NormalizationResult:
    smiles: Optional[str]
    status: str
    error_message: Optional[str] = None


class MoleculeNormalizer:
    def __init__(self, timeout_seconds: int = 10):
        self.timeout_handler = TimeoutHandler(timeout_seconds)

    def _result_from_native(self, kind, out, err, blob) -> NormalizationResult:
        if kind == "ok":
            return NormalizationResult(out, "SUCCESS", None)
        if kind == "invalid":
            return NormalizationResult(
                None, "INVALID_SMILES", err or "Could not parse SMILES"
            )
        if kind == "mixture":
            return NormalizationResult(None, "MIXTURE", err)
        if kind == "non_small":
            return NormalizationResult(None, "NON_SMALL_MOLECULE", err)
        if kind == "need_chembl2":
            try:
                mol = Chem.Mol(blob)
                mol = Papyrus_standardizer._apply_chembl_standardization(mol)
                return NormalizationResult(
                    Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True),
                    "SUCCESS",
                    None,
                )
            except Exception as exc:
                raise RuntimeError("Molecule could not be standardized") from exc
        if kind == "error":
            return NormalizationResult(None, "GENERAL_ERROR", err)
        return NormalizationResult(None, "GENERAL_ERROR", f"Unknown native result: {kind}")

    def _normalize_once(self, smiles: str) -> NormalizationResult:
        kind, out, err, blob = _native_normalize_once(smiles)
        if kind == "need_chembl1":
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return NormalizationResult(None, "INVALID_SMILES", "Could not parse SMILES")
            try:
                mol = Papyrus_standardizer._apply_chembl_standardization(mol)
                kind, out, err, blob = _native_papyrus_rest(bytes(mol.ToBinary()))
            except Exception as exc:
                raise RuntimeError("Molecule could not be standardized") from exc
        return self._result_from_native(kind, out, err, blob)

    def _normalize_until_stable(self, smiles, deadline, cancelled):
        current = smiles
        seen = set()
        for _ in range(MAX_NORMALIZATION_PASSES):
            if cancelled.is_set() or time.monotonic() >= deadline:
                return None
            try:
                result = self._normalize_once(current)
            except Exception as exc:
                result = NormalizationResult(None, "GENERAL_ERROR", str(exc))
            # Native chemistry cannot be interrupted safely. Once it returns,
            # never accept a late success or begin another pass after timeout.
            if cancelled.is_set() or time.monotonic() >= deadline:
                return None
            if result.status != "SUCCESS":
                return result
            if result.smiles == current:
                return result
            seen.add(current)
            if result.smiles in seen:
                return NormalizationResult(None, "NORMALIZATION_CYCLE", "Normalization entered a cycle")
            current = result.smiles
        return NormalizationResult(
            None, "NORMALIZATION_LIMIT",
            f"Normalization did not stabilize within {MAX_NORMALIZATION_PASSES} passes",
        )

    def standardize_normalize(self, smiles: str) -> NormalizationResult:
        cancelled = threading.Event()
        deadline = time.monotonic() + self.timeout_handler.timeout_seconds
        try:
            result = self.timeout_handler.run_with_timeout(
                self._normalize_until_stable, smiles, deadline, cancelled,
            )
            if result is None:
                return NormalizationResult(None, "TIMEOUT", "Standardization timeout")
            return result
        except Exception as exc:
            return NormalizationResult(None, "GENERAL_ERROR", str(exc))
        finally:
            cancelled.set()
