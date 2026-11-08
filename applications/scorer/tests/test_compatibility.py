import importlib
import inspect
from pathlib import Path

import pytest


def test_legacy_module_and_new_module_are_identical():
    old = importlib.import_module("applications.scorer.predictor")
    new = importlib.import_module("applications.scorer.psichic_plus.predictor")
    assert old is new
    assert list(inspect.signature(old.Predictor).parameters) == ["fasta", "model_dir", "device", "cache", "batch_size"]


def test_default_screening_path_survives_relocation(monkeypatch, tmp_path):
    pytest.importorskip("torch")
    from applications.scorer.psichic_plus import predictor

    monkeypatch.delenv("PSICHIC_SCREENING_DIR", raising=False)
    seen = []

    def load(name, path):
        seen.append(Path(path))
        raise RuntimeError("stop before model loading")

    monkeypatch.setattr(predictor.importlib.util, "spec_from_file_location", load)
    with pytest.raises(RuntimeError, match="stop before"):
        predictor.Predictor("target.fasta", "weights", "cpu", str(tmp_path), 1)
    assert seen == [Path(__file__).resolve().parents[3] / "psichic_plus_screening/inference.py"]
    assert seen[0].is_file()
