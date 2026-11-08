import hashlib
import sys
from types import SimpleNamespace

from applications.scorer.psichic_plus.predictor import load_helpers
from applications.scorer.psichic_plus.provenance import source_provenance


class Predictor:
    def __init__(self, inference):
        self.helpers = SimpleNamespace(__file__=str(inference))


def test_source_fingerprint_tracks_override_and_actual_loaded_modules(tmp_path, monkeypatch):
    screening = tmp_path / "override"
    screening.mkdir()
    inference = screening / "inference.py"
    inference.write_text("implementation = 1\n")
    cached_model = tmp_path / "other_model.py"
    cached_model.write_text("model = 2\n")
    monkeypatch.setitem(sys.modules, "psichic_model", SimpleNamespace(__file__=str(cached_model)))
    predictor = Predictor(inference)
    before = source_provenance(predictor)
    assert before["screening_directory"] == str(screening)
    assert before["loaded_modules"]["psichic_model"] == str(cached_model)
    hashes = {item["path"]: item["sha256"] for item in before["source_files"]}
    assert hashes[str(inference)] == hashlib.sha256(inference.read_bytes()).hexdigest()
    assert hashes[str(cached_model)] == hashlib.sha256(cached_model.read_bytes()).hexdigest()
    inference.write_text("implementation = 3\n")
    assert source_provenance(predictor)["source_sha256"] != before["source_sha256"]


def test_load_helpers_registers_the_screening_module(tmp_path, monkeypatch):
    screening = tmp_path / "screening"
    screening.mkdir()
    (screening / "inference.py").write_text("value = 1\n")
    monkeypatch.setenv("PSICHIC_SCREENING_DIR", str(screening))
    monkeypatch.delitem(sys.modules, "adaptive_screening_inference", raising=False)

    helpers = load_helpers()

    assert helpers.value == 1
    assert sys.modules["adaptive_screening_inference"] is helpers
