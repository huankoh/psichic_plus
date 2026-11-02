import os
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from applications.scorer import Failure, Success
from applications.scorer.predictor import Predictor, _protein_state


def test_protein_features_use_the_requested_cuda_device(monkeypatch):
    model = SimpleNamespace(device=None)
    model.eval = lambda: None
    model.to = lambda device: setattr(model, "device", device)
    alphabet = SimpleNamespace(get_batch_converter=lambda: None)
    featurizers = SimpleNamespace(
        esm=SimpleNamespace(pretrained=SimpleNamespace(load_model_and_alphabet=lambda _: (model, alphabet))),
        pro_res_table={},
        replace_non_standard_residues=lambda sequence, _: sequence,
        seq_feature=lambda sequence: np.zeros((len(sequence), 1)),
        esm_extract=lambda *args, **kwargs: (torch.zeros((2, 1)), torch.zeros((2, 2)), None),
        contact_map=lambda _: (torch.zeros((2, 1), dtype=torch.long), torch.zeros(1)),
    )
    monkeypatch.setitem(sys.modules, "featurizers", featurizers)
    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    protein = _protein_state(("target", "AC"), torch.device("cuda:1"))

    assert model.device == torch.device("cuda:1")
    assert protein["target"]["seq"] == "AC"


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_real_predictor_preserves_ids_and_honors_device(tmp_path, device):
    model_dir = os.environ.get("PSICHIC_TEST_MODEL_DIR")
    if not model_dir:
        pytest.skip("Set PSICHIC_TEST_MODEL_DIR to authorized PSICHIC+ weights")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    gpu_baseline = 0
    if torch.cuda.is_available():
        torch.cuda.init()
        gpu_baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()

    fasta = Path(__file__).resolve().parents[2] / "generator/example/target.fasta"
    predictor = Predictor(str(fasta), model_dir, device, str(tmp_path), 2)
    inputs = [
        ("None", "CNc1ccccc1"),
        ("nan", "O=C(O)c1ccccc1"),
        ("", "invalid"),
        ("Row_1", "CNc1ccccc1"),
        ("None", "O=C(O)c1ccccc1"),
    ]
    results = predictor.score(inputs)

    assert [result.token for result in results] == [token for token, _ in inputs]
    assert isinstance(results[2], Failure)
    assert all(isinstance(results[index], Success) for index in (0, 1, 3, 4))
    assert next(predictor.model.parameters()).device.type == device
    if torch.cuda.is_available():
        if device == "cpu":
            assert torch.cuda.max_memory_allocated() - gpu_baseline < 1024 * 1024
        else:
            assert torch.cuda.max_memory_allocated() > gpu_baseline
