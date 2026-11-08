from argparse import Namespace
import pickle

import pytest

from applications.scorer.psichic_plus.predictor import _load_local_esm


class UnsupportedConfiguration:
    pass


def test_local_esm_namespace_checkpoint_preserves_all_weights_without_network(tmp_path, monkeypatch):
    import esm
    import torch

    original = esm.model.esm2.ESM2(num_layers=1, embed_dim=8, attention_heads=2)
    weights = original.state_dict()
    regression = {name: weights.pop(name) for name in tuple(weights) if name.startswith("contact_head.regression.")}
    path = tmp_path / "esm2_fixture.pt"
    torch.save({"cfg": {"model": Namespace(encoder_layers=1, encoder_embed_dim=8,
                                          encoder_attention_heads=2, token_dropout=True)}, "model": weights}, path)
    torch.save({"model": regression}, tmp_path / "esm2_fixture-contact-regression.pt")
    allowed_before = torch.serialization.get_safe_globals().copy()
    with pytest.raises(pickle.UnpicklingError, match="Namespace"):
        torch.load(path, weights_only=True)

    def no_network(*args, **kwargs):
        raise AssertionError("Local checkpoint loading attempted a download")

    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", no_network)
    restored, alphabet = _load_local_esm(esm.pretrained, path)
    for name, value in original.state_dict().items():
        assert torch.equal(restored.state_dict()[name], value)
    assert alphabet.all_toks == original.alphabet.all_toks
    assert torch.serialization.get_safe_globals() == allowed_before
    torch.save({"cfg": UnsupportedConfiguration()}, path)
    with pytest.raises(pickle.UnpicklingError, match="UnsupportedConfiguration"):
        _load_local_esm(esm.pretrained, path)
    assert torch.serialization.get_safe_globals() == allowed_before
