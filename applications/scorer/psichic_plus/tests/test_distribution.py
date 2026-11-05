import json
from dataclasses import FrozenInstanceError, asdict, replace
from types import SimpleNamespace

import pytest

from applications.generator.scorer import PSICHICScorer
from applications.scorer import (
    BetaDistribution,
    DistributionFailure,
    DistributionNotRequested,
    Failure,
    Output,
    Success,
)
from applications.scorer.psichic_plus.predictor import Predictor, _distribution_outcome


@pytest.fixture
def fake_inference(tmp_path):
    import pandas as pd

    def make(rows):
        calls = []

        def run(frame, *args):
            calls.append(frame.copy())
            return pd.DataFrame([
                {"ID": row_id, "inference_status": "success", **row}
                for row_id, row in zip(frame.ID, rows, strict=True)
            ]), None

        predictor = Predictor.__new__(Predictor)
        predictor.helpers = SimpleNamespace(
            prepare_input_frame=lambda frame, *args: frame,
            initialize_output_columns=lambda frame: frame,
            build_loader=lambda *args: (SimpleNamespace(dataset=SimpleNamespace(ligands={})),),
            run_screening=run,
            apply_invalid_ligand_policy=lambda frame, policy: (frame, None),
        )
        predictor.fasta = ("target", "AC")
        predictor.proteins = {}
        predictor.protein_folder = None
        predictor.cache = tmp_path
        predictor.batch_size = 2
        predictor.device = "cpu"
        predictor.model = object()
        return predictor, calls

    return make


def test_success_default_construction_and_computed_concentration():
    assert Success("token", 4.0, -4.0).distribution == DistributionNotRequested()
    distribution = BetaDistribution(1.0, 2.0)
    assert distribution.state == "valid"
    assert distribution.concentration == 3.0
    assert replace(distribution, alpha=4.0).concentration == 6.0
    with pytest.raises(FrozenInstanceError):
        distribution.concentration = 99.0


@pytest.mark.parametrize("output", [None, "affinity", Output.AFFINITY])
@pytest.mark.parametrize("parameters", [{}, {"reg_alpha": None, "reg_beta": float("inf")}])
def test_default_affinity_ignores_unavailable_distribution(fake_inference, output, parameters):
    predictor, calls = fake_inference([{"predicted_binding_affinity": 4.125, **parameters}])
    kwargs = {} if output is None else {"output": output}
    assert predictor.score([("token", "smiles")], **kwargs) == [Success("token", 4.125, -4.125)]
    assert len(calls) == 1


@pytest.mark.parametrize("output", ["distribution", Output.DISTRIBUTION])
def test_rich_batch_preserves_native_values_identities_and_one_inference(fake_inference, output):
    tokens = ["None", "nan", "", "重复/../\n", "None"]
    native_affinities = [6.0000004, 4.0, 3.0, 8.0, 9.0]
    parameters = [(2.0, 2.0), (1.0, 2.0), (1.0, 3.0), (4.0, 2.0), (3.0, 1.0)]
    predictor, calls = fake_inference([
        {"predicted_binding_affinity": affinity, "reg_alpha": alpha, "reg_beta": beta}
        for affinity, (alpha, beta) in zip(native_affinities, parameters, strict=True)
    ])
    inputs = [(token, f"smiles-{index}") for index, token in enumerate(tokens)]
    results = predictor.score(inputs, output=output)

    assert [result.token for result in results] == tokens
    assert [result.raw_affinity for result in results] == native_affinities
    assert [result.adaptive_score for result in results] == [-value for value in native_affinities]
    assert [result.distribution for result in results] == [BetaDistribution(*pair) for pair in parameters]
    assert len(calls) == 1
    assert calls[0].ID.tolist() == ["0", "1", "2", "3", "4"]
    assert calls[0].smiles.tolist() == [smiles for _, smiles in inputs]
    assert json.loads(json.dumps([asdict(result) for result in results], allow_nan=False))[0]["distribution"] == {
        "alpha": 2.0, "beta": 2.0, "concentration": 4.0, "state": "valid",
    }


@pytest.mark.parametrize("parameters", [
    {},
    {"reg_alpha": 2.0},
    {"reg_beta": 2.0},
    {"reg_alpha": None, "reg_beta": 2.0},
    {"reg_alpha": 2.0, "reg_beta": None},
    {"reg_alpha": "invalid", "reg_beta": 2.0},
    {"reg_alpha": 1e308, "reg_beta": 1e308},
    {"reg_alpha": 2.0, "reg_beta": 6.0},
] + [
    {"reg_alpha": value, "reg_beta": 2.0} for value in [float("nan"), float("inf"), -float("inf"), 0.0, -1.0]
] + [
    {"reg_alpha": 2.0, "reg_beta": value} for value in [float("nan"), float("inf"), -float("inf"), 0.0, -1.0]
])
def test_invalid_distribution_preserves_valid_affinity_and_diagnostics(fake_inference, parameters):
    predictor, calls = fake_inference([{"predicted_binding_affinity": 6.0, **parameters}])
    result, = predictor.score([("token", "smiles")], output="distribution")

    assert isinstance(result, Success)
    assert (result.token, result.raw_affinity, result.adaptive_score) == ("token", 6.0, -6.0)
    assert isinstance(result.distribution, DistributionFailure)
    assert result.distribution.state == "distribution-failure"
    assert result.distribution.reason
    assert result.distribution.raw_alpha == repr(parameters.get("reg_alpha"))
    assert result.distribution.raw_beta == repr(parameters.get("reg_beta"))
    json.dumps(asdict(result), allow_nan=False)
    assert len(calls) == 1


def test_pure_conversion_preserves_parameters_and_handles_large_finite_shapes():
    result = _distribution_outcome(8.0, 1e307, 5e306)
    assert isinstance(result, BetaDistribution)
    assert (result.alpha, result.beta, result.concentration) == (1e307, 5e306, 1.5e307)
    result = _distribution_outcome(6.0, 8e307, 8e307)
    assert isinstance(result, BetaDistribution)
    assert result.concentration == 1.6e308


@pytest.mark.parametrize("affinity,alpha,beta,valid", [
    (6.000006, 2.0, 2.0, True),
    (6.00002, 2.0, 2.0, False),
    (0.0000005, 1e-8, 1.0, True),
    (0.000002, 1e-8, 1.0, False),
    (0.5, 2.0, 2.0, False),
])
def test_scale_check_allows_float32_rounding_and_rejects_wrong_scale(affinity, alpha, beta, valid):
    result = _distribution_outcome(affinity, alpha, beta)
    assert isinstance(result, BetaDistribution if valid else DistributionFailure)


@pytest.mark.parametrize("output", ["", "Distribution", "uncertainty", None, 1, [], {}])
def test_invalid_output_is_rejected_before_model_or_input_work(output):
    predictor = Predictor.__new__(Predictor)

    def unread_inputs():
        raise AssertionError("Invalid output must be rejected before reading inputs")
        yield

    with pytest.raises(ValueError):
        predictor.score(unread_inputs(), output=output)


def test_output_is_keyword_only():
    with pytest.raises(TypeError):
        Predictor.__new__(Predictor).score([], "distribution")


@pytest.mark.parametrize("output", ["affinity", "distribution"])
def test_prediction_failures_remain_failures_in_both_modes(fake_inference, output):
    predictor, calls = fake_inference([
        {"predicted_binding_affinity": float("nan")},
        {"predicted_binding_affinity": float("inf")},
        {"inference_status": "failed", "failure_stage": "ligand_standardization", "failure_reason": "invalid"},
    ])
    results = predictor.score([("nan", "a"), ("inf", "b"), ("bad", "c")], output=output)
    assert results == [
        Failure("nan", "prediction", "Non-finite affinity"),
        Failure("inf", "prediction", "Non-finite affinity"),
        Failure("bad", "ligand_standardization", "invalid"),
    ]
    assert len(calls) == 1


def test_generator_callback_still_returns_one_float(fake_inference, tmp_path):
    predictor, calls = fake_inference([{"predicted_binding_affinity": 6.25}])
    scorer = PSICHICScorer(tmp_path / "target", tmp_path / "weights", "cpu", tmp_path)
    scorer.predictor = predictor
    value = scorer("smiles")
    assert type(value) is float
    assert value == 6.25
    assert len(calls) == 1
