import importlib.util
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from applications.scorer import (
    BetaDistribution,
    DistributionFailure,
    DistributionNotRequested,
    Failure,
    Success,
)
from applications.scorer.psichic_plus.predictor import Predictor


@pytest.fixture
def native_inference(tmp_path, monkeypatch):
    screening = Path(__file__).resolve().parents[4] / "psichic_plus_screening"
    monkeypatch.syspath_prepend(str(screening))
    spec = importlib.util.spec_from_file_location("prediction_cleanup_native", screening / "inference.py")
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)

    def make(rows):
        calls = []

        def run(frame, *args):
            calls.append(frame.copy())
            return pd.DataFrame([
                {"ID": row_id, "Ligand": ligand, **row}
                for row_id, ligand, row in zip(frame.ID, frame.Ligand, rows, strict=True)
            ], dtype=object), None

        predictor = Predictor.__new__(Predictor)
        predictor.helpers = SimpleNamespace(
            prepare_input_frame=native.prepare_input_frame,
            initialize_output_columns=native.initialize_output_columns,
            build_loader=lambda *args: (SimpleNamespace(dataset=SimpleNamespace(ligands={})),),
            run_screening=run,
            apply_invalid_ligand_policy=native.apply_invalid_ligand_policy,
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


@pytest.mark.parametrize("output", [None, "affinity", "distribution"])
@pytest.mark.parametrize("parameters", [
    {},
    {"reg_beta": 4.0},
    {"reg_alpha": 2.0},
    {"reg_alpha": None, "reg_beta": 4.0},
    {"reg_alpha": 2.0, "reg_beta": None},
    {"reg_alpha": float("nan"), "reg_beta": 4.0},
    {"reg_alpha": 2.0, "reg_beta": float("nan")},
])
def test_native_cleanup_retains_affinity_without_distribution(native_inference, output, parameters):
    predictor, calls = native_inference([{"predicted_binding_affinity": 4.0, **parameters}])
    kwargs = {} if output is None else {"output": output}
    result, = predictor.score([("retained", "CNc1ccccc1")], **kwargs)

    assert isinstance(result, Success)
    assert (result.token, result.raw_affinity, result.adaptive_score) == ("retained", 4.0, -4.0)
    if output == "distribution":
        assert isinstance(result.distribution, DistributionFailure)
        assert result.distribution.reason
        assert result.distribution.raw_alpha == repr(parameters.get("reg_alpha"))
        assert result.distribution.raw_beta == repr(parameters.get("reg_beta"))
    else:
        assert result.distribution == DistributionNotRequested()
    json.dumps(asdict(result), allow_nan=False)
    assert len(calls) == 1


@pytest.mark.parametrize("output", [None, "affinity", "distribution"])
def test_native_cleanup_preserves_mixed_outcomes_and_input_order(native_inference, output):
    rows = [
        {"predicted_binding_affinity": 4.0, "reg_alpha": 2.0, "reg_beta": 4.0},
        {"predicted_binding_affinity": 5.0, "reg_alpha": None, "reg_beta": 4.0},
        {"predicted_binding_affinity": None},
        {"predicted_binding_affinity": float("nan")},
        {"predicted_binding_affinity": float("inf"), "reg_alpha": 2.0, "reg_beta": 4.0},
        {"predicted_binding_affinity": -float("inf"), "reg_alpha": 2.0, "reg_beta": 4.0},
        {"predicted_binding_affinity": 6.0, "reg_alpha": 2.0, "reg_beta": 2.0,
         "inference_status": "failed", "failure_stage": "ligand_standardization", "failure_reason": "invalid ligand"},
        {"predicted_binding_affinity": 6.0, "reg_alpha": 2.0, "reg_beta": 2.0,
         "inference_status": "failed", "failure_stage": "prediction", "failure_reason": "model failed"},
        {"predicted_binding_affinity": None,
         "inference_status": "failed", "failure_stage": "prediction", "failure_reason": "model failed"},
        {"predicted_binding_affinity": None},
        {"predicted_binding_affinity": 3.0, "reg_alpha": 1.0, "reg_beta": 3.0},
    ]
    tokens = ["duplicate", "重复/../\n", "bad", "missing", "inf", "-inf", "bad-native", "model", "model-null", "empty", "duplicate"]
    inputs = [(token, "invalid" if index in (2, 6) else "" if index == 9 else "CNc1ccccc1")
              for index, token in enumerate(tokens)]
    predictor, calls = native_inference(rows)
    kwargs = {} if output is None else {"output": output}
    results = predictor.score(inputs, **kwargs)

    assert [result.token for result in results] == tokens
    for index, affinity in ((0, 4.0), (1, 5.0), (10, 3.0)):
        assert isinstance(results[index], Success)
        assert results[index].raw_affinity == affinity
        assert results[index].adaptive_score == -affinity
    if output == "distribution":
        assert results[0].distribution == BetaDistribution(2.0, 4.0)
        assert isinstance(results[1].distribution, DistributionFailure)
        assert results[10].distribution == BetaDistribution(1.0, 3.0)
    else:
        assert all(results[index].distribution == DistributionNotRequested() for index in (0, 1, 10))
    assert isinstance(results[2], Failure)
    assert results[2].stage == "ligand_standardization"
    assert "PREDICTION_MISSING" in results[3].reason
    assert results[4:9] == [
        Failure("inf", "prediction", "Non-finite affinity"),
        Failure("-inf", "prediction", "Non-finite affinity"),
        Failure("bad-native", "ligand_standardization", "invalid ligand"),
        Failure("model", "prediction", "model failed"),
        Failure("model-null", "prediction", "model failed"),
    ]
    assert isinstance(results[9], Failure)
    assert results[9].stage == "ligand_input"
    assert results[9].reason.startswith("EMPTY_LIGAND")
    json.dumps([asdict(result) for result in results], allow_nan=False)
    assert len(calls) == 1


@pytest.mark.parametrize("output", ["affinity", "distribution"])
def test_native_cleanup_diagnoses_absent_affinity_column(native_inference, output):
    predictor, _ = native_inference([{"reg_alpha": 2.0, "reg_beta": 4.0}])
    result, = predictor.score([("missing", "CNc1ccccc1")], output=output)

    assert isinstance(result, Failure)
    assert result.token == "missing"
    assert result.stage == "prediction"
    assert "PREDICTION_MISSING" in result.reason


def test_public_evidence_retains_full_native_rows_from_one_pass(native_inference):
    predictor, calls = native_inference([
        {'predicted_binding_affinity': 9., 'reg_alpha': float('inf'), 'reg_beta': 1.,
         'diagnostic': 'original native value'},
        {'predicted_binding_affinity': None},
    ])
    results = predictor.score_with_evidence([('duplicate', 'CCO'), ('duplicate', '')], output='distribution')
    assert len(calls) == 1
    assert isinstance(results[0].outcome, Success)
    assert isinstance(results[0].outcome.distribution, DistributionFailure)
    assert results[0].native['reg_alpha'] == float('inf')
    assert results[0].native['diagnostic'] == 'original native value'
    assert results[0].native['predicted_binding_affinity'] == 9.
    assert isinstance(results[1].outcome, Failure)
    assert results[1].outcome.stage == 'ligand_input'
    assert results[1].native['predicted_binding_affinity'] is None
