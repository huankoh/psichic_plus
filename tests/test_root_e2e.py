import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from psichic_ligand import FEATURIZATION_VERSION

ROOT = Path(__file__).resolve().parents[1]
pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        os.environ.get("PSICHIC_E2E") != "1",
        reason="Set PSICHIC_E2E=1 to run pretrained CPU CLI checks",
    ),
]


def run_cli(args, *, cwd=ROOT):
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", TMPDIR=".")
    result = subprocess.run(
        [sys.executable, *map(str, args)],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def assert_predictions_and_interpretations(output):
    frame = pd.read_csv(output / "inference_results.csv")
    assert len(frame) == 3
    assert np.isfinite(
        frame[["predicted_binding_affinity", "reg_alpha", "reg_beta"]].to_numpy()
    ).all()
    for pair_id in frame["ID"]:
        pair = output / "interpretations" / pair_id
        assert (pair / "protein.csv").is_file()
        assert np.isfinite(np.load(pair / "fingerprint.npy")).all()
        with (pair / "ligand.pkl").open("rb") as handle:
            molecule = pickle.load(handle)
        assert all(
            np.isfinite(float(atom.GetProp("PSICHIC_Atom_Score")))
            for atom in molecule.GetAtoms()
        )
    return frame


def test_root_cpu_cli_with_one_clique_and_interpretations(tmp_path):
    output = tmp_path / "root"
    run_cli(
        [
            ROOT / "inference.py",
            "--input_file",
            ROOT / "tests/data/e2e.csv",
            "--output_folder",
            output,
            "--device",
            "cpu",
            "--save_interpret",
            "--batch_size",
            "1",
        ]
    )
    assert_predictions_and_interpretations(output)
    assert not (output / "ligands").exists()


def test_screening_cpu_cli_reuses_a_versioned_cache(tmp_path):
    cache = tmp_path / "ligands"
    cache.mkdir()
    legacy = cache / (hashlib.sha256(b"CC(=O)Oc1ccccc1C(=O)O").hexdigest()[:32] + ".pt")
    legacy.write_bytes(b"legacy file stays untouched")
    args = [
        ROOT / "psichic_plus_screening/inference.py",
        "--input_file",
        ROOT / "tests/data/e2e.csv",
        "--model_dir",
        ROOT / "psichic_plus_screening/pretrained_weights/PSICHIC_plus",
        "--device",
        "cpu",
        "--save_interpret",
        "--num_workers",
        "2",
        "--batch_size",
        "1",
        "--precision",
        "fp32",
        "--batching_strategy",
        "protein_size_bucketed",
        "--ligand_cache_dir",
        cache,
        "--protein_cache_dir",
        tmp_path / "proteins",
    ]
    first, second = tmp_path / "first", tmp_path / "second"
    run_cli([*args, "--output_folder", first])
    paths = list((cache / FEATURIZATION_VERSION).glob("*.pt"))
    assert len(paths) == 3
    before = {p.name: (p.stat().st_mtime_ns, p.read_bytes()) for p in paths}
    run_cli([*args, "--output_folder", second])
    pd.testing.assert_frame_equal(
        assert_predictions_and_interpretations(first),
        assert_predictions_and_interpretations(second),
        check_exact=True,
    )
    assert {p.name: (p.stat().st_mtime_ns, p.read_bytes()) for p in paths} == before
    assert legacy.read_bytes() == b"legacy file stays untouched"


def test_fewshot_cpu_smoke(tmp_path):
    frame = pd.read_csv(ROOT / "tests/data/e2e.csv")
    frame = pd.concat([frame, frame], ignore_index=True)
    frame["ID"] = [f"fewshot-{i}" for i in range(6)]
    frame["regression_label"] = np.arange(6) * 0.5 + 4
    frame.to_csv(tmp_path / "fep_test.csv", index=False)
    output = tmp_path / "result"
    run_cli(
        [
            ROOT / "fewshot.py",
            "--device",
            "cpu",
            "--trained_model_path",
            ROOT / "trained_weights/PSICHIC_plus",
            "--datafolder",
            tmp_path,
            "--result_path",
            output,
            "--total_iters",
            "1",
            "--eval_interval",
            "1",
            "--batch_size",
            "3",
            "--train_frac",
            "0.5",
        ],
        cwd=tmp_path,
    )
    summary = pd.read_csv(output / "test-protein_seed168_summary.csv")
    assert summary.loc[0, "best_iter"] == 1
    assert np.isfinite(summary.loc[0, "final_rmse"])
    entries = json.loads((output / "test-protein_seed168_log.json").read_text())
    assert len(entries) == 1 and np.isfinite(entries[0]["loss"])
