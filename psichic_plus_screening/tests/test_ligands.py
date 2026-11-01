from pathlib import Path
import pickle
import sys

import numpy as np
import pandas as pd
import pytest
import torch

from psichic_ligand import LigandStore

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "psichic_plus_screening"))
import featurizers

from psichic_plus_screening import inference

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, ("ligand_input", "EMPTY_LIGAND: Ligand value is empty")),
        ("", ("ligand_input", "EMPTY_LIGAND: Ligand value is empty")),
        ("  ", ("ligand_input", "EMPTY_LIGAND: Ligand value is empty")),
        (
            "not-a-smiles",
            ("ligand_standardization", "INVALID_SMILES: Could not parse SMILES"),
        ),
        (
            ASPIRIN,
            (
                "prediction",
                "PREDICTION_MISSING: Ligand featurization succeeded but no prediction was emitted",
            ),
        ),
        (
            "[Na+].O=C([O-])c1ccccc1",
            (
                "prediction",
                "PREDICTION_MISSING: Ligand featurization succeeded but no prediction was emitted",
            ),
        ),
    ],
)
def test_diagnosis_preserves_csv_stage_and_reason(value, expected):
    assert inference._diagnose_ligand(value) == expected


def payload():
    return dict(
        interaction_keys=[("P", ASPIRIN)],
        ids=["pair"],
        protein_sequences=["ACDE"],
        residue_scores=[torch.arange(4).float()],
        atom_scores=[torch.arange(13).float()],
        fingerprints=torch.ones(1, 10),
    )


@pytest.mark.parametrize(
    "writer_type", [inference.AsyncInterpretWriter, inference.ProcessInterpretWriter]
)
def test_interpretation_writers_read_the_versioned_store(tmp_path, writer_type):
    store = LigandStore(tmp_path / "cache")
    graph = store.get(ASPIRIN)
    writer = writer_type(store, tmp_path / "interpret")
    writer.enqueue(payload())
    stats = writer.close()
    assert stats["writer_pairs"] == 1
    with (tmp_path / "interpret/pair/ligand.pkl").open("rb") as handle:
        mol = pickle.load(handle)
    assert [a.GetAtomicNum() for a in mol.GetAtoms()] == graph[
        "atom_types"
    ].flatten().tolist()
    assert (tmp_path / "interpret/pair/fingerprint.npy").exists()


def test_shape_precompute_uses_the_dataset_store(tmp_path, monkeypatch):
    frame = pd.DataFrame({"ID": ["pair"], "Ligand": [ASPIRIN], "Protein": ["P"]})
    dataset = featurizers.ProteinMoleculeDataset(
        frame, molecule_folder=tmp_path / "cache", result_path=str(tmp_path)
    )
    graph = dataset.ligands.get(ASPIRIN)
    calls = []

    def get(raw):
        calls.append(raw)
        return graph

    monkeypatch.setattr(dataset.ligands, "get", get)
    protein = dict(seq="ACDE", num_nodes=4, edge_index=torch.tensor([[0, 1], [1, 2]]))
    inference.compute_ligand_clique_counts(frame, dataset)
    dataset.prot_dict = {"P": protein}
    inference.compute_size_shape_records(frame, dataset)
    assert calls == [ASPIRIN, ASPIRIN]


def test_constant_attention_scores_are_finite():
    with np.errstate(divide="raise", invalid="raise"):
        assert np.array_equal(featurizers.minmax_norm(np.full(8, 0.5)), np.zeros(8))
