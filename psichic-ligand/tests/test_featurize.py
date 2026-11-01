from types import SimpleNamespace

import pytest

from helpers import graph_digest
from psichic_ligand import LigandFailure, featurize_ligand
from psichic_ligand import _ligands
from psichic_ligand.normalization import MoleculeNormalizer
from psichic_ligand.graph import smiles_to_graph

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"


@pytest.mark.parametrize("value", [None, 3, [], b"CC"])
def test_only_strings_are_accepted(value):
    with pytest.raises(TypeError, match="str"):
        featurize_ligand(value)


@pytest.mark.parametrize(
    "stage,status,message,reason",
    [
        (
            "ligand_standardization",
            "INVALID_SMILES",
            "Could not parse SMILES",
            "INVALID_SMILES: Could not parse SMILES",
        ),
        (
            "ligand_standardization",
            "TIMEOUT",
            "Standardization timeout",
            "TIMEOUT: Standardization timeout",
        ),
        (
            "ligand_standardization",
            "GENERAL_ERROR",
            None,
            "GENERAL_ERROR: Ligand could not be standardized",
        ),
        (
            "ligand_graph",
            "GRAPH_CONSTRUCTION_ERROR",
            None,
            "GRAPH_CONSTRUCTION_ERROR: Molecule graph is None",
        ),
        (
            "ligand_graph",
            "GRAPH_CONSTRUCTION_ERROR",
            "bad graph",
            "GRAPH_CONSTRUCTION_ERROR: bad graph",
        ),
    ],
)
def test_failure_reason(stage, status, message, reason):
    assert LigandFailure(stage, status, message).reason == reason


@pytest.mark.parametrize(
    "status",
    [
        "INVALID_SMILES",
        "TIMEOUT",
        "NON_SMALL_MOLECULE",
        "INORGANIC_MOLECULE",
        "MIXTURE",
        "STANDARDIZATION_ERROR",
        "GENERAL_ERROR",
        "NORMALIZATION_CYCLE",
        "NORMALIZATION_LIMIT",
    ],
)
def test_normalizer_failures_keep_stage_status_and_message(monkeypatch, status):
    normalizer = SimpleNamespace(
        standardize_normalize=lambda raw: SimpleNamespace(
            smiles=None, status=status, error_message="detail"
        )
    )
    monkeypatch.setattr(_ligands, "MoleculeNormalizer", lambda timeout: normalizer)
    assert featurize_ligand(ASPIRIN) == LigandFailure(
        "ligand_standardization", status, "detail"
    )


@pytest.mark.parametrize("raises", [False, True])
def test_graph_failure_becomes_a_value(monkeypatch, raises):
    def build(smiles):
        if raises:
            raise ValueError("bad graph")
        return None

    monkeypatch.setattr(_ligands, "smiles_to_graph", build)
    assert featurize_ligand(ASPIRIN) == LigandFailure(
        "ligand_graph", "GRAPH_CONSTRUCTION_ERROR", "bad graph" if raises else None
    )


@pytest.mark.parametrize(
    "raw", [ASPIRIN, "[Na+].O=C([O-])c1ccccc1", "  ", "not-a-smiles"]
)
def test_public_function_equals_stage_composition(raw):
    normalized = MoleculeNormalizer().standardize_normalize(raw)
    actual = featurize_ligand(raw)
    if normalized.smiles is None:
        assert actual == LigandFailure(
            "ligand_standardization", normalized.status, normalized.error_message
        )
    else:
        assert graph_digest(actual) == graph_digest(smiles_to_graph(normalized.smiles))
