import hashlib
import json
from pathlib import Path

import pytest

from helpers import graph_digest as independent_digest
from psichic_ligand import FEATURIZATION_VERSION, LigandFailure, featurize_ligand
from psichic_ligand._contract import graph_digest, load_contract, verify
from psichic_ligand.normalization import MoleculeNormalizer
from psichic_ligand.graph import smiles_to_graph
from psichic_ligand.normalization.tautomer_contract import RDKIT_RELEASE


def test_contract_provenance_selection_and_failure_coverage():
    header, rows = load_contract(FEATURIZATION_VERSION)
    assert header.contract == FEATURIZATION_VERSION
    assert header.rdkit == RDKIT_RELEASE
    assert header.source == "huankoh/psichic-plus@3c8e5e4 mol_featurizer"
    assert len({r.raw for r in rows}) == header.rows == len(rows)
    assert {"NON_SMALL_MOLECULE", "GENERAL_ERROR", "INVALID_SMILES"} <= {
        r.status for r in rows
    }
    assert (
        header.panel_sha256
        == "e7cf9301c66a03990252156f4cc6988f32f6d9ce9a9bf79f7f3e576d77a0dc64"
    )
    hashes = json.loads(
        Path(__file__).with_name("policy_change_hashes.json").read_text()
    )
    assert len(hashes) == 90
    assert set(hashes) <= {hashlib.sha256(r.raw.encode()).hexdigest() for r in rows}
    smiles = [
        "CC(=O)Oc1ccccc1C(=O)O",
        "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
        "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O",
        "[Na+].O=C([O-])c1ccccc1",
        "[13CH3]C(=O)Oc1ccccc1",
        "C[N+](C)(C)CC(=O)[O-]",
        "Oc1ccccn1",
        "O=C1C=CC=CN1",
        "I",
        "CC",
        "not-a-smiles",
        "",
        "C" * 160,
        "CC(=O)Oc1ccccc1C(=O)O.Cn1c(=O)c2c(ncn2C)n(C)c1=O",
    ]
    assert set(smiles) <= {r.raw for r in rows}


def test_verify_installed_artifact_matches_every_recorded_output():
    report = verify()
    assert report.checked > 2200
    assert report.ok, report.mismatches[:3]


def test_missing_version_fails_with_its_name():
    with pytest.raises(FileNotFoundError, match="missing-version"):
        load_contract("missing-version")


def test_public_function_equals_composition_on_fifty_contract_rows():
    _, rows = load_contract(FEATURIZATION_VERSION)
    for row in rows[:50]:
        result = featurize_ligand(row.raw)
        normalized = MoleculeNormalizer().standardize_normalize(row.raw)
        if normalized.smiles is None:
            assert result == LigandFailure(
                "ligand_standardization", normalized.status, normalized.error_message
            )
        else:
            assert (
                graph_digest(result)
                == independent_digest(smiles_to_graph(normalized.smiles))
                == row.digest
            )
