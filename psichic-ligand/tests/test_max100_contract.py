"""RDKit 2024.03.3 maxTransforms=100 contract pins."""
from __future__ import annotations

from pathlib import Path

from psichic_ligand.normalization.pipeline import NORMALIZATION_VERSION
from psichic_ligand.normalization.tautomer_contract import (
    PREPROCESSING_POLICY,
    TAUTOMER_MAX_TAUTOMERS,
    TAUTOMER_MAX_TRANSFORMS,
)

NATIVE = Path(__file__).resolve().parents[1] / "src/psichic_ligand/normalization/native/tautomer_ext.cpp"


def test_policy_is_2024_max100_not_max_tautomers():
    assert NORMALIZATION_VERSION == PREPROCESSING_POLICY
    assert PREPROCESSING_POLICY == "rdkit-2024.03.3-enumpick-max100-v1"
    assert TAUTOMER_MAX_TRANSFORMS == 100
    assert TAUTOMER_MAX_TAUTOMERS == 2**32 - 1


def test_native_loop_literal_is_100_not_1000():
    text = NATIVE.read_text()
    assert "const unsigned int maxTransforms = 100;" in text
    assert "const unsigned int maxTransforms = 1000;" not in text


def test_native_pick_returns_mol_for_stereo_and_plain_inputs():
    from rdkit import Chem

    from psichic_ligand.normalization.native import pick_canonical_tautomer

    smiles = (
        "CC(=O)Oc1ccccc1C(=O)O",
        "CC(=O)CC(C)=O",
        "COC(=O)c1ccc(NC(=O)Nc2ccc(/N=N/c3ccc(C(=O)Nc4ccc(O)cc4)cc3)cc2)cc1",
        "C[C@H](O)C(=O)O",
    )
    for raw in smiles:
        mol = Chem.MolFromSmiles(raw)
        pickled, status = pick_canonical_tautomer(
            bytes(mol.ToBinary()), 2**32 - 1, True, True
        )
        assert Chem.Mol(pickled).GetNumAtoms() >= 1
        assert status in (0, 1, 2)
