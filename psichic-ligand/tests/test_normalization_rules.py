"""Rule precompilation must preserve the unchanged chemistry helpers."""

import contextlib
import io

import pytest
from rdkit import Chem
from chembl_structure_pipeline import standardizer as original_chembl


from psichic_ligand.normalization.chembl_structure_pipeline import standardizer as chembl
from psichic_ligand.normalization.papyrus_structure_pipeline import standardizer as papyrus
from papyrus_structure_pipeline import standardizer as original_papyrus


ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
SUPPLEMENTARY_RULES = tuple(dict.fromkeys(papyrus.SALTS + papyrus.METALS))


def molecule_signature(mol):
    return (
        Chem.MolToSmiles(mol),
        [(atom.GetAtomicNum(), atom.GetFormalCharge(), atom.GetIsotope(),
          atom.GetChiralTag(), atom.GetNumExplicitHs(), atom.GetNoImplicit())
         for atom in mol.GetAtoms()],
        [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond.GetBondType(),
          bond.GetStereo(), bond.GetBondDir()) for bond in mol.GetBonds()],
    )


@pytest.mark.parametrize("salt", SUPPLEMENTARY_RULES)
@pytest.mark.parametrize("include_metals", [False, True])
def test_every_supplementary_rule_matches_original(monkeypatch, salt, include_metals):
    # The imported helper mutates SALTS when metals are disabled. Isolate that
    # inherited side effect so one reference call cannot contaminate the next.
    monkeypatch.setattr(original_papyrus, "SALTS", list(papyrus.SALTS))
    for raw in [salt, ASPIRIN + "." + salt]:
        original_papyrus.SALTS = list(papyrus.SALTS)
        expected, expected_type = original_papyrus._remove_supplementary_salts(
            Chem.MolFromSmiles(raw), include_metals, return_type=True,
        )
        actual, actual_type = papyrus._remove_supplementary_salts(
            Chem.MolFromSmiles(raw), include_metals, return_type=True,
        )
        assert actual_type.value == expected_type.value
        assert molecule_signature(actual) == molecule_signature(expected)


def test_changed_supplementary_definitions_are_used(monkeypatch):
    for module in [original_papyrus, papyrus]:
        monkeypatch.setattr(module, "SALTS", ["C=C"])
        monkeypatch.setattr(module, "METALS", ["[Li+]"])
    raw = ASPIRIN + ".C=C.[Li+]"
    for salt in ["C=C", "CC"]:
        # Change list contents, not just its identity.
        original_papyrus.SALTS[:] = [salt]
        papyrus.SALTS[:] = [salt]
        expected = original_papyrus._remove_supplementary_salts(Chem.MolFromSmiles(raw))
        actual = papyrus._remove_supplementary_salts(Chem.MolFromSmiles(raw))
        assert molecule_signature(actual) == molecule_signature(expected)


FRAGMENT_CASES = [
    "", "O", "[Na+].[Cl-]", "O.CC(=O)O", "CCN.CCN", "CCC.CNC",
    ASPIRIN, ASPIRIN + ".O.CCO.[Na+].[Cl-]",
    "[13CH3]C(=O)O.[2H]O[2H]", "[Na+].O=C([O-])c1ccccc1",
    "C[N+](C)(C)CC(=O)[O-].[Cl-]", "[Fe]." + ASPIRIN,
    "O=C(O)[C@@H](O)[C@H](O)C(=O)O." + ASPIRIN,
]


@pytest.mark.parametrize("raw", FRAGMENT_CASES)
@pytest.mark.parametrize("neutralize,check_exclusion", [(False, False), (True, False), (True, True)])
def test_fragment_parent_and_matching_order_match_original(raw, neutralize, check_exclusion):
    results = []
    for module in [original_chembl, chembl]:
        output = io.StringIO()
        try:
            with contextlib.redirect_stdout(output):
                mol, exclude = module.get_fragment_parent_mol(
                    Chem.MolFromSmiles(raw), neutralize=neutralize,
                    check_exclusion=check_exclusion, verbose=True,
                )
            results.append((molecule_signature(mol), exclude, output.getvalue()))
        except Exception as error:
            results.append((type(error), str(error), output.getvalue()))
    assert results[1] == results[0]


@pytest.mark.parametrize("raw", [
    "O=C(O)[C@@H](O)[C@H](O)C(=O)O",
    "COC(=O)[C@@H](O)[C@H](O)C(=O)O",
    ASPIRIN,
])
def test_compiled_tartrate_query_matches_original(raw):
    expected = original_chembl.flatten_tartrate_mol(Chem.MolFromSmiles(raw))
    actual = chembl.flatten_tartrate_mol(Chem.MolFromSmiles(raw))
    assert molecule_signature(actual) == molecule_signature(expected)
