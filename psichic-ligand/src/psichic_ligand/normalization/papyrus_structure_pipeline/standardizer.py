# -*- coding: utf-8 -*-

#
#  Copyright (c) 2023 Olivier J. M. Béquignon
#  All rights reserved.
#
#  This file is part of the Papyrus_StructurePipeline project.
#  The contents are covered by the terms of the MIT license
#  which is included in the file LICENSE, found at the root
#  of the source tree.

"""Standardizing methods."""

from enum import Enum, auto
from typing import Optional, Tuple, Union

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize.rdMolStandardize import TautomerEnumerator, Uncharger
from rdkit.rdBase import BlockLogs

from ..native import pick_canonical_tautomer as _native_pick_canonical_tautomer
from ..tautomer_contract import TAUTOMER_MAX_TRANSFORMS

# Remove warnings of importing the ChEMBL Structure Pipeline
with BlockLogs():
    from ..chembl_structure_pipeline import standardizer
    from ..chembl_structure_pipeline.exclude_flag import exclude_flag


class StandardizationResult(Enum):
    """Result of the standardization process."""
    CORRECT_MOLECULE = auto()
    NON_SMALL_MOLECULE = auto()
    INORGANIC_MOLECULE = auto()
    MIXTURE_MOLECULE = auto()
    STANDARDIZATION_ERROR = auto()


class InorganicSubtype(Enum):
    """Reason the compound is deemed inorganic."""
    IS_ORGANIC = auto()
    NO_CC_BOND = auto()
    NOT_CHONPSFClIBrB = auto()
    EXCLUSION_FLAG_SET = auto()  # molecule was flagged by the ChEMBL Structure Pipeline


class SaltStrippingResult(Enum):
    """Result of removing salts, metals and additional substructures."""
    STRIPPED_MOLECULE = auto()
    EMPTY_MOLECULE = auto()


# Define salts not flagged by the ChEMBL structure pipeline
SALTS = ['[Na+]', '[K+]', 'Cl', 'C1=NC=CC=C1', '[O-][Cl+3]([O-])([O-])[O-]', 'O=CN(C)C', '[No]',
         'C', 'C1COCCO1', 'CCCCCCN(C)C', 'CCCCCC', 'CN(C)C(=N)N', 'CN(C)C',
         'c1ccc([B-](c2ccccc2)(c2ccccc2)c2ccccc2)cc1']

# Define metals as all but H, B, C, N, O, P, F, S, Cl, Br, I
METALS = ["He", "Li", "Be", "Ne", "Na", "Mg", "Al", "Si", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co",
          "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd",
          "Ag", "Cd", "In", "Sn", "Sb", "Te", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb",
          "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi",
          "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md",
          "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"]
METALS = [f'[{metal}]' for metal in METALS]

# Define organic atoms
ORGANIC_ATOMS = ['C', 'H', 'O', 'N', 'P', 'S', 'F', 'Cl', 'I', 'Br']


# Compile immutable chemistry rules once. Each call gets its own enumerator
# so timeout workers cannot share mutable enumeration state.
_tautomer_enumerator_template = TautomerEnumerator()

# ChEMBL restandardization after tautomer is a no-op unless one of these
# residual features remains. Keep the SMARTS aligned with chembl normalize_mol.
_chembl_rest_patterns = tuple(pattern for pattern in (
    Chem.MolFromSmarts("[N;X3](=[O])=[O]"),
    Chem.MolFromSmarts("[N;X2]#[N;X1]"),
    Chem.MolFromSmarts("[N;X4;v4;+0]"),
    Chem.MolFromSmarts("[O;X2;v3;+0]"),
    Chem.MolFromSmarts("[S+0;D3](=[O])"),
    Chem.MolFromSmarts("[SH1+1;D3](=[O])"),
    Chem.MolFromSmarts("[O]=[S;D2;+0]"),
    Chem.MolFromSmarts("[C]([OH1;D1])=;!@[NH1]"),
    Chem.MolFromSmarts("[C]([OH1;D1])=;!@[NH0]"),
    Chem.MolFromSmarts("[F,Cl,Br,I;X0;+0]"),
    Chem.MolFromSmarts("[C,N;-;D2,D3]-[N+2;D3]-[O-;D1]"),
    Chem.MolFromSmarts("[N-][N+]#[N]"),
    Chem.MolFromSmarts("[Li,Na,K;+0]-[#7,#8;+0]"),
    Chem.MolFromSmarts("OC(=O)C(O)C(O)C(=O)O"),
) if pattern is not None)


def _chembl_may_change(mol: Chem.Mol) -> bool:
    """True when a ChEMBL parent/standardize pass can still change mol.

    Explicit hydrogens are not a trigger. Tautomer products set NoImplicit on
    atoms that gained or lost H, so GetNumExplicitHs() is true on many
    molecules whose ChEMBL parent/standardize result matches later
    RemoveAllHs export. Charge, isotope, fragments, conformers, and the
    residual SMARTS catalog still run parent/standardize.
    """
    if mol is None:
        return True
    if mol.GetNumConformers():
        return True
    if len(Chem.GetMolFrags(mol)) != 1:
        return True
    for atom in mol.GetAtoms():
        if atom.GetIsotope() or atom.GetFormalCharge():
            return True
    return any(mol.HasSubstructMatch(pattern) for pattern in _chembl_rest_patterns)
_neutralizing_smiles = [  # Imidazoles
    ('[n+;H]', 'n'),
    # Amines
    ('[N+;!H0]', 'N'),
    # Carboxylic acids and alcohols
    ('[$([O-]);!$([O-][#7])]', 'O'),
    # Thiols
    ('[S-;X1]', 'S'),
    # Sulfonamides
    ('[$([N-;X2]S(=O)=O)]', 'N'),
    # Enamines
    ('[$([N-;X2][C,N]=C)]', 'N'),
    # Tetrazoles
    ('[n-]', '[nH]'),
    # Sulfoxides
    ('[$([S-]=O)]', 'S'),
    # Amides
    ('[$([N-]C=O)]', 'N')
]
_neutralizing_reactions = [(Chem.MolFromSmarts(x), Chem.MolFromSmiles(y, False)) for x, y in _neutralizing_smiles]


def _compile_salt_patterns(smiles):
    patterns = {}
    for value in smiles:
        salt = Chem.MolFromSmiles(value)
        size = salt.GetNumAtoms(), salt.GetNumBonds()
        patterns.setdefault(size, []).append(salt)
    return {size: tuple(group) for size, group in patterns.items()}


# These are fixed chemical rules, never results for an input molecule. Keep
# custom SALTS/METALS working by compiling any changed definitions per call.
_supplementary_salt_definitions = {False: tuple(SALTS), True: tuple(SALTS + METALS)}
_supplementary_salt_patterns = {
    metals: _compile_salt_patterns(values)
    for metals, values in _supplementary_salt_definitions.items()
}


def standardize(mol: Chem.Mol,
                remove_additional_salts: bool = True,
                remove_additional_metals: bool = True,
                filter_mixtures: bool = True,
                filter_inorganic: bool = True,
                uncharge: bool = True,
                filter_non_small_molecule: bool = True,
                canonicalize_tautomer: bool = True,
                small_molecule_min_mw: float = 200,
                small_molecule_max_mw: float = 800,
                tautomer_allow_stereo_removal: bool = True,
                tautomer_max_tautomers: int = 2 ** 32 - 1,
                return_type: bool = False,
                raise_error: bool = True
                ) -> Union[Chem.Mol, Tuple[Optional[Chem.Mol], StandardizationResult]]:
    """Standardize a molecule from either a RDKit molecule or SMILES.

    Steps:
    1) Use the ChEMBL structure pipeline (get parent molecule & standardization)
    2) remove supplementary salts & metals
    3) check if a mixture
    4) check if organic
    5) uncharge
    5) check if 200 <= MW <= 800
    6) canonicalize tautomer
    7) Use the ChEMBL structure pipeline (get parent molecule & standardization)

    :param mol: RDKit molecule to be standardized
    :param remove_additional_salts: should additional salts then those
    dealt with in the ChEMBL Structure Pipeline be removed
    :param remove_additional_metals: should salt removal also remove metal atoms
    :param filter_mixtures: should mixtures be filtered out
    :param filter_inorganic: should inorganic molecules be filtered out
    :param uncharge: should formal charges be removed (based on a list of substructures)
    :param filter_non_small_molecule: should non-small molecules be filtered out
    :param canonicalize_tautomer: should a canonical tautomer be determined
    :param small_molecule_min_mw: minimum molecular weight to be a valid small molecule
    :param small_molecule_max_mw: maximum molecular weight to be a valid small molecule
    :param tautomer_allow_stereo_removal: allow stereocenters to be removed by tautomerization
    :param tautomer_max_tautomers: maximum number of tautomers to enumerate (ignored if 0)
    :param return_type: If True, include the `StandardizationResult` in the return value
    :param raise_error: If True, standardization failures raise exceptions, if False, None is returned
    (with the standardization error if `return_type` is set).
    :return: a tuple of either the standardized molecule or None and a standardization_result flag
    """
    # Apply ChEMBL standardization
    try:
        # First ChEMBL parent/standardize is a no-op when residual features
        # are absent. Use the same gate as the second pass.
        if _chembl_may_change(mol):
            std_mol = _apply_chembl_standardization(mol)
        else:
            std_mol = mol
        # Remove additional salts
        if remove_additional_salts:
            std_mol = _remove_supplementary_salts(std_mol, include_metals=remove_additional_metals)
        # Ensure is not a mixture
        if filter_mixtures and is_mixture(std_mol):
            return (None, StandardizationResult.MIXTURE_MOLECULE) if return_type else None
        # Ensure organic
        if filter_inorganic and not is_organic(std_mol):
            return (None, StandardizationResult.INORGANIC_MOLECULE) if return_type else None
        # Remove charges
        if uncharge:
            std_mol = _uncharge(std_mol)
        # Ensure small molecule
        if filter_non_small_molecule and not is_small_molecule(std_mol,
                                                               min_molwt=small_molecule_min_mw,
                                                               max_molwt=small_molecule_max_mw):
            return (None, StandardizationResult.NON_SMALL_MOLECULE) if return_type else None
        # Obtain canonical tautomer
        if canonicalize_tautomer:
            std_mol = _canonicalize_tautomer(std_mol,
                                             allow_stereo_removal=tautomer_allow_stereo_removal,
                                             max_tautomers=tautomer_max_tautomers)
        # Second ChEMBL pass repairs tautomer products such as sulfoxide and
        # hydroxamic acid. Skip it when those residual features are absent.
        if _chembl_may_change(std_mol):
            final_mol = _apply_chembl_standardization(std_mol)
        else:
            final_mol = std_mol
        return (final_mol, StandardizationResult.CORRECT_MOLECULE) if return_type else final_mol
    except Exception as e:
        if raise_error:
            raise RuntimeError('Molecule could not be standardized') from e
        elif return_type:
            return (None, StandardizationResult.STANDARDIZATION_ERROR)
        else:
            return None


def _apply_chembl_standardization(mol: Chem.Mol) -> Chem.Mol:
    """Apply the ChEMBL structure standardization pipeline to a RDKit molecule.

    :param mol: RDKit molecule to be standardized
    """
    if mol is None:
        raise ValueError('Either RDKit molecule or SMILES must be specified')
    # Use the ChEMBL structure pipeline to standardize the molecule.
    # Return that mol directly. The original SMILES round-trip is a no-op for
    # the serialized Papyrus result; downstream export still canonicalizes.
    block = BlockLogs()  # Disable RDKit outputs
    standardized_mol, _ = standardizer.get_parent_mol(standardizer.standardize_mol(mol))
    del block  # Re-enable them
    if standardized_mol is None:
        raise ValueError('Could not apply ChEMBL structure standardization')
    return standardized_mol


def _canonicalize_tautomer(mol: Chem.Mol, allow_stereo_removal: bool = True,
                           max_tautomers: int = 2 ** 32 - 1) -> Chem.Mol:
    """Obtain the RDKit canonical tautomer of the given RDKit molecule.

    Contract ``rdkit-2024.03.3-enumpick-max100-v1``: native copied 2024
    enumerate+pickCanonical with ``maxTransforms=100``. Cap is not
    ``SetMaxTautomers(100)``. ``maxTautomers`` stays ``2**32-1`` unless the
    caller overrides it.
    """
    if mol is None:
        raise ValueError('A RDKit molecule must be specified')
    block = BlockLogs()  # Disable RDKit outputs
    if allow_stereo_removal:
        # Same truncated BFS as RDKit 2024.03.3 Enumerate+PickCanonical, with
        # per-molecule findSSSR reused across tautomer products.
        pickled, _status = _native_pick_canonical_tautomer(
            bytes(mol.ToBinary()), int(max_tautomers), True, True)
        del block
        can_tauto = Chem.Mol(pickled)
        if can_tauto is not None:
            return can_tauto
        raise ValueError(f'Could not obtain canonical tautomer: {Chem.MolToSmiles(mol)}')
    # Stereo-preserving fallback: same maxTransforms=100 contract, existing chiral filter.
    enumerator = TautomerEnumerator(_tautomer_enumerator_template)
    enumerator.SetMaxTautomers(max_tautomers)
    enumerator.SetMaxTransforms(TAUTOMER_MAX_TRANSFORMS)
    enumerator.SetRemoveSp3Stereo(False)
    enumerator.SetRemoveBondStereo(False)
    tautos = enumerator.Enumerate(mol)
    def get_num_ciral_centers(mol_: Chem.Mol):
        """Utility function."""
        return len([atom.GetChiralTag()
                    for atom in mol_.GetAtoms()
                    if atom.GetChiralTag() != AllChem.ChiralType.CHI_UNSPECIFIED])

    orig_chiral_centers = get_num_ciral_centers(mol)
    if orig_chiral_centers > 0:
        tautos = [tauto for tauto in tautos
                  if get_num_ciral_centers(tauto) == orig_chiral_centers]
    can_tauto = enumerator.PickCanonical(tautos)
    del block  # Re-enable them
    if can_tauto is not None:
        return can_tauto
    raise ValueError(f'Could not obtain canonical tautomer: {Chem.MolToSmiles(mol)}')


def is_organic(mol: Chem.Mol, return_type: bool = False) -> Union[bool, Tuple[bool, InorganicSubtype]]:
    """Return whether the RKDit molecule is organic or not.

    Makes use of the `ORGANIC_ATOMS` variable to identify inorganic atoms.

    :param mol: RDKit molecule
    :param return_type: If True, include the `InorganicSubtype` in the return value
    :return: True if return_type is True and the molecule has no exclusion flag
    set by the `chembl_structure_pipeline`, has at least one C-C bond and is only
    made of C, H, O, N, P, S and halogen atoms.
    """
    if mol is None:
        raise ValueError('A RDKit molecule must be specified')
    # Obtain ChEMBL structure exclusion flag
    if exclude_flag(Chem.MolToMolBlock(mol)):
        return False if not return_type else (False, InorganicSubtype.EXCLUSION_FLAG_SET)
    # Ensure at least 1 C-C bond
    ccbond = Chem.MolFromSmarts('[#6]~[#6]')
    if len(mol.GetSubstructMatches(ccbond)) == 0:
        return False if not return_type else (False, InorganicSubtype.NO_CC_BOND)
    # Ensure contains only C, H, O, N, P, S and halogens
    global ORGANIC_ATOMS
    if len(set(atom.GetSymbol() for atom in mol.GetAtoms()).difference(set(ORGANIC_ATOMS))):
        return False if not return_type else (False, InorganicSubtype.NOT_CHONPSFClIBrB)
    return True if not return_type else (True, InorganicSubtype.IS_ORGANIC)


def is_small_molecule(mol: Chem.Mol,
                      min_molwt: float = 200,
                      max_molwt: float = 800,
                      ) -> bool:
    """Return whether the RDKit molecule is a 'small molecule'.

    :param mol: molecule to examine
    :param min_molwt: molecular weight under which the molecule is not considered a small molecule
    :param max_molwt: molecular weight above which the molecule is not considered a small molecule
    """
    if mol is None:
        raise ValueError('A RDKit molecule must be specified')
    return min_molwt <= AllChem.CalcExactMolWt(mol) <= max_molwt


def is_mixture(mol: Chem.Mol) -> bool:
    """Return whether the molecule is composed of multiple fragments."""
    if mol is None:
        raise ValueError('A RDKit molecule must be specified')
    return len(Chem.GetMolFrags(mol)) > 1


def _remove_supplementary_salts(mol: Chem.Mol,
                                include_metals: bool = True,
                                return_type: bool = False
                                ) -> Union[Chem.Mol, Tuple[Chem.Mol, SaltStrippingResult]]:
    """Remove salts not dealt with by the ChEMBL pipeline structure.

    Adapted from the ChEMBL structure pipeline.
    https://github.com/chembl/ChEMBL_Structure_Pipeline.git

    Makes use of the `SALTS` and `METALS` variables to identify substructures to remove.

    :param include_metals: should the variable `METALS` be used toremove additional substructures.
    :param return_type: If True, include the `SaltStrippingResult` in the return value
    :return: the stripped molecule
    """
    if mol is None:
        raise ValueError('A RDKit molecule must be specified')
    definitions = tuple(SALTS) + (tuple(METALS) if include_metals else ())
    if definitions == _supplementary_salt_definitions[bool(include_metals)]:
        salts = _supplementary_salt_patterns[bool(include_metals)]
    else:
        salts = _compile_salt_patterns(definitions)
    # Obtain fragments as molecules
    frags = list(Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False))
    # Sanitize fragments
    for i, _ in enumerate(frags):
        frag = Chem.RemoveHs(frags[i], sanitize=False)
        frag.UpdatePropertyCache(strict=False)
        Chem.SetAromaticity(frag)
        frags[i] = frag
    # Flag fragments to keep
    keep = [1] * len(frags)
    for i, frag in enumerate(frags):
        size = frag.GetNumAtoms(), frag.GetNumBonds()
        for salt in salts.get(size, ()):
            if (keep[i] and frag.HasSubstructMatch(salt)
                    and salt.HasSubstructMatch(frag)):
                keep[i] = 0
            if not max(keep):
                # All was removed, return initial molecule
                return Chem.Mol(mol) if not return_type else (Chem.Mol(mol), SaltStrippingResult.EMPTY_MOLECULE)
    # Collect fragments to keep
    frags = [frags[i] for i, x in enumerate(keep) if x == 1]
    # Only one fragment left
    if len(frags) == 1:
        return frags[0] if not return_type else (frags[0], SaltStrippingResult.STRIPPED_MOLECULE)
    # Otherwise combine remaining fragments
    else:
        res = frags[0]
        for frag in frags[1:]:
            res = Chem.CombineMols(res, frag)
        return res if not return_type else (res, SaltStrippingResult.STRIPPED_MOLECULE)


def _uncharge(mol: Chem.Mol) -> Chem.Mol:
    """Uncharge the given RDKit molecule.

    Adapted from Hans de Winter's contribution to B3DB
    (https://github.com/theochem/B3DB/blob/main/cleaning/02_clean_smiles_chembl_way_20210215.py)

    Neutral molecules skip Uncharger, the zwitterion SMARTS pass, and the
    trailing SanitizeMol. Those steps are no-ops when no atom has a formal
    charge; the neutralizing patterns all require +/−.
    """
    if mol is None:
        raise ValueError('A RDKit molecule must be specified')
    if not any(atom.GetFormalCharge() for atom in mol.GetAtoms()):
        return mol
    block = BlockLogs()
    uncharger = Uncharger()
    mol = uncharger.uncharge(mol)
    # Do a second pass on zwitterions
    romol = Chem.RWMol(mol)
    for i, (reactant, product) in enumerate(_neutralizing_reactions):
        while romol.HasSubstructMatch(reactant):
            rms = AllChem.ReplaceSubstructs(romol, reactant, product)
            romol = rms[0]
    mol = Chem.Mol(romol)
    Chem.SanitizeMol(mol)
    del block
    return mol
