"""Atom and bond features with the original PSICHIC values and tensor layout."""

from __future__ import annotations

from collections import OrderedDict
from itertools import accumulate
from typing import Any

import numpy as np
import torch
from rdkit import Chem

from .native import from_rdmol as _native_from_rdmol


x_map = OrderedDict(
    [
        (
            "chirality",
            [
                "CHI_UNSPECIFIED",
                "CHI_TETRAHEDRAL_CW",
                "CHI_TETRAHEDRAL_CCW",
                "CHI_OTHER",
                "CHI_TETRAHEDRAL",
                "CHI_ALLENE",
                "CHI_SQUAREPLANAR",
                "CHI_TRIGONALBIPYRAMIDAL",
                "CHI_OCTAHEDRAL",
            ],
        ),
        ("degree", list(range(0, 11))),
        ("formal_charge", list(range(-5, 10))),
        ("num_hs", list(range(0, 9))),
        ("num_radical_electrons", list(range(0, 5))),
        (
            "hybridization",
            [
                "UNSPECIFIED",
                "S",
                "SP",
                "SP2",
                "SP3",
                "SP3D",
                "SP3D2",
                "OTHER",
            ],
        ),
        ("is_aromatic", [False, True]),
        ("is_in_ring", [False, True]),
    ]
)

e_map = OrderedDict(
    [
        (
            "bond_type",
            [
                "UNSPECIFIED",
                "SINGLE",
                "DOUBLE",
                "TRIPLE",
                "QUADRUPLE",
                "QUINTUPLE",
                "HEXTUPLE",
                "ONEANDAHALF",
                "TWOANDAHALF",
                "THREEANDAHALF",
                "FOURANDAHALF",
                "FIVEANDAHALF",
                "AROMATIC",
                "IONIC",
                "HYDROGEN",
                "THREECENTER",
                "DATIVEONE",
                "DATIVE",
                "DATIVEL",
                "DATIVER",
                "OTHER",
                "ZERO",
            ],
        ),
        ("stereo", ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE", "STEREOCIS", "STEREOTRANS"]),
        ("is_conjugated", [False, True]),
    ]
)

metals = [3, 4, 11, 12, 13] + list(range(19, 32)) + list(range(37, 51)) + list(range(55, 84)) + list(range(87, 104))
atom_classes = [
    (5, "B"),
    (6, "C"),
    (7, "N"),
    (8, "O"),
    (15, "P"),
    (16, "S"),
    (34, "Se"),
    ([9, 17, 35, 53], "halogen"),
    (metals, "metal"),
]
ATOM_CODES = {}
for code, (atom, _) in enumerate(atom_classes):
    if isinstance(atom, list):
        for atomic_num in atom:
            ATOM_CODES[atomic_num] = code
    else:
        ATOM_CODES[atom] = code

x_map_length = {key: len(value) for key, value in x_map.items()}
e_map_length = {key: len(value) for key, value in e_map.items()}


def map_atom_code(atomic_num):
    try:
        return ATOM_CODES[atomic_num] + 1
    except KeyError:
        return 0


def one_hot_encode_features(features, feature_lengths):
    widths = list(feature_lengths.values())[:features.shape[1]]
    offsets = [0, *accumulate(widths[:-1])]
    indices = features + features.new_tensor(offsets)
    # Each categorical column owns a disjoint block in a fresh output tensor.
    # This avoids allocating and indexing an identity matrix for every block.
    encoded = torch.zeros((features.shape[0], sum(widths)))
    return encoded.scatter_(1, indices, 1)


def _long_tensor(array, shape=None):
    array = np.ascontiguousarray(array, dtype=np.int64)
    if array.size == 0:
        if shape is None:
            return torch.empty(tuple(int(dim) for dim in array.shape), dtype=torch.long)
        return torch.empty(0, dtype=torch.long).view(*shape)
    tensor = torch.from_numpy(array)
    if shape is not None:
        tensor = tensor.view(*shape)
    return tensor


def from_rdmol(mol: Any, one_hot=True):
    assert isinstance(mol, Chem.Mol)
    arrays = _native_from_rdmol(bytes(mol.ToBinary()))
    atom_ids_tensor = _long_tensor(arrays["atom_ids"], (-1, 1))
    atomic_nums_tensor = _long_tensor(arrays["atomic_nums"], (-1, 1))
    x = _long_tensor(arrays["x"], (-1, 8))
    edge_index = _long_tensor(arrays["edge_index"])
    edge_attr = _long_tensor(arrays["edge_attr"], (-1, 3))
    if one_hot:
        x = one_hot_encode_features(x, x_map_length)
        edge_attr = one_hot_encode_features(edge_attr, e_map_length)
    else:
        x = x.float()
        edge_attr = edge_attr.float()
    return atom_ids_tensor, atomic_nums_tensor, x, edge_index, edge_attr
