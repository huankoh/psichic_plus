"""Runtime graph featurizers and PyG data plumbing for PSICHIC+ inference."""

from __future__ import annotations

import gc
import hashlib
import logging
import math
import os
import pickle
import tempfile
import threading
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import chain
from typing import Any, Callable, List, Optional, Sequence, Tuple, TypeVar, Union

import esm
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, PropertyPickleOptions
from scipy.sparse.csgraph import minimum_spanning_tree
from torch import Tensor
from torch.utils.data.dataloader import default_collate
from torch_geometric.data import Batch, Data, Dataset
from torch_geometric.data.data import BaseData
from torch_geometric.data.datapipes import DatasetAdapter
from torch_geometric.typing import TensorFrame, torch_frame
from torch_geometric.utils import (
    add_self_loops,
    coalesce,
    degree,
    from_scipy_sparse_matrix,
    is_torch_sparse_tensor,
    remove_self_loops,
    scatter,
    to_edge_index,
    to_scipy_sparse_matrix,
    to_undirected,
)
from tqdm import tqdm

from papyrus_structure_pipeline import standardizer as Papyrus_standardizer

T = TypeVar("T")


class TimeoutHandler:
    def __init__(self, timeout_seconds: int = 10):
        self.timeout_seconds = timeout_seconds
        self._result: Optional[Any] = None
        self._exception: Optional[Exception] = None

    def run_with_timeout(self, func: Callable[..., T], *args, **kwargs) -> Optional[T]:
        def worker():
            try:
                self._result = func(*args, **kwargs)
            except Exception as exc:  # pragma: no cover - propagated below
                self._exception = exc

        thread = threading.Thread(target=worker)
        thread.daemon = True
        thread.start()
        thread.join(timeout=self.timeout_seconds)

        if thread.is_alive():
            return None
        if self._exception is not None:
            raise self._exception
        return self._result


@dataclass
class NormalizationResult:
    smiles: Optional[str]
    status: str
    error_message: Optional[str] = None


@dataclass
class LigandFailureDiagnosis:
    valid: bool
    failure_stage: Optional[str] = None
    failure_reason: Optional[str] = None


class MoleculeNormalizer:
    def __init__(self, timeout_seconds: int = 10):
        self.timeout_handler = TimeoutHandler(timeout_seconds)

    def _standardize_molecule(self, mol):
        return Papyrus_standardizer.standardize(
            mol,
            filter_inorganic=False,
            small_molecule_min_mw=100,
            small_molecule_max_mw=2000,
            return_type=True,
        )

    def standardize_normalize(self, smiles: str) -> NormalizationResult:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return NormalizationResult(None, "INVALID_SMILES", "Could not parse SMILES")

            result = self.timeout_handler.run_with_timeout(self._standardize_molecule, mol)
            if result is None:
                return NormalizationResult(None, "TIMEOUT", "Standardization timeout")

            out_type = result[1].value
            status_mapping = {
                1: ("SUCCESS", None),
                2: ("NON_SMALL_MOLECULE", "Molecular weight outside allowed range"),
                3: ("INORGANIC_MOLECULE", "Molecule contains non-organic elements"),
                4: ("MIXTURE", "Multiple fragments detected"),
                5: ("STANDARDIZATION_ERROR", "General standardization error"),
            }
            if out_type != 1:
                status, message = status_mapping.get(
                    out_type,
                    ("UNKNOWN_ERROR", f"Unknown standardization result type: {out_type}"),
                )
                return NormalizationResult(None, status, message)

            mol = result[0]
            mol = Chem.RemoveHs(mol, sanitize=True)
            mol = AllChem.RemoveAllHs(mol)
            normalized_smiles = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
            return NormalizationResult(normalized_smiles, "SUCCESS", None)
        except Exception as exc:
            return NormalizationResult(None, "GENERAL_ERROR", str(exc))


def diagnose_ligand_failure(smiles: Any, timeout_seconds: int = 10) -> LigandFailureDiagnosis:
    if smiles is None or pd.isna(smiles):
        return LigandFailureDiagnosis(False, "ligand_input", "EMPTY_LIGAND: Ligand value is empty")

    smiles_text = str(smiles).strip()
    if not smiles_text:
        return LigandFailureDiagnosis(False, "ligand_input", "EMPTY_LIGAND: Ligand value is empty")

    normalizer = MoleculeNormalizer(timeout_seconds=timeout_seconds)
    result = normalizer.standardize_normalize(smiles_text)
    if result.smiles is None:
        message = result.error_message or "Ligand could not be standardized"
        return LigandFailureDiagnosis(
            False,
            "ligand_standardization",
            f"{result.status}: {message}",
        )

    try:
        graph = smiles_to_graph(result.smiles)
    except Exception as exc:
        return LigandFailureDiagnosis(
            False,
            "ligand_graph",
            f"GRAPH_CONSTRUCTION_ERROR: {exc}",
        )

    if graph is None:
        return LigandFailureDiagnosis(
            False,
            "ligand_graph",
            "GRAPH_CONSTRUCTION_ERROR: Molecule graph is None",
        )
    return LigandFailureDiagnosis(True)


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
    one_hot_encoded = []
    for i in range(features.shape[1]):
        feature_column = features[:, i]
        num_classes = feature_lengths[list(feature_lengths.keys())[i]]
        one_hot_encoded.append(torch.eye(num_classes)[feature_column])
    return torch.concat(one_hot_encoded, dim=1)


def from_rdmol(mol: Any, one_hot=True):
    assert isinstance(mol, Chem.Mol)

    xs: List[List[int]] = []
    atom_ids: List[int] = []
    atomic_nums: List[int] = []

    for atom in mol.GetAtoms():
        atomic_num = atom.GetAtomicNum()
        atom_ids.append(map_atom_code(atomic_num))
        atomic_nums.append(atomic_num)
        xs.append(
            [
                x_map["chirality"].index(str(atom.GetChiralTag())) if str(atom.GetChiralTag()) in x_map["chirality"] else 0,
                x_map["degree"].index(atom.GetTotalDegree()) if atom.GetTotalDegree() in x_map["degree"] else len(x_map["degree"]) - 1,
                x_map["formal_charge"].index(atom.GetFormalCharge())
                if atom.GetFormalCharge() in x_map["formal_charge"]
                else len(x_map["formal_charge"]) - 1,
                x_map["num_hs"].index(atom.GetTotalNumHs()) if atom.GetTotalNumHs() in x_map["num_hs"] else len(x_map["num_hs"]) - 1,
                x_map["num_radical_electrons"].index(atom.GetNumRadicalElectrons())
                if atom.GetNumRadicalElectrons() in x_map["num_radical_electrons"]
                else len(x_map["num_radical_electrons"]) - 1,
                x_map["hybridization"].index(str(atom.GetHybridization())) if str(atom.GetHybridization()) in x_map["hybridization"] else 0,
                x_map["is_aromatic"].index(atom.GetIsAromatic()),
                x_map["is_in_ring"].index(atom.IsInRing()),
            ]
        )

    atom_ids_tensor = torch.tensor(atom_ids, dtype=torch.long).view(-1, 1)
    atomic_nums_tensor = torch.tensor(atomic_nums, dtype=torch.long).view(-1, 1)
    x = torch.tensor(xs, dtype=torch.long).view(-1, 8)

    edge_indices, edge_attrs = [], []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_attr = [
            e_map["bond_type"].index(str(bond.GetBondType())),
            e_map["stereo"].index(str(bond.GetStereo())),
            e_map["is_conjugated"].index(bond.GetIsConjugated()),
        ]
        edge_indices += [[i, j], [j, i]]
        edge_attrs += [edge_attr, edge_attr]

    edge_index = torch.tensor(edge_indices).t().to(torch.long).view(2, -1)
    edge_attr = torch.tensor(edge_attrs, dtype=torch.long).view(-1, 3)
    if edge_index.numel() > 0:
        perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
        edge_index, edge_attr = edge_index[:, perm], edge_attr[perm]

    if one_hot:
        x = one_hot_encode_features(x, x_map_length)
        edge_attr = one_hot_encode_features(edge_attr, e_map_length)
    else:
        x = x.float()
        edge_attr = edge_attr.float()
    return atom_ids_tensor, atomic_nums_tensor, x, edge_index, edge_attr


def tree_decomposition(mol: Any, return_vocab: bool = False) -> Union[Tuple[Tensor, Tensor, int], Tuple[Tensor, Tensor, int, Tensor]]:
    cliques = [list(x) for x in Chem.GetSymmSSSR(mol)]
    xs = [0] * len(cliques)
    for bond in mol.GetBonds():
        if not bond.IsInRing():
            cliques.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
            xs.append(1)

    atom2clique = [[] for _ in range(mol.GetNumAtoms())]
    for c, clique in enumerate(cliques):
        for atom in clique:
            atom2clique[atom].append(c)

    for c1 in range(len(cliques)):
        for atom in cliques[c1]:
            for c2 in atom2clique[atom]:
                if c1 >= c2 or len(cliques[c1]) <= 2 or len(cliques[c2]) <= 2:
                    continue
                if len(set(cliques[c1]) & set(cliques[c2])) > 2:
                    cliques[c1] = set(cliques[c1]) | set(cliques[c2])
                    xs[c1] = 2
                    cliques[c2] = []
                    xs[c2] = -1
    cliques = [c for c in cliques if len(c) > 0]
    xs = [x for x in xs if x >= 0]

    atom2clique = [[] for _ in range(mol.GetNumAtoms())]
    for c, clique in enumerate(cliques):
        for atom in clique:
            atom2clique[atom].append(c)

    edges = {}
    for atom in range(mol.GetNumAtoms()):
        cs = atom2clique[atom]
        if len(cs) <= 1:
            continue
        bonds = [c for c in cs if len(cliques[c]) == 2]
        rings = [c for c in cs if len(cliques[c]) > 4]
        if len(bonds) > 2 or (len(bonds) == 2 and len(cs) > 2):
            cliques.append([atom])
            xs.append(3)
            c2 = len(cliques) - 1
            for c1 in cs:
                edges[(c1, c2)] = 1
        elif len(rings) > 2:
            cliques.append([atom])
            xs.append(3)
            c2 = len(cliques) - 1
            for c1 in cs:
                edges[(c1, c2)] = 99
        else:
            for i in range(len(cs)):
                for j in range(i + 1, len(cs)):
                    c1, c2 = cs[i], cs[j]
                    count = len(set(cliques[c1]) & set(cliques[c2]))
                    edges[(c1, c2)] = min(count, edges.get((c1, c2), 99))

    atom2clique = [[] for _ in range(mol.GetNumAtoms())]
    for c, clique in enumerate(cliques):
        for atom in clique:
            atom2clique[atom].append(c)

    if len(edges) > 0:
        edge_index_t, weight = zip(*edges.items())
        edge_index = torch.tensor(edge_index_t).t()
        inv_weight = 100 - torch.tensor(weight)
        graph = to_scipy_sparse_matrix(edge_index, inv_weight, len(cliques))
        junction_tree = minimum_spanning_tree(graph)
        edge_index, _ = from_scipy_sparse_matrix(junction_tree)
        edge_index = to_undirected(edge_index, num_nodes=len(cliques))
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    rows = [[i] * len(atom2clique[i]) for i in range(mol.GetNumAtoms())]
    row = torch.tensor(list(chain.from_iterable(rows)))
    col = torch.tensor(list(chain.from_iterable(atom2clique)))
    atom2clique_tensor = torch.stack([row, col], dim=0).to(torch.long)

    if return_vocab:
        return edge_index, atom2clique_tensor, len(cliques), torch.tensor(xs, dtype=torch.long)
    return edge_index, atom2clique_tensor, len(cliques)


def add_rw_pe(edge_index, num_nodes, edge_weight=None, walk_length=20, device="cpu"):
    num_edges = edge_index.size(1)
    edge_index = edge_index.to(device)
    row, col = edge_index
    if edge_weight is None:
        value = torch.ones(num_edges, device=row.device)
    else:
        value = edge_weight

    value = torch.ones(num_edges, device=device)
    value = scatter(value, row, dim_size=num_nodes, reduce="sum").clamp(min=1)[row]
    value = 1.0 / value

    adj = torch.zeros((num_nodes, num_nodes), device=device)
    adj[row, col] = value
    loop_index = torch.arange(num_nodes, device=device)

    def get_pe(out: Tensor) -> Tensor:
        if is_torch_sparse_tensor(out):
            return get_self_loop_attr(*to_edge_index(out), num_nodes=num_nodes)
        return out[loop_index, loop_index]

    out = adj
    pe_list = [get_pe(out)]
    for _ in range(walk_length - 1):
        out = out @ adj
        pe_list.append(get_pe(out))
    return torch.stack(pe_list, dim=-1)


def get_self_loop_attr(edge_index, edge_attr, num_nodes):
    loop_mask = edge_index[0] == edge_index[1]
    loop_index = edge_index[0][loop_mask]
    loop_attr = edge_attr[loop_mask] if edge_attr is not None else torch.ones_like(loop_index, dtype=torch.float)
    full_loop_attr = loop_attr.new_zeros((num_nodes,) + loop_attr.size()[1:])
    full_loop_attr[loop_index] = loop_attr
    return full_loop_attr


def smiles_to_graph(smiles, one_hot=True):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    atom_ids, atomic_nums, atom_feat, edge_index, edge_attr = from_rdmol(mol, one_hot)
    tree_edge_index, atom2clique_index, num_cliques, x_clique = tree_decomposition(mol, return_vocab=True)
    if atom2clique_index.nelement() == 0:
        num_cliques = len(mol.GetAtoms())
        x_clique = torch.tensor([3] * num_cliques)
        atom2clique_index = torch.stack([torch.arange(num_cliques), torch.arange(num_cliques)])

    graph_dict = {
        "smiles": smiles,
        "atom_idx": atom_ids,
        "atom_types": atomic_nums,
        "atom_feature": atom_feat,
        "atom_edge_index": edge_index,
        "atom_edge_attr": edge_attr,
        "atom_num_nodes": atom_ids.shape[0],
        "tree_edge_index": tree_edge_index.long(),
        "atom2clique_index": atom2clique_index.long(),
        "x_clique": x_clique.long().view(-1, 1),
        "clique_num_nodes": num_cliques,
    }
    if num_cliques == 1:
        graph_dict["clique_pe"] = torch.zeros(1, 20).float()
    else:
        graph_dict["clique_pe"] = add_rw_pe(graph_dict["tree_edge_index"], num_cliques, edge_weight=None, walk_length=20, device="cpu")
    return graph_dict


def safe_smiles_filename(smiles):
    return hashlib.sha256(smiles.encode()).hexdigest()[:32]


def process_molecule_file(smiles, ligand_path):
    mol_filename = safe_smiles_filename(smiles)
    mol_dict_path = os.path.join(ligand_path, f"{mol_filename}.pt")
    if not os.path.exists(mol_dict_path):
        mol_graph = smiles_to_graph(smiles)
        if mol_graph is None:
            raise Exception("Molecule is None")
        torch.save(mol_graph, mol_dict_path)
    else:
        mol_graph = torch.load(mol_dict_path, weights_only=False)
    return mol_graph


class Collater:
    def __init__(
        self,
        dataset: Union[Dataset, Sequence[BaseData], DatasetAdapter],
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
    ):
        self.dataset = dataset
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

    def __call__(self, batch: List[Any]) -> Any:
        batch = [elem for elem in batch if elem is not None]
        if len(batch) == 0:
            return None

        elem = batch[0]
        if isinstance(elem, BaseData):
            return Batch.from_data_list(batch, follow_batch=self.follow_batch, exclude_keys=self.exclude_keys)
        if isinstance(elem, torch.Tensor):
            return default_collate(batch)
        if isinstance(elem, TensorFrame):
            return torch_frame.cat(batch, dim=0)
        if isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float)
        if isinstance(elem, int):
            return torch.tensor(batch)
        if isinstance(elem, str):
            return batch
        if isinstance(elem, Mapping):
            return {key: self([data[key] for data in batch if data[key] is not None]) for key in elem}
        if isinstance(elem, tuple) and hasattr(elem, "_fields"):
            return type(elem)(*(self(s) for s in zip(*batch)))
        if isinstance(elem, Sequence) and not isinstance(elem, str):
            return [self(s) for s in zip(*batch)]
        raise TypeError(f"DataLoader found invalid type: '{type(elem)}'")


class DataLoader(torch.utils.data.DataLoader):
    def __init__(
        self,
        dataset: Union[Dataset, Sequence[BaseData], DatasetAdapter],
        batch_size: int = 1,
        shuffle: bool = False,
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        kwargs.pop("collate_fn", None)
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys
        batch_sampler = kwargs.pop("batch_sampler", None)
        if batch_sampler is not None:
            super().__init__(
                dataset,
                batch_sampler=batch_sampler,
                collate_fn=Collater(dataset, follow_batch, exclude_keys),
                **kwargs,
            )
        else:
            super().__init__(
                dataset,
                batch_size,
                shuffle,
                collate_fn=Collater(dataset, follow_batch, exclude_keys),
                **kwargs,
            )


def dic_normalize(dic):
    max_value = dic[max(dic, key=dic.get)]
    min_value = dic[min(dic, key=dic.get)]
    interval = float(max_value) - float(min_value)
    for key in dic.keys():
        dic[key] = (dic[key] - min_value) / interval
    dic["X"] = (max_value + min_value) / 2.0
    return dic


pro_res_table = ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y", "X"]
pro_res_aliphatic_table = ["A", "I", "L", "M", "V"]
pro_res_aromatic_table = ["F", "W", "Y"]
pro_res_polar_neutral_table = ["C", "N", "Q", "S", "T"]
pro_res_acidic_charged_table = ["D", "E"]
pro_res_basic_charged_table = ["H", "K", "R"]

res_weight_table = dic_normalize({"A": 71.08, "C": 103.15, "D": 115.09, "E": 129.12, "F": 147.18, "G": 57.05, "H": 137.14, "I": 113.16, "K": 128.18, "L": 113.16, "M": 131.20, "N": 114.11, "P": 97.12, "Q": 128.13, "R": 156.19, "S": 87.08, "T": 101.11, "V": 99.13, "W": 186.22, "Y": 163.18})
res_pka_table = dic_normalize({"A": 2.34, "C": 1.96, "D": 1.88, "E": 2.19, "F": 1.83, "G": 2.34, "H": 1.82, "I": 2.36, "K": 2.18, "L": 2.36, "M": 2.28, "N": 2.02, "P": 1.99, "Q": 2.17, "R": 2.17, "S": 2.21, "T": 2.09, "V": 2.32, "W": 2.83, "Y": 2.32})
res_pkb_table = dic_normalize({"A": 9.69, "C": 10.28, "D": 9.60, "E": 9.67, "F": 9.13, "G": 9.60, "H": 9.17, "I": 9.60, "K": 8.95, "L": 9.60, "M": 9.21, "N": 8.80, "P": 10.60, "Q": 9.13, "R": 9.04, "S": 9.15, "T": 9.10, "V": 9.62, "W": 9.39, "Y": 9.62})
res_pkx_table = dic_normalize({"A": 0.00, "C": 8.18, "D": 3.65, "E": 4.25, "F": 0.00, "G": 0, "H": 6.00, "I": 0.00, "K": 10.53, "L": 0.00, "M": 0.00, "N": 0.00, "P": 0.00, "Q": 0.00, "R": 12.48, "S": 0.00, "T": 0.00, "V": 0.00, "W": 0.00, "Y": 0.00})
res_pl_table = dic_normalize({"A": 6.00, "C": 5.07, "D": 2.77, "E": 3.22, "F": 5.48, "G": 5.97, "H": 7.59, "I": 6.02, "K": 9.74, "L": 5.98, "M": 5.74, "N": 5.41, "P": 6.30, "Q": 5.65, "R": 10.76, "S": 5.68, "T": 5.60, "V": 5.96, "W": 5.89, "Y": 5.96})
res_hydrophobic_ph2_table = dic_normalize({"A": 47, "C": 52, "D": -18, "E": 8, "F": 92, "G": 0, "H": -42, "I": 100, "K": -37, "L": 100, "M": 74, "N": -41, "P": -46, "Q": -18, "R": -26, "S": -7, "T": 13, "V": 79, "W": 84, "Y": 49})
res_hydrophobic_ph7_table = dic_normalize({"A": 41, "C": 49, "D": -55, "E": -31, "F": 100, "G": 0, "H": 8, "I": 99, "K": -23, "L": 97, "M": 74, "N": -28, "P": -46, "Q": -10, "R": -14, "S": -5, "T": 13, "V": 76, "W": 97, "Y": 63})


def replace_non_standard_residues(sequence, res_table):
    res_set = set(res_table)
    return "".join([res if res in res_set else "X" for res in sequence])


def residue_features(residue):
    res_property1 = [
        1 if residue in pro_res_aliphatic_table else 0,
        1 if residue in pro_res_aromatic_table else 0,
        1 if residue in pro_res_polar_neutral_table else 0,
        1 if residue in pro_res_acidic_charged_table else 0,
        1 if residue in pro_res_basic_charged_table else 0,
    ]
    res_property2 = [
        res_weight_table[residue],
        res_pka_table[residue],
        res_pkb_table[residue],
        res_pkx_table[residue],
        res_pl_table[residue],
        res_hydrophobic_ph2_table[residue],
        res_hydrophobic_ph7_table[residue],
    ]
    return np.array(res_property1 + res_property2)


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception(f"input {x} not in allowable set{allowable_set}:")
    return list(map(lambda s: x == s, allowable_set))


def seq_feature(pro_seq):
    if "U" in pro_seq or "B" in pro_seq:
        print("U or B in Sequence")
    pro_seq = pro_seq.replace("U", "X").replace("B", "X")
    pro_hot = np.zeros((len(pro_seq), len(pro_res_table)))
    pro_property = np.zeros((len(pro_seq), 12))
    for i in range(len(pro_seq)):
        pro_hot[i,] = one_of_k_encoding(pro_seq[i], pro_res_table)
        pro_property[i,] = residue_features(pro_seq[i])
    return np.concatenate((pro_hot, pro_property), axis=1)


def contact_map(contact_map_proba, contact_threshold=0.5):
    num_residues = contact_map_proba.shape[0]
    prot_contact_adj = (contact_map_proba >= contact_threshold).long()
    edge_index = prot_contact_adj.nonzero(as_tuple=False).t().contiguous()
    row, col = edge_index
    edge_weight = contact_map_proba[row, col].float()

    seq_edge_head1 = torch.stack([torch.arange(num_residues)[:-1], (torch.arange(num_residues) + 1)[:-1]])
    seq_edge_tail1 = torch.stack([(torch.arange(num_residues))[1:], (torch.arange(num_residues) - 1)[1:]])
    seq_edge_weight1 = torch.ones(seq_edge_head1.size(1) + seq_edge_tail1.size(1)) * contact_threshold
    edge_index = torch.cat([edge_index, seq_edge_head1, seq_edge_tail1], dim=-1)
    edge_weight = torch.cat([edge_weight, seq_edge_weight1], dim=-1)

    seq_edge_head2 = torch.stack([torch.arange(num_residues)[:-2], (torch.arange(num_residues) + 2)[:-2]])
    seq_edge_tail2 = torch.stack([(torch.arange(num_residues))[2:], (torch.arange(num_residues) - 2)[2:]])
    seq_edge_weight2 = torch.ones(seq_edge_head2.size(1) + seq_edge_tail2.size(1)) * contact_threshold
    edge_index = torch.cat([edge_index, seq_edge_head2, seq_edge_tail2], dim=-1)
    edge_weight = torch.cat([edge_weight, seq_edge_weight2], dim=-1)

    edge_index, edge_weight = coalesce(edge_index, edge_weight, reduce="max")
    edge_index, edge_weight = to_undirected(edge_index, edge_weight, reduce="max")
    edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
    edge_index, edge_weight = add_self_loops(edge_index, edge_weight, fill_value=1)
    return edge_index, edge_weight


def esm_extract(model, batch_converter, seq, layer=36, approach="mean", dim=2560):
    pro_id = "A"
    if len(seq) <= 700:
        _, _, batch_tokens = batch_converter([(pro_id, seq)])
        batch_tokens = batch_tokens.to(next(model.parameters()).device, non_blocking=True)
        with torch.inference_mode():
            results = model(batch_tokens, repr_layers=[i for i in range(1, layer + 1)], return_contacts=True)

        logits = results["logits"][0].cpu().numpy()[1 : len(seq) + 1]
        contact_prob_map = results["contacts"][0].cpu().numpy()
        token_representation = torch.cat([results["representations"][i] for i in range(1, layer + 1)])
        assert token_representation.size(0) == layer
        if approach == "last":
            token_representation = token_representation[-1]
        elif approach == "sum":
            token_representation = token_representation.sum(dim=0)
        elif approach == "mean":
            token_representation = token_representation.mean(dim=0)
        token_representation = token_representation.cpu().numpy()[1 : len(seq) + 1]
    else:
        contact_prob_map = np.zeros((len(seq), len(seq)))
        token_representation = np.zeros((len(seq), dim))
        logits = np.zeros((len(seq), layer))
        interval = 350
        for s in range(math.ceil(len(seq) / interval)):
            start = s * interval
            end = min((s + 2) * interval, len(seq))
            temp_seq = seq[start:end]
            _, _, batch_tokens = batch_converter([(pro_id, temp_seq)])
            batch_tokens = batch_tokens.to(next(model.parameters()).device, non_blocking=True)
            with torch.inference_mode():
                results = model(batch_tokens, repr_layers=[i for i in range(1, layer + 1)], return_contacts=True)

            row, col = np.where(contact_prob_map[start:end, start:end] != 0)
            row = row + start
            col = col + start
            contact_prob_map[start:end, start:end] = contact_prob_map[start:end, start:end] + results["contacts"][0].cpu().numpy()
            contact_prob_map[row, col] = contact_prob_map[row, col] / 2.0
            logits[start:end] += results["logits"][0].cpu().numpy()[1 : len(temp_seq) + 1]
            logits[row] = logits[row] / 2.0

            subtoken_repr = torch.cat([results["representations"][i] for i in range(1, layer + 1)])
            assert subtoken_repr.size(0) == layer
            if approach == "last":
                subtoken_repr = subtoken_repr[-1]
            elif approach == "sum":
                subtoken_repr = subtoken_repr.sum(dim=0)
            elif approach == "mean":
                subtoken_repr = subtoken_repr.mean(dim=0)
            subtoken_repr = subtoken_repr.cpu().numpy()[1 : len(temp_seq) + 1]

            trow = np.where(token_representation[start:end].sum(axis=-1) != 0)[0]
            trow = trow + start
            token_representation[start:end] = token_representation[start:end] + subtoken_repr
            token_representation[trow] = token_representation[trow] / 2.0
            if end == len(seq):
                break

    return torch.from_numpy(token_representation), torch.from_numpy(contact_prob_map), torch.from_numpy(logits)


def protein_init_with_keys(keys, seqs):
    assert len(seqs) == len(keys)
    result_dict = {}
    model, alphabet = esm.pretrained.load_model_and_alphabet("esm2_t33_650M_UR50D")
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    batch_converter = alphabet.get_batch_converter()

    for key, seq in tqdm(zip(keys, seqs), total=len(seqs)):
        seq = replace_non_standard_residues(seq, pro_res_table)
        seq_feat = seq_feature(seq)
        token_repr, contact_map_proba, _ = esm_extract(model, batch_converter, seq, layer=33, approach="last", dim=1280)
        assert len(contact_map_proba) == len(seq)
        edge_index, edge_weight = contact_map(contact_map_proba)
        result_dict[key] = {
            "seq": seq,
            "seq_feat": torch.from_numpy(seq_feat),
            "token_representation": token_repr.half(),
            "num_nodes": len(seq),
            "num_pos": torch.arange(len(seq)).reshape(-1, 1),
            "edge_index": edge_index,
            "edge_weight": edge_weight,
        }

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return result_dict


class MultiGraphData(Data):
    def __inc__(self, key, item, *args, **kwargs):
        if key == "mol_edge_index":
            return self.mol_x.size(0)
        if key == "clique_edge_index":
            return self.clique_x.size(0)
        if key == "atom2clique_index":
            return torch.tensor([[self.mol_x.size(0)], [self.clique_x.size(0)]])
        if key == "prot_edge_index":
            return self.prot_node_aa.size(0)
        if key == "prot_struc_edge_index":
            return self.prot_node_aa.size(0)
        if key == "m2p_edge_index":
            return torch.tensor([[self.mol_x.size(0)], [self.prot_node_aa.size(0)]])
        return super().__inc__(key, item, *args, **kwargs)


class ProteinMoleculeDataset(Dataset):
    def __init__(
        self,
        sequence_data,
        molecule_dict=None,
        molecule_folder="",
        protein_dict=None,
        protein_folder="",
        result_path="",
        dataset_tag="",
        source_data_column=None,
        device="cpu",
        molecule_error_log="molecule_error_log.txt",
        standardize=True,
        timeout=10,
        cache_molecules=True,
    ):
        super().__init__()
        if molecule_dict is None:
            molecule_dict = {}
        if protein_dict is None:
            protein_dict = {}
        if not isinstance(sequence_data, pd.core.frame.DataFrame):
            raise Exception("Must be a pandas dataframe..")
        assert "Ligand" in sequence_data.columns and "Protein" in sequence_data.columns, "DataFrame must contain 'Ligand' and 'Protein' columns"

        if "ID" in sequence_data.columns:
            id_series = sequence_data["ID"].astype(str).replace(["", "nan", "NaN", "None"], pd.NA)
            default_ids = pd.Series("Row_" + (sequence_data.index.astype(int) + 1).astype(str), index=sequence_data.index)
            ids = id_series.fillna(default_ids).tolist()
        else:
            ids = ("Row_" + (sequence_data.index.astype(int) + 1).astype(str)).tolist()

        ligands = sequence_data["Ligand"].to_numpy()
        proteins = sequence_data["Protein"].to_numpy()
        reg_labels = sequence_data.get("regression_label", np.full_like(ligands, np.nan, dtype=float))
        cls_labels = sequence_data.get("classification_label", np.full_like(ligands, np.nan, dtype=float))
        oi_labels = sequence_data.get("orthosteric_inhibitor", np.full_like(ligands, -1.0, dtype=float))
        oa_labels = sequence_data.get("orthosteric_activator", np.full_like(ligands, -1.0, dtype=float))
        ai_labels = sequence_data.get("allosteric_inhibitor", np.full_like(ligands, -1.0, dtype=float))
        aa_labels = sequence_data.get("allosteric_activator", np.full_like(ligands, -1.0, dtype=float))
        self.mcls_labels = torch.from_numpy(np.column_stack((oi_labels, oa_labels, ai_labels, aa_labels))).float()

        if source_data_column is not None:
            source_data = sequence_data.get(source_data_column, np.full_like(ligands, np.nan, dtype=float))
        else:
            source_data = np.full_like(ligands, np.nan, dtype=float)

        self.ids = ids
        row_positions = list(range(len(sequence_data)))
        self.pairs = list(zip(ligands, proteins, reg_labels, cls_labels, source_data, ids, row_positions))
        self.mol_dict = molecule_dict
        self.mol_folder = molecule_folder
        self.prot_dict = protein_dict
        self.prot_folder = protein_folder
        self.result_path = result_path
        self.dataset_tag = dataset_tag
        self.device = device
        self.molecule_error_log = molecule_error_log
        self.timeout = timeout
        self.cache_molecules = cache_molecules
        if not standardize:
            raise ValueError(
                "PSICHIC+ requires Papyrus ligand standardization; "
                "RDKit-only normalization is not supported."
            )
        self.standardize = standardize
        self.error_molecule = set()
        self.setup_logging()

    def setup_logging(self):
        log_file_name = f"{self.dataset_tag}_dataset_errors.log" if self.dataset_tag else "dataset_errors.log"
        log_file_path = os.path.join(self.result_path, log_file_name)
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        logging.basicConfig(filename=log_file_path, filemode="a", format="%(asctime)s - %(levelname)s - %(message)s", level=logging.ERROR)

    def safe_smiles_filename(self, smiles):
        return hashlib.sha256(smiles.encode()).hexdigest()[:32]

    def mol_graph(self, mol_key):
        if mol_key in self.mol_dict:
            return self.mol_dict[mol_key]

        mol_filename = self.safe_smiles_filename(mol_key)
        mol_dict_path = os.path.join(self.mol_folder, f"{mol_filename}.pt")
        if self.cache_molecules and os.path.exists(mol_dict_path):
            mol_graph = torch.load(mol_dict_path, weights_only=False)
            if mol_graph is None:
                raise Exception("Molecule is None")
            return mol_graph

        if mol_key in self.error_molecule:
            raise Exception("Molecule is None as previously failed")

        normalizer = MoleculeNormalizer(timeout_seconds=self.timeout)
        result = normalizer.standardize_normalize(mol_key)
        norm_smi = result.smiles
        if norm_smi is None:
            self.error_molecule.add(mol_key)
            with open(self.molecule_error_log, "a") as handle:
                handle.write(
                    f"||||| {mol_key} ||||| with status (failed to normalize because [{result.status}]); "
                    f"and error message - {result.error_message}\n"
                )
            raise Exception("Molecule SMILES is None")

        mol_graph = smiles_to_graph(norm_smi)
        if mol_graph is None:
            self.error_molecule.add(mol_key)
            with open(self.molecule_error_log, "a") as handle:
                handle.write(f"||||| {mol_key} ||||| with status (successful normalization but fail at graph construction); and error message - None\n")
            raise Exception("Molecule is None")

        if self.cache_molecules:
            os.makedirs(self.mol_folder, exist_ok=True)
            torch.save(mol_graph, mol_dict_path)
        return mol_graph

    def prot_graph(self, prot_key):
        if prot_key in self.prot_dict:
            return self.prot_dict[prot_key]

        prot_dict_path = os.path.join(self.prot_folder, f"{prot_key}.pt")
        if os.path.exists(prot_dict_path):
            return torch.load(prot_dict_path, weights_only=False)
        raise Exception("cannot find in dictionary or folder.")

    def get(self, index):
        return self.__getitem__(index)

    def len(self):
        return self.__len__()

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        try:
            mol_key, prot_key, reg_y, cls_y, source_y, id_val, row_position = self.pairs[idx]
            mcls_y = self.mcls_labels[idx].view(1, -1)
            reg_y = torch.tensor(reg_y if not pd.isna(reg_y) else float("nan")).float()
            cls_y = torch.tensor(cls_y if not pd.isna(cls_y) else float("nan")).float()
            mol = self.mol_graph(str(mol_key))
            prot = self.prot_graph(str(prot_key))
            pl_pair = self.create_multigraph_data(mol, prot, reg_y, cls_y, mcls_y, str(mol_key), str(prot_key), str(source_y))
            pl_pair.id = id_val
            pl_pair.row_position = int(row_position)
            return pl_pair
        except Exception as exc:
            mol_key = self.pairs[idx][0] if idx < len(self.pairs) else "<unknown>"
            logging.error(f"Error processing mol_key: ||| {mol_key} |||, error: {str(exc)}")
            return None

    def create_multigraph_data(self, mol, prot, reg_y, cls_y, mcls_y, mol_key, prot_key, source_y):
        return MultiGraphData(
            mol_x=mol["atom_idx"],
            mol_x_feat=mol["atom_feature"],
            mol_x_pe=mol["ligand_pe"] if "ligand_pe" in mol else None,
            mol_edge_index=mol["atom_edge_index"],
            mol_edge_attr=mol["atom_edge_attr"],
            mol_num_nodes=mol["atom_num_nodes"],
            clique_x=mol["x_clique"],
            clique_x_pe=mol["clique_pe"] if "clique_pe" in mol else None,
            clique_edge_index=mol["tree_edge_index"],
            atom2clique_index=mol["atom2clique_index"],
            clique_num_nodes=mol["clique_num_nodes"],
            prot_node_aa=prot["seq_feat"].float(),
            prot_node_evo=prot["token_representation"].float(),
            prot_node_pe=prot["protein_pe"] if "protein_pe" in prot else None,
            prot_seq=prot["seq"],
            prot_edge_index=prot["edge_index"].long(),
            prot_edge_weight=prot["edge_weight"].float(),
            prot_num_nodes=prot["num_nodes"],
            reg_y=reg_y,
            cls_y=cls_y,
            mcls_y=mcls_y,
            mol_key=mol_key,
            prot_key=prot_key,
            source_y=source_y,
        )


def unbatch(src, batch, dim: int = 0):
    sizes = degree(batch, dtype=torch.long).tolist()
    return src.split(sizes, dim)


def unbatch_nodes(data_tensor, index_tensor):
    return [data_tensor[index_tensor == i] for i in index_tensor.unique()]


def minmax_norm(arr):
    return (arr - arr.min()) / (arr.max() - arr.min())


def percentile_rank(arr):
    return np.argsort(np.argsort(arr)) / (len(arr) - 1)


def load_ligand_from_path(ligand_path, ligand_key):
    ligand_filename = safe_smiles_filename(ligand_key)
    ligand_file = os.path.join(ligand_path, f"{ligand_filename}.pt")
    if not os.path.exists(ligand_file):
        raise FileNotFoundError(f"Ligand file not found: {ligand_file}")
    return torch.load(ligand_file, weights_only=False)


def store_ligand_score(ligand_smiles, atom_types, atom_scores, ligand_path):
    mol = Chem.MolFromSmiles(ligand_smiles)
    for i, atom in enumerate(mol.GetAtoms()):
        if atom_types[i] == atom.GetAtomicNum():
            atom.SetProp("PSICHIC_Atom_Score", str(atom_scores[i]))
        else:
            return False
    Chem.SetDefaultPickleProperties(PropertyPickleOptions.AllProps)
    tmp_path = _temporary_output_path(ligand_path)
    try:
        with open(tmp_path, "wb") as handle:
            pickle.dump(mol, handle)
        os.replace(tmp_path, ligand_path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
    return True


def _temporary_output_path(path):
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=f".{os.path.basename(path)}.",
        suffix=".tmp",
        dir=directory,
        delete=False,
    )
    tmp_path = handle.name
    handle.close()
    return tmp_path


def _write_dataframe_csv_atomic(frame, path):
    tmp_path = _temporary_output_path(path)
    try:
        frame.to_csv(tmp_path, index=False)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _write_numpy_atomic(path, array):
    tmp_path = _temporary_output_path(path)
    try:
        with open(tmp_path, "wb") as handle:
            np.save(handle, array)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def build_interpret_result_payload(
    attention_dict,
    ids,
    interaction_keys,
    protein_dict,
    save_cluster=False,
):
    residue_score = attention_dict["residue_final_score"].detach().cpu()
    protein_residue_index = attention_dict["protein_residue_index"].detach().cpu()
    atom_score = attention_dict["atom_final_score"].detach().cpu()
    drug_atom_index = attention_dict["drug_atom_index"].detach().cpu()
    fingerprint = attention_dict["interaction_fingerprint"].detach().cpu()

    payload = {
        "ids": list(ids),
        "interaction_keys": list(interaction_keys),
        "protein_sequences": [protein_dict[key[0]]["seq"] for key in interaction_keys],
        "residue_scores": unbatch(residue_score, protein_residue_index),
        "atom_scores": unbatch(atom_score, drug_atom_index),
        "fingerprints": fingerprint,
        "cluster_scores": None,
    }
    if save_cluster and all(id_ in attention_dict["cluster_s"] for id_ in range(3)):
        payload["cluster_scores"] = [
            unbatch_nodes(attention_dict["cluster_s"][layer].detach().cpu().softmax(dim=-1), protein_residue_index)
            for layer in range(3)
        ]
    return payload


def save_interpret_result_payload(payload, ligand_path, result_path, save_cluster=False):
    os.makedirs(result_path, exist_ok=True)
    saved_pairs = 0
    cluster_scores = payload.get("cluster_scores") if save_cluster else None
    for idx, key in enumerate(payload["interaction_keys"]):
        pair_id = payload["ids"][idx]
        pair_path = os.path.join(result_path, pair_id)
        os.makedirs(pair_path, exist_ok=True)

        protein_interpret = pd.DataFrame(
            {
                "Residue_Type": list(payload["protein_sequences"][idx]),
                "PSICHIC_Residue_Score": minmax_norm(payload["residue_scores"][idx].flatten().numpy()),
            }
        )
        protein_interpret["Residue_ID"] = protein_interpret.index + 1
        protein_interpret["PSICHIC_Residue_Percentile"] = percentile_rank(protein_interpret["PSICHIC_Residue_Score"])
        protein_interpret = protein_interpret[["Residue_ID", "Residue_Type", "PSICHIC_Residue_Score", "PSICHIC_Residue_Percentile"]]
        if cluster_scores is not None:
            for ci in range(5):
                protein_interpret["Layer0_Cluster" + str(ci)] = cluster_scores[0][idx][:, ci].flatten().numpy()
            for ci in range(10):
                protein_interpret["Layer1_Cluster" + str(ci)] = cluster_scores[1][idx][:, ci].flatten().numpy()
            for ci in range(20):
                protein_interpret["Layer2_Cluster" + str(ci)] = cluster_scores[2][idx][:, ci].flatten().numpy()
        _write_dataframe_csv_atomic(protein_interpret, os.path.join(pair_path, "protein.csv"))

        ligand_data = load_ligand_from_path(ligand_path, key[1])
        successful_ligand = store_ligand_score(
            ligand_data["smiles"],
            ligand_data["atom_types"],
            minmax_norm(payload["atom_scores"][idx].flatten().numpy()),
            os.path.join(pair_path, "ligand.pkl"),
        )
        if not successful_ligand:
            raise RuntimeError(f"Ligand interpretation for {pair_id} failed due to not matching atom order.")
        _write_numpy_atomic(
            os.path.join(pair_path, "fingerprint.npy"),
            payload["fingerprints"][idx].numpy(),
        )
        saved_pairs += 1
    return saved_pairs


def store_result(
    df,
    attention_dict,
    ids,
    interaction_keys,
    protein_dict,
    ligand_path,
    reg_pred=None,
    reg_alpha=None,
    reg_beta=None,
    result_path="",
    save_interpret=True,
    save_cluster=False,
    row_id_to_position=None,
    row_positions=None,
):
    if not save_interpret:
        return _store_result_fast(
            df,
            ids,
            reg_pred=reg_pred,
            reg_alpha=reg_alpha,
            reg_beta=reg_beta,
            row_id_to_position=row_id_to_position,
            row_positions=row_positions,
        )

    if save_interpret:
        unbatched_residue_score = unbatch(attention_dict["residue_final_score"], attention_dict["protein_residue_index"])
        unbatched_atom_score = unbatch(attention_dict["atom_final_score"], attention_dict["drug_atom_index"])

    for idx, key in enumerate(interaction_keys):
        matching_row = df["ID"] == ids[idx]
        if reg_pred is not None:
            if "predicted_binding_affinity" not in df.columns:
                df["predicted_binding_affinity"] = None
            df.loc[matching_row, "predicted_binding_affinity"] = reg_pred[idx]
        if reg_alpha is not None:
            if "reg_alpha" not in df.columns:
                df["reg_alpha"] = None
            df.loc[matching_row, "reg_alpha"] = reg_alpha[idx]
        if reg_beta is not None:
            if "reg_beta" not in df.columns:
                df["reg_beta"] = None
            df.loc[matching_row, "reg_beta"] = reg_beta[idx]

        if save_interpret:
            for pair_id in df[matching_row]["ID"]:
                pair_path = os.path.join(result_path, pair_id)
                os.makedirs(pair_path, exist_ok=True)

                protein_seq = protein_dict[key[0]]["seq"]
                protein_interpret = pd.DataFrame(
                    {
                        "Residue_Type": list(protein_seq),
                        "PSICHIC_Residue_Score": minmax_norm(unbatched_residue_score[idx].cpu().flatten().numpy()),
                    }
                )
                protein_interpret["Residue_ID"] = protein_interpret.index + 1
                protein_interpret["PSICHIC_Residue_Percentile"] = percentile_rank(protein_interpret["PSICHIC_Residue_Score"])
                protein_interpret = protein_interpret[["Residue_ID", "Residue_Type", "PSICHIC_Residue_Score", "PSICHIC_Residue_Percentile"]]
                if save_cluster and all(id_ in attention_dict["cluster_s"] for id_ in range(3)):
                    unbatched_cluster_s0 = unbatch_nodes(attention_dict["cluster_s"][0].softmax(dim=-1), attention_dict["protein_residue_index"])
                    unbatched_cluster_s1 = unbatch_nodes(attention_dict["cluster_s"][1].softmax(dim=-1), attention_dict["protein_residue_index"])
                    unbatched_cluster_s2 = unbatch_nodes(attention_dict["cluster_s"][2].softmax(dim=-1), attention_dict["protein_residue_index"])
                    for ci in range(5):
                        protein_interpret["Layer0_Cluster" + str(ci)] = unbatched_cluster_s0[idx][:, ci].cpu().flatten().numpy()
                    for ci in range(10):
                        protein_interpret["Layer1_Cluster" + str(ci)] = unbatched_cluster_s1[idx][:, ci].cpu().flatten().numpy()
                    for ci in range(20):
                        protein_interpret["Layer2_Cluster" + str(ci)] = unbatched_cluster_s2[idx][:, ci].cpu().flatten().numpy()
                _write_dataframe_csv_atomic(protein_interpret, os.path.join(pair_path, "protein.csv"))

                ligand_data = load_ligand_from_path(ligand_path, key[1])
                successful_ligand = store_ligand_score(
                    ligand_data["smiles"],
                    ligand_data["atom_types"],
                    minmax_norm(unbatched_atom_score[idx].cpu().flatten().numpy()),
                    os.path.join(pair_path, "ligand.pkl"),
                )
                if not successful_ligand:
                    print(f"Ligand Intepretation for {pair_id} failed due to not matching atom order.")
                _write_numpy_atomic(os.path.join(pair_path, "fingerprint.npy"), attention_dict["interaction_fingerprint"][idx].detach().cpu().numpy())
    return df


def _store_result_fast(
    df,
    ids,
    reg_pred=None,
    reg_alpha=None,
    reg_beta=None,
    row_id_to_position=None,
    row_positions=None,
):
    if row_positions is not None:
        row_positions_array = np.asarray(row_positions).reshape(-1).astype(int).tolist()
        _assign_result_column(df, row_positions_array, "predicted_binding_affinity", reg_pred)
        _assign_result_column(df, row_positions_array, "reg_alpha", reg_alpha)
        _assign_result_column(df, row_positions_array, "reg_beta", reg_beta)
        return df

    if row_id_to_position is None:
        if not df["ID"].is_unique:
            return _store_result_by_id_mask(
                df,
                ids,
                reg_pred=reg_pred,
                reg_alpha=reg_alpha,
                reg_beta=reg_beta,
            )
        row_id_to_position = {id_value: idx for idx, id_value in enumerate(df["ID"].tolist())}

    try:
        row_positions = [row_id_to_position[id_value] for id_value in ids]
    except KeyError:
        return _store_result_by_id_mask(
            df,
            ids,
            reg_pred=reg_pred,
            reg_alpha=reg_alpha,
            reg_beta=reg_beta,
        )

    _assign_result_column(df, row_positions, "predicted_binding_affinity", reg_pred)
    _assign_result_column(df, row_positions, "reg_alpha", reg_alpha)
    _assign_result_column(df, row_positions, "reg_beta", reg_beta)
    return df


def _assign_result_column(df, row_positions, column, values):
    if values is None:
        return
    if column not in df.columns:
        df[column] = None

    values_array = np.asarray(values).reshape(-1)
    if len(values_array) != len(row_positions):
        raise ValueError(
            f"Expected {len(row_positions)} values for {column}, got {len(values_array)}"
        )
    column_index = df.columns.get_loc(column)
    df.iloc[row_positions, column_index] = values_array


def _store_result_by_id_mask(
    df,
    ids,
    reg_pred=None,
    reg_alpha=None,
    reg_beta=None,
):
    for idx, pair_id in enumerate(ids):
        matching_row = df["ID"] == pair_id
        if reg_pred is not None:
            if "predicted_binding_affinity" not in df.columns:
                df["predicted_binding_affinity"] = None
            df.loc[matching_row, "predicted_binding_affinity"] = reg_pred[idx]
        if reg_alpha is not None:
            if "reg_alpha" not in df.columns:
                df["reg_alpha"] = None
            df.loc[matching_row, "reg_alpha"] = reg_alpha[idx]
        if reg_beta is not None:
            if "reg_beta" not in df.columns:
                df["reg_beta"] = None
            df.loc[matching_row, "reg_beta"] = reg_beta[idx]
    return df
