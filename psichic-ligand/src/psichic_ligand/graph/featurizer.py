"""Molecule graph assembly with the original PSICHIC tensor contract."""

from __future__ import annotations

from typing import Any, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from torch_geometric.utils import (
    is_torch_sparse_tensor,
    scatter,
    to_edge_index,
)

from .atom_bond import one_hot_encode_features, x_map_length, e_map_length, _long_tensor
from .native import molecule_from_smiles as _native_molecule_from_smiles
from .native import tree_parts as _native_tree_parts


def _tree_from_parts(parts, return_vocab: bool = False):
    n_cliques = int(parts["n_cliques"])
    edge_index = _long_tensor(parts["tree_index"])
    atom_row = np.ascontiguousarray(parts["atom2clique_row"], dtype=np.int64)
    atom_col = np.ascontiguousarray(parts["atom2clique_col"], dtype=np.int64)
    if atom_row.size == 0:
        atom2clique_tensor = torch.empty((2, 0), dtype=torch.long)
    else:
        atom2clique_tensor = torch.stack(
            [torch.from_numpy(atom_row), torch.from_numpy(atom_col)], dim=0
        ).to(torch.long)

    if return_vocab:
        xs = np.ascontiguousarray(parts["xs"], dtype=np.int64)
        vocab = torch.empty((0,), dtype=torch.long) if xs.size == 0 else torch.from_numpy(xs).to(torch.long)
        return edge_index, atom2clique_tensor, n_cliques, vocab
    return edge_index, atom2clique_tensor, n_cliques


def tree_decomposition(mol: Any, return_vocab: bool = False) -> Union[Tuple[Tensor, Tensor, int], Tuple[Tensor, Tensor, int, Tensor]]:
    return _tree_from_parts(_native_tree_parts(bytes(mol.ToBinary())), return_vocab)


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
    parts = _native_molecule_from_smiles(smiles)
    if parts is None:
        return None

    atom_ids = _long_tensor(parts["atom_ids"], (-1, 1))
    atomic_nums = _long_tensor(parts["atomic_nums"], (-1, 1))
    x = _long_tensor(parts["x"], (-1, 8))
    edge_index = _long_tensor(parts["edge_index"])
    edge_attr = _long_tensor(parts["edge_attr"], (-1, 3))
    if one_hot:
        atom_feat = one_hot_encode_features(x, x_map_length)
        edge_attr = one_hot_encode_features(edge_attr, e_map_length)
    else:
        atom_feat = x.float()
        edge_attr = edge_attr.float()
    tree_edge_index, atom2clique_index, num_cliques, x_clique = _tree_from_parts(parts, return_vocab=True)
    if atom2clique_index.nelement() == 0:
        num_cliques = int(atom_ids.shape[0])
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
