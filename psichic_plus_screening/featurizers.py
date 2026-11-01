"""Runtime graph featurizers and PyG data plumbing for PSICHIC+ inference."""

from __future__ import annotations

import gc
import logging
import math
import os
import pickle
import tempfile
from collections.abc import Mapping
from typing import Any, List, Optional, Sequence, Union

import esm
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import PropertyPickleOptions
from torch.utils.data.dataloader import default_collate
from torch_geometric.data import Batch, Data, Dataset
from torch_geometric.data.data import BaseData
from torch_geometric.data.datapipes import DatasetAdapter
from torch_geometric.typing import TensorFrame, torch_frame
from torch_geometric.utils import (
    add_self_loops,
    coalesce,
    degree,
    remove_self_loops,
    to_undirected,
)
from tqdm import tqdm

from psichic_ligand import LigandStore



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
        *,
        molecule_folder=None,
        protein_dict=None,
        protein_folder="",
        result_path="",
        dataset_tag="",
        source_data_column=None,
        device="cpu",
        molecule_error_log="molecule_error_log.txt",
        timeout=10,
    ):
        super().__init__()
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
        self.ligands = LigandStore(molecule_folder, error_log=molecule_error_log, timeout_seconds=timeout)
        self.prot_dict = protein_dict
        self.prot_folder = protein_folder
        self.result_path = result_path
        self.dataset_tag = dataset_tag
        self.device = device
        self.molecule_error_log = molecule_error_log
        self.timeout = timeout
        self.setup_logging()

    def setup_logging(self):
        log_file_name = f"{self.dataset_tag}_dataset_errors.log" if self.dataset_tag else "dataset_errors.log"
        log_file_path = os.path.join(self.result_path, log_file_name)
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        logging.basicConfig(filename=log_file_path, filemode="a", format="%(asctime)s - %(levelname)s - %(message)s", level=logging.ERROR)



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
            mol = self.ligands.get(str(mol_key))
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
    span = arr.max() - arr.min()
    return np.zeros_like(arr) if span == 0 else (arr - arr.min()) / span


def percentile_rank(arr):
    return np.argsort(np.argsort(arr)) / (len(arr) - 1)




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


def save_interpret_result_payload(payload, ligands: LigandStore, result_path, save_cluster=False):
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

        ligand_data = ligands.get(key[1])
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
    ligands: LigandStore,
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

                ligand_data = ligands.get(key[1])
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
