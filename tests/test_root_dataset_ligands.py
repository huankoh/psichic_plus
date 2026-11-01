import pickle

import pandas as pd
import torch

from psichic_ligand import FEATURIZATION_VERSION, LigandStore
from tests.contract_helpers import contract_rows, digest
from utils.dataset import ProteinMoleculeDataset
from utils.utils import store_result


def protein_graph():
    return dict(
        seq="ACDE",
        seq_feat=torch.zeros(4, 33),
        token_representation=torch.zeros(4, 1280),
        num_nodes=4,
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]),
        edge_weight=torch.ones(3),
    )


def test_root_dataset_uses_versioned_store_and_contract_graphs(tmp_path):
    rows = [r for r in contract_rows() if r["status"] == "SUCCESS"][:5]
    frame = pd.DataFrame({"Ligand": [r["raw"] for r in rows], "Protein": ["P"] * 5})
    dataset = ProteinMoleculeDataset(
        frame,
        molecule_folder=tmp_path / "graphs",
        protein_dict={"P": protein_graph()},
        result_path=str(tmp_path),
    )
    assert isinstance(dataset.ligands, LigandStore)
    for index, row in enumerate(rows):
        graph = dataset.ligands.get(row["raw"])
        assert digest(graph) == row["digest"]
        item = dataset[index]
        assert item is not None
        assert torch.equal(item.mol_x_feat, graph["atom_feature"])
        assert item.mol_key == row["raw"]
    assert dataset.ligands.directory == tmp_path / "graphs" / FEATURIZATION_VERSION
    assert len(list(dataset.ligands.directory.glob("*.pt"))) == 5


def test_invalid_row_returns_none_and_logs_once(tmp_path):
    log = tmp_path / "errors.log"
    dataset = ProteinMoleculeDataset(
        pd.DataFrame({"Ligand": ["not-a-smiles"], "Protein": ["P"]}),
        molecule_folder=tmp_path / "cache",
        result_path=str(tmp_path),
        molecule_error_log=log,
    )
    assert dataset[0] is None
    assert dataset[0] is None
    assert (
        log.read_text()
        == "||||| not-a-smiles ||||| with status (failed to normalize because [INVALID_SMILES]); and error message - Could not parse SMILES\n"
    )


def test_interpretation_reads_the_same_store(tmp_path, monkeypatch):
    raw = "CC(=O)Oc1ccccc1C(=O)O"
    store = LigandStore(tmp_path / "cache")
    graph = store.get(raw)
    calls = []
    original = store.get

    def traced(key):
        calls.append(key)
        return original(key)

    monkeypatch.setattr(store, "get", traced)
    frame = pd.DataFrame({"ID": ["pair"], "Protein": ["P"], "Ligand": [raw]})
    attention = dict(
        residue_final_score=torch.arange(4).float(),
        protein_residue_index=torch.zeros(4, dtype=torch.long),
        atom_final_score=torch.arange(13).float(),
        drug_atom_index=torch.zeros(13, dtype=torch.long),
        residue_layer_scores=torch.zeros(4, 3),
        clique_layer_scores=torch.zeros(graph["clique_num_nodes"], 3),
        drug_clique_index=torch.zeros(graph["clique_num_nodes"], dtype=torch.long),
        interaction_fingerprint=torch.zeros(1, 10),
    )
    store_result(
        frame,
        attention,
        ["pair"],
        [("P", raw)],
        {"P": protein_graph()},
        store,
        result_path=str(tmp_path / "interpret"),
    )
    assert calls == [raw]
    with (tmp_path / "interpret/pair/ligand.pkl").open("rb") as handle:
        molecule = pickle.load(handle)
    assert [a.GetAtomicNum() for a in molecule.GetAtoms()] == graph[
        "atom_types"
    ].flatten().tolist()
    assert (tmp_path / "interpret/pair/protein.csv").is_file()


def test_symmetric_interpretation_scores_are_finite():
    import numpy as np
    from utils.utils import minmax_norm
    with np.errstate(divide="raise", invalid="raise"):
        assert np.array_equal(minmax_norm(np.full(8, 0.5)), np.zeros(8))
