import json
from pathlib import Path

import pytest
import torch

from helpers import graph_digest
from psichic_ligand.graph import smiles_to_graph

CASES = json.loads((Path(__file__).with_name("graph_cases.json")).read_text())


@pytest.mark.parametrize("case", CASES)
def test_categories_junction_ties_and_rings_match_independent_build(case, tmp_path):
    graph = smiles_to_graph(case["raw"], case["one_hot"])
    assert graph_digest(graph) == case["digest"]
    path = tmp_path / "graph.pt"
    torch.save(graph, path)
    assert graph_digest(torch.load(path, weights_only=True)) == case["digest"]
    graph["atom_feature"].fill_(19)
    graph["atom_edge_attr"].fill_(23)
    assert graph_digest(smiles_to_graph(case["raw"], case["one_hot"])) == case["digest"]
