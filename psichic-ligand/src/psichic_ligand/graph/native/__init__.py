"""Local graph extension: 2024.03.3 atom/bond features and clique merge."""

from ..._native import load_native

_ext = load_native(__name__ + ".graph_ext")
from_rdmol = _ext.from_rdmol
tree_parts = _ext.tree_parts
molecule_parts = _ext.molecule_parts
molecule_from_smiles = _ext.molecule_from_smiles
SOURCE_REVISION = "Release_2024_03_3"
PATCH_NOTE = (
    "native from_rdmol, tree clique-merge, Kruskal MST, and SMILES parse; "
    "PE stays in Python"
)
