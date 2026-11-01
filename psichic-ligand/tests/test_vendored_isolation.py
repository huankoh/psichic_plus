import subprocess
import sys


def test_installed_package_uses_only_local_pipeline_code():
    code = """
import importlib.abc
import pathlib
import sys
class BlockOriginal(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'utils', 'mol_utils', 'featurizers', 'chembl_structure_pipeline', 'papyrus_structure_pipeline', 'esm'}:
            raise ImportError('Original dependency used: ' + fullname)
sys.meta_path.insert(0, BlockOriginal())
import psichic_ligand
from psichic_ligand.normalization import MoleculeNormalizer
from psichic_ligand.graph import smiles_to_graph
from psichic_ligand.normalization.pipeline import Papyrus_standardizer
result = MoleculeNormalizer().standardize_normalize('CC(=O)Oc1ccccc1C(=O)O')
assert result.status == 'SUCCESS'
assert smiles_to_graph(result.smiles)['atom_num_nodes'] == 13
root = pathlib.Path(psichic_ligand.__file__).parent
for path in [Papyrus_standardizer.__file__, Papyrus_standardizer.standardizer.__file__,
             Papyrus_standardizer.standardizer._salts_file, Papyrus_standardizer.standardizer._solvents_file]:
    assert pathlib.Path(path).is_relative_to(root), path
"""
    process = subprocess.run([sys.executable, "-I", "-B", "-c", code], capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
