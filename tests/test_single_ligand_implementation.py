import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = {
    "smiles_to_graph",
    "MoleculeNormalizer",
    "tree_decomposition",
    "from_rdmol",
    "add_rw_pe",
    "safe_smiles_filename",
}
VENDORED = {"papyrus_structure_pipeline", "chembl_structure_pipeline"}


def test_only_the_installed_package_owns_ligand_featurization():
    assert not (ROOT / "mol_utils").exists()
    assert not (ROOT / "utils/ligand_init.py").exists()
    for path in ROOT.rglob("*"):
        if path.relative_to(ROOT).parts[0] in {".git", "psichic-ligand", "runs"}:
            continue
        if path.is_dir():
            assert path.name not in {"papyrus_structure_pipeline", "mol_utils"}, path
        if path.suffix != ".py" or not path.is_file():
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                assert node.name not in FORBIDDEN, (path, node.name)
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                assert module.split(".")[0] not in VENDORED, (path, module)
                assert not module.startswith("psichic_ligand."), (path, module)
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".")[0] not in VENDORED, (path, alias.name)
                    assert not alias.name.startswith("psichic_ligand."), (
                        path,
                        alias.name,
                    )
