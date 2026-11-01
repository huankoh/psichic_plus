import ast
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_supported_platforms_equal_the_ligand_ci_matrix():
    tree = ast.parse((ROOT / "psichic-ligand/setup.py").read_text())
    supported = next(
        ast.literal_eval(n.value)
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == "SUPPORTED_SUBDIRS" for t in n.targets
        )
    )
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    matrix = workflow["jobs"]["ligand"]["strategy"]["matrix"]["include"]
    assert set(supported) == {entry["subdir"] for entry in matrix}
