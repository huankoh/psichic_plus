import ast
import json
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
    assert (
        workflow["jobs"]["ligand"]["strategy"]["matrix"]
        == "${{ fromJSON(needs.portability.outputs.matrix) }}"
    )
    gate = next(
        step
        for step in workflow["jobs"]["portability"]["steps"]
        if step.get("id") == "platforms"
    )
    assert ".github/ligand-platforms.json" in gate["run"]
    matrix = json.loads((ROOT / ".github/ligand-platforms.json").read_text())
    assert set(supported) == {entry["subdir"] for entry in matrix}
