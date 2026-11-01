import ast
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONDA_PINS = {
    "python": "3.12",
    "numpy": "1.26.4",
    "pandas": "2.2.2",
    "scipy": "1.13.1",
    "rdkit": "2024.03.3",
    "libboost-devel": "1.84.0",
    "cxx-compiler": "1.11.0",
    "setuptools": "80.9.0",
    "pybind11": "3.1.0",
}
PIP_PINS = {
    "torch": "2.6.0",
    "torch-geometric": "2.7.0",
    "torch-scatter": "2.1.2",
    "torch-sparse": "0.6.18",
    "fair-esm": "2.0.0",
}


def test_all_environments_share_one_pinned_stack():
    path = ROOT / "psichic-ligand/src/psichic_ligand/normalization/tautomer_contract.py"
    constants = {
        n.targets[0].id: ast.literal_eval(n.value)
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.Assign) and isinstance(n.value, ast.Constant)
    }
    assert CONDA_PINS["rdkit"] == constants["RDKIT_RELEASE"]
    for variant in ("cpu", "gpu", "osx"):
        env = yaml.safe_load((ROOT / f"environment_{variant}.yml").read_text())
        assert env["channels"] == (
            ["nvidia", "conda-forge"] if variant == "gpu" else ["conda-forge"]
        )
        conda = dict(
            p.split("=", 1)
            for p in env["dependencies"]
            if isinstance(p, str) and "=" in p
        )
        assert {k: conda[k] for k in CONDA_PINS} == CONDA_PINS
        requirements = next(
            p["pip"] for p in env["dependencies"] if isinstance(p, dict)
        )
        pip = dict(p.split("==", 1) for p in requirements if "==" in p)
        assert {k: pip[k].split("+")[0] for k in PIP_PINS} == PIP_PINS
        for p in requirements:
            assert p.split("=")[0].lower() not in {
                "rdkit",
                "chembl-structure-pipeline",
                "papyrus-structure-pipeline",
            }
