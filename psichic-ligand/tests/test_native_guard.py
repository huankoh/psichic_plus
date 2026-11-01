import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "mutation,expected",
    [
        ("rdkit.__version__ = '2099.01.1'", "was built for RDKit 2024.03.3"),
        ("rdkit.__file__ = '/outside-prefix/rdkit/__init__.py'", "Recreate the env"),
        (
            """
import importlib.abc
class BlockNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.endswith(('tautomer_ext', 'graph_ext')):
            raise ModuleNotFoundError(fullname)
sys.meta_path.insert(0, BlockNative())
""",
            "pip install --no-build-isolation --no-deps ./psichic-ligand",
        ),
    ],
)
def test_import_refuses_drifted_or_missing_native(mutation, expected):
    code = "import sys, rdkit\n" + mutation + "\nimport psichic_ligand.normalization\n"
    process = subprocess.run(
        [sys.executable, "-I", "-B", "-c", code], capture_output=True, text=True
    )
    assert process.returncode != 0
    assert "ImportError" in process.stderr
    assert expected in process.stderr


def test_rdkit_shadow_package_on_sys_path_is_refused(tmp_path):
    from pathlib import Path
    import rdkit

    original = Path(rdkit.__file__).resolve()
    shadow = tmp_path / "rdkit"
    shadow.mkdir()
    (shadow / "__init__.py").write_text(
        f"__path__ = [{str(original.parent)!r}]\n"
        f"exec(compile(open({str(original)!r}).read(), {str(original)!r}, 'exec'))\n"
    )
    code = "import sys; sys.path.insert(0, sys.argv[1]); import psichic_ligand"
    process = subprocess.run(
        [sys.executable, "-I", "-B", "-c", code, str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert process.returncode != 0
    assert "was built for RDKit 2024.03.3" in process.stderr
    assert str(shadow.resolve()) in process.stderr
    assert "Recreate the env" in process.stderr
