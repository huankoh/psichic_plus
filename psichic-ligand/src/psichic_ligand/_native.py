"""Load installed native extensions after checking RDKit provenance."""

import importlib
import os
import sys
from pathlib import Path
from types import ModuleType

_DLL_HANDLES = []


def load_native(qualified_name: str) -> ModuleType:
    import rdkit
    from .normalization.tautomer_contract import RDKIT_RELEASE

    path = Path(rdkit.__file__).resolve()
    if rdkit.__version__ != RDKIT_RELEASE or not path.is_relative_to(
        Path(sys.prefix).resolve()
    ):
        raise ImportError(
            f"psichic_ligand was built for RDKit {RDKIT_RELEASE} but this process loads "
            f"RDKit {rdkit.__version__} from {path}. "
            "Recreate the env from environment_*.yml and reinstall psichic-ligand."
        )
    if sys.platform == "win32" and not _DLL_HANDLES:
        _DLL_HANDLES.append(os.add_dll_directory(str(Path(sys.prefix) / "Library/bin")))
    try:
        return importlib.import_module(qualified_name)
    except ImportError as exc:
        raise ImportError(
            "psichic_ligand native extensions are not built or could not be loaded. "
            "Inside the activated env run: "
            "pip install --no-build-isolation --no-deps ./psichic-ligand"
        ) from exc
