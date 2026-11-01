"""Build native extensions against the activated conda-forge environment."""

import ast
import importlib.util
import json
import os
import platform
import re
import shlex
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

SUPPORTED_SUBDIRS = ("osx-arm64",)
RDKIT_LIBRARIES = (
    "RDKitGraphMol",
    "RDKitSmilesParse",
    "RDKitSubstructMatch",
    "RDKitMolStandardize",
    "RDKitDescriptors",
    "RDKitRDGeneral",
    "RDKitDataStructs",
    "RDKitRDGeometryLib",
    "RDKitRingDecomposerLib",
    "RDKitCatalogs",
    "RDKitChemReactions",
    "RDKitChemTransforms",
    "RDKitFileParsers",
    "RDKitGenericGroups",
    "RDKitDepictor",
    "RDKitSubgraphs",
    "RDKitFingerprints",
    "RDKitMolTransforms",
)
ROOT = Path(__file__).parent


class BuildEnvironmentError(SystemExit):
    """An unsupported build environment with an actionable message."""


@dataclass(frozen=True)
class RdkitToolchain:
    prefix: Path
    include_dirs: list[str]
    library_dirs: list[str]
    runtime_library_dirs: list[str]


def read_contract_constant(name: str) -> str:
    path = ROOT / "src/psichic_ligand/normalization/tautomer_contract.py"
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise BuildEnvironmentError(f"Missing contract constant {name} in {path}")


def conda_subdir() -> str:
    machine = platform.machine().lower()
    if sys.platform == "darwin":
        return "osx-arm64" if machine == "arm64" else "osx-64"
    if sys.platform == "win32":
        return "win-64" if machine in ("amd64", "x86_64") else f"win-{machine}"
    if sys.platform.startswith("linux"):
        return "linux-64" if machine == "x86_64" else f"linux-{machine}"
    return f"{sys.platform}-{machine}"


def locate_rdkit_toolchain() -> RdkitToolchain:
    subdir = conda_subdir()
    if subdir not in SUPPORTED_SUBDIRS:
        raise BuildEnvironmentError(
            f"psichic-ligand: {subdir} has not passed the byte-identity contract "
            f"(verified: {', '.join(SUPPORTED_SUBDIRS)}). Refusing to build an unverified featurizer."
        )
    if sys.version_info[:2] != (3, 12):
        raise BuildEnvironmentError(
            "psichic-ligand requires Python 3.12. Create the env from environment_*.yml."
        )
    prefix = Path(sys.prefix).resolve()
    dev = prefix / "Library" if sys.platform == "win32" else prefix
    release = read_contract_constant("RDKIT_RELEASE")
    records = [
        json.loads(p.read_text()) for p in (prefix / "conda-meta").glob("rdkit-*.json")
    ]
    header = dev / "include/rdkit/GraphMol/ROMol.h"
    matching = [
        r
        for r in records
        if r.get("version") == release
        and "conda-forge" in (r.get("channel", "") + r.get("url", ""))
    ]
    if not matching or not header.is_file():
        spec = importlib.util.find_spec("rdkit")
        found = (
            f"conda rdkit {records[0]['version']}"
            if records
            else f"pip rdkit at {spec.origin}"
            if spec
            else "nothing"
        )
        raise BuildEnvironmentError(
            f"psichic-ligand needs conda-forge rdkit={release} in {prefix} (found: {found}). "
            "Create the env from environment_*.yml."
        )
    config = (dev / "lib/cmake/rdkit/rdkit-config.cmake").read_text()
    expected = re.search(r"set\(RDKit_USE_BOOST_VERSION\s+([\d.]+)\)", config).group(1)
    boost_header = dev / "include/boost/version.hpp"
    if not boost_header.exists():
        raise BuildEnvironmentError(
            f"Missing Boost headers. Pin libboost-devel={expected}."
        )
    value = int(
        re.search(r"#define BOOST_VERSION\s+(\d+)", boost_header.read_text()).group(1)
    )
    actual = f"{value // 100000}.{value // 100 % 1000}.{value % 100}"
    if actual != expected:
        raise BuildEnvironmentError(
            f"Boost headers {actual} do not match the Boost {expected} RDKit {release} was built with. "
            f"Pin libboost-devel={expected}."
        )
    if sys.platform != "win32" and not shutil.which(
        shlex.split(os.environ.get("CXX", "c++"))[0]
    ):
        raise BuildEnvironmentError(
            "No C++ compiler. Run inside the activated env (conda activate psichic_fp) "
            "so conda-forge cxx-compiler is on PATH."
        )
    return RdkitToolchain(
        prefix,
        [str(dev / "include/rdkit"), str(dev / "include")],
        [str(dev / "lib")],
        [] if sys.platform == "win32" else [str(dev / "lib")],
    )


def extensions(toolchain: RdkitToolchain) -> list[Pybind11Extension]:
    return [
        Pybind11Extension(
            f"psichic_ligand.{stage}.native.{name}",
            [f"src/psichic_ligand/{stage}/native/{name}.cpp"],
            include_dirs=toolchain.include_dirs,
            library_dirs=toolchain.library_dirs,
            runtime_library_dirs=toolchain.runtime_library_dirs,
            libraries=list(RDKIT_LIBRARIES),
            cxx_std=17,
            # Hidden RDKit template symbols change normalization results on macOS.
            extra_compile_args=[]
            if sys.platform == "win32"
            else ["-fvisibility=default"],
        )
        for stage, name in (("normalization", "tautomer_ext"), ("graph", "graph_ext"))
    ]


if __name__ == "__main__":
    setup(
        ext_modules=extensions(locate_rdkit_toolchain()),
        cmdclass={"build_ext": build_ext},
        options={"build_ext": {"force": True}},
    )
