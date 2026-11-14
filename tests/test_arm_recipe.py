import hashlib
import io
import runpy
import tarfile
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def local_package(tmp_path, change):
    package = tmp_path / "rdkit.tar.bz2"
    recipe = ROOT / "psichic-ligand/packaging/rdkit-arm64"
    with tarfile.open(package, "w:bz2") as archive:
        for name in ("meta.yaml", "build.sh", "pyproject.patch"):
            data = (recipe / name).read_bytes()
            if change == "recipe" and name == "build.sh":
                data += b"\n# altered build\n"
            archived = "meta.yaml.template" if name == "meta.yaml" else name
            info = tarfile.TarInfo(f"info/recipe/{archived}")
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
        config = yaml.safe_load((recipe / "conda_build_config.yaml").read_text())
        config = {key: str(value[0]) for key, value in config.items()}
        if change == "configuration":
            config["numpy"] = "2.0"
        data = yaml.safe_dump(config).encode()
        info = tarfile.TarInfo("info/recipe/conda_build_config.yaml")
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
    return {
        "subdir": "linux-aarch64",
        "url": package.as_uri(),
        "sha256": hashlib.sha256(package.read_bytes()).hexdigest(),
    }


@pytest.mark.parametrize(
    "change",
    [None, "digest", "architecture", "channel", "bytes", "recipe", "configuration"],
)
def test_only_matching_local_arm_recipe_is_accepted(tmp_path, change):
    setup = runpy.run_path(str(ROOT / "psichic-ligand/setup.py"))
    record = local_package(tmp_path, change)
    if change == "digest":
        record["sha256"] = "0" * 64
    elif change == "architecture":
        record["subdir"] = "linux-64"
    elif change == "channel":
        record["url"] = "https://example.invalid/rdkit.tar.bz2"
    elif change == "bytes":
        package = tmp_path / "rdkit.tar.bz2"
        package.write_bytes(b"not a package archive")
        record["sha256"] = hashlib.sha256(package.read_bytes()).hexdigest()
    assert setup["matches_arm_recipe"](record, "linux-aarch64") is (change is None)
    assert setup["matches_arm_recipe"](record, "linux-64") is False
