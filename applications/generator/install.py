"""Install the pinned, patched upstream source without changing its dependency stack."""
import argparse
import hashlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from . import UPSTREAM_REVISION

PATCH_SHA256 = "25b0c8d6111afe24bc5ae975d566962cadaf5a7e0a83c2cc75150b7a4932ad68"
UPSTREAM_URL = "https://github.com/swansonk14/SyntheMol.git"


def install(source: str = UPSTREAM_URL) -> None:
    patch = Path(__file__).resolve().parent / "patches/synthemol-6030233.patch"
    if hashlib.sha256(patch.read_bytes()).hexdigest() != PATCH_SHA256:
        raise ValueError("SyntheMol patch SHA-256 mismatch")
    with tempfile.TemporaryDirectory(prefix="psichic-synthemol-") as directory:
        checkout = Path(directory) / "source"
        subprocess.run(["git", "clone", "--no-hardlinks", source, str(checkout)], check=True)
        subprocess.run(["git", "checkout", "--detach", UPSTREAM_REVISION], cwd=checkout, check=True)
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip()
        if revision != UPSTREAM_REVISION:
            raise ValueError(f"Unexpected SyntheMol revision: {revision}")
        subprocess.run(["git", "apply", "--check", str(patch)], cwd=checkout, check=True)
        subprocess.run(["git", "apply", str(patch)], cwd=checkout, check=True)
        shutil.rmtree(checkout / "synthemol/resources")
        subprocess.run([
            sys.executable, "-m", "pip", "install", "--no-deps", "--no-build-isolation",
            "--no-cache-dir", str(checkout),
        ], check=True)
    print(f"Installed SyntheMol {revision}, patch SHA-256 {PATCH_SHA256}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=UPSTREAM_URL, help="Git URL or read-only local source checkout")
    install(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
