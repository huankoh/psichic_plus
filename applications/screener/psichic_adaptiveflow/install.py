"""Apply the bundled patch only to an explicitly supplied, pinned AFVS checkout."""

import argparse
from importlib.resources import files
from pathlib import Path
import subprocess
import tempfile

REVISION = "df74e005e456e94654fa93a40d748de399450b81"
TOUCHED = ["tools/templates/afvs_run.py", "tools/afvs_prepare_folders.py"]


def install(checkout):
    checkout = Path(checkout).resolve(strict=True)

    def git(*args, **kwargs):
        return subprocess.run(
            ["git", "-C", str(checkout), *args], capture_output=True, **kwargs
        )

    head = git("rev-parse", "HEAD")
    if head.returncode or head.stdout.decode().strip() != REVISION:
        raise ValueError(f"Expected AdaptiveFlow-VS revision {REVISION}")
    if git("diff", "--cached", "--name-only", "--", *TOUCHED).stdout:
        raise ValueError("Refusing staged changes to patch targets")
    patch = files("psichic_adaptiveflow").joinpath("native.patch").read_bytes()
    reverse = git("apply", "--reverse", "--check", "-", input=patch)
    if reverse.returncode == 0:
        # A reverse check alone also accepts unrelated edits in the same files.
        with tempfile.TemporaryDirectory() as directory:
            for path in TOUCHED:
                target = Path(directory) / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(git("show", f"{REVISION}:{path}").stdout)
            applied = subprocess.run(
                ["git", "apply", "-"], cwd=directory, input=patch, capture_output=True
            )
            if applied.returncode:
                raise ValueError(applied.stderr.decode())
            if any(
                (checkout / path).read_bytes() != (Path(directory) / path).read_bytes()
                for path in TOUCHED
            ):
                raise ValueError("Patch targets contain additional changes")
        return "already applied"
    if git("status", "--porcelain", "--", *TOUCHED).stdout:
        raise ValueError("Refusing dirty patch targets")
    checked = git("apply", "--check", "-", input=patch)
    if checked.returncode:
        raise ValueError(checked.stderr.decode())
    applied = git("apply", "-", input=patch)
    if applied.returncode:
        raise ValueError(applied.stderr.decode())
    return "applied"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", type=Path)
    args = parser.parse_args()
    print(install(args.checkout))


if __name__ == "__main__":
    main()
