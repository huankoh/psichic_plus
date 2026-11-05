import os
from pathlib import Path
import subprocess
import sys
import tempfile

import pytest


@pytest.mark.parametrize("inside_git", [False, True], ids=["outside-git", "inside-git"])
def test_cli_installer_is_exact_and_idempotent_with_nested_tmpdir(tmp_path, inside_git):
    upstream = os.environ.get("AFVS_TEST_UPSTREAM")
    if not upstream:
        pytest.skip("Set AFVS_TEST_UPSTREAM to a pinned AFVS checkout")
    checkout = tmp_path / "afvs"
    subprocess.run(
        ["git", "clone", "--quiet", "--no-hardlinks", upstream, str(checkout)],
        cwd=tmp_path,
        capture_output=True,
        check=True,
    )
    screener = Path(__file__).resolve().parents[1]
    expected_patch = (screener / "psichic_adaptiveflow/native.patch").read_bytes()
    with tempfile.TemporaryDirectory(prefix="afvs-installer-", dir="/tmp") as directory:
        workspace = Path(directory).resolve()
        temporary_root = workspace / "tmp"
        if inside_git:
            unrelated = workspace / "unrelated"
            subprocess.run(
                ["git", "init", "--quiet", str(unrelated)], cwd=workspace, check=True
            )
            temporary_root = unrelated / "tmp"
        temporary_root.mkdir()
        env = dict(
            os.environ,
            PYTHONPATH=str(screener),
            PYTHONDONTWRITEBYTECODE="1",
            TMPDIR=str(temporary_root),
        )
        discovery = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=temporary_root,
            env=env,
            capture_output=True,
            text=True,
        )
        assert (discovery.returncode == 0) == inside_git
        if inside_git:
            assert Path(discovery.stdout.strip()) == unrelated

        def install():
            return subprocess.run(
                [sys.executable, "-m", "psichic_adaptiveflow.install", str(checkout)],
                cwd=workspace,
                env=env,
                capture_output=True,
                text=True,
                timeout=10,
            )

        runner = checkout / "tools/templates/afvs_run.py"
        original = runner.read_bytes()
        runner.write_bytes(original + b"\n# unrelated edit\n")
        dirty = install()
        assert dirty.returncode != 0
        assert "Refusing dirty patch targets" in dirty.stderr
        assert runner.read_bytes() == original + b"\n# unrelated edit\n"
        runner.write_bytes(original)

        for expected in ("applied\n", "already applied\n"):
            result = install()
            assert result.returncode == 0, result.stderr
            assert result.stdout == expected
            assert result.stderr == ""
            diff = subprocess.run(
                ["git", "diff", "--no-ext-diff"],
                cwd=checkout,
                capture_output=True,
                check=True,
            )
            assert diff.stdout == expected_patch

        for name in ("tools/templates/afvs_run.py", "tools/afvs_prepare_folders.py"):
            target = checkout / name
            patched = target.read_bytes()
            target.write_bytes(patched + b"\n# unrelated edit\n")
            modified = install()
            assert modified.returncode != 0
            assert "Patch targets contain additional changes" in modified.stderr
            assert target.read_bytes() == patched + b"\n# unrelated edit\n"
            target.write_bytes(patched)

        subprocess.run(["git", "add", str(runner)], cwd=checkout, check=True)
        staged = install()
        assert staged.returncode != 0
        assert "Refusing staged changes to patch targets" in staged.stderr
