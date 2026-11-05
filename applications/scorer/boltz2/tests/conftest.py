import sys

import pytest


@pytest.fixture
def fake_boltz(tmp_path):
    directory = tmp_path / "fake-environment/bin"
    directory.mkdir(parents=True)
    python = directory / "python"
    python.write_text(f"#!{sys.executable}\nimport json\nprint(json.dumps({{'boltz':'2.2.1','python':'test fixture','packages':{{}}}}))\n")
    executable = directory / "boltz"
    executable.write_text(f"#!{sys.executable}\n" + r'''
import json
import os
from pathlib import Path
import sys
import time
import yaml

if "--help" in sys.argv:
    print("  --accelerator TEXT\n  --sampling_steps INTEGER\n  --output_format TEXT\n  --seed INTEGER\n  --cache PATH")
    sys.exit(0)
args = sys.argv[1:]
assert args[0] == "predict"
source = Path(args[1])
destination = Path(args[args.index("--out_dir") + 1]) / f"boltz_results_{source.stem}" / "predictions"
if os.environ.get("FAKE_BOLTZ_TIMEOUT"):
    time.sleep(20)
for index, path in enumerate(sorted(source.glob("*.yaml"))):
    document = yaml.safe_load(path.read_text())
    binder = document["properties"][-1]["affinity"]["binder"]
    ligand = next(item["ligand"] for item in document["sequences"] if "ligand" in item and item["ligand"]["id"] == binder)
    if ligand["smiles"] == "invalid":
        continue
    output = destination / path.stem
    output.mkdir(parents=True)
    payload = {"affinity_probability_binary": (index + 1) / 10, "affinity_pred_value": -index,
               "affinity_pred_value1": -index - .1, "affinity_pred_value2": -index + .1}
    (output / f"affinity_{path.stem}.json").write_text(json.dumps(payload))
    (output / f"{path.stem}_model_0.cif").write_text("native structure fixture")
(destination.parent / "argv.json").write_text(json.dumps(args))
sys.exit(int(os.environ.get("FAKE_BOLTZ_EXIT", "0")))
''')
    python.chmod(0o755)
    executable.chmod(0o755)
    return python


@pytest.fixture
def fasta(tmp_path):
    path = tmp_path / "target.fasta"
    path.write_text(">test\nMKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ\n")
    return path
