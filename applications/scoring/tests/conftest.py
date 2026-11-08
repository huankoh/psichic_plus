from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import math

import pytest

from applications.scorer import Output
from applications.scoring import (
    Artifact, Asset, BatchContext, Engine, Entry, Fingerprint, Ownership, Receipt, ScoringRun, Settings, Target,
)


def sha(data=b"controlled evidence"):
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def target():
    return Target("controlled-target", "ACDE", (Asset("target.fasta", sha(b">target\nACDE\n")),))


@pytest.fixture
def make_run(tmp_path, target):
    def create(mode="psichic-only", n=3, **kwargs):
        options = {"psichic_options": '{"preset":"controlled"}'} if mode != "boltz-only" else {}
        if mode != "psichic-only":
            options["boltz_options"] = '{"preset":"controlled","msa":"single"}'
        settings = Settings(mode, **options, **kwargs)
        path = tmp_path / f"run-{len(list(tmp_path.glob('run-*')))}"
        return ScoringRun.create(path, (Entry("duplicate", "CCO", f"fixture.csv:{i}", '{"route":"kept"}') for i in range(n)),
                                 target=target, settings=settings)
    return create


def declarations(run, rows, engine):
    if not rows:
        return []
    batch = BatchContext.for_rows(rows, seed=42)
    return [(row.key, Fingerprint(
        engine, "2.2.1" if engine is Engine.BOLTZ else "controlled-v1", row.entry.smiles, run.target,
        sha(row.entry.smiles.encode()), (Asset("weights", sha()),), (Asset("source.py", sha()),),
        '{"python":"3.12","packages":{"controlled":"1"},"runtime":"test fixture"}',
        "cpu", "float32", "raw-v1", "controlled-no-inference-v1", getattr(run.settings, engine.value + "_options"),
        batch, run.settings.required_output if engine is Engine.PSICHIC else Output.AFFINITY, "single", (),
    )) for row in rows]


def prepare(run, engine=Engine.PSICHIC, *, limit=256, ownership=Ownership.LOCAL):
    return run.prepare(engine, declarations(run, run.pending(engine, limit=limit), engine), ownership=ownership)


def transfer(root, task, *, affinity=6., alpha=4., beta=4., boltz=-3., probability=.5, failure=None, distribution_failure=False):
    """Controlled synthetic observations, never model inference evidence."""
    root = Path(root)
    root.mkdir()
    def put(name, data):
        (root / name).write_text(data if isinstance(data, str) else json.dumps(data, allow_nan=False))
    put("input.txt", task.fingerprint.submitted_smiles)
    put("provenance.json", asdict(task.fingerprint))
    put("execution.json", {"schema_version": 1, "invocation_id": task.invocation_id,
                           "command": ["controlled-fixture", "no-inference"], "returncode": 17 if failure else 0})
    put("stdout.log", "controlled fixture\n")
    put("stderr.log", failure or "")
    roles = {"native_input": "input.txt", "provenance": "provenance.json", "execution": "execution.json",
             "stdout": "stdout.log", "stderr": "stderr.log"}
    if failure:
        record = {"status": "failed", "stage": "controlled-engine", "reason": failure}
    elif task.engine is Engine.BOLTZ:
        native = {"affinity_pred_value": boltz, "affinity_probability_binary": probability, "fixture": True}
        put("affinity.json", native)
        put("structure.cif", "data_controlled\n# synthetic structure evidence\n")
        roles.update(native_affinity="affinity.json", structure="structure.cif")
        record = {"status": "succeeded", **{k: native[k] for k in ("affinity_pred_value", "affinity_probability_binary")},
                  "native": native, "submitted_smiles": task.fingerprint.submitted_smiles}
    else:
        distribution = {"state": "not-requested"}
        if task.fingerprint.output is Output.DISTRIBUTION:
            distribution = ({"state": "distribution-failure", "reason": "controlled invalid distribution",
                             "raw_alpha": repr(alpha), "raw_beta": repr(beta)} if distribution_failure else
                            {"state": "valid", "alpha": alpha, "beta": beta, "concentration": alpha + beta})
        record = {"status": "succeeded", "raw_affinity": affinity, "adaptive_score": -affinity,
                  "submitted_smiles": task.fingerprint.submitted_smiles, "normalized_smiles": None, "distribution": distribution}
        put("raw.json", {"schema_version": 1, "task": task.to_dict(), "native": {
            "predicted_binding_affinity": affinity,
            "reg_alpha": {"nonfinite_float": repr(alpha)} if isinstance(alpha, float) and not math.isfinite(alpha) else alpha,
            "reg_beta": {"nonfinite_float": repr(beta)} if isinstance(beta, float) and not math.isfinite(beta) else beta,
            "full_native_extra": "retained"}})
        roles["raw_psichic"] = "raw.json"
    receipt = Receipt(task, json.dumps(record), tuple(Artifact.from_file(root, role, path) for role, path in roles.items()))
    receipt.publish(root)
    return root
