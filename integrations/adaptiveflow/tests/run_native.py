"""Real native AFVS fixture run. This test helper never prepares user libraries."""

import argparse
import gzip
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
CAFFEINE = "Cn1c(=O)c2c(ncn2C)n(C)c1=O"
IBUPROFEN = "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O"
PARACETAMOL = "CC(=O)Nc1ccc(O)cc1"
FULL = {
    "AA_0000001": {
        "shared": ASPIRIN,
        "a": ASPIRIN,
        "acid": IBUPROFEN,
        "bad": "not_smiles",
        "empty": "",
    },
    "BB_0000001": {"shared": CAFFEINE, "b": CAFFEINE, "other": PARACETAMOL},
}
SEQUENCE = "MKTAYIAKQRQISFVKSHFSRQDILD"


def write_archive(path, records, listing=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w:gz") as archive:
        for name, value in records.items():
            content = (value + "\n").encode()
            member = tarfile.TarInfo("0000001/" + name + ".smi")
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
        if listing is not None:
            content = listing.encode()
            member = tarfile.TarInfo("0000001/.listing")
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))


def run(command, cwd, env, log):
    print("RUN", command, "IN", cwd, flush=True)
    with log.open("w") as output:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
            timeout=240,
        )
    if completed.returncode:
        raise RuntimeError(
            f"Command failed ({completed.returncode}): {log}\n{log.read_text()[-5000:]}"
        )


def prepare_case(
    base,
    source,
    model,
    env,
    name,
    prescreen,
    todo,
    replicas=1,
    mixed=False,
    library_name=None,
):
    case = base / name
    shutil.copytree(source / "tools", case / "tools")
    tools = case / "tools"
    (case / "input-files" / "target").mkdir(parents=True)
    (case / "input-files" / "target" / "target.fasta").write_text(
        ">tiny\n" + SEQUENCE + "\n"
    )
    (case / "input-files" / "target" / "config.txt").write_text(
        f"protein_fasta=target.fasta\nmodel_dir={model}\ndevice=cpu\n"
    )
    (tools / "templates" / "todo.all").write_text(todo)
    library = base / (library_name or ("sparse" if prescreen else "full"))
    replacements = {
        "job_name": name,
        "threads_per_docking": "1",
        "threads_to_use": "1",
        "program_timeout": "120",
        "batchsystem": "bash",
        "bash_array_job_size": "1",
        "data_storage_mode": "sharedfs",
        "job_storage_mode": "sharedfs",
        "data_collection_addressing_mode": "metatranche",
        "data_collection_identifier": "",
        "collection_folder": str(library),
        "summary_formats": "parquet,csv.gz",
        "ligand_library_format": "smi",
        "dynamic_tranche_filtering": "0",
        "tempdir_default": str(base / "tmp"),
        "docking_scenario_names": "target",
        "docking_scenario_methods": "psichic_plus",
        "docking_scenario_replicas": str(replicas),
        "docking_scenario_batchsizes": "2",
        "docking_scenario_basefolder": "../input-files",
        "prescreen_mode": str(int(prescreen)),
        "prescreen_ligands_per_tranche": "1",
        "run_atom_check": "0",
        "print_attrs_in_summary": "smi,psichic_affinity",
        "dockings_per_subjob": "100",
    }
    if mixed:
        legacy = case / "input-files" / "legacy"
        legacy.mkdir()
        (legacy / "config.txt").write_text("")
        executable = tools / "bin" / "vina"
        executable.write_text('#!/bin/sh\necho "test-only native legacy failure"\n')
        executable.chmod(0o755)
        replacements.update(
            docking_scenario_names="legacy:target",
            docking_scenario_methods="vina:psichic_plus",
            docking_scenario_replicas="1:1",
            docking_scenario_batchsizes="2:2",
        )
    ctrl = (tools / "templates" / "all.ctrl").read_text()
    for key, value in replacements.items():
        ctrl, count = re.subn(r"^" + key + "=.*$", key + "=" + value, ctrl, flags=re.M)
        if not count:
            ctrl += "\n" + key + "=" + value + "\n"
    (tools / "templates" / "all.ctrl").write_text(ctrl)
    # The stock Bash template assumes /dev/shm. Only this copied test template changes.
    template = tools / "templates" / "template1.bash"
    template.write_text(
        template.read_text().replace(
            "export AFVS_TMP_PATH=/dev/shm", f"export AFVS_TMP_PATH={base / 'tmp'}"
        )
    )
    run([sys.executable, "afvs_prepare_folders.py"], tools, env, case / "prepare.log")
    run(
        [sys.executable, "afvs_prepare_workunits.py"],
        tools,
        env,
        case / "workunits.log",
    )
    run(
        [sys.executable, "afvs_submit_jobs.py", "1", "1"],
        tools,
        env,
        case / "submit.log",
    )
    for generated in (case / "workflow" / "run.bash").read_text().splitlines():
        if generated.strip() and not generated.startswith("#"):
            run(["bash", generated], tools, env, case / "generated.log")
    return case


def read_case(case):
    import pandas as pd

    parquet = list((case / "output-files").rglob("*.parquet"))
    assert parquet, f"No native summaries in {case}"
    frame = pd.concat([pd.read_parquet(path) for path in parquet], ignore_index=True)
    overviews = []
    for path in (case / "output-files").rglob("*.json.gz"):
        with gzip.open(path, "rt") as handle:
            data = json.load(handle)
        if "failed_list" in data:
            overviews.append(data)
    assert overviews, f"No overview in {case}"
    return frame, overviews


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base = args.output.resolve()
    base.mkdir(parents=True, exist_ok=False)
    (base / "tmp").mkdir()
    source = args.upstream.resolve()
    model = args.model_dir.resolve()
    env = dict(
        os.environ,
        AWS_EC2_METADATA_DISABLED="true",
        AWS_DEFAULT_REGION="us-east-1",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        CUDA_VISIBLE_DEVICES="",
    )
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env["PATH"]
    if sys.platform == "darwin":
        compatibility = base / "mac-test-only"
        compatibility.mkdir()
        (compatibility / "sitecustomize.py").write_text(
            "import multiprocessing\nimport multiprocessing.queues\nmultiprocessing.set_start_method('fork', force=True)\nmultiprocessing.queues.Queue.qsize = lambda self: 0\n"
        )
        env["PYTHONPATH"] = str(compatibility) + os.pathsep + env.get("PYTHONPATH", "")
    for key, records in FULL.items():
        tranche, number = key.split("_")
        write_archive(base / "full" / tranche / (number + ".tar.gz"), records)
    write_archive(
        base / "sparse" / "sparse" / "0000001.tar.gz",
        {"a": ASPIRIN, "b": CAFFEINE, "bad": "not_smiles", "empty": ""},
        "AA_0000001,a,0\nBB_0000001,b,0\nAA_0000001,bad,0\nAA_0000001,empty,0\n",
    )
    prescreen = prepare_case(
        base, source, model, env, "prescreen", True, "sparse_0000001 4\n"
    )
    pre, pre_overviews = read_case(prescreen)
    assert set(zip(pre.collection_key, pre.ligand)) == {
        ("AA_0000001", "a"),
        ("BB_0000001", "b"),
    }
    assert sum(data["total_dockings"] for data in pre_overviews) == 4
    assert sum(len(data["failed_list"]) for data in pre_overviews) == 2
    import pandas as pd

    inventory = pd.DataFrame(
        [
            {
                "Tranche": key.split("_")[0],
                "Collection": key.split("_")[1],
                "LigandCount": len(records),
            }
            for key, records in FULL.items()
        ]
    )
    inventory.to_parquet(base / "collections.parquet", index=False)
    inventory.groupby("Tranche", as_index=False).LigandCount.sum().to_parquet(
        base / "tranches.parquet", index=False
    )
    # Same four native columns as afvs_postprocess_atg-prescreen.sh, without its AWS-only Athena fetch.
    subset = pre[["collection_key", "ligand", "score_min"]].copy()
    subset[["Tranche", "Collection"]] = subset.collection_key.str.split(
        "_", expand=True
    )
    subset = subset.rename(columns={"ligand": "LigandAFID", "score_min": "ScoreMin"})[
        ["Tranche", "Collection", "LigandAFID", "ScoreMin"]
    ]
    subset.to_csv(base / "ranking.subset-1.csv.gz", index=False, compression="gzip")
    selected = base / "selected.todo"
    run(
        [
            sys.executable,
            str(source / "tools/templates/create_todofile_atg-primaryscreen.py"),
            str(base / "ranking.subset-1.csv.gz"),
            str(base / "collections.parquet"),
            str(base / "tranches.parquet"),
            "tranche_ave_score",
            str(selected),
            "1",
            ".*",
        ],
        source / "tools",
        env,
        base / "selector.log",
    )
    best = pre.sort_values("score_min").iloc[0].collection_key
    assert selected.read_text().split()[0] == best
    primary = prepare_case(
        base, source, model, env, "primary", False, selected.read_text()
    )
    primary_frame, primary_overviews = read_case(primary)
    assert set(primary_frame.collection_key) == {best}
    full_todo = "".join(f"{key} {len(records)}\n" for key, records in FULL.items())
    full = prepare_case(
        base, source, model, env, "full-library", False, full_todo, replicas=2
    )
    actual, overviews = read_case(full)
    assert len(actual) == 6
    assert set(actual.loc[actual.ligand == "shared", "collection_key"]) == set(FULL)
    assert sum(data["total_dockings"] for data in overviews) == 16
    failures = [row for data in overviews for row in data["failed_list"]]
    assert len(failures) == 4
    assert {row["ligand_key"] for row in failures} == {"bad", "empty"}
    assert all(
        {
            "collection_key",
            "ligand_key",
            "scenario_key",
            "replica_index",
            "stage",
            "reason",
        }
        <= row.keys()
        for row in failures
    )
    assert (actual.score_0 - actual.score_1).abs().max() < 1e-5
    for _, row in actual.iterrows():
        assert row.score_0 == -row.attr_psichic_affinity
        assert abs(row.score_average - (row.score_0 + row.score_1) / 2) < 1e-12
        assert row.score_min == min(row.score_0, row.score_1)
    logs = []
    for path in (full / "output-files").rglob("*.tar.gz"):
        with tarfile.open(path) as archive:
            logs.extend(
                member.name for member in archive if member.name.endswith("/stdout")
            )
    shared_logs = [name for name in logs if "/shared/" in name]
    assert len(shared_logs) == 4
    assert any("/AA_0000001/" in name for name in shared_logs) and any(
        "/BB_0000001/" in name for name in shared_logs
    )
    native_log = (full / "workflow/workunits/1/0.out").read_text()
    assert native_log.count("PSICHIC+ initialized target and model once") == 1
    assert native_log.count("PSICHIC+ completed batch") >= 2
    direct_input = base / "direct.csv"
    pd.DataFrame(
        [
            {"ID": key + "/" + ligand, "Ligand": smiles}
            for key, records in FULL.items()
            for ligand, smiles in records.items()
        ]
    ).to_csv(direct_input, index=False)
    screening = (
        Path(__file__).resolve().parents[3] / "psichic_plus_screening/inference.py"
    )
    run(
        [
            sys.executable,
            str(screening),
            "--input_file",
            str(direct_input),
            "--protein_fasta",
            str(full / "input-files/target/target.fasta"),
            "--model_dir",
            str(model),
            "--output_folder",
            str(base / "direct"),
            "--device",
            "cpu",
            "--batch_size",
            "2",
            "--num_workers",
            "0",
            "--precision",
            "fp32",
            "--inference_preset",
            "conservative",
            "--invalid_ligand_policy",
            "emit_null",
        ],
        base,
        env,
        base / "direct.log",
    )
    direct = pd.read_csv(base / "direct/inference_results.csv").set_index("ID")
    differences = []
    for _, row in actual.iterrows():
        expected = direct.loc[
            row.collection_key + "/" + row.ligand, "predicted_binding_affinity"
        ]
        differences.append(abs(row.attr_psichic_affinity - expected))
    assert max(differences) < 1e-5, differences
    from psichic_adaptiveflow.adapter import (
        Affinity,
        Failure,
        LigandKey,
        ScoreInput,
        Scorer,
        Target,
    )

    recovery = []
    target = Target(full / "input-files/target/target.fasta", model, "cpu")
    fault_child = Path(__file__).with_name("fault_child.py")
    for mode in ("before-read", "initialize", "predict", "partial"):
        scorer = Scorer(target, 1, [sys.executable, str(fault_child), mode])
        fault_input = ScoreInput(
            LigandKey("AA_0000001", "a"),
            0,
            direct_input,
            "C" * 1_000_000 if mode == "before-read" else ASPIRIN,
        )
        try:
            failed = scorer.score([fault_input], 0.5)
            assert len(failed) == 1 and isinstance(failed[0], Failure)
            assert scorer.process is None
            scorer.command = None
            recovered = scorer.score(
                [ScoreInput(fault_input.key, 0, direct_input, ASPIRIN)], 120
            )
            assert isinstance(recovered[0], Affinity), recovered
            assert (
                abs(
                    recovered[0].value
                    - direct.loc["AA_0000001/a", "predicted_binding_affinity"]
                )
                < 1e-5
            )
            recovery.append(mode)
        finally:
            pid = scorer.process.pid if scorer.process else None
            scorer.close()
        if pid is not None:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                pass
            else:
                raise AssertionError("Recovered inference child was not reaped")
    mixed = prepare_case(
        base,
        source,
        model,
        env,
        "mixed-methods",
        True,
        "sparse_0000001 4\n",
        mixed=True,
    )
    mixed_frame, mixed_overviews = read_case(mixed)
    assert set(mixed_frame.scenario) == {"target"}
    assert len(mixed_frame) == 2
    assert sum(data["total_dockings"] for data in mixed_overviews) == 8
    assert sum(len(data["failed_list"]) for data in mixed_overviews) == 6
    mixed_log = (mixed / "workflow/workunits/1/0.out").read_text()
    assert "processing a" in mixed_log and "Could not find score" in mixed_log
    ambiguous_dir = base / "ambiguous" / "AA"
    ambiguous_dir.mkdir(parents=True)
    with tarfile.open(ambiguous_dir / "0000001.tar.gz", "w:gz") as archive:
        for smiles in (ASPIRIN, CAFFEINE):
            content = (smiles + "\n").encode()
            member = tarfile.TarInfo("0000001/shared.smi")
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
    rejected = prepare_case(
        base,
        source,
        model,
        env,
        "rejected-archive",
        False,
        "AA_0000001 2\n",
        library_name="ambiguous",
    )
    rejected_overviews = []
    for path in (rejected / "output-files").rglob("*.json.gz"):
        with gzip.open(path, "rt") as handle:
            data = json.load(handle)
        if "failed_list" in data:
            rejected_overviews.append(data)
    rejected_failures = [
        row for data in rejected_overviews for row in data["failed_list"]
    ]
    assert len(rejected_failures) == 1
    assert rejected_failures[0]["ligand_key"] == "shared"
    assert "Ambiguous duplicate ligand ID" in rejected_failures[0]["reason"]
    assert not list((rejected / "output-files").rglob("*.parquet"))
    evidence = {
        "native_commands_passed": [
            "afvs_prepare_folders.py",
            "afvs_prepare_workunits.py",
            "afvs_submit_jobs.py 1 1",
            "generated workflow/workunits/1/run.bash",
            "create_todofile_atg-primaryscreen.py tranche_ave_score",
        ],
        "prescreen_completions": 4,
        "primary_collection": best,
        "full_library_completions": 16,
        "full_summary_rows": len(actual),
        "visible_failures": len(failures),
        "duplicate_id_logs": len(shared_logs),
        "target_model_initializations": 1,
        "completed_model_batches": native_log.count("PSICHIC+ completed batch"),
        "max_direct_affinity_difference": max(differences),
        "max_replica_affinity_difference": float(
            (actual.score_0 - actual.score_1).abs().max()
        ),
        "mixed_method_completions": 8,
        "all_failure_first_scenario_preserves_later_summary": True,
        "ambiguous_archive_rejected": True,
        "mac_test_only": sys.platform == "darwin",
        "real_model_recovery_after_faults": recovery,
        "gaps": ["CUDA hardware", "Slurm and AWS submission", "cluster scaling"],
    }
    (base / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
