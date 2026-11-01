"""Run one local AFVS prescreen and adaptive primary screen with PSICHIC+."""

import argparse
import csv
import gzip
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile

import pandas as pd

from library import FIELDS, collection_key, convert_library, write_csv


UPSTREAM_REVISION = "df74e005e456e94654fa93a40d748de399450b81"
HERE = Path(__file__).resolve().parent
RESULT_FIELDS = FIELDS + [
    "status",
    "predicted_binding_affinity",
    "score",
    "reason",
    "seconds",
    "worker_pid",
    "batch_number",
]


def check_upstream(upstream):
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=upstream, text=True
    ).strip()
    if revision != UPSTREAM_REVISION:
        raise ValueError(
            f"Use AdaptiveFlow-VS revision {UPSTREAM_REVISION}, found {revision}"
        )
    subprocess.run(
        ["git", "diff", "--exit-code", "HEAD", "--", "tools"], cwd=upstream, check=True
    )


def run_command(command, cwd, log):
    env = dict(
        os.environ, AWS_EC2_METADATA_DISABLED="true", AWS_DEFAULT_REGION="us-east-1"
    )
    with Path(log).open("w") as handle:
        subprocess.run(
            [str(arg) for arg in command],
            cwd=cwd,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=True,
        )


def prepare_stage(
    upstream, stage, library, todo, predictor_config, batch_size, samples_per_tranche=0
):
    stage = Path(stage).resolve()
    tools = stage / "tools"
    (tools / "templates").mkdir(parents=True)
    (stage / "tmp").mkdir()
    inputs = stage / "input-files"
    (inputs / "psichic").mkdir(parents=True)
    (inputs / "psichic" / "config.txt").write_text(json.dumps(predictor_config))
    total = sum(int(line.split()[1]) for line in Path(todo).read_text().splitlines())
    if not total:
        raise ValueError("Todo file has no ligands")
    config = {
        "job_name": "psichic",
        "threads_to_use": 1,
        "threads_per_docking": 1,
        "job_storage_mode": "sharedfs",
        "data_storage_mode": "sharedfs",
        "data_collection_addressing_mode": "meta-tranche",
        "job_addressing_mode": "meta-tranche",
        "data_collection_identifier": "",
        "object_store_job_prefix": "local",
        "object_store_job_bucket": "unused",
        "batchsystem": "bash",
        "bash_template": str(upstream / "tools/templates/template1.bash"),
        "bash_array_job_size": 1,
        "dockings_per_subjob": total + 1,
        "ligand_library_format": "json",
        "program_timeout": 3600,
        "docking_scenario_basefolder": str(inputs),
        "docking_scenario_names": "psichic",
        "docking_scenario_methods": "psichic_plus",
        "docking_scenario_replicas": 1,
        "docking_scenario_batchsizes": batch_size,
        "summary_formats": "csv.gz,parquet",
        "run_atom_check": 0,
        "dynamic_tranche_filtering": 0,
        "tempdir_default": str(stage / "tmp"),
        "collection_list_type": "standard",
        "collection_list_file": str(Path(todo).resolve()),
        "collection_folder": str(Path(library).resolve() / "collections"),
        "prescreen_mode": int(samples_per_tranche > 0),
        "prescreen_ligands_per_tranche": samples_per_tranche,
        "aws_region": "us-east-1",
    }
    (tools / "templates/all.ctrl").write_text(
        "".join(f"{key}={value}\n" for key, value in config.items())
    )
    for name in ["afvs_prepare_folders.py", "afvs_prepare_workunits.py"]:
        run_command(
            [sys.executable, upstream / "tools" / name], tools, stage / f"{name}.log"
        )
    status = json.loads((stage / "workflow/status.json").read_text())
    if list(status["workunits"]) != ["1"] or list(
        status["workunits"]["1"]["subjobs"]
    ) != ["0"]:
        raise RuntimeError("Expected one workunit and one subjob")
    return stage / "workflow/workunits/1.tar.gz"


def execute_stage(upstream, stage, start_method):
    run_command(
        [
            sys.executable,
            HERE / "worker.py",
            "--upstream",
            upstream,
            "--workunit",
            stage / "workflow/workunits/1.tar.gz",
            "--temp-dir",
            stage / "tmp",
            "--start-method",
            start_method,
        ],
        HERE.parents[1],
        stage / "worker.log",
    )


def collect_results(stage, manifest):
    output = stage / "output-files"
    with gzip.open(output / "summary/1/0.json.gz", "rt") as handle:
        overview = json.load(handle)
    if overview["failed_downloads"] or overview["skipped_ligands"]:
        raise RuntimeError(f"Unexpected upstream download failures or skips in {stage}")
    results = []
    with tarfile.open(output / "psichic/logs/1/0.tar.gz") as archive:
        for member in archive:
            if member.isfile() and member.name.endswith("/output"):
                record = json.load(archive.extractfile(member))
                expected = manifest[record["ligand_id"]]
                if any(record[field] != expected[field] for field in FIELDS):
                    raise RuntimeError("Worker output does not match the manifest")
                if record["status"] == "success":
                    if (
                        not math.isfinite(record["predicted_binding_affinity"])
                        or record["score"] != -record["predicted_binding_affinity"]
                    ):
                        raise RuntimeError("Invalid successful score")
                elif (
                    record["score"] is not None
                    or record["predicted_binding_affinity"] is not None
                    or not record["reason"]
                ):
                    raise RuntimeError("Failed ligand needs a reason and null scores")
                results.append(record)
    if (
        len({row["ligand_id"] for row in results}) != len(results)
        or len(results) != overview["total_dockings"]
    ):
        raise RuntimeError("Missing or duplicate worker results")
    successes = {row["ligand_id"]: row for row in results if row["status"] == "success"}
    if len(successes) != overview["dockings_status"]["success"]:
        raise RuntimeError("Worker results disagree with the upstream overview")
    if successes:
        summary = pd.read_parquet(output / "psichic/parquet/1/0.parquet")
        if set(summary["ligand"]) != set(successes):
            raise RuntimeError("Worker results disagree with the upstream summary")
        for row in summary.to_dict("records"):
            record = successes[row["ligand"]]
            if row["score_min"] != record["score"] or row[
                "collection_key"
            ] != collection_key(record):
                raise RuntimeError(
                    "Upstream summary lost a score or collection identity"
                )
    failures = {row["ligand_key"]: row["reason"] for row in overview["failed_list"]}
    if failures != {
        row["ligand_id"]: row["reason"] for row in results if row["status"] == "failed"
    }:
        raise RuntimeError("Worker results disagree with upstream failure records")
    results.sort(
        key=lambda row: (
            row["status"] != "success",
            row["score"] if row["score"] is not None else 0,
            row["ligand_id"],
        )
    )
    write_csv(stage / "predictions.csv", RESULT_FIELDS, results)
    return results


def select_primary(upstream, library, results, budget, tranche_regex, destination):
    regex = re.compile(tranche_regex)
    inventory = pd.read_parquet(library / "collections.parquet")
    available = inventory.loc[
        inventory["Tranche"].map(lambda value: bool(regex.fullmatch(value))),
        "LigandCount",
    ].sum()
    if budget < 1 or budget > available:
        raise ValueError(
            f"Primary budget must be between 1 and {available} for the tranche regex"
        )
    successes = [row for row in results if row["status"] == "success"]
    if not successes:
        raise ValueError(
            "Prescreen has no successful predictions for upstream selection"
        )
    destination.mkdir()
    scores_path = destination / "prescreen.csv.gz"
    with gzip.open(scores_path, "wt", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Tranche", "Collection", "LigandAFID", "ScoreMin"])
        writer.writerows(
            [row["tranche"], row["collection_id"], row["ligand_id"], row["score"]]
            for row in successes
        )
    todo = destination / "primary.todo"
    run_command(
        [
            sys.executable,
            upstream / "tools/templates/create_todofile_atg-primaryscreen.py",
            scores_path,
            library / "collections.parquet",
            library / "tranches.parquet",
            "tranche_min_score",
            todo,
            budget,
            tranche_regex,
        ],
        destination,
        destination / "selector.log",
    )
    return todo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--target-fasta", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--budget",
        type=int,
        required=True,
        help="Primary ligand budget, rounded up to whole collections",
    )
    parser.add_argument("--samples-per-tranche", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tranche-regex", default=".*")
    parser.add_argument("--start-method", choices=["spawn", "fork"], default="spawn")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    upstream = args.upstream.resolve()
    check_upstream(upstream)
    output = args.output.resolve()
    output.mkdir(parents=True)
    records = convert_library(
        args.library, output / "library", args.samples_per_tranche
    )
    manifest = {row["ligand_id"]: row for row in records}
    config = {
        "target_fasta": str(args.target_fasta.resolve()),
        "model_dir": str(args.model_dir.resolve()),
        "device": args.device,
        "threads": 1,
    }
    library = output / "library"
    prescreen = output / "prescreen"
    prepare_stage(
        upstream,
        prescreen,
        library,
        library / "prescreen.todo",
        config,
        args.batch_size,
        args.samples_per_tranche,
    )
    execute_stage(upstream, prescreen, args.start_method)
    results = collect_results(prescreen, manifest)
    todo = select_primary(
        upstream,
        library,
        results,
        args.budget,
        args.tranche_regex,
        output / "selection",
    )
    primary = output / "primary"
    prepare_stage(upstream, primary, library, todo, config, args.batch_size)
    execute_stage(upstream, primary, args.start_method)
    primary_results = collect_results(primary, manifest)
    combined = {row["ligand_id"]: row for row in results + primary_results}
    ranked = sorted(
        combined.values(),
        key=lambda row: (
            row["status"] != "success",
            row["score"] or 0,
            row["ligand_id"],
        ),
    )
    write_csv(output / "ranked.csv", RESULT_FIELDS, ranked)
    print(f"Ranked model predictions and failures written to {output / 'ranked.csv'}")


if __name__ == "__main__":
    main()
