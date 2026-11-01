"""Rerun real CPU inference parity and the external adaptive selector."""

import argparse
import csv
import json
import math
from pathlib import Path
import sys
import tarfile

from library import FIELDS, convert_library, write_csv
from run import HERE, check_upstream, prepare_stage, run_command, select_primary


ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
CAFFEINE = "Cn1c(=O)c2c(ncn2C)n(C)c1=O"


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def verify_selector(upstream, output, config):
    source = output / "synthetic.csv"
    rows = [
        {
            "molecule_id": "duplicate",
            "collection_id": number,
            "tranche": tranche,
            "smiles": ASPIRIN,
        }
        for tranche in ["AA", "BB"]
        for number in ["0000001", "0000042"]
        for _ in range(2)
    ]
    write_csv(source, FIELDS[1:], rows)
    library = output / "synthetic-library"
    records = convert_library(source, library, 1)
    results = [
        {**row, "status": "success", "score": -2.0 if row["tranche"] == "AA" else -9.0}
        for row in records
    ]
    todo = select_primary(
        upstream, library, results, 3, ".*", output / "synthetic-selection"
    )
    selected = dict(line.split() for line in todo.read_text().splitlines())
    assert selected == {"BB_0000001": "2", "BB_0000042": "2"}, selected
    stage = output / "synthetic-primary"
    workunit = prepare_stage(upstream, stage, library, todo, config, 2)
    with tarfile.open(workunit) as archive:
        payload = json.load(archive.extractfile("af_input/config.json"))
    collections = payload["subjobs"]["0"]["collections"]
    assert set(collections) == set(selected), collections
    assert all(row["mode"] == "all" for row in collections.values())
    assert all(Path(row["sharedfs_path"]).is_file() for row in collections.values())
    filtered = select_primary(
        upstream, library, results, 1, "AA", output / "synthetic-filtered"
    )
    assert all(line.startswith("AA_") for line in filtered.read_text().splitlines())
    return {
        "selected_collections": selected,
        "budget": 3,
        "selected_ligands": 4,
        "regex_filter": "pass",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=HERE.parents[1]
        / "psichic_plus_screening/pretrained_weights/PSICHIC_plus",
    )
    parser.add_argument("--start-method", choices=["spawn", "fork"], default="spawn")
    args = parser.parse_args()
    upstream = args.upstream.resolve()
    check_upstream(upstream)
    output = args.output.resolve()
    output.mkdir(parents=True)
    target = output / "target.fasta"
    target.write_text(">synthetic-test-target\nMKTAYIAKQRQISFVKSHFSRQDILD\n")
    source = output / "library.csv"
    rows = [
        {
            "molecule_id": "duplicate,id",
            "collection_id": "0000001",
            "tranche": "AA",
            "smiles": ASPIRIN,
        },
        {
            "molecule_id": "caffeine",
            "collection_id": "0000001",
            "tranche": "AA",
            "smiles": CAFFEINE,
        },
        {
            "molecule_id": "duplicate,id",
            "collection_id": "0000042",
            "tranche": "BB",
            "smiles": ASPIRIN,
        },
        {
            "molecule_id": "invalid",
            "collection_id": "0000042",
            "tranche": "BB",
            "smiles": "invalid",
        },
    ]
    write_csv(source, FIELDS[1:], rows)
    actual = output / "actual"
    command = [
        sys.executable,
        HERE / "run.py",
        "--upstream",
        upstream,
        "--library",
        source,
        "--target-fasta",
        target,
        "--model-dir",
        args.model_dir.resolve(),
        "--output",
        actual,
        "--budget",
        "2",
        "--samples-per-tranche",
        "2",
        "--batch-size",
        "2",
        "--device",
        "cpu",
        "--start-method",
        args.start_method,
    ]
    run_command(command, HERE.parents[1], output / "integration.log")
    manifest = read_csv(actual / "library/manifest.csv")
    assert len({row["ligand_id"] for row in manifest}) == 4
    assert [{field: row[field] for field in FIELDS[1:]} for row in manifest] == rows
    write_csv(
        output / "direct.csv",
        ["ID", "Ligand"],
        [{"ID": row["ligand_id"], "Ligand": row["smiles"]} for row in manifest],
    )
    run_command(
        [
            sys.executable,
            HERE.parents[1] / "psichic_plus_screening/inference.py",
            "--input_file",
            output / "direct.csv",
            "--protein_fasta",
            target,
            "--output_folder",
            output / "direct",
            "--model_dir",
            args.model_dir.resolve(),
            "--device",
            "cpu",
            "--batch_size",
            "2",
            "--num_workers",
            "0",
            "--inference_preset",
            "conservative",
            "--precision",
            "fp32",
            "--invalid_ligand_policy",
            "emit_null",
            "--no-ligand-disk-cache",
        ],
        output,
        output / "direct.log",
    )
    direct = {
        row["ID"]: row for row in read_csv(output / "direct/inference_results.csv")
    }
    prescreen = read_csv(actual / "prescreen/predictions.csv")
    assert len(prescreen) == 4
    assert {row["batch_number"] for row in prescreen} == {"1", "2"}
    assert len({row["worker_pid"] for row in prescreen}) == 1
    worker_log = (actual / "prescreen/worker.log").read_text()
    assert worker_log.count("model and target initialized in docking child") == 1
    differences = []
    for row in prescreen:
        prediction = direct[row["ligand_id"]]
        assert row["status"] == prediction["inference_status"]
        if row["status"] == "success":
            affinity = float(row["predicted_binding_affinity"])
            assert math.isfinite(affinity)
            assert float(row["score"]) == -affinity
            difference = abs(affinity - float(prediction["predicted_binding_affinity"]))
            assert difference <= 1e-5, (row, prediction)
            differences.append(difference)
        else:
            assert not row["score"] and not row["predicted_binding_affinity"]
            assert row["reason"].startswith("INVALID_SMILES")
    successes = [row for row in prescreen if row["status"] == "success"]
    affinities = [float(row["predicted_binding_affinity"]) for row in successes]
    assert affinities == sorted(affinities, reverse=True)
    duplicates = [row for row in prescreen if row["molecule_id"] == "duplicate,id"]
    assert (
        len(duplicates) == 2 and len({row["collection_id"] for row in duplicates}) == 2
    )
    assert duplicates[0]["smiles"] == duplicates[1]["smiles"] == ASPIRIN
    assert abs(float(duplicates[0]["score"]) - float(duplicates[1]["score"])) <= 1e-5
    ranked = read_csv(actual / "ranked.csv")
    assert len(ranked) == 4 and sum(row["status"] == "failed" for row in ranked) == 1
    config = {
        "target_fasta": str(target),
        "model_dir": str(args.model_dir.resolve()),
        "device": "cpu",
        "threads": 1,
    }
    selection = verify_selector(upstream, output, config)
    report = {
        "real_cpu": "pass",
        "prescreen_batches": 2,
        "prescreen_model_initializations": 1,
        "successes": 3,
        "visible_failures": 1,
        "max_direct_affinity_difference": max(differences),
        "upstream_selector": selection,
        "start_method": args.start_method,
        "docker": "not exercised by this CPU verifier",
        "cuda": "not exercised by this CPU verifier",
        "integration_command": [str(part) for part in command],
    }
    (output / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
