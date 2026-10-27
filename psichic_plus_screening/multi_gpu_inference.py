#!/usr/bin/env python3
"""Data-parallel multi-GPU wrapper for PSICHIC+ screening."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROW_POSITION_COLUMN = "__psichic_multi_gpu_row_position"
OUTPUT_NAME = "inference_results.csv"

MANAGED_OPTIONS = {
    "--input_file",
    "--input-file",
    "--output_folder",
    "--output-folder",
    "--device",
    "--metrics_json",
    "--metrics-json",
    "--validation_json",
    "--validation-json",
    "--shape_metrics_csv",
    "--shape-metrics-csv",
    "--shape_metrics_json",
    "--shape-metrics-json",
}

PRESET_OR_STRATEGY_OPTIONS = {
    "--inference_preset",
    "--inference-preset",
    "--motifpool_dense_strategy",
    "--motifpool-dense-strategy",
    "--protein_mincut_strategy",
    "--protein-mincut-strategy",
    "--protein_dense_cache_strategy",
    "--protein-dense-cache-strategy",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Shard an input CSV across multiple GPUs, run independent inference.py "
            "workers, and merge outputs back in input order."
        )
    )
    parser.add_argument("--input_file", "--input-file", dest="input_file", type=Path, required=True)
    parser.add_argument("--output_folder", "--output-folder", dest="output_folder", type=Path, required=True)
    parser.add_argument(
        "--gpus",
        default=None,
        help="Comma-separated GPU indices. Defaults to CUDA_VISIBLE_DEVICES or 0.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used for shard workers.",
    )
    parser.add_argument(
        "--inference_script",
        "--inference-script",
        dest="inference_script",
        type=Path,
        default=Path(__file__).resolve().parent / "inference.py",
    )
    parser.add_argument(
        "--metrics_json",
        "--metrics-json",
        dest="metrics_json",
        type=Path,
        default=None,
        help="Optional JSON summary path. Defaults to <output_folder>/multi_gpu_metrics.json.",
    )
    parser.add_argument(
        "inference_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to inference.py after an optional '--'.",
    )
    args = parser.parse_args()
    if args.inference_args and args.inference_args[0] == "--":
        args.inference_args = args.inference_args[1:]
    return args


def parse_gpus(value: str | None) -> list[str]:
    if value:
        gpus = [item.strip() for item in value.split(",") if item.strip()]
    else:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        gpus = [item.strip() for item in visible.split(",") if item.strip()]
        if not gpus:
            gpus = ["0"]
    if not gpus:
        raise ValueError("No GPUs were selected")
    return gpus


def has_any_option(args: list[str], names: set[str]) -> bool:
    return any(item in names or any(item.startswith(f"{name}=") for name in names) for item in args)


def reject_managed_options(args: list[str]) -> None:
    present = [item for item in args if item in MANAGED_OPTIONS or any(item.startswith(f"{name}=") for name in MANAGED_OPTIONS)]
    if present:
        joined = ", ".join(sorted(present))
        raise ValueError(f"multi_gpu_inference.py manages these options itself: {joined}")


def ensure_global_ids(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "ID" in df.columns:
        ids = df["ID"].astype(str).replace(["", "nan", "NaN", "None"], pd.NA)
        defaults = pd.Series("Row_" + (df.index.astype(int) + 1).astype(str), index=df.index)
        df["ID"] = ids.fillna(defaults)
    else:
        df["ID"] = "Row_" + (df.index.astype(int) + 1).astype(str)
    return df


def write_shards(df: pd.DataFrame, gpus: list[str], work_dir: Path) -> list[dict]:
    if ROW_POSITION_COLUMN in df.columns:
        raise ValueError(f"Input CSV already contains reserved column {ROW_POSITION_COLUMN}")

    df = ensure_global_ids(df)
    df[ROW_POSITION_COLUMN] = range(len(df))
    shard_dir = work_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    shards = []
    for shard_index, gpu in enumerate(gpus):
        shard_df = df.iloc[shard_index::len(gpus)].copy()
        shard_csv = shard_dir / f"shard_{shard_index:03d}.csv"
        shard_df.to_csv(shard_csv, index=False)
        shards.append(
            {
                "shard_index": shard_index,
                "gpu": gpu,
                "rows": int(len(shard_df)),
                "csv": shard_csv,
                "output_folder": work_dir / f"shard_{shard_index:03d}_output",
                "log": work_dir / f"shard_{shard_index:03d}.log",
                "metrics": work_dir / f"shard_{shard_index:03d}_metrics.json",
                "validation": work_dir / f"shard_{shard_index:03d}_validation.json",
            }
        )
    return shards


def build_worker_command(args: argparse.Namespace, shard: dict, forwarded_args: list[str]) -> list[str]:
    command = [
        args.python,
        str(args.inference_script),
        "--input_file",
        str(shard["csv"]),
        "--output_folder",
        str(shard["output_folder"]),
        "--device",
        f"cuda:{shard['gpu']}",
        "--metrics_json",
        str(shard["metrics"]),
        "--validation_json",
        str(shard["validation"]),
    ]
    if not has_any_option(forwarded_args, PRESET_OR_STRATEGY_OPTIONS):
        command.extend(["--inference_preset", "fast_screening"])
    if not has_any_option(forwarded_args, {"--protein_cache_dir", "--protein-cache-dir"}):
        command.extend(["--protein_cache_dir", str(shard["output_folder"] / "cache" / "proteins")])
    if not has_any_option(forwarded_args, {"--ligand_cache_dir", "--ligand-cache-dir"}):
        command.extend(["--ligand_cache_dir", str(shard["output_folder"] / "cache" / "ligands")])
    command.extend(forwarded_args)
    return command


def run_workers(args: argparse.Namespace, shards: list[dict], forwarded_args: list[str]) -> list[dict]:
    processes = []
    start = time.perf_counter()
    for shard in shards:
        shard["output_folder"].mkdir(parents=True, exist_ok=True)
        command = build_worker_command(args, shard, forwarded_args)
        shard["command"] = command
        log_handle = open(shard["log"], "w")
        process = subprocess.Popen(command, stdout=log_handle, stderr=subprocess.STDOUT)
        processes.append((shard, process, log_handle))

    failures = []
    for shard, process, log_handle in processes:
        returncode = process.wait()
        log_handle.close()
        shard["returncode"] = returncode
        if returncode != 0:
            failures.append(shard)

    elapsed_s = time.perf_counter() - start
    if failures:
        failed = ", ".join(f"shard {item['shard_index']} gpu {item['gpu']}" for item in failures)
        raise RuntimeError(f"One or more shard workers failed: {failed}")
    for shard in shards:
        shard["elapsed_s"] = elapsed_s
    return shards


def merge_outputs(input_rows: int, shards: list[dict], output_folder: Path) -> tuple[Path, dict]:
    frames = []
    for shard in shards:
        result_csv = shard["output_folder"] / OUTPUT_NAME
        if not result_csv.exists():
            raise FileNotFoundError(f"Missing shard output: {result_csv}")
        frame = pd.read_csv(result_csv)
        if ROW_POSITION_COLUMN not in frame.columns:
            raise ValueError(f"Shard output is missing {ROW_POSITION_COLUMN}: {result_csv}")
        frames.append(frame)

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.sort_values(ROW_POSITION_COLUMN, kind="stable")
    if len(merged) != input_rows:
        raise ValueError(f"Merged row count mismatch: expected {input_rows}, got {len(merged)}")
    if merged[ROW_POSITION_COLUMN].tolist() != list(range(input_rows)):
        raise ValueError("Merged row positions are not exactly the original input order")
    merged = merged.drop(columns=[ROW_POSITION_COLUMN])

    output_csv = output_folder / OUTPUT_NAME
    merged.to_csv(output_csv, index=False)
    protected = ["predicted_binding_affinity", "reg_alpha", "reg_beta"]
    complete_mask = merged[protected].notna().all(axis=1) if all(column in merged.columns for column in protected) else pd.Series(False, index=merged.index)
    return output_csv, {
        "rows": int(len(merged)),
        "protected_complete_rows": int(complete_mask.sum()),
        "ids_unique": bool(merged["ID"].is_unique) if "ID" in merged.columns else False,
        "output_csv": str(output_csv),
    }


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as handle:
        return json.load(handle)


def write_summary(args: argparse.Namespace, gpus: list[str], shards: list[dict], merge_summary: dict, elapsed_s: float) -> None:
    metrics_path = args.metrics_json or (args.output_folder / "multi_gpu_metrics.json")
    payload = {
        "status": "pass",
        "input_file": str(args.input_file),
        "output_folder": str(args.output_folder),
        "gpus": gpus,
        "rows": merge_summary["rows"],
        "protected_complete_rows": merge_summary["protected_complete_rows"],
        "wall_time_s": elapsed_s,
        "throughput_pairs_per_s": merge_summary["rows"] / elapsed_s if elapsed_s > 0 else 0.0,
        "output_csv": merge_summary["output_csv"],
        "shards": [
            {
                "shard_index": shard["shard_index"],
                "gpu": shard["gpu"],
                "rows": shard["rows"],
                "returncode": shard["returncode"],
                "log": str(shard["log"]),
                "metrics": str(shard["metrics"]),
                "validation": str(shard["validation"]),
                "worker_metrics": load_json(shard["metrics"]),
            }
            for shard in shards
        ],
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> None:
    args = parse_args()
    reject_managed_options(args.inference_args)
    gpus = parse_gpus(args.gpus)
    args.output_folder.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_file)
    if len(df) == 0:
        raise ValueError("Input CSV is empty")
    gpus = gpus[: min(len(gpus), len(df))]
    start = time.perf_counter()
    shards = write_shards(df, gpus, args.output_folder)
    run_workers(args, shards, args.inference_args)
    output_csv, merge_summary = merge_outputs(len(df), shards, args.output_folder)
    elapsed_s = time.perf_counter() - start
    write_summary(args, gpus, shards, merge_summary, elapsed_s)
    print(f"Saved merged predictions to {output_csv}")


if __name__ == "__main__":
    main()
