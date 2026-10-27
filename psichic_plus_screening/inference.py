#!/usr/bin/env python3
"""Standalone PSICHIC+ production inference pipeline."""

from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
import os
import queue
import shutil
import socket
import threading
import time
import traceback
from collections import Counter
from contextlib import contextmanager, nullcontext
from pathlib import Path

import pandas as pd
import torch

from featurizers import (
    DataLoader,
    ProteinMoleculeDataset,
    build_interpret_result_payload,
    diagnose_ligand_failure,
    protein_init_with_keys,
    save_interpret_result_payload,
    store_result,
)
from psichic_model import (
    DEFAULT_MODEL_DIR,
    INFERENCE_PRESET_CHOICES,
    MOTIFPOOL_DENSE_STRATEGIES,
    PROTEIN_DENSE_CACHE_STRATEGIES,
    PROTEIN_MINCUT_STRATEGIES,
    PSICHICPlusModel,
    resolve_inference_preset,
)

VALIDATION_COLUMNS = (
    "predicted_binding_affinity",
    "reg_alpha",
    "reg_beta",
)

BATCHING_STRATEGIES = (
    "input_order",
    "clique_count_bucketed",
    "protein_size_bucketed",
)

INVALID_LIGAND_POLICIES = (
    "fail",
    "emit_null",
)

PRECISION_MODES = (
    "fp32",
    "tf32",
    "bf16",
    "fp16",
)

STATUS_COLUMNS = (
    "inference_status",
    "failure_stage",
    "failure_reason",
)


def add_bool_argument(
    parser: argparse.ArgumentParser,
    name: str,
    default: bool,
    help_text: str,
) -> None:
    option = f"--{name}"
    negative_option = f"--no-{name.replace('_', '-')}"
    parser.add_argument(option, dest=name, action="store_true", help=help_text)
    parser.add_argument(negative_option, dest=name, action="store_false", help=f"Disable {help_text.lower()}")
    parser.set_defaults(**{name: default})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PSICHIC+ production inference")
    parser.add_argument("--input_file", type=Path, required=True, help="Path to the PSICHIC+ input CSV.")
    parser.add_argument("--output_folder", type=Path, required=True, help="Directory for outputs.")
    parser.add_argument(
        "--model_dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Directory with config.json, degree.pt, and model.pt.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument(
        "--inference_preset",
        "--inference-preset",
        dest="inference_preset",
        choices=INFERENCE_PRESET_CHOICES,
        default="fast_screening",
        help=(
            "Inference preset. fast_screening is the default screening path; "
            "conservative restores the historical strategy trio; none honors explicit strategy flags."
        ),
    )
    parser.add_argument(
        "--batching_strategy",
        choices=BATCHING_STRATEGIES,
        default="input_order",
        help=(
            "Batch ordering strategy. input_order preserves current behavior; "
            "clique_count_bucketed and protein_size_bucketed are opt-in profiling candidates."
        ),
    )
    parser.add_argument(
        "--motifpool_dense_strategy",
        "--motifpool-dense-strategy",
        dest="motifpool_dense_strategy",
        choices=MOTIFPOOL_DENSE_STRATEGIES,
        default="pyg_to_dense",
        help="MotifPool dense conversion strategy. pyg_to_dense preserves current behavior; scatter strategies are opt-in candidates.",
    )
    parser.add_argument(
        "--protein_mincut_strategy",
        "--protein-mincut-strategy",
        dest="protein_mincut_strategy",
        choices=PROTEIN_MINCUT_STRATEGIES,
        default="pyg_dense_mincut",
        help="Protein dense MinCut strategy. pyg_dense_mincut preserves current behavior; inference_no_loss skips auxiliary losses in eval inference.",
    )
    parser.add_argument(
        "--protein_dense_cache_strategy",
        "--protein-dense-cache-strategy",
        dest="protein_dense_cache_strategy",
        choices=PROTEIN_DENSE_CACHE_STRATEGIES,
        default="none",
        help="Protein-side dense cache strategy. none preserves current behavior; protein_dense_inputs reuses safe layer-0 protein-only dense intermediates.",
    )
    parser.add_argument(
        "--collect_batching_metrics",
        action="store_true",
        help="Collect clique-count batch padding metrics without collecting full shape metrics.",
    )
    parser.add_argument(
        "--ligand_cache_dir",
        type=Path,
        default=None,
        help="Optional persistent ligand cache directory. Defaults to <output_folder>/ligands.",
    )
    parser.add_argument(
        "--protein_cache_dir",
        type=Path,
        default=None,
        help="Optional persistent protein cache directory keyed by Protein. Defaults to in-memory protein initialization only.",
    )
    parser.add_argument(
        "--protein_fasta",
        type=Path,
        default=None,
        help="Optional FASTA file for single-protein screening inputs. Overrides Protein and Protein_Sequence columns.",
    )
    parser.add_argument(
        "--protein_name",
        default=None,
        help="Protein key to use with --protein_fasta. Defaults to the FASTA header.",
    )
    parser.add_argument(
        "--ligand_column",
        default=None,
        help="Input CSV column containing SMILES. Defaults to Ligand, or smiles when Ligand is absent.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=None,
        help="Stream input CSV in chunks of this many rows. Intended for large single-protein screening inputs.",
    )
    parser.add_argument(
        "--invalid_ligand_policy",
        "--invalid-ligand-policy",
        dest="invalid_ligand_policy",
        choices=INVALID_LIGAND_POLICIES,
        default="emit_null",
        help=(
            "How to handle rows whose ligand cannot be parsed, Papyrus-standardized, or featurized. "
            "Supported ligands are always standardized before graph construction; this flag does not disable normalization. "
            "emit_null preserves the row with null prediction columns and status metadata; "
            "fail raises an error before writing that chunk."
        ),
    )
    parser.add_argument("--save_interpret", action="store_true", help="Save interpretation outputs.")
    parser.add_argument(
        "--interpret_writer_backend",
        choices=("thread", "process"),
        default="process",
        help="Background worker backend for --save_interpret result serialization.",
    )
    add_bool_argument(
        parser,
        "ligand_disk_cache",
        True,
        "Read and write hashed ligand graph .pt files on disk.",
    )
    add_bool_argument(parser, "pin_memory", True, "Enable pinned host memory in the DataLoader.")
    add_bool_argument(
        parser,
        "persistent_workers",
        False,
        "Enable persistent DataLoader workers.",
    )
    add_bool_argument(
        parser,
        "preserve_ligand_cache",
        False,
        "Keep the hashed ligand cache directory after inference.",
    )
    add_bool_argument(
        parser,
        "preserve_protein_cache",
        False,
        "Keep the protein cache directory after inference when using the default cache location.",
    )
    add_bool_argument(
        parser,
        "inference_mode",
        False,
        "Use torch.inference_mode() instead of torch.no_grad().",
    )
    add_bool_argument(
        parser,
        "amp",
        False,
        "Enable fp16 autocast on CUDA during inference. Legacy alias for --precision fp16.",
    )
    add_bool_argument(
        parser,
        "tf32",
        False,
        "Enable TF32 matmul and cuDNN on Ampere-class GPUs. Legacy alias for --precision tf32.",
    )
    parser.add_argument(
        "--precision",
        choices=PRECISION_MODES,
        default="fp32",
        help=(
            "Inference precision mode. fp32 is the control path; tf32 enables Ampere TF32; "
            "bf16 and fp16 use CUDA autocast. Existing --tf32 and --amp flags remain aliases."
        ),
    )
    add_bool_argument(
        parser,
        "nvtx",
        False,
        "Emit NVTX ranges for Nsight Systems profiling.",
    )
    parser.add_argument(
        "--compile_mode",
        default="none",
        help="torch.compile mode to use for the loaded model. Use 'none' to disable.",
    )
    parser.add_argument(
        "--metrics_json",
        type=Path,
        default=None,
        help="Optional path for structured runtime and validation metrics.",
    )
    parser.add_argument(
        "--shape_metrics_csv",
        type=Path,
        default=None,
        help="Optional path for per-batch graph shape metrics as CSV.",
    )
    parser.add_argument(
        "--shape_metrics_json",
        type=Path,
        default=None,
        help="Optional path for per-batch graph shape metrics as JSON.",
    )
    parser.add_argument(
        "--validation_json",
        type=Path,
        default=None,
        help="Optional path for validation-only metrics.",
    )
    parser.add_argument(
        "--validation_tolerance",
        type=float,
        default=1.0e-5,
        help="Maximum allowed absolute delta for protected prediction columns.",
    )
    parser.add_argument(
        "--sync_timing",
        "--sync-timing",
        dest="sync_timing",
        action="store_true",
        help="Synchronize CUDA around batch timing phases for profiling. Adds overhead.",
    )
    return parser.parse_args()


def resolve_cli_presets(args: argparse.Namespace) -> None:
    requested_strategies = {
        "motifpool_dense_strategy": args.motifpool_dense_strategy,
        "protein_mincut_strategy": args.protein_mincut_strategy,
        "protein_dense_cache_strategy": args.protein_dense_cache_strategy,
    }
    resolved = resolve_inference_preset(
        args.inference_preset,
        args.motifpool_dense_strategy,
        args.protein_mincut_strategy,
        args.protein_dense_cache_strategy,
    )
    args.requested_strategies = requested_strategies
    args.resolved_inference_preset = resolved["inference_preset"]
    args.motifpool_dense_strategy = resolved["motifpool_dense_strategy"]
    args.protein_mincut_strategy = resolved["protein_mincut_strategy"]
    args.protein_dense_cache_strategy = resolved["protein_dense_cache_strategy"]


def ensure_ids(df: pd.DataFrame, row_offset: int = 0) -> pd.DataFrame:
    if "ID" in df.columns:
        id_series = df["ID"].astype(str)
        id_series = id_series.replace(["", "nan", "NaN", "None"], pd.NA)
        default_ids = pd.Series("Row_" + (df.index + row_offset + 1).astype(str), index=df.index)
        ids = id_series.fillna(default_ids).tolist()
    else:
        ids = ("Row_" + (df.index + row_offset + 1).astype(str)).tolist()
    df = df.copy()
    df["ID"] = ids
    return df


def read_single_fasta(path: Path) -> tuple[str, str]:
    name: str | None = None
    sequence_parts: list[str] = []
    with open(path) as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    raise ValueError(f"Expected one FASTA record in {path}, found multiple records")
                name = line[1:].split()[0]
            else:
                sequence_parts.append(line)
    if name is None or not sequence_parts:
        raise ValueError(f"No FASTA sequence found in {path}")
    return name, "".join(sequence_parts)


def prepare_input_frame(
    df: pd.DataFrame,
    ligand_column: str | None,
    fasta_record: tuple[str, str] | None,
    protein_name: str | None,
    row_offset: int = 0,
) -> pd.DataFrame:
    df = df.copy()
    selected_ligand_column = ligand_column
    if selected_ligand_column is None:
        if "Ligand" in df.columns:
            selected_ligand_column = "Ligand"
        elif "smiles" in df.columns:
            selected_ligand_column = "smiles"
    if selected_ligand_column is None or selected_ligand_column not in df.columns:
        raise ValueError("Input CSV must contain a Ligand column, a smiles column, or pass --ligand_column")
    if selected_ligand_column != "Ligand":
        df["Ligand"] = df[selected_ligand_column]

    if fasta_record is not None:
        fasta_name, fasta_sequence = fasta_record
        df["Protein"] = protein_name or fasta_name
        df["Protein_Sequence"] = fasta_sequence

    if "Protein" not in df.columns:
        raise ValueError("Input CSV must contain Protein, or pass --protein_fasta for single-protein screening")

    return ensure_ids(df, row_offset=row_offset)


def extract_protein_keys_and_seqs(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    if "Protein_Sequence" in df.columns:
        all_proteins = df["Protein"].tolist()
        all_seqs = df["Protein_Sequence"].tolist()
        protein_seqs = dict(zip(all_proteins, all_seqs))
        keys = list(protein_seqs.keys())
        seqs = list(protein_seqs.values())
    else:
        keys = df["Protein"].unique().tolist()
        seqs = copy.deepcopy(keys)
    return keys, seqs


def load_cached_protein_dict(keys: list[str], protein_cache_dir: Path) -> dict:
    return {key: torch.load(protein_cache_dir / f"{key}.pt", weights_only=False) for key in keys}


def build_protein_state(
    df: pd.DataFrame,
    protein_cache_dir: Path | None,
    save_interpret: bool,
) -> tuple[dict, Path | None, float, int, int, str]:
    keys, seqs = extract_protein_keys_and_seqs(df)
    start = time.perf_counter()

    if protein_cache_dir is None:
        protein_dict = protein_init_with_keys(keys, seqs)
        return protein_dict, None, time.perf_counter() - start, 0, len(keys), "in_memory"

    protein_cache_dir.mkdir(parents=True, exist_ok=True)
    missing_pairs = [(key, seq) for key, seq in zip(keys, seqs) if not (protein_cache_dir / f"{key}.pt").exists()]
    protein_cache_hits = len(keys) - len(missing_pairs)
    protein_cache_misses = len(missing_pairs)

    if missing_pairs:
        missing_keys = [key for key, _ in missing_pairs]
        missing_seqs = [seq for _, seq in missing_pairs]
        missing_dict = protein_init_with_keys(missing_keys, missing_seqs)
        for key, protein_value in missing_dict.items():
            torch.save(protein_value, protein_cache_dir / f"{key}.pt")

    protein_dict = {}
    if save_interpret:
        protein_dict = load_cached_protein_dict(keys, protein_cache_dir)

    return (
        protein_dict,
        protein_cache_dir,
        time.perf_counter() - start,
        protein_cache_hits,
        protein_cache_misses,
        "disk_cache",
    )


def ligand_clique_count(graph: dict) -> int:
    value = graph.get("clique_num_nodes")
    if value is not None:
        return int(value)
    clique_x = graph.get("x_clique")
    if clique_x is not None:
        return int(clique_x.size(0))
    raise ValueError("Ligand graph is missing clique count metadata")


def ligand_atom_count(graph: dict) -> int:
    value = graph.get("atom_num_nodes")
    if value is not None:
        return int(value)
    atom_x = graph.get("atom_idx")
    if atom_x is not None:
        return int(atom_x.size(0))
    raise ValueError("Ligand graph is missing atom count metadata")


def protein_residue_count(graph: dict) -> int:
    value = graph.get("num_nodes")
    if value is not None:
        return int(value)
    seq = graph.get("seq")
    if seq is not None:
        return int(len(seq))
    seq_feat = graph.get("seq_feat")
    if seq_feat is not None:
        return int(seq_feat.size(0))
    raise ValueError("Protein graph is missing residue count metadata")


def protein_edge_count(graph: dict) -> int:
    edge_index = graph.get("edge_index")
    if edge_index is None:
        return 0
    return int(edge_index.size(1))


def compute_ligand_clique_counts(
    df: pd.DataFrame,
    dataset: ProteinMoleculeDataset,
) -> tuple[list[int], int]:
    counts: list[int] = []
    ligand_counts: dict[str, int] = {}
    failures = 0
    for ligand_value in df["Ligand"].astype(str).tolist():
        if ligand_value not in ligand_counts:
            try:
                ligand_counts[ligand_value] = ligand_clique_count(dataset.mol_graph(ligand_value))
            except Exception:
                ligand_counts[ligand_value] = 0
                failures += 1
        counts.append(ligand_counts[ligand_value])
    return counts, failures


def compute_size_shape_records(
    df: pd.DataFrame,
    dataset: ProteinMoleculeDataset,
) -> tuple[list[dict], int]:
    records: list[dict] = []
    ligand_shapes: dict[str, tuple[int, int]] = {}
    protein_shapes: dict[str, tuple[int, int]] = {}
    failures = 0

    for row_position, row in enumerate(df.itertuples(index=False)):
        ligand_value = str(getattr(row, "Ligand"))
        protein_value = str(getattr(row, "Protein"))

        if ligand_value not in ligand_shapes:
            try:
                ligand_graph = dataset.mol_graph(ligand_value)
                ligand_shapes[ligand_value] = (
                    ligand_atom_count(ligand_graph),
                    ligand_clique_count(ligand_graph),
                )
            except Exception:
                ligand_shapes[ligand_value] = (0, 0)
                failures += 1

        if protein_value not in protein_shapes:
            try:
                protein_graph = dataset.prot_graph(protein_value)
                protein_shapes[protein_value] = (
                    protein_residue_count(protein_graph),
                    protein_edge_count(protein_graph),
                )
            except Exception:
                protein_shapes[protein_value] = (0, 0)
                failures += 1

        ligand_atoms, clique_count = ligand_shapes[ligand_value]
        protein_residues, protein_edges = protein_shapes[protein_value]
        records.append(
            {
                "row_index": row_position,
                "protein_key": protein_value,
                "protein_residue_count": int(protein_residues),
                "protein_edge_count": int(protein_edges),
                "ligand_atom_count": int(ligand_atoms),
                "clique_count": int(clique_count),
            }
        )
    return records, failures


def make_ordered_batches(indices: list[int], batch_size: int) -> list[list[int]]:
    if batch_size <= 0:
        raise ValueError("--batch_size must be positive")
    return [indices[start:start + batch_size] for start in range(0, len(indices), batch_size)]


def make_protein_size_bucketed_batches(
    shape_records: list[dict],
    batch_size: int,
) -> list[list[int]]:
    ordered = sorted(
        shape_records,
        key=lambda record: (
            record["protein_residue_count"],
            record["protein_edge_count"],
            record["ligand_atom_count"],
            record["clique_count"],
            record["row_index"],
        ),
    )
    ordered_indices = [int(record["row_index"]) for record in ordered]
    return make_ordered_batches(ordered_indices, batch_size)


def _padding_summary(values: list[int]) -> tuple[int, int, float]:
    if not values:
        return 0, 0, 0.0
    actual = int(sum(values))
    dense = int(max(values) * len(values))
    return actual, dense, 0.0 if dense == 0 else (dense - actual) / dense


def summarize_clique_batches(
    batches: list[list[int]],
    clique_counts: list[int],
    shape_key_precompute_s: float,
    shape_key_failures: int,
) -> dict:
    batch_records = []
    total_actual = 0
    total_dense = 0
    for batch_index, batch in enumerate(batches):
        counts = [clique_counts[index] for index in batch]
        if counts:
            actual = int(sum(counts))
            dense = int(max(counts) * len(counts))
            min_count = int(min(counts))
            max_count = int(max(counts))
            mean_count = float(actual / len(counts))
        else:
            actual = 0
            dense = 0
            min_count = 0
            max_count = 0
            mean_count = 0.0
        total_actual += actual
        total_dense += dense
        batch_records.append(
            {
                "batch_index": batch_index,
                "batch_row_count": len(batch),
                "clique_count_min": min_count,
                "clique_count_mean": mean_count,
                "clique_count_max": max_count,
                "clique_actual_slots": actual,
                "clique_dense_slots": dense,
                "clique_padding_slots": dense - actual,
                "clique_padding_ratio": 0.0 if dense == 0 else (dense - actual) / dense,
            }
        )

    total_padding = total_dense - total_actual
    return {
        "shape_key_precompute_s": shape_key_precompute_s,
        "shape_key_failures": shape_key_failures,
        "batches": len(batches),
        "actual_clique_slots": int(total_actual),
        "dense_clique_slots": int(total_dense),
        "clique_padding_slots": int(total_padding),
        "clique_padding_ratio": 0.0 if total_dense == 0 else total_padding / total_dense,
        "batch_records": batch_records,
    }


def summarize_size_batches(
    batches: list[list[int]],
    shape_records: list[dict],
    shape_key_precompute_s: float,
    shape_key_failures: int,
) -> dict:
    records_by_index = {int(record["row_index"]): record for record in shape_records}
    batch_records = []
    total_protein_actual = 0
    total_protein_dense = 0
    total_ligand_actual = 0
    total_ligand_dense = 0
    total_clique_actual = 0
    total_clique_dense = 0

    for batch_index, batch in enumerate(batches):
        batch_shapes = [records_by_index[index] for index in batch]
        protein_counts = [int(record["protein_residue_count"]) for record in batch_shapes]
        protein_edge_counts = [int(record["protein_edge_count"]) for record in batch_shapes]
        ligand_counts = [int(record["ligand_atom_count"]) for record in batch_shapes]
        clique_counts = [int(record["clique_count"]) for record in batch_shapes]

        protein_actual, protein_dense, protein_padding_ratio = _padding_summary(protein_counts)
        ligand_actual, ligand_dense, ligand_padding_ratio = _padding_summary(ligand_counts)
        clique_actual, clique_dense, clique_padding_ratio = _padding_summary(clique_counts)

        total_protein_actual += protein_actual
        total_protein_dense += protein_dense
        total_ligand_actual += ligand_actual
        total_ligand_dense += ligand_dense
        total_clique_actual += clique_actual
        total_clique_dense += clique_dense

        batch_records.append(
            {
                "bucket_id": batch_index,
                "batch_index": batch_index,
                "batch_row_count": len(batch),
                "protein_residue_count_min": int(min(protein_counts)) if protein_counts else 0,
                "protein_residue_count_mean": float(sum(protein_counts) / len(protein_counts)) if protein_counts else 0.0,
                "protein_residue_count_max": int(max(protein_counts)) if protein_counts else 0,
                "protein_edge_count_min": int(min(protein_edge_counts)) if protein_edge_counts else 0,
                "protein_edge_count_mean": float(sum(protein_edge_counts) / len(protein_edge_counts)) if protein_edge_counts else 0.0,
                "protein_edge_count_max": int(max(protein_edge_counts)) if protein_edge_counts else 0,
                "protein_actual_slots": protein_actual,
                "protein_dense_slots": protein_dense,
                "protein_padding_slots": protein_dense - protein_actual,
                "protein_padding_ratio": protein_padding_ratio,
                "ligand_atom_count_min": int(min(ligand_counts)) if ligand_counts else 0,
                "ligand_atom_count_mean": float(sum(ligand_counts) / len(ligand_counts)) if ligand_counts else 0.0,
                "ligand_atom_count_max": int(max(ligand_counts)) if ligand_counts else 0,
                "ligand_actual_slots": ligand_actual,
                "ligand_dense_slots": ligand_dense,
                "ligand_padding_slots": ligand_dense - ligand_actual,
                "ligand_padding_ratio": ligand_padding_ratio,
                "clique_count_min": int(min(clique_counts)) if clique_counts else 0,
                "clique_count_mean": float(sum(clique_counts) / len(clique_counts)) if clique_counts else 0.0,
                "clique_count_max": int(max(clique_counts)) if clique_counts else 0,
                "clique_actual_slots": clique_actual,
                "clique_dense_slots": clique_dense,
                "clique_padding_slots": clique_dense - clique_actual,
                "clique_padding_ratio": clique_padding_ratio,
            }
        )

    total_protein_padding = total_protein_dense - total_protein_actual
    total_ligand_padding = total_ligand_dense - total_ligand_actual
    total_clique_padding = total_clique_dense - total_clique_actual
    return {
        "shape_key_precompute_s": shape_key_precompute_s,
        "shape_key_failures": shape_key_failures,
        "batches": len(batches),
        "actual_protein_slots": int(total_protein_actual),
        "dense_protein_slots": int(total_protein_dense),
        "protein_padding_slots": int(total_protein_padding),
        "protein_padding_ratio": 0.0 if total_protein_dense == 0 else total_protein_padding / total_protein_dense,
        "actual_ligand_slots": int(total_ligand_actual),
        "dense_ligand_slots": int(total_ligand_dense),
        "ligand_padding_slots": int(total_ligand_padding),
        "ligand_padding_ratio": 0.0 if total_ligand_dense == 0 else total_ligand_padding / total_ligand_dense,
        "actual_clique_slots": int(total_clique_actual),
        "dense_clique_slots": int(total_clique_dense),
        "clique_padding_slots": int(total_clique_padding),
        "clique_padding_ratio": 0.0 if total_clique_dense == 0 else total_clique_padding / total_clique_dense,
        "batch_records": batch_records,
    }


def build_loader(
    df: pd.DataFrame,
    protein_dict: dict,
    protein_folder: Path | None,
    output_folder: Path,
    ligand_path: Path,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    device: str,
    ligand_disk_cache: bool,
    batching_strategy: str,
    collect_batching_metrics: bool,
) -> tuple[DataLoader, float, dict]:
    start = time.perf_counter()
    dataset = ProteinMoleculeDataset(
        df,
        molecule_dict={},
        molecule_folder=str(ligand_path),
        protein_dict=protein_dict,
        protein_folder=str(protein_folder) if protein_folder is not None else None,
        result_path=str(output_folder),
        dataset_tag="inference",
        device=device,
        cache_molecules=ligand_disk_cache,
    )
    if batching_strategy not in BATCHING_STRATEGIES:
        raise ValueError(f"Unknown batching strategy: {batching_strategy}")
    if batching_strategy in {"clique_count_bucketed", "protein_size_bucketed"} and not ligand_disk_cache:
        raise ValueError(f"--batching_strategy {batching_strategy} requires ligand disk cache")

    batch_sampler = None
    batching_metrics = {
        "strategy": batching_strategy,
        "shape_key_precompute_s": 0.0,
        "shape_key_failures": 0,
        "batches": None,
        "actual_clique_slots": None,
        "dense_clique_slots": None,
        "clique_padding_slots": None,
        "clique_padding_ratio": None,
    }
    should_compute_shape_keys = (
        collect_batching_metrics
        or batching_strategy in {"clique_count_bucketed", "protein_size_bucketed"}
    )
    if should_compute_shape_keys:
        if batching_strategy == "protein_size_bucketed" or (
            batching_strategy == "input_order" and collect_batching_metrics
        ):
            shape_start = time.perf_counter()
            shape_records, shape_key_failures = compute_size_shape_records(df, dataset)
            shape_key_precompute_s = time.perf_counter() - shape_start
            if batching_strategy == "protein_size_bucketed":
                batch_sampler = make_protein_size_bucketed_batches(shape_records, batch_size)
                batches_for_metrics = batch_sampler
            else:
                batches_for_metrics = make_ordered_batches(list(range(len(shape_records))), batch_size)
            batching_metrics.update(
                summarize_size_batches(
                    batches_for_metrics,
                    shape_records,
                    shape_key_precompute_s,
                    shape_key_failures,
                )
            )
        elif batching_strategy == "clique_count_bucketed":
            shape_start = time.perf_counter()
            clique_counts, shape_key_failures = compute_ligand_clique_counts(df, dataset)
            shape_key_precompute_s = time.perf_counter() - shape_start
            ordered_indices = sorted(range(len(clique_counts)), key=lambda index: clique_counts[index])
            batch_sampler = make_ordered_batches(ordered_indices, batch_size)
            batches_for_metrics = batch_sampler
            batching_metrics.update(
                summarize_clique_batches(
                    batches_for_metrics,
                    clique_counts,
                    shape_key_precompute_s,
                    shape_key_failures,
                )
            )
        else:
            shape_start = time.perf_counter()
            clique_counts, shape_key_failures = compute_ligand_clique_counts(df, dataset)
            shape_key_precompute_s = time.perf_counter() - shape_start
            batches_for_metrics = make_ordered_batches(list(range(len(clique_counts))), batch_size)
            batching_metrics.update(
                summarize_clique_batches(
                    batches_for_metrics,
                    clique_counts,
                    shape_key_precompute_s,
                    shape_key_failures,
                )
            )

    loader_kwargs = {
        "follow_batch": ["mol_x", "clique_x", "prot_node_aa"],
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers if num_workers > 0 else False,
    }
    if batch_sampler is not None:
        loader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            **loader_kwargs,
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            **loader_kwargs,
        )
    return loader, time.perf_counter() - start, batching_metrics


def capture_baseline(df: pd.DataFrame) -> pd.DataFrame | None:
    present = [column for column in VALIDATION_COLUMNS if column in df.columns]
    if not present:
        return None
    return df[["ID", *present]].copy()


def initialize_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for column in VALIDATION_COLUMNS:
        df[column] = pd.NA
    return df


def _prediction_complete_mask(df: pd.DataFrame) -> pd.Series:
    for column in VALIDATION_COLUMNS:
        if column not in df.columns:
            df[column] = pd.NA
    return df[list(VALIDATION_COLUMNS)].notna().all(axis=1)


def _failure_reason_counts(df: pd.DataFrame) -> dict[str, int]:
    if "failure_reason" not in df.columns:
        return {}
    counts = df.loc[df["failure_reason"].notna(), "failure_reason"].value_counts()
    return {str(reason): int(count) for reason, count in counts.items()}


def _diagnose_unpredicted_rows(df: pd.DataFrame, missing_mask: pd.Series) -> dict[object, tuple[str, str]]:
    diagnoses: dict[object, tuple[str, str]] = {}
    diagnosis_cache: dict[str, tuple[str, str]] = {}
    for row_position, ligand_value in df.loc[missing_mask, "Ligand"].items():
        if ligand_value is None or pd.isna(ligand_value):
            cache_key = "<EMPTY_LIGAND>"
        else:
            cache_key = str(ligand_value)
        if cache_key not in diagnosis_cache:
            diagnosis = diagnose_ligand_failure(ligand_value)
            if diagnosis.valid:
                diagnosis_cache[cache_key] = (
                    "prediction",
                    "PREDICTION_MISSING: Ligand featurization succeeded but no prediction was emitted",
                )
            else:
                diagnosis_cache[cache_key] = (
                    diagnosis.failure_stage or "ligand_featurization",
                    diagnosis.failure_reason or "LIGAND_FEATURIZATION_FAILED",
                )
        diagnoses[row_position] = diagnosis_cache[cache_key]
    return diagnoses


def apply_invalid_ligand_policy(df: pd.DataFrame, policy: str) -> tuple[pd.DataFrame, dict]:
    if policy not in INVALID_LIGAND_POLICIES:
        raise ValueError(f"Unknown invalid ligand policy: {policy}")

    predicted_mask = _prediction_complete_mask(df)
    missing_mask = ~predicted_mask
    failed_rows = int(missing_mask.sum())

    if policy == "fail" and failed_rows:
        diagnoses = _diagnose_unpredicted_rows(df, missing_mask)
        stage_counts = Counter(stage for stage, _ in diagnoses.values())
        reason_counts = Counter(reason for _, reason in diagnoses.values())
        top_reasons = ", ".join(f"{reason} ({count})" for reason, count in reason_counts.most_common(3))
        raise RuntimeError(
            "Invalid ligand policy 'fail' rejected "
            f"{failed_rows} row(s); stages={dict(stage_counts)}; top_reasons={top_reasons}"
        )

    if policy == "emit_null":
        for column in STATUS_COLUMNS:
            if column not in df.columns:
                df[column] = pd.NA
        df.loc[predicted_mask, "inference_status"] = "success"
        df.loc[predicted_mask, ["failure_stage", "failure_reason"]] = pd.NA
        if failed_rows:
            diagnoses = _diagnose_unpredicted_rows(df, missing_mask)
            df.loc[missing_mask, "inference_status"] = "failed"
            for row_position, (stage, reason) in diagnoses.items():
                df.at[row_position, "failure_stage"] = stage
                df.at[row_position, "failure_reason"] = reason
        status_counts = df["inference_status"].value_counts(dropna=False)
        stage_counts = df.loc[df["failure_stage"].notna(), "failure_stage"].value_counts()
    else:
        status_counts = pd.Series(dtype=int)
        stage_counts = pd.Series(dtype=int)

    summary = {
        "policy": policy,
        "rows": int(len(df)),
        "valid_rows": int(predicted_mask.sum()),
        "failed_rows": failed_rows,
        "status_counts": {str(status): int(count) for status, count in status_counts.items()},
        "failure_stage_counts": {str(stage): int(count) for stage, count in stage_counts.items()},
        "failure_reason_counts": _failure_reason_counts(df),
    }
    return df, summary


def merge_invalid_ligand_summaries(summaries: list[dict], policy: str) -> dict:
    merged_status_counts: Counter = Counter()
    merged_stage_counts: Counter = Counter()
    merged_reason_counts: Counter = Counter()
    rows = 0
    valid_rows = 0
    failed_rows = 0
    for summary in summaries:
        rows += int(summary.get("rows", 0))
        valid_rows += int(summary.get("valid_rows", 0))
        failed_rows += int(summary.get("failed_rows", 0))
        merged_status_counts.update(summary.get("status_counts", {}))
        merged_stage_counts.update(summary.get("failure_stage_counts", {}))
        merged_reason_counts.update(summary.get("failure_reason_counts", {}))
    return {
        "policy": policy,
        "rows": rows,
        "valid_rows": valid_rows,
        "failed_rows": failed_rows,
        "status_counts": dict(merged_status_counts),
        "failure_stage_counts": dict(merged_stage_counts),
        "failure_reason_counts": dict(merged_reason_counts),
    }


def _sync_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def resolve_precision_args(args: argparse.Namespace) -> str:
    legacy_modes = []
    if args.amp:
        legacy_modes.append("fp16")
    if args.tf32:
        legacy_modes.append("tf32")

    if args.precision == "fp32":
        if len(legacy_modes) > 1:
            raise ValueError("Use --precision instead of combining legacy --amp and --tf32 flags")
        if legacy_modes:
            args.precision = legacy_modes[0]
    else:
        for legacy_mode in legacy_modes:
            if legacy_mode != args.precision:
                raise ValueError(
                    f"Conflicting precision arguments: --precision {args.precision} with legacy {legacy_mode} flag"
                )

    args.amp = args.precision == "fp16"
    args.tf32 = args.precision == "tf32"
    return args.precision


def autocast_dtype_for_precision(precision: str) -> torch.dtype:
    if precision == "bf16":
        return torch.bfloat16
    return torch.float16


class AsyncInterpretWriter:
    _STOP = object()

    def __init__(
        self,
        ligand_path: Path,
        result_path: Path,
        max_queue_size: int = 2,
        save_cluster: bool = False,
    ) -> None:
        self.ligand_path = str(ligand_path)
        self.result_path = str(result_path)
        self.save_cluster = save_cluster
        self.queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self._lock = threading.Lock()
        self._error: BaseException | None = None
        self._active_s = 0.0
        self._payloads = 0
        self._pairs = 0
        self._queue_put_wait_s = 0.0
        self._closed = False
        self.thread = threading.Thread(target=self._run, name="psichic-interpret-writer")
        self.thread.start()

    def _set_error(self, exc: BaseException) -> None:
        with self._lock:
            if self._error is None:
                self._error = exc

    def _raise_if_failed(self) -> None:
        with self._lock:
            error = self._error
        if error is not None:
            raise RuntimeError("Interpret result writer failed") from error

    def _put_with_backpressure(self, item) -> float:
        wait_s = 0.0
        while True:
            self._raise_if_failed()
            start = time.perf_counter()
            try:
                self.queue.put(item, timeout=0.1)
                wait_s += time.perf_counter() - start
                break
            except queue.Full:
                wait_s += time.perf_counter() - start
        with self._lock:
            self._queue_put_wait_s += wait_s
        return wait_s

    def enqueue(self, payload: dict) -> float:
        if self._closed:
            raise RuntimeError("Cannot enqueue interpret results after writer close")
        return self._put_with_backpressure(payload)

    def _run(self) -> None:
        while True:
            item = self.queue.get()
            try:
                if item is self._STOP:
                    return
                start = time.perf_counter()
                pair_count = save_interpret_result_payload(
                    item,
                    ligand_path=self.ligand_path,
                    result_path=self.result_path,
                    save_cluster=self.save_cluster,
                )
                active_s = time.perf_counter() - start
                with self._lock:
                    self._active_s += active_s
                    self._payloads += 1
                    self._pairs += pair_count
            except BaseException as exc:  # pragma: no cover - propagated by close/enqueue
                self._set_error(exc)
                return
            finally:
                self.queue.task_done()

    def close(self) -> dict:
        if not self._closed:
            self._put_with_backpressure(self._STOP)
            self._closed = True
        while True:
            self._raise_if_failed()
            with self.queue.all_tasks_done:
                if self.queue.unfinished_tasks == 0:
                    break
            time.sleep(0.05)
        self.thread.join()
        self._raise_if_failed()
        return self.stats()

    def stats(self) -> dict:
        with self._lock:
            return {
                "writer_active_s": self._active_s,
                "writer_payloads": self._payloads,
                "writer_pairs": self._pairs,
                "queue_backpressure_wait_s": self._queue_put_wait_s,
                "queue_max_size": self.queue.maxsize,
            }


def _interpret_process_worker(queue_, result_queue, ligand_path: str, result_path: str, save_cluster: bool) -> None:
    active_s = 0.0
    payloads = 0
    pairs = 0
    try:
        while True:
            item = queue_.get()
            if item is None:
                result_queue.put(
                    {
                        "status": "ok",
                        "writer_active_s": active_s,
                        "writer_payloads": payloads,
                        "writer_pairs": pairs,
                    }
                )
                return
            start = time.perf_counter()
            pair_count = save_interpret_result_payload(
                item,
                ligand_path=ligand_path,
                result_path=result_path,
                save_cluster=save_cluster,
            )
            active_s += time.perf_counter() - start
            payloads += 1
            pairs += pair_count
    except BaseException as exc:  # pragma: no cover - propagated to parent
        result_queue.put(
            {
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "writer_active_s": active_s,
                "writer_payloads": payloads,
                "writer_pairs": pairs,
            }
        )


class ProcessInterpretWriter:
    def __init__(
        self,
        ligand_path: Path,
        result_path: Path,
        max_queue_size: int = 2,
        save_cluster: bool = False,
    ) -> None:
        self.ctx = mp.get_context("spawn")
        self.queue_max_size = max_queue_size
        self.queue = self.ctx.Queue(maxsize=max_queue_size)
        self.result_queue = self.ctx.Queue(maxsize=1)
        self._queue_put_wait_s = 0.0
        self._closed = False
        self._stats = {
            "writer_active_s": 0.0,
            "writer_payloads": 0,
            "writer_pairs": 0,
        }
        self.process = self.ctx.Process(
            target=_interpret_process_worker,
            args=(self.queue, self.result_queue, str(ligand_path), str(result_path), save_cluster),
            name="psichic-interpret-writer-process",
        )
        self.process.start()

    def _read_result_nowait(self) -> dict | None:
        try:
            return self.result_queue.get_nowait()
        except queue.Empty:
            return None

    def _raise_for_result(self, result: dict | None) -> None:
        if result is None:
            return
        self._stats.update(
            {
                "writer_active_s": result.get("writer_active_s", 0.0),
                "writer_payloads": result.get("writer_payloads", 0),
                "writer_pairs": result.get("writer_pairs", 0),
            }
        )
        if result.get("status") == "error":
            raise RuntimeError(
                "Interpret result writer failed in process: "
                f"{result.get('error_type')}: {result.get('error')}\n{result.get('traceback', '')}"
            )

    def _raise_if_failed(self) -> None:
        result = self._read_result_nowait()
        self._raise_for_result(result)
        if self.process.exitcode not in (None, 0):
            raise RuntimeError(f"Interpret result writer process exited with code {self.process.exitcode}")

    def _put_with_backpressure(self, item) -> float:
        wait_s = 0.0
        while True:
            self._raise_if_failed()
            start = time.perf_counter()
            try:
                self.queue.put(item, timeout=0.1)
                wait_s += time.perf_counter() - start
                break
            except queue.Full:
                wait_s += time.perf_counter() - start
        self._queue_put_wait_s += wait_s
        return wait_s

    def enqueue(self, payload: dict) -> float:
        if self._closed:
            raise RuntimeError("Cannot enqueue interpret results after writer close")
        return self._put_with_backpressure(payload)

    def close(self) -> dict:
        if not self._closed:
            self._put_with_backpressure(None)
            self._closed = True
        while self.process.is_alive():
            self._raise_if_failed()
            self.process.join(timeout=0.1)
        result = self._read_result_nowait()
        self._raise_for_result(result)
        self._raise_if_failed()
        return self.stats()

    def stats(self) -> dict:
        return {
            **self._stats,
            "queue_backpressure_wait_s": self._queue_put_wait_s,
            "queue_max_size": self.queue_max_size,
        }


@contextmanager
def nvtx_range(name: str, enabled: bool):
    if not enabled or not torch.cuda.is_available():
        with nullcontext():
            yield
        return

    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def _node_count(data, name: str) -> int:
    value = getattr(data, name, None)
    if value is None:
        return 0
    return int(value.size(0))


def _edge_count(data, name: str) -> int:
    value = getattr(data, name, None)
    if value is None or value.dim() < 2:
        return 0
    return int(value.size(1))


def _batch_item_counts(batch_tensor) -> list[int]:
    if batch_tensor is None or batch_tensor.numel() == 0:
        return []
    return torch.bincount(batch_tensor.detach().cpu()).tolist()


def collect_shape_metrics(data, batch_index: int, batch_ids: list[str], forward_s: float) -> dict:
    protein_residue_count = _node_count(data, "prot_node_aa")
    protein_edge_count = _edge_count(data, "prot_edge_index")
    ligand_atom_count = _node_count(data, "mol_x")
    ligand_edge_count = _edge_count(data, "mol_edge_index")
    clique_count = _node_count(data, "clique_x")
    clique_edge_count = _edge_count(data, "clique_edge_index")
    atom2clique_edge_count = _edge_count(data, "atom2clique_index")
    per_graph_clique_counts = _batch_item_counts(getattr(data, "clique_x_batch", None))
    if per_graph_clique_counts:
        clique_dense_slots = max(per_graph_clique_counts) * len(per_graph_clique_counts)
        clique_padding_slots = clique_dense_slots - sum(per_graph_clique_counts)
        clique_count_min = min(per_graph_clique_counts)
        clique_count_max = max(per_graph_clique_counts)
        clique_count_mean = sum(per_graph_clique_counts) / len(per_graph_clique_counts)
    else:
        clique_dense_slots = 0
        clique_padding_slots = 0
        clique_count_min = 0
        clique_count_max = 0
        clique_count_mean = 0.0
    return {
        "batch_index": batch_index,
        "batch_row_count": len(batch_ids),
        "protein_residue_count": protein_residue_count,
        "protein_edge_count": protein_edge_count,
        "ligand_atom_count": ligand_atom_count,
        "ligand_edge_count": ligand_edge_count,
        "clique_count": clique_count,
        "clique_graph_count": len(per_graph_clique_counts),
        "clique_count_min": clique_count_min,
        "clique_count_mean": clique_count_mean,
        "clique_count_max": clique_count_max,
        "clique_dense_slots": clique_dense_slots,
        "clique_padding_slots": clique_padding_slots,
        "clique_padding_ratio": 0.0 if clique_dense_slots == 0 else clique_padding_slots / clique_dense_slots,
        "clique_edge_count": clique_edge_count,
        "atom2clique_edge_count": atom2clique_edge_count,
        "total_nodes": protein_residue_count + ligand_atom_count + clique_count,
        "total_edges": protein_edge_count + ligand_edge_count + clique_edge_count + atom2clique_edge_count,
        "forward_time_s": forward_s,
    }


def _batch_row_positions(data) -> list[int] | None:
    value = getattr(data, "row_position", None)
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return [int(item) for item in value.detach().cpu().reshape(-1).tolist()]
    if isinstance(value, (list, tuple)):
        positions = []
        for item in value:
            if isinstance(item, torch.Tensor):
                positions.extend(int(pos) for pos in item.detach().cpu().reshape(-1).tolist())
            else:
                positions.append(int(item))
        return positions
    return [int(value)]


def run_screening(
    df: pd.DataFrame,
    model: PSICHICPlusModel,
    loader: DataLoader,
    protein_dict: dict,
    ligand_path: Path,
    output_folder: Path,
    device: torch.device,
    save_interpret: bool,
    use_inference_mode: bool,
    precision: str,
    use_nvtx: bool,
    sync_timing: bool,
    interpret_writer_backend: str,
    collect_shapes: bool,
) -> tuple[pd.DataFrame, float, float, dict, list[dict]]:
    interpret_dir = output_folder / "interpretations"
    peak_gpu_mem_gb = 0.0
    timing = {
        "batches": 0,
        "dataloader_wait_s": 0.0,
        "to_device_s": 0.0,
        "forward_s": 0.0,
        "cpu_extract_s": 0.0,
        "interpret_payload_copy_s": 0.0,
        "interpret_enqueue_s": 0.0,
        "queue_backpressure_wait_s": 0.0,
        "writer_active_s": 0.0,
        "writer_wait_s": 0.0,
        "writer_payloads": 0,
        "writer_pairs": 0,
        "queue_max_size": 0,
        "store_result_s": 0.0,
    }

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        _sync_cuda(device)

    if hasattr(model, "set_forward_nvtx"):
        model.set_forward_nvtx(use_nvtx)

    context = torch.inference_mode if use_inference_mode else torch.no_grad
    autocast_enabled = precision in {"bf16", "fp16"} and device.type == "cuda"
    autocast_dtype = autocast_dtype_for_precision(precision)
    shape_metrics: list[dict] = []
    row_id_to_position = None
    if df["ID"].is_unique:
        row_id_to_position = {
            id_value: row_position
            for row_position, id_value in enumerate(df["ID"].tolist())
        }
    interpret_writer = None
    if save_interpret:
        writer_class = ProcessInterpretWriter if interpret_writer_backend == "process" else AsyncInterpretWriter
        interpret_writer = writer_class(ligand_path, interpret_dir, max_queue_size=2)
    start = time.perf_counter()
    pending_error: BaseException | None = None
    try:
        with context(), nvtx_range("screening", use_nvtx):
            batch_index = 0
            loader_iter = iter(loader)
            while True:
                wait_start = time.perf_counter()
                try:
                    data = next(loader_iter)
                except StopIteration:
                    break
                timing["dataloader_wait_s"] += time.perf_counter() - wait_start
                if data is None:
                    continue
                timing["batches"] += 1
                with nvtx_range(f"batch_{batch_index:04d}_to_device", use_nvtx):
                    phase_start = time.perf_counter()
                    data = data.to(device)
                    if sync_timing:
                        _sync_cuda(device)
                    timing["to_device_s"] += time.perf_counter() - phase_start
                batch_ids = data.id if isinstance(data.id, (list, tuple)) else [data.id]
                with nvtx_range(f"batch_{batch_index:04d}_forward", use_nvtx), torch.autocast(
                    device_type="cuda",
                    dtype=autocast_dtype,
                    enabled=autocast_enabled,
                ):
                    phase_start = time.perf_counter()
                    reg_pred, reg_alpha, reg_beta, attention_dict = model.predict_batch(data)
                    if sync_timing:
                        _sync_cuda(device)
                    forward_s = time.perf_counter() - phase_start
                    timing["forward_s"] += forward_s
                if collect_shapes:
                    shape_metrics.append(collect_shape_metrics(data, batch_index, batch_ids, forward_s))
                batch_row_positions = _batch_row_positions(data)
                phase_start = time.perf_counter()
                reg_pred_np = reg_pred.squeeze().reshape(-1).detach().cpu().numpy()
                reg_alpha_np = reg_alpha.squeeze().reshape(-1).detach().cpu().numpy()
                reg_beta_np = reg_beta.squeeze().reshape(-1).detach().cpu().numpy()
                interaction_keys = list(zip(data.prot_key, data.mol_key)) if save_interpret else ()
                timing["cpu_extract_s"] += time.perf_counter() - phase_start
                interpret_payload = None
                if save_interpret:
                    with nvtx_range(f"batch_{batch_index:04d}_interpret_payload_copy", use_nvtx):
                        phase_start = time.perf_counter()
                        interpret_payload = build_interpret_result_payload(
                            attention_dict,
                            batch_ids,
                            interaction_keys,
                            protein_dict,
                            save_cluster=False,
                        )
                        timing["interpret_payload_copy_s"] += time.perf_counter() - phase_start
                with nvtx_range(f"batch_{batch_index:04d}_store_result", use_nvtx):
                    phase_start = time.perf_counter()
                    df = store_result(
                        df,
                        {},
                        batch_ids,
                        (),
                        protein_dict,
                        str(ligand_path),
                        reg_pred=reg_pred_np,
                        reg_alpha=reg_alpha_np,
                        reg_beta=reg_beta_np,
                        result_path=str(interpret_dir),
                        save_interpret=False,
                        save_cluster=False,
                        row_id_to_position=row_id_to_position,
                        row_positions=batch_row_positions,
                    )
                    timing["store_result_s"] += time.perf_counter() - phase_start
                if interpret_writer is not None and interpret_payload is not None:
                    with nvtx_range(f"batch_{batch_index:04d}_interpret_enqueue", use_nvtx):
                        phase_start = time.perf_counter()
                        queue_wait_s = interpret_writer.enqueue(interpret_payload)
                        timing["interpret_enqueue_s"] += time.perf_counter() - phase_start
                        timing["queue_backpressure_wait_s"] += queue_wait_s
                batch_index += 1
    except BaseException as exc:
        pending_error = exc
    finally:
        if interpret_writer is not None:
            phase_start = time.perf_counter()
            try:
                writer_stats = interpret_writer.close()
            except BaseException as exc:  # pragma: no cover - surfaced to caller
                if pending_error is None:
                    pending_error = exc
                writer_stats = interpret_writer.stats()
            timing["writer_wait_s"] += time.perf_counter() - phase_start
            for key, value in writer_stats.items():
                if key == "queue_backpressure_wait_s":
                    timing[key] = value
                else:
                    timing[key] = value
    if pending_error is not None:
        raise pending_error
    _sync_cuda(device)
    screening_s = time.perf_counter() - start

    if device.type == "cuda":
        peak_gpu_mem_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)

    timing["unattributed_screening_s"] = max(
        0.0,
        screening_s
        - timing["dataloader_wait_s"]
        - timing["to_device_s"]
        - timing["forward_s"]
        - timing["cpu_extract_s"]
        - timing["interpret_payload_copy_s"]
        - timing["interpret_enqueue_s"]
        - timing["writer_wait_s"]
        - timing["store_result_s"],
    )

    return df, screening_s, peak_gpu_mem_gb, timing, shape_metrics


def validate_predictions(
    baseline_df: pd.DataFrame | None,
    result_df: pd.DataFrame,
    tolerance: float,
) -> dict:
    if baseline_df is None:
        return {
            "status": "skipped",
            "tolerance": tolerance,
            "checked_columns": [],
            "columns": {},
        }

    checked_columns = [column for column in VALIDATION_COLUMNS if column in baseline_df.columns and column in result_df.columns]
    merged = baseline_df.merge(
        result_df[["ID", *checked_columns]],
        on="ID",
        how="inner",
        suffixes=("_baseline", "_actual"),
    )

    column_metrics: dict[str, dict[str, float | None | int]] = {}
    passes = True
    for column in checked_columns:
        baseline_values = pd.to_numeric(merged[f"{column}_baseline"], errors="coerce")
        actual_values = pd.to_numeric(merged[f"{column}_actual"], errors="coerce")
        delta = (actual_values - baseline_values).abs()
        max_abs = None if delta.isna().all() else float(delta.max())
        mean_abs = None if delta.isna().all() else float(delta.mean())
        column_pass = max_abs is not None and max_abs <= tolerance
        column_metrics[column] = {
            "rows_compared": int(delta.notna().sum()),
            "max_abs_diff": max_abs,
            "mean_abs_diff": mean_abs,
            "pass": column_pass,
        }
        passes = passes and column_pass

    return {
        "status": "pass" if passes else "fail",
        "tolerance": tolerance,
        "checked_columns": checked_columns,
        "columns": column_metrics,
        "rows_compared": int(len(merged)),
    }


def write_json(path: Path | None, payload: dict) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def summarize_shape_metrics(rows: list[dict]) -> dict:
    if not rows:
        return {"batches": 0}
    frame = pd.DataFrame(rows)
    summary: dict[str, dict[str, float]] = {}
    for column in frame.columns:
        if column in {"batch_index", "chunk_index", "global_batch_index"}:
            continue
        if not pd.api.types.is_numeric_dtype(frame[column]):
            continue
        series = frame[column]
        summary[column] = {
            "min": float(series.min()),
            "max": float(series.max()),
            "mean": float(series.mean()),
            "sum": float(series.sum()),
        }
    return {
        "batches": int(len(rows)),
        "columns": summary,
    }


def write_shape_metrics(csv_path: Path | None, json_path: Path | None, rows: list[dict]) -> None:
    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(csv_path, index=False)
    write_json(json_path, {"batches": len(rows), "rows": rows})


def summarize_batching_metrics(chunks: list[dict]) -> dict:
    if not chunks:
        return {"chunks": [], "summary": {"batches": 0}}
    total_clique_actual = sum(int(chunk.get("actual_clique_slots") or 0) for chunk in chunks)
    total_clique_dense = sum(int(chunk.get("dense_clique_slots") or 0) for chunk in chunks)
    total_clique_padding = sum(int(chunk.get("clique_padding_slots") or 0) for chunk in chunks)
    total_protein_actual = sum(int(chunk.get("actual_protein_slots") or 0) for chunk in chunks)
    total_protein_dense = sum(int(chunk.get("dense_protein_slots") or 0) for chunk in chunks)
    total_protein_padding = sum(int(chunk.get("protein_padding_slots") or 0) for chunk in chunks)
    total_ligand_actual = sum(int(chunk.get("actual_ligand_slots") or 0) for chunk in chunks)
    total_ligand_dense = sum(int(chunk.get("dense_ligand_slots") or 0) for chunk in chunks)
    total_ligand_padding = sum(int(chunk.get("ligand_padding_slots") or 0) for chunk in chunks)
    return {
        "chunks": chunks,
        "summary": {
            "batches": sum(int(chunk.get("batches") or 0) for chunk in chunks),
            "shape_key_precompute_s": sum(float(chunk.get("shape_key_precompute_s") or 0.0) for chunk in chunks),
            "shape_key_failures": sum(int(chunk.get("shape_key_failures") or 0) for chunk in chunks),
            "actual_protein_slots": total_protein_actual,
            "dense_protein_slots": total_protein_dense,
            "protein_padding_slots": total_protein_padding,
            "protein_padding_ratio": 0.0 if total_protein_dense == 0 else total_protein_padding / total_protein_dense,
            "actual_ligand_slots": total_ligand_actual,
            "dense_ligand_slots": total_ligand_dense,
            "ligand_padding_slots": total_ligand_padding,
            "ligand_padding_ratio": 0.0 if total_ligand_dense == 0 else total_ligand_padding / total_ligand_dense,
            "actual_clique_slots": total_clique_actual,
            "dense_clique_slots": total_clique_dense,
            "clique_padding_slots": total_clique_padding,
            "clique_padding_ratio": 0.0 if total_clique_dense == 0 else total_clique_padding / total_clique_dense,
        },
    }


def merge_validation_metrics(validations: list[dict], tolerance: float) -> dict:
    checked_columns = sorted({column for item in validations for column in item.get("checked_columns", [])})
    if not checked_columns:
        return {
            "status": "skipped",
            "tolerance": tolerance,
            "checked_columns": [],
            "columns": {},
        }

    merged_columns = {}
    passes = True
    rows_compared = 0
    for column in checked_columns:
        compared = 0
        weighted_sum = 0.0
        max_abs = None
        for item in validations:
            column_metrics = item.get("columns", {}).get(column)
            if not column_metrics:
                continue
            row_count = int(column_metrics.get("rows_compared", 0))
            mean_abs = column_metrics.get("mean_abs_diff")
            column_max = column_metrics.get("max_abs_diff")
            compared += row_count
            if mean_abs is not None:
                weighted_sum += float(mean_abs) * row_count
            if column_max is not None:
                max_abs = float(column_max) if max_abs is None else max(max_abs, float(column_max))
        mean_abs = None if compared == 0 else weighted_sum / compared
        column_pass = max_abs is not None and max_abs <= tolerance
        passes = passes and column_pass
        rows_compared = max(rows_compared, compared)
        merged_columns[column] = {
            "rows_compared": compared,
            "max_abs_diff": max_abs,
            "mean_abs_diff": mean_abs,
            "pass": column_pass,
        }

    return {
        "status": "pass" if passes else "fail",
        "tolerance": tolerance,
        "checked_columns": checked_columns,
        "columns": merged_columns,
        "rows_compared": rows_compared,
    }


def main() -> None:
    args = parse_args()
    resolve_cli_presets(args)
    resolve_precision_args(args)
    device = torch.device(args.device)
    if args.precision == "tf32" and device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = args.precision == "tf32"
        torch.backends.cudnn.allow_tf32 = args.precision == "tf32"
    if args.precision in {"bf16", "fp16"} and device.type == "cuda":
        # cuDNN's SDPA fp16 plan does not support PSICHIC+'s MotifPool attention
        # on A40 ("No execution plans support the graph"); fall back to flash /
        # mem-efficient / math kernels which all handle this configuration.
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
    output_folder = args.output_folder
    output_folder.mkdir(parents=True, exist_ok=True)
    ligand_path = args.ligand_cache_dir or (output_folder / "ligands")
    if args.ligand_disk_cache:
        ligand_path.mkdir(parents=True, exist_ok=True)
    protein_cache_dir = args.protein_cache_dir
    fasta_record = read_single_fasta(args.protein_fasta) if args.protein_fasta is not None else None
    if args.save_interpret and not args.ligand_disk_cache:
        raise ValueError("--save_interpret requires ligand disk cache because ligand interpretation reloads ligand graphs")
    if args.batching_strategy == "clique_count_bucketed" and not args.ligand_disk_cache:
        raise ValueError("--batching_strategy clique_count_bucketed requires ligand disk cache")
    if args.chunk_size is not None and fasta_record is None:
        raise ValueError("--chunk_size currently requires --protein_fasta so protein initialization can stay bounded")

    first_df = pd.read_csv(args.input_file, nrows=args.chunk_size or None)
    first_df = prepare_input_frame(first_df, args.ligand_column, fasta_record, args.protein_name)
    pairs = len(first_df)
    unique_proteins_seen = set(first_df["Protein"].astype(str).unique().tolist())
    unique_ligands_seen = set(first_df["Ligand"].astype(str).unique().tolist())

    wall_start = time.perf_counter()
    with nvtx_range("protein_init", args.nvtx):
        (
            protein_dict,
            protein_folder,
            protein_init_s,
            protein_cache_hits,
            protein_cache_misses,
            protein_cache_mode,
        ) = build_protein_state(first_df, protein_cache_dir, args.save_interpret)
    with nvtx_range("model_load", args.nvtx):
        model = PSICHICPlusModel.from_pretrained(
            args.model_dir,
            device=device,
            motifpool_dense_strategy=args.motifpool_dense_strategy,
            protein_mincut_strategy=args.protein_mincut_strategy,
            protein_dense_cache_strategy=args.protein_dense_cache_strategy,
        )
        if args.compile_mode != "none":
            if not hasattr(torch, "compile"):
                raise RuntimeError("torch.compile is not available in this PyTorch build")
            model.model = torch.compile(model.model, mode=args.compile_mode)

    output_csv = output_folder / "inference_results.csv"
    if output_csv.exists():
        output_csv.unlink()

    loader_setup_s = 0.0
    screening_s = 0.0
    writeout_s = 0.0
    peak_gpu_mem_gb = 0.0
    screening_timing = {
        "batches": 0,
        "dataloader_wait_s": 0.0,
        "to_device_s": 0.0,
        "forward_s": 0.0,
        "cpu_extract_s": 0.0,
        "store_result_s": 0.0,
        "unattributed_screening_s": 0.0,
    }
    validations: list[dict] = []
    shape_metrics: list[dict] = []
    batching_metric_chunks: list[dict] = []
    invalid_ligand_summaries: list[dict] = []
    chunk_count = 0

    def process_frame(frame: pd.DataFrame, append: bool) -> None:
        nonlocal loader_setup_s, screening_s, writeout_s, peak_gpu_mem_gb, screening_timing, validations, shape_metrics, batching_metric_chunks, invalid_ligand_summaries, chunk_count
        chunk_count += 1
        frame_baseline = capture_baseline(frame)
        frame = initialize_output_columns(frame)
        with nvtx_range(f"loader_setup_chunk_{chunk_count:04d}", args.nvtx):
            loader, chunk_loader_setup_s, chunk_batching_metrics = build_loader(
                frame,
                protein_dict,
                protein_folder,
                output_folder,
                ligand_path,
                args.batch_size,
                args.num_workers,
                args.pin_memory,
                args.persistent_workers,
                args.device,
                args.ligand_disk_cache,
                args.batching_strategy,
                (
                    args.collect_batching_metrics
                    or args.shape_metrics_csv is not None
                    or args.shape_metrics_json is not None
                ),
            )
        chunk_batching_metrics["chunk_index"] = chunk_count
        batching_metric_chunks.append(chunk_batching_metrics)
        result_df, chunk_screening_s, chunk_peak_gpu_mem_gb, chunk_screening_timing, chunk_shape_metrics = run_screening(
            frame,
            model,
            loader,
            protein_dict,
            ligand_path,
            output_folder,
            device,
            args.save_interpret,
            args.inference_mode,
            args.precision,
            args.nvtx,
            args.sync_timing,
            args.interpret_writer_backend,
            args.shape_metrics_csv is not None or args.shape_metrics_json is not None,
        )
        result_df, invalid_ligand_summary = apply_invalid_ligand_policy(result_df, args.invalid_ligand_policy)
        invalid_ligand_summary["chunk_index"] = chunk_count
        invalid_ligand_summaries.append(invalid_ligand_summary)
        batch_offset = len(shape_metrics)
        for row in chunk_shape_metrics:
            row["chunk_index"] = chunk_count
            row["global_batch_index"] = batch_offset + int(row["batch_index"])
        shape_metrics.extend(chunk_shape_metrics)
        writeout_start = time.perf_counter()
        result_df.to_csv(output_csv, mode="a" if append else "w", header=not append, index=False)
        writeout_s += time.perf_counter() - writeout_start
        loader_setup_s += chunk_loader_setup_s
        screening_s += chunk_screening_s
        peak_gpu_mem_gb = max(peak_gpu_mem_gb, chunk_peak_gpu_mem_gb)
        for key, value in chunk_screening_timing.items():
            screening_timing[key] = screening_timing.get(key, 0.0) + value
        validations.append(validate_predictions(frame_baseline, result_df, args.validation_tolerance))

    if args.chunk_size is None:
        process_frame(first_df, append=False)
    else:
        process_frame(first_df, append=False)
        rows_processed = len(first_df)
        for chunk in pd.read_csv(args.input_file, chunksize=args.chunk_size, skiprows=range(1, args.chunk_size + 1)):
            chunk = prepare_input_frame(chunk, args.ligand_column, fasta_record, args.protein_name, row_offset=rows_processed)
            pairs += len(chunk)
            unique_proteins_seen.update(chunk["Protein"].astype(str).unique().tolist())
            unique_ligands_seen.update(chunk["Ligand"].astype(str).unique().tolist())
            process_frame(chunk, append=True)
            rows_processed += len(chunk)

    wall_time_s = time.perf_counter() - wall_start
    throughput_pairs_per_s = pairs / wall_time_s if wall_time_s > 0 else 0.0

    validation = merge_validation_metrics(validations, args.validation_tolerance)
    invalid_ligand_metrics = merge_invalid_ligand_summaries(
        invalid_ligand_summaries,
        args.invalid_ligand_policy,
    )

    metrics = {
        "runtime_variant": "production",
        "status": validation["status"],
        "host": socket.gethostname(),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" and torch.cuda.is_available() else None,
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "device": str(device),
        "precision": args.precision,
        "autocast_dtype": str(autocast_dtype_for_precision(args.precision)) if args.precision in {"bf16", "fp16"} else None,
        "allow_tf32_matmul": (
            bool(torch.backends.cuda.matmul.allow_tf32)
            if device.type == "cuda" and torch.cuda.is_available()
            else None
        ),
        "allow_tf32_cudnn": (
            bool(torch.backends.cudnn.allow_tf32)
            if device.type == "cuda" and torch.cuda.is_available()
            else None
        ),
        "input_file": str(args.input_file),
        "output_file": str(output_csv),
        "model_dir": str(args.model_dir),
        "protein_fasta": str(args.protein_fasta) if args.protein_fasta is not None else None,
        "protein_name": args.protein_name,
        "ligand_column": args.ligand_column,
        "chunk_size": args.chunk_size,
        "chunks": chunk_count,
        "invalid_ligand_policy": args.invalid_ligand_policy,
        "invalid_ligands": invalid_ligand_metrics,
        "invalid_ligand_chunks": invalid_ligand_summaries,
        "ligand_disk_cache": args.ligand_disk_cache,
        "ligand_cache_dir": str(ligand_path) if args.ligand_disk_cache else None,
        "protein_cache_dir": str(protein_folder) if protein_folder is not None else None,
        "protein_cache_mode": protein_cache_mode,
        "protein_cache_hits": protein_cache_hits,
        "protein_cache_misses": protein_cache_misses,
        "rows": pairs,
        "unique_proteins": len(unique_proteins_seen),
        "unique_ligands": len(unique_ligands_seen),
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "inference_preset": args.inference_preset,
        "resolved_inference_preset": args.resolved_inference_preset,
        "requested_strategies": args.requested_strategies,
        "resolved_strategies": {
            "motifpool_dense_strategy": args.motifpool_dense_strategy,
            "protein_mincut_strategy": args.protein_mincut_strategy,
            "protein_dense_cache_strategy": args.protein_dense_cache_strategy,
        },
        "batching_strategy": args.batching_strategy,
        "motifpool_dense_strategy": args.motifpool_dense_strategy,
        "protein_mincut_strategy": args.protein_mincut_strategy,
        "protein_dense_cache_strategy": args.protein_dense_cache_strategy,
        "protein_dense_cache": model.protein_dense_cache_stats(),
        "collect_batching_metrics": args.collect_batching_metrics,
        "pin_memory": args.pin_memory,
        "persistent_workers": args.persistent_workers if args.num_workers > 0 else False,
        "inference_mode": args.inference_mode,
        "amp": args.amp,
        "tf32": args.tf32,
        "nvtx": args.nvtx,
        "sync_timing": args.sync_timing,
        "compile_mode": args.compile_mode,
        "save_interpret": args.save_interpret,
        "interpret_writer_backend": args.interpret_writer_backend if args.save_interpret else None,
        "wall_time_s": wall_time_s,
        "protein_init_s": protein_init_s,
        "loader_setup_s": loader_setup_s,
        "screening_s": screening_s,
        "screening_timing": screening_timing,
        "batching_metrics": summarize_batching_metrics(batching_metric_chunks),
        "shape_metrics": {
            "csv": str(args.shape_metrics_csv) if args.shape_metrics_csv is not None else None,
            "json": str(args.shape_metrics_json) if args.shape_metrics_json is not None else None,
            "summary": summarize_shape_metrics(shape_metrics),
        },
        "writeout_s": writeout_s,
        "throughput_pairs_per_s": throughput_pairs_per_s,
        "peak_gpu_mem_gb": peak_gpu_mem_gb,
        "validation": validation,
    }

    if args.ligand_cache_dir is None and not args.preserve_ligand_cache and ligand_path.exists():
        shutil.rmtree(ligand_path)
    if protein_folder is not None and args.protein_cache_dir is None and not args.preserve_protein_cache and protein_folder.exists():
        shutil.rmtree(protein_folder)

    log_file = output_folder / "inference_dataset_errors.log"
    if log_file.exists() and log_file.stat().st_size == 0:
        log_file.unlink()

    write_shape_metrics(args.shape_metrics_csv, args.shape_metrics_json, shape_metrics)
    write_json(args.metrics_json, metrics)
    write_json(args.validation_json, validation)
    print(f"Saved predictions to {output_csv}")


if __name__ == "__main__":
    main()
