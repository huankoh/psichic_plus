import argparse
import csv
import gc
import hashlib
from dataclasses import asdict
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import yaml

from applications.scorer import Output
from .adapter import BoltzRunner
from .models import Candidate, EnsembleMethod, Metric, Mode, OMITTED, Plan
from .target import Target
from .workflow import row_record, run_workflow, score_psichic


def read_candidates(path, id_column="ID", smiles_column="SMILES"):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle, strict=True)
        fields = reader.fieldnames or []
        if len(set(fields)) != len(fields):
            raise ValueError("Candidate CSV has duplicate column names")
        if id_column not in fields or smiles_column not in fields:
            raise ValueError(f"Candidate CSV requires {id_column!r} and {smiles_column!r} columns")
        rows = []
        for index, row in enumerate(reader, 1):
            if None in row or row[id_column] is None or row[smiles_column] is None:
                raise ValueError(f"Malformed candidate CSV row {index}")
            rows.append(Candidate(index, row[id_column], row[smiles_column]))
    if not rows:
        raise ValueError("Candidate CSV has no rows")
    return rows


def parser():
    result = argparse.ArgumentParser(description="Optional Boltz-2 scoring and experimental PSICHIC+ ensembles.")
    result.add_argument("--mode", required=True, choices=[mode.value for mode in Mode])
    result.add_argument("--candidates", required=True, type=Path)
    result.add_argument("--id-column", default="ID")
    result.add_argument("--smiles-column", default="SMILES")
    source = result.add_mutually_exclusive_group(required=True)
    source.add_argument("--fasta", type=Path)
    source.add_argument("--target-yaml", type=Path)
    result.add_argument("--ligand-id")
    result.add_argument("--psichic-chain")
    result.add_argument("--msa-mode", choices=["local", "server", "single"], required=True)
    result.add_argument("--msa", type=Path, help="Local MSA for --fasta. Native YAML owns its per-protein MSA paths.")
    result.add_argument("--output", type=Path, required=True, help="A new run directory. Existing directories are rejected.")
    result.add_argument("--boltz-python", default=os.environ.get("BOLTZ_PYTHON"), help="Python in the separate Boltz 2.2.1 environment, with boltz in the same bin directory.")
    result.add_argument("--boltz-timeout", type=float, help="Optional timeout in seconds for the native prediction process.")
    result.add_argument("--rank-by", choices=[metric.value for metric in Metric], default=OMITTED)
    result.add_argument("--ensemble", choices=[method.value for method in EnsembleMethod], default=OMITTED,
                        help="Ensemble method. Omitted ensemble modes retain rank-v1.")
    result.add_argument("--psichic-output", choices=[output.value for output in Output], default=OMITTED,
                        help="Output detail. concentration-v1 always requests distribution internally.")
    result.add_argument("--shortlist-size", type=int)
    result.add_argument("--psichic-weight", type=float, default=OMITTED)
    result.add_argument("--boltz-weight", type=float, default=OMITTED)
    result.add_argument("--model-dir", type=Path, help="Existing PSICHIC+ weights, required except in boltz-only mode.")
    result.add_argument("--psichic-device", default="cpu")
    result.add_argument("--psichic-cache", type=Path)
    result.add_argument("--batch-size", type=int, default=32)
    result.add_argument("boltz_options", nargs=argparse.REMAINDER, help="Pass native prediction options after --. See boltz predict --help.")
    return result


def _write_json(path, content):
    path.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n")


def execute(args):
    plan = Plan(Mode(args.mode), args.shortlist_size,
                Metric(args.rank_by) if args.rank_by is not OMITTED else OMITTED,
                args.psichic_weight, args.boltz_weight,
                EnsembleMethod(args.ensemble) if args.ensemble is not OMITTED else OMITTED, args.psichic_output)
    if not args.boltz_python:
        raise ValueError("Provide --boltz-python or BOLTZ_PYTHON for the isolated environment")
    if args.batch_size < 1:
        raise ValueError("Batch size must be positive")
    if plan.mode is not Mode.BOLTZ and (args.model_dir is None or not args.model_dir.is_dir()):
        raise ValueError("This mode requires --model-dir pointing to PSICHIC+ weights")
    if plan.mode is Mode.BOLTZ and any(value is not None for value in (args.model_dir, args.psichic_chain, args.psichic_cache)):
        raise ValueError("PSICHIC options do not apply to boltz-only mode")
    candidates = read_candidates(args.candidates, args.id_column, args.smiles_column)
    target = Target.load(fasta=args.fasta, native=args.target_yaml, ligand_id=args.ligand_id, msa_mode=args.msa_mode, msa=args.msa)
    protein = target.protein_sequence(args.psichic_chain) if plan.mode is not Mode.BOLTZ else None
    output = args.output.expanduser().absolute()
    if output.exists():
        raise ValueError(f"Output already exists; choose a new run directory: {output}")
    options = args.boltz_options[1:] if args.boltz_options[:1] == ["--"] else args.boltz_options
    runner = BoltzRunner(args.boltz_python, target, output / "boltz", options=options, timeout=args.boltz_timeout)
    runner.check()
    output.mkdir(parents=True, exist_ok=False)
    runner.directory.mkdir()
    (runner.directory / "native_help.txt").write_text(runner.metadata["native_help"])
    shutil.copyfile(args.candidates, output / "source_candidates.csv")
    target_copy = output / ("source_target.yaml" if args.target_yaml else "source_target.fasta")
    shutil.copyfile(target.source, target_copy)
    root = Path(__file__).resolve().parents[3]
    revision, source_dirty = None, None
    if shutil.which("git"):
        result = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True)
        revision = result.stdout.strip() or None
        result = subprocess.run(["git", "-C", str(root), "status", "--porcelain"], capture_output=True, text=True)
        source_dirty = bool(result.stdout.strip()) if result.returncode == 0 else None
    weight_hashes = []
    if args.model_dir:
        for path in (args.model_dir / name for name in ("config.json", "degree.pt", "model.pt")):
            if path.is_file():
                with path.open("rb") as handle:
                    weight_hashes.append({"path": str(path.absolute()), "sha256": hashlib.file_digest(handle, "sha256").hexdigest()})
    packages = {}
    for package in ("torch", "numpy", "psichic-ligand", "rdkit", "PyYAML"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    manifest = {
        "started_at": datetime.now(timezone.utc).isoformat(), "plan": asdict(plan),
        "row_count": len(candidates), "target": str(target_copy), "ligand_id": target.ligand_id,
        "msa_mode": target.msa_mode, "status": "running", "source_revision": revision,
        "source_dirty": source_dirty,
        "build_revision": os.environ.get("PSICHIC_WORKFLOW_BUILD_REVISION"),
        "python": sys.version, "packages": packages,
        "psichic": {"chain": protein[0] if protein else None, "model_dir": str(args.model_dir.absolute()) if args.model_dir else None,
                    "device": args.psichic_device, "batch_size": args.batch_size, "weights": weight_hashes,
                    "requested_output": plan.psichic_output, "required_output": plan.required_psichic_output,
                    "chemical_representation": "Original SMILES submitted; normalized representation is not exposed by Predictor."},
        "ranking": {"boltz_field": plan.metric.value, "boltz_direction": "descending" if plan.metric.higher_is_better else "ascending",
                    "psichic_direction": "descending", "ties": "average ranks; shortlist boundary ties use input row order",
                    "ensemble": ("relative concentration-weighted normalized score; higher is better"
                                 if plan.ensemble is EnsembleMethod.CONCENTRATION else
                                 "experimental weighted mean of ranks within the both-successful cohort; lower is better"),
                    "ensemble_method": ({"method": plan.ensemble.value, "version": 1,
                                  "reference_commit": plan.ensemble.reference_commit,
                                  "reference_source": plan.ensemble.reference_source,
                                  "direction": plan.ensemble.direction,
                                  "meaning": ("relative normalized score, not a calibrated affinity"
                                              if plan.ensemble is EnsembleMethod.CONCENTRATION else "weighted average of ranks")}
                                 if plan.ensemble else None),
                    "shortlist": ("exactly K valid affinity and distribution rows"
                                  if plan.ensemble is EnsembleMethod.CONCENTRATION else "exactly K successful PSICHIC rows")
                                 + "; no replenishment after Boltz failure"},
    }
    _write_json(output / "manifest.json", manifest)
    psichic = None
    if protein:
        protein_file = output / "psichic_target.fasta"
        protein_file.write_text(f">{protein[0]}\n{protein[1]}\n")

        def psichic(rows):
            from applications.scorer.psichic_plus.predictor import Predictor
            from applications.scorer.psichic_plus.provenance import source_provenance

            predictor = Predictor(str(protein_file), str(args.model_dir.absolute()), args.psichic_device,
                                  str(args.psichic_cache or output / "psichic_cache"), args.batch_size)
            try:
                provenance = source_provenance(predictor)
                manifest["psichic"]["source"] = provenance
                _write_json(output / "manifest.json", manifest)
                run = score_psichic(rows, predictor, output=plan.required_psichic_output)
                run.metadata["source"] = provenance
                return run
            finally:
                device = getattr(predictor, "device", None)
                del predictor
                gc.collect()
                if device is not None and device.type == "cuda":
                    import torch

                    with torch.cuda.device(device):
                        torch.cuda.empty_cache()

    report = run_workflow(candidates, plan, boltz=runner, psichic=psichic)
    records = [row_record(row) for row in report.rows]
    _write_json(output / "results.json", records)
    with (output / "results.csv").open("w", newline="") as handle:
        fields = ["row", "key", "ID", "SMILES", "selection", "selection_reason", "psichic_status", "psichic_affinity",
                  "psichic_failure", "boltz_status", "affinity_probability_binary", "affinity_pred_value", "boltz_failure",
                  "psichic_screen_rank", "boltz_rank", "ensemble_psichic_rank", "ensemble_boltz_rank",
                  "experimental_combined_value", "final_rank", "unranked_reason",
                  "normalized_ensemble_score", "ensemble_method", "ensemble_reference_commit", "ensemble_score_direction",
                  "ensemble_reference_source", "ensemble_version", "ensemble_eligible", "ensemble_exclusion_reason", "psichic_distribution_status",
                  "psichic_alpha", "psichic_beta", "psichic_concentration", "psichic_distribution_failure",
                  "psichic_raw_alpha", "psichic_raw_beta"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            candidate, a, b = record["candidate"], record["psichic"], record["boltz"]
            values = {key: record[key] for key in fields if key in record}
            values.update(row=candidate["row"], key=candidate["key"], ID=candidate["id"], SMILES=candidate["smiles"],
                          psichic_status=a["status"], psichic_affinity=a.get("raw_affinity"), psichic_failure=a.get("reason"),
                          boltz_status=b["status"], affinity_probability_binary=b.get("affinity_probability_binary"),
                          affinity_pred_value=b.get("affinity_pred_value"), boltz_failure=b.get("reason"))
            distribution = a.get("distribution", {})
            eligibility = record["ensemble_eligibility"]
            values.update(ensemble_eligible=eligibility["eligible"], ensemble_exclusion_reason=eligibility["reason"],
                          psichic_distribution_status=distribution.get("state"), psichic_alpha=distribution.get("alpha"),
                          psichic_beta=distribution.get("beta"), psichic_concentration=distribution.get("concentration"),
                          psichic_distribution_failure=distribution.get("reason"),
                          psichic_raw_alpha=distribution.get("raw_alpha"), psichic_raw_beta=distribution.get("raw_beta"))
            writer.writerow(values)
    manifest.update(status="partial" if report.failed else "complete", engines=report.engines, errors=report.errors,
                    ensemble_cohort=report.ensemble.cohort if report.ensemble else [],
                    ensemble=report.ensemble.metadata() if report.ensemble else None,
                    finished_at=datetime.now(timezone.utc).isoformat())
    _write_json(output / "manifest.json", manifest)
    print(f"{manifest['status']}: {len(records)} input rows retained in {output}")
    return 1 if report.failed else 0


def main(argv=None):
    arguments = parser().parse_args(argv)
    try:
        return execute(arguments)
    except (ValueError, OSError, csv.Error, yaml.YAMLError) as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
