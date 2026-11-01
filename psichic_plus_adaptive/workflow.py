import gzip
import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from . import Failure, Molecule, Success, UPSTREAM_REVISION
from .library import collection_key, convert_library, read_library, safe_token

PACKAGE = Path(__file__).resolve().parent


def command(args: list[str], cwd: Path, env: dict | None = None, timeout: float | None = None) -> None:
    with (cwd / "commands.log").open("a") as handle:
        handle.write(json.dumps(args) + "\n")
        handle.flush()
        with subprocess.Popen(args, cwd=cwd, env=env, stdout=handle, stderr=subprocess.STDOUT,
                              start_new_session=True) as process:
            try:
                returncode = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                raise
            if returncode:
                raise subprocess.CalledProcessError(returncode, args)


def checked_upstream(source: Path, destination: Path) -> Path:
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if revision != UPSTREAM_REVISION:
        raise ValueError(f"AdaptiveFlow-VS must be checked out at {UPSTREAM_REVISION}, got {revision}")
    dirty = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"], text=True,
    )
    if dirty:
        raise ValueError("AdaptiveFlow-VS has tracked changes; supply a clean pinned checkout")
    command(["git", "clone", "--no-hardlinks", "--no-checkout", str(source), str(destination)], destination.parent)
    command(["git", "checkout", "--detach", UPSTREAM_REVISION], destination)
    patch = str(PACKAGE / "patches" / "af-vs-df74e005.patch")
    command(["git", "apply", "--check", patch], destination)
    command(["git", "apply", patch], destination)
    return destination


def local_environment(output: Path) -> dict:
    env = os.environ.copy()
    env['PATH'] = str(Path(sys.executable).parent) + os.pathsep + env.get('PATH', '')
    env['PYTHONPATH'] = str(PACKAGE.parent) + os.pathsep + env.get('PYTHONPATH', '')
    env['PSICHIC_ADAPTIVE_TMP'] = str(output / "tmp")
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        env[key] = "1"
    # The local upstream backend still constructs unused AWS clients.
    for key in tuple(env):
        if key.startswith('AWS_'):
            del env[key]
    env.update(AWS_EC2_METADATA_DISABLED="true", AWS_DEFAULT_REGION="us-east-1",
               AWS_ACCESS_KEY_ID="unused-local-backend", AWS_SECRET_ACCESS_KEY="unused-local-backend")
    return env


def stage_tools(upstream: Path, stage: Path) -> Path:
    tools = stage / "tools"
    (tools / "templates").mkdir(parents=True)
    for name in ('afvs_prepare_folders.py', 'afvs_prepare_workunits.py', 'afvs_submit_jobs.py'):
        shutil.copy2(upstream / "tools" / name, tools / name)
    for name in ('afvs_run.py', 'template1.bash'):
        shutil.copy2(upstream / "tools" / "templates" / name, tools / "templates" / name)
    (tools / "templates" / "afvs_run.py").chmod(0o755)
    return tools


def reconcile(stage: Path, expected: list[Molecule], workunits: dict, execution_error: str | None = None) -> pd.DataFrame:
    by_token = {m.token: m for m in expected}
    outcomes = {}
    issues = [execution_error] if execution_error else []
    total = 0

    def add(result, af_collection):
        if result.token not in by_token:
            issues.append(f"Unexpected token {result.token}")
        elif result.token in outcomes:
            issues.append(f"Duplicate completion {result.token}")
        elif collection_key(by_token[result.token]) != af_collection:
            issues.append(f"Collection mismatch for {result.token}")
        else:
            outcomes[result.token] = result

    for workunit, info in workunits.items():
        for subjob in info['subjobs']:
            overview = stage / "output-files" / "summary" / workunit / f"{subjob}.json.gz"
            if not overview.is_file():
                issues.append(f"Missing overview {workunit}/{subjob}")
                continue
            try:
                with gzip.open(overview, "rt") as handle:
                    summary = json.load(handle)
            except (OSError, ValueError) as error:
                issues.append(f"Unreadable overview {workunit}/{subjob}: {error}")
                continue
            total += summary['total_dockings']
            if summary['failed_downloads'] or summary['skipped_ligands']:
                issues.append(f"Upstream skipped inputs or failed downloads in {workunit}/{subjob}")
            for failure in summary['failed_list']:
                add(Failure(failure['ligand_key'], failure.get('stage', 'upstream'), failure['reason']),
                    failure['collection_key'])
            parquet = stage / "output-files" / "psichic" / "parquet" / workunit / f"{subjob}.parquet"
            if parquet.is_file():
                try:
                    successes = pd.read_parquet(parquet)
                except (OSError, ValueError) as error:
                    issues.append(f"Unreadable predictions {workunit}/{subjob}: {error}")
                    continue
                required = {'ligand', 'collection_key', 'attr_raw_affinity', 'score_min'}
                if not required.issubset(successes.columns):
                    issues.append(f"Missing prediction columns in {workunit}/{subjob}")
                    continue
                for row in successes.itertuples(index=False):
                    try:
                        raw, score = float(row.attr_raw_affinity), float(row.score_min)
                    except (ValueError, TypeError):
                        issues.append(f"Invalid score for {row.ligand}")
                        continue
                    if not math.isfinite(raw) or not math.isfinite(score) or score != -raw:
                        issues.append(f"Invalid score for {row.ligand}")
                        continue
                    add(Success(row.ligand, raw, score), row.collection_key)
    if total != len(expected):
        issues.append(f"Upstream completed {total} of {len(expected)} expected items")
    records = []
    for molecule in expected:
        result = outcomes.get(molecule.token)
        if result is None:
            issues.append(f"Missing completion {molecule.token}")
            result = Failure(molecule.token, "completion", "Missing upstream completion")
        record = {**asdict(molecule), "screen_stage": stage.name}
        if isinstance(result, Success):
            record.update(status="success", raw_affinity=result.raw_affinity, adaptive_score=result.adaptive_score,
                          failure_stage="", failure_reason="")
        else:
            record.update(status="failed", raw_affinity=None, adaptive_score=None,
                          failure_stage=result.stage, failure_reason=result.reason)
        records.append(record)
    frame = pd.DataFrame(records)
    frame.to_csv(stage / "results.csv", index=False)
    frame[frame.status == "failed"].to_csv(stage / "failures.csv", index=False)
    if issues:
        (stage / "incomplete.json").write_text(json.dumps(issues, indent=2) + "\n")
        raise RuntimeError(f"Incomplete upstream output in {stage}; inspect incomplete.json and results.csv")
    return frame


def run_stage(upstream: Path, stage: Path, library: Path, todo: Path, expected: list[Molecule],
              config: dict, samples_per_tranche: int, env: dict, stage_timeout: float = 3600) -> pd.DataFrame:
    tools = stage_tools(upstream, stage)
    inputs = stage / "input-files" / "psichic"
    inputs.mkdir(parents=True)
    (inputs / "config.txt").write_text(json.dumps(config, indent=2) + "\n")
    shutil.copyfile(todo, tools / "templates" / "todo.all")
    prescreen = stage.name == "prescreen"
    values = {
        "job_name": stage.name,
        "threads_to_use": 1, "threads_per_docking": 1,
        "data_collection_addressing_mode": "standard", "job_addressing_mode": "standard",
        "data_collection_identifier": "", "job_storage_mode": "sharedfs", "data_storage_mode": "sharedfs",
        "batchsystem": "bash", "bash_template": "templates/template1.bash", "bash_array_job_size": 1,
        "dockings_per_subjob": max(1, len(expected)),
        "ligand_library_format": "smi", "program_timeout": 3600,
        "docking_scenario_basefolder": str(stage / "input-files"),
        "docking_scenario_names": "psichic", "docking_scenario_methods": "external_batch",
        "docking_scenario_replicas": 1, "docking_scenario_batchsizes": config['batch_size'],
        "summary_formats": "parquet",
        "collection_folder": str(library / ("sparse" if prescreen else "full")),
        "collection_list_file": "templates/todo.all", "collection_list_type": "standard",
        "tempdir_default": env['PSICHIC_ADAPTIVE_TMP'],
        "prescreen_mode": int(prescreen), "prescreen_ligands_per_tranche": samples_per_tranche,
        "run_atom_check": 0, "dynamic_tranche_filtering": 0,
        "object_store_job_prefix": "unused", "object_store_job_bucket": "unused", "aws_region": "us-east-1",
    }
    (tools / "templates" / "all.ctrl").write_text("".join(f"{key}={value}\n" for key, value in values.items()))
    for name in ('afvs_prepare_folders.py', 'afvs_prepare_workunits.py'):
        command([sys.executable, name], tools, env)
    workunits = json.loads((stage / "workflow" / "status.json").read_text())['workunits']
    command([sys.executable, "afvs_submit_jobs.py", "1", str(len(workunits))], tools, env)
    execution_error = None
    deadline = time.monotonic() + stage_timeout
    for workunit in workunits:
        try:
            command(["bash", str(stage / "workflow" / "workunits" / workunit / "run.bash")], tools, env,
                    timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            execution_error = f"Stage exceeded {stage_timeout} seconds; worker process group stopped"
            break
        except subprocess.CalledProcessError as error:
            execution_error = f"Upstream worker exited with {error.returncode}; inspect tools/commands.log"
            break
    return reconcile(stage, expected, workunits, execution_error)


def select_primary(upstream: Path, library: Path, prescreen: pd.DataFrame,
                   target: int, output: Path, env: dict) -> Path:
    if target < 1:
        raise ValueError("primary_target must be positive")
    scores = prescreen[prescreen.status == "success"].copy()
    if scores.empty:
        raise ValueError("All prescreen predictions failed; no tranches can be selected")
    raw = scores.raw_affinity.astype(float)
    score = scores.adaptive_score.astype(float)
    if not (score.map(math.isfinite).all() and raw.map(math.isfinite).all()
            and (score < 0).all() and (score == -raw).all()):
        raise ValueError("Selector requires finite negative affinities as adaptive scores")
    scores['Tranche'] = scores.tranche.map(lambda value: safe_token("t", value))
    collections = pd.read_parquet(library / "collections.parquet")
    eligible = collections[collections.Tranche.isin(scores.Tranche)].copy()
    available = int(eligible.LigandCount.sum())
    if target > available:
        raise ValueError(f"Primary target {target} is unreachable; scored tranches contain {available} molecules")
    eligible_path = output / "eligible.collections.parquet"
    eligible.to_parquet(eligible_path, index=False)
    ranking = pd.DataFrame({"Tranche": scores.Tranche, "Collection": scores.collection_id,
                            "Ligand": scores.token, "ScoreMin": score})
    ranking_path = output / "prescreen.ranking.csv.gz"
    ranking.to_csv(ranking_path, index=False, compression="gzip")
    todo = output / "primary.todo"
    command([sys.executable, str(upstream / "tools" / "templates" / "create_todofile_atg-primaryscreen.py"),
             str(ranking_path), str(eligible_path), str(library / "tranches.parquet"),
             "tranche_ave_score", str(todo), str(target), ".*"], output, env)
    counts = {f"{row.Tranche}_{row.Collection}": row.LigandCount for row in eligible.itertuples(index=False)}
    selected = {}
    for line in todo.read_text().splitlines():
        key, count = line.split()
        if key not in counts or key in selected or int(count) != counts[key]:
            raise RuntimeError("Upstream selector generated an invalid collection todo")
        selected[key] = int(count)
    if sum(selected.values()) < target:
        raise RuntimeError("Upstream selector did not reach the primary target")
    (output / "selection.json").write_text(json.dumps({
        "scoring_mode": "tranche_ave_score", "primary_target": target,
        "selected_molecules": sum(selected.values()), "selected_collections": selected,
        "excluded_unscored_collections": len(collections) - len(eligible),
        "todo_sha256": hashlib.sha256(todo.read_bytes()).hexdigest(),
    }, indent=2) + "\n")
    return todo


def run(library_csv: Path, fasta: Path, model_dir: Path, upstream: Path, output: Path,
        samples_per_tranche: int, primary_target: int, device: str, batch_size: int, stage_timeout: float = 3600) -> None:
    if not math.isfinite(stage_timeout) or stage_timeout <= 0:
        raise ValueError("Stage timeout must be positive and finite")
    molecules = read_library(library_csv)
    if samples_per_tranche < 1 or primary_target < 1 or batch_size < 1:
        raise ValueError("Sample count, primary target, and batch size must be positive")
    if primary_target > len(molecules):
        raise ValueError(f"Primary target {primary_target} is unreachable; library has {len(molecules)} molecules")
    if not fasta.is_file():
        raise ValueError(f"Target FASTA does not exist: {fasta}")
    for filename in ('config.json', 'degree.pt', 'model.pt'):
        if not (model_dir / filename).is_file():
            raise ValueError(f"Missing model file: {model_dir / filename}")
    output = output.resolve()
    if not re.fullmatch(r"[A-Za-z0-9_./-]+", str(output)):
        raise ValueError("The upstream Bash backend requires an output path without spaces or shell metacharacters")
    output.mkdir(parents=True, exist_ok=False)
    (output / "tmp").mkdir()
    shutil.copyfile(fasta, output / "target.fasta")
    env = local_environment(output)
    prepared = checked_upstream(upstream.resolve(), output / "upstream")
    library = output / "library"
    samples = set(convert_library(molecules, library, samples_per_tranche))
    config = {"fasta": str(output / "target.fasta"), "model_dir": str(model_dir.resolve()),
              "device": device, "cache": str(output / "cache"), "batch_size": batch_size}
    prescreen = run_stage(prepared, output / "prescreen", library, library / "prescreen.todo",
                          [m for m in molecules if m.token in samples], config, samples_per_tranche, env, stage_timeout)
    todo = select_primary(prepared, library, prescreen, primary_target, output, env)
    selected = {line.split()[0] for line in todo.read_text().splitlines()}
    primary = run_stage(prepared, output / "primaryscreen", library, todo,
                        [m for m in molecules if collection_key(m) in selected], config, samples_per_tranche, env, stage_timeout)
    ranked = primary[primary.status == "success"].sort_values(['raw_affinity', 'token'], ascending=[False, True])
    ranked.to_csv(output / "ranked.csv", index=False)
    (output / "run.json").write_text(json.dumps({
        "upstream_revision": UPSTREAM_REVISION, "device": device, "scoring_processes": 1,
        "prescreen": prescreen.status.value_counts().to_dict(),
        "primaryscreen": primary.status.value_counts().to_dict(),
        "ranked": len(ranked),
    }, indent=2) + "\n")
