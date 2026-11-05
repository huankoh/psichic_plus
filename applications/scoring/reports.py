"""Disk-backed global ranking and immutable report snapshots."""

import csv
from dataclasses import asdict, dataclass
import hashlib
import math
import os
from pathlib import Path
import uuid

from applications.scorer import Output
from applications.scorer.boltz2.ensemble import (
    average_tie_rank, concentration_value, normalize_value, weighted_rank_value,
)
from applications.scorer.boltz2.models import EnsembleMethod
from .contracts import Completion, ContractError, Engine, SelectionState, canonical, loads
from .evidence import atomic_pointer, fsync_directory, read_document, write_file


@dataclass(frozen=True)
class Waiting:
    pending: int
    submitted: int
    unknown: int


@dataclass(frozen=True)
class Report:
    version: int
    report_id: str
    status: Completion
    directory: Path
    accepted_receipt_digest: str


def _snapshot(run, db):
    checksum = hashlib.sha256((canonical({"settings": asdict(run.settings), "population": run.population_digest,
                                        "selection": run._get(db, "selection")}) + "\n").encode())
    accepted = hashlib.sha256()
    for row in db.execute("SELECT ordinal,engine,state,expected,accepted FROM evaluations ORDER BY ordinal,engine"):
        checksum.update((canonical(list(row)) + "\n").encode())
        if row["accepted"]:
            accepted.update((canonical([row["ordinal"], row["engine"], row["accepted"]]) + "\n").encode())
    for row in db.execute("SELECT ordinal FROM selection ORDER BY ordinal"):
        checksum.update(f"selected:{row[0]}\n".encode())
    return checksum.hexdigest(), accepted.hexdigest()


def _score(run, db):
    settings = run.settings
    db.create_function("tie_rank", 2, average_tie_rank, deterministic=True)
    db.create_function("normalize", 3, normalize_value, deterministic=True)
    db.create_function("concentration_score", 3, concentration_value, deterministic=True)
    db.create_function("weighted_rank", 4, weighted_rank_value, deterministic=True)
    db.create_function("natural_log", 1, math.log, deterministic=True)
    db.execute("""CREATE TEMP TABLE scores (
        ordinal INTEGER PRIMARY KEY, p REAL, b REAL, c REAL, p_rank REAL, b_rank REAL, score REAL)""")
    db.execute("""CREATE TEMP VIEW accepted AS SELECT v.ordinal,v.engine,r.*
                  FROM evaluations v JOIN receipts r ON r.digest=v.accepted WHERE r.status='succeeded'""")
    if settings.mode == "psichic-only":
        db.execute("INSERT INTO scores(ordinal,p,score) SELECT ordinal,p,p FROM accepted WHERE engine='psichic'")
        direction = "DESC"
    elif settings.ensemble:
        b_column = "b" if settings.metric.value == "affinity_pred_value" else "probability"
        distribution = " AND p.distribution='valid'" if settings.ensemble is EnsembleMethod.CONCENTRATION else ""
        db.execute(f"""INSERT INTO scores(ordinal,p,b,c)
            SELECT s.ordinal,p.p,b.{b_column},p.concentration FROM selection s
            JOIN accepted p ON p.ordinal=s.ordinal AND p.engine='psichic'
            JOIN accepted b ON b.ordinal=s.ordinal AND b.engine='boltz' WHERE 1=1 {distribution}""")
        direction = "DESC" if settings.ensemble is EnsembleMethod.CONCENTRATION else "ASC"
    else:
        b_column = "b" if settings.metric.value == "affinity_pred_value" else "probability"
        db.execute(f"""INSERT INTO scores(ordinal,b,score) SELECT s.ordinal,b.{b_column},b.{b_column}
                       FROM selection s JOIN accepted b ON b.ordinal=s.ordinal AND b.engine='boltz'""")
        direction = "DESC" if settings.metric.higher_is_better else "ASC"
    constants = []
    if settings.ensemble is EnsembleMethod.CONCENTRATION:
        bounds = db.execute("SELECT min(p),max(p),min(-b),max(-b),min(natural_log(c)),max(natural_log(c)) FROM scores").fetchone()
        if bounds[0] is not None:
            for name, start in (("psichic_affinity", 0), ("boltz_affinity", 2), ("log_concentration", 4)):
                if bounds[start] == bounds[start + 1]:
                    constants.append(name)
            db.execute("""UPDATE scores SET score=concentration_score(
                normalize(p,?,?),normalize(-b,?,?),normalize(natural_log(c),?,?))""", tuple(bounds))
    elif settings.ensemble is EnsembleMethod.RANK:
        b_direction = "DESC" if settings.metric.higher_is_better else "ASC"
        db.execute(f"""CREATE TEMP TABLE ranks AS SELECT ordinal,
            tie_rank(rank() OVER (ORDER BY p DESC), count(*) OVER (PARTITION BY p)) AS p_rank,
            tie_rank(rank() OVER (ORDER BY b {b_direction}), count(*) OVER (PARTITION BY b)) AS b_rank FROM scores""")
        db.execute("CREATE UNIQUE INDEX rank_ordinal ON ranks(ordinal)")
        db.execute("""UPDATE scores SET (p_rank,b_rank)=(SELECT p_rank,b_rank FROM ranks WHERE ranks.ordinal=scores.ordinal)""")
        db.execute("UPDATE scores SET score=weighted_rank(p_rank,b_rank,?,?)", (settings.psichic_weight, settings.boltz_weight))
    db.execute(f"""CREATE TEMP TABLE ranked AS SELECT *,
        tie_rank(rank() OVER (ORDER BY score {direction}),count(*) OVER (PARTITION BY score)) AS final_rank,
        row_number() OVER (ORDER BY score {direction},ordinal) AS position FROM scores""")
    db.execute("CREATE UNIQUE INDEX ranked_ordinal ON ranked(ordinal)")
    count, distinct = db.execute("SELECT count(*),count(DISTINCT score) FROM scores").fetchone()
    discrimination = ("no_scored_entries" if not count else "singleton_lacks_discrimination" if count == 1 else
                      "all_tied_lacks_discrimination" if distinct == 1 else "multiple_distinct_scores")
    return count, constants, discrimination, direction.lower()


def _engine_record(run, row, engine):
    state = row[engine + "_state"]
    if state is None:
        return {"state": "not-run", "attempt_id": None, "raw": None, "receipt": None, "artifacts": []}
    accepted = row[engine + "_receipt"]
    artifacts = []
    if accepted:
        receipt = read_document(run.path, f"receipts/{accepted}.json")
        artifacts = [{**artifact, "link": f"../../blobs/{artifact['sha256']}"} for artifact in receipt["artifacts"]]
    return {"state": state, "attempt_id": row[engine + "_attempt"],
            "raw": loads(row[engine + "_record"]) if accepted else None,
            "receipt": f"../../receipts/{accepted}.json" if accepted else None, "artifacts": artifacts}


def _store_rows(run, db, version, shortage):
    query = """SELECT e.*,s.ordinal AS selected,r.score,r.final_rank,r.position,r.p_rank,r.b_rank,
        p.state AS psichic_state,p.expected AS psichic_attempt,p.accepted AS psichic_receipt,pr.record AS psichic_record,
        b.state AS boltz_state,b.expected AS boltz_attempt,b.accepted AS boltz_receipt,br.record AS boltz_record
        FROM entries e LEFT JOIN selection s USING(ordinal) LEFT JOIN ranked r USING(ordinal)
        LEFT JOIN evaluations p ON p.ordinal=e.ordinal AND p.engine='psichic'
        LEFT JOIN receipts pr ON pr.digest=p.accepted
        LEFT JOIN evaluations b ON b.ordinal=e.ordinal AND b.engine='boltz'
        LEFT JOIN receipts br ON br.digest=b.accepted ORDER BY e.ordinal"""
    for row in db.execute(query):
        psichic, boltz = _engine_record(run, row, "psichic"), _engine_record(run, row, "boltz")
        reason = None
        if row["final_rank"] is None:
            if psichic["state"] == "failed":
                reason = "psichic_failed"
            elif run.settings.ensemble is EnsembleMethod.CONCENTRATION and psichic["raw"] and psichic["raw"]["distribution"]["state"] != "valid":
                reason = "psichic_distribution_failed"
            elif shortage:
                reason = "insufficient_psichic_successes"
            elif row["selected"] is None:
                reason = "below_shortlist"
            elif boltz["state"] == "failed":
                reason = "boltz_failed"
            else:
                reason = "no_successful_outcome"
        record = {"entry": run._row(row).to_dict(), "selected": row["selected"] is not None,
                  "status": "ranked" if row["final_rank"] is not None else "unranked", "reason": reason,
                  "psichic": psichic, "boltz": boltz, "score": row["score"], "final_rank": row["final_rank"],
                  "position": row["position"], "ensemble_psichic_rank": row["p_rank"], "ensemble_boltz_rank": row["b_rank"]}
        db.execute("INSERT INTO report_rows VALUES (?,?,?,?)", (version, row["ordinal"], canonical(record), row["position"]))


def _metadata(run, db, version, snapshot, accepted_digest, selection):
    ranked, constants, discrimination, direction = _score(run, db)
    counts = {"entries": run.row_count, "ranked": ranked,
              "selected": db.execute("SELECT count(*) FROM selection").fetchone()[0]}
    counts["unranked"] = run.row_count - ranked
    counts["not_selected"] = run.row_count - counts["selected"]
    for engine in Engine:
        for state in ("succeeded", "failed"):
            counts[f"{engine.value}_{state}"] = db.execute("SELECT count(*) FROM evaluations WHERE engine=? AND state=?", (engine.value, state)).fetchone()[0]
    counts["distribution_failed"] = db.execute("""SELECT count(*) FROM evaluations v JOIN receipts r ON r.digest=v.accepted
                                               WHERE v.engine='psichic' AND r.distribution='distribution-failure'""").fetchone()[0]
    shortage = selection.state is SelectionState.SHORTAGE
    status = Completion.SHORTAGE if shortage else (Completion.PARTIAL if counts["psichic_failed"] or counts["boltz_failed"] or
             (run.settings.required_output is Output.DISTRIBUTION and counts["distribution_failed"]) else Completion.COMPLETE)
    method = run.settings.ensemble
    meaning = ("relative concentration-weighted normalized score, not calibrated affinity" if method is EnsembleMethod.CONCENTRATION else
               "weighted average of ranks in the common successful cohort" if method else
               "raw PSICHIC binding affinity" if run.settings.mode == "psichic-only" else f"raw Boltz {run.settings.metric.value}")
    warnings = []
    if "lacks_discrimination" in discrimination:
        warnings.append("No discrimination in this cohort; computed scores and average ranks are retained.")
    if constants:
        warnings.append("Constant series normalize to 0.5: " + ", ".join(constants))
    if version > 1:
        warnings.append("Accepted observations changed. Cohort changes can alter every rank and normalized score; previous reports are preserved.")
    return {"schema_version": 1, "run_id": run.run_id, "population_digest": run.population_digest,
            "report_version": version, "report_id": f"{run.run_id}:{version}", "snapshot_digest": snapshot,
            "accepted_receipt_digest": accepted_digest, "status": status.value, "counts": counts,
            "shortage": {"required": selection.required, "eligible": selection.eligible} if shortage else None,
            "selection_frozen": run._get(db, "selection") == "committed", "settings": asdict(run.settings),
            "ranking": {"method": method.value if method else "psichic-affinity-v1" if run.settings.mode == "psichic-only" else "boltz-rank-v1",
                        "reference_source": method.reference_source if method else "applications/scorer/boltz2/workflow.py",
                        "reference_commit": method.reference_commit if method else "85fe5eafa9b9ede93e7c025cc4494c1e25ef656b",
                        "direction": direction, "meaning": meaning, "constant_series": constants,
                        "discrimination": discrimination, "ties": "average ranks; display ties use manifest order",
                        "distribution_scale": "normalized activity [0,1]; affinity = 12 * alpha / (alpha + beta)"},
            "warnings": warnings}


def _report(run, row):
    metadata = loads(row["metadata"])
    return Report(row["version"], metadata["report_id"], Completion(metadata["status"]),
                  run.path / "reports" / f"v{row['version']:08d}", metadata["accepted_receipt_digest"])


def _files_digest(directory):
    result = {}
    for name in ("results.json", "results.csv", "summary.md"):
        with (directory / name).open("rb") as handle:
            result[name] = {"size": os.fstat(handle.fileno()).st_size, "sha256": hashlib.file_digest(handle, "sha256").hexdigest()}
    return result


def _publish(run, db, row, top_n):
    report = _report(run, row)
    metadata = loads(row["metadata"])
    if report.directory.exists():
        expected = read_document(report.directory, "files.json")
        if expected != _files_digest(report.directory):
            raise ContractError("Immutable published report was modified")
    else:
        staging = run.path / "reports" / f".v{report.version:08d}.{uuid.uuid4().hex}.tmp"
        staging.mkdir()
        with (staging / "results.json").open("x") as output, (staging / "results.csv").open("x", newline="") as csv_file:
            output.write(canonical(metadata)[:-1] + ',"rows":[')
            fields = ["ordinal", "key", "id", "smiles", "source", "metadata", "selected", "status", "reason", "score",
                      "final_rank", "position", "ensemble_psichic_rank", "ensemble_boltz_rank", "psichic", "boltz"]
            writer = csv.DictWriter(csv_file, fieldnames=fields)
            writer.writeheader()
            first = True
            for record, in db.execute("SELECT record FROM report_rows WHERE version=? ORDER BY ordinal", (report.version,)):
                output.write(("" if first else ",") + record)
                first = False
                value = loads(record)
                flat = {**value.pop("entry"), **value}
                for key in ("metadata", "psichic", "boltz"):
                    flat[key] = canonical(flat[key])
                writer.writerow(flat)
            output.write("]}\n")
            for handle in (output, csv_file):
                handle.flush()
                os.fsync(handle.fileno())
        summary = [f"# {metadata['status'].upper()}\n", f"{metadata['counts']['ranked']} of {run.row_count} entries ranked.\n",
                   f"Method: {metadata['ranking']['method']}.\n", f"Meaning: {metadata['ranking']['meaning']}.\n",
                   f"Direction: {metadata['ranking']['direction']}.\n", "Raw PSICHIC affinity is higher-better; raw Boltz affinity is lower-better and binder probability is higher-better.\n",
                   "Final rank uses average ties; position is the human display order.\n"]
        if metadata["shortage"]:
            summary.append(f"Shortlist shortage: {metadata['shortage']['eligible']} eligible, {metadata['shortage']['required']} required. No Boltz tasks were prepared.\n")
        summary.extend(warning + "\n" for warning in metadata["warnings"])
        summary.append("Counts: " + ", ".join(f"{name}={count}" for name, count in metadata["counts"].items()) + "\n")
        columns = ["Position", "Average rank", "Entry key", "Original ID", "SMILES", "PSICHIC affinity", "Boltz affinity", "Binder probability"]
        distribution = run.settings.psichic_output is Output.DISTRIBUTION
        if distribution:
            columns += ["Alpha", "Beta", "Concentration"]
        columns += ["Score", "Status"]
        summary.append("\n| " + " | ".join(columns) + " |\n|" + "---|" * len(columns) + "\n")
        for record, in db.execute("SELECT record FROM report_rows WHERE version=? AND position IS NOT NULL ORDER BY position LIMIT ?", (report.version, top_n)):
            value = loads(record)
            p, b = value["psichic"]["raw"] or {}, value["boltz"]["raw"] or {}
            cells = [value["position"], value["final_rank"], value["entry"]["key"], value["entry"]["id"], value["entry"]["smiles"],
                     p.get("raw_affinity"), b.get("affinity_pred_value"), b.get("affinity_probability_binary")]
            if distribution:
                d = p.get("distribution", {})
                cells += [d.get(name) for name in ("alpha", "beta", "concentration")]
            cells += [value["score"], value["status"]]
            escaped = [("-" if cell is None else str(cell)).replace("|", "\\|").replace("\n", " ").replace("\r", " ")
                       for cell in cells]
            summary.append("| " + " | ".join(escaped) + " |\n")
        summary.append("\nFull population and evidence links: [JSON](results.json), [CSV](results.csv).\n")
        write_file(staging / "summary.md", "\n".join(summary))
        write_file(staging / "files.json", canonical(_files_digest(staging)) + "\n")
        for path in staging.iterdir():
            path.chmod(0o444)
        fsync_directory(staging)
        os.rename(staging, report.directory)
        fsync_directory(report.directory.parent)
    db.execute("BEGIN IMMEDIATE")
    db.execute("UPDATE reports SET state='published' WHERE version=?", (report.version,))
    db.execute("COMMIT")
    _repair_latest(run, db)
    return report


def _repair_latest(run, db):
    newest = db.execute("SELECT version,snapshot FROM reports WHERE state='published' ORDER BY version DESC LIMIT 1").fetchone()
    if newest is None:
        return
    pointer = {"schema_version": 1, "version": newest["version"], "snapshot_digest": newest["snapshot"],
               "directory": f"reports/v{newest['version']:08d}"}
    if not (run.path / "latest.json").exists() or read_document(run.path, "latest.json") != pointer:
        atomic_pointer(run.path / "latest.json", pointer)


def finalize(run, *, top_n):
    if type(top_n) is not int or not 0 <= top_n <= 100:
        raise ContractError("Summary top_n must be between 0 and 100")
    with run._locked() as db:
        for interrupted in db.execute("SELECT * FROM reports WHERE state='building' ORDER BY version"):
            _publish(run, db, interrupted, top_n)
        _repair_latest(run, db)
        selection = run._select(db)
        unresolved = dict(db.execute("SELECT state,count(*) FROM evaluations WHERE state NOT IN ('succeeded','failed') GROUP BY state"))
        if unresolved:
            return Waiting(unresolved.get("pending", 0), unresolved.get("submitted", 0), unresolved.get("interrupted-or-unknown", 0))
        snapshot, accepted_digest = _snapshot(run, db)
        previous = db.execute("SELECT * FROM reports ORDER BY version DESC LIMIT 1").fetchone()
        if previous is not None and previous["snapshot"] == snapshot:
            return _publish(run, db, previous, top_n)
        db.execute("BEGIN IMMEDIATE")
        version = db.execute("INSERT INTO reports(snapshot,metadata,state) VALUES (?,'{}','building')", (snapshot,)).lastrowid
        metadata = _metadata(run, db, version, snapshot, accepted_digest, selection)
        db.execute("UPDATE reports SET metadata=? WHERE version=?", (canonical(metadata), version))
        _store_rows(run, db, version, selection.state is SelectionState.SHORTAGE)
        db.execute("COMMIT")
        row = db.execute("SELECT * FROM reports WHERE version=?", (version,)).fetchone()
        return _publish(run, db, row, top_n)


def latest(run):
    if not (run.path / "latest.json").exists():
        return None
    with run._locked() as db:
        pointer = read_document(run.path, "latest.json")
        row = db.execute("SELECT * FROM reports WHERE version=? AND state='published'", (pointer["version"],)).fetchone()
        if row is None or row["snapshot"] != pointer["snapshot_digest"]:
            raise ContractError("Latest pointer does not name a published snapshot")
        return _report(run, row)


def results(run, version):
    if version is None:
        current = latest(run)
        if current is None:
            return
        version = current.version
    with run._locked() as db:
        if db.execute("SELECT 1 FROM reports WHERE version=? AND state='published'", (version,)).fetchone() is None:
            raise ContractError("Report version is not published")
        for record, in db.execute("SELECT record FROM report_rows WHERE version=? ORDER BY ordinal", (version,)):
            yield loads(record)
