"""Single-host coordinator for a frozen, disk-backed scoring population."""

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import fcntl
import hashlib
import os
from pathlib import Path
import socket
import sqlite3
import uuid

from applications.scorer import Output
from applications.scorer.boltz2.models import EnsembleMethod

from .contracts import (
    Acceptance, BatchContext, ContractError, Engine, Entry, Fingerprint, IncompleteRun, ManifestRow,
    MAX_BATCH, Ownership, Resolution, RunBusy, SCHEMA, SelectionState, Settings, State, StateError,
    Target, Task, canonical, key, loads, row_key, text,
)
from .evidence import fsync_directory, publish_evidence, stage_receipt

DDL = """
CREATE TABLE meta (name TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE entries (ordinal INTEGER PRIMARY KEY, id TEXT NOT NULL, smiles TEXT NOT NULL,
                      source TEXT NOT NULL, metadata TEXT NOT NULL);
CREATE TABLE evaluations (
    ordinal INTEGER NOT NULL REFERENCES entries, engine TEXT NOT NULL,
    state TEXT NOT NULL CHECK(state IN ('pending','submitted','succeeded','failed','interrupted-or-unknown')),
    expected TEXT, accepted TEXT, stopped INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY(ordinal,engine));
CREATE INDEX dispatch ON evaluations(engine,state,ordinal);
CREATE TABLE attempts (id TEXT PRIMARY KEY, ordinal INTEGER NOT NULL, engine TEXT NOT NULL,
                       task TEXT NOT NULL, receipt TEXT);
CREATE INDEX attempt_entry ON attempts(ordinal,engine);
CREATE TABLE receipts (
    digest TEXT PRIMARY KEY, attempt TEXT NOT NULL UNIQUE REFERENCES attempts,
    record TEXT NOT NULL,
    status TEXT GENERATED ALWAYS AS (json_extract(record,'$.status')) VIRTUAL,
    p REAL GENERATED ALWAYS AS (json_extract(record,'$.raw_affinity')) VIRTUAL,
    concentration REAL GENERATED ALWAYS AS (json_extract(record,'$.distribution.concentration')) VIRTUAL,
    distribution TEXT GENERATED ALWAYS AS (json_extract(record,'$.distribution.state')) VIRTUAL,
    b REAL GENERATED ALWAYS AS (json_extract(record,'$.affinity_pred_value')) VIRTUAL,
    probability REAL GENERATED ALWAYS AS (json_extract(record,'$.affinity_probability_binary')) VIRTUAL);
CREATE TABLE reconciliations (sequence INTEGER PRIMARY KEY, attempt TEXT NOT NULL REFERENCES attempts,
                             resolution TEXT NOT NULL, evidence TEXT NOT NULL);
CREATE TABLE selection (ordinal INTEGER PRIMARY KEY REFERENCES entries);
CREATE TABLE reports (version INTEGER PRIMARY KEY AUTOINCREMENT, snapshot TEXT NOT NULL,
                      metadata TEXT NOT NULL, state TEXT NOT NULL CHECK(state IN ('building','published')));
CREATE TABLE report_rows (version INTEGER NOT NULL REFERENCES reports, ordinal INTEGER NOT NULL,
                          record TEXT NOT NULL, position INTEGER,
                          PRIMARY KEY(version,ordinal));
CREATE INDEX report_order ON report_rows(version,position,ordinal);
"""


@dataclass(frozen=True)
class Selection:
    state: SelectionState
    eligible: int
    required: int
    selected: int


@dataclass(frozen=True)
class Diagnostic:
    entry_key: str
    engine: Engine
    state: State
    attempt_id: str | None
    resumable: bool


class ScoringRun:
    def __init__(self, path):
        self.path = Path(path).absolute()

    @property
    def run_id(self):
        return self._run_id

    @property
    def population_digest(self):
        return self._population_digest

    @property
    def row_count(self):
        return self._row_count

    @property
    def target(self):
        return self._target

    @property
    def settings(self):
        return self._settings

    @contextmanager
    def _locked(self):
        try:
            fd = os.open(self.path / "coordinator.lock", os.O_RDWR | os.O_NOFOLLOW)
        except FileNotFoundError as error:
            raise IncompleteRun("Run creation did not finish; preserve it and create a new run path") from error
        connection = None
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RunBusy("Another coordinator owns this run") from error
            database = self.path / "state.sqlite3"
            if not database.is_file():
                raise IncompleteRun("Run creation did not finish")
            connection = sqlite3.connect(database, isolation_level=None)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute("PRAGMA temp_store=FILE")
            connection.execute("PRAGMA cache_size=-8192")
            if connection.execute("PRAGMA journal_mode").fetchone()[0] != "delete":
                raise IncompleteRun("Unsupported journal configuration")
            yield connection
        finally:
            if connection is not None:
                connection.close()
            os.close(fd)

    @staticmethod
    def _get(db, name):
        value = db.execute("SELECT value FROM meta WHERE name=?", (name,)).fetchone()
        if value is None:
            raise IncompleteRun(f"Run has no committed {name}")
        return loads(value[0])

    @staticmethod
    def _set(db, name, value):
        db.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (name, canonical(value)))

    @classmethod
    def create(cls, path, entries, *, target, settings, storage="same-host-local"):
        if not isinstance(target, Target) or not isinstance(settings, Settings):
            raise ContractError("Create requires typed target and settings")
        if storage != "same-host-local":
            raise ContractError("Only same-host local filesystem coordination is supported")
        run = cls(path)
        run.path.mkdir(exist_ok=False)
        fd = os.open(run.path / "coordinator.lock", os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            for name in ("staging", "blobs", "receipts", "reports"):
                (run.path / name).mkdir()
            db = sqlite3.connect(run.path / "state.sqlite3", isolation_level=None)
            try:
                db.execute("PRAGMA journal_mode=DELETE")
                db.execute("PRAGMA synchronous=FULL")
                db.executescript(DDL)
                db.execute("BEGIN IMMEDIATE")
                population = hashlib.sha256((canonical({"schema_version": SCHEMA, "target": asdict(target)}) + "\n").encode())
                count = 0
                for count, entry in enumerate(entries, 1):
                    if not isinstance(entry, Entry):
                        raise ContractError("Population must yield Entry values")
                    row = ManifestRow(count, entry)
                    population.update((canonical(row.to_dict()) + "\n").encode())
                    db.execute("INSERT INTO entries VALUES (?,?,?,?,?)", (count, entry.id, entry.smiles, entry.source, entry.metadata_json))
                    for engine in settings.engines:
                        if engine is Engine.BOLTZ and settings.pyramid:
                            continue
                        db.execute("INSERT INTO evaluations(ordinal,engine,state) VALUES (?,?,'pending')", (count, engine.value))
                for name, value in {"schema_version": SCHEMA, "run_id": uuid.uuid4().hex,
                                    "target": asdict(target), "settings": asdict(settings),
                                    "population_digest": population.hexdigest(), "row_count": count,
                                    "host": socket.gethostname(), "storage": storage,
                                    "selection": "waiting" if settings.pyramid else "committed"}.items():
                    run._set(db, name, value)
                if not settings.pyramid:
                    db.execute("INSERT INTO selection SELECT ordinal FROM entries")
                db.execute("COMMIT")
            finally:
                db.close()
            fsync_directory(run.path)
            fsync_directory(run.path.parent)
        finally:
            os.close(fd)
        return cls.open(path)

    @classmethod
    def open(cls, path):
        run = cls(path)
        try:
            with run._locked() as db:
                if run._get(db, "schema_version") != SCHEMA:
                    raise IncompleteRun("Unsupported run schema")
                if run._get(db, "host") != socket.gethostname():
                    raise StateError("Run belongs to another host; cross-host coordination is unsupported")
                run._run_id = run._get(db, "run_id")
                run._population_digest = run._get(db, "population_digest")
                run._row_count = run._get(db, "row_count")
                run._target = Target.from_dict(run._get(db, "target"))
                run._settings = Settings.from_dict(run._get(db, "settings"))
        except sqlite3.DatabaseError as error:
            raise IncompleteRun("Run creation is incomplete or database is damaged; preserve evidence") from error
        return run

    @staticmethod
    def _row(record):
        return ManifestRow(record["ordinal"], Entry(record["id"], record["smiles"], record["source"], record["metadata"]))

    def population(self):
        with self._locked() as db:
            for record in db.execute("SELECT * FROM entries ORDER BY ordinal"):
                yield self._row(record)

    def pending(self, engine, *, after=0, limit=256):
        engine = Engine(engine)
        if type(limit) is not int or not 1 <= limit <= MAX_BATCH or type(after) is not int or after < 0:
            raise ContractError("Use a positive bounded limit and nonnegative ordinal cursor")
        with self._locked() as db:
            records = db.execute("""SELECT e.* FROM entries e JOIN evaluations v USING(ordinal)
                WHERE engine=? AND ordinal>? AND (state='pending' OR (state='interrupted-or-unknown' AND stopped=1))
                ORDER BY ordinal LIMIT ?""", (engine.value, after, limit))
            return tuple(self._row(record) for record in records)

    def prepare(self, engine, declarations, *, ownership):
        engine, ownership = Engine(engine), Ownership(ownership)
        bounded = []
        for declaration in declarations:
            if len(bounded) == MAX_BATCH:
                raise ContractError("Prepare accepts at most 4096 declarations")
            bounded.append(declaration)
        if not bounded:
            return ()
        keys = [item[0] for item in bounded]
        if len(set(keys)) != len(keys):
            raise ContractError("Duplicate entry/evaluator declaration")
        with self._locked() as db:
            db.execute("BEGIN IMMEDIATE")
            rows = []
            for entry_key, fingerprint in bounded:
                if not isinstance(fingerprint, Fingerprint):
                    raise ContractError("Missing typed execution fingerprint")
                key(entry_key)
                record = db.execute("SELECT * FROM entries WHERE ordinal=?", (int(entry_key[4:]),)).fetchone()
                if record is None:
                    raise ContractError("Unexpected entry identity")
                row = self._row(record)
                state = db.execute("SELECT * FROM evaluations WHERE ordinal=? AND engine=?", (row.ordinal, engine.value)).fetchone()
                if state is None or not (state["state"] == "pending" or (state["state"] == State.UNKNOWN.value and state["stopped"])):
                    raise StateError("Entry is not dispatchable; submitted jobs require reconciliation and failures require retry")
                if fingerprint.engine != engine or fingerprint.submitted_smiles != row.entry.smiles or fingerprint.target != self.target:
                    raise ContractError("Fingerprint input, target, or engine differs from frozen population")
                if fingerprint.resolved_options != getattr(self.settings, engine.value + "_options"):
                    raise ContractError("Fingerprint differs from frozen resolved settings")
                required = self.settings.required_output if engine is Engine.PSICHIC else Output.AFFINITY
                if fingerprint.output != required:
                    raise ContractError("Fingerprint output does not satisfy the run")
                rows.append(row)
            batch = BatchContext.for_rows(rows, seed=bounded[0][1].batch.seed)
            if any(f.batch != batch for _, f in bounded):
                raise ContractError("Fingerprint batch differs from actual submitted context")
            invocation = uuid.uuid4().hex
            tasks = []
            for row, (_, fingerprint) in zip(rows, bounded):
                task = Task(self.run_id, self.population_digest, row.key, engine, uuid.uuid4().hex, invocation, ownership, fingerprint)
                db.execute("INSERT INTO attempts(id,ordinal,engine,task) VALUES (?,?,?,?)",
                           (task.attempt_id, row.ordinal, engine.value, task.to_json()))
                db.execute("UPDATE evaluations SET state='submitted',expected=?,accepted=NULL,stopped=0 WHERE ordinal=? AND engine=?",
                           (task.attempt_id, row.ordinal, engine.value))
                tasks.append(task)
            db.execute("COMMIT")
            return tuple(tasks)

    def task(self, attempt_id):
        with self._locked() as db:
            row = db.execute("SELECT task FROM attempts WHERE id=?", (attempt_id,)).fetchone()
            if row is None:
                raise ContractError("Unexpected attempt identity")
            return Task.from_json(row[0])

    def diagnostics(self):
        with self._locked() as db:
            for row in db.execute("SELECT * FROM evaluations ORDER BY ordinal,engine"):
                yield Diagnostic(row_key(row["ordinal"]), Engine(row["engine"]), State(row["state"]), row["expected"],
                                 row["state"] == "pending" or (row["state"] == State.UNKNOWN.value and bool(row["stopped"])))

    def reconcile(self, attempt_id, resolution, *, evidence):
        resolution = Resolution(resolution)
        text(evidence, "reconciliation evidence")
        with self._locked() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM evaluations WHERE expected=?", (attempt_id,)).fetchone()
            if row is None or row["state"] not in (State.SUBMITTED.value, State.UNKNOWN.value):
                raise StateError("Only a current unresolved attempt can be reconciled")
            db.execute("INSERT INTO reconciliations(attempt,resolution,evidence) VALUES (?,?,?)", (attempt_id, resolution.value, evidence))
            state = State.SUBMITTED if resolution is Resolution.RUNNING else State.UNKNOWN
            db.execute("UPDATE evaluations SET state=?,stopped=? WHERE expected=?", (state.value, int(resolution is Resolution.STOPPED), attempt_id))
            db.execute("COMMIT")
            return state

    def retry(self, entry_key, engine):
        key(entry_key)
        engine = Engine(engine)
        with self._locked() as db:
            db.execute("BEGIN IMMEDIATE")
            if engine is Engine.PSICHIC and self.settings.pyramid and self._get(db, "selection") == "committed":
                raise StateError("PSICHIC selection is frozen; changing it requires a new run")
            row = db.execute("""SELECT v.state,r.distribution FROM evaluations v LEFT JOIN receipts r ON r.digest=v.accepted
                                 WHERE v.ordinal=? AND v.engine=?""", (int(entry_key[4:]), engine.value)).fetchone()
            if row is None or not (row["state"] == "failed" or row["distribution"] == "distribution-failure"):
                raise StateError("Explicit retry requires a recorded engine or distribution failure")
            db.execute("UPDATE evaluations SET state='pending',expected=NULL,accepted=NULL,stopped=0 WHERE ordinal=? AND engine=?",
                       (int(entry_key[4:]), engine.value))
            if self.settings.pyramid and engine is Engine.PSICHIC:
                self._set(db, "selection", "waiting")
            db.execute("COMMIT")

    def accept(self, transfer_directory):
        try:
            receipt, staging = stage_receipt(self.path, Path(transfer_directory))
        except FileNotFoundError:
            return Acceptance.INCOMPLETE
        task = receipt.task
        with self._locked() as db:
            db.execute("BEGIN IMMEDIATE")
            expected = db.execute("SELECT * FROM attempts WHERE id=?", (task.attempt_id,)).fetchone()
            if expected is None or Task.from_json(expected["task"]) != task:
                raise ContractError("Receipt does not match exact expected task identity and fingerprint")
            publish_evidence(self.path, receipt, staging)
            if expected["receipt"] is not None and expected["receipt"] != receipt.sha256:
                raise ContractError("Contradictory receipt for the same attempt; evidence retained")
            current = db.execute("SELECT * FROM evaluations WHERE ordinal=? AND engine=?", (expected["ordinal"], task.engine.value)).fetchone()
            if expected["receipt"] is None:
                db.execute("INSERT INTO receipts(digest,attempt,record) VALUES (?,?,?)", (receipt.sha256, task.attempt_id, receipt.record_json))
                db.execute("UPDATE attempts SET receipt=? WHERE id=?", (receipt.sha256, task.attempt_id))
            if current["expected"] != task.attempt_id:
                result = Acceptance.STALE
            elif current["accepted"] == receipt.sha256:
                result = Acceptance.IDENTICAL
            else:
                db.execute("UPDATE evaluations SET state=?,accepted=?,stopped=0 WHERE ordinal=? AND engine=?",
                           (loads(receipt.record_json)["status"], receipt.sha256, expected["ordinal"], task.engine.value))
                result = Acceptance.ACCEPTED
            db.execute("COMMIT")
            return result

    def check_coverage(self, identities):
        """Check an export against every currently expected task, not just counts."""
        with self._locked() as db:
            db.execute("CREATE TEMP TABLE coverage (entry_key TEXT, engine TEXT, attempt TEXT, PRIMARY KEY(entry_key,engine))")
            try:
                for entry_key, engine, attempt_id in identities:
                    db.execute("INSERT INTO coverage VALUES (?,?,?)", (entry_key, Engine(engine).value, attempt_id))
            except sqlite3.IntegrityError as error:
                raise ContractError("Duplicate exported entry/evaluator identity") from error
            expected = "SELECT printf('row_%012d',ordinal),engine,expected FROM evaluations WHERE expected IS NOT NULL"
            given = "SELECT entry_key,engine,attempt FROM coverage"
            if db.execute(f"SELECT * FROM ({expected} EXCEPT {given}) LIMIT 1").fetchone() or db.execute(f"SELECT * FROM ({given} EXCEPT {expected}) LIMIT 1").fetchone():
                raise ContractError("Missing or unexpected exported task identity")

    def select(self):
        with self._locked() as db:
            return self._select(db)

    def _select(self, db):
        db.execute("BEGIN IMMEDIATE")
        count = db.execute("SELECT count(*) FROM selection").fetchone()[0]
        if self._get(db, "selection") == "committed":
            db.execute("COMMIT")
            return Selection(SelectionState.COMMITTED, count, self.settings.shortlist_size or count, count)
        required = self.settings.shortlist_size
        unresolved = db.execute("SELECT count(*) FROM evaluations WHERE engine='psichic' AND state NOT IN ('succeeded','failed')").fetchone()[0]
        if unresolved:
            db.execute("COMMIT")
            return Selection(SelectionState.WAITING, 0, required, 0)
        condition = " AND r.distribution='valid'" if self.settings.ensemble is EnsembleMethod.CONCENTRATION else ""
        eligible = "SELECT v.ordinal,r.p FROM evaluations v JOIN receipts r ON r.digest=v.accepted WHERE v.engine='psichic' AND r.status='succeeded'" + condition
        available = db.execute(f"SELECT count(*) FROM ({eligible})").fetchone()[0]
        if available < required:
            self._set(db, "selection", "shortage")
            db.execute("COMMIT")
            return Selection(SelectionState.SHORTAGE, available, required, 0)
        db.execute(f"INSERT INTO selection SELECT ordinal FROM ({eligible}) ORDER BY p DESC,ordinal LIMIT ?", (required,))
        db.execute("INSERT INTO evaluations(ordinal,engine,state) SELECT ordinal,'boltz','pending' FROM selection")
        self._set(db, "selection", "committed")
        db.execute("COMMIT")
        return Selection(SelectionState.COMMITTED, available, required, required)

    def finalize(self, *, top_n=20):
        from .reports import finalize
        return finalize(self, top_n=top_n)

    def results(self, version=None):
        from .reports import results
        yield from results(self, version)

    def latest(self):
        from .reports import latest
        return latest(self)
