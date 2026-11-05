# Durable scoring core

`applications.scoring` owns a saved population and its accepted observations.
The core does not invoke models or submit scheduler jobs.
The optional local executor invokes the existing Predictor and isolated pinned Boltz runtime.
Use [the CLI guide](CLI.md) for installation, validation, ranking, resume, and explicit failure retry.
The coordinator requires Python 3.11 or newer and SQLite with JSON, generated columns, and window functions.
This unit is verified on Python 3.12.15 and SQLite 3.53.4.
The original `applications.scorer.boltz2` interface and its `rank-v1` default remain independent.

Create a run with `ScoringRun.create(path, entries, target=target, settings=settings)`.
`entries` is an iterable of `Entry(id, smiles, source, metadata_json)`.
Creation consumes the iterable once and inserts each occurrence into SQLite.
The ordinal key and population digest distinguish repeated IDs and structures.
The digest includes the target, exact SMILES, original ID, source locator, metadata, and row order.
An existing directory is rejected.
A failed create remains evidence and `open` rejects it if its manifest transaction never committed.
Create a fresh path after diagnosing that failure.

Use `pending(engine, limit=256)` to obtain a bounded batch.
Build a `BatchContext.for_rows(rows, seed=42)` and a complete `Fingerprint` for each row.
Call `prepare(engine, [(row.key, fingerprint), ...], ownership=Ownership.LOCAL)` before dispatch.
Preparation commits the expected attempt IDs and returns versioned `Task` declarations.
The driver must dispatch exactly that batch context.
A changed chunk or seed requires a new declaration and attempt.
The core checks declarations and equality; it cannot establish that a worker actually loaded the declared source or weights.
The real executor owns that evidence.

Workers publish a `Receipt` and its required artifacts in their own transfer directory.
`Receipt.publish(directory)` validates hashes and uses an atomic no-clobber publication of `receipt.json`.
The coordinator calls `accept(directory)` to copy and validate evidence before accepting it.
A missing transfer returns `Acceptance.INCOMPLETE` and leaves the attempt unresolved.
An explicit engine failure is a terminal receipt with retained execution evidence and logs.
An identical receipt is idempotent.
A conflicting receipt for the same attempt is rejected and retained.
A superseded receipt is retained without replacing the accepted observation.

`open` never redispatches a submitted attempt.
Use `diagnostics()` and `task(attempt_id)` to reconcile jobs with their owner.
`reconcile(attempt_id, Resolution.STOPPED, evidence=...)` records the driver's observation that the old process or scheduler job has stopped.
Only then can `pending` return that unknown row for a fresh attempt.
`Resolution.UNKNOWN` alone does not authorize dispatch.
`retry(entry_key, engine)` explicitly supersedes an accepted engine or distribution failure.
Successful predictions remain accepted.
After a pyramid selection commits, PSICHIC retries require a new run because the shortlist is frozen.

`select()` waits for every expected PSICHIC outcome before selecting global K.
A shortage produces no Boltz tasks.
Concentration selection requires both affinity and a valid distribution.
Other selection modes require affinity only.
`check_coverage(identities)` checks exact exported entry, evaluator, and attempt tuples, including duplicate-for-missing errors.

## Storage boundary

The supported contract is one host and a local filesystem with cooperative `flock`, SQLite, hard links, atomic rename, and directory fsync.
A stable `coordinator.lock` inode serializes coordinator commands.
Workers never write the database or accepted references.
SQLite uses DELETE journaling and FULL synchronous commits.
Files are flushed and fsynced before publication, and containing directories are fsynced.
This protocol is tested for process interruption on local storage.
It is not a power-loss, cross-host, network-filesystem, or object-store guarantee.

`state.sqlite3` owns manifest rows, attempts, reconciliation observations, accepted receipt pointers, and selection.
`receipts/<digest>.json` and `blobs/<sha256>` are immutable coordinator-owned evidence.
Artifacts with equal content share a blob, so invocation-level evidence and checkpoints need not be copied per row.
The receipt still binds every role, path, size, and digest to the exact task.
`staging/` retains incomplete imports and publication remnants as evidence.
No automatic cleanup or garbage collection runs.

See [the format reference](FORMAT.md) and [the executor and AFVS guide](ADAPTERS.md).
Run the controlled CPU tests with `python -m pytest applications/scoring/tests` from the repository root.
The synthetic receipts are explicitly labeled fixtures and do not claim model inference.

## Finalize and read reports

Call `finalize(top_n=20)` after ingesting available receipts.
It returns `Waiting(pending, submitted, unknown)` while any expected outcome remains unresolved.
No final ranking is published in that state.
A terminal population returns `Report(version, report_id, status, directory, accepted_receipt_digest)`.
The status is `complete`, `partial`, or `shortage`.
Failures remain visible without numeric penalties.

`results(version=None)` streams stored report records in manifest order.
`latest()` returns the current published report, or `None` before the first publication.
Each record has a separate `position` for human display and `final_rank` for average-tie ranking.
Raw outcomes, original source metadata, expected attempt IDs, accepted receipts, and content-addressed artifact links accompany every row.
`results.json`, `results.csv`, and the bounded `summary.md` come from that same stored snapshot.
JSON never emits NaN.

The new `Settings` defaults ensemble modes to `concentration-v1`.
Pass `ensemble=EnsembleMethod.RANK` to request `rank-v1` with finite nonnegative weights whose sum is positive and finite.
PSICHIC-only needs no Boltz options.
The four standalone `Plan` modes and defaults remain unchanged.

Ranking joins accepted outcomes on disk and uses SQLite window functions and file-backed temporary sorting.
Ensemble normalization and component ranks use only the selected cohort successful on both engines.
Concentration also requires a valid PSICHIC distribution.
Its Boltz affinity is negated before normalization.
Scalar numerical rules are shared with the standalone scorer.
Constant series normalize to 0.5, and singleton or tied cohorts retain their calculated ranks with warnings.
Report rows remain in manifest order, with ties displayed in manifest order.

`reports/v00000001/` and subsequent versions are immutable.
Finalization commits a database snapshot before writing a temporary report directory.
It fsyncs the report files and directory, renames the directory, and then atomically replaces `latest.json`.
An interrupted publication can resume from the stored snapshot without changing the previous report.
Temporary directories remain as evidence.
No-change finalization returns the existing version.
An explicit retry that changes the successful cohort publishes a new version because every normalized score or rank can change.
