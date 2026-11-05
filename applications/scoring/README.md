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

## Installed runtime source inventory

Local preflight freezes source before inference, including files from every installed distribution visible to that engine's Python.
This matches the environment contract, which already records all installed package versions.
Coverage does not depend on which packages preflight happened to import.
The inventory reads distribution file lists without importing their packages.
It includes existing recorded `.py`, `.so`, `.pyd`, `.dylib`, `.so.*`, `.jinja`, `.smi`, and `.jsonl` files, including recorded package test helpers.
It also hashes `RECORD`, `METADATA`, `PKG-INFO`, `SOURCES.txt`, and `direct_url.json` when listed.
Explicit application and engine roots retain their existing source and resource coverage.
The Python executable is hashed too.

Each resolved file path has one record with its exact preflight hash and filesystem stamp.
Overlapping roots, distribution records, and import aliases share that path record.
Sorted logical names and hashes determine source identity, independent of metadata enumeration order and imports performed later.
Namespace packages need no package `__file__` because their installed children have file records.
An editable or external source file is covered only if its path is recorded by installation metadata, belongs to an explicit root, or was captured by the existing preflight import contract.
A distribution without a file list adds no implicit directory authorization.
Python's metadata API can omit missing listed files.
Those paths receive no authorization and fail the loaded-source check if created and imported after freezing.

Workers still require every loaded non-stdlib module to match an exact frozen path and hash.
A file added after freezing fails even inside an inventoried package directory.
A changed recorded file, unknown temporary generated module, or unrecorded late external module also fails.
Preflight-generated temporary modules retain their module identity and exact-byte check across processes.
PSICHIC's three expected PyG modules retain their separately predicted exact-byte contract, including GCN fusion.
Boltz 2.2.1 still requires its installed Python source to match its RECORD before preflight and execution.

Installed optional packages are included even when the current prediction does not use them.
Changing their inventoried source requires a new run.
This avoids inferring which optional dependencies inference might load, at the cost of hashing the broader installed runtime during preflight.
The contract does not attest arbitrary in-memory monkeypatches, every library loaded outside Python module specifications, or unlisted non-source data files.
Validation remains offline and does not construct prediction models, deserialize checkpoints, initialize CUDA, or create a run directory.

## Runtime native generation

Triton host extensions have a separate `native_generation` policy.
They do not exist at preflight and are never added to the installed source inventory after inference.
The Linux Triton 3.3.1 policy freezes all installed Triton resources, including C sources, headers, compiler binaries, and bitcode.
It also records GCC, its programs and resources, the compiler sysroot, Python and system headers, link inputs, loader configuration, and CUDA driver library bytes.
The policy and its input hashes contribute to prediction fingerprints and reopened-run identity.
Runtime signatures and temporary output paths belong to invocation evidence instead.

Validation inspects files and runs read-only compiler queries, including preprocessing an empty input to discover include paths.
It creates no native binary, Triton cache, or run directory.
It does not construct a prediction model, deserialize a checkpoint, or initialize CUDA.
The driver link library must be available even in a CPU-only qualification container.
Native compilation is an execution preparation step inside the dedicated Boltz worker.

Each invocation exclusively creates its own empty Triton cache.
A scoped Python profile observer watches the unmodified Triton generator, compiler, and cache functions.
It matches the registry's `nvidia` alias and the canonical import by code equality and the same frozen file path.
It accepts the utility source only from `CudaUtils` and launchers only from `CudaLauncher` with a frozen JIT function source and a supported argument signature.
The observer regenerates the expected C source and checks the exact compiler command, dependency bytes, and build options.
Before cache publication and loading, it independently rebuilds the actual temporary source and requires exact binary equality.
The rebuild uses the original source path because `driver.c` embeds `__FILE__` in diagnostics.
No numerical model code or upstream function is replaced.
The observer is removed when the scope exits.

A cache hit needs a matching build observed in the same worker.
An existing binary or a saved build receipt cannot authorize a new worker's cache.
Corruption, an unknown generated module, a different path, changed native inputs, or an unsupported compiler control prevents output acceptance.
Every observed variant retains its signature, kernel source hash, command, generated C, dependencies, linker trace, rebuild, actual binary, and loaded CUDA driver hash.
Receipts also retain the invocation's native cache files and final loaded-module hashes.
The installed-source guard, PyG generation contract, and restricted ESM loading remain in force.

This contract establishes file provenance for Triton's host CPython extensions under the recorded toolchain.
The source of truth is the frozen input policy plus a live observed build and independent byte comparison.
It is not a sandbox or an attestation of arbitrary in-memory changes, hostile concurrent writers, the operating-system kernel, GPU firmware, or scientific accuracy.
Retained Triton device-kernel cache files are output evidence, not independently rebuilt device-kernel attestations.
Other compiler families, Triton versions, custom cache managers, and custom include or compiler search controls fail closed.
Moving installed toolchains or changing driver libraries can require a new run.
Changing only runtime temporary paths does not change prediction identity.

The real CPU qualification probe is `applications/scoring/tests/native_build_probe.py`.
It runs in the pinned Linux environment with a task-owned driver library copy and no GPU devices.
Running it twice with distinct output directories and one profile checks fresh-process behavior without reusing compilation receipts.
The ordinary local test suite skips this lane when Triton is unavailable.
