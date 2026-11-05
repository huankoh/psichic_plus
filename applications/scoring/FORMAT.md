# Scoring storage and receipt format v1

## Ownership and durability

The coordinator is the only writer of `state.sqlite3`, `receipts/`, `blobs/`, `reports/`, and `latest.json`.
Worker and transport directories are outside that state.
A worker publishes `receipt.json` only after its artifact bytes are complete.
The coordinator copies those bytes into its own unique `staging/` directory before validation.
Every relative artifact path component is opened without following symlinks.
Traversal, absolute paths, duplicate artifact roles, duplicate paths, nonregular files, missing required roles, size mismatches, and digest mismatches prevent acceptance.
The coordinator does not rename, delete, or modify worker artifacts.

SQLite stores the accepted receipt pointer and the receipt's canonical raw record in one transaction.
Generated SQLite columns expose numeric fields without introducing a separately writable outcome representation.
A successful import fsyncs artifact bytes, publishes content-addressed links, fsyncs their directories, and commits acceptance with SQLite `synchronous=FULL` and DELETE journaling.
A crash before acceptance can leave unreferenced evidence, which a repeated import can accept.
A crash after acceptance preserves the accepted pointer under the tested process-interruption conditions.
Neither outcome requires rewriting accepted evidence.

`coordinator.lock` is created once and never replaced or removed.
All coordinator operations acquire its stable-inode exclusive `flock` nonblockingly.
Contention raises `RunBusy`.
The run records its creating hostname and rejects opening on another hostname.
The operator must supply a same-host local filesystem that implements the stated primitives.
Hostname checks do not qualify a shared filesystem.
Power loss, cross-host coordination, NFS, S3, scheduler transport, and adversarial modification of coordinator-owned files are outside this qualification.

Streaming read iterators hold the coordinator lock until exhausted or closed.
Consume or close `population()`, `diagnostics()`, and `results()` before another command on that run.
`pending(engine, after=ordinal, limit=n)` returns at most 4096 rows and releases its lock before dispatch work.
An accepted or unresolved submitted attempt is never returned by `pending`.

## Manifest and settings

`Entry` retains the original string ID, exact submitted SMILES string, source locator, and JSON metadata object.
Creation assigns `row_000000000001` and subsequent twelve-digit keys in manifest order.
Keys are bounded and scoped by the population digest, never by user IDs or structures.
Repeated occurrences remain separate rows.
The digest hashes canonical schema and target JSON followed by each canonical manifest row, each terminated by a newline.
The target contains an identity, selected normalized sequence, and nonempty named asset digests.
`metadata_json` and engine option JSON are canonical immutable strings at the Python boundary.

`Settings` supports `psichic-only`, `boltz-only`, `full-ensemble`, `pyramid-ensemble`, and `pyramid-refinement`.
It resolves its Boltz and ensemble semantics through the existing four-mode `Plan`.
The new ensemble default is `concentration-v1`.
The standalone default remains `rank-v1`.
Only one replica is supported.
An unused engine cannot have options, and a required engine needs a nonempty resolved options object.
A settings change requires a new run.

## Task and fingerprint

`Task.to_json()` and `Task.from_json()` define the versioned worker task contract.
The exact top-level fields are `schema_version`, `run_id`, `population_digest`, `entry_key`, `engine`, `attempt_id`, `invocation_id`, `ownership`, `fingerprint`, and `fingerprint_sha256`.
UUID identities use 32 lowercase hex characters.
Digests use 64 lowercase SHA256 hex characters.
Missing, additional, or duplicate JSON fields at closed contract boundaries are rejected.

Every fingerprint requires the following fields.

| Field | Contract |
|---|---|
| `schema_version` | Integer 1. |
| `engine`, `engine_version` | Typed engine and declared version; Boltz requires 2.2.1. |
| `submitted_smiles` | Exact manifest input. |
| `target` | Exact run target identity, normalized sequence, and named asset digests. |
| `native_input_sha256` | Digest of the exact retained native input artifact, including rendered YAML for Boltz. |
| `weights` | Nonempty tuple of unique named SHA256 declarations for all loaded model assets. |
| `loaded_source` | Nonempty tuple of unique named SHA256 declarations for the loaded implementation. |
| `environment_json` | Object with exactly `python`, nonempty package-version mapping `packages`, and `runtime`. |
| `device`, `precision` | Nonempty resolved device and precision declarations. |
| `parser_contract`, `call_profile` | Nonempty versioned parser and invocation-profile identifiers. |
| `resolved_options` | Canonical nonempty options object equal to the run's frozen engine options. |
| `batch` | Ordered row keys, digest of complete submitted manifest rows, actual batch size, explicit integer seed, and replica zero. |
| `output` | `affinity` or `distribution`; must equal the run's required output. |
| `msa_mode`, `msa_assets` | `single` and an empty tuple for this integration contract. |

The SHA256 covers every serialized fingerprint field.
The first prepared batch freezes each engine's execution profile for that run.
Later batches cannot change its weights, loaded source, environment, device, precision, version, parser, call profile, options, output, target, or MSA declarations.
Per-row input and batch context remain explicit in each task.
Changing a chunk or seed changes its fingerprint and requires a new expected attempt.
The invocation and attempt IDs identify a stochastic observation separately from the comparable declared payload.
The core offers no automatic structure-based or cross-row reuse.

A complete declaration is necessary evidence, not proof that a remote worker actually loaded those bytes.
The real executor must derive and verify the complete source, weights, target, environment, rendered input, and resolved settings.
The core enforces completeness and exact equality against that committed expectation.
It does not promote arbitrary user-supplied digests into verified external provenance.
Legacy search scores and affinity-only results cannot satisfy a distribution task.

## Receipt and required artifacts

A receipt has exactly `schema_version`, `task`, `record`, and `artifacts`.
`task` is the complete task declaration, including its fingerprint digest.
Each artifact has exactly `role`, normalized relative `path`, byte `size`, and `sha256`.
Artifact roles and paths are unique within the receipt.
Only stdout and stderr may be empty.
Additional named artifacts are retained, but cannot replace required roles.

| Outcome | Required artifact roles |
|---|---|
| Every outcome | `native_input`, `provenance`, `execution`, `stdout`, `stderr`. |
| PSICHIC success | All common roles and `raw_psichic`. |
| Boltz success | All common roles, `native_affinity`, and `structure`. |
| Explicit failure | All common roles, with a typed failure record and execution evidence. |

`provenance` contains the exact fingerprint object.
`execution` contains exactly `schema_version`, `invocation_id`, nonempty command argument list `command`, and observed integer `returncode`.
A failed batch may still salvage individual successes if their native outputs and required artifacts validate.
A zero exit code alone never establishes prediction success.
Missing execution evidence is an incomplete transfer or invalid receipt, not an inferred engine failure.

A PSICHIC success record contains `status="succeeded"`, `raw_affinity`, `adaptive_score`, `submitted_smiles`, `normalized_smiles`, and `distribution`.
Affinity is finite and adaptive score equals its negation.
The distribution is the existing tagged union of `not-requested`, `valid`, and `distribution-failure`.
Valid alpha and beta are positive finite values, concentration is their exact sum, and affinity agrees with `12 * alpha / concentration` at the existing `1e-6` relative and absolute validation tolerance.
A distribution failure preserves the affinity and original diagnostic values.
Its `raw_alpha` and `raw_beta` are Python `repr` strings of the native values.
Missing native values use `None`.
Non-finite float parameters use the tagged object `{"nonfinite_float":"nan"}`, `{"nonfinite_float":"inf"}`, or `{"nonfinite_float":"-inf"}` because contract JSON prohibits non-finite numbers.
Use `encode_native_parameter` for `reg_alpha` and `reg_beta` when retaining raw native rows.
The validator decodes this tag before checking the original diagnostic representation.
Ordinary strings remain strings and missing values remain null, preserving their distinct `repr` diagnostics.
A distribution-required task rejects `not-requested` evidence.

`raw_psichic` has exactly `schema_version`, the complete `task`, and `native`.
`native` retains the full per-row same-pass output, including `predicted_binding_affinity`, `reg_alpha`, `reg_beta`, and any other native columns.
Validation compares affinity and distribution against this evidence.
Adapters must not combine an old affinity with a new distribution.

A Boltz success record contains `status="succeeded"`, `affinity_pred_value`, `affinity_probability_binary`, the complete `native` JSON object, and `submitted_smiles`.
The existing native affinity parser validates the retained `native_affinity` file.
The entire parsed native object and both numeric fields must equal the receipt.
Binder probability is finite and in [0,1].
The structure is retained as a nonempty digest-validated file; the core does not assess structural quality.

An explicit failure record contains exactly `status="failed"`, nonempty `stage`, and nonempty `reason`.
It has no invented numeric score.
A transfer with missing files returns `Acceptance.INCOMPLETE` and leaves state unresolved.
Malformed documents, contradictory declarations, unsafe paths, and tampering raise validation errors while leaving the expected attempt unaccepted.
Partially written receipts and temporary files are not terminal predictions.

Common invocation artifacts can use the same content digest across receipts.
The coordinator retains one blob per digest while validating every declared transfer path.
The engine adapter may retain shared weight or invocation evidence once and refer to it from all affected tasks.
The fingerprint must still describe each task's exact input and actual submitted batch.

## Attempts, selection, and reports

Each expected entry and engine has one of `pending`, `submitted`, `succeeded`, `failed`, and `interrupted-or-unknown`.
Preparation commits the attempt before a task leaves the coordinator.
Open and ordinary finalization never turn missing outcomes into failures or redispatch submitted jobs.
`reconcile` records `still-running`, `unresolved`, or `confirmed-stopped` with a nonempty evidence description.
Only confirmed stopped attempts become dispatchable again.
The driver owns process reaping or scheduler reconciliation; the core does not infer liveness from a restarted coordinator.
Explicit retry creates a fresh attempt and preserves earlier evidence.
An old attempt's receipt is retained as superseded.
A conflicting same-attempt receipt is retained but rejected.
Different numeric values across distinct attempts are permitted.

A pyramid selection commits exactly K rows after every expected PSICHIC outcome is terminal.
A shortage has zero Boltz tasks.
Concentration eligibility requires valid affinity and distribution; refinement and rank selection require affinity.
Boundary ties use manifest order, and Boltz failures never cause backfill.
PSICHIC retry after selection requires a new run.
Boltz failure retry retains the original selected rows and publishes a new report when terminal.

A report snapshot contains every manifest row and uses accepted raw records as its source.
Ensemble calculations use exactly the selected common-success cohort.
SQLite joins and window functions perform global sorting and average-tie ranking.
The scalar formulas are shared with the in-memory scorer.
JSON and CSV stream snapshot rows in manifest order.
`position` distinguishes human display order from the potentially fractional `final_rank`.
A bounded summary shows at most 100 rows.

Reports carry schema, run, population, report, snapshot, and accepted-receipt identities.
They include completion counts, shortage details, method, source reference, direction, meaning, raw outcomes, distribution states, reasons, attempts, and artifact links.
`files.json` records the byte sizes and hashes of the JSON, CSV, and summary.
Publication orders the durable database snapshot, fsynced temporary files, atomic report-directory rename, published database marker, and atomic latest pointer.
Reopening can finish publication without replacing any previous version.
Finalization first publishes committed interrupted snapshots, even if current observations changed or are still pending.
The latest pointer always names the highest published version.
Unfinished populations return diagnostics rather than a final global ranking.
