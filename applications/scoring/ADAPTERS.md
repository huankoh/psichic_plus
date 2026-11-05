# Connect an executor to a saved scoring run

Import `ScoringRun` from `applications.scoring`.
Use the public API to create tasks and accept evidence.
Do not write SQLite, accepted pointers, blobs, or report versions from an adapter.
The core remains separate from engine execution and scheduling.
`applications.scoring.executor` supplies the local execution boundary and `python -m applications.scoring` supplies the durable CLI.
A distributed scheduler adapter remains a separate unit.

## Freeze the population

Stream native candidates as `Entry` objects into `ScoringRun.create`.
Keep every generator route column actually present in metadata.
For a screener, keep collection, ligand, scenario, replica zero, and occurrence in the source locator or metadata.
Preserve exact IDs and SMILES without deduplication.
Reject multi-replica inputs until an aggregation contract exists.
Retain the native source file separately in the application's saved-run artifacts.

Resolve one target sequence and its asset digests before creation.
Resolve presets and engine options before constructing `Settings`.
Use PSICHIC-only settings without Boltz options when no Boltz stage was requested.
Validate the actual storage deployment before advertising resume support.

## Prepare and dispatch bounded tasks

Obtain a tuple with `rows = run.pending(Engine.PSICHIC, limit=chunk_size)`.
The maximum is 4096 entries per preparation transaction.
Use a smaller chunk when native batching or the scheduler requires it.
Create the exact native input for each row and hash its bytes.
Derive all fingerprint declarations from the real runtime and loaded assets, using the [format reference](FORMAT.md).
Use `BatchContext.for_rows(rows, seed=resolved_seed)` to bind the actual ordered invocation context.

With those verified declarations, the coordinator sequence is:

```python
from applications.scoring import Engine, Ownership, ScoringRun, SelectionState, Waiting

run = ScoringRun.open(run_directory)
tasks = run.prepare(Engine.PSICHIC, declarations, ownership=Ownership.LOCAL)
serialized_tasks = [task.to_json() for task in tasks]
```

Send `serialized_tasks` through the existing local process or scheduler transport.
Each task already has a durable expected attempt and invocation identity.
The executor must use the exact prepared batch and must retain the same-pass raw PSICHIC output.
Use `Ownership.EXTERNAL` for native scheduler jobs.
Record the mapping from scheduler workunit and job identities to these task IDs in adapter-owned evidence.
Do not treat a returned task as authorization to swap in different fingerprints, chunks, or seeds.

Workers build `Receipt` objects and call `receipt.publish(worker_directory)` after writing the required artifacts.
They can send serialized JSON through Python 3.10 AFVS workers without importing this coordinator package there.
Preserve `schema_version=1` and all task identities without translating them into ligand names.
The coordinator then ingests each transferred directory:

```python
for transfer_directory in available_transfers:
    acceptance = run.accept(transfer_directory)

selection = run.select()
if selection.state is SelectionState.COMMITTED:
    boltz_rows = run.pending(Engine.BOLTZ, limit=chunk_size)

outcome = run.finalize()
if isinstance(outcome, Waiting):
    diagnostics = run.diagnostics()
else:
    report_directory = outcome.directory
    for row in run.results(outcome.version):
        consume_report_row(row)
```

These snippets show coordinator calls at adapter boundaries.
`declarations`, `available_transfers`, and the real executor remain the adapter's inputs.
The controlled executable examples in `tests/conftest.py` show complete fingerprint, raw-record, artifact, and receipt construction without claiming inference.

For Boltz, retain the exact rendered input, native affinity JSON, structure, logs, execution evidence, and full provenance.
The existing `parse_affinity` parser is reused by receipt acceptance.
For PSICHIC, preserve the full raw row and distribution diagnostics from the same invocation as affinity.
Pass native alpha and beta through `encode_native_parameter` when writing raw JSON so non-finite floats retain their original type and diagnostics.
An affinity-only legacy saved result cannot satisfy distribution-required tasks.
A generator search score is not an accepted inference receipt.

## Recover without duplicating live jobs

Read `diagnostics()` to identify expected attempts.
Close the streaming iterator before calling another coordinator command.
Retrieve any task declaration with `task(attempt_id)`.
Check local child ownership or ask the existing scheduler for its job state.
Open alone does not perform that reconciliation.

Record `Resolution.RUNNING` while the job remains live.
Record `Resolution.UNKNOWN` when its liveness cannot be established.
Neither makes it dispatchable.
After the driver reaps an owned process or confirms that the external job has stopped, record `Resolution.STOPPED` with the observed evidence.
`pending` then exposes that row for a fresh attempt.
An already validated output from the old task can still be accepted before a replacement is prepared.
After replacement, the old receipt is retained as superseded.

Use `retry(entry_key, engine)` only on an explicit recorded engine or distribution failure.
Preserve the old attempt's outputs.
PSICHIC retries after an immutable pyramid shortlist require a new run.
Boltz retries use the same shortlist and can change all final scores through cohort renormalization.

## Keep native AFVS identity lossless

Use the twelve-digit internal row keys as synthetic native ligand names.
Package a digest-bound map to the original collection, ligand, scenario, and occurrence.
Validate the packaged task tuples with `check_coverage` when exporting a complete expected stage.
For a subset workunit, compare its exact bounded task tuples against the prepared tasks before dispatch.
A duplicate-for-missing result must fail even when counts agree.
Do not equate a native worker exit zero or overview file with successful predictions.
The coordinator must receive and validate the complete required artifact set.

AFVS retains scheduling and resource allocation ownership.
The next adapter unit must qualify the synthetic named-ligand path through native preparation and execution, the Boltz registration and path protections, and actual transfer completeness.
Retire task-owned PSICHIC processes before reusing the same allocation for Boltz, or use separately allocated workunits.
The core makes no GPU, cluster-scheduler, remote-storage, or scientific-quality claim.

## Reuse the local runner from the next application unit

Import `resolve` from `applications.scoring.config`, then pass its `LocalPlan` to `executor.create_run` and `executor.execute`.
The CLI parser defines the current named configuration inputs.
An application that already owns a frozen `ScoringRun` must retain a compatible `local.json` and input assets before calling the local executor.
Do not reinterpret existing generator search scores as accepted evidence.

`executor.prepare_invocation` is the bounded rendering and task preparation boundary.
It creates the exact YAML before committing tasks and maps each twelve-digit core entry key to a fresh eight-digit native Candidate key within that invocation.
Legacy Candidate keys are unchanged.
The mapping is retained in receipts and checked before execution.
Receipt recovery rechecks the same ordered task, native row, key, and filename mapping before publishing an observation.

`worker.publish` can salvage completed observations using the recorded actual process exit status.
It requires a loaded runtime record and the full required native artifacts.
A missing structure never becomes Boltz success.
A nonzero batch exit without complete per-row artifacts remains unknown for explicit resume.
A zero exit with a missing or invalid result becomes a recorded output failure.

Each invocation has a separate lifecycle owner and native process group.
The owner registers its identity, starts the model behind a pipe gate, records the child identity, then releases the gate.
The child watches the pipe and stops its own group when the owner disappears.
The coordinator records the owner mapping before releasing its own registration gate.
A stable `driver.lock` serializes a complete local invocation sequence alongside the core command lock.
The next AFVS adapter must keep scheduler ownership and use `Ownership.EXTERNAL` instead of this local driver.

Runtime fingerprints contain digests of complete named source and weight inventories.
The full inventories, absolute loaded paths, file hashes, process commands, and actual exit statuses accompany receipts.
Inventory names normalize checkout and installed-package locations, while file provenance retains their actual paths.
Import-time generated Python source is hashed from real files in preflight and matched by module identity and exact bytes in each child, retaining both absolute locations.
It is never treated as reusable merely because the generating package version matches.
Normalization alone does not authorize reuse between different runtimes or call profiles.
Preflight hashes once per command; immutable file identity checks and child import verification guard each batch.
Changed source or assets on resume requires a new run.

The local CLI has no generator `--source-run` option or distributed adapter flags.
Those are reserved for the next application units.
