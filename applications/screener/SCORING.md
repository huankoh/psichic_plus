# Final scoring in native AdaptiveFlow workunits

`psichic-screener` freezes a native selection, prepares AdaptiveFlow workunits, accepts qualified receipts, and produces the shared scoring reports.
AdaptiveFlow retains scheduler submission and workunit ownership.
The equivalent module command is `python -m applications.screener`.
The original affinity-only adapter remains available as described in [README.md](README.md).

## Install and configure the deployment

Install both wheels in a compatible coordinator environment with Python 3.11 or newer.
Install the standalone adapter wheel in each native worker's Python 3.10 or newer environment.
The native worker needs AdaptiveFlow's dependencies; it calls the coordinator's explicit Python executable for inference supervision.
The qualified PSICHIC environment and the isolated Boltz 2.2.1 environment retain their existing dependencies and compiled extensions.
See [shared runtime setup](../scoring/CLI.md) for cached assets, runtime qualification, and supported Boltz builds.

```sh
python -m pip wheel --no-deps --no-build-isolation \
  ./applications ./applications/screener --wheel-dir /task/wheels
python -m pip install --no-deps /task/wheels/*.whl
/opt/afvs/bin/python -m pip install --no-deps \
  /task/wheels/psichic_adaptiveflow-0.1.0-py3-none-any.whl
```

Use a pinned AFVS checkout at `df74e005e456e94654fa93a40d748de399450b81` with its real `tools/templates/all.ctrl`, library paths, selection file, scenario, and scheduler templates configured.
Preparation reads that checkout, validates the pinned preparation interfaces, and patches a run-owned copy.
It accepts unpatched sources or the exact current adapter patch.
It does not modify the source checkout.
Site scheduler templates must select the native worker Python and the actual GPU allocation when a GPU is needed.
A CPU Bash template does not allocate GPUs.

This deployment requires one coordinator writer on a qualified same-host filesystem with working POSIX locks.
Native workers, coordinator, model environments, and artifacts must see the same absolute paths on that host.
Containers must retain the same explicit hostname and mount paths on resume.
Cross-host SQLite, S3 transfer, live cluster scheduler behavior, and GPU inference for this integration require separate deployment qualification.

```sh
export PSICHIC_MODEL_DIR=/data/weights/PSICHIC_plus
export PSICHIC_ESM_CHECKPOINT=/data/torch/hub/checkpoints/esm2_t33_650M_UR50D.pt
export PSICHIC_CACHE=/task/cache/psichic
export BOLTZ_PYTHON=/opt/boltz/bin/python
export BOLTZ_CACHE=/data/cache/boltz
export AWS_EC2_METADATA_DISABLED=true
```

The last setting disables AWS instance-metadata credential lookup for offline sharedfs workers.

## Prepare a screen

Select one scenario with one replica and per-ligand `.smi` archives.
Full-library todo files, named CSV selections, and sparse prescreen `.listing` selections are supported.
The exact source order, repeated collection/ligand pairs, duplicate names across collections, original strings, and metadata are retained.
Sparse selections retain the original full collection from `.listing`, its index, and its source archive identity.
Atom and dynamic tranche filters must be disabled for this SMILES integration.
Use the native selector to produce the intended todo or named selection first.
Missing identities and malformed archives fail before native preparation.
Multiple replicas are rejected because this integration has no approved scientific aggregation rule for them.

```sh
psichic-screener prepare \
  --afvs /data/configured-afvs --scenario target \
  --afvs-python /opt/afvs/bin/python \
  --fasta /data/target.fasta --msa-mode single \
  --mode pyramid-refinement --shortlist-size 100 \
  --psichic-device cpu --batch-size 32 --chunk-size 64 \
  --boltz-timeout 7200 --output /task/screens/refinement-001 \
  --validate-only -- --accelerator cpu
```

Validation imports and fingerprints the configured runtimes, checks the native source and exact archive identities, and constructs no model or run output.
Remove `--validate-only` to prepare the first stage.
Set native `program_timeout` long enough for the complete qualified invocation, including initialization and receipt publication.
The Boltz executor timeout applies inside that native deadline.
`--stage-size` bounds native preparation to at most 4096 entries per stage; `--chunk-size` bounds each model invocation.
A generated collection contains one whole invocation and stays in one native subjob.
The adapter uses one docking worker per native allocation to avoid simultaneous invocations on the same allocation.

The shared modes and scientific defaults are unchanged.
`psichic-only` and `boltz-only` score the entire population with one engine.
`full-ensemble` scores every entry with both engines.
Pyramid modes finish all PSICHIC observations before freezing one global K, resolving cutoff ties in manifest order.
A shortage starts zero Boltz workunits; failed Boltz entries are never backfilled.
`pyramid-refinement` defaults to binder probability.
Ensembles default to `concentration-v1`; `--ensemble rank-v1` selects the existing rank formula.
Affinity and distribution come from the same PSICHIC observation, and a distribution failure retains any valid affinity.
Normal upstream sampling remains the default; `--preset quick-check` is an explicit reduced-sampling diagnostic.

## Submit and resume

Preparation returns `waiting` with the newly created `prepared_stage`, native submission directory, and explicit command.
For example, run the printed command from that stage's tools directory using its actual final workunit number.

```sh
cd /task/screens/refinement-001/screening/stages/PSICHIC_STAGE/tools
/opt/afvs/bin/python afvs_submit_jobs.py 1 2
```

Use the configured native scheduler's normal execution procedure.
For the Bash backend, submission creates scripts; run each generated `workflow/workunits/N/run.bash` explicitly with Bash.
Inspect those templates and their environment before execution.
PSICHIC and Boltz are prepared as separate stages and must run in separate allocated workunits.
Retire each workunit's task-owned children before reusing an allocation.
GPU visibility alone is not authorization to use that GPU.

```sh
psichic-screener resume --run /task/screens/refinement-001
psichic-screener status --run /task/screens/refinement-001 --json
```

Resume imports native result archives and prepares the next stage only when the exact required terminal identities permit it.
Submit only the newly prepared stage.
Status includes the latest 20 stage descriptions and the full stage directory, but does not infer scheduler liveness or authorize resubmission.
Inspect AFVS's `workflow/status.json` and the actual scheduler before any manual submission.
Exit codes are 0 for validated or complete, 1 for waiting/partial/shortage, and 2 for configuration errors.
`--json` keeps stdout machine-readable; inference logs remain in artifacts.

Workers write immutable attempt artifacts and versioned receipts under externally owned attempts.
They do not update the coordinator database.
Native per-item log archives carry each receipt and all of its required artifacts.
The coordinator snapshots those archives into `screening/imports`, checks exact task identities, qualified runtime evidence, and artifact sizes/digests, then calls the durable acceptance API.
Short copies remain pending; identical receipts are idempotent; contradictory receipts are rejected; superseded attempts cannot replace current results.
Native exit zero and overview counts never establish prediction success.
Invocation stdout, stderr, loaded-runtime evidence, process identities, native preparation logs, row maps, and transferred archives remain available for diagnosis.
The original native selection and configuration bytes are retained with their hashes under `screening/source`.
Unchanged native archives share one content-addressed import snapshot across resume calls.

Unknown scheduler state prevents redispatch.
If a workunit stopped without receipts, explicitly reconcile each affected current attempt using observed scheduler evidence, after checking that its task-owned processes have exited.
The local process/lock guard rejects a stopped declaration while an invocation remains live or uninspectable.

```sh
psichic-screener reconcile --run /task/screens/refinement-001 \
  --attempt ATTEMPT_ID --resolution confirmed-stopped --job-id NATIVE_JOB_ID \
  --evidence 'Scheduler accounting confirms termination; task-owned processes exited.'
psichic-screener resume --run /task/screens/refinement-001
psichic-screener retry --run /task/screens/refinement-001 --failed --engine boltz
```

Use `--resolution unresolved` when liveness cannot be established, or `--resolution still-running` for a live job.
Retry is explicit, keeps prior receipts and report versions, and obeys frozen selection rules.
Changing settings or a committed PSICHIC shortlist requires a new run.

## Rank an existing completed screen

The original ordered `collection_list_file`, native configuration, archives, and selected scenario must still be available.
Grouped native JSON cannot recover a lost global CSV order.
Native CSV summaries are retained as source metadata; legacy scores and first-replica attributes do not qualify as inference provenance.
An imported legacy population is rescored into a new durable run.

```sh
python -m applications.screener rank \
  --source-run /data/completed-afvs-run --afvs /data/pinned-afvs \
  --scenario target --afvs-python /opt/afvs/bin/python \
  --fasta /data/target.fasta --msa-mode single \
  --mode full-ensemble --output /task/screens/reranked-001 \
  -- --accelerator cpu
```

Only compatible validated receipts inside the durable run are reused by resume.
The adapter does not import unproven legacy numbers as fresh observations.
The shared versioned `results.json`, `results.csv`, and `summary.md` reports retain source identities, failures, selection, and ranking evidence.
See [verification](VERIFICATION.md) and the saved-observation comparison helper in `tests/compare_saved.py`.
