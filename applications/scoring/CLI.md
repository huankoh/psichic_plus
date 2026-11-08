# Rank a candidate list locally

Install the focused applications package into an existing compatible PSICHIC environment.
The wheel contains the scoring core, executor, shared Predictor, Boltz adapter, and screening Python sources.
It does not contain weights, Boltz, SyntheMol, or AdaptiveFlow.
The separate `psichic-ligand` package and its compiled extensions must already match that environment's RDKit installation.
Use the repository's `environment_cpu.yml` or `environment_gpu.yml` for the PSICHIC dependencies.
The coordinator requires Python 3.11 or newer.
Boltz runs in a separate Python 3.12 environment with unmodified Boltz 2.2.1.
The independently packaged AdaptiveFlow worker remains Python 3.10 compatible.

```sh
python -m pip install --no-build-isolation --no-deps ./psichic-ligand
python -m pip install ./applications
psichic --help
```

The console command works outside the source checkout.
`python -m applications.scoring` provides the same interface.
For a task-owned `pip --target /absolute/install` smoke installation, set `PYTHONPATH=/absolute/install` and invoke `/absolute/install/bin/psichic` with the same dependency-equipped Python.
An ordinary environment installation needs no checkout path or `PYTHONPATH` setting.

Prepare Boltz in a separate Python 3.12 environment.
For qualified Linux AMD64 execution, install the hash-locked dependency set used by the combined Docker image.
That lock pins Boltz 2.2.1, Torch 2.7.1, Triton 3.3.1, and the psutil required by the local process guardian.

```sh
python3.12 -m venv /opt/boltz
/opt/boltz/bin/python -m pip install -r ./applications/scorer/boltz2/requirements-boltz.lock
```

## Configure cached assets

Supply the PSICHIC model directory containing `config.json`, `degree.pt`, and `model.pt`.
Supply `esm2_t33_650M_UR50D.pt` beside `esm2_t33_650M_UR50D-contact-regression.pt`.
Preflight hashes both ESM checkpoints.
Execution reads them with PyTorch's restricted `weights_only=True` loader, allowing only the checkpoint's `argparse.Namespace` configuration class in a scoped context, then calls ESM's native model constructor.
No missing model is downloaded.

When running in Docker, use the same explicit `--hostname` for creation, status, and resume containers on the same physical host, and mount the saved run at the same path.
For example, use `--hostname psichic-huan-main-a100` consistently on this host.
The default random container hostname will fail the saved run's same-host check.
An explicit hostname does not authorize moving the run between machines or using an unqualified shared filesystem.

```sh
export PSICHIC_MODEL_DIR=/data/weights/PSICHIC_plus
export PSICHIC_ESM_CHECKPOINT=/data/torch/hub/checkpoints/esm2_t33_650M_UR50D.pt
export PSICHIC_CACHE=/data/cache/psichic
export PSICHIC_DEVICE=cpu
export BOLTZ_PYTHON=/opt/boltz/bin/python
export BOLTZ_CACHE=/data/cache/boltz
```

The Boltz cache must already contain `boltz2_conf.ckpt`, `boltz2_aff.ckpt`, `mols.tar`, and the extracted `mols` directory.
The native download routine checks the default checkpoints even when explicit checkpoint overrides are supplied.
Preflight requires those defaults and fingerprints every selected checkpoint and extracted molecule file.
It checks the molecule archive's file inventory against the extraction and rejects missing or truncated records before starting a model.
Prepare missing assets separately, before validation.
Do not use the executor as a model downloader.

For a GPU allocation, set `CUDA_VISIBLE_DEVICES` to the assigned GPU index or full UUID.
Use `--psichic-device cuda:0` and native `--accelerator gpu --devices 1` within that allocation.
The executor never chooses a GPU or stops another workload.
CPU runs use `--psichic-device cpu` and native `--accelerator cpu`.
One invocation runs at a time.
Every PSICHIC model process exits before Boltz starts.

## Validate before inference

The CSV defaults to `ID` and `SMILES` columns.
Override either name explicitly when needed.
Extra columns, duplicate IDs, duplicate structures, and row order are retained.
Malformed rows, duplicate headers, and empty populations are rejected.

```sh
psichic rank \
  --candidates /data/molecules.csv --id-column ID --smiles-column SMILES \
  --fasta /data/target.fasta --msa-mode single \
  --mode psichic-only --batch-size 32 --chunk-size 64 \
  --output /data/results/psichic-001 --validate-only
```

Validation checks the complete input, cached files, installed dependencies, runtime source, and resolved settings.
It imports runtime code and hashes checkpoint files without deserializing them.
It constructs no Predictor, ESM model, or PSICHIC layer, initializes no CUDA context, and creates no run output or inference cache.
For PSICHIC, it inspects class signatures and renders PyG templates in memory without importing or executing the rendered source.
It displays evaluation counts, not a runtime estimate.
Remove `--validate-only` to execute.
PSICHIC-only requires no Boltz configuration.
Options for unused engines are rejected.

Native complex input uses `--target-yaml /data/target.yaml --ligand-id B` instead of FASTA.
For PSICHIC with multiple protein chains, also select `--psichic-chain A`.
PSICHIC sees that selected sequence, while Boltz sees the retained complex.
Referenced templates are copied and hashed into the frozen run.
Only `--msa-mode single` is accepted.
Native server, authentication, and remote-MSA options are rejected, including equals syntax.

## Rank with Boltz

```sh
psichic rank \
  --candidates /data/molecules.csv --fasta /data/target.fasta --msa-mode single \
  --mode pyramid-refinement --shortlist-size 100 \
  --boltz-timeout 7200 --output /data/results/refinement-001 \
  -- --accelerator cpu
```

K is a required global row count for either pyramid mode.
No universal K is supplied.
The shortlist is frozen after all PSICHIC outcomes become terminal.
A shortage starts zero Boltz jobs, and a failed shortlisted candidate is never replaced.

Choose `boltz-only` to evaluate all rows with Boltz.
Choose `full-ensemble` to evaluate all rows with both engines.
Choose `pyramid-ensemble` to combine both scores after global shortlisting.
The new ensemble default is `concentration-v1`, which requires Boltz affinity and valid PSICHIC distribution evidence.
It rejects manual weights and binder-probability ranking.
Use `--ensemble rank-v1` for weighted average ranks with explicit compatible metric and weight options.
The older standalone `applications.scorer.boltz2` command keeps its `rank-v1` default.

`--psichic-output affinity` is the default display detail.
`--psichic-output distribution` displays alpha, beta, and concentration from the same forward pass.
Concentration ensembles retain the required distribution evidence regardless of display detail.
A finite affinity with an invalid distribution remains visible with the original diagnostics.
Pandas missing values are represented as null in retained native rows and as `None` in the matching distribution diagnostic.

## Distinguish normal sampling from quick checks

Normal runs resolve the installed Click defaults, including values that differ from the Python function signature or help prose.
Normal preserves upstream sampling unless explicitly overridden.
Both presets record an explicit seed, default 42, for each exact submitted batch.
Every native default, internal configuration class, selected file, environment control, and explicit override is retained before task preparation.

`--preset quick-check` expands to one recycling step, two structure sampling steps, one structure sample, two affinity sampling steps, one affinity sample, zero data-loader workers, and one preprocessing thread.
Explicit compatible native options after `--` override that preset and are recorded.
Native option abbreviations, duplicate options, unsupported names, and workflow-owned collisions are rejected.
No shell command string is evaluated.
Quick-check is an integration exercise and does not establish production prediction quality.
Normal sampling and scientific ranking quality still require representative GPU qualification.

## Resume and retry

```sh
psichic status --run /data/results/refinement-001
psichic resume --run /data/results/refinement-001
psichic retry --run /data/results/refinement-001 --failed --engine boltz
```

Resume ingests completed receipts before reconciling missing outcomes.
It leaves matching live processes alone and continues pending or confirmed stopped work.
It does not retry recorded failures.
An owner process survives coordinator loss, records actual native completion, and publishes task-bound receipts.
A model process watches an ownership pipe and stops its own process group if its owner dies.
A small member of that group also clears descendants if both model and owner disappear.
Process start identity and a held invocation lock prevent PID-reuse mistakes.
External scheduler ownership is not treated as local ownership.

Retry preserves prior attempts and immutable reports.
A multirow retry is one transaction, so a prohibited PSICHIC retry cannot partially reset Boltz rows.
PSICHIC retry after a committed shortlist requires a new run.
Boltz retry preserves K and may change every ensemble score because the common-success cohort changes.
Changed source, weights, environment, or frozen settings requires a new run before inference.
Original CSV or target paths can change without altering the retained population and target assets.

The supported durability boundary is one POSIX host and cooperative local storage with the core's documented locking and fsync semantics.
Cross-host scheduling, network filesystems, power loss, and object-store transport are unqualified.

## Read the result

Human output starts with resolved settings and evaluation counts, then prints status, counts, a bounded ranking, and report paths.
The top table includes original IDs, SMILES, raw scores, optional distribution fields, final scores, and statuses.
The full population stays in versioned `reports/v00000001/results.json` and `results.csv`, with a readable `summary.md`.
`latest.json` identifies the newest immutable version.
The CLI never rewrites a root-level report.

`--json` emits exactly one schema-versioned envelope on stdout.
Native Python and C-extension output goes to invocation logs.
CLI progress goes to stderr.
The response contains run and report identities, counts, bounded top rows, report paths, and an actionable next step.
It does not materialize the full population.
Exit 0 means complete or validated, exit 1 means partial, shortage, or waiting, and exit 2 means invalid configuration.
JSON uses null instead of NaN.
Inspect `local/invocations/<invocation-id>/owner.log`, `stdout.log`, `stderr.log`, `completion.json`, and transferred receipt artifacts for incomplete work.

The local implementation is covered by controlled CPU process and artifact tests.
Preflight fingerprints imported package source, compiled libraries, and Jinja templates.
The PSICHIC profile also records expected SHA-256 hashes for the `PNAConv`, `DrugProteinConv`, and `GCNConv` propagate modules generated by PyG 2.7.0.
The `generated_source` records identify the class, template, generator contract, and fusion configuration.
The two custom layers use non-fused propagation; `GCNCluster` uses `GCNConv` with its default add aggregation and fused propagation.
Their expected bytes come from class introspection and template rendering before model construction, independently of actual generated files.
Both the profile identity and each prediction's source fingerprint include these records.
Workers verify actual generated bytes by exact module identity and hash, and retain the actual paths and hashes in their loaded-source reports.
Temporary paths may differ between processes.
Changed bytes, unexpected generated modules, and unsupported generator versions or changes to the qualified fusion and edge-update configurations fail explicitly.
Existing import-time temporary modules still require their pre-recorded identity and exact bytes.
If inference dynamically loads an additional package absent from that inventory, acceptance fails explicitly.
That runtime needs a qualified preflight update before its outputs can be reused.
Boltz retains its installed-RECORD and loaded-source checks and has no new dynamic-source authorization.
Those controlled fixtures are not trained-model inference evidence.
Retained A100 evidence separately qualifies real Boltz and PSICHIC+ execution with quick-check sampling for the tested source revisions.
That evidence does not establish normal-sampling scientific quality.
The [generator adapter](../generator/README.md) imports native products and offers automatic final scoring through this executor.
Its independent real-model A100 verification is recorded separately from the shared executor fixtures.
Docker images must be rebuilt from the exact source revision being released.
The distributed AdaptiveFlow deployment remains a separate qualification from this local executor.
