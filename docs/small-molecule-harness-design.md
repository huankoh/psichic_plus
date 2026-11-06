# Small-molecule harness design

This document designs a Chaperone-style harness for small-molecule scoring, generation, and screening.
It targets a new private repository with a clean break from current entry points.
It covers local and SLURM execution in version 1.
It designs block manifests so MCP tools and agent skills can drive the same harness later without a redesign.
It is a design only.
No code in this repository changes under this plan.

## 1. What exists today

The current system has one model and three separate embeddings of it.
Each embedding re-solves target loading, batching, failure handling, and execution.
That duplication is the problem this design removes.

### 1.1 Scorer as a library

`Predictor` in `applications/scorer/psichic_plus/predictor.py` loads ESM and PSICHIC+ once per process.
It featurizes one FASTA target into an in-memory protein state.
It scores `(token, SMILES)` pairs and returns `Success` or `Failure` records.
A success carries `raw_affinity`, `adaptive_score` as its negation, and an optional beta distribution.
`score_with_evidence` also retains the native row from the same forward pass.
The inference preset, precision, and batching are fixed to `fast_screening`, FP32, and input order.

### 1.2 Scorer as durable ranking

`applications/scoring` freezes a candidate population into a SQLite-backed `ScoringRun`.
Workers execute versioned `Task` declarations and publish `Receipt` evidence.
The coordinator accepts receipts, commits an optional pyramid shortlist, and finalizes immutable reports.
`applications/scorer/boltz2` adds Boltz 2.2.1 in a separate environment through its unmodified CLI.
Pure functions in `ensemble.py` implement `rank-v1` and `concentration-v1` combination over a common-success cohort.
The local executor runs one invocation at a time under process-group ownership.
The contracts already distinguish `Ownership.LOCAL` from `Ownership.EXTERNAL` for scheduler jobs.
No scheduler adapter exists yet.
The run is locked to one hostname and one local filesystem.

### 1.3 Generator embedding

`applications/generator` patches pinned SyntheMol to add a `psichic_plus` score type.
`PSICHICScorer` scores one SMILES per call with batch size 1 and returns one float.
SyntheMol owns MCTS, RL, score vectors, weights, caches, and output.
Building blocks need a separate pre-scoring command before generation.
The generator therefore cannot batch scoring, cannot see distributions, and cannot swap scorers without another upstream patch.

### 1.4 Screener embedding

`applications/screener` patches pinned AdaptiveFlow-VS to add a `psichic_plus` docking method.
Each native worker supervises a long-lived inference child over JSON lines with an 8 MiB frame limit.
AdaptiveFlow owns tranches, workunits, scheduling templates, prescreen selection, and summaries.
Brute-force screening is `prescreen_mode=0` with a full-library todo file.
Adaptive screening is a prescreen, a tranche-average selector, and a primary screen.
The adapter re-implements target parsing and model supervision instead of reusing `Predictor`.

### 1.5 Screening runtime

`psichic_plus_screening/inference.py` is a monolithic CLI for CSV-in and CSV-out screening.
`multi_gpu_inference.py` shards rows across GPUs by launching one CLI process per GPU.
Ligand graphs use a versioned content-hash cache keyed by `FEATURIZATION_VERSION`.
Protein featurization repeats per process unless a protein cache directory is configured.

### 1.6 Consequences

A new scorer today needs three integrations.
A new generator needs a new patch or a new callback shape.
A new screening strategy needs either AdaptiveFlow internals or a new bespoke driver.
Execution knowledge is scattered across Docker flags, `CUDA_VISIBLE_DEVICES` conventions, AFVS templates, and local process guardians.
Provenance is strong in the durable ranking path and weak in the generator and screener paths.
There is no registry, no common candidate table, and no agent-callable surface.

## 2. What Chaperone does

Chaperone is a harness, not a model.
Its transferable ideas are listed here in harness terms, stripped of protein specifics.

### 2.1 Blocks over a universal table

Every tool is a block with a stable string name in a `Blocks` registry.
Every block takes a `PipelineDataHandler` and returns a `PipelineDataHandler`.
A handler row carries an ordered lineage of artifacts plus a JSON metrics bag.
Blocks append artifacts and namespaced metrics instead of inventing output files.
Pipelines are ordinary Python scripts that pass handlers between `engine.run` calls.

### 2.2 One engine for execution

`ExecutionEngine` owns the run directory, manifest, chunking, caching, retries, and compilation.
`engine.run` executes one block.
`engine.submit` plus `gather` runs independent blocks concurrently.
`engine.cycle` repeats a sub-pipeline for iterative refinement.
`RuntimeArgs` selects `local`, `slurm`, or cloud execution without changing block code.
`JobArgs` declares portable resource needs such as a GPU size class, memory, time, and rows per job.
Site settings in `.env` files map those portable needs to real partitions and gres strings.
Blocks never name cluster resources directly.

### 2.3 Isolation by container

Each heavy tool ships as a named container installed on demand.
Conflicting dependency stacks never share one environment.
The engine prepares inputs, runs the container command per chunk, and compiles outputs back into the handler.
Local GPUs are auto-detected and used instead of SLURM unless the caller forces submission.

### 2.4 Selection as data operations

A `filter` block prunes rows with a boolean expression over namespaced metrics.
Missing metrics fail selection instead of passing silently.
Diversity selection is an explicit operation with required oracle steps.
Cheap checks run before expensive inference.
Expensive predictions are reused for cheap downstream analyses.

### 2.5 Agents as first-class users

The repository ships `AGENTS.md` plus task skills for design, scripting, debugging, and SLURM operation.
A monitor MCP exposes campaign status and a `wait` tool for detached drivers.
A single tested `score.py` acts as the oracle for iterative research.
Drivers encode smoke tests, pilots, promotion rules, and stop rules so agents spend compute only on evidence.

## 3. Design goals and non-goals

### 3.1 Goals

One candidate table flows through sources, scorers, combiners, filters, and sinks.
One scorer integration serves ranking, generation guidance, and screening.
A generator swap changes configuration, not harness code.
Composite scoring is a pure function over table columns with explicit cohort rules.
Brute-force and adaptive screening are two strategies over the same blocks.
The same pipeline script runs on a laptop CPU, a workstation GPU, and a SLURM cluster.
Every block declares its config schema, resource needs, environment, and fingerprint inputs in one manifest.
An agent can discover blocks, validate a pipeline, run it, wait for it, and read bounded results through MCP tools generated from those manifests.
Failures stay per-row and visible.
No failure becomes a silent zero, a dropped row, or a placeholder score.

### 3.2 Non-goals for version 1

Cloud execution is deferred behind the executor seam.
Training and fine-tuning stay outside the harness.
Multi-target and multi-replica aggregation need their own contracts and stay out until designed.
Power-loss and cross-host database guarantees stay out.
The harness does not re-implement SyntheMol search or AdaptiveFlow library preparation.
Scientific ranking quality is validated by benchmarks, not asserted by the harness.

## 4. Core vocabulary

This design uses Module, Interface, Seam, and Adapter in the codebase-design sense.
A Module is anything with an Interface and an implementation.
An Interface includes types, invariants, error modes, required configuration, and performance characteristics.
A Seam is where behavior can change without editing the caller.
An Adapter satisfies an Interface at a Seam.
A Seam is real only when at least two Adapters vary across it.
Deep Modules hide chunking, retries, caching, and process supervision behind small Interfaces.

## 5. The candidate table

The candidate table replaces Chaperone's structure handler.
It is the single data shape every block reads and writes.
Arrow and Parquet are the canonical on-disk form.
Blocks may use pandas internally only when an upstream tool requires it.

### 5.1 Row identity

Each row has a stable `row_key` assigned at population freeze.
Each row retains the original `candidate_id` exactly as submitted, including duplicates.
Each row retains the exact submitted SMILES and, after standardization, the normalized SMILES.
Duplicate IDs and duplicate structures remain separate rows.
Ordering is manifest order unless a block explicitly sorts a view.
This preserves the identity discipline already proven in `ScoringRun`.

### 5.2 Columns

Core columns are `row_key`, `candidate_id`, `smiles_submitted`, `smiles_normalized`, `target_id`, and `source`.
The `source` column is a structured locator, not free text.
A library source records library, tranche, collection, and member.
A generator source records generator name, round, reaction steps, and building-block IDs.
A file source records path digest and row number.
Metric columns are namespaced by block instance, for example `psichic.affinity`, `psichic.alpha`, `psichic.beta`, `psichic.concentration`, `boltz.affinity_pred_value`, and `boltz.affinity_probability_binary`.
Status columns are namespaced the same way, for example `psichic.status`, `psichic.stage`, and `psichic.reason`.
Artifact columns hold content-addressed references, for example a Boltz structure, a docked pose, or an interpretation bundle.
Selection columns record `selected`, `selection_reason`, and `final_rank` per selection step name.

### 5.3 Target as an artifact

A target is a versioned artifact, not a FASTA path passed around.
Target resolution happens once at the boundary from FASTA, native YAML, or a structure.
The resolved target records identity, normalized sequence or sequences, chain selection, MSA assets, and digests of every input file.
Featurized target state, such as ESM embeddings and contact graphs, is a cached derived artifact keyed by target digest plus featurizer version.
Scorer blocks consume the target artifact.
Workers never re-parse FASTA files with three different parsers.

## 6. Block roles and Seams

Five Seams carry real variation.
Everything else is ordinary code inside a Module.

### 6.1 Source seam

A Source produces candidate rows.
Adapters include a CSV or Parquet file, a prepared library with tranches, and a generator round.
A Source never scores.
A Source must emit stable source locators so every score stays traceable to its origin.

### 6.2 Scorer seam

A Scorer consumes pending rows for one target and appends namespaced metrics and statuses.
Adapters include PSICHIC+, Boltz-2, a docking program, an RDKit property calculator, and a fake scorer for tests.
The Interface is batch-first.
A Scorer declares its preferred batch size, its maximum batch size, and whether batch composition can change results.
A Scorer loads its model once per worker process and scores many chunks.
A Scorer returns one outcome per submitted row, in order, with explicit failures.
Single-molecule callers are served by a micro-batching service over the same Interface, never by a separate batch-size-1 integration.
This one decision removes the SyntheMol callback and the AdaptiveFlow child as special cases.

### 6.3 Generator seam

Generators come in two kinds, and the Interface names both honestly.
A proposer emits candidates given a count, a seed, and optional scored context from prior rounds.
A guided search owns an internal loop and needs scoring during the loop.
SyntheMol is a guided search.
Most samplers and enumeration tools are proposers.
The harness runs proposers through an explicit propose, score, select cycle.
The harness runs guided searches as a block that receives a scoring service handle with micro-batching, timeouts, and a call budget.
Both kinds emit ordinary candidate rows with full route metadata.
Search-internal scores are recorded as generator metrics.
They never become accepted scorer evidence without a separate scoring pass.

### 6.4 Combiner seam

A Combiner is a pure function over metric columns.
Adapters include single-metric ranking, weighted average ranks (`rank-v1`), concentration-weighted combination (`concentration-v1`), z-score blends, and Pareto fronts.
A Combiner declares required metrics, the eligibility rule, the cohort it normalizes over, tie handling, and score direction.
Normalization uses only the common-success cohort unless the Adapter documents otherwise.
Missing or failed metrics make a row ineligible with a named reason.
Combiners never invoke models and never need a GPU.
This keeps the proven `ensemble.py` shape and generalizes it beyond two engines.

### 6.5 Filter and selector seam

Filters prune rows by boolean expressions over namespaced metrics, following Chaperone's `expr` language.
Selectors implement top-K, quantile, threshold, cluster-diverse, and tranche-statistic selection.
A selector records its rule, its input count, its output count, and a per-row reason.
Pyramid shortlisting is a selector followed by a second scorer, not a mode flag inside a ranker.
A shortage stops downstream scoring explicitly.
A shortlisted failure is never silently replaced.

### 6.6 Sink seam

A Sink writes reports and exports.
Adapters include the immutable versioned report directory, a plain CSV export, and a next-round candidate file for a generator cycle.
Sinks read stored snapshots.
Sinks never recompute scores.

## 7. Screening strategies

A screener is not a block kind.
A screener is a strategy that composes sources, scorers, selectors, and sinks under a budget.

### 7.1 Brute-force strategy

Brute force maps one or more scorers over every row, then combines, selects, and reports.
It is the correct default when the library fits the compute budget.
It needs no tranche logic.
Chunking and caching carry the whole cost story.

### 7.2 Adaptive strategy

The adaptive strategy samples each tranche, scores the sample, aggregates per-tranche statistics, selects whole tranches or collections, and scores the selection in full.
Tranche identity comes from the source locator.
The tranche statistic is a declared Combiner over the sample, for example mean of the best decile.
The selector takes a target ligand budget and records that whole-collection selection can overshoot it.
This re-implements the AdaptiveFlow prescreen logic natively over the candidate table.
Prepared AdaptiveFlow libraries remain readable through a library Source Adapter.
Running inside native AdaptiveFlow workunits remains possible through an external-system Adapter for sites that already operate it.
Version 1 should implement the native strategy first because it reuses the same executor, cache, and reports as brute force.

### 7.3 Active-learning strategy

An active-learning loop is a cycle of score, train a cheap surrogate outside the harness, propose or select by acquisition, and rescore.
The harness provides the cycle, the budget, and the lineage.
The surrogate itself is a Scorer Adapter once trained.
This falls out of the design without new concepts, so version 1 only needs the cycle primitive.

## 8. Plugin model

### 8.1 Discovery

Blocks register through Python entry points under one group per role, for example `molharness.scorers`.
Built-in blocks and third-party blocks use the same mechanism.
The registry resolves a string name plus config into a validated block instance.
Duplicate names fail at startup with both providers named.
A pipeline script names blocks by string, exactly as Chaperone scripts do.

### 8.2 Manifest

Every block ships one manifest.
The manifest declares name, role, version, description, config schema as JSON Schema, required input columns, produced columns, resource needs, environment reference, batching limits, determinism notes, and fingerprint inputs.
The manifest is the contract for the CLI, the Python API, validation, and MCP tool generation.
No second hand-written tool schema exists.
Config resolves in layers with highest priority first.
The layers are CLI overrides, pipeline script values, block defaults, then site settings.

### 8.3 Environments as Adapters

Conflicting stacks are normal in this domain.
PSICHIC+ pins Torch 2.6 with PyG 2.7.
Boltz 2.2.1 needs its own Torch.
AdaptiveFlow workers historically expect older Python.
The manifest therefore names an environment, and three transport Adapters satisfy the same block Interface.
An in-process Adapter imports the block directly when environments match.
A subprocess Adapter runs the block under a named Python executable with a framed request and response protocol.
A container Adapter runs a named image per chunk with mounted inputs and outputs.
The engine picks the transport from the manifest and the site config.
Block authors write scoring logic once against files and tables, never against transport details.

### 8.4 Versioning

A block name plus manifest version plus resolved config plus environment digest identifies an execution profile.
Changing any of them changes the fingerprint.
Results from different profiles never mix inside one run.
This generalizes the execution-profile rule already enforced by `ScoringRun.prepare`.

## 9. Execution engine

### 9.1 Shape

The engine mirrors Chaperone's `ExecutionEngine`.
Construction takes a run directory, a resume flag, and default runtime and job arguments.
`run` executes one block over a handler and returns a handler.
`submit` and `gather` run independent blocks concurrently, for example PSICHIC+ and Boltz over the same rows.
Concurrent GPU blocks still serialize per allocated GPU unless site config provides more devices.
`cycle` repeats a named sub-pipeline with its own lineage scope.
`compile` materializes the final table and report snapshot.
Pipelines are Python scripts.
A YAML pipeline format can compile to the same calls later, but Python stays the source of truth in version 1.

### 9.2 Chunking

The engine splits pending rows into chunks.
Chunk size is rows per job.
Batch size is rows per model forward pass inside one worker.
These are different numbers and the design keeps them separate.
The durable core already bounds a prepared batch to 4096 rows.
The manifest gives each scorer a preferred chunk range.
Job arguments can override rows per job per run.
A worker loads the model once, then processes its chunk in internal batches, then exits.
Cold-start cost therefore amortizes over the chunk, which is the main efficiency lever on clusters.

### 9.3 Runtime modes

`local` prepares, executes, and compiles on the current host.
`slurm` prepares locally, executes chunks as a job array, and compiles locally after ingestion.
The mode lives in runtime arguments, never in block code.
A backend setting resolves the default mode.
Local CUDA detection can route GPU blocks locally instead of submitting, with an explicit off switch for login nodes with visible GPUs.
Cloud modes slot in later as new executors without touching blocks or strategies.

### 9.4 Job arguments and site config

Job arguments declare partition class, GPU class, memory, wall time, CPUs per task, account, quality of service, extra scheduler options, and rows per job.
Blocks declare portable needs only, for example GPU class `small` or `large`.
Site settings map classes to partitions and gres strings.
Site files load in a documented priority order ending with `.env.local` and process environment.
The CLI can override partition, gres, account, and rows per job per run.
No pipeline script contains a cluster-specific string.

### 9.5 GPU contract

One chunk job owns one GPU allocation.
The executor sets `CUDA_VISIBLE_DEVICES` to exactly one assigned device.
Inside the job, the block always addresses `cuda:0`.
The executor never picks a free GPU by scanning and never stops another workload.
CPU blocks run with CUDA hidden before Torch import.
Thread counts are set explicitly per worker from CPUs per task.
These rules codify conventions the current executor and AdaptiveFlow child already follow ad hoc.
Multi-GPU single-node runs are multiple concurrent chunk jobs, not one sharding wrapper script.
Data-parallel training-style execution stays out of scope.

### 9.6 SLURM specifics

Each chunk becomes one array task with a rendered `sbatch` script stored in the run directory.
Workers write only to their own transfer directories.
Workers never write the run database.
The coordinator ingests transfer directories and accepts or rejects evidence centrally.
Job identity maps array index to prepared task IDs in coordinator-owned evidence.
Reconciliation queries the scheduler for live, stopped, or unknown states.
Only a confirmed-stopped attempt becomes dispatchable again.
Retries are bounded per chunk with the failure evidence retained.
This is the `Ownership.EXTERNAL` path the durable core already anticipates.

### 9.7 Scoring service for guided search

Job arrays fit batch scoring.
They do not fit a guided search that needs thousands of low-latency scores.
The engine therefore offers a service handle backed by one persistent worker process per GPU allocation.
Locally, the worker is a supervised subprocess with framed requests, timeouts that include cold start, and respawn after failure.
On SLURM, the guided-search block and its scoring worker are co-scheduled in one job so no cross-node service discovery is needed in version 1.
The service micro-batches concurrent single requests up to the scorer batch size or a short wait window.
Every service result is logged as generator-internal evidence with the same fingerprint rules.

## 10. Durable run store

### 10.1 Keep the proven core idea

The new repository should carry over the `ScoringRun` idea, generalized.
A run freezes its population, target, resolved settings, and execution profiles at creation.
Attempts are prepared before dispatch with durable IDs.
Workers publish receipts to transfer directories.
The coordinator validates hashes, publishes content-addressed blobs, and accepts idempotently.
Conflicting receipts are retained and rejected.
Reports are immutable versioned snapshots with a `latest` pointer.
Resume ingests first, reconciles second, and never redispatches a live attempt.

### 10.2 Generalize the closed parts

The current core hardcodes an `Engine` enum with two members, five mode strings, and PSICHIC-specific receipt validation.
The new core keys evaluations by block instance name instead of an enum.
Receipt validation dispatches to the block's declared output schema.
Selection becomes a recorded step that gates downstream pending sets, instead of a PSICHIC-then-Boltz special case.
Settings become the resolved pipeline graph plus per-block resolved config.
Any settings change still requires a new run.

### 10.3 Storage boundary

Version 1 keeps the single-host coordinator with SQLite, `flock`, fsync, and atomic rename on a local or site-qualified filesystem.
SLURM workers reach the coordinator only through transfer directories on shared storage.
That keeps the database off network filesystems while still using the cluster.
Object-store transport and cross-host coordination remain future work with their own qualification.

### 10.4 Provenance tiers

The current executor fingerprints full installed source inventories, generated PyG modules, and Triton native builds.
That strictness is valuable for audited ranking and expensive for everyday screening.
The design splits provenance into two tiers.
The standard tier fingerprints resolved config, weights, target assets, native inputs, package versions, device, precision, seeds, and batch context.
The strict tier adds loaded-source inventories and native-build evidence per block that supports it.
The tier is a run setting recorded in every fingerprint.
Receipts from different tiers never mix.

## 11. Caching

Three caches matter, and they key differently.
The ligand graph cache keys by featurization version plus SMILES hash and is shared across runs.
The target state cache keys by target digest plus featurizer version and is shared across runs.
The result cache keys by full task fingerprint and lives inside the run store as accepted receipts.
Step-level and chunk-level skip flags exist for debugging, following Chaperone's runtime arguments.
Cache writes publish atomically.
Corrupt cache entries recompute and report, never fail silently into wrong scores.
No automatic garbage collection runs in version 1.

## 12. Efficiency rules

Model load and ESM featurization dominate small runs, so workers are long-lived per chunk and targets featurize once.
Batch composition effects are declared per scorer, and fingerprinting binds the exact submitted batch context where composition matters.
Size-bucketed batching from the screening runtime becomes an optional per-scorer batching policy, measured before adoption.
Cheap filters run before expensive scorers in every strategy template.
Pyramid strategies spend Boltz-class compute only on a frozen shortlist.
Guided search uses micro-batching instead of batch size 1.
Concurrency caps per GPU, per partition, and per run keep the cluster usable.
Every run records evaluation counts before execution.
The harness reports counts as counts, never as time estimates.

## 13. Agent and MCP surface

### 13.1 Tools generated from manifests

Each block manifest generates one MCP tool description with the config JSON Schema as its input schema.
Engine tools cover list blocks, describe block, validate pipeline, run pipeline, status, wait, cancel, and read results.
Result reads return bounded envelopes with counts, top rows, report paths, and a next action, following the current `psichic --json` discipline.
Full populations stay in Parquet and report files, never in tool responses.
Validation constructs no model, initializes no CUDA, and creates no run.

### 13.2 Skills and project context

The new repository ships an `AGENTS.md` that routes agents to skills.
Skills cover pipeline authoring, block authoring, SLURM background runs, input debugging, and result interpretation.
A tested `score.py` oracle pattern supports iterative campaigns.
Drivers encode smoke, pilot, promotion, and stop rules.
Agents monitor detached drivers through the wait tool instead of polling the scheduler by hand.

### 13.3 CLI

The CLI mirrors the engine with subcommands for blocks, validate, run, status, resume, retry, and results.
Per-block flags generate from manifests.
Global flags cover mode, partition, rows per job, cache control, and test-size limits.
Exit codes distinguish complete, partial or waiting, and invalid configuration.
Human output stays bounded.
Machine output is one versioned JSON envelope on stdout.

## 14. Worked examples

The examples use illustrative names for the new repository.
They show shape, not final spelling.

### 14.1 Brute-force composite ranking

```python
from molharness import ExecutionEngine, JobArgs, RuntimeArgs, init_cli

init_cli()
engine = ExecutionEngine(rundir="runs/rank-001")
h = engine.run("csv_source", path="inputs/molecules.csv", id_column="ID", smiles_column="SMILES")
h = engine.run("psichic", data=h, target="inputs/target.fasta", output="distribution",
               job_args=JobArgs(rows_per_job=512))
h = engine.run("topk", data=h, name="shortlist", by="psichic.affinity", k=100)
h = engine.run("boltz2", data=h, only_selected="shortlist",
               job_args=JobArgs(gpu_class="large", rows_per_job=8))
h = engine.run("concentration_v1", data=h)
engine.run("report", data=h, top_n=20)
```

The same script runs locally by default on a GPU workstation.
Adding `--partition` and site settings submits the scorer chunks to SLURM.
Swapping `concentration_v1` for `rank_v1` with weights changes only the combiner line.

### 14.2 Generator swap

```python
h = engine.run("synthemol", target="inputs/target.fasta", scorer="psichic",
               building_blocks="inputs/scored_blocks.csv", n_rollout=100, seed=0)
h = engine.run("psichic", data=h, target="inputs/target.fasta")
engine.run("report", data=h)
```

Replacing `synthemol` with a proposer block changes the first line only.
The harness then runs propose, score, and select cycles under `engine.cycle`.
The rescore pass is what makes generated candidates comparable to library candidates.

### 14.3 Adaptive screening

```python
h = engine.run("library_source", root="/libraries/prepared", todo="inputs/full.todo")
h = engine.run("tranche_sample", data=h, per_tranche=200, seed=0)
h = engine.run("psichic", data=h, target="inputs/target.fasta")
h = engine.run("tranche_select", data=h, statistic="mean_top_decile", budget=100000)
h = engine.run("psichic", data=h, name="psichic_primary", only_selected="tranche_select", full_collections=True)
engine.run("report", data=h)
```

Deleting the sample and select steps yields brute force over the same library.
No block changes.

## 15. New repository layout

```text
molharness/
  core/          candidate table, target artifacts, run store, receipts, reports
  engine/        ExecutionEngine, chunking, cycle, service handle
  executors/     local, slurm
  registry/      entry-point discovery, manifests, config resolution
  blocks/
    sources/     csv, parquet, prepared library
    scorers/     psichic, boltz2, rdkit_props, fake
    generators/  synthemol (guided), enumerate (proposer)
    combiners/   rank_v1, concentration_v1, pareto
    selectors/   topk, filter_expr, tranche_select, diverse
    sinks/       report, csv export
  transports/    in_process, subprocess, container
  mcp/           tool generation and server
  cli/           generated command line
containers/      one Dockerfile per isolated environment
examples/        brute force, pyramid, generation, adaptive screening
skills/          agent skills and AGENTS.md routing
tests/           contract, fake-block end-to-end, executor, slurm rendering
```

Core never imports a model stack.
Blocks import heavy stacks only inside their worker entry points.
The coordinator stays light enough to run on a login node.

## 16. What to port, rewrite, or leave

Port the durable run-store concepts and tests, generalized as in section 10.
Port the pure combiner functions and their cohort tests unchanged in behavior.
Port the target YAML validation and Boltz input rendering into the Boltz block.
Port the ligand featurization version discipline and cache layout.
Rewrite `Predictor` as the PSICHIC+ scorer block over the candidate table, keeping its outcome and distribution semantics.
Rewrite building-block pre-scoring as an ordinary scorer pass plus a CSV sink.
Rewrite the AdaptiveFlow child protocol as the generic subprocess transport plus the scoring service.
Leave the SyntheMol and AdaptiveFlow upstream patches behind.
Replace them with a guided-search Adapter that drives SyntheMol through its public scorer factory from outside the upstream tree where licensing and upstream shape allow, or keep a thin patch only if no external hook exists.
Leave the monolithic screening CLI and the multi-GPU sharding wrapper behind.
Their batching policies survive as scorer options with measurements.

## 17. Testing strategy

Contract tests run every registered block against its manifest for schema, columns, and failure behavior.
Fake scorers and fake generators drive full engine tests on CPU in seconds.
SLURM tests render `sbatch` scripts and ingest synthetic transfer directories without a cluster.
One small real-model CPU end-to-end test per scorer guards the worker entry points.
GPU qualification runs the same pipelines on one assigned GPU and compares within declared tolerance.
Resume tests kill workers and coordinators at each lifecycle point.
A generator test proves micro-batching changes throughput without changing accepted rescore results.
An adaptive test proves tranche selection on a synthetic library with planted signal.

## 18. Phasing

Phase 1 builds core, registry, local executor, fake blocks, CSV source, report sink, and contract tests.
Phase 2 adds the PSICHIC+ scorer, combiners, top-K, and brute-force plus pyramid examples locally on CPU and GPU.
Phase 3 adds the SLURM executor, site config, reconciliation, and cluster qualification.
Phase 4 adds Boltz-2 through the subprocess or container transport and composite examples.
Phase 5 adds the SyntheMol guided-search Adapter, the scoring service, and generation examples.
Phase 6 adds the prepared-library source, tranche sampling and selection, and the adaptive strategy example.
Phase 7 adds MCP tool generation, the monitor wait tool, skills, and `AGENTS.md`.
Each phase ends with a runnable example and its tests green.
No phase depends on a later phase to be useful.

## 19. Risks and open decisions

The SyntheMol external-hook question decides whether a thin patch survives.
It needs a spike against the pinned upstream before phase 5 planning.
Batch-composition sensitivity of PSICHIC+ decides how strictly fingerprints must bind batch context for cross-run reuse.
It needs a measurement, not an assumption.
Shared-storage behavior for transfer directories decides SLURM ingestion throughput.
It needs site qualification on the target cluster filesystem.
Container versus subprocess transport for Boltz depends on site container policy.
The manifest keeps both open until a site decides.
Tranche statistic choice is scientific, not architectural.
The design makes it explicit and recorded so benchmarks can settle it.
License boundaries for SyntheMol, AdaptiveFlow, Boltz, and model weights carry over unchanged.
A process or container boundary does not by itself establish license compatibility.

## 20. Decision summary

Build a new private repository with a clean break.
Center it on one candidate table, five Seams, manifest-driven plugins, one engine, and two executors.
Treat screening strategies and generator loops as compositions, not block kinds.
Keep the durable evidence discipline, generalized beyond two hardcoded engines.
Generate CLI and MCP surfaces from manifests so humans and agents drive identical machinery.
Qualify GPU and SLURM behavior with the same example pipelines that users run.

## 21. Multi-target and multi-scorer extension

This section amends the version 1 non-goal on multi-target work.
Multi-scorer support needs no amendment.
Multi-target support changes row identity and must be designed into the core from the start, even if the first shipped pipelines use one target.

### 21.1 Multi-scorer per molecule

Each scorer instance owns a namespaced column set and a namespaced status set.
Instances run in sequence with `run` or concurrently with `submit` plus `gather`, subject to GPU allocation limits.
A combiner declares the exact metric columns it requires, regardless of how many scorers produced them.
Target-free scorers, such as RDKit property calculators, run once per molecule.
Target-dependent scorers run once per molecule and target pair.
Replicas are scorer instances that differ only by seed or replica index.
A replica aggregation combiner then owns mean, minimum, or another declared reduction.
This replaces a special replica contract with ordinary instances plus one combiner.

### 21.2 Multi-target row identity

The canonical scored unit becomes a pair of one molecule and one target.
A pair table carries `pair_key`, `molecule_key`, and `target_id` alongside the molecule fields.
The population freeze records an explicit pair list.
A full cross product is one way to build that list, never an implicit assumption.
Sparse pair lists cover proteome panels where only named pairs are in scope.
The existing screening runtime already accepts this input shape as per-row protein keys and sequences.
Target-free metrics stay on a molecule table keyed by `molecule_key`.
Pair-level scorers join molecule fields at worker preparation time.
A wide per-molecule view, with one column group per target, is a derived pivot for selectivity combiners and reports.
The wide view is never the storage form, because sparse panels and partial target completion resume poorly in wide form.

### 21.3 Execution consequences

Chunks group by target.
A worker loads one target state and scores only pairs for that target.
Target state caching already keys by target digest, so many targets share infrastructure without sharing batches.
Pending, fingerprints, and receipts key by pair and block instance, with the target digest inside the fingerprint as today.
Budgets, shortlists, and shortage rules declare whether they apply per target or jointly.
A selectivity shortlist is a joint selector over the wide view that then gates pair-level downstream scoring.
Reports publish per-target rankings plus any joint ranking the pipeline requested.
The simplest supported pattern remains one run per target with a later join.
Pair-table runs are preferred when joint selection, shared generation, or one lineage across targets matters.

## 22. REINVENT as a generator

REINVENT 4 is a SMILES language-model generator with reinforcement learning against a composed scoring function.
Its generators cover de novo design, LibInvent scaffold decoration, LinkInvent fragment linking, and Mol2Mol analogue generation.
Its run modes cover sampling, transfer learning, staged reinforcement learning, and standalone scoring.
A reinforcement learning run samples a batch of SMILES, scores each scoring component, transforms each raw value into a 0 to 1 reward, aggregates the rewards, applies a diversity filter, and updates the agent toward higher reward.
Configuration is a single TOML document naming the prior checkpoint, the generator, the scoring components, and the run mode.

REINVENT in sampling mode is a proposer under section 6.3.
REINVENT in reinforcement learning mode is a guided search under section 6.3.
It is an easier guided search than SyntheMol because scoring already happens in batches per epoch.
Version 1 integration treats one reinforcement learning run as one generator block.
The harness renders the TOML config from resolved block config.
Harness scorers plug in as scoring components through the micro-batching scoring service.
The block emits every sampled molecule as candidate rows with epoch, step, raw component values, transformed rewards, and diversity-filter outcomes as generator metrics.
A normal scorer pass after the block produces accepted evidence, exactly as for SyntheMol.
Splitting epochs across `engine.cycle` rounds is deferred until per-epoch checkpoint and resume behavior is qualified by a spike.
Prior training and transfer learning stay outside the harness and produce versioned checkpoint artifacts.
Reward transforms and aggregation inside the loop belong to REINVENT.
Final ranking outside the loop belongs to harness combiners over raw rescored metrics.
No value is normalized twice without both transforms being recorded.
For multi-target generation, the service returns raw per-target scores and each target becomes its own REINVENT scoring component, with any selectivity term declared as an explicit additional component.

## 23. Contract reuse decision

Chaperone is MIT licensed, so reuse is legally available with attribution.
The question is technical fit at five layers.
The layers are the handler schema, the block interface, the engine interface, the on-disk layout, and the container convention.

The handler schema does not fit as the canonical small-molecule contract.
Chaperone rows carry an ordered list of structure files with residue-level metadata plus a JSON metrics string.
Small-molecule screening rows carry SMILES identity, structured source locators, target identity, and typed metric columns at million-row scale.
Pair rows for multi-target work have no Chaperone analogue.
Forcing SMILES into structure metadata, or minting a structure file per row, would tax every operation to serve the minority of rows that ever gain a structure.

The engine and block interface shapes do fit and should be mirrored.
The mirrored surface includes `ExecutionEngine` with `run`, `submit`, `gather`, and `cycle`, portable `JobArgs`, modal `RuntimeArgs`, a string-name block registry, per-block resource declarations, site settings files, CLI override generation, and the filter expression language.
Mirroring means same concepts and, where sensible, same spellings, implemented against the candidate and pair tables.
It does not mean importing `chaperone-pipelines` as a runtime dependency in version 1.
That dependency is pre-production, installed from the repository, and pulls a structural-biology stack the coordinator must not carry.

A bridge remains valuable after structure-producing steps.
A bridge block can export a shortlisted subset with Boltz structures into a Chaperone handler to reuse Chaperone analysis blocks, then import their metrics back by row key.
The bridge is an ordinary Adapter at the sink and source seams.
Adopting the Chaperone handler wholesale should be reconsidered only if a benchmark spike shows million-row handler parquet, filtering, and tranche grouping within agreed cost of the native table.
Until that evidence exists, the decision is a new data contract with a Chaperone-shaped engine contract.


