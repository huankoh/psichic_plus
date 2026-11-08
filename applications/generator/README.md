# Generate molecules with SyntheMol and PSICHIC+

Generate molecules with the complete native SyntheMol launcher at revision `60302339148339355fc3e568ce200ed06207f99c`.
The [versioned patch](patches/synthemol-6030233.patch) adds `psichic_plus` to its individual scorer factory and passes three optional PSICHIC settings.
SyntheMol retains ownership of reaction matching, MCTS, RL, score vectors, weights, caches, logging, and CSV output.
The [verification record](VERIFICATION.md) separates tested controls from known upstream limits.
The [shared image](../README.md) also includes AdaptiveFlow screening.

## Install the patched upstream package

Start in the repository root with the verified PSICHIC+ environment and native ligand package installed.

```bash
python -m pip install -c applications/generator/constraints.txt -r applications/generator/requirements.txt
python -m applications.generator.install
python -m pip check
python -m applications.generator --help
```

The installer clones the exact revision into a temporary directory, checks the patch SHA-256 and `git apply --check`, and installs with `--no-deps`.
Use `--source /path/to/read-only/SyntheMol` to clone an existing checkout instead.
It never installs from or writes to that source checkout.
Bundled catalogue resources are excluded from the installed package and Docker image.
Supply your own authorized CSV and trusted reaction-map pickle files.
Native default resource paths therefore do not exist in this installation.
Keep this repository on Python's import path when using the installed `psichic_plus` scorer.

Install the workflow CLI separately to rank existing products without SyntheMol.
This command installs no model or native generator dependency.

```bash
python -m pip install --no-deps --no-build-isolation ./applications
python -m applications.generator rank --help
```

The wheel includes the generator adapter, checked patch, installer inputs, and open example resources.
Prepare the [shared scoring dependencies and offline caches](../scoring/CLI.md) before final inference.
The native generator remains optional when you rank an already-produced file.

## Prepare building-block scores

Native generation requires one precomputed CSV column for every score, including PSICHIC+.
For a plain CSV, run the explicit preparation command first.

```bash
python -m applications.generator score-building-blocks \
  --input /inputs/building_blocks.csv \
  --output /inputs/scored_blocks.csv \
  --target-fasta /inputs/target.fasta \
  --model-dir /weights/PSICHIC_plus \
  --cache-dir runs/block-cache \
  --device cpu
```

The command adds `score` with one finite raw affinity per row using the same predictor as generation.
It preserves IDs as strings, all other columns, row order, and existing score columns with other names.
Use `--smiles-column` and `--score-column` for custom column names.
An existing output file or destination score column causes an error unless you supply `--overwrite-output` or `--overwrite-score-column`, respectively.
Prediction failures stop preparation without writing placeholder scores.
All predictions finish before the output CSV is opened.

## Run native generation

Supply a single-record target FASTA and authorized weights containing `config.json`, `degree.pt`, and `model.pt`.
Set `TORCH_HOME` to an existing ESM cache, or allow the first run to download ESM.

```bash
python -m applications.generator \
  --search_type mcts \
  --save_dir runs/generation \
  --score_types psichic_plus \
  --score_model_paths /weights/PSICHIC_plus \
  --score_names psichic_affinity \
  --psichic_target_fasta /inputs/target.fasta \
  --building_blocks_paths /inputs/scored_blocks.csv \
  --reaction_to_building_blocks_paths /inputs/real_map.pkl \
  --chemical_spaces real \
  --n_rollout 100 \
  --max_reactions 2 \
  --rng_seed 0 \
  --save_frequency 10 \
  --store_nodes
```

Generation uses upstream underscore option names and defaults, including `n_rollout=10`.
`--psichic_device` defaults to `cuda` when native `--use_gpu` is enabled and to `cpu` otherwise.
`--psichic_cache_dir` defaults to `save_dir/psichic_cache`.
The model and target initialize lazily on the first prediction outside the CSV cache.
The predictor scores one SMILES at a time with batch size one.
PSICHIC+ returns raw affinity, with larger values preferred by the default positive sign.
Native `--score_signs`, `--base_score_weights`, and `--success_thresholds` retain their upstream behavior.
Multi-objective search has a known upstream accumulator defect described in the verification record.

For combined spaces, pass matching ordered lists to `--chemical_spaces real wuxi`, `--building_blocks_paths`, and `--reaction_to_building_blocks_paths`.
Use `--building_blocks_id_column`, `--building_blocks_smiles_column`, and `--building_blocks_score_columns` for custom CSV columns.
For multiple scores, supply the full ordered vector in CSV columns and align all native per-score lists.
Use literal `None` entries in `--score_model_paths` for non-model scores.
The original score types and all RL, replication, solvent, wavelength, and W&B flags remain exposed.
Exposing a flag does not establish that its upstream combination works.

Read `molecules.csv`, `generation_stats.csv`, and `logs.pkl` in the save directory.
The native writer retains score names, node statistics, reaction IDs, chemical spaces, and building-block IDs for each step.
An intermediate has building-block ID `-1`.
Reserve that ID for intermediates in your catalogue.
Upstream logs reaction `7_2` as `7`, and the CSV can display it as `7.0`.
That native output loses the sub-reaction suffix; the adapter preserves this behavior.
Identical products from different spaces share a native node and can retain only one route.

## Rank completed native products

Use an explicit import after you have confirmed that native generation finished.
`--source-run` reads that directory's `molecules.csv`.
The native CSV has no trustworthy process-completion marker, so this command records your completion claim rather than claiming to verify the earlier process.
It rejects malformed or empty product populations and detects source changes during import.
Run it after the native writer has stopped.

```bash
psichic-generator rank \
  --source-run runs/generation \
  --fasta /inputs/target.fasta --msa-mode single \
  --mode psichic-only --model-dir /weights/PSICHIC_plus \
  --esm-checkpoint /cache/esm2_t33_650M_UR50D.pt \
  --output runs/final-ranking --validate-only
```

Remove `--validate-only` to import and score the completed products.
The Python module equivalent is `python -m applications.generator rank` with the same arguments.
All scoring options come from the [shared rank interface](../scoring/CLI.md).
Defaults are native `node_id` and `smiles` columns, with explicit `--id-column` and `--smiles-column` overrides.
The importer retains exact strings, row order, duplicate IDs, duplicate structures, and every native output column.
Each row has a bounded internal key, its original file and row locator, and the native byte digest.
User IDs never become artifact paths.

The final stage rescores products because native search scores lack the full model, batch, loaded-source, and same-pass receipt provenance.
It never infers alpha or beta from a search affinity or joins values from separate attempts.
Compatible accepted receipts resume through the shared durable contract.
Native search scores, routes, and building-block columns stay in metadata and in `inputs/molecules.csv`.
The importer cannot recover routes or sub-reaction identifiers already lost by native deduplication or serialization.
It never overwrites the source `molecules.csv`.

Choose `boltz-only`, `full-ensemble`, `pyramid-ensemble`, or `pyramid-refinement` to opt into Boltz final scoring.
These choices rank the completed products and do not change search.
Refinement with binder probability remains the suggested starting workflow.
Supply K explicitly for either pyramid mode.

```bash
psichic-generator rank \
  --source-run runs/generation \
  --fasta /inputs/target.fasta --msa-mode single \
  --mode pyramid-refinement --shortlist-size 100 \
  --rank-by affinity_probability_binary \
  --output runs/refined-products
```

Model paths may use the shared documented environment variables.
An explicitly chosen ensemble defaults to `concentration-v1`, which requires same-pass PSICHIC distribution evidence and Boltz affinity.
`--ensemble rank-v1` retains weighted average rank support.
Incompatible metrics and manual concentration weights fail before inference.
`--psichic-output distribution` changes displayed detail, while concentration ensembles always retain the required distribution evidence.
`--preset normal` remains the default.
Use `--preset quick-check` explicitly for reduced sampling integration checks.
Only single-sequence MSA is supported and no sequence goes to a remote MSA service.

## Generate and score automatically

The opt-in wrapper runs the complete native launcher in a separate owned process.
The native parser accepts its existing underscore flags and defaults after `--native-args`.
The wrapper requires a fresh native `--save_dir` and a search objective that includes `psichic_plus`.
Native signs, weights, caches, and search remain native behavior.
The separate final scoring mode never supplies Boltz feedback to search.

```bash
psichic-generator generate-and-rank \
  --fasta /inputs/target.fasta --msa-mode single \
  --mode pyramid-refinement --shortlist-size 100 \
  --output runs/generation-workflow \
  --native-args \
  --search_type mcts --save_dir runs/new-generation \
  --score_types psichic_plus \
  --score_model_paths /weights/PSICHIC_plus \
  --score_names psichic_affinity \
  --psichic_target_fasta /inputs/target.fasta \
  --building_blocks_paths /inputs/scored_blocks.csv \
  --reaction_to_building_blocks_paths /inputs/real_map.pkl \
  --chemical_spaces real --n_rollout 100 --rng_seed 0
```

Put shared native Boltz overrides after `--` and before `--native-args`, for example `-- --accelerator gpu --devices 1 --native-args ...`.
The generation and selected final-scoring protein sequence must agree.
For a final PSICHIC stage, the resolved cached ESM checkpoint also configures the native child.
For Boltz-only final scoring, configure the native search checkpoint with `PSICHIC_ESM_CHECKPOINT`.
Prepare every required cache offline before execution.
The wrapper starts no installer or downloader and denies network access in its native child.
Use an offline configuration for optional native services such as W&B.
The original native launch and `score-building-blocks` commands keep their behavior.

The workflow saves native logs and process identities under `runs/generation-workflow/generation`.
It observes the actual exit status, retires the entire owned process group, and binds a product snapshot digest to successful completion.
Only then can the final stage start in `runs/generation-workflow/scoring`.
This process boundary retires the search predictor before final inference can use the allocation.
Cancellation or generation failure stops dependent scoring and retains logs.
The pipe guard and group guardian stop owned descendants if the coordinator disappears.
No unrelated process is signaled.

Add `--validate-only` before `--native-args` to preflight the future workflow without generation, inference, or output creation.
Future validation checks native argument parsing, the target, scoring dependencies, and resolved settings.
It reports unknown product counts rather than validating a population that does not yet exist.
Existing-run validation parses the actual population and reports its row count.
The automatic boundary proves an observed successful native process, not experimental binding, synthesis success, or upstream search correctness.

## Read and recover final results

Human output identifies the native source and explains that the report ranks completed products with search unchanged.
The scoring directory contains `generation.json`, `generation.md`, the frozen native CSV, and the shared durable scoring state.
Read `latest.json` for immutable ranked `summary.md`, `results.csv`, and `results.json` paths.
The full reports retain original source locators and all native metadata alongside final scores and accepted receipts.
Counts, failures, shortlist shortages, and singleton or tied populations use the shared reporting rules.

```bash
psichic-generator status --run runs/generation-workflow
psichic-generator resume --run runs/generation-workflow
psichic-generator retry --run runs/generation-workflow --failed --engine boltz
psichic resume --run runs/generation-workflow/scoring
```

The first three commands also accept a scoring directory created by an explicit import.
Resume never reruns generation, including after a completed final report.
If generation succeeded but final scoring was interrupted before initialization, resume uses the saved completion and product digest to initialize the final directory.
It refuses an unconfirmed, failed, cancelled, or modified generation output.
Changed scoring runtime or settings requires an explicit new import.
An incomplete scoring-directory creation is retained as evidence and needs a new scoring path through `rank --source-run`.
Retry of failed predictions remains explicit and preserves earlier attempts and reports.
Recovery is limited to the original POSIX host, paths, and qualified local filesystem assumptions of the shared executor.

`--json` emits one schema-versioned stdout envelope.
Native progress stays in logs and coordinator progress goes to stderr.
An automatic response includes the recoverable workflow and scoring directories even when final inference is interrupted.
Exit 0 means complete or validated, exit 1 means incomplete execution or cancelled generation, and exit 2 means invalid input or recovery configuration.
Use `rank --help` or `generate-and-rank --help` for the full scoring options.

## Use native catalogue and failure behavior

The default CSV columns are `reagent_id`, `smiles`, and `score`.
IDs retain leading zeros through native string loading.
Reaction maps have the upstream shape `dict[str, dict[int, set[str]]]`, keyed by reaction ID and zero-based reactant slot.
Use only trusted pickle files because loading a pickle can execute code.
The upstream loader requires entries for its selected reactions.
No adapter catalogue validator or reaction filter changes that contract.

Empty slots remain in the search because an intermediate can fill them later.
No fully populated route is required by the adapter.
Native dead ends can raise an error, and the adapter preserves that error.
It does not prune routes, retry searches, skip failures, or invent scores.
A PSICHIC failure includes the exact SMILES, failure stage, and reason.
Native precomputed full vectors take precedence over predictor inference, including supplied values that differ from model predictions.
Users are responsible for supplying valid, correctly ordered vectors.
These predictions do not establish experimental binding or synthesis success.

## Verify with the open example

The committed example contains two open molecules and maps with only REAL 22 or WuXi 1 populated.
It includes no licensed catalogue data.

```bash
python -m applications.generator.example.verify \
  --model-dir /weights/PSICHIC_plus --output runs/synthemol-verified --device cpu
SYNTHEMOL_SOURCE=/path/to/pinned/SyntheMol python -m pytest applications/generator/tests/test_synthemol.py -q
```

The real verifier prepares the CSV, runs each space and both spaces together, checks native saves and provenance, and compares affinities with direct screening.
The tests also run a native two-step CLI route through REAL 22 and 7_2, whose first slot is empty in the catalogue.
Pinned baseline comparisons cover working MCTS and regression RL with explicit `rdkit` fingerprints, plus preserved upstream failures.

## Licenses

PSICHIC+ code and weights remain subject to this repository's [non-commercial research license](../../LICENSE).
SyntheMol retains its original MIT license and copyright notices in the installed distribution.
See the [patch notice](patches/NOTICE.md).
