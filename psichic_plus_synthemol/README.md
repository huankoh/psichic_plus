# Generate molecules with SyntheMol and PSICHIC+

Generate molecules with the complete native SyntheMol launcher at revision `60302339148339355fc3e568ce200ed06207f99c`.
The [versioned patch](patches/synthemol-6030233.patch) adds `psichic_plus` to its individual scorer factory and passes three optional PSICHIC settings.
SyntheMol retains ownership of reaction matching, MCTS, RL, score vectors, weights, caches, logging, and CSV output.
The [verification record](VERIFICATION.md) separates tested controls from known upstream limits.
The [shared image](../psichic_plus_workflows/README.md) also includes AdaptiveFlow screening.

## Install the patched upstream package

Start in the repository root with the verified PSICHIC+ environment and native ligand package installed.

```bash
python -m pip install -c psichic_plus_synthemol/constraints.txt -r psichic_plus_synthemol/requirements.txt
python -m psichic_plus_synthemol.install
python -m pip check
python -m psichic_plus_synthemol --help
```

The installer clones the exact revision into a temporary directory, checks the patch SHA-256 and `git apply --check`, and installs with `--no-deps`.
Use `--source /path/to/read-only/SyntheMol` to clone an existing checkout instead.
It never installs from or writes to that source checkout.
Bundled catalogue resources are excluded from the installed package and Docker image.
Supply your own authorized CSV and trusted reaction-map pickle files.
Native default resource paths therefore do not exist in this installation.
Keep this repository on Python's import path when using the installed `psichic_plus` scorer.

## Prepare building-block scores

Native generation requires one precomputed CSV column for every score, including PSICHIC+.
For a plain CSV, run the explicit preparation command first.

```bash
python -m psichic_plus_synthemol score-building-blocks \
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
python -m psichic_plus_synthemol \
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
Upstream can format a reaction ID as `7.0` when some rows lack that reaction step.
Identical products from different spaces share a native node and can retain only one route.

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
python -m psichic_plus_synthemol.example.verify \
  --model-dir /weights/PSICHIC_plus --output runs/synthemol-verified --device cpu
SYNTHEMOL_SOURCE=/path/to/pinned/SyntheMol python -m pytest tests/test_synthemol.py -q
```

The real verifier prepares the CSV, runs each space and both spaces together, checks native saves and provenance, and compares affinities with direct screening.
The tests also run a native two-step CLI route through REAL 22 and 7_2, whose first slot is empty in the catalogue.
Pinned baseline comparisons cover working MCTS and regression RL with explicit `rdkit` fingerprints, plus preserved upstream failures.

## Licenses

PSICHIC+ code and weights remain subject to this repository's [non-commercial research license](../LICENSE).
SyntheMol retains its original MIT license and copyright notices in the installed distribution.
See the [patch notice](patches/NOTICE.md).
