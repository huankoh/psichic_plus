# Generate molecules with SyntheMol and PSICHIC+

Use PSICHIC+ predicted affinity to guide SyntheMol's MCTS search for one target.
The adapter uses upstream SyntheMol at `60302339148339355fc3e568ce200ed06207f99c` and its unmodified generator and CSV writer.
It supports Enamine REAL and WuXi GalaXi reaction definitions.
The [shared Docker image](../psichic_plus_workflows/README.md) contains PSICHIC+, this adapter, and the AdaptiveFlow adapter.

## Install in an existing PSICHIC+ environment

Start with the repository's verified screening environment and installed native ligand package.
Install the additional dependencies under constraints that preserve the PSICHIC+ stack.
Install upstream source separately so its dependency resolution cannot replace that stack.

```bash
python -m pip install -c psichic_plus_synthemol/constraints.txt -r psichic_plus_synthemol/requirements.txt
python -m pip install --no-deps 'synthemol @ git+https://github.com/swansonk14/SyntheMol.git@60302339148339355fc3e568ce200ed06207f99c'
python -m pip check
```

## Generate from your catalogue

Supply a single-record target FASTA, authorized weights containing `config.json`, `degree.pt`, and `model.pt`, and your own catalogue files.
Read the [catalogue contract](#catalogue-contract) before preparing the reaction map.
Use a new output directory for each run.

```bash
python -m psichic_plus_synthemol \
  --target-fasta /inputs/target.fasta \
  --model-dir /weights/PSICHIC_plus \
  --building-blocks /inputs/building_blocks.csv \
  --reaction-map /inputs/reaction_to_building_blocks.pkl \
  --chemical-space real \
  --output runs/generation \
  --rollouts 100 \
  --max-reactions 1 \
  --seed 0 \
  --device cpu
```

For WuXi, supply its map and use `--chemical-space wuxi`.
For a compatible CUDA environment, use `--device cuda:0`.
The adapter loads one model and target per run and scores one SMILES at a time.
Set `TORCH_HOME` to an existing ESM cache to avoid downloading ESM on first use.

Read `molecules.csv` for generated products in descending affinity order.
The upstream writer retains `smiles`, `score`, `psichic_affinity`, rollout and node statistics, `num_reactions`, and the reaction and building-block columns for every step.
Both `score` and `psichic_affinity` contain raw PSICHIC+ predicted affinity, where larger values are preferred.
`run.json` records the search settings and pinned upstream revision.
These predictions and proposed routes do not establish experimental binding or synthesis success.

## Catalogue contract

The CSV requires `reagent_id,smiles` columns.
IDs remain strings, including leading zeros.
Empty values, duplicate IDs, duplicate exact SMILES, and invalid SMILES cause an error.
ID `-1` is reserved for upstream intermediate reactants and causes an error.
SMILES identities are exact input strings, without canonicalization or deduplication across equivalent representations.

The reaction map is a trusted Python pickle with shape `dict[str, dict[int, list[str] | set[str]]]`.
Loading a pickle can execute code, so use only a file from a source you trust.
Each outer key is an upstream reaction ID string, such as `"22"` or `"1"`.
Include exactly every reaction in the chosen upstream chemical space, including its sub-reaction IDs.
Each reaction contains exactly its zero-based integer reactant slots.
The loader normalizes slot lists to sets.
Every mapped SMILES must appear in the CSV and match the slot's upstream SMARTS query.
A reaction with any empty slot is excluded from search, including reactions that intermediates could otherwise complete.
At least one reaction must have a building block in every slot.
Do not commit licensed catalogue data to this repository.

SyntheMol scores building blocks, intermediate combinations, and generated products during search.
Every scored SMILES must meet the existing PSICHIC+ ligand contract, including its molecular-weight limits.
A predictor failure stops generation with the SMILES, failure stage, and reason.
The adapter never substitutes an affinity for a failed prediction.
The exact-SMILES cache contains only successful finite raw affinities.
A run with no reaction products fails explicitly.

## Check the open example

The example uses `CNc1ccccc1`, `O=C(O)c1ccccc1`, and target sequence `MKTAYIAKQRQISFVKSHFSRQDILD`.
Only REAL reaction 22 or WuXi reaction 1 has populated slots.
The committed pickles contain only these two open example molecules and empty slots for other reactions.
Regenerate them with `python -m psichic_plus_synthemol.example.make_maps` after installing the pinned upstream source.

```bash
python -m psichic_plus_synthemol.example.verify \
  --model-dir /weights/PSICHIC_plus \
  --output runs/synthemol-verified \
  --device cpu
python -m pytest tests/test_synthemol.py -q
```

The verifier runs both generation CLIs, checks route columns and leading-zero IDs, and compares each product with the unmodified screening CLI at batch size one.
It also sends an undersized ligand through the real scorer and requires a visible rejection.
The behavior tests separately prove that changing predicted affinities changes the branch visited by upstream MCTS.
See [verification evidence](VERIFICATION.md) for executed commands, results, and hardware limits.

## Licenses

PSICHIC+ code and weights remain subject to the repository's [non-commercial research license](../LICENSE).
Supply authorized weights at runtime.
SyntheMol retains its [upstream MIT license](https://github.com/swansonk14/SyntheMol/blob/60302339148339355fc3e568ce200ed06207f99c/LICENSE) in the installed distribution.
This adapter imports upstream code and does not copy its implementation.
