# Boltz-2 scoring reference

## Native targets

`--target-yaml /inputs/target.yaml --ligand-id B` replaces the designated ligand's `smiles` or `ccd` with each candidate's original SMILES.
The ligand chain ID, other entities, protein modifications, pocket and contact constraints, and templates remain in the document.
An affinity property is added only when none exists.
An existing affinity property must already refer to that designated ligand.
There is no implicit protein truncation, cofactor removal, constraint generation, or template generation.

The following native target illustrates an optional cofactor and pocket constraint supplied by the user.

```yaml
version: 1
sequences:
  - protein:
      id: A
      sequence: MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ
      msa: target.a3m
  - ligand:
      id: B
      smiles: CCO
  - ligand:
      id: C
      ccd: MG
constraints:
  - pocket:
      binder: B
      contacts: [[A, 5], [A, 10]]
      max_distance: 6
properties:
  - affinity:
      binder: B
```

Local MSA and CIF/PDB template paths resolve relative to the source YAML, then are copied into the run's `boltz/assets/` directory.
Generated inputs point to those copies.
Asset hashes and source paths are recorded.
Use native CSV MSAs for paired multichain input as described by upstream.
Single-sequence mode retains every chain and sets each protein's MSA to `empty`.
It rejects a target that already supplies a local MSA instead of silently discarding that MSA.
Server mode allows upstream to request missing MSAs remotely.
Boltz 2.2.1 rejects mixing local MSAs with automatically generated MSAs in one target, so the integration rejects that combination before execution.
Use local MSAs for every protein, or omit local MSA paths when requesting automatic generation.

For PSICHIC+ modes, a target with multiple protein chain IDs requires `--psichic-chain`.
PSICHIC+ scores the selected protein sequence alone.
Boltz receives the entire native complex.
This distinction is recorded in the manifest and matters when interpreting an experimental ensemble.

The adapter rejects ambiguous candidate ligand entities, duplicate chain IDs, duplicate YAML mapping keys, mismatched affinity binders, missing assets, and unsupported schema versions.
A covalent bond constraint to the replaced ligand is rejected because upstream requires CCD atoms for bonds and candidates use SMILES.
Other native chemistry, contacts, atom naming, and geometry validation remains upstream's responsibility.
Upstream rejections are reported as failed rows with retained native logs.

## Native CLI options

The authoritative option list is the installed `boltz predict --help` for version 2.2.1.
Each run saves that help in `boltz/native_help.txt`.
Pass options after `--`, for example `-- --accelerator gpu --recycling_steps 5 --diffusion_samples 3 --output_format pdb --seed 42`.
Sampling, recycling, affinity sampling, checkpoints, caches, potentials, devices, embeddings, and output format remain native options.
Their upstream defaults are unchanged unless explicitly passed.
The integration controls only the input directory, `--out_dir`, `--model boltz2`, and `--use_msa_server` through `--msa-mode server`.
It rejects passthrough `--override`, `--help`, and an additional `--` because runs require fresh outputs and predictions.
Both `--option value` and `--option=value` forms are checked.

The integration invokes the `boltz` executable beside `--boltz-python` and checks that environment's package version.
It removes the coordinator's `PYTHONPATH`, `PYTHONHOME`, and environment-specific library directory from the child environment.
The native process otherwise retains user environment settings.
Passwords and API-key values supplied as native options are redacted from saved command metadata.
Environment-based credentials remain available to the native CLI.
Native output logs remain upstream's output.

## Ranking and selection

`affinity_probability_binary` is the default Boltz ranking metric.
Larger values rank first and represent the upstream binder-probability head intended for hit discovery.
`--rank-by affinity_pred_value` uses the native affinity head, where smaller values rank first.
Upstream defines this value as `log10(IC50)` with IC50 measured in micromolar and recommends it for comparing known binders.
PSICHIC+ uses its raw affinity with larger values ranked first.
Neither engine's raw score is converted into the other's units.

Ties use the average of their occupied ranking positions, with rank 1 best.
Ties use exact numeric equality without rounding.
For ensembles, each engine is ranked again within the cohort of rows that succeeded on both engines.
The experimental combined value is `(psichic_weight * psichic_rank + boltz_weight * boltz_rank) / (psichic_weight + boltz_weight)`.
Smaller combined values rank first, with average final ranks for ties.
Weights default to 1 each, must be nonnegative and finite, and must have a positive finite sum.
A zero-weight engine must still succeed for a row to enter the common cohort.

Pyramid modes select exactly `--shortlist-size` successful PSICHIC+ rows.
Descending raw affinity determines selection, and input row order resolves a boundary tie.
If fewer than that many rows succeed, Boltz is not started and the report records the shortage.
Boltz failures do not trigger replacement candidates.
All inputs remain in the report with their selection decisions and failure reasons.

## Output and failure contract

Each row has an internal key derived from its input row number.
IDs remain strings, including blank IDs, duplicate IDs, `None`, `nan`, and leading zeroes.
User IDs and SMILES are never used as filenames or mapping keys.
Original SMILES, submitted representations, independent engine scores, ranks, selection, and failures are kept separately.
The current PSICHIC+ result API does not expose its normalized SMILES, so that field is null.
Boltz's processed native artifacts remain available for inspection of its representation.

Native affinity files are read only from `boltz/native/boltz_results_inputs/predictions/<row_key>/affinity_<row_key>.json`.
The complete JSON object, including per-head values, is retained alongside native structures and confidence files.
Missing files, malformed JSON, non-finite scores, boolean scores, and out-of-range probabilities are failures.
Valid per-row outputs survive a nonzero native exit, but the run is marked partial.
Failures never become zero scores or weak ranks.
Unranked rows have a null final rank and an explicit reason.

Exit code 0 means the requested execution completed without failed rows.
Intentional shortlist exclusions do not cause failure.
Exit code 1 means partial execution, failed rows, or an insufficient shortlist.
Exit code 2 means invalid configuration or an unusable environment.
`--boltz-timeout` bounds the native prediction process and kills its process group on timeout.
Interrupted or crashed coordinators may leave an unfinished manifest, which must not be interpreted as a completed run.
Every rerun requires a new output directory to prevent stale predictions from being reused.

The manifest records the coordinator Git revision when available and the optional `PSICHIC_WORKFLOW_BUILD_REVISION` supplied at image build time.
PSICHIC+ provenance separately records the resolved screening directory, hashes of its Python sources, and the actual loaded module paths and hashes, including the model, featurizers, and ligand package.
These source fingerprints also identify overrides selected through `PSICHIC_SCREENING_DIR` and builds without Git metadata.
The build revision describes the coordinator build and does not substitute for the PSICHIC+ source fingerprints.

## Scope and scientific limits

No target-specific screening improvement has been demonstrated by these workflow tests.
Shortlist recall, enrichment, failure rates, runtime, and resource use require an independent labeled evaluation for the intended target.
This integration does not implement FEP or ABFE.
It does not add per-molecule scoring to AdaptiveFlow's distributed collection selector.
No cluster or cloud compatibility is claimed.

## Upstream sources

- [Boltz 2.2.1 prediction inputs, options, and outputs](https://github.com/jwohlwend/boltz/blob/v2.2.1/docs/prediction.md).
- [Boltz 2.2.1 CLI implementation](https://github.com/jwohlwend/boltz/blob/v2.2.1/src/boltz/main.py).
- [Boltz 2.2.1 native output writer](https://github.com/jwohlwend/boltz/blob/v2.2.1/src/boltz/data/write/writer.py).
