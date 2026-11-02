# Native SyntheMol fidelity verification

This record supersedes the earlier verification descriptions on this branch.
Those descriptions validated a reduced wrapper that filtered reactions with empty slots and used a separate score-vector wrapper.
They do not establish native launcher parity.
The current implementation deletes that wrapper and uses the pinned upstream launcher with a scalar scorer-factory extension.
No upstream search correctness fixes are included.
Two fresh code reviewers found no actionable issues in upstream fidelity.
A separate worker reran the real workflows and native parity checks as recorded below.

## Fidelity and limits

| Control or behavior | Executed evidence | Result or limit |
| --- | --- | --- |
| Every original `generate` parameter, annotation, default, and CLI flag | AST signature comparison and real `--help` against a temporary clone of revision `6030233` | Preserved, including underscore names and `n_rollout=10`; only three optional PSICHIC parameters added |
| MCTS, REAL, WuXi, and combined spaces | Native CLI tests and real model verifier | Separate and combined runs succeed; identical SMILES can retain one upstream route |
| Custom ID, SMILES, and score columns | Native CLI with leading-zero IDs and renamed columns | IDs and route columns preserved |
| `save_frequency`, `store_nodes`, verbose output, local logging, and stats | Two one-rollout chunks, `logs.pkl`, `generation_stats.csv`, and native CSV | Executed through the full launcher |
| PSICHIC preparation and product affinity | Real weights, batch size one, direct screening CLI comparison | Five host comparisons have maximum absolute difference `0.0` |
| Precomputed vectors, exact-SMILES cache, and lazy initialization | Native `MoleculeScorer` with a predictor boundary fixture | Supplied vectors win; cached molecules trigger no prediction; predictor initializes once on an uncached call |
| Signs, base weights, and score composition | Native CLI single-score negative sign and native mixed-score scorer with reweighting | No adapter vector or weight wrapper |
| Intermediate-only reaction | Actual native CLI with REAL 22 followed by 7_2 | Two-step product retained despite empty slot zero in 7_2; second route step uses intermediate ID `-1` |
| Seeded native parity | Baseline and installed launcher CSV bytes and non-time stats | MCTS, two-step generation, explicit regression RL, and mixed-score baseline comparisons |
| PSICHIC factory parity | Installed PSICHIC CLI with an explicit descriptor predictor fixture versus original native descriptor scorer | Same seeded two-step CSV; fixture scores are descriptors, not claimed binding affinities |
| Regression RL | Explicit `mlp`, `rdkit`, `regression`, one training epoch per rollout | Baseline comparison and real PSICHIC host run; no claim for other RL combinations |
| Dead ends from incomplete routes | Actual CLI against original and patched package | Same native failure with and without another complete route; no filtering, retry, skip, or fallback |
| `qed` | Baseline and patched CLI | Preserved upstream factory fallthrough raises `ValueError` |
| Default RL fingerprint | Baseline and patched CLI | Preserved `FEATURES_SIZE_MAPPING[None]` raises `KeyError` |
| Replication | Baseline and patched `replicate_mcts` CLI | Preserved integer/string map-key mismatch raises `KeyError`; `replicate_rl` not independently exercised |
| Mixed-objective search | Native baseline comparison and source inspection | Upstream adds scalar property rewards to every accumulator component; output parity does not establish correct per-objective exploitation |
| W&B, wavelength, solvents, pretrained RL and other score models, custom reaction space, extended RL evaluation | Signature and flag parity only | Exposed, runtime behavior unverified |
| CUDA | A100 real SyntheMol and AdaptiveFlow workflows, plus same-input repeated inference | SyntheMol passed once; AdaptiveFlow completed but exceeded direct-score tolerance twice; see GPU results below |
| Default catalogue paths | Resource directory removed before wheel installation | Users must supply authorized CSV and map files |

## Repeat host verification

Commands run from the repository root use an existing verified environment, authorized model weights, and ESM cache.
Set the paths for your machine.
Use new run paths when repeating these commands.
The reference source is read only.
The installer and baseline tests clone temporary copies instead of installing from that reference.

```bash
old=/Users/huanyeekoh/.t3/scratch/2026-10-27-https-github-com-huankoh-psichic-7a152a8a
python="$old/work-scratch/fresh-env/bin/python"
weights="$old/work-impl/psichic_plus_screening/pretrained_weights/PSICHIC_plus"
export TORCH_HOME="$old/work-scratch/torch" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
"$python" -m psichic_plus_synthemol.install --source "$old/refs/synthemol"
"$python" -m pip check
"$python" -m psichic_plus_synthemol.example.verify \
  --model-dir "$weights" --output runs/parity-implementation/host-real --device cpu
SYNTHEMOL_SOURCE="$old/refs/synthemol" AFVS_UPSTREAM="$old/refs/af-vs" \
  "$python" -m pytest tests psichic_plus_screening/tests psichic-ligand/tests -q \
  --basetemp runs/parity-implementation/pytest-final
ruff check psichic_plus_synthemol tests/test_synthemol.py --ignore E402
"$python" -m compileall -q psichic_plus_synthemol
git diff --check
```

The final full regression passed 485 tests, with three opt-in root E2E tests skipped.
Running those three tests separately with `PSICHIC_E2E=1` passed with real weights.
Ruff, compilation, `pip check`, and `git diff --check` passed.
The installer rejected a changed patch digest before attempting to access the source.
The actual CLI rejected an unprepared CSV with the same missing-score-column contract as upstream.
The reference checkout's tracked source remained clean.

The real host verifier produced `CN(C(=O)c1ccccc1)c1ccccc1` in all three generation runs.
Preparation retained IDs `0001` and `0002` and raw affinities `3.6013894081115723` and `4.581860542297363`.
Product raw affinity was `1.6210767030715942`.
The real PSICHIC regression RL run trained after each rollout and generated the same product with zero difference from direct inference.
Its command and result are in `run-host-rl.sh` and `host-rl/verification.json` under the artifact directory.
The predictor rejected `C` with stage `ligand_standardization` and reason `NON_SMALL_MOLECULE: Molecular weight outside allowed range`.
Artifacts reside under `runs/parity-implementation`, including commands, logs, CSVs, and JSON summaries.
The decision trail continues in `runs/implementation/decisions.tsv`.

## Repeat Docker verification

The shared Dockerfile uses the same checked patch installer and excludes licensed catalogue resources and weights.
AdaptiveFlow-VS remains unchanged at `df74e005e456e94654fa93a40d748de399450b81` under `/opt/AdaptiveFlow-VS`.
The image targets CPU `linux/amd64` on the arm64 host to preserve the native ligand platform contract.
The public Docker configuration avoids the credential helper without changing the user's configuration.

```bash
docker --config "$PWD/runs/implementation/docker-config" \
  --host unix:///Users/huanyeekoh/.docker/run/docker.sock \
  build --platform linux/amd64 --progress=plain \
  -f psichic_plus_workflows/Dockerfile -t psichic-workflows:parity .
bash runs/parity-implementation/run-containers.sh
```

The container script mounts only weights, the ESM cache, and writable run artifacts.
Both workflow modules run from the rebuilt image, serially, without host source mounts.
See `runs/parity-implementation/docker-build.log`, `container-synthemol.log`, `container-adaptive.log`, and the per-workflow `verification.json` records.

The tested image is `psichic-workflows:parity`, with ID `sha256:2d0a9262b2fceb13db624a71a91ac8960a90b1c4d39ba86967989efba8bbc060`.
Both real workflow verifiers completed serially from this image.
SyntheMol completed five affinity comparisons with maximum absolute difference `4.440892098500626e-16`.
AdaptiveFlow completed five comparisons with maximum absolute difference `0.0`, preserved failed-ligand reporting, and retained the selected work list.
Runtime inspection confirmed no installed SyntheMol resource directory, a retained MIT license file, no broken requirements, and a clean AdaptiveFlow checkout at the pinned revision.
The image contains the runtime code tested here.
This final verification record was updated on the host after those executions.

## Independent fidelity review and runtime checks

Review and runtime verification used commit `bdbb9228ec0a7bf4bb60256b4c0d424a4772bda2`.
Both code reviewers found no actionable critical, warning, or nit findings.
They confirmed all 43 original launcher parameters retain their annotations and defaults, with only three optional PSICHIC flags added.
They found no unintended reaction pruning, input-validation changes, vector replacement, or change in precomputed-score precedence.
The original and patched upstream packages differ only in the three extension files listed in the checked patch.

A fresh worker passed all 31 focused tests and reran the real image workflows serially without host source mounts.
SyntheMol completed separate REAL, WuXi, and combined runs with five direct-affinity comparisons and maximum absolute difference `4.440892098500626e-16`.
AdaptiveFlow used bundled `/opt/AdaptiveFlow-VS`, passed five direct comparisons with zero difference, and preserved its selected work list.
Explicit host MLP/RDKit regression RL and a same-host repeated-request check also passed.
The independent worker did not repeat the full regression suite; the earlier logs support 485 regression passes and three separate real root E2E passes.
Evidence is under `runs/parity-independent`, including `run-containers.sh`, `run-focused.sh`, `focused.log`, and each container's `verification.json`.

The two-step fixture enables REAL `7_2` after reaction 22 even though its first catalogue slot is empty.
Its intermediate retains ID `-1`.
The unchanged upstream logger drops the sub-reaction suffix and writes reaction `7`, displayed as `7.0` in the CSV.
Original and patched output CSVs are byte-identical, so preserving upstream output also preserves that loss of detail.
See `runs/parity-independent/route-audit.json`.

An exploratory comparison between host and image predictions exceeded an assumed `1e-5` tolerance.
Absolute differences were `0.00021982192993164062` for the acid and `0.000047087669372558594` for the product.
The cause was not isolated.
That driver exited nonzero after its successful RL check; a separate comparison against direct inference in the same host environment passed.
Affinity agreement claims therefore apply within the same runtime and do not imply exact equality across environments.
See `runs/parity-independent/cross-environment-affinities.json`, `host-rl-verification.json`, and `real-repeat-host.json`.

The independently tested image predates final README and verification-record edits.
A documentation-only rebuild refreshes those bundled files without changing the reviewed runtime code or upstream patch.
Its build and file comparison are recorded under `runs/parity-independent`.

## A100 and A40 checks

GPU verification tested commit `bbf0b66aa4eacf6c52f300179b376bac536a780a` on an NVIDIA A100-PCIE-40GB.
The host driver was `550.90.07`.
The container used PyTorch `2.6.0+cu124`, CUDA `12.4`, PyG `2.7.0`, torch-scatter `2.1.2+pt26cu124`, and RDKit `2024.03.3`.
Forward hooks confirmed that the model, ligand inputs, and protein inputs were on `cuda:0`.
A source audit found no mismatches against the tested commit, and the rebuilt native ligand package passed all 2,242 contract cases.

SyntheMol preparation and separate REAL, WuXi, and combined generation passed its CUDA verifier once.
Five direct-inference comparisons had maximum absolute difference `4.76837158203125e-06`, below the existing `1e-5` threshold.
The checked routes were preserved and the undersized ligand was visibly rejected.

AdaptiveFlow completed prescreen and primary screening in two unchanged CUDA runs, but both verifiers failed their score-comparison assertion.
For `C1CCCCCCC1`, differences from direct inference were `2.002716064453125e-05` and `4.1484832763671875e-05`, exceeding `1e-5`.
Independent output checks passed for the selected work list, ranking, and failure details.
This is not a full AdaptiveFlow CUDA verification pass.
The tolerance was not relaxed and no scores or search behavior were changed.

Ten calls to the same evaluation-mode model with byte-identical tensor inputs produced scores spanning `5.1975250244140625e-05`.
This demonstrates repeated-inference variation in that CUDA runtime.
The responsible operation was not identified, and CUDA reproducibility remains unresolved.

The shared CUDA image was `sha256:1a0e08743d9b849bf89ce64a76b19cbf300825edf796fbc94280e4bb1faabc24`.
It extended a pre-existing authorized CUDA base after refreshing its screening code and rebuilding the native package to match the tested commit.
The original base already contained weights in an inherited layer.
No new external weights were added, but this derived image is not a weights-free distributable image.
Authorized test weights and ESM cache were mounted read-only.
The repository's CUDA environment library path was applied after reproducing an initial RDKit import-order failure in the old base.
No tracked runtime code was changed for the GPU tests.

The A40 server passed a read-only CUDA tensor smoke test.
Its main data filesystem had only 925 MB free, so neither full workflow nor a large image transfer was attempted there.
A40 workflow verification remains pending.

All local GPU artifacts are under `runs/gpu-verification/a100-bbf0b66-xAzrhM4J`.
`report.md` links the environment and source audits, exact build and run scripts, original failing verifier logs, and repeated-input diagnostic.
`outputs/synthemol/verification.json` is the SyntheMol verifier's own successful result.
`outputs/adaptive-independent-verification.json` and `outputs/adaptive-repeat-independent-verification.json` summarize outputs from the failed AdaptiveFlow verifiers.
`outputs/identical-input-diagnostic.json` records the ten identical-input predictions.
These results supersede the earlier absence of CUDA runtime evidence, without changing the CPU verdicts.
