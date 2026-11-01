# SyntheMol and shared-image verification

Implementation started at `3305eb664f31aeaf7ccb6bc0b7b17f4956a31aa1` on `feat/synthemol-workflows`.
The checkout was `/Users/huanyeekoh/.t3/scratch/2026-11-01-t3-pstack-science-poteto-mode-a9c6cba2`.
The previous SyntheMol draft was inspected without changes.
The reference checkout's tracked source remains clean.
The host wheel installation refreshed its pre-existing ignored `synthemol.egg-info` metadata, contrary to the instruction to leave that checkout untouched.
No generated upstream metadata was manually restored because its original contents and timestamps were not recorded.
The existing `psichic_plus_adaptive.predictor.Predictor` was reused without changes.
No PR or push was performed.

## Repeat the host checks

The following paths name the actual environment, weights, and cache used during implementation.
Use a new output directory when repeating a verifier.

```bash
synthemol_repo=/Users/huanyeekoh/.t3/scratch/2026-11-01-t3-pstack-science-poteto-mode-a9c6cba2
synthemol_old=/Users/huanyeekoh/.t3/scratch/2026-10-27-https-github-com-huankoh-psichic-7a152a8a
synthemol_python="$synthemol_old/work-scratch/fresh-env/bin/python"
synthemol_weights="$synthemol_old/work-impl/psichic_plus_screening/pretrained_weights/PSICHIC_plus"
export TORCH_HOME="$synthemol_old/work-scratch/torch"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$synthemol_repo"
"$synthemol_python" -m pip install -c psichic_plus_synthemol/constraints.txt -r psichic_plus_synthemol/requirements.txt
"$synthemol_python" -m pip install --no-deps "$synthemol_old/refs/synthemol"
"$synthemol_python" -m pip check
"$synthemol_python" -m psichic_plus_synthemol.example.verify \
  --model-dir "$synthemol_weights" --output runs/synthemol-host --device cpu
AFVS_UPSTREAM="$synthemol_old/refs/af-vs" \
  "$synthemol_python" -m pytest tests psichic_plus_screening/tests psichic-ligand/tests -q
"$synthemol_python" -m pytest tests/test_synthemol.py -q
PSICHIC_E2E=1 "$synthemol_python" -m pytest tests/test_root_e2e.py -q
ruff check psichic_plus_synthemol tests/test_synthemol.py --ignore E402
"$synthemol_python" -m compileall -q psichic_plus_synthemol
git diff --check
```

The supplied SyntheMol reference checkout was at `60302339148339355fc3e568ce200ed06207f99c`.
The supplied AdaptiveFlow reference checkout was at `df74e005e456e94654fa93a40d748de399450b81`.
The host environment gained only the additional adapter dependencies.
`pip check` reported no broken requirements.
The stack retained Python 3.12, NumPy 1.26.4, pandas 2.2.2, SciPy 1.13.1, RDKit 2024.03.3, PyTorch 2.6.0, and PyG 2.7.0.

The real host verifier generated one product through REAL reaction 22 and one through WuXi reaction 1.
Both routes produced `CN(C(=O)c1ccccc1)c1ccccc1` and retained building-block IDs `0001` and `0002`.
The affinity was `1.6210767030715942` in both host runs.
Both predictions matched the unmodified direct screening CLI with maximum absolute difference `0.0` at batch size one.
The scorer rejected `C` with stage `ligand_standardization` and reason `NON_SMALL_MOLECULE: Molecular weight outside allowed range`.

The regression run passed 481 tests and skipped three opt-in root E2E tests.
The final SyntheMol suite passed 27 tests after adding one more empty-result error case.
The three opt-in root CLI E2E tests then passed with the real pretrained model and existing ESM cache.
The behavior test changed the favored building block and observed the corresponding change in the branch visited by the actual upstream MCTS generator.
Ruff passed with E402 excluded for pytest's dependency-gated imports.
Compilation and whitespace checks passed.

## Build the shared image

Docker Desktop reported server version `29.2.0` on an arm64 host.
There were no existing images to reuse.
The image targets `linux/amd64` under emulation because the native ligand package's verified platforms exclude Linux arm64.
The existing native platform and RDKit provenance guards remain enabled.

The first attempt waited on Docker Desktop's credential helper while resolving the public Dockerfile frontend.
The retry used a temporary unauthenticated configuration for public image pulls.
The user's Docker configuration was left unchanged.
The temporary configuration includes the Docker Desktop CLI plugin directory so BuildKit remains available.

```bash
mkdir -p runs/implementation/docker-config
cat > runs/implementation/docker-config/config.json <<'EOF'
{"auths":{},"cliPluginsExtraDirs":["/Applications/Docker.app/Contents/Resources/cli-plugins"]}
EOF
docker --config "$PWD/runs/implementation/docker-config" \
  --host unix:///Users/huanyeekoh/.docker/run/docker.sock \
  build --platform linux/amd64 --progress=plain \
  -f psichic_plus_workflows/Dockerfile -t psichic-workflows:local .
```

The final Dockerfile removes SyntheMol's bundled catalogue resources before installing its pinned source with `--no-deps`.
The build context excludes weights and generated run directories.
Only the two-molecule open fixture is included.
The existing GPU screening Docker ignore file also excludes `pretrained_weights/`.
The built image is `psichic-workflows:local`, with ID `sha256:390c49ac0e100a4de9f1976f9f16342a1f7b67ee4b0d5803b671544425453dcd`.
The build passed native ligand verification before and after installing the workflow dependencies.
Runtime inspection confirmed RDKit 2024.03.3 loads beneath `/opt/conda/envs/psichic_fp`, the Python interpreter's own prefix.
It also confirmed PyTorch `2.6.0+cpu`, no CUDA device, no installed SyntheMol resource directory, and the retained upstream MIT license.

## Run the real workflows inside the image

These commands used the weights, ESM cache, and upstream checkout from the host checks as read-only mounts.
The adapters ran from the built image, with no source-code bind mount.

```bash
mkdir -p runs/shared-image
docker run --rm --platform linux/amd64 --shm-size=4g --name psichic-synthemol-verification \
  -e TORCH_HOME=/torch \
  -v "$synthemol_weights:/weights:ro" \
  -v "$synthemol_old/work-scratch/torch:/torch:ro" \
  -v "$PWD/runs/shared-image:/runs" \
  psichic-workflows:local python -m psichic_plus_synthemol.example.verify \
  --model-dir /weights --output /runs/synthemol --device cpu
docker run --rm --platform linux/amd64 --shm-size=4g --name psichic-adaptive-verification \
  -e TORCH_HOME=/torch \
  -v "$synthemol_weights:/weights:ro" \
  -v "$synthemol_old/work-scratch/torch:/torch:ro" \
  -v "$synthemol_old/refs/af-vs:/upstream:ro" \
  -v "$PWD/runs/shared-image:/runs" \
  psichic-workflows:local python -m psichic_plus_adaptive.example.verify \
  --upstream /upstream --model-dir /weights --output /runs/adaptive --device cpu
```

The SyntheMol verifier passed inside the image for both chemical spaces.
Each space generated one amide with the expected route and leading-zero building-block IDs.
Both affinities matched direct inference with maximum absolute difference `0.0`.
The real scorer rejected the undersized ligand with the same stage and reason as on the host.

The existing AdaptiveFlow verifier also passed in the same image.
Its five successful stage predictions matched direct inference with maximum absolute difference `0.0` at batch size one.
The direct run recorded five successes and one invalid-SMILES failure.
The upstream selector chose tranche beta and its whole three-molecule collection for a primary target of two.
The final ranking contained two predictions, and the generated primary todo remained byte-identical to the executed todo.

CUDA execution and a CUDA-based shared-image build were not tested because this machine has no CUDA device.
Large licensed catalogues and large search budgets were not exercised.
The open fixture establishes the integration behavior for these two small routes.

## Evidence files

All paths below are relative to the implementation checkout.
Run outputs and the decision trail remain local under the ignored `runs/` directory.

| Evidence | Path |
| --- | --- |
| Host dependency installation | `runs/implementation/host-install.log` |
| Host generation and direct comparison | `runs/synthemol-host/verification.json` |
| Generated host routes | `runs/synthemol-host/{real,wuxi}/molecules.csv` |
| Exact host subprocess commands | `runs/synthemol-host/commands.json` |
| Full regression | `runs/implementation/regression.log` |
| Final SyntheMol behavior tests | `runs/implementation/synthemol-tests-final.log` |
| Opt-in root CLI E2E tests | `runs/implementation/root-e2e.log` |
| Lint | `runs/implementation/lint.log` |
| Final Docker build | `runs/implementation/docker-build-final.log` |
| Image package and provenance inspection | `runs/implementation/docker-runtime.json` |
| Container SyntheMol verification | `runs/shared-image/synthemol/verification.json` |
| Container SyntheMol commands | `runs/shared-image/synthemol/commands.json` |
| Container SyntheMol log | `runs/implementation/docker-synthemol.log` |
| Container AdaptiveFlow verification | `runs/shared-image/adaptive/verification.json` |
| Container AdaptiveFlow log | `runs/implementation/docker-adaptive.log` |
| Local decisions | `runs/implementation/decisions.tsv` |
| Reference checkout audit | `runs/implementation/upstream-status.txt` |

Independent code review and a fresh runtime check belong to the parent task.
This implementation child spawned no subagents.

## Independent review and fixes

The independent panel found three issues in the first implementation.
The fixes exclude reactions with any empty slot before constructing the upstream generator, reject catalogue ID `-1` reserved for intermediates, and bundle the pinned AdaptiveFlow-VS checkout in the image.
Two new behavior tests reproduced the catalogue issues before the fix in `runs/independent/review-reproductions.log`.
After the fix, all 29 focused tests passed in `runs/independent/review-fixes-tests.log`.
The partial-route regression runs upstream MCTS with both one- and two-reaction budgets.

An earlier independent run at `4439949` passed both SyntheMol spaces and AdaptiveFlow serially with zero prediction differences.
Its first concurrent AdaptiveFlow run had a worker terminated by signal 9 and reported incomplete output.
The cause was not proven.
Fresh serial output is under `runs/independent/adaptive/serial`.
Core dependencies are pinned, but ranged auxiliary dependencies and the tagged base mean future builds are not fully locked.
