# Screen a library with AdaptiveFlow and PSICHIC+

Run a sparse prescreen, let AdaptiveFlow select whole collections, and score the selected collections with PSICHIC+.
The adapter runs AdaptiveFlow-VS's local Bash backend with one scoring process.
The outputs are model predictions.
Read the [license notices](patches/NOTICE.md) before using or distributing the integration.

## Install the adapter dependencies

Start in the PSICHIC+ repository root with its working screening environment.
Install the additional packages without replacing the PSICHIC+ environment pins.

```bash
python -m pip install -r psichic_plus_adaptive/requirements.txt
```

Supply your own clean AdaptiveFlow-VS checkout at the supported revision.
The launcher verifies the revision and tracked files, clones the checkout into the new run directory, checks the patch with `git apply --check`, and applies it there.
Your supplied checkout stays unchanged.

```bash
git clone https://github.com/QuantumAI4Bio/AdaptiveFlow-VS /path/to/af-vs
git -C /path/to/af-vs checkout --detach df74e005e456e94654fa93a40d748de399450b81
```

## Run the two-tranche example

Point `--model-dir` at authorized PSICHIC+ weights containing `config.json`, `degree.pt`, and `model.pt`.
Set `TORCH_HOME` to an existing ESM cache if you have one.
An empty cache triggers the existing protein helper's ESM download.

```bash
python -m psichic_plus_adaptive.example.verify \
  --upstream /path/to/af-vs \
  --model-dir /path/to/authorized/PSICHIC_plus \
  --output runs/adaptive-example \
  --device cpu
```

The example uses six molecules in two tranches, including one invalid SMILES and molecule ID `0001` in both collections.
It runs the upstream worker in both stages, calls the real upstream selector, and compares every attempted item with the existing unmodified inference CLI.
The equality check uses batch size one in both paths to keep graph batch composition identical.
It writes `verification.json` only after the affinity comparison, failure checks, ranking check, and todo comparison pass.
The primary target is two molecules, so upstream selects a whole three-molecule collection.
Use a new output directory for every run.
Each stage has a worker execution deadline of 3600 seconds.
Set `--stage-timeout` to change it.
On expiry, the launcher stops the worker process group and exports missing attempts as failures with an `incomplete.json` report.

## Screen your library

Create a CSV with `collection_id,molecule_id,tranche,smiles` columns and a single-record target FASTA.
Keep each collection in one tranche and make each molecule ID unique within its collection.
See the [format reference](FORMAT.md) for sampling, budgets, identifiers, and output files.

```bash
python -m psichic_plus_adaptive \
  --library /inputs/library.csv \
  --target-fasta /inputs/target.fasta \
  --model-dir /weights/PSICHIC_plus \
  --upstream /path/to/af-vs \
  --output runs/my-screen \
  --samples-per-tranche 10 \
  --primary-target 1000 \
  --batch-size 32 \
  --device cpu
```

Read `ranked.csv` for primary-stage successes in descending predicted affinity.
Read `prescreen/results.csv` and `primaryscreen/results.csv` for all attempted molecules, including failures.
Inspect each stage's `failures.csv` before using the rankings.
If a run exits with an incomplete-output error, inspect its `incomplete.json` and upstream worker logs.

## Run inside your screening image

For one image containing PSICHIC+ and both workflow adapters, use the [shared workflow image](../psichic_plus_workflows/README.md).
The standalone adapter image below extends a screening image you supply.

Choose a base image you are authorized to use.
The base must provide the existing screening modules at `/opt/min-psichic_plus`, their Python dependencies, Git, Bash, and `/usr/bin/tini`.
The adapter Dockerfile copies no weights and downloads no AdaptiveFlow source.
Supply weights and the external upstream checkout as read-only mounts.

```bash
docker build \
  --build-arg PSICHIC_BASE_IMAGE=your-authorized-screening-image:tag \
  -f psichic_plus_adaptive/Dockerfile \
  -t psichic-adaptive:local .
mkdir -p runs/adaptive-docker runs/torch-cache
docker run --rm --shm-size=4g \
  -v "$PWD/psichic_plus_adaptive/example:/inputs:ro" \
  -v /path/to/authorized/PSICHIC_plus:/weights:ro \
  -v /path/to/af-vs:/upstream:ro \
  -v "$PWD/runs/adaptive-docker:/runs" \
  -v "$PWD/runs/torch-cache:/root/.cache/torch" \
  psichic-adaptive:local \
  --library /inputs/library.csv \
  --target-fasta /inputs/target.fasta \
  --model-dir /weights \
  --upstream /upstream \
  --output /runs/example \
  --samples-per-tranche 2 \
  --primary-target 2 \
  --batch-size 2 \
  --device cpu
```

To check the Docker run against direct inference, use the example verifier through the image's entrypoint override.

```bash
docker run --rm --shm-size=4g \
  --entrypoint /usr/bin/tini \
  -v /path/to/authorized/PSICHIC_plus:/weights:ro \
  -v /path/to/af-vs:/upstream:ro \
  -v "$PWD/runs/adaptive-docker:/runs" \
  -v "$PWD/runs/torch-cache:/root/.cache/torch" \
  psichic-adaptive:local -- python -m psichic_plus_adaptive.example.verify \
  --upstream /upstream --model-dir /weights --output /runs/verified --device cpu
```

For CUDA, add `--gpus all` and use `--device cuda:0` with a compatible base image and driver.
The [shared CPU image](../psichic_plus_workflows/README.md) has passed this verifier with real weights.
See the [shared-image evidence](../psichic_plus_synthemol/VERIFICATION.md) for the executed commands and results.
CUDA execution remains unverified.

## Run the behavior tests

Set `AFVS_UPSTREAM` to enable the checked-patch, actual archive-reader, completion-queue, and selector tests.
These tests use a stub predictor for worker behavior and do not load model weights.
The separate example above verifies the real model and the complete adaptive handoff.

```bash
AFVS_UPSTREAM=/path/to/af-vs python -m pytest tests/test_adaptive.py -q
```
