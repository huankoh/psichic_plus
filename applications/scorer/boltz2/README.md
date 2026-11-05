# Run optional Boltz-2 scoring

Run these commands from the repository root.
Boltz-2 is optional and runs through the unmodified `boltz predict` CLI in a separate environment.
The supported release is **Boltz 2.2.1**.
The integration rejects other releases before prediction.

The four workflows are `boltz-only`, `full-ensemble`, `pyramid-ensemble`, and `pyramid-refinement`.
Ensemble modes offer `rank-v1` and `concentration-v1` and remain experimental.
Omitting `--ensemble` preserves the existing `rank-v1` standalone default.
Read [ranking and input rules](REFERENCE.md) before interpreting results.
See [verification evidence and limits](VERIFICATION.md) for what has actually run.
The [GPU handoff](HANDOFF.md) describes the remaining checks on `huan-main-a100`.

## Install separate environments

Keep your existing PSICHIC+ environment for combined modes.
Install the workflow's small additional dependency there.

```bash
python -m pip install -r applications/scorer/boltz2/requirements.txt
```

For Boltz-only mode, the coordinator needs Python 3.11 or newer and that requirements file.
It does not import Torch, PSICHIC+, or Boltz.

Create a separate environment outside the source tree for the native CLI.
This CPU setup was used for the upstream example attempt on macOS ARM64.

```bash
uv venv --python 3.11.18 "$HOME/.venvs/boltz-2.2.1"
uv pip install --python "$HOME/.venvs/boltz-2.2.1/bin/python" \
  'boltz==2.2.1' 'torch==2.5.1'
export BOLTZ_PYTHON="$HOME/.venvs/boltz-2.2.1/bin/python"
"$HOME/.venvs/boltz-2.2.1/bin/boltz" predict --help
```

The combined Linux image uses a separate Python 3.12.8 environment with the hash-locked CUDA dependencies in `requirements-boltz.lock`.
It pins Torch 2.7.1 because the Boltz CUDA extras conflict with the PSICHIC+ Torch 2.6.0 CUDA libraries.
The PSICHIC+ environment keeps its existing versions.

## Prepare a target and candidates

Place your inputs outside the source tree, for example under `/data/screen`.
The candidate CSV requires `ID` and `SMILES` columns by default.
Use `--id-column` and `--smiles-column` for other column names, including existing `smiles` or `Ligand` files.

```csv
ID,SMILES
aspirin,CC(=O)Oc1ccccc1C(=O)O
caffeine,Cn1c(=O)c2c(ncn2C)n(C)c1=O
```

Provide a FASTA with exactly one protein, or use a [native target YAML](REFERENCE.md#native-targets).
Choose an MSA mode explicitly.

- `--msa-mode local --msa /data/screen/target.a3m` supplies a local MSA with a FASTA target.
- `--msa-mode server` opts in to sending protein sequences to the upstream MSA service.
- `--msa-mode single` explicitly uses single-sequence inputs without an MSA service.

Single-sequence mode can reduce prediction quality.
No MSA server, structural template, pocket constraint, or contact constraint is enabled by default.

## Run each workflow

The examples use CPU execution and explicit single-sequence mode.
Replace `--accelerator cpu` with `--accelerator gpu` on a supported CUDA host.
Set `--psichic-device cuda:0` separately if PSICHIC+ should also use the GPU.
PSICHIC+ finishes and releases its model before the native Boltz process starts.

Score all candidates with Boltz-2.

```bash
python -m applications.scorer.boltz2 \
  --mode boltz-only --candidates /data/screen/molecules.csv \
  --fasta /data/screen/target.fasta --msa-mode single \
  --output /data/results/boltz-only \
  -- --accelerator cpu --cache /data/cache/boltz
```

Score all candidates with both engines and combine their ranks using `rank-v1`.

```bash
python -m applications.scorer.boltz2 \
  --mode full-ensemble --candidates /data/screen/molecules.csv \
  --fasta /data/screen/target.fasta --msa-mode single \
  --model-dir /data/weights/PSICHIC_plus --psichic-device cpu \
  --psichic-weight 1 --boltz-weight 1 --output /data/results/full-ensemble \
  -- --accelerator cpu --cache /data/cache/boltz
```

Use the source concentration ensemble explicitly.
It requests PSICHIC+ distribution output internally and uses native Boltz affinity, even when `--psichic-output` is omitted.

```bash
python -m applications.scorer.boltz2 \
  --mode full-ensemble --ensemble concentration-v1 \
  --candidates /data/screen/molecules.csv --fasta /data/screen/target.fasta \
  --msa-mode single --model-dir /data/weights/PSICHIC_plus \
  --output /data/results/concentration-ensemble \
  -- --accelerator cpu --cache /data/cache/boltz
```

For a shortlist concentration ensemble, use `--mode pyramid-ensemble --shortlist-size 20 --ensemble concentration-v1`.
That shortlist requires 20 valid affinity and distribution results before Boltz starts.
`concentration-v1` rejects `--rank-by affinity_probability_binary` and either manual weight option, including an explicitly supplied weight of 1.
Its `normalized_ensemble_score` is a relative score with larger values better, not a calibrated affinity.
The existing lower-better `experimental_combined_value` remains exclusive to `rank-v1`.

`--psichic-output distribution` also retains distribution parameters in rank ensemble or refinement reports without changing their affinity-only selection rules.
A requested distribution failure retains valid affinity, records the failure, and marks the run partial.
Both methods reproduce software rules; these checks do not establish scientific ranking accuracy.

Select the top 20 successful PSICHIC+ rows, then combine both engines' ranks within their common successful shortlist.
Provide at least 20 candidates that PSICHIC+ can score, or choose a smaller shortlist.

```bash
python -m applications.scorer.boltz2 \
  --mode pyramid-ensemble --shortlist-size 20 \
  --candidates /data/screen/molecules.csv --fasta /data/screen/target.fasta \
  --msa-mode single --model-dir /data/weights/PSICHIC_plus \
  --output /data/results/pyramid-ensemble \
  -- --accelerator cpu --cache /data/cache/boltz
```

Select the top 20 successful PSICHIC+ rows, then rank them using Boltz binder probability alone.

```bash
python -m applications.scorer.boltz2 \
  --mode pyramid-refinement --shortlist-size 20 \
  --candidates /data/screen/molecules.csv --fasta /data/screen/target.fasta \
  --msa-mode single --model-dir /data/weights/PSICHIC_plus \
  --rank-by affinity_probability_binary --output /data/results/pyramid-refinement \
  -- --accelerator cpu --cache /data/cache/boltz
```

Every output path must be new.
Each run writes `results.csv`, `results.json`, and `manifest.json` and retains the native inputs, structures, affinity JSON, logs, and copied target assets under `boltz/`.
Sort results by `final_rank` ascending for ranked output.
The stored report remains in original input order and includes unselected and failed rows.

## Build the PSICHIC+ only image

The existing image keeps PSICHIC+, SyntheMol, and AdaptiveFlow without installing Boltz.
Build it from this repository revision.

```bash
docker build --platform linux/amd64 -f applications/Dockerfile -t psichic-workflows:latest .
docker run --rm psichic-workflows:latest python -m applications.generator --help
docker run --rm psichic-workflows:latest python -c \
  "import importlib.util; assert importlib.util.find_spec('boltz') is None"
```

Existing application commands remain documented in [applications](../../README.md).
Mount PSICHIC+ weights read-only and mount writable outputs and caches when running predictions.
Use the existing native CPU option, or add `--gpus all` and select CUDA for GPU execution.

## Build the combined image

Build the PSICHIC+ only image first using the command above.
The second image inherits that image and installs Boltz in `/opt/boltz`.
Its default `BOLTZ_PYTHON` points to `/opt/boltz/bin/python`.

```bash
docker build --platform linux/amd64 \
  -f applications/scorer/boltz2/Dockerfile \
  --build-arg PSICHIC_IMAGE=psichic-workflows:latest \
  --build-arg SOURCE_REVISION="$(git rev-parse HEAD)" \
  -t psichic-boltz2:2.2.1 .
docker run --rm psichic-boltz2:2.2.1 python -m applications.scorer.boltz2 --help
docker run --rm psichic-boltz2:2.2.1 /opt/boltz/bin/boltz predict --help
```

Run the full ensemble on a Linux CUDA host with a driver compatible with the locked PyTorch CUDA runtime.
The lock currently uses CUDA 12.6 libraries in the Boltz environment.
Docker builds and GPU inference have not been verified on the development Mac.

```bash
docker run --rm --gpus all \
  -v /data/screen:/inputs:ro -v /data/weights:/weights:ro \
  -v /data/results:/outputs -v /data/cache:/cache \
  psichic-boltz2:2.2.1 python -m applications.scorer.boltz2 \
  --mode full-ensemble --candidates /inputs/molecules.csv \
  --fasta /inputs/target.fasta --msa-mode single \
  --model-dir /weights/PSICHIC_plus --psichic-device cuda:0 \
  --output /outputs/full-ensemble \
  -- --accelerator gpu --cache /cache/boltz
```

For CPU execution, omit `--gpus all`, use `--psichic-device cpu`, and pass `--accelerator cpu`.
CPU inference can take substantially longer than the smoke-test time budget.
Change the workflow arguments to use any of the other three modes.
Omit `--model-dir` in Boltz-only mode.
For native YAML, mount every referenced asset or use paths relative to the mounted YAML.
The build excludes weights and user data, and it does not download model weights.
Boltz downloads its public models into the mounted cache at runtime.
PSICHIC+ uses the weights supplied through `--model-dir`.
