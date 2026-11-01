# PSICHIC+ screening runtime


The screening CLI uses the installed `psichic_ligand` package shared with root inference.
The package supports Linux x86-64 and Apple Silicon macOS with the pinned environments.
See [the root build status](../README.md#ligand-build-status) before using the commands below.
CI builds the image and verifies the ligand package inside it.
The screening CLI was checked on CPU, but inference on a GPU has not been tested for this port.

## Contents

```text
inference.py                         # primary screening CLI
multi_gpu_inference.py               # data-parallel row-sharding wrapper
psichic_model.py                     # model definition and fast inference modes
featurizers.py                       # protein graphs and shared ligand-store adapters
pretrained_weights/PSICHIC_plus/     # config.json, degree.pt, model.pt
docker/Dockerfile.gpu                # CUDA 12.4 / Torch 2.6 / PyG 2.7 image
../environment_gpu.yml              # shared pinned GPU environment
../psichic-ligand/                   # installed ligand package
```

The CLI defaults to these inference settings:

```text
--inference_preset fast_screening
--precision fp32
```

For molecular screening against one or a few proteins, keep input-order batching:

```text
--batching_strategy input_order
--batch_size 128
```

For proteome-wide or varied-protein screening on A40-class memory, use protein-size batching:

```text
--batching_strategy protein_size_bucketed
--batch_size 100
```

## Build

Run from the repository root.
The image builds the ligand package and runs `python -m psichic_ligand verify` before copying the screening scripts.
A failed contract or unsupported platform stops the image build.

```bash
docker build --platform linux/amd64 \
  --build-arg VCS_REF="$(git rev-parse --short HEAD 2>/dev/null || echo production-bundle)" \
  --build-arg BUILD_DATE="$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  -f psichic_plus_screening/docker/Dockerfile.gpu \
  -t psichic-plus-production:gpu .
```

## Standard screening

Input can be a ligand CSV plus one FASTA file:

```bash
mkdir -p runs/torch_cache runs/screen
docker run --rm --gpus all --shm-size=16g \
  -v "$PWD/runs:/opt/min-psichic_plus/runs" \
  -v "$PWD/runs/torch_cache:/root/.cache/torch" \
  -v "$PWD/inputs:/opt/min-psichic_plus/inputs:ro" \
  psichic-plus-production:gpu \
  python inference.py \
    --input_file inputs/ligands.csv \
    --protein_fasta inputs/target.fasta \
    --output_folder runs/screen \
    --model_dir pretrained_weights/PSICHIC_plus \
    --ligand_column smiles \
    --batch_size 128 \
    --num_workers 4 \
    --batching_strategy input_order \
    --device cuda:0 \
    --metrics_json runs/screen_metrics.json
```

## Proteome-wide screening

Use this when the input CSV carries protein identifiers and sequences, or when screening many proteins:

```bash
mkdir -p runs/torch_cache runs/proteome
docker run --rm --gpus all --shm-size=32g \
  -v "$PWD/runs:/opt/min-psichic_plus/runs" \
  -v "$PWD/runs/torch_cache:/root/.cache/torch" \
  -v "$PWD/inputs:/opt/min-psichic_plus/inputs:ro" \
  psichic-plus-production:gpu \
  python inference.py \
    --input_file inputs/proteome_pairs.csv \
    --output_folder runs/proteome \
    --model_dir pretrained_weights/PSICHIC_plus \
    --ligand_column smiles \
    --batch_size 100 \
    --num_workers 4 \
    --batching_strategy protein_size_bucketed \
    --device cuda:0 \
    --metrics_json runs/proteome_metrics.json
```

The `runs/torch_cache` mount persists the Fair ESM checkpoint cache across ephemeral containers.
The first protein-featurization run downloads the ESM backbone when the cache is empty.

## Exclusions

This bundle intentionally does not include:

- FEP benchmark fixtures
- proteome validation panels
- Nsight Systems helpers or artifacts
- optimization history
- verifier-only scripts
- Docker upgrade sandboxes

## Ligand cache

Pass `--ligand_cache_dir runs/ligands` to share a persistent cache across runs.
Graphs live below the `FEATURIZATION_VERSION` subdirectory, currently `rdkit-2024.03.3-enumpick-max100-v1+psichic-graph-v1`.
Legacy unversioned cache files are ignored and remain untouched.
A second run uses the stored graphs without featurization, including shape precompute and interpretation.
`--no-ligand_disk_cache` disables persistent ligand graph storage.
Metrics record `ligand_featurization_version`.

## Acknowledgements

Papyrus 0.0.5 and ChEMBL Structure Pipeline 1.2.4 are vendored once inside `psichic-ligand` with their MIT licenses.
The adapted RDKit code retains its BSD notice.
See [psichic-ligand/NOTICE](../psichic-ligand/NOTICE) and the files it names.
