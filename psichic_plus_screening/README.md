# PSICHIC+ Production Runtime Bundle

This folder is the slim production-facing PSICHIC+ runtime bundle. It is copied
from the validated minimal package, but excludes benchmark fixtures, profiling
artifacts, optimization logs, validation reports, and sandbox Docker images.

## Contents

```text
inference.py                         # primary screening CLI
multi_gpu_inference.py               # data-parallel row-sharding wrapper
psichic_model.py                     # model definition and fast inference modes
featurizers.py                       # ligand/protein featurization
papyrus_structure_pipeline/          # vendored ligand standardizer
pretrained_weights/PSICHIC_plus/     # config.json, degree.pt, model.pt
docker/Dockerfile.gpu                # CUDA 12.4 / Torch 2.6 / PyG 2.7 image
docker/environment-gpu-a40.yml       # pinned GPU environment
```

The CLI defaults to the validated fast path:

```text
--inference_preset fast_screening
--precision fp32
```

For molecular screening against one or a few proteins, keep input-order
batching:

```text
--batching_strategy input_order
--batch_size 128
```

For proteome-wide or varied-protein screening on A40-class memory, use
protein-size batching:

```text
--batching_strategy protein_size_bucketed
--batch_size 100
```

## Build

```bash
docker build --platform linux/amd64 \
  --build-arg VCS_REF="$(git rev-parse --short HEAD 2>/dev/null || echo production-bundle)" \
  --build-arg BUILD_DATE="$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  -f docker/Dockerfile.gpu \
  -t psichic-plus-production:gpu .
```

## Standard Screening

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

## Proteome-Wide Screening

Use this when the input CSV already carries protein identifiers and sequences,
or when screening many different proteins:

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

The `runs/torch_cache` mount persists the Fair ESM checkpoint cache across
ephemeral containers. Without it, the first protein-featurization run downloads
the ESM backbone inside the container.

## Exclusions

This bundle intentionally does not include:

- FEP benchmark fixtures
- proteome validation panels
- Nsight Systems helpers or artifacts
- optimization history
- verifier-only scripts
- Docker upgrade sandboxes

## Acknowledgements

This bundle vendors ligand standardization code from
`papyrus-structure-pipeline` version `0.0.5`, licensed under the MIT License.
Copyright (c) 2023 Olivier J. M. Béquignon.
