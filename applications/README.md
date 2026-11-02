# PSICHIC+ applications

These applications use the existing PSICHIC+ predictor.

- [scorer](scorer/): load a target and model once and score molecules in batches.
- [generator](generator/): use PSICHIC+ to guide native SyntheMol generation.
- [screener](screener/): use PSICHIC+ in native AdaptiveFlow-VS screening workunits.

## Shared Docker image

Build one CUDA image from the repository root on an NVIDIA-enabled Linux host.
The build installs both integrations and their pinned upstream versions.

```sh
docker build --platform linux/amd64 -f applications/Dockerfile -t psichic-applications .
docker run --rm --gpus all psichic-applications
```

Use `--gpus all` and select `cuda` in a workflow to use the GPU.
The CUDA image can also run a workflow on CPU by selecting `cpu` and omitting `--gpus all`.

Provide your own model weights, target sequence, input data, and ESM cache as mounted volumes when running a workflow.
See the individual application guides for their inputs and commands.
The image does not include model weights or SyntheMol catalogue data.
AdaptiveFlow workers need access to the image and the same input files.

For example, prepare SyntheMol building-block scores with a target and weights you provide.
Create local `inputs`, `outputs`, `weights`, and `torch-cache` directories first.
Place `building_blocks.csv` and `target.fasta` in `inputs`, and the PSICHIC+ model files in `weights`.

```sh
docker run --rm --gpus all \
  -v "$PWD/inputs:/inputs:ro" \
  -v "$PWD/outputs:/outputs" \
  -v "$PWD/weights:/weights:ro" \
  -v "$PWD/torch-cache:/torch-cache" \
  -e TORCH_HOME=/torch-cache \
  psichic-applications python -m applications.generator score-building-blocks \
  --input /inputs/building_blocks.csv --output /outputs/scored_blocks.csv \
  --target-fasta /inputs/target.fasta --model-dir /weights \
  --cache-dir /outputs/cache --device cuda
```

The [generator guide](generator/README.md) shows how to use the scored CSV for native generation.
For AdaptiveFlow, the image contains the patched checkout at `/opt/AdaptiveFlow-VS`.
Copy it to a mounted working directory, configure it with your libraries and target, and follow the [screener guide](screener/README.md).
