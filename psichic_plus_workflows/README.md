# Run both workflows in one image

The shared image contains the existing PSICHIC+ screening runtime, SyntheMol generation, and AdaptiveFlow screening.
The default build uses `environment_cpu.yml` and preserves its Python 3.12, RDKit 2024.03.3, PyTorch 2.6.0, and PyG 2.7.0 pins.
It builds and verifies the native ligand package against RDKit in the same conda prefix.
The image contains the two-molecule open SyntheMol fixture, with no model weights or user catalogues.
The build removes SyntheMol's bundled catalogue resources before installing its pinned source with `--no-deps`.

## Build the image

Run from the repository root.
Use `linux/amd64` on Apple Silicon because the native ligand contract does not support Linux arm64.

```bash
docker build --platform linux/amd64 \
  -f psichic_plus_workflows/Dockerfile \
  -t psichic-workflows:local .
```

To extend an existing authorized CUDA screening image, supply its tag as `--build-arg PSICHIC_BASE_IMAGE=your-screening-image:tag`.
That base must provide the existing screening modules at `/opt/min-psichic_plus`, their dependencies, Bash, Git, and `/usr/bin/tini`.
The shared image adds both adapters and compatible upstream dependencies under constraints.
It does not install SyntheMol's older environment pins.
CUDA execution requires the matching driver and remains unverified in this implementation environment.

## Verify SyntheMol generation

Mount authorized PSICHIC+ weights and an ESM cache at runtime.
Use a writable ESM cache if it needs to download missing files.
Use a new output path for every verification run.

```bash
mkdir -p runs/shared-image runs/torch-cache
docker run --rm --platform linux/amd64 --shm-size=4g \
  -v /path/to/authorized/PSICHIC_plus:/weights:ro \
  -v "$PWD/runs/shared-image:/runs" \
  -v "$PWD/runs/torch-cache:/root/.cache/torch" \
  psichic-workflows:local \
  python -m psichic_plus_synthemol.example.verify \
  --model-dir /weights --output /runs/synthemol --device cpu
```

For your catalogue, mount `/path/to/inputs:/inputs:ro` and replace the verifier command with the CLI in the [generation guide](../psichic_plus_synthemol/README.md).
Use output paths beneath `/runs` to retain results after the container exits.

## Verify AdaptiveFlow screening

Supply a clean upstream checkout at `df74e005e456e94654fa93a40d748de399450b81` as described in the [AdaptiveFlow guide](../psichic_plus_adaptive/README.md).
The workflow copies and patches it inside the output directory.

```bash
docker run --rm --platform linux/amd64 --shm-size=4g \
  -v /path/to/authorized/PSICHIC_plus:/weights:ro \
  -v /path/to/af-vs:/upstream:ro \
  -v "$PWD/runs/shared-image:/runs" \
  -v "$PWD/runs/torch-cache:/root/.cache/torch" \
  psichic-workflows:local \
  python -m psichic_plus_adaptive.example.verify \
  --upstream /upstream --model-dir /weights --output /runs/adaptive --device cpu
```

For a CUDA image, add `--gpus all` and use `--device cuda:0`.
The image entrypoint is `tini`, so both workflows and the existing `python /opt/min-psichic_plus/inference.py` CLI accept ordinary command arguments.
