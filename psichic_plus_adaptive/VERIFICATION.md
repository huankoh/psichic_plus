# Implementation verification

The implementation ran in the isolated clone `/Users/huanyeekoh/.t3/scratch/2026-11-01-t3-pstack-science-poteto-mode-9bd25403/src` on branch `feat/adaptiveflow`.
The base was `9288a0957145966abff22e600a0abd98af34ef66`.
The supplied upstream checkout remained clean at `df74e005e456e94654fa93a40d748de399450b81`.
Source changes and real-model run artifacts stayed in this clone.
The user-authorized existing `fresh-env` received the optional adapter dependencies.

## Commands executed

These shell variables abbreviate the exact paths used in the commands below.
The final run directory already exists, so a repeat run needs a different output path.

```bash
cd /Users/huanyeekoh/.t3/scratch/2026-11-01-t3-pstack-science-poteto-mode-9bd25403/src
adaptive_python=/Users/huanyeekoh/.t3/scratch/2026-10-27-https-github-com-huankoh-psichic-7a152a8a/work-scratch/fresh-env/bin/python
adaptive_upstream=/Users/huanyeekoh/.t3/scratch/2026-10-27-https-github-com-huankoh-psichic-7a152a8a/refs/af-vs
export TORCH_HOME=/Users/huanyeekoh/.t3/scratch/2026-10-27-https-github-com-huankoh-psichic-7a152a8a/work-scratch/torch
"$adaptive_python" -m pip install -r psichic_plus_adaptive/requirements.txt
"$adaptive_python" -m pip check
AFVS_UPSTREAM="$adaptive_upstream" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  "$adaptive_python" -m pytest tests psichic_plus_screening/tests psichic-ligand/tests -q
"$adaptive_python" -m psichic_plus_adaptive.example.verify \
  --upstream "$adaptive_upstream" \
  --model-dir psichic_plus_screening/pretrained_weights/PSICHIC_plus \
  --output runs/adaptive-acceptance-2 \
  --device cpu
"$adaptive_python" -m compileall -q psichic_plus_adaptive
git diff --cached --check
git -C runs/adaptive-acceptance-2/upstream diff --check
docker info
"$adaptive_python" -c 'import torch; print("cuda_available", torch.cuda.is_available()); print("cuda_version", torch.version.cuda)'
```

The environment used Python 3.12.15 and PyTorch 2.6.0.
The added packages included boto3 1.43.123, PyArrow 23.0.1, IPython 9.18.0, and the existing Jinja2 3.1.6.
The boto3 dependency resolved python-dateutil to 2.9.0.post0.
`pip check` reported no broken requirements.

## Observed results

The final regression command passed 454 tests and skipped three opt-in root E2E tests.
The adapter contributed 23 passing behavior tests, including the real upstream archive reader, checked patch, completion queue, and selector.
The warnings came from upstream tar extraction and PyG deprecations.

The separate real-model example executed upstream folder preparation, work-unit creation, Bash job generation, and the patched upstream worker in both stages.
It invoked the real `create_todofile_atg-primaryscreen.py` with `tranche_ave_score`.
The prescreen returned three successes and one invalid-SMILES failure.
The primary screen returned two successes and the same invalid-SMILES failure.
Upstream selected tranche beta and attempted its whole three-molecule collection for a target of two.
Both collections retained molecule ID `0001` with different tokens.

Five successful stage predictions matched the direct, unmodified screening CLI with maximum absolute affinity difference `0.0`.
Both paths used batch size one for this equality check.
An earlier batch-size-two comparison used different graph batch companions and differed by `1.38e-5` for one affinity.
The equality result does not establish bitwise agreement across different batch compositions or devices.
The generated primary todo matched the primary input todo byte for byte.
The final ranking contained two predictions in descending raw affinity.

The final evidence is relative to the isolated clone root.

| Evidence | Path |
| --- | --- |
| Equality and handoff checks | `runs/adaptive-acceptance-2/verification.json` |
| Run counts | `runs/adaptive-acceptance-2/run.json` |
| Actual selection and todo digest | `runs/adaptive-acceptance-2/selection.json` |
| Generated upstream todo | `runs/adaptive-acceptance-2/primary.todo` |
| Original-ID manifest | `runs/adaptive-acceptance-2/library/manifest.csv` |
| Ranked primary predictions | `runs/adaptive-acceptance-2/ranked.csv` |
| Prescreen results and failures | `runs/adaptive-acceptance-2/prescreen/results.csv`, `runs/adaptive-acceptance-2/prescreen/failures.csv` |
| Primary results and failures | `runs/adaptive-acceptance-2/primaryscreen/results.csv`, `runs/adaptive-acceptance-2/primaryscreen/failures.csv` |
| Worker logs | `runs/adaptive-acceptance-2/<stage>/workflow/workunits/1/0.out` |
| Upstream command arguments and output | `runs/adaptive-acceptance-2/<stage>/tools/commands.log` |
| Selector command and output | `runs/adaptive-acceptance-2/commands.log` |
| Direct inference command and output | `runs/adaptive-acceptance-2/direct.log` |

Docker image build and container execution could not run because the Docker daemon socket was absent.
`docker info` failed to connect to `/Users/huanyeekoh/.docker/run/docker.sock`.
PyTorch reported `cuda_available False` and `cuda_version None`, so CUDA execution remains unverified.
The user-specified-base Dockerfile copies adapter code and license notices, with weights supplied through a mount.
No shared Docker files or existing inference modules changed.
No independent reviewer ran in this implementation-only subagent.
Independent verification belongs to the parent task.
