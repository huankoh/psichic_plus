# AdaptiveFlow integration verification

Verified on macOS arm64 with Python 3.12.15, CPU FP32, the existing PSICHIC+ checkpoint, and cached ESM weights.
The base checkout was `9288a0957145966abff22e600a0abd98af34ef66`.
External AdaptiveFlow-VS was clean at `df74e005e456e94654fa93a40d748de399450b81`.
No shared inference, Docker, or root README files changed.

## Completion checkpoint

Before adding orchestration, a real upstream workunit completed four rows in two batches through the downloader, unpacker, collection, docking, summary, and uploader processes.
The output contained three successful predictions, one invalid-SMILES failure, CSV and parquet summaries, the failure overview, and the per-ligand output archive.
One docking child initialized the model and target once.
This established the completion contract before building the adaptive handoff.

## Commands

Run these commands from the exclusive checkout.
The dependency overlay leaves the parallel runtime unchanged.
Use the original interpreter because the native ligand package checks that RDKit belongs to its interpreter prefix.
Each verification output directory must be new.

```bash
cd /Users/huanyeekoh/.t3/scratch/2026-11-01-t3-pstack-science-poteto-mode-4215a796
AF_PYTHON=/Users/huanyeekoh/.t3/scratch/2026-10-27-https-github-com-huankoh-psichic-7a152a8a/work-scratch/fresh-env/bin/python
AF_UPSTREAM=/Users/huanyeekoh/.t3/scratch/2026-10-27-https-github-com-huankoh-psichic-7a152a8a/refs/af-vs
"$AF_PYTHON" -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r integrations/adaptiveflow/requirements.txt
export PYTHONPATH="$PWD/.venv/lib/python3.12/site-packages"
export TORCH_HOME=/Users/huanyeekoh/.t3/scratch/2026-10-27-https-github-com-huankoh-psichic-7a152a8a/work-scratch/torch
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
"$AF_PYTHON" integrations/adaptiveflow/verify.py --upstream "$AF_UPSTREAM" --output runs/adaptiveflow-verified
"$AF_PYTHON" integrations/adaptiveflow/verify.py --upstream "$AF_UPSTREAM" --output runs/adaptiveflow-verify-fork --start-method fork
"$AF_PYTHON" -m pytest -q integrations/adaptiveflow/test_library.py
ruff check integrations/adaptiveflow
ruff format --check integrations/adaptiveflow
git diff --check
```

`verify.py` retains the exact integration command in `verification.json` and the subprocess output in its logs.
It generates a four-row library with duplicate original identifiers across two collections, duplicate aspirin SMILES, caffeine, and invalid SMILES.
Its target FASTA contains the synthetic test sequence `MKTAYIAKQRQISFVKSHFSRQDILD`.
The direct comparison uses the standalone screening CLI with the conservative inference preset, batch size two, and `emit_null` invalid-ligand policy.
This is an integration fixture, not a biological validation of the model.

## Observed results

Both multiprocessing methods passed the actual prescreen, external selector, upstream preparation, and primary worker execution.
The final `spawn` evidence is in `runs/adaptiveflow-verified/verification.json`.
The `fork` evidence is in `runs/adaptiveflow-verify-fork/verification.json`.

- The prescreen completed two batches with one model and target initialization in one docking child.
- All three successful affinities matched direct inference with maximum absolute difference `0.0` in these CPU fixtures.
- Successful upstream scores equaled negative raw affinity, and ascending scores produced descending affinity order.
- Both original duplicate IDs and both duplicate SMILES rows survived with distinct internal IDs.
- Invalid SMILES produced a reason, null affinity, and null score in the outputs and upstream failure overview.
- The deterministic two-tranche fixture selected `BB_0000001` and `BB_0000042` for budget three, totaling four ligands through whole-collection rounding.
- The explicit `AA` regex selected only AA collections, despite their worse synthetic scores.
- The selected todo file produced a real upstream primary workunit containing exactly the selected collections in `all` mode.
- All seven conversion and input-contract tests passed.
- Ruff checks, formatting checks, and whitespace checks passed.

## Independent verification after review fixes

A fresh verifier repeated the actual-model workflow into `runs/adaptiveflow-independent` after the input fixes.
It confirmed three successful predictions, one visible failure, one model initialization across two batches, and exact agreement with direct CPU inference.
The actual upstream selector and primary workunit preparation passed again.
All 15 input-contract tests and Ruff checks passed in the final verification.

The converter now rejects tranche names that upstream CSV parsing turns into missing values or booleans.
Regression cases include `NA`, `NULL`, `NaN`, `True`, and `FALSE`.
Generated sparse names skip existing input tranche names without regard to case.
Source tranche names that differ only by case are rejected before output creation.
An additional real run with input collection `SPARSE0_0000000` used separate sparse archives, preserved the original archive, and produced primary predictions for both original ligand IDs.
Its evidence is in `runs/adaptiveflow-collision-selected/verification.json`.

## Gaps

`docker info --format '{{.ServerVersion}}'` failed because `/Users/huanyeekoh/.docker/run/docker.sock` did not exist.
No image build or container execution was tested.
PyTorch reported `torch.cuda.is_available() == False`, and `nvidia-smi` was unavailable.
CUDA execution remains unverified.
The runtime mount command is documented without changing shared Docker files or distributing a combined image.
Two reviewers completed the full change review.
All three input-correctness findings were fixed and the final actual-model verification passed.
The final evidence is in `runs/adaptiveflow-final/verification.json` and `runs/adaptiveflow-lowercase-final/verification.json`.
