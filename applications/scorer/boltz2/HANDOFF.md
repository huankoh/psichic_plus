# Continue GPU verification on huan-main-a100

This handoff was executed on 2026-11-05.
See the [GPU verification results](VERIFICATION.md#gpu-verification-on-2026-11-05) for completed native inference, exact parsing checks, all four real workflows, both Docker images, and remaining limits.
The instructions below retain the original reproduction procedure.

At handoff time, the implementation was ready for GPU verification, and the PR was to remain a draft until the outstanding checks were assessed.
Use a PC with an existing authorized connection to `huan-main-a100`.
The development session could not reach that server and performed no GPU work there.

## Retrieve the implementation

```bash
git clone --branch feat/optional-boltz2-scoring https://github.com/huankoh/psichic_plus.git
cd psichic_plus
git status --short --branch
git rev-parse HEAD
```

The branch contains all implementation code, focused tests, the generated dependency lock, both image instructions, and this handoff.
Downloaded weights, Python environments, user data, and local execution artifacts are not committed.
The [verification record](VERIFICATION.md) preserves the results and limitations of the completed checks.

## Current implementation and evidence

All four modes are implemented through `python -m applications.scorer.boltz2`.
Boltz runs through the unmodified 2.2.1 CLI in a separate environment.
PSICHIC+ resides in `applications/scorer/psichic_plus/`, and the old predictor import remains compatible.
The existing PSICHIC+ Dockerfile and shared result types are unchanged.
No FEP or ABFE implementation is included.

The final application and selected repository suite passed 107 tests with 14 skips.
Two skips require PSICHIC+ weights, eleven require a pinned SyntheMol source checkout, and one requires a pinned AFVS checkout.
Lint, dependency checks, CLI help, and whitespace checks passed.
Independent review findings were corrected for YAML merges, mixed custom and automatic MSAs, loaded PSICHIC+ source provenance, and extreme ensemble weights.

The real upstream example reached CPU structure prediction but timed out after 360 seconds without structures or affinity scores.
Two bounded coordinator runs verified actual subprocess isolation and explicit timeout failures with retained row identity and null scores.
Neither Docker image was built because Docker was unavailable on the development Mac.
Completed GPU inference and numerical comparison with a native result remained open at handoff time.

## Run the GPU checks

1. Inspect the server's instructions, scheduler, `nvidia-smi`, free GPU memory, active jobs, disk space, and Docker access.
   Use one available GPU through the server's normal allocation mechanism.
   Preserve existing jobs and environments.
2. Transfer or check out this exact branch on the server and record its commit.
   Keep environments, public model downloads, inputs, caches, and outputs outside the checkout.
3. Create an isolated Python 3.12 environment for Boltz and install `applications/scorer/boltz2/requirements-boltz.lock` on Linux AMD64.
   The lock pins Boltz 2.2.1, Torch 2.7.1, and CUDA 12.6 libraries.
   Run `pip check`, `boltz predict --help`, and a Torch CUDA availability probe in that environment.
   Record the GPU, driver, Python, package versions, and allocation method.
4. Create a separate coordinator environment with Python 3.11 or newer and `applications/scorer/boltz2/requirements.txt`.
   For the three PSICHIC+ modes, use the documented PSICHIC+ environment and authorized model weights.
   Do not install Boltz into that environment.
5. Download the [pinned upstream affinity example](https://github.com/jwohlwend/boltz/blob/v2.2.1/examples/affinity.yaml) outside the checkout.
   Retain the original and a copy that adds only `msa: empty` to protein A.
   Keep all other target fields unchanged.
   Use the example's ligand B SMILES as the candidate CSV entry.
   No remote MSA service is authorized by this handoff.
6. Run the native CLI on the single-sequence example with the command below, using absolute paths for the environment, input, cache, and output.
   Bound each prediction to 30 minutes initially and retain the real exit status and native logs.
   Reduced sampling is an execution check, not a scientific evaluation.
7. Run the coordinator command below with the same target, candidate, native environment, cache, and prediction settings.
   Set `--boltz-python` to the isolated environment's Python and `--output` to a fresh directory.
8. If PSICHIC+ weights are available, run the three remaining modes using the [run guide](README.md#run-each-workflow).
   Use a small candidate file with duplicate IDs and an explicit shortlist size smaller than the input count.
   Confirm all input rows survive, selection is explicit, and both engines actually use the intended devices and environments.
   Report unavailable PSICHIC+ weights as a verification gap.

Native prediction options for the first GPU execution check follow.

```bash
/absolute/boltz-env/bin/boltz predict /absolute/inputs/affinity-single.yaml \
  --out_dir /absolute/results/direct-native \
  --cache /absolute/cache/boltz --model boltz2 \
  --accelerator gpu --devices 1 --seed 42 \
  --recycling_steps 1 --sampling_steps 2 --diffusion_samples 1 \
  --sampling_steps_affinity 2 --diffusion_samples_affinity 1 \
  --num_workers 0 --preprocessing-threads 1
```

Run this coordinator command from the repository root.

```bash
python -m applications.scorer.boltz2 \
  --mode boltz-only --candidates /absolute/inputs/molecules.csv \
  --target-yaml /absolute/inputs/affinity-single.yaml --ligand-id B \
  --msa-mode single --boltz-python /absolute/boltz-env/bin/python \
  --boltz-timeout 1800 --output /absolute/results/coordinator \
  -- --accelerator gpu --devices 1 --seed 42 \
  --cache /absolute/cache/boltz \
  --recycling_steps 1 --sampling_steps 2 --diffusion_samples 1 \
  --sampling_steps_affinity 2 --diffusion_samples_affinity 1 \
  --num_workers 0 --preprocessing-threads 1
```

## Compare native outputs and reports

Confirm each successful run produced a predicted structure and a native affinity JSON file.
Pass the direct native affinity file to `applications.scorer.boltz2.adapter.parse_affinity` with its candidate row.
Require exact equality between the parsed fields and that file's `affinity_probability_binary` and `affinity_pred_value`.
Also compare the coordinator's result fields against its own retained native affinity JSON under `boltz/native/boltz_results_inputs/predictions/row_00000001/`.
Inspect the rendered target to confirm the original complex and explicit MSA choice survived.

Record any numerical difference between the two separate inference runs.
Do not confuse stochastic differences between predictions with parser mismatches on the same JSON file.
Do not treat a timeout, missing output, or zero wrapper exit status as successful inference.

## Verify images and return evidence

Build and smoke-test both images using the [Docker instructions](README.md#build-the-psichic-only-image) if the server permits Docker.
Check that the PSICHIC+ image has no Boltz package and the combined image invokes the isolated Boltz environment.
Run at least the completed Boltz example inside the combined image with a GPU allocation and mounted cache.
Host-environment inference alone does not verify the Docker image.

Retain commands, actual exit codes, input hashes, model hashes, package versions, GPU details, native logs, structures, affinity JSON, coordinator manifests, and field-comparison assertions outside tracked source.
Update `VERIFICATION.md` with the completed checks and remaining gaps, and link portable evidence in the draft PR.
Do not claim better screening performance without an independent labeled evaluation.
Keep the PR in draft and do not merge it as part of this handoff.
