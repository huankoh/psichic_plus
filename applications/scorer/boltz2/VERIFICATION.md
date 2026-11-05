# Verification record

The implementation starts from PSICHIC+ revision `93f357b97b91843e918bcc8ea04ce0e9d8574e9c`.
Checks ran on an ARM64 Mac with 24 GiB of memory on 2026-11-05.

## Application checks

The pre-change application suite passed with 26 tests and 4 skips in an environment without Boltz.
After implementation, review corrections, and installation of the existing pinned SyntheMol integration, the application and selected repository checks passed with 107 tests and 14 skips.
Two skipped checks require PSICHIC+ weights, eleven require a pinned SyntheMol source checkout, and one requires a pinned AFVS checkout.
The application environment still has no Boltz package.

```bash
PYTHONPATH=.:applications/screener python -m pytest \
  applications/scorer applications/generator/tests applications/screener/tests \
  tests/test_env_pins.py tests/test_supported_platforms.py \
  tests/test_single_ligand_implementation.py -q
python -m pip check
ruff check applications/scorer applications/generator/scorer.py
```

The controlled tests exercise all four CLI modes with a native-process fixture and controlled PSICHIC+ results.
They cover duplicate and blank IDs, original row order, shortlist ties, insufficient shortlists, ranking directions, weights, common-cohort ranks, engine failures, partial native exits, process timeouts, stale-output rejection, native file paths, malformed scores, and preserved native fields.
Native-target tests cover multiple chains, cofactors, local MSAs, template assets, constraints, incompatible binders, duplicate YAML keys, and native YAML merge precedence.
Additional regressions verify early rejection of mixed custom and automatic MSAs, finite ranking with extreme weights, and fingerprints of the PSICHIC+ source actually loaded.
Compatibility tests cover the legacy predictor module identity, constructor signature, relocated default screening path, SyntheMol, and the existing AdaptiveFlow adapter.
Controlled outputs do not establish predictive accuracy.

## Real upstream example

An isolated environment installed unmodified Boltz 2.2.1 with Python 3.11.18 and Torch 2.5.1.
The dependency check passed for 71 packages, and the real `boltz predict --help` command succeeded.
The [upstream affinity example](https://github.com/jwohlwend/boltz/blob/v2.2.1/examples/affinity.yaml) was changed only by adding `msa: empty` to protein A.
No remote MSA service was used.
The public model assets were downloaded to an external cache, and their SHA-256 values matched the recorded Hugging Face linked hashes.

The actual prediction command used these options.

```text
boltz predict affinity.yaml --out_dir outputs --cache cache/boltz --model boltz2
  --accelerator cpu --devices 1 --recycling_steps 1 --sampling_steps 2
  --diffusion_samples 1 --sampling_steps_affinity 2 --diffusion_samples_affinity 1
  --num_workers 0 --preprocessing-threads 1
```

Boltz accepted the target, reported explicit single-sequence mode, produced processed inputs, and began CPU structure prediction.
It exceeded the 360-second smoke-test limit before producing a structure or an affinity JSON file.
The original test wrapper recorded the timeout but returned zero.
That wrapper was corrected to return 124 on timeout.
The run is classified as incomplete regardless of the wrapper's original exit code.

Completed native inference and numerical comparison of adapter scores against a direct upstream result remain unverified.
The retained evidence includes the original example, modified input, download hashes, package list, native logs, processed inputs, and corrected reproduction script outside the tracked source tree.
The reduced sampling settings were an execution smoke test, not a scientific evaluation.

## Real coordinator execution

The final verifier ran the coordinator against the same unmodified Boltz 2.2.1 environment and upstream affinity example with a 45-second prediction timeout.
The native CLI reached CPU prediction, and the coordinator returned exit code 1 with a partial manifest.
The retained result preserves the original candidate ID, SMILES, and row key and records the timeout without any score or rank.
A second bounded run captured the actual prediction subprocess and confirmed that its environment omitted `PYTHONPATH` and `PYTHONHOME`.
Both runs used single-sequence mode and did not enable remote MSA generation.
The real coordinator also rejected a mixed custom and automatic MSA target with exit code 2 before creating an output directory.

The local evidence directory is `.audit/final-verification-v2/` in the enclosing workspace, outside this source checkout.
It contains `summary.md`, `pytest.log`, `boltz-run/manifest.json`, `boltz-run/results.json`, native logs and inputs, and `predict-process-isolation.log`.
These runs verify execution isolation, native input handling, and failure reporting.
They do not establish completed inference or numerical parity.

## Containers

`applications/Dockerfile` remains unchanged and does not install Boltz.
The combined Dockerfile adds a separate environment and includes build-time dependency, version, import-isolation, and CLI checks.
The Linux AMD64 CUDA requirements resolve to a hash-locked set with Boltz 2.2.1, Torch 2.7.1, and cuEquivariance 0.5.1.
The initial attempt with Torch 2.6.0 failed dependency resolution because its CUDA libraries conflict with the Boltz CUDA extras.
That conflict is isolated from the existing PSICHIC+ environment.

Docker is not installed on the verification Mac.
Neither image was built or smoke-tested here, and GPU inference was not run.
Dependency resolution is not evidence of a successful image build or GPU execution.

Regenerate the lock using the command recorded in its generated header.
Build and smoke-test both images with the commands in the [run guide](README.md#build-the-psichic-only-image) on a Linux AMD64 Docker host.
