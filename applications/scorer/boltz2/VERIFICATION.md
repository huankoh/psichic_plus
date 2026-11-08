# Verification record

## PSICHIC+ and Boltz combo verification on 2026-11-06 and 2026-11-07

This section records the merge-ready checks for the combined PSICHIC+ and Boltz branch.
Each check below names the revision it actually ran.
Evidence remains outside tracked source under `/Users/huanyeekoh/.t3/projects/psichic-workflow/evidence/integration-20261106/` on the authorized PC and under `/mnt/hykoh3/psichic-merge-ready-20261107/` and `/mnt/hykoh3/psichic-e2e-20261106/` on `ai-gpcr-1`.

### Passed

- At revision `10e1f5b33f8fd9fd3a3827c7dd8c88563ea0986b`, direct native Boltz and every durable mode exited 0 on the A100.
  The modes were PSICHIC-only, Boltz-only, full ensemble, pyramid ensemble with K=1, and pyramid refinement with K=1.
  Every reported Boltz score matched its retained native affinity JSON exactly.
  The runs used single-sequence MSA mode, the quick-check preset, and seed 42.
- At revision `8236e00`, the final native screening rerun passed its audits for every mode on the A100.
  Full ensemble checked 584 artifact digests and the concentration formula.
  Pyramid ensemble checked 246 digests, pyramid refinement checked 246 digests, Boltz-only checked 548 digests, rank-v1 checked 584 digests, and PSICHIC-only checked 36 digests.
  The screening fixture used a genuinely read-only AFVS checkout.
  Completed-run status and resume in fresh containers exited 0 with unchanged report hashes and invocation sets.
- Two screening harness attempts were rejected by the local GPU gate on transient utilization samples with no compute processes and zero final GPU memory.
  The preserved retries passed.
  Those gate rejections were harness flakes, not product failures.
- At revision `48f12cdf0d0696a15a001afe9956bb22db7e1f92`, both Docker images built and passed smoke tests on the A100 host.
  The base image is `psichic-workflows:merge-ready-48f12cdf0d06`, ID `sha256:591913d37c2596aecc82299c418a507345f1f40e8227ee39cdc73e91950f89c0`.
  The combined image is `psichic-boltz:merge-ready-48f12cdf0d06`, ID `sha256:e31204d477f41cb3e81dfd4c031da666b5b8cac5086493a01b6e1ee3f4f70318`.
  Base smoke confirmed the installed CLIs, the ligand contract, clean dependencies, and the absence of Boltz in the coordinator environment.
  Combined smoke confirmed Boltz 2.2.1, Torch 2.7.1+cu126, clean dependencies in both environments, and the recorded source revision.
  The only change from `7531e95` to this revision is the `./applications` install path in `applications/Dockerfile`.
- The broad local core suite passed 455 tests with 15 skipped before the final review fixes.
  The root CI selection passed 417 tests locally.
  The SyntheMol generator file passed all 30 tests on rerun with the pinned runtime overlay first on `PYTHONPATH`.
  The first generator attempt failed 11 tests with `ModuleNotFoundError: No module named 'psichic_plus_synthemol'` because that overlay was missing.
  The focused review-fix tests passed 4 tests and the Predictor slice passed 78 tests with 4 skipped after the read-only AFVS and helper-registration fixes.
- At revision `48f12cdf0d0696a15a001afe9956bb22db7e1f92`, the full generator test suite passed 35 tests with 19 skipped inside the combined Docker image on the A100 host.
  The controlled Boltz fixture used `CONTROLLED_BOLTZ_PYTHON=/opt/boltz/bin/python` and the container ran with the GPU driver files mounted.
  Two earlier harness attempts failed five Boltz-mode tests each for those two missing setup settings, and those logs are retained.
- Five additional local executor tests failed on the development Mac with 120-second subprocess timeouts while the machine load was between 49 and 88 from unrelated work.
  The same test that failed in the first A100 container attempts passed individually once the qualified interpreter and GPU mount were present.
- At revision `48f12cdf0d0696a15a001afe9956bb22db7e1f92`, the full core suite inside the combined Docker image on the A100 host reported 449 passed, 20 skipped, and 2 failed in 1855.99 seconds.
  Both failures were in `applications/scoring/tests/test_native_build.py`.
  The real native compiler contract ran under the coordinator Python, whose Triton is 3.2.0, instead of the qualified Triton 3.3.1 interpreter.
  The stamp test asserted that unlinking and rewriting a file changes its filesystem stamp, and the A100 container filesystems returned identical stamps in 20 of 20 probes.
  Neither failure showed a product error.
- A test-only follow-up at revision `b4e013ac290e3579f910072f143e9c70efb8c9d4` skips the qualified lane under an unqualified Triton interpreter and removes the filesystem stamp assumption.
  No product code changed in that commit.
  After the fix, that test file passed 5 tests with 1 skipped on the Mac and inside the combined image.
  The qualified contract passed on its own under `/opt/boltz/bin/python` inside the combined image.
  Two direct native probe runs under that interpreter also passed with the same profile digest `23ece49f70238977c3ec9f5cd2673e006389803dab07820255fdd75e91a2e13d`.
  A parallel Mac core rerun was stopped after 35 passes with no failures shown, so it is not counted as a completed suite.

### Limits

Quick-check sampling proves execution and native score parsing only.
It does not prove normal-sampling prediction quality, ranking quality, or scientific accuracy.
Live cluster scheduling, cross-host or S3 coordination, and multi-replica screening remain unqualified.
Only one AdaptiveFlow replica is supported.
`verify_files` now requires recorded runtime files to exist and does not compare their contents.
Runtime assets must therefore remain immutable during execution.
Loaded Python source is still verified, and a later resume still reprobes the recorded profile.
No remote MSA service was used, upstream Boltz source was not modified, and no FEP or ABFE was implemented.

## GPU verification on 2026-11-05

The GPU continuation tested handoff revision `988358b641c5a2fa16084bc8481fd6cfbca99504` on host `ai-gpcr-1`, reached through the authorized `Huan-MainA100` SSH alias.
The device was GPU 0, NVIDIA A100-PCIE-40GB, UUID `GPU-0def26fd-cf53-a700-89fc-d07bbab6dfc3`, with driver 535.247.01.
No scheduler or active GPU processes were found before execution.
Checks ran serially on that GPU, without changing existing environments, stopping services, or modifying upstream Boltz code.
Docker used its existing `/mnt/docker` storage.

### Passed

- The hash-locked Linux environment installed Boltz 2.2.1, Python 3.12.8, and Torch 2.7.1+cu126.
  Dependency checks, native CLI help, and a CUDA tensor operation passed.
  All 107 installed Boltz Python source files matched their package RECORD hashes.
- The pinned upstream affinity example completed native GPU structure and affinity prediction with exit code 0.
  The only semantic input change was `msa: empty` on protein A.
  The original and modified inputs are retained.
  Seed 42, one recycling step, two structure sampling steps, two affinity sampling steps, and one sample for each stage were used, with zero workers and one preprocessing thread.
- The separate coordinator completed the same target and settings with exit code 0.
  Direct native scores were `affinity_probability_binary = 0.44567131996154785` and `affinity_pred_value = 2.4228515625`.
  Both parsed fields matched their actual native JSON exactly, and the direct/coordinator difference was zero for both fields.
- Both Dockerfiles built successfully from the handoff revision.
  The PSICHIC+ base image passed generator CLI help and confirmed Boltz was absent.
  The combined image passed both CLI help checks and ran Boltz through `/opt/boltz/bin/python`.
  Base image ID: `sha256:8b92e07c5f60e6d7cba220898d11a322d431c5cb518db883f0c0ccd0d71bacd5`.
  Combined image ID: `sha256:1d2161ca91fa874b9151741f783a3ecff6f56143fea1a649aff17f42020228a6`.
- All four workflows completed inside the combined image on the GPU, with exit code 0 and complete manifests.
  Boltz-only scored one row; full ensemble scored two rows with both engines.
  Each pyramid workflow scored two rows with PSICHIC+ and selected exactly one for Boltz.
  Duplicate user IDs and original row order survived, excluded rows remained unranked, and all selected Boltz fields matched their own native JSON exactly.
  Predicted CIF structures were present for every selected row.
  PSICHIC+ used authorized weights mounted read-only and `cuda:0`; its loaded source fingerprints and weight hashes are retained.
- The focused coordinator suite passed 56 tests on both the Mac and Linux.
  The broader application and selected repository suite passed 107 tests with 14 external-resource skips inside the new base image.
- The weights-enabled predictor suite passed all three tests inside the combined image.
  The real CPU and CUDA cases verified duplicate IDs, invalid-ligand failure handling, model placement, and GPU memory behavior.
  This separately closes the two weight-dependent skips in the broader suite.
  Ruff 0.17.3 passed the prior baseline rule set, explicitly selected as `E4,E7,E9,F`.

### Evidence and reproduction

The complete server evidence remains outside the checkout at `/mnt/hykoh3/boltz2-pr7-20261105/`.
A portable copy of logs, commands, inputs, structures, native JSON, reports, versions, image IDs, and hashes is retained on the authorized PC at `/Users/huanyeekoh/.t3/projects/psichic-workflow/evidence/server/`.
The local evidence index is `/Users/huanyeekoh/.t3/projects/psichic-workflow/evidence/summary.md`.
Model caches and environments remain on the server and are excluded from the portable copy.
Reproduction scripts are `run-host.sh`, `run-docker.sh`, `run-modes.sh`, `run-weight-tests.sh`, and `compare.py`.
The comparison assertions and separate-run differences are recorded in `comparison.json`.
Actual exit codes are recorded in `*.exit`, and the exact commands appear in the run logs.

### Limits and remaining checks

No native, coordinator, or container execution check failed in this continuation.
Upstream emitted RDKit pickle-version warnings but completed every requested prediction; those warnings remain in the native logs.
Tiny PSICHIC+ floating-point differences between identical candidates can change their shortlist order across runs.
Exact equality was established for parsing each native JSON, not general cross-run determinism.
The broader suite retained its 14 external-resource skips; its two weight-dependent cases passed in the separate weights-enabled run.
The remaining 12 skipped checks require pinned SyntheMol and AFVS source checkouts and were not rerun here.
These reduced-sampling checks establish execution and score parsing only.
Default-sampling scientific accuracy, screening enrichment, multi-GPU execution, and other hardware remain unverified.
No remote MSA service was used, and no FEP or ABFE implementation was added.
PR #7 remains a draft.

## Earlier CPU verification (historical)

The limitations below describe the earlier Mac session and are superseded by the GPU results above where applicable.

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
