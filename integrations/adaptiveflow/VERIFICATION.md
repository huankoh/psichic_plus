# Native verification evidence

The implementation was verified against AdaptiveFlow-VS `df74e005e456e94654fa93a40d748de399450b81` on 2026-11-01.
All fixtures are software tests with a short synthetic protein sequence and small public molecule SMILES.
No model weights are included in the wheel or verification artifacts.

## Observed results

Both the [macOS CPU run](tests/evidence/native-macos.json) and the [Docker Linux CPU run](tests/evidence/native-docker-cpu.json) passed the same real native workflow.
The installed wheel supplies all inference modules when the working directory is a copied AFVS `tools` directory.
The full-library workunit used one docking worker, two replicas, and a native batch size of two.

The runs produced these results.

- Native prescreen preparation, workunit preparation, submission, and generated Bash execution completed four submitted items.
- The actual upstream `tranche_ave_score` selector consumed the prescreen results and complete-library inventories, then selected `AA_0000001` for a ligand target of one.
- Native primary preparation and generated workunit execution consumed that selected todo file and the original full-library archives.
- A full-library run with prescreening disabled completed 16 items and emitted six distinct success rows plus four visible invalid or empty-input failures.
- Both full collections retained ligand ID `shared`, with four distinct archived replica logs.
- The model child initialized the target and model once across seven batches, including partial batches.
- A mixed-method run completed eight items, and an all-failure legacy scenario before PSICHIC+ did not suppress the later successful summary.
- An archive containing duplicate member ligand IDs produced a visible failure before extraction could overwrite a ligand.
- Real subprocess fault fixtures hung before reading, during simulated initialization or prediction, or after a partial response.
  Each failed batch completed, its child was reaped, and the next request successfully used the actual PSICHIC+ model.

The maximum difference from standalone direct inference was `0.000005723` on macOS.
The Docker result and replica differences are recorded in its linked JSON artifact.
Checks use a `0.00001` FP32 tolerance for repeated inference and require exact sign conversion for the first raw result and exact native mean and minimum aggregation.
The summary attribute retains the first completed replica's raw affinity, matching native attribute aggregation.
Each individual replica's raw affinity is also retained as the negative of its native `score_N` column.

The 17 subprocess, archive, configuration, and installer regression tests passed on macOS.
They cover transport EOF, malformed correspondence, nonfinite scores, partial writes, bounded TERM and KILL, a descendant that ignores TERM, parent death, stdout diagnostics, recovery, and exact patch idempotence.
The installer tests also run actual native preparation and verify early rejection of an unsupported library format.

## Runtime and reproduction

The host run used Python 3.12, PyTorch 2.6.0, RDKit 2024.03.3, the repository checkpoint, and the existing cached ESM model.
It used a private wheel installation overlay without modifying the shared runtime environment.
On macOS, the test helper changes only a copied Bash template's temporary directory and provides `fork` and a test-only `Queue.qsize` fallback through `sitecustomize.py`.
Those adjustments are absent from the adapter and checked upstream patch.

The Docker run used the already available `psichic-workflows:local` image, ID `sha256:390c49ac0e100a4de9f1976f9f16342a1f7b67ee4b0d5803b671544425453dcd`.
It ran the image's Linux amd64 CPU runtime under Docker Desktop on an ARM host with PyTorch `2.6.0+cpu` and RDKit `2024.03.3`.
The image was neither rebuilt nor retagged.
The wheel was installed only inside the disposable container.
Weights and the ESM cache were mounted read-only.
The final run explicitly executed its verification driver as container PID 1 and passed child recovery with that legitimate parent.
Linux native queue operations were used without the macOS compatibility module.

The [README](README.md#reproduce-the-fixture-verification) gives portable rerun commands.
The local implementation run retained full commands and native outputs under `runs/adaptiveflow-native-design/`.
Its final host artifacts are in `native-final-code/`, and its Docker artifacts are in `docker-final/`.
`native-final-code-console.log`, `docker-final-console.log`, and `pytest-final.log` record the corresponding executions.
Each native case keeps preparation and submission logs, generated scripts, worker `.out` files, overview files, parquet summaries, and archived per-item logs.
Earlier failed attempts remain in separate directories for diagnosis.

The checked patch changes only `tools/afvs_prepare_folders.py` and `tools/templates/afvs_run.py` in explicitly supplied external checkouts.
The original external reference checkout was still clean at the pinned revision after verification.
`git diff --check` passes for the applied upstream files and this integration.
The wheel was inspected and contains no `.pt`, `.pth`, or `.ckpt` files.

## Remaining verification

CUDA hardware was unavailable, so GPU execution has not been verified.
Slurm, AWS Batch, LSF, scheduler scale, shared storage behavior across machines, and cluster scaling have not been exercised.
The Docker test used an existing authorized local worker image rather than building a new distributable image.
The mixed-method test uses a failing fixture executable for native Vina dispatch, so it does not validate a Vina calculation.
The upstream Athena postprocessing wrapper was not run because these are shared-filesystem Bash runs.
The fixture helper extracts the native summary columns into the wrapper's existing ranking schema and invokes the real upstream selector directly.

Independent review, independent reruns, and PR creation belong to the root task and remain pending.
