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

CUDA hardware was unavailable for these initial CPU checks.
The later A100 evaluation below completed the native workflow but failed its frozen numerical agreement check.
Slurm, AWS Batch, LSF, scheduler scale, shared storage behavior across machines, and cluster scaling have not been exercised.
The Docker test used an existing authorized local worker image rather than building a new distributable image.
The mixed-method test uses a failing fixture executable for native Vina dispatch, so it does not validate a Vina calculation.
The upstream Athena postprocessing wrapper was not run because these are shared-filesystem Bash runs.
The fixture helper extracts the native summary columns into the wrapper's existing ranking schema and invokes the real upstream selector directly.

## Independent verification after review fixes

Both independent code reviewers completed their full review.
All five actionable findings were fixed, including malformed sparse identifiers, invalid UTF-8 completion, the stdout regression, normalization timeout retirement, and forced-termination workspace cleanup.
The final installed wheel passed all 26 regression tests, Ruff lint and formatting checks, and whitespace checks.
Fresh host and Docker Linux CPU runs passed the complete native workflow again.
The linked JSON evidence records these final independent runs.

Actual native workunits retained a failed invalid-UTF-8 ligand while its valid companion succeeded.
Unsafe traversal, absolute, and missing-separator sparse identifiers produced native failures without an output escape or summary crash.
A targeted run blocked the real normalizer inside the unchanged production child.
Its actual failure response contained `TIMEOUT: Standardization timeout`, while its companion retained a successful affinity.
The supervisor reaped that child, removed its workspace, and recovered with the production child in generation two.
A second targeted run invoked the patched native dispatcher with a hung child, verified the failed completion and cleanup, and recovered through the same callback using the actual model.

The full independent run commands and outputs are retained locally in `runs/adaptiveflow-native-final-independent/commands.sh`.
The two targeted production lifecycle checks are retained in `runs/adaptiveflow-native-production-lifecycle/checks.py` and `checks.log`.
These final checks used the same application patch, SHA-256 `d85f8ff4e242c09b60016a460ef6fd100c4bdae37236c15db84574d342399561`, before this evidence record was updated.
These CPU checks do not establish CUDA numerical correctness, remote scheduler behavior, or scaling.

## A100 adaptive evaluation

The [A100 evidence](tests/evidence/native-a100.json) records one completed engineering run on 2026-11-02.
It uses human JNK3, UniProt P53779, with its full 464-residue sequence.
The input library contains 1,000,000 distinct canonical isomeric SMILES from existing Papyrus and PCBA datasets, with source identifiers retained.
Prepared native archives, sparse listings, native workunit commands, and the actual upstream `tranche_ave_score` selector were used.
No protein structure, conformer generation, 3D pose sampling, or docking was performed.

The native workflow passed its independent artifact audit.
The prescreen completed 4,061 predictions in 98.200 seconds.
The selector took 0.956 seconds and chose 104 full collections containing 100,871 molecules for a budget of 100,000.
The primary screen completed all 100,871 predictions in 1,512.791 seconds, or 14.997 milliseconds of amortized wall time per attempted or successful pair.
Both stages reported zero failures, skips, download failures, or retries.
There were 104,932 attempts, 104,424 distinct molecules, and 508 molecules evaluated in both stages.
The remaining 895,576 molecules have no prediction from this run.
The complete adaptive workflow took 1,625.378 seconds, including native preparation, selection, consolidation, ranking, export, and orchestration between stage timers.
Runtime construction, library preparation, standalone validation, diagnostic reruns, and two earlier aborted attempts are outside that duration.

This run used one scoring worker, one CPU thread per worker, batch size 128, one replica, conservative FP32 CUDA, and disabled TF32.
Each stage initialized the target and model once.
The A100 sampled peak device memory was 4,205 MiB.
The scorer occupied approximately one CPU core while sampled primary GPU utilization averaged 9.35 percent.
The idle device also reported unexplained utilization, so these readings do not establish exclusive GPU occupancy.
This is a measurement of the recorded configuration, not an optimized-throughput or implementation-speedup comparison.
The host's glibc 2.31 prevents optional `pyg-lib` and `torch-sparse` extensions from loading.
The exercised conservative dense CUDA path completed despite that limitation.

The frozen numerical agreement check failed.
All 128 standalone predictions from the deterministically selected sample succeeded, but one exceeded the absolute tolerance of 0.0001.
The maximum difference was 0.00014257431030229029 and the mean difference was 0.000006930902600271044.
Diagnostic repeats using the original batch also produced an adapter-versus-standalone difference of 0.00016164779663085938.
Batch composition alone does not explain the discrepancy, and no sole cause has been established.
The tolerance and failed validation record were preserved.
This run therefore does not establish an overall numerical validation pass.

An earlier attempted prescreen exposed a CUDA child shutdown failure after processing its full sample.
Commit `77e52b5` increases only the bounded post-SIGKILL reap wait from 0.5 seconds to 5 seconds.
The completed run used an installed wheel matching that commit's adapter SHA-256.
All 26 regression tests and Ruff checks passed, and an independent reviewer accepted the focused fix.
No shared Docker image or shared environment was changed.

Native commands, configurations, summaries, logs, failed attempts, and diagnostic artifacts remain under `Huan-MainA100:/mnt/hykoh3/adaptiveflow_pr5_eval_20261102`.
Use `PYTHONNOUSERSITE=1` with the recorded private runtime when reproducing selection.
The user-site pandas version changes tied collection ordering.
The source data may overlap training, and 58,713 library molecules have JNK3 source activity records.
The study does not measure adaptive recall or validate novel biological hits.
Remote schedulers and cluster scaling remain untested.
