# Final A100 screening rerun, 2026-11-07

At revision `8236e00`, the final native screening rerun passed its audits for every mode on `ai-gpcr-1`.
Full ensemble checked 584 artifact digests and the concentration formula.
Pyramid ensemble checked 246 digests, pyramid refinement checked 246 digests, Boltz-only checked 548 digests, rank-v1 checked 584 digests, and PSICHIC-only checked 36 digests.
The fixture used a genuinely read-only AFVS checkout with modes 555 and 444.
Completed-run status and resume in fresh containers exited 0 with unchanged report hashes and invocation sets.
Two harness attempts were rejected by the local GPU gate on transient utilization samples with no compute processes and zero final GPU memory.
The preserved retries passed.
These runs used quick-check sampling and prove execution, transfer, and score handling only.
They do not prove normal-sampling prediction quality, live cluster scheduling, or multi-replica screening.
Evidence remains outside tracked source under `/mnt/hykoh3/psichic-merge-ready-20261107/screener-final` on the server and in the project evidence copy at `evidence/integration-20261106/merge-ready-20261107/screener-final`.

# Durable screening integration, 2026-11-06

The author CPU qualification passed 28 checks against the real pinned Linux AFVS preparation and worker, with controlled predictions and qualified runtime evidence.
Native workers ran under Python 3.10.14 and launched the separate Python 3.12 inference supervisor.
All six configurations passed: PSICHIC-only, Boltz-only, full concentration-v1, global K=2 concentration-v1, global K=2 refinement, and full rank-v1.
The four-entry population retained duplicate names across collections and a repeated identical source pair across multiple workunits.
The final follow-up passed two additional checks for legacy `rank --source-run` and installed packaging.
An old native affinity of 99 remained metadata while the fresh qualified observation supplied affinity 6.

The fault cases passed owner-death cleanup, live-process reconciliation guards, explicit stopped recovery, sparse selection across bounded preparation windows, retry with immutable report history, and shortage with zero Boltz workunits.
Transfer checks passed short copies, idempotent redelivery, contradictory delivery rejection, superseded attempts, wrong run identity, artifact tampering, missing loaded-runtime evidence, and zero-exit overview rejection.
Installed console and module validation ran outside the checkout, with human and JSON output, invalid-option errors, no inference during validation, and actual Python 3.10 worker imports.
The rich legacy transport retained valid affinity when same-pass distribution validation failed.

A separate saved-artifact audit checked 392 artifact references across seven completed runs, reproduced standalone scores and selections from identical saved observations, and matched the hash-verified original concentration formula at `1e-12`.
All model observations in this author qualification were controlled test values.
No trained-model GPU inference or live scheduler submission was performed in this unit.
The CPU containers had no GPU device request, network disabled, and read-only source, image root, model assets, and dependency mounts.
Independent GPU and deployment qualification remain separate work.

The source-level reproduction entry points are `tests/test_integration.py`, `tests/test_transfers.py`, `tests/test_packaging.py`, and `tests/compare_saved.py`.
Set `AFVS_TEST_UPSTREAM` to the pinned reference and `AFVS_WORKER_PYTHON` to the dependency-equipped Python 3.10 native worker.
The controlled Boltz fixture can use `CONTROLLED_BOLTZ_PYTHON` for its isolated interpreter.
Exact commands, immutable source snapshots, retained failures, raw native archives, process records, and the unexecuted independent A100 handoff are in the project-wrapper evidence directory `evidence/integration-20261106/screener-integration-implementation`.
See [SCORING.md](SCORING.md) for the deployment boundary and public commands.

> The remaining sections are historical verification from the original pull request, before relocation to `applications/`.
> Commands and paths below describe that earlier layout.

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
CUDA, remote schedulers, and scaling remain unverified.
