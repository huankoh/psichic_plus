# Run PSICHIC+ in native AdaptiveFlow-VS workunits

Use your existing prepared AdaptiveFlow-VS libraries, storage configuration, worker count, and scheduler.
This adapter adds `docking_scenario_methods=psichic_plus` to [AdaptiveFlow-VS revision df74e005e456e94654fa93a40d748de399450b81](https://github.com/QuantumAI4Bio/AdaptiveFlow-VS/tree/df74e005e456e94654fa93a40d748de399450b81).
It scores native per-ligand `.smi` archives and leaves collection selection to the upstream workflow.
Affinities are model predictions, not experimental measurements.

## Install on the submission host and every worker

Use a working PSICHIC+ environment from this repository, including its installed `psichic-ligand` package, matching RDKit build, PyTorch, and ESM model access.
Provide the upstream Python dependencies, including `boto3`, `jinja2`, `pyarrow`, and `IPython` for its selector.
The adapter does not replace that environment or install another PyTorch stack.

Build the adapter wheel from this repository root.
The wheel includes the existing screening Python modules and the checked upstream patch.
It contains no model weights.

```sh
python -m pip wheel --no-deps --no-build-isolation integrations/adaptiveflow --wheel-dir runs/afvs-wheels
python -m pip install --no-deps runs/afvs-wheels/psichic_adaptiveflow-0.1.0-py3-none-any.whl
python -m psichic_adaptiveflow.install /absolute/path/to/AdaptiveFlow-VS
```

Install the same wheel into the Python environment used by `afvs_run.py` on **every worker**.
An installation on the login node alone is insufficient.
The installed package supplies its screening imports independently of the working directory.
Use the activated environment's actual Python executable rather than a symlink inside an unrelated virtual environment.

The installer accepts only the pinned revision.
It runs `git apply --check` before applying the patch and recognizes its exact previous application.
It refuses changed revisions, staged patch targets, and conflicting or additional edits in the two patched files.
Only the explicitly supplied checkout is modified.
Keep other local changes outside those files.

## Configure the usual native inputs

Keep your existing `all.ctrl` storage and scheduler settings.
Set these values in `AdaptiveFlow-VS/tools/templates/all.ctrl` for one scenario named `target`.
Use the upstream colon-separated settings for multiple scenarios.

```ini
ligand_library_format=smi
docking_scenario_names=target
docking_scenario_methods=psichic_plus
docking_scenario_replicas=1
docking_scenario_batchsizes=32
docking_scenario_basefolder=../input-files
threads_per_docking=1
program_timeout=600
print_attrs_in_summary=smi,psichic_affinity
summary_formats=parquet,csv.gz
run_atom_check=0
dynamic_tranche_filtering=0
```

Set the timeout for a whole native batch, including cold model and ESM initialization.
Batch size bounds inference memory and each private request has an 8 MiB limit.
The adapter honors `threads_per_docking` and uses FP32, input-order batches, no interpretation, and no data-loader workers.
Each AFVS docking worker owns one model child per PSICHIC+ scenario and reuses it across batches.
AFVS still decides the number of workers and workunits.

Create `input-files/target/config.txt` with these key-value settings.

```ini
protein_fasta=target.fasta
model_dir=/absolute/worker-visible/authorized-weights/PSICHIC_plus
device=cpu
```

Place your single-record FASTA at `input-files/target/target.fasta` so native workunit preparation packs it with the scenario.
A relative `protein_fasta` resolves against the scenario directory, including on a worker after unpacking.
An absolute FASTA path must also exist on every worker.
`model_dir` must be an absolute path visible on every worker and contain `config.json`, `degree.pt`, and `model.pt`.
Weights are not packed into workunits.
Set `device=cuda` or `device=cuda:0` for an assigned CUDA device.
The selected CUDA device also receives protein initialization.
CPU mode hides CUDA before importing PyTorch or the featurizers.

The adapter validates format and scenario files during native folder preparation and repeats the checks after worker unpacking.
Only per-ligand SMILES files are supported.
Each archive member contains one SMILES line, optionally followed by a whitespace-separated name.
Do not enable the upstream atom or tranche filters that inspect structure-file metadata.
Select existing tranches with your native todo files and selector settings.
No CSV conversion or new tranche labels are required.

## Run the existing preparation and scheduler commands

Run these commands from `AdaptiveFlow-VS/tools`.

```sh
python afvs_prepare_folders.py
python afvs_prepare_workunits.py
python afvs_submit_jobs.py 1 1
```

Use your actual final workunit number in the submission range.
Slurm, AWS Batch, and LSF continue to use their configured templates and submission commands.
For the Bash backend, run each generated workunit script with Bash.
For a one-workunit example, use this command from `tools`.

```sh
bash ../workflow/workunits/1/run.bash
```

The upstream aggregate Bash launcher references workunit files that are not executable at this pinned revision.
Calling their generated scripts with Bash avoids that launcher permission issue.
Use a worker temporary directory supported by your host in your native Bash template.

Successful native summaries retain `ligand`, `collection_key`, `scenario`, replica scores, `score_average`, and `score_min`.
`attr_psichic_affinity` contains the raw affinity prediction and every native score is its negative.
Lower native scores therefore rank higher predicted affinity first.
Replicas keep the native mean and minimum aggregation, with one replica as the default.
For multiple replicas, `attr_psichic_affinity` is the first completed replica's raw value, and `-score_0`, `-score_1`, and subsequent score columns retain each raw value.
FP32 results can differ slightly between batch positions.
Failures appear in the native overview's `failed_list` with collection, ligand, scenario, replica, stage, and reason.
They also have archived per-item diagnostics.
Every submitted item produces one native completion, including a batch that times out or loses its child.
A failed model child is terminated and reaped before the next batch starts a fresh child.
A normalization timeout also retires the child after preserving the completed batch results.
The worker owns the child workspace and removes it after the child is reaped, including after forced termination.
A child that cannot be reaped disables respawn and reports failures.

Original IDs remain unchanged.
PSICHIC+ summaries distinguish identical ligand IDs in separate full collections.
Its archived log directories include the source archive collection and original full collection before ligand and replica.
A sparse archive must have unique member ligand IDs and unique IDs in its `.listing`.
Ambiguous archives are rejected visibly before extraction can overwrite a ligand.
An archive without readable members reports an archive failure with an empty ligand identifier.

## Continue from prescreen to primary screen

Use your existing prepared sparse archives, `.listing` indices, and native `prescreen_mode=1` settings.
The upstream `prescreen_ligands_per_tranche` controls which existing indices run.
The adapter preserves the full collection identifiers from `.listing`.

Postprocess the native summaries with your normal workflow.
The upstream AWS postprocessing wrapper uses Athena and hardcodes dataset locations in its selector wrapper.
For a shared-filesystem run, extract the native `collection_key`, `ligand`, and `score_min` columns into its existing four-column ranking format `Tranche,Collection,LigandAFID,ScoreMin`.
Split `collection_key` at its native underscore separator.
Pass that gzipped ranking and your **actual complete-library inventories** directly to the upstream selector.

```sh
python templates/create_todofile_atg-primaryscreen.py \
    /results/target.ranking.subset-1.csv.gz \
    /library/complete.collections.parquet \
    /library/complete.tranches.parquet \
    tranche_ave_score /results/primary.todo 100000 '.*'
```

The final arguments are the primary-screen ligand target and your existing tranche filter regex.
This is the upstream whole-collection selector, so the selected count can exceed the target.
Point the primary screen's native `collection_list_file` at the selected todo file, set `prescreen_mode=0`, and use the full prepared library's original archive paths.
Run the same folder, workunit, and submission commands in the primary-screen workflow directory.
To score the full library directly, use `prescreen_mode=0` with its existing full-library todo file.

## Use your existing Docker worker runtime

Install the wheel inside your authorized PSICHIC+ worker runtime.
No shared Dockerfile or image is changed by this integration.
This example uses a prebuilt image selected by you and writable mounts for the external AFVS checkout and its results.
Configure `all.ctrl` for `/library` and `/afvs/input-files`, and set the scenario's `model_dir=/weights`.
Mount ESM's cache if your workers cannot download it.

```sh
docker run --rm --gpus all \
    -v "$PWD/runs/afvs-wheels:/wheels:ro" \
    -v "$AFVS_CHECKOUT:/afvs" \
    -v "$PREPARED_LIBRARY:/library:ro" \
    -v "$AUTHORIZED_WEIGHTS:/weights:ro" \
    -v "$TORCH_CACHE:/root/.cache/torch" \
    -w /afvs/tools "$PSICHIC_WORKER_IMAGE" bash -c '
        python -m pip install --no-deps /wheels/psichic_adaptiveflow-0.1.0-py3-none-any.whl &&
        python -m psichic_adaptiveflow.install /afvs &&
        python afvs_prepare_folders.py &&
        python afvs_prepare_workunits.py &&
        python afvs_submit_jobs.py 1 1 &&
        bash ../workflow/workunits/1/run.bash
    '
```

For CPU mode, omit `--gpus all` and set `device=cpu`.
Include AFVS's Python and command-line dependencies in your chosen worker runtime.
Docker CPU verification is recorded in the verification evidence.
An A100 native adaptive run completed over a million-input library, but its frozen numerical agreement check failed.
See the [A100 verification record](VERIFICATION.md#a100-adaptive-evaluation) for counts, timing, and the unresolved numerical difference.
Slurm and AWS execution and cluster scaling remain unverified.

## Reproduce the fixture verification

The [example](example) contains tiny **prepared** native archives and todo files with their original fixture names.
The short FASTA is only a software test target.
The invalid and empty inputs intentionally exercise visible failures.
Fixture generation belongs only to the test helper, not the user workflow.

Install the wheel, then run the tests with your PSICHIC+ Python environment.
Provide a pinned external checkout for installer tests.
Use an ignored copy of that checkout for the native run, never a shared upstream source checkout.

```sh
AFVS_TEST_UPSTREAM=/absolute/path/to/pinned-afvs python -m pytest -q integrations/adaptiveflow/tests/test_adapter.py
python integrations/adaptiveflow/tests/run_native.py \
    --upstream /absolute/path/to/patched-afvs-copy \
    --model-dir /absolute/path/to/authorized-weights/PSICHIC_plus \
    --output /absolute/path/to/new-test-output
```

The helper runs real native preparation, submission, generated Bash scripts, the upstream `tranche_ave_score` selector, primary preparation, and the full library without prescreening.
It compares raw affinities with standalone direct inference and checks replicas, repeated IDs, archived logs, invalid inputs, partial batches, and mixed methods with an all-failure scenario first.
The legacy method in the mixed-method fixture uses a test-only failing executable, not a real Vina calculation.
On macOS only, a test-specific `sitecustomize.py` supplies the unavailable `Queue.qsize` operation and selects `fork` for native multiprocessing.
A copied test Bash template supplies a local temporary directory instead of `/dev/shm`.
These adjustments do not ship in the adapter or upstream patch.
See [verification evidence](VERIFICATION.md) for observed results and limits.

## Software and weight terms

The bundled patch modifies AdaptiveFlow-VS files under their upstream GPL-2.0-or-later notices.
Its complete upstream license is included as [COPYING](psichic_adaptiveflow/COPYING), and the patch carries modification notices.
The adapter and bundled screening modules remain subject to this repository's [non-commercial research license](../../LICENSE) and dependency notices.
Model weights and ESM weights have separate rights and access requirements.
Supply only weights you are authorized to use.
No weights are distributed in the adapter wheel.
The process boundary does not establish license compatibility, and this integration makes no such claim.
