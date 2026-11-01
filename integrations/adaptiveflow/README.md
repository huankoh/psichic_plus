# Screen a CSV library with AdaptiveFlow and PSICHIC+

This integration runs AdaptiveFlow-VS's sparse prescreen, tranche-min collection selector, and primary screen on a local shared filesystem.
It uses one scenario, one replica, and one docking child per stage.
The child loads the PSICHIC+ model and target lazily, then reuses them across batches.
Scoring uses the conservative FP32 inference path and the existing `emit_null` invalid-ligand policy.
Outputs are model predictions, not experimental affinity measurements.

## Install the external workflow

Use an installed PSICHIC+ runtime with its native ligand featurizer, authorized model weights, and ESM checkpoint access.
Install these additional dependencies in that runtime.

```bash
python -m pip install -r integrations/adaptiveflow/requirements.txt
git clone https://github.com/QuantumAI4Bio/AdaptiveFlow-VS.git /path/to/afvs
git -C /path/to/afvs checkout df74e005e456e94654fa93a40d748de399450b81
```

Keep the upstream `tools` directory clean.
The runner checks the revision and imports the separately installed `tools/templates/afvs_run.py` without modifying it.

## Run a screen

Prepare one FASTA record and a CSV with these columns.
Keep collection numbers as strings, including their leading zeros.
Tranches must start with an ASCII letter and contain only ASCII letters and digits.
Tranche names must be distinct without regard to case.
Names that upstream CSV parsing treats as missing values or booleans, such as `NA`, `NULL`, and `True`, are rejected before conversion.
Collection IDs must contain only digits because upstream collection keys use `tranche_collection`.

```csv
molecule_id,collection_id,tranche,smiles
aspirin,0000001,AA,CC(=O)Oc1ccccc1C(=O)O
caffeine,0000001,AA,Cn1c(=O)c2c(ncn2C)n(C)c1=O
aspirin,0000042,BB,CC(=O)Oc1ccccc1C(=O)O
invalid,0000042,BB,invalid
```

Run from the PSICHIC+ checkout with a new output directory.

```bash
python integrations/adaptiveflow/run.py \
	--upstream /path/to/afvs \
	--library /path/to/library.csv \
	--target-fasta /path/to/target.fasta \
	--model-dir /path/to/authorized/PSICHIC_plus \
	--output runs/adaptiveflow \
	--samples-per-tranche 2 --budget 2 --batch-size 2 --device cpu
```

The prescreen takes the first `--samples-per-tranche` rows of each tranche in CSV order.
Shuffle the CSV beforehand if you need random sampling.
The primary budget counts ligands in whole collections, so upstream selection can exceed the budget by the final collection size.
Prescreen evaluations are additional to that budget, and primary screening can evaluate sampled ligands again.
`--tranche-regex` defaults to `.*` and is passed explicitly to the upstream selector.
Tranches without successful prescreen scores follow upstream's unscored-tranche behavior.
If every prescreen row fails, selection stops and the prescreen outputs retain the failures.
The converter loads the CSV into memory.

Use `--device cuda:0` in a compatible GPU runtime.
The default multiprocessing method is `spawn`, which keeps model initialization out of the parent.
`--start-method fork` is also supported for CPU execution on systems that provide it.
The worker supplies a portable queue counter for upstream backpressure on macOS and uses the stage's temporary directory instead of `/dev/shm`.

## Inspect the results

Open `ranked.csv` for successful predictions followed by visible failures.
Each row has a unique internal `ligand_id`, the original `molecule_id`, `collection_id`, tranche, and verbatim SMILES.
Duplicate original IDs and duplicate SMILES remain separate rows.
Successes have finite `predicted_binding_affinity` and `score = -predicted_binding_affinity`, so lower upstream scores rank higher affinities first.
Failures have a reason and empty affinity and score fields.
When a row is evaluated twice, `ranked.csv` keeps its primary result.
Rows never selected for evaluation remain in `library/manifest.csv`.

Each stage retains `predictions.csv`, its upstream workunit, preparation logs, and `worker.log`.
Upstream `output-files` contains the success summaries, the overview with `failed_list`, and the per-ligand JSON outputs and logs archive.
The hook emits `docking_complete` records with per-ligand files and timing, leaving collection preparation, queues, summaries, and cleanup to upstream.
Other docking methods delegate to the original upstream dispatch.

`library/collections` contains the full and sparse tar collections.
Each tar has a collection-number directory, one JSON ligand file per row, and a `.listing` with original collection key, internal ligand key, and sample index.
The manifest and parquet collection inventory preserve collection-number strings.
`selection/prescreen.csv.gz` has `Tranche,Collection,LigandAFID,ScoreMin` in that order.
The external `create_todofile_atg-primaryscreen.py` receives that file, both inventories, `tranche_min_score`, the budget, and the regex.
Upstream preparation consumes `selection/primary.todo` to create the primary workunit.

## Mount the integration into an existing runtime

Use an existing compatible PSICHIC+ runtime image and mount the checkout, external upstream checkout, inputs, authorized weights, and output directory.
This integration does not build or publish an image.
The following example installs only its extra dependencies into an ephemeral overlay.

```bash
docker run --rm --entrypoint /bin/bash \
	-v "$PSICHIC_CHECKOUT:/workspace/psichic_plus:ro" \
	-v "$AFVS_CHECKOUT:/opt/afvs:ro" \
	-v "$INPUT_DIR:/input:ro" \
	-v "$MODEL_DIR:/models:ro" \
	-v "$OUTPUT_DIR:/output" \
	-v "$ESM_CACHE:/root/.cache/torch" \
	"$RUNTIME_IMAGE" -c '
	python -m pip install --target /tmp/adaptiveflow-deps -r /workspace/psichic_plus/integrations/adaptiveflow/requirements.txt &&
	PYTHONPATH=/tmp/adaptiveflow-deps python /workspace/psichic_plus/integrations/adaptiveflow/run.py \
		--upstream /opt/afvs --library /input/library.csv --target-fasta /input/target.fasta \
		--model-dir /models --output /output/screen --samples-per-tranche 2 \
		--budget 2 --batch-size 2 --device cpu
	'
```

This mount command has not been container-tested because the verification host has no Docker daemon.
For CUDA, add the runtime's GPU options and select `--device cuda:0` after validating the runtime on your hardware.

## Verify with real inference

Run the retained verification script in the same runtime with a new output directory.
It invokes the real upstream worker and selector, then compares against the standalone PSICHIC+ CLI.

```bash
python integrations/adaptiveflow/verify.py --upstream /path/to/afvs --output runs/adaptiveflow-check
python -m pytest -q integrations/adaptiveflow/test_library.py
```

The verifier covers two real CPU batches, duplicate identifiers across collections, duplicate SMILES, invalid SMILES, negative score ordering, and one model initialization per child.
A separate deterministic fixture checks tranche selection across two tranches, whole-collection budget rounding, regex filtering, and primary workunit preparation.
See [the verification record](VERIFICATION.md) for commands and observed results.

## License and artifact rights

PSICHIC+ has a noncommercial research license, and external AdaptiveFlow-VS is GPL-2.0-or-later.
This integration copies no upstream implementation and adds no weights or combined image distribution.
An import or process boundary does not establish license compatibility.
The PSICHIC+ software license does not grant weight or dataset rights unless expressly provided in writing.
Obtain the necessary grants for your use and any combined redistribution.
