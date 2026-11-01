# Adapter formats and limits

## Library identity

The input CSV requires `collection_id`, `molecule_id`, `tranche`, and `smiles`.
Additional columns are ignored.
Identifiers are strings, including leading zeros and strings such as `NA`.
A collection belongs to exactly one tranche.
The pair `collection_id,molecule_id` identifies one molecule.
Repeated molecule IDs across different collections are allowed.
Duplicate pairs, missing identifiers, malformed rows, and collections spanning tranches are errors.
SMILES may be invalid or empty, but may not contain tabs or line breaks.

`Molecule(token, collection_id, molecule_id, tranche, smiles)` is the library record.
The token is `m` followed by the SHA-256 digest of a JSON-encoded identifier pair.
The converter checks token uniqueness within the library.
Separate hashes encode tranche and collection names for upstream paths.
`library/manifest.csv` joins these safe names to the exact original strings and SMILES.

## Upstream collections

The converter creates `library/full/<safe-tranche>/<safe-collection>.tar.gz`.
Each archive contains `<safe-collection>/<token>.smi` files and a `.listing`.
A molecule file contains the original SMILES, a tab, the token, and a newline.
The full listing contains `upstream_collection_key,token,index` records.

The sparse archives are `library/sparse/<safe-tranche>/0000001.tar.gz`.
Their `.listing` records name the full upstream collection key, token, and zero-based tranche sample index.
The existing upstream `unpack_item` reads this listing in prescreen mode.
The converter also writes full and prescreen todo files, collection and tranche Parquet metadata, and `prescreen.tokens.json`.
No prepared three-dimensional structures or AdaptiveFlow-LP run are required.

## Sampling and budgets

The prescreen takes the first `--samples-per-tranche` rows of each tranche in input order, or all rows if the tranche is smaller.
Input order therefore determines the sample.
Shuffle or stratify the CSV before conversion when the row order is not representative.
The converter holds the library in memory.

A successful prediction is `Success(token, raw_affinity, adaptive_score)`.
`adaptive_score` is the negative of the raw predicted affinity because upstream considers lower scores better.
A failure is `Failure(token, stage, reason)` and has no score.
Only finite, strictly negative adaptive scores from successes enter selection.

The launcher passes successful prescreen scores to upstream `create_todofile_atg-primaryscreen.py` with `tranche_ave_score`.
This mode averages successful scores by tranche and selects collections in ascending tranche-score order.
Collections in tranches without any successful prescreen score are excluded before the selector runs.
An all-failed prescreen or a target larger than the eligible molecule count is an error.

`--primary-target` is a target number of molecules attempted in the primary stage.
Upstream selects whole collections, so the count can exceed that target by the size of the last selected collection minus one.
The primary stage re-scores its complete selected collections, including prescreen molecules.
Failures count toward attempted molecules and do not trigger replacement sampling.
The overall scoring effort includes the prescreen and these repeated primary-stage predictions.

The launcher preserves the generated `primary.todo`, records its SHA-256 digest in `selection.json`, and copies its bytes unchanged into primary-stage preparation.
One scoring process runs at a time.
The model and target features are loaded once per upstream scoring process and reused across batches.
Each work-unit subjob starts a new worker, so reuse does not extend across subjobs or stages.

## Outputs and completion

| Path | Contents |
| --- | --- |
| `library/manifest.csv` | Original string IDs, SMILES, tokens, and upstream collection keys. |
| `prescreen/results.csv` | Every expected prescreen molecule with raw affinity or failure. |
| `primaryscreen/results.csv` | Every expected primary molecule with raw affinity or failure. |
| `<stage>/failures.csv` | Stage-specific failures with their original IDs. |
| `ranked.csv` | Primary-stage successes sorted by descending raw affinity. |
| `selection.json` | The requested and actual primary counts, selected collections, excluded collection count, and todo digest. |
| `run.json` | Completed stage counts, upstream revision, device, and scoring-process count. |
| `<stage>/workflow/workunits/*/*.out` | Actual upstream worker logs. |
| `<stage>/output-files` | Upstream overview records, success Parquet files, and archived item logs. |
| `<stage>/incomplete.json` | Completion mismatches, present only on an incomplete run. |
| `commands.log` and nested `commands.log` files | Exact subprocess argument lists and their output. |

The result columns are the molecule fields plus `screen_stage`, `status`, `raw_affinity`, `adaptive_score`, `failure_stage`, and `failure_reason`.
A failed prediction has empty affinity fields.
Reconciliation uses upstream success summaries and the completion queue's failure records.
Missing, duplicate, unexpected, mis-grouped, or inconsistent-score completions fail the run.
Missing completions appear as explicit `completion` failures in the stage exports.
A run writes its final `run.json` only after both stages reconcile.
A fully completed primary stage can still have no successful predictions, in which case `ranked.csv` contains only its header.

## Supported execution

The adapter supports the pinned upstream local Bash backend on a shared filesystem.
It creates a new run directory and refuses to overwrite an existing one.
The upstream Bash templates require output paths without spaces or shell metacharacters.
Cluster, AWS, multiple scoring processes, resumable runs, and distributed scheduling are outside this adapter's supported path.
Existing inference modules and shared Docker files are unchanged.
