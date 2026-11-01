# PSICHIC+ ligand featurization

This package owns ligand normalization, graph construction, and versioned caching for both PSICHIC+ entry points.
It ports `mol_featurizer` from `huankoh/psichic-plus@3c8e5e4` without changing its algorithms.

## Current build status

There are no supported subdirs for the frozen v1 contract.
`setup.py` refuses all new builds because `SUPPORTED_SUBDIRS` is empty.
A local Apple M4 build passes all 2,242 contract rows and the 20,000-row benchmark replay, but Apple Silicon CI differs on 651 rows.
Linux CPU and CUDA environments each differ on 563 rows.
All these differences are confined to `clique_pe`; normalized SMILES and every other graph value agree.
Windows' pinned RDKit package lacks the C++ development headers, and Intel macOS cannot install the pinned PyTorch wheel.
A new graph version requires a separate decision before these platforms can be supported.

## Build and verify

After a platform passes the contract and is restored to the allowlist, create and activate its root `environment_*.yml` environment.
Run these commands from the repository root.

```bash
pip install --no-build-isolation --no-deps ./psichic-ligand
python -m psichic_ligand verify
```

Builds require Python 3.12, conda-forge RDKit 2024.03.3 with C++ headers, Boost headers matching RDKit's build, pybind11 3.1.0, and a C++17 compiler.
The build checks RDKit provenance and Boost compatibility before invoking a compiler.
Every installation recompiles both extensions so a previous environment's object files cannot leak into the new build.
Imports check the loaded RDKit version and path and load installed extensions without compiling.

`verify` compares statuses, error messages, normalized SMILES, and graph digests through the public featurizer.
It exits 1 on any difference and prints per-key hashes for up to 20 graph mismatches.
A missing contract file also fails.

## Public API

```python
from psichic_ligand import LigandFailure, LigandStore, featurize_ligand

result = featurize_ligand("CC(=O)Oc1ccccc1C(=O)O")
if isinstance(result, LigandFailure):
    print(result.stage, result.reason)
else:
    print(result["smiles"], result["atom_num_nodes"], result["clique_pe"].shape)

store = LigandStore("runs/ligands")
graph = store.get("CC(=O)Oc1ccccc1C(=O)O")
```

The public names are `featurize_ligand`, `LigandStore`, `LigandGraph`, `LigandFailure`, `LigandFeaturizationError`, and `FEATURIZATION_VERSION`.
Submodules are internal.
`featurize_ligand` accepts raw strings without stripping and returns a graph dictionary or a failure value.
`LigandStore.get` raises `LigandFeaturizationError` on failure and exposes the failure through `error.failure`.

## Cache and version rules

`FEATURIZATION_VERSION` combines `NORMALIZATION_VERSION` and `GRAPH_VERSION`.
The current value is `rdkit-2024.03.3-enumpick-max100-v1+psichic-graph-v1`.
The store writes `<cache_dir>/<FEATURIZATION_VERSION>/<sha256(raw UTF-8)[:32]>.pt`.
Its `directory` property returns that versioned directory.

The cache contains plain graph dictionaries loaded on CPU with `weights_only=True`.
Writes use a temporary file in the destination directory and an atomic replacement.
Unreadable files are cache misses and are replaced after successful recomputation.
Legacy flat cache files are warned about, ignored, and left unchanged.
Failures are memoized per process and never persisted as graphs.
`LigandStore(None)` keeps nothing on disk.
The store is picklable for spawned DataLoader and interpretation workers.

Any change to normalization or graph bytes requires a version bump and a newly recorded independent fixture.
Per-platform expected outputs and numerical tolerances are not permitted.

## Contract provenance

The fixture header records the exact selection rule and SHA-256 hashes of its inputs.
The 2,242 unique raw inputs combine a seeded stratified panel sample, all 90 normalization policy changes, all 17 benchmark failures, the 100 largest molecules, and hand-picked cases including the private audit's exact 14 SMILES.
Expected outputs come from benchmark records and the untouched Python 3.10 pip-wheel build of `3c8e5e4`, using RDKit 2024.3.3 and PyTorch 2.4.1.
Every row was independently checked with the untouched Python 3.12 conda-forge build, using RDKit 2024.03.3, Boost 1.84.0, and PyTorch 2.6.0.
The generator remains outside this repository.

The graph digest hashes sorted keys and each tensor's dtype, shape, and contiguous CPU bytes.
It uses the `sha256-bench-v2-no-clique-pe` algorithm, which compares every graph key exactly except `clique_pe`. That key is 20 chained float32 matmuls whose bytes depend on the local BLAS, and it is derived from `tree_edge_index` and `clique_num_nodes`, which are compared exactly.

## Notices

See [NOTICE](NOTICE), [LICENSE](LICENSE), the retained Papyrus and ChEMBL MIT licenses under `src/psichic_ligand/normalization/`, and [the RDKit BSD license](licenses/RDKit-BSD-3-Clause.txt).
