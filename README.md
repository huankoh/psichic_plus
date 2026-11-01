# psichic_plus

**Public alpha / pre-release.**

`psichic_plus` extends the published [PSICHIC platform](https://github.com/huankoh/PSICHIC) with uncertainty-aware binding-affinity prediction from protein sequences and ligand SMILES.
The repository is intended for non-commercial research and evaluation.
APIs, checkpoint formats, model behavior, and package layout may change before a stable release.

## Ligand build status

The native ligand package supports Linux x86-64 and Apple Silicon macOS with the pinned environments below.
CI verifies the ligand contract on both platforms, including Linux CPU and CUDA environments.
The graph contract checks `clique_pe` with a tolerance of 0.000001 for floating-point differences.
It checks the other graph values exactly.
Windows and Intel macOS are not supported by the pinned build dependencies.
See [the ligand package](psichic-ligand/README.md) for the contract, cache rules, and build requirements.

## Environment and installation

Use these commands on Apple Silicon macOS.
For Linux, create the corresponding CPU or GPU environment instead.

```bash
conda env create -f environment_osx.yml
conda activate psichic_fp
pip install --no-build-isolation --no-deps ./psichic-ligand
python -m psichic_ligand verify
```

Use `environment_cpu.yml` for Linux CPU environments, or `environment_gpu.yml` for CUDA 12.4 environments.
All three files pin Python 3.12, numpy 1.26.4, RDKit 2024.03.3, Boost 1.84.0, PyTorch 2.6.0, and PyG 2.7.0.
The environment files include the PyG extension wheels and native build dependencies.
On macOS, install Xcode Command Line Tools with `xcode-select --install`.
Windows compilation also requires Visual Studio 2022 Build Tools with "Desktop development with C++" and a matching RDKit development package.

On Linux, run this after activation so RDKit and PyTorch use the conda C++ runtime.

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

The two native extensions compile during package installation.
Imports never invoke a compiler.
Reinstall the package after changing native source or recreating the environment, then run `verify`.

## Input and inference

Prepare a CSV with `Protein` and `Ligand` columns.
`Protein` can contain a sequence, or a protein key paired with a `Protein_Sequence` column.
An optional `ID` column identifies rows; missing IDs become `Row_<index>`.
The small example in [tests/data/e2e.csv](tests/data/e2e.csv) includes both protein keys and sequences.

With a verified installed build and the checkpoint in `trained_weights/PSICHIC_plus`, run CPU inference with interpretations.

```bash
python inference.py \
  --input_file tests/data/e2e.csv \
  --output_folder runs/e2e \
  --device cpu \
  --save_interpret
```

Predictions are written to `runs/e2e/inference_results.csv`.
Each pair gets protein scores, an RDKit ligand pickle with atom scores, and a NumPy fingerprint under `runs/e2e/interpretations/`.
The ESM protein model downloads on first use.
The root CLI removes its temporary ligand cache after interpretation.
`fewshot.py` keeps its versioned ligand cache under `<datafolder>/ligands`.

## Screening and ligand caches

The [screening runtime](psichic_plus_screening/README.md) provides batching strategies, persistent caches, and a Docker build from the repository root.
Both entry points use the installed `psichic_ligand` package.

`FEATURIZATION_VERSION` is `rdkit-2024.03.3-enumpick-max100-v1+psichic-graph-v1`.
Compared with the initial public alpha, 0.09% of benchmark ligands normalize differently because cleanup repeats to stability and tautomer search is capped at 100 transforms.
These changes can change predictions for the affected ligands.

Graphs are stored at `<cache_dir>/<FEATURIZATION_VERSION>/<sha256(raw UTF-8)[:32]>.pt`.
Legacy unversioned files are ignored and left untouched.
A corrupt graph is recomputed, and writes publish atomically.
Interpretation reads through the same store as featurization.

## Acknowledgements and license

The code builds on [PSICHIC](https://github.com/huankoh/PSICHIC).
The previous PSICHIC model has a [web demo](https://www.psichicwebserver.com).
Vendored Papyrus 0.0.5 and ChEMBL Structure Pipeline 1.2.4 code, their MIT licenses, and the adapted RDKit BSD notice are retained in [psichic-ligand/NOTICE](psichic-ligand/NOTICE) and its referenced files.
This public alpha is available for non-commercial research and evaluation under [LICENSE](LICENSE).

For adaptive collection screening, see the optional [AdaptiveFlow adapter](psichic_plus_adaptive/README.md).
