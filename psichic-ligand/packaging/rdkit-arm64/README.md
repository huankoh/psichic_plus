# RDKit 2024.03.3 for Linux ARM

This recipe builds the exact RDKit release required by the existing ligand contract.
Conda-forge did not publish that release for `linux-aarch64`.
The recipe comes from [rdkit-feedstock at 5a836e8](https://github.com/conda-forge/rdkit-feedstock/tree/5a836e8b064e226f4f8b76b2ff3f3242f00fe0fe/recipe).
It enables ARM, supplies NumPy's include directory explicitly, removes a patch already present in the released source, and uses CMake 3 with a pinned compiler and sysroot configuration.
The source archive checksum remains `52f79c6bf1d446cdb5c86a35de655d96bad0c52a5f4ecbe15f08eaf334e6f76a`.
No RDKit chemistry code changes.

Run from the repository root on Linux ARM with conda or micromamba available.
Create the model environment from `environment_cpu_arm64.yml` first.
This environment deliberately omits RDKit until the local package is built.
It explicitly pins Boost.Python 1.84 because installing a local Conda archive by filename does not resolve the archive's runtime dependencies.
The PyTorch extensions are compiled after PyTorch is installed.

```bash
conda env create -f environment_cpu_arm64.yml
conda create -n rdkit-builder --override-channels -c conda-forge -y conda-build conda-libmamba-solver
arm_build_root=$(mktemp -d /tmp/rdkit-arm64.XXXXXX)
CPU_COUNT=12 conda run -n rdkit-builder conda build \
  --override-channels -c conda-forge --no-anaconda-upload --package-format 1 \
  --output-folder "$arm_build_root/packages" --croot "$arm_build_root/build" \
  psichic-ligand/packaging/rdkit-arm64
conda activate psichic_fp
conda install --override-channels -c conda-forge --freeze-installed -y \
  "$arm_build_root"/packages/linux-aarch64/rdkit-2024.03.3-py312*.tar.bz2
MAX_JOBS=8 pip install --no-build-isolation --no-deps -r requirements_cpu_arm64.txt
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
pip install --no-build-isolation --no-deps ./psichic-ligand
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -I -m psichic_ligand verify
```

Keep the original package archive at its installed file URL.
The ligand build checks its SHA-256 against the actual package-manager record and compares the embedded recipe files with this directory.
An arbitrary local RDKit package, a changed archive, or a changed recipe is refused.
Existing conda-forge installations keep their existing provenance checks.
Python 3.12, RDKit 2024.03.3, matching Boost headers, and the compiler checks still apply.

The verifier must match the existing 2,242 recorded inputs before this build is used for inference.
Normalized SMILES and the recorded graph values match exactly.
The existing `clique_pe` check retains its absolute tolerance of `1e-6` and zero relative tolerance.
The contract panel does not prove arbitrary molecular inputs or biological accuracy.

The [Spark qualification summary](proof.json) records the source/package hashes, exact model-library versions, unchanged contract result and archive-guard fault probe.

The copied recipe retains the conda-forge contributors' [BSD-3-Clause notice](LICENSE).
