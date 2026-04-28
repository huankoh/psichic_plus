# PSICHIC-plus Alpha

**Public alpha / pre-release.**

PSICHIC-plus Alpha is an unpublished extension of the published
[PSICHIC platform](https://github.com/huankoh/PSICHIC). It provides research
code for uncertainty-aware binding-affinity prediction directly from protein
sequence and ligand SMILES inputs.

This repository is intended for non-commercial research and evaluation. APIs,
checkpoint formats, model behavior, and package layout may change before a
stable release.

## Pretrained Weights

Pretrained PSICHIC-plus model weights are not included in this repository.

The default checkpoint directory is:

```text
trained_weights/PSICHIC_plus/
  config.json
  degree.pt
  model.pt
```

Only `config.json` is tracked. The `degree.pt` and `model.pt` checkpoint files
must be supplied separately before inference or few-shot fine-tuning can run.
You can either place compatible local files in the default directory or pass a
different checkpoint directory with `--model_dir`.

## Environment Setup

PSICHIC-plus Alpha has been validated on macOS, Linux, and Windows. Conda or
mamba is recommended.

```bash
# macOS
conda env create -f environment_osx.yml

# Linux or Windows GPU
conda env create -f environment_gpu.yml
conda activate psichic_fp
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv \
  -f https://data.pyg.org/whl/torch-2.1.0+cu118.html

# Linux or Windows CPU
conda env create -f environment_cpu.yml
conda activate psichic_fp
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv \
  -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
```

Alternative Linux setup tested with Python 3.8:

```bash
conda create --name psichic_fp python=3.8
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install pyg -c pyg
conda install -c conda-forge rdkit==2024.03.3
pip install scipy biopython pandas biopandas timeout_decorator py3Dmol umap-learn plotly mplcursors lifelines reprint
pip install "fair-esm"
```

## Input Format

Prepare a CSV file with protein sequences and ligand SMILES strings:

| ID | Protein | Ligand |
|:---|:---|:---|
| P1_LX | MMTGSP... | C1CCCCC1 |
| ROW_2 | TDGRTA... | O=C(C)Oc1ccccc1C(=O)O |
| GLP1_XXF | MAKAVLTGEYKKDELL | CCO |

If the CSV file does not include an `ID` column, row IDs are generated as
`Row_<index>`.

## Inference

Run prediction with the default checkpoint directory:

```bash
python inference.py \
  --input_file YOUR_CSV.csv \
  --output_folder YOUR_OUTPUT_DIR
```

Or provide a separate checkpoint directory:

```bash
python inference.py \
  --input_file YOUR_CSV.csv \
  --output_folder YOUR_OUTPUT_DIR \
  --model_dir /path/to/PSICHIC_plus_checkpoint
```

Predictions are written to:

```text
YOUR_OUTPUT_DIR/inference_results.csv
```

To save interpretation outputs:

```bash
python inference.py \
  --input_file YOUR_CSV.csv \
  --output_folder YOUR_OUTPUT_DIR \
  --save_interpret
```

Interpretation outputs are written under `YOUR_OUTPUT_DIR/interpretations/`:

- Protein residue importance scores are saved as CSV files.
- Ligand atom importance scores are saved as pickle files for RDKit-based
  downstream analysis.
- Protein-ligand interaction fingerprints are saved as NumPy arrays.

## Related Work

PSICHIC-plus Alpha builds on the published PSICHIC platform:

- Code: https://github.com/huankoh/PSICHIC
- Demo for the previous PSICHIC model: https://www.psichicwebserver.com

## License

This public alpha / pre-release is available for non-commercial research and
evaluation under the terms in `LICENSE`.
