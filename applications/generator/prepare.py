import argparse
import csv
from pathlib import Path

from .scorer import PSICHICScorer


def score_building_blocks(
    input_path: Path, output_path: Path, target_fasta: Path, model_dir: Path,
    cache_dir: Path, device: str = "cpu", smiles_column: str = "smiles",
    score_column: str = "score", overwrite_output: bool = False,
    overwrite_score_column: bool = False,
) -> None:
    if output_path.exists() and not overwrite_output:
        raise FileExistsError(f"Output already exists: {output_path}. Use --overwrite-output to replace it.")
    with input_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames
        if not columns or len(set(columns)) != len(columns) or smiles_column not in columns:
            raise ValueError(f"CSV needs unique column names and a {smiles_column!r} column")
        if score_column == smiles_column:
            raise ValueError("The score column must differ from the SMILES column")
        if score_column in columns and not overwrite_score_column:
            raise ValueError(f"Score column {score_column!r} already exists. Use --overwrite-score-column to replace it.")
        rows = list(reader)
    if not rows:
        raise ValueError("Building-block CSV is empty")
    scorer = PSICHICScorer(target_fasta, model_dir, device, cache_dir)
    for row in rows:
        if None in row or any(value is None for value in row.values()) or not row[smiles_column].strip():
            raise ValueError("CSV contains an incomplete row or empty SMILES")
        row[score_column] = str(scorer(row[smiles_column]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w" if overwrite_output else "x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns if score_column in columns else columns + [score_column])
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Add raw PSICHIC+ affinities to a building-block CSV.")
    parser.add_argument("--input", type=Path, required=True, dest="input_path")
    parser.add_argument("--output", type=Path, required=True, dest="output_path")
    parser.add_argument("--target-fasta", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--smiles-column", default="smiles")
    parser.add_argument("--score-column", default="score")
    parser.add_argument("--overwrite-output", action="store_true")
    parser.add_argument("--overwrite-score-column", action="store_true")
    score_building_blocks(**vars(parser.parse_args(argv)))


if __name__ == "__main__":
    main()
