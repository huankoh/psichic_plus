import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem

from psichic_plus_adaptive.predictor import Predictor
from psichic_plus_synthemol import UPSTREAM_REVISION
from psichic_plus_synthemol.scorer import PSICHICScorer


def main() -> None:
    parser = argparse.ArgumentParser(description="Check real generation, route provenance, and direct PSICHIC+ affinity.")
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    example = Path(__file__).resolve().parent
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    frames = []
    commands = []
    for space, reaction in [("real", "22"), ("wuxi", "1")]:
        command = [sys.executable, "-m", "psichic_plus_synthemol",
                   "--target-fasta", str(example / "target.fasta"),
                   "--model-dir", str(args.model_dir.resolve()),
                   "--building-blocks", str(example / "building_blocks.csv"),
                   "--reaction-map", str(example / f"{space}.pkl"),
                   "--chemical-space", space, "--output", str(output / space),
                   "--rollouts", "2", "--max-reactions", "1", "--seed", "0", "--device", args.device]
        commands.append(command)
        with (output / f"{space}.log").open("w") as handle:
            subprocess.run(command, check=True, stdout=handle, stderr=subprocess.STDOUT)
        frame = pd.read_csv(output / space / "molecules.csv", dtype=str, keep_default_na=False)
        assert len(frame) > 0
        assert frame.num_reactions.tolist() == ["1"] * len(frame)
        assert frame.reaction_1_id.tolist() == [reaction] * len(frame)
        assert frame.reaction_1_chemical_space.tolist() == [space] * len(frame)
        for row in frame.itertuples(index=False):
            assert Chem.MolFromSmiles(row.smiles) is not None
            assert {row.building_block_1_1_id, row.building_block_1_2_id} == {"0001", "0002"}
            assert {row.building_block_1_1_smiles, row.building_block_1_2_smiles} == {
                "CNc1ccccc1", "O=C(O)c1ccccc1",
            }
        frames.append(frame)
    products = pd.concat(frames, ignore_index=True)
    unique = products[["smiles"]].drop_duplicates().reset_index(drop=True)
    unique.insert(0, "ID", [f"product-{index}" for index in range(len(unique))])
    unique.to_csv(output / "direct-input.csv", index=False)
    screening = Path(os.environ.get("PSICHIC_SCREENING_DIR", example.parents[1] / "psichic_plus_screening"))
    command = [sys.executable, str(screening / "inference.py"),
               "--input_file", str(output / "direct-input.csv"),
               "--protein_fasta", str(example / "target.fasta"),
               "--output_folder", str(output / "direct"), "--model_dir", str(args.model_dir.resolve()),
               "--ligand_column", "smiles", "--device", args.device,
               "--batch_size", "1", "--num_workers", "0", "--no-pin-memory"]
    commands.append(command)
    with (output / "direct.log").open("w") as handle:
        subprocess.run(command, check=True, stdout=handle, stderr=subprocess.STDOUT)
    direct = pd.read_csv(output / "direct/inference_results.csv").set_index("smiles")
    assert direct.inference_status.tolist() == ["success"] * len(unique)
    maximum_delta = 0.0
    for row in products.itertuples(index=False):
        delta = abs(float(row.psichic_affinity) - direct.loc[row.smiles].predicted_binding_affinity)
        assert delta <= 1e-5, (row.smiles, delta)
        assert float(row.score) == float(row.psichic_affinity)
        maximum_delta = max(maximum_delta, delta)
    scorer = PSICHICScorer(Predictor(str(example / "target.fasta"), str(args.model_dir.resolve()),
                                    args.device, str(output / "rejection-cache"), batch_size=1))
    try:
        scorer("C")
    except RuntimeError as error:
        rejection = str(error)
        assert "SMILES 'C'" in rejection and "stage" in rejection and "reason" in rejection
        assert scorer.scores == {}
    else:
        raise AssertionError("The undersized ligand must fail visibly")
    (output / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
    summary = {"synthemol_revision": UPSTREAM_REVISION, "device": args.device,
               "products_by_space": {space: len(frame) for space, frame in zip(("real", "wuxi"), frames)},
               "compared_predictions": len(products), "max_absolute_affinity_delta": maximum_delta,
               "provenance_preserved": True, "visible_rejection": rejection}
    (output / "verification.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
