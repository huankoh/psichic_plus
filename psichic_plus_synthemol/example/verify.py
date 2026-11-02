import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem

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
    preparation = [sys.executable, "-m", "psichic_plus_synthemol", "score-building-blocks",
                   "--input", str(example / "building_blocks.csv"),
                   "--output", str(output / "scored-blocks.csv"),
                   "--target-fasta", str(example / "target.fasta"),
                   "--model-dir", str(args.model_dir.resolve()), "--device", args.device,
                   "--cache-dir", str(output / "preparation-cache")]
    commands.append(preparation)
    with (output / "preparation.log").open("w") as handle:
        subprocess.run(preparation, check=True, stdout=handle, stderr=subprocess.STDOUT)
    blocks = pd.read_csv(output / "scored-blocks.csv", dtype=str)
    assert blocks.reagent_id.tolist() == ["0001", "0002"]
    for label, spaces in [("real", ["real"]), ("wuxi", ["wuxi"]), ("combined", ["real", "wuxi"])]:
        command = [sys.executable, "-m", "psichic_plus_synthemol",
                   "--search_type", "mcts", "--save_dir", str(output / label),
                   "--score_types", "psichic_plus", "--score_model_paths", str(args.model_dir.resolve()),
                   "--score_names", "psichic_affinity",
                   "--psichic_target_fasta", str(example / "target.fasta"),
                   "--psichic_device", args.device,
                   "--building_blocks_paths", *[str(output / "scored-blocks.csv")] * len(spaces),
                   "--reaction_to_building_blocks_paths", *[str(example / f"{space}.pkl") for space in spaces],
                   "--chemical_spaces", *spaces, "--n_rollout", "2", "--rng_seed", "0",
                   "--save_frequency", "1", "--store_nodes"]
        commands.append(command)
        with (output / f"{label}.log").open("w") as handle:
            subprocess.run(command, check=True, stdout=handle, stderr=subprocess.STDOUT,
                           env=dict(os.environ, PYTHONHASHSEED="0"))
        frame = pd.read_csv(output / label / "molecules.csv", dtype=str, keep_default_na=False)
        assert len(frame) > 0
        assert frame.num_reactions.tolist() == ["1"] * len(frame)
        assert set(frame.reaction_1_chemical_space) <= set(spaces)
        for row in frame.itertuples(index=False):
            assert row.reaction_1_id == {"real": "22", "wuxi": "1"}[row.reaction_1_chemical_space]
            assert Chem.MolFromSmiles(row.smiles) is not None
            assert {row.building_block_1_1_id, row.building_block_1_2_id} == {"0001", "0002"}
            assert {row.building_block_1_1_smiles, row.building_block_1_2_smiles} == {
                "CNc1ccccc1", "O=C(O)c1ccccc1",
            }
        assert (output / label / "logs.pkl").stat().st_size > 0
        stats = pd.read_csv(output / label / "generation_stats.csv")
        assert stats.num_nodes_searched.iloc[0] > 0
        frames.append(frame)
    products = pd.concat(frames, ignore_index=True)
    unique = pd.concat([products[["smiles"]], blocks[["smiles"]]]).drop_duplicates().reset_index(drop=True)
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
    for row in blocks.itertuples(index=False):
        delta = abs(float(row.score) - direct.loc[row.smiles].predicted_binding_affinity)
        assert delta <= 1e-5, (row.smiles, delta)
        maximum_delta = max(maximum_delta, delta)
    scorer = PSICHICScorer(example / "target.fasta", args.model_dir.resolve(),
                           args.device, output / "rejection-cache")
    try:
        scorer("C")
    except RuntimeError as error:
        rejection = str(error)
        assert "SMILES 'C'" in rejection and "stage" in rejection and "reason" in rejection
    else:
        raise AssertionError("The undersized ligand must fail visibly")
    (output / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
    summary = {"synthemol_revision": UPSTREAM_REVISION, "device": args.device,
               "products_by_space": {space: len(frame) for space, frame in zip(("real", "wuxi", "combined"), frames)},
               "compared_predictions": len(products) + len(blocks), "max_absolute_affinity_delta": maximum_delta,
               "provenance_preserved": True, "visible_rejection": rejection}
    (output / "verification.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
