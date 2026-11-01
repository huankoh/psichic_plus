"""One target and model per AdaptiveFlow docking child."""

import hashlib
import sys
from pathlib import Path


class Predictor:
    def __init__(self, config, output_dir):
        sys.path.insert(
            0, str(Path(__file__).resolve().parents[2] / "psichic_plus_screening")
        )
        import inference
        import torch

        self.inference = inference
        torch.set_num_threads(config["threads"])
        self.device = torch.device(config["device"])
        _, self.sequence = inference.read_single_fasta(Path(config["target_fasta"]))
        self.protein_key = (
            "target_" + hashlib.sha256(self.sequence.encode()).hexdigest()
        )
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        seed = self.frame([{"ligand_id": "seed", "smiles": "C"}])
        self.proteins, self.protein_folder, *_ = inference.build_protein_state(
            seed, None, False
        )
        self.model = inference.PSICHICPlusModel.from_pretrained(
            config["model_dir"], device=self.device
        )
        self.batch_number = 0

    def frame(self, records):
        import pandas as pd

        return pd.DataFrame(
            {
                "ID": [record["ligand_id"] for record in records],
                "Ligand": [record["smiles"] for record in records],
                "Protein": self.protein_key,
                "Protein_Sequence": self.sequence,
            }
        )

    def predict(self, records):
        api = self.inference
        self.batch_number += 1
        frame = api.initialize_output_columns(self.frame(records))
        loader, *_ = api.build_loader(
            frame,
            self.proteins,
            self.protein_folder,
            self.output_dir,
            self.output_dir / "ligands",
            len(frame),
            0,
            False,
            False,
            str(self.device),
            False,
            "input_order",
            False,
        )
        result, *_ = api.run_screening(
            frame,
            self.model,
            loader,
            self.proteins,
            loader.dataset.ligands,
            self.output_dir,
            self.device,
            False,
            True,
            "fp32",
            False,
            False,
            "sync",
            False,
        )
        result, _ = api.apply_invalid_ligand_policy(result, "emit_null")
        return result.to_dict("records")
