"""Reuse the existing screening helpers without modifying their modules."""

import importlib.util
import math
import os
import sys
from pathlib import Path

from . import Failure, Success


class Predictor:
    def __init__(self, fasta: str, model_dir: str, device: str, cache: str, batch_size: int):
        import pandas as pd
        import torch

        screening = Path(os.environ.get(
            "PSICHIC_SCREENING_DIR", Path(__file__).resolve().parents[1] / "psichic_plus_screening",
        ))
        sys.path.insert(0, str(screening))
        spec = importlib.util.spec_from_file_location("adaptive_screening_inference", screening / "inference.py")
        self.helpers = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.helpers)
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
        self.device = torch.device(device)
        self.cache = Path(cache)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        self.fasta = self.helpers.read_single_fasta(Path(fasta))
        frame = self.helpers.prepare_input_frame(
            pd.DataFrame({"ID": ["target"], "smiles": ["CCO"]}), "smiles", self.fasta, "target",
        )
        self.proteins, self.protein_folder, *_ = self.helpers.build_protein_state(frame, None, False)
        strategies = self.helpers.resolve_inference_preset("fast_screening", "pyg_to_dense", "pyg_dense_mincut", "none")
        self.model = self.helpers.PSICHICPlusModel.from_pretrained(
            model_dir, device=self.device,
            **{key: value for key, value in strategies.items() if key != "inference_preset"},
        )

    def score(self, inputs: list[tuple[str, str]]) -> list[Success | Failure]:
        import pandas as pd

        frame = pd.DataFrame(inputs, columns=["ID", "smiles"])
        frame = self.helpers.prepare_input_frame(frame, "smiles", self.fasta, "target")
        frame = self.helpers.initialize_output_columns(frame)
        loader, *_ = self.helpers.build_loader(
            frame, self.proteins, self.protein_folder, self.cache, self.cache / "ligands",
            self.batch_size, 0, False, False, str(self.device), True, "input_order", False,
        )
        frame, *_ = self.helpers.run_screening(
            frame, self.model, loader, self.proteins, loader.dataset.ligands, self.cache,
            self.device, False, False, "fp32", False, False, "thread", False,
        )
        frame, _ = self.helpers.apply_invalid_ligand_policy(frame, "emit_null")
        results = []
        for row in frame.itertuples(index=False):
            if row.inference_status == "success":
                affinity = float(row.predicted_binding_affinity)
                if math.isfinite(affinity):
                    results.append(Success(row.ID, affinity, -affinity))
                else:
                    results.append(Failure(row.ID, "prediction", "Non-finite affinity"))
            else:
                results.append(Failure(row.ID, str(row.failure_stage), str(row.failure_reason)))
        return results
