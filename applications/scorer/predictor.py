import importlib.util
import gc
import math
import os
import sys
from pathlib import Path

from . import Failure, Success


def _protein_state(fasta, device):
    import featurizers
    import torch

    _, sequence = fasta
    model, alphabet = featurizers.esm.pretrained.load_model_and_alphabet("esm2_t33_650M_UR50D")
    model.eval()
    model.to(device)
    sequence = featurizers.replace_non_standard_residues(sequence, featurizers.pro_res_table)
    sequence_features = featurizers.seq_feature(sequence)
    tokens, contacts, _ = featurizers.esm_extract(
        model, alphabet.get_batch_converter(), sequence, layer=33, approach="last", dim=1280,
    )
    edge_index, edge_weight = featurizers.contact_map(contacts)
    protein = {
        "seq": sequence,
        "seq_feat": torch.from_numpy(sequence_features),
        "token_representation": tokens.half(),
        "num_nodes": len(sequence),
        "num_pos": torch.arange(len(sequence)).reshape(-1, 1),
        "edge_index": edge_index,
        "edge_weight": edge_weight,
    }
    del model
    if device.type == "cuda":
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
    gc.collect()
    return {"target": protein}


class Predictor:
    def __init__(self, fasta: str, model_dir: str, device: str, cache: str, batch_size: int):
        import torch

        screening = Path(os.environ.get(
            "PSICHIC_SCREENING_DIR", Path(__file__).resolve().parents[2] / "psichic_plus_screening",
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
        self.proteins = _protein_state(self.fasta, self.device)
        self.protein_folder = None
        strategies = self.helpers.resolve_inference_preset("fast_screening", "pyg_to_dense", "pyg_dense_mincut", "none")
        self.model = self.helpers.PSICHICPlusModel.from_pretrained(
            model_dir, device=self.device,
            **{key: value for key, value in strategies.items() if key != "inference_preset"},
        )

    def score(self, inputs: list[tuple[str, str]]) -> list[Success | Failure]:
        import pandas as pd

        tokens = {str(index): token for index, (token, _) in enumerate(inputs)}
        frame = pd.DataFrame({"ID": list(tokens), "smiles": [smiles for _, smiles in inputs]})
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
            token = tokens[row.ID]
            if row.inference_status == "success":
                affinity = float(row.predicted_binding_affinity)
                if math.isfinite(affinity):
                    results.append(Success(token, affinity, -affinity))
                else:
                    results.append(Failure(token, "prediction", "Non-finite affinity"))
            else:
                results.append(Failure(token, str(row.failure_stage), str(row.failure_reason)))
        return results
