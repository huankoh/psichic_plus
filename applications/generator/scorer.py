import math
from pathlib import Path

from applications.scorer import Failure, Success


class PSICHICScorer:
    """Return one raw affinity; SyntheMol owns score composition and caching."""

    def __init__(self, target_fasta: Path, model_dir: Path, device: str, cache_dir: Path):
        self.target_fasta = target_fasta
        self.model_dir = model_dir
        self.device = device
        self.cache_dir = cache_dir
        self.predictor = None

    def __call__(self, smiles: str) -> float:
        try:
            if self.predictor is None:
                from applications.scorer.predictor import Predictor

                self.predictor = Predictor(
                    str(self.target_fasta), str(self.model_dir), self.device,
                    str(self.cache_dir), batch_size=1,
                )
            results = self.predictor.score([(smiles, smiles)])
        except Exception as error:
            raise RuntimeError(f"SMILES {smiles!r}, stage prediction, reason {error}") from error
        if len(results) != 1 or results[0].token != smiles:
            raise RuntimeError(f"SMILES {smiles!r}, stage prediction, reason expected one matching result")
        result = results[0]
        if isinstance(result, Failure):
            raise RuntimeError(f"SMILES {smiles!r}, stage {result.stage}, reason {result.reason}")
        if not isinstance(result, Success) or not math.isfinite(result.raw_affinity):
            raise RuntimeError(f"SMILES {smiles!r}, stage prediction, reason non-finite or invalid affinity")
        return result.raw_affinity
