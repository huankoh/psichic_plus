import math

from synthemol.generate.score_weights import ScoreWeights

from psichic_plus_adaptive import Failure, Success


class PSICHICScorer:
    num_scores = 1

    def __init__(self, predictor):
        self.predictor = predictor
        self.score_weights = ScoreWeights(
            [1.0], immutable=True, score_names=["psichic_affinity"], score_signs=[1],
        )
        self.scores: dict[str, float] = {}

    def compute_individual_scores(self, smiles: str) -> list[float]:
        if smiles not in self.scores:
            try:
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
            self.scores[smiles] = result.raw_affinity
        return [self.scores[smiles]]

    def __call__(self, smiles: str) -> float:
        return self.compute_individual_scores(smiles)[0]
