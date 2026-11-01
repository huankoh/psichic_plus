"""The lazy external_batch entry point called by the upstream worker."""

import json
import time
from dataclasses import asdict
from pathlib import Path

from . import Failure, Success

_predictor = None


def score_items(items: list[dict]) -> None:
    global _predictor

    started = time.perf_counter()
    results = {}
    inputs = []
    for item in items:
        token = item['ligand_key']
        try:
            smiles, archived_token = Path(item['ligand_path']).read_text().rstrip("\n").split("\t")
            if archived_token != token:
                raise ValueError("Archive token does not match the upstream ligand key")
            inputs.append((token, smiles))
        except (OSError, ValueError) as error:
            results[token] = Failure(token, "input", str(error))

    if inputs:
        stage = "predictor_init"
        try:
            if _predictor is None:
                from .predictor import Predictor

                config = json.loads(Path(items[0]['config_path']).read_text())
                _predictor = Predictor(**config)
            stage = "prediction"
            predictions = _predictor.score(inputs)
            expected = {token for token, _ in inputs}
            if len(predictions) != len(inputs) or {result.token for result in predictions} != expected:
                raise ValueError("Predictor did not return exactly one result per input")
            results.update((result.token, result) for result in predictions)
        except Exception as error:
            results.update((token, Failure(token, stage, f"{type(error).__name__}: {error}")) for token, _ in inputs)

    for item in items:
        result = results[item['ligand_key']]
        item['seconds'] = time.perf_counter() - started
        if isinstance(result, Success):
            item['status'] = "success"
            item['score'] = result.adaptive_score
            item['attrs']['raw_affinity'] = result.raw_affinity
        else:
            item['status'] = "failed"
            item['log']['stage'] = result.stage
            item['log']['reason'] = result.reason
        Path(item['log_path']).write_text(json.dumps(asdict(result), allow_nan=False) + "\n")
