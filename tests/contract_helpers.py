import hashlib
import json
from importlib.resources import files

import torch

from psichic_ligand import FEATURIZATION_VERSION


def contract_rows():
    path = files("psichic_ligand").joinpath(
        "contract", FEATURIZATION_VERSION + ".jsonl"
    )
    return [json.loads(line) for line in path.read_text().splitlines()[1:]]


def digest(graph):
    h = hashlib.sha256()
    for key in sorted(graph):
        if key == "clique_pe":  # BLAS-dependent bytes, see psichic_ligand._contract
            continue
        h.update(f"k:{key};".encode())
        value = graph[key]
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu().contiguous()
            h.update(f"t:{tensor.dtype}:{tuple(tensor.shape)};".encode())
            h.update(tensor.numpy().tobytes())
        else:
            h.update(f"v:{type(value).__name__}:{value!r};".encode())
    return h.hexdigest()
