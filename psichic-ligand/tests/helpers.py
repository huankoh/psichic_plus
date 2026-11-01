import hashlib


def graph_digest(graph):
    h = hashlib.sha256()
    for key in sorted(graph):
        if key == "clique_pe":  # BLAS-dependent bytes, see psichic_ligand._contract
            continue
        h.update(f"k:{key};".encode())
        value = graph[key]
        if hasattr(value, "detach"):
            value = value.detach().cpu().contiguous()
            h.update(f"t:{value.dtype}:{tuple(value.shape)};".encode())
            h.update(value.numpy().tobytes())
        else:
            h.update(f"v:{type(value).__name__}:{value!r};".encode())
    return h.hexdigest()
