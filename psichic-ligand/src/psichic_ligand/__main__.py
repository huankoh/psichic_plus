"""Command-line validation of this installed ligand build."""

import argparse
from collections.abc import Sequence

from ._contract import DIGEST_EXCLUDED_KEYS, graph_digest, verify
from ._ligands import LigandFailure, featurize_ligand


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m psichic_ligand")
    parser.add_argument("command", choices=["verify"])
    parser.parse_args(argv)
    report = verify()
    print(f"psichic_ligand verify: contract {report.contract}")
    print(
        f"platform {report.platform}, python {report.python}, rdkit {report.rdkit} ({report.rdkit_path}), torch {report.torch}"
    )
    if report.ok:
        print(
            f"{report.checked} of {report.checked} molecules match the contract (clique_pe tolerance: 1e-6). This build is supported."
        )
        return 0
    failed = len({m.index for m in report.mismatches})
    print(
        f"FAIL: {failed} of {report.checked} molecules differ from the contract. This build is unsupported; do not use it."
    )
    for mismatch in report.mismatches[:20]:
        expected = "valid random-walk values" if mismatch.field == "clique_pe" else getattr(mismatch.row, mismatch.field)
        print(
            f"row {mismatch.index} {mismatch.field}: raw={mismatch.row.raw!r} expected {expected!r} got {mismatch.actual!r}"
        )
        if mismatch.field == "digest":
            graph = featurize_ligand(mismatch.row.raw)
            if not isinstance(graph, LigandFailure):
                print(
                    f"normalized SMILES identical: {graph['smiles'] == mismatch.row.smiles}"
                )
                for key in sorted(graph.keys() - DIGEST_EXCLUDED_KEYS):
                    print(f"  {key}: {graph_digest({key: graph[key]})}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
