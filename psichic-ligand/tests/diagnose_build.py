"""Write per-key hashes for CI rows that fail the aggregate graph contract."""

import json
from dataclasses import asdict
from pathlib import Path
import sys

from psichic_ligand import FEATURIZATION_VERSION, LigandFailure, featurize_ligand
from psichic_ligand._contract import graph_digest, load_contract


def main():
    _, rows = load_contract(FEATURIZATION_VERSION)
    differences = []
    for index, row in enumerate(rows):
        result = featurize_ligand(row.raw)
        if isinstance(result, LigandFailure):
            if (result.stage, result.status, result.message) != (
                row.stage,
                row.status,
                row.message,
            ):
                differences.append(
                    dict(index=index, expected=asdict(row), failure=asdict(result))
                )
        elif result["smiles"] != row.smiles or graph_digest(result) != row.digest:
            differences.append(
                dict(
                    index=index,
                    expected=asdict(row),
                    smiles=result["smiles"],
                    digest=graph_digest(result),
                    keys={
                        key: graph_digest({key: value}) for key, value in result.items()
                    },
                )
            )
    Path(sys.argv[1]).write_text(json.dumps(differences, indent=2) + "\n")
    print(f"Wrote per-key diagnostics for {len(differences)} differing rows.")


if __name__ == "__main__":
    main()
