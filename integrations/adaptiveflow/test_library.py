import csv
import json
import tarfile

import pandas as pd
import pytest

from library import FIELDS, convert_library, write_csv
from run import select_primary


def test_conversion_keeps_row_identity_and_sparse_original_collections(tmp_path):
    source = tmp_path / "input.csv"
    rows = [
        {
            "molecule_id": "duplicate,\nID",
            "collection_id": "0000001",
            "tranche": "AA",
            "smiles": " CCO ",
        },
        {
            "molecule_id": "duplicate,\nID",
            "collection_id": "0000042",
            "tranche": "AA",
            "smiles": " CCO ",
        },
        {
            "molecule_id": "invalid",
            "collection_id": "0000001",
            "tranche": "BB",
            "smiles": "invalid",
        },
        {
            "molecule_id": "empty",
            "collection_id": "0000001",
            "tranche": "AA",
            "smiles": "",
        },
    ]
    write_csv(source, FIELDS[1:], rows)
    destination = tmp_path / "converted"
    records = convert_library(source, destination, 2)
    with (destination / "manifest.csv").open(newline="") as handle:
        manifest = list(csv.DictReader(handle))
    assert records == manifest
    assert len({row["ligand_id"] for row in manifest}) == 4
    assert [{field: row[field] for field in FIELDS[1:]} for row in manifest] == rows
    inventory = pd.read_parquet(destination / "collections.parquet")
    assert inventory["Collection"].tolist() == ["0000001", "0000042", "0000001"]
    assert inventory["LigandCount"].tolist() == [2, 1, 1]
    assert pd.read_parquet(destination / "tranches.parquet")[
        "LigandCount"
    ].tolist() == [3, 1]
    with tarfile.open(destination / "collections/SPARSE0/0000000.tar.gz") as archive:
        listing = archive.extractfile("0000000/.listing").read().decode().splitlines()
        assert listing == [
            f"AA_0000001,{records[0]['ligand_id']},0",
            f"AA_0000042,{records[1]['ligand_id']},1",
        ]
        assert len(archive.getmembers()) == 3
    for record in records:
        path = (
            destination
            / "collections"
            / record["tranche"]
            / f"{record['collection_id']}.tar.gz"
        )
        with tarfile.open(path) as archive:
            assert (
                json.load(
                    archive.extractfile(
                        f"{record['collection_id']}/{record['ligand_id']}.json"
                    )
                )
                == record
            )


@pytest.mark.parametrize(
    "field,value",
    [
        ("tranche", "AA_BB"),
        ("tranche", "01"),
        ("tranche", "../AA"),
        ("collection_id", "../1"),
        ("collection_id", "1.0"),
    ],
)
def test_rejects_keys_upstream_cannot_round_trip(tmp_path, field, value):
    row = {
        "molecule_id": "x",
        "collection_id": "0000001",
        "tranche": "AA",
        "smiles": "invalid",
    }
    row[field] = value
    source = tmp_path / "input.csv"
    write_csv(source, FIELDS[1:], [row])
    with pytest.raises(ValueError):
        convert_library(source, tmp_path / "output", 1)
    assert not (tmp_path / "output").exists()


def test_rejects_impossible_budget_before_external_selector(tmp_path):
    library = tmp_path / "library"
    library.mkdir()
    pd.DataFrame(
        [
            {"Tranche": "AA", "Collection": "0000001", "LigandCount": 2},
            {"Tranche": "BB", "Collection": "0000001", "LigandCount": 4},
        ]
    ).to_parquet(library / "collections.parquet")
    with pytest.raises(ValueError, match="between 1 and 2"):
        select_primary(
            tmp_path / "no-upstream", library, [], 3, "AA", tmp_path / "selection"
        )
    assert not (tmp_path / "selection").exists()
