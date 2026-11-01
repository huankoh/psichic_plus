"""Convert CSV rows into manifest-backed AFVS collections and inventories."""

import csv
import io
import json
from pathlib import Path
import re
import tarfile

import pandas as pd


FIELDS = ["ligand_id", "molecule_id", "collection_id", "tranche", "smiles"]


def write_csv(path, fields, records):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def collection_key(record):
    return f"{record['tranche']}_{record['collection_id']}"


def write_archive(path, number, records, sample_indices):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "x:gz") as archive:
        members = {f"{row['ligand_id']}.json": json.dumps(row) for row in records}
        members[".listing"] = "".join(
            f"{collection_key(row)},{row['ligand_id']},{sample_indices[row['ligand_id']]}\n"
            for row in records
        )
        for name, contents in members.items():
            data = contents.encode()
            member = tarfile.TarInfo(f"{number}/{name}")
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))


def convert_library(source, destination, samples_per_tranche):
    if samples_per_tranche < 1:
        raise ValueError("samples_per_tranche must be positive")
    with Path(source).open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not set(FIELDS[1:]).issubset(reader.fieldnames or []):
            raise ValueError(
                "Library needs molecule_id,collection_id,tranche,smiles columns"
            )
        records = []
        for index, row in enumerate(reader):
            if any(row[field] is None for field in FIELDS[1:]):
                raise ValueError(f"Missing field in CSV row {index + 2}")
            if not re.fullmatch(r"[A-Za-z][A-Za-z0-9]*", row["tranche"]):
                raise ValueError(
                    "Tranche must start with an ASCII letter and contain only ASCII letters and digits"
                )
            if not re.fullmatch(r"[0-9]+", row["collection_id"]):
                raise ValueError(
                    "collection_id must be a string of digits, including any leading zeros"
                )
            records.append(
                {
                    "ligand_id": f"L{index:012d}",
                    **{field: row[field] for field in FIELDS[1:]},
                }
            )
    if not records:
        raise ValueError("Library has no molecules")

    destination = Path(destination)
    destination.mkdir(parents=True)
    write_csv(destination / "manifest.csv", FIELDS, records)
    collections = {}
    tranches = {}
    sample_indices = {}
    for row in records:
        collections.setdefault(collection_key(row), []).append(row)
        tranche_rows = tranches.setdefault(row["tranche"], [])
        sample_indices[row["ligand_id"]] = len(tranche_rows)
        tranche_rows.append(row)

    inventory = []
    for rows in collections.values():
        row = rows[0]
        write_archive(
            destination
            / "collections"
            / row["tranche"]
            / f"{row['collection_id']}.tar.gz",
            row["collection_id"],
            rows,
            sample_indices,
        )
        inventory.append(
            {
                "Tranche": row["tranche"],
                "Collection": row["collection_id"],
                "LigandCount": len(rows),
            }
        )
    pd.DataFrame(inventory).to_parquet(destination / "collections.parquet", index=False)
    pd.DataFrame(
        [
            {"Tranche": tranche, "LigandCount": len(rows)}
            for tranche, rows in tranches.items()
        ]
    ).to_parquet(destination / "tranches.parquet", index=False)

    with (destination / "prescreen.todo").open("w") as todo:
        for index, rows in enumerate(tranches.values()):
            sparse_tranche = f"SPARSE{index}"
            number = "0000000"
            sampled = rows[:samples_per_tranche]
            write_archive(
                destination / "collections" / sparse_tranche / f"{number}.tar.gz",
                number,
                sampled,
                sample_indices,
            )
            todo.write(f"{sparse_tranche}_{number} {len(sampled)}\n")
    return records
