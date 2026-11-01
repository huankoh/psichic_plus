"""Translate a string-ID library into AdaptiveFlow collections."""

import csv
import hashlib
import io
import json
import tarfile
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from . import Molecule

COLUMNS = ("collection_id", "molecule_id", "tranche", "smiles")


def safe_token(prefix: str, *parts: str) -> str:
    encoded = json.dumps(parts, ensure_ascii=False, separators=(",", ":")).encode()
    return prefix + hashlib.sha256(encoded).hexdigest()


def collection_key(molecule: Molecule) -> str:
    return f"{safe_token('t', molecule.tranche)}_{safe_token('c', molecule.collection_id)}"


def read_library(path: Path) -> list[Molecule]:
    molecules = []
    seen = set()
    collections = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not set(COLUMNS).issubset(reader.fieldnames):
            raise ValueError(f"CSV requires columns {', '.join(COLUMNS)}")
        if len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError("CSV contains duplicate column names")
        for line, row in enumerate(reader, 2):
            if None in row or any(row[key] is None for key in COLUMNS):
                raise ValueError(f"Malformed CSV row {line}")
            values = {key: row[key] for key in COLUMNS}
            if any(not values[key].strip() for key in COLUMNS[:-1]):
                raise ValueError(f"Empty identifier or tranche on row {line}")
            if any(char in values['smiles'] for char in "\r\n\t"):
                raise ValueError(f"SMILES must occupy one field on one line, row {line}")
            identity = (values['collection_id'], values['molecule_id'])
            if identity in seen:
                raise ValueError(f"Duplicate collection_id/molecule_id on row {line}: {identity!r}")
            seen.add(identity)
            previous = collections.setdefault(values['collection_id'], values['tranche'])
            if previous != values['tranche']:
                raise ValueError(f"Collection {values['collection_id']!r} spans multiple tranches")
            molecules.append(Molecule(safe_token("m", *identity), **values))
    if not molecules:
        raise ValueError("Library is empty")
    if len({m.token for m in molecules}) != len(molecules):
        raise ValueError("Molecule token collision")
    return molecules


def write_archive(path: Path, members: list[Molecule], listing: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    directory = path.name.removesuffix(".tar.gz")
    with tarfile.open(path, "w:gz") as archive:
        contents = {f"{m.token}.smi": f"{m.smiles}\t{m.token}\n" for m in members}
        contents['.listing'] = "".join(listing)
        for name, content in contents.items():
            data = content.encode()
            entry = tarfile.TarInfo(f"{directory}/{name}")
            entry.size = len(data)
            entry.mode = 0o644
            archive.addfile(entry, io.BytesIO(data))


def convert_library(molecules: list[Molecule], output: Path, samples_per_tranche: int) -> list[str]:
    if samples_per_tranche < 1:
        raise ValueError("samples_per_tranche must be positive")
    output.mkdir(parents=True, exist_ok=False)
    full = defaultdict(list)
    tranches = defaultdict(list)
    for molecule in molecules:
        full[collection_key(molecule)].append(molecule)
        tranches[molecule.tranche].append(molecule)

    metadata = []
    full_todo = []
    for key, members in full.items():
        tranche, collection = key.split("_")
        listing = [f"{key},{m.token},{i}\n" for i, m in enumerate(members)]
        write_archive(output / "full" / tranche / f"{collection}.tar.gz", members, listing)
        metadata.append({"Tranche": tranche, "Collection": collection, "LigandCount": len(members)})
        full_todo.append(f"{key} {len(members)}\n")

    samples = []
    sparse_todo = []
    for tranche, members in tranches.items():
        selected = members[:samples_per_tranche]
        af_tranche = safe_token("t", tranche)
        listing = [f"{collection_key(m)},{m.token},{i}\n" for i, m in enumerate(selected)]
        write_archive(output / "sparse" / af_tranche / "0000001.tar.gz", selected, listing)
        sparse_todo.append(f"{af_tranche}_0000001 {len(selected)}\n")
        samples.extend(m.token for m in selected)

    pd.DataFrame(metadata).to_parquet(output / "collections.parquet", index=False)
    pd.DataFrame(metadata).groupby("Tranche", as_index=False)['LigandCount'].sum().to_parquet(
        output / "tranches.parquet", index=False,
    )
    with (output / "manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["token", *COLUMNS, "af_collection"])
        writer.writeheader()
        for molecule in molecules:
            writer.writerow({**asdict(molecule), "af_collection": collection_key(molecule)})
    (output / "full.todo").write_text("".join(full_todo))
    (output / "prescreen.todo").write_text("".join(sparse_todo))
    (output / "prescreen.tokens.json").write_text(json.dumps(samples, indent=2) + "\n")
    return samples
