import copy
from dataclasses import dataclass
import hashlib
from pathlib import Path
import shutil

import yaml


class _UniqueLoader(yaml.SafeLoader):
    def __init__(self, stream):
        super().__init__(stream)
        self._checked_mappings = set()

    def flatten_mapping(self, node):
        if node not in self._checked_mappings:
            self._checked_mappings.add(node)
            keys = set()
            for key_node, _ in node.value:
                key = "<<" if key_node.tag == "tag:yaml.org,2002:merge" else self.construct_object(key_node)
                if key in keys:
                    raise ValueError(f"Duplicate YAML key: {key!r}")
                keys.add(key)
        super().flatten_mapping(node)


def read_fasta(path):
    lines = Path(path).read_text().splitlines()
    headers = [line for line in lines if line.startswith(">")]
    if len(headers) != 1 or next((line for line in lines if line.strip()), "") != headers[0]:
        raise ValueError("Provide a FASTA containing exactly one protein")
    sequence = "".join(line.strip() for line in lines if not line.startswith(">")).upper()
    if not sequence or not sequence.isascii() or not sequence.isalpha():
        raise ValueError("The FASTA sequence must contain amino-acid letters")
    return sequence


@dataclass(frozen=True)
class Target:
    document: dict
    ligand_id: str
    msa_mode: str
    base_directory: Path
    source: Path

    @classmethod
    def load(cls, *, fasta=None, native=None, ligand_id=None, msa_mode, msa=None):
        if (fasta is None) == (native is None):
            raise ValueError("Provide exactly one of FASTA or native YAML")
        if msa_mode not in ("local", "server", "single"):
            raise ValueError("Choose local, server, or single MSA mode explicitly")
        if native is not None:
            if not ligand_id:
                raise ValueError("Native YAML requires the designated --ligand-id")
            if msa is not None:
                raise ValueError("Put local MSA paths in the native YAML for each protein")
            source = Path(native).absolute()
            document = yaml.load(source.read_text(), Loader=_UniqueLoader)
        else:
            source = Path(fasta).absolute()
            ligand_id = ligand_id or "B"
            protein = {"id": "A", "sequence": read_fasta(source)}
            if msa is not None:
                if msa_mode != "local":
                    raise ValueError("--msa requires local MSA mode")
                protein["msa"] = str(Path(msa).absolute())
            document = {"version": 1, "sequences": [{"protein": protein}, {"ligand": {"id": ligand_id, "smiles": "C"}}]}
        target = cls(document, ligand_id, msa_mode, source.parent, source)
        target.validate()
        return target

    def validate(self):
        doc = self.document
        if not isinstance(doc, dict) or doc.get("version", 1) != 1:
            raise ValueError("Expected a native Boltz version 1 YAML mapping")
        sequences = doc.get("sequences")
        if not isinstance(sequences, list) or not sequences:
            raise ValueError("Target requires a nonempty sequences list")
        ids, ligands, proteins = set(), [], []
        for entry in sequences:
            if not isinstance(entry, dict) or len(entry) != 1:
                raise ValueError("Each sequence must have exactly one entity type")
            kind, value = next(iter(entry.items()))
            if kind not in ("protein", "ligand", "dna", "rna") or not isinstance(value, dict):
                raise ValueError("Unsupported sequence entity")
            chain_ids = value.get("id")
            chain_ids = chain_ids if isinstance(chain_ids, list) else [chain_ids]
            if not chain_ids or any(not isinstance(i, str) or not i for i in chain_ids):
                raise ValueError("Each chain requires a nonempty string ID")
            for chain in chain_ids:
                if chain in ids:
                    raise ValueError(f"Duplicate target chain ID: {chain}")
                ids.add(chain)
            if kind == "protein":
                if not isinstance(value.get("sequence"), str) or not value["sequence"].strip():
                    raise ValueError("Each protein requires a sequence")
                proteins.append(value)
            if self.ligand_id in chain_ids:
                if kind != "ligand" or len(chain_ids) != 1:
                    raise ValueError("Designate a single ligand chain, not a shared multi-chain entity")
                ligands.append(value)
        if len(ligands) != 1 or not proteins:
            raise ValueError("Target requires a protein and exactly one designated ligand")
        ligand = ligands[0]
        if ("smiles" in ligand) == ("ccd" in ligand):
            raise ValueError("The designated ligand must contain exactly one of smiles or ccd")
        properties = doc.get("properties", [])
        if not isinstance(properties, list):
            raise ValueError("properties must be a list")
        if any(not isinstance(prop, dict) for prop in properties):
            raise ValueError("Each property must be a mapping")
        affinities = [p["affinity"] for p in properties if isinstance(p, dict) and "affinity" in p]
        if affinities and (len(affinities) != 1 or not isinstance(affinities[0], dict) or affinities[0].get("binder") != self.ligand_id):
            raise ValueError("Affinity must refer only to the designated candidate ligand")
        for protein in proteins:
            msa = protein.get("msa")
            if self.msa_mode == "local" and (not isinstance(msa, str) or msa in ("", "empty")):
                raise ValueError("Local MSA mode requires a file for every protein")
            if self.msa_mode == "single" and msa not in (None, "empty"):
                raise ValueError("Single-sequence mode would discard an existing MSA; remove it or choose local/server")
            if msa not in (None, "empty"):
                self._asset(msa)
        if self.msa_mode == "server" and any(p.get("msa") is None for p in proteins) and any(
            p.get("msa") not in (None, "empty") for p in proteins
        ):
            raise ValueError("Boltz cannot mix custom and auto-generated MSAs in the same input")
        templates = doc.get("templates", [])
        if not isinstance(templates, list):
            raise ValueError("templates must be a list")
        for template in templates:
            if not isinstance(template, dict) or ("cif" in template) == ("pdb" in template):
                raise ValueError("Each template requires exactly one cif or pdb path")
            self._asset(template.get("cif", template.get("pdb")))
        constraints = doc.get("constraints", [])
        if not isinstance(constraints, list):
            raise ValueError("constraints must be a list")
        for constraint in constraints:
            if not isinstance(constraint, dict):
                raise ValueError("Each constraint must be a mapping")
            bond = constraint.get("bond", {})
            if not isinstance(bond, dict):
                raise ValueError("A bond constraint must be a mapping")
            if any(isinstance(bond.get(atom), list) and bond[atom] and bond[atom][0] == self.ligand_id for atom in ("atom1", "atom2")):
                raise ValueError("Boltz bond constraints require CCD atoms; replacing this bonded ligand with SMILES is incompatible")

    def _asset(self, value):
        if not isinstance(value, str) or not value:
            raise ValueError("An asset path must be a nonempty string")
        path = Path(value).expanduser()
        path = path if path.is_absolute() else self.base_directory / path
        if not path.is_file():
            raise ValueError(f"Target asset does not exist: {path}")
        return path.absolute()

    def protein_sequence(self, chain_id=None):
        proteins = {}
        for entry in self.document["sequences"]:
            if "protein" in entry:
                protein = entry["protein"]
                ids = protein["id"] if isinstance(protein["id"], list) else [protein["id"]]
                proteins.update({key: protein["sequence"] for key in ids})
        if chain_id is None and len(proteins) != 1:
            raise ValueError("Select --psichic-chain explicitly for a multichain native target")
        chain_id = chain_id or next(iter(proteins))
        if chain_id not in proteins:
            raise ValueError(f"PSICHIC protein chain not found: {chain_id}")
        return chain_id, proteins[chain_id]

    def prepare(self, run_directory):
        doc = copy.deepcopy(self.document)
        assets = Path(run_directory) / "assets"
        assets.mkdir()
        manifest, copied = [], {}

        def preserve(value):
            source = self._asset(value)
            if source not in copied:
                destination = assets / f"asset_{len(copied):04d}{source.suffix}"
                shutil.copyfile(source, destination)
                with destination.open("rb") as handle:
                    digest = hashlib.file_digest(handle, "sha256").hexdigest()
                manifest.append({"source": str(source), "copy": str(destination), "sha256": digest})
                copied[source] = destination
            return str(copied[source])

        for entry in doc["sequences"]:
            if "protein" in entry:
                protein = entry["protein"]
                if self.msa_mode == "single":
                    protein["msa"] = "empty"
                elif protein.get("msa") not in (None, "empty"):
                    protein["msa"] = preserve(protein["msa"])
        for template in doc.get("templates", []):
            key = "cif" if "cif" in template else "pdb"
            template[key] = preserve(template[key])
        if not any("affinity" in prop for prop in doc.get("properties", [])):
            doc.setdefault("properties", []).append({"affinity": {"binder": self.ligand_id}})
        return doc, manifest

    def render(self, prepared, candidate, destination):
        doc = copy.deepcopy(prepared)
        for entry in doc["sequences"]:
            ligand = entry.get("ligand")
            if ligand is not None and ligand["id"] in (self.ligand_id, [self.ligand_id]):
                ligand.pop("ccd", None)
                ligand["smiles"] = candidate.smiles
        destination.write_text(yaml.safe_dump(doc, sort_keys=False))
