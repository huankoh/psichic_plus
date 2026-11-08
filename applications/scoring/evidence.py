"""Worker publications and coordinator-owned immutable artifact imports."""

from dataclasses import asdict, dataclass
import hashlib
import math
import os
from pathlib import Path, PurePosixPath
import stat
import uuid

from applications.scorer import BetaDistribution
from applications.scorer.boltz2.adapter import parse_affinity
from applications.scorer.boltz2.models import Candidate, PSICHICScore, finite_number
from .contracts import (
    ContractError, Engine, SCHEMA, State, Task, canonical, digest, fields, loads, sha, text, validate_record,
)

MAX_DOCUMENT = 16 * 1024 * 1024
COMMON_ROLES = frozenset({"native_input", "provenance", "execution", "stdout", "stderr"})


def encode_native_parameter(value):
    """Retain non-finite float identity without ambiguous JSON strings or nulls."""
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite_float": repr(value)}
    return value


def _decode_native_parameter(value):
    if isinstance(value, dict):
        fields(value, "nonfinite_float")
        if value["nonfinite_float"] not in ("nan", "inf", "-inf"):
            raise ContractError("Invalid non-finite native parameter encoding")
        return float(value["nonfinite_float"])
    return value


def fsync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_file(path, content):
    with Path(path).open("x", encoding="utf-8") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def atomic_pointer(path, value):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    write_file(temporary, canonical(value) + "\n")
    os.replace(temporary, path)
    fsync_directory(path.parent)


def safe_path(value):
    text(value, "artifact path")
    path = PurePosixPath(value)
    if path.is_absolute() or str(path) != value or any(p in (".", "..") for p in path.parts) or "\\" in value:
        raise ContractError("Artifact paths must be normalized relative paths without traversal")
    return path


def open_relative(root, relative):
    """Open without following a symlink in any path component, including races."""
    parts = safe_path(relative).parts
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in parts[:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = child
        leaf = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
        if not stat.S_ISREG(os.fstat(leaf).st_mode):
            os.close(leaf)
            raise ContractError("Artifact must be a regular file")
        return os.fdopen(leaf, "rb")
    finally:
        os.close(fd)


def read_document(root, relative):
    with open_relative(root, relative) as handle:
        data = handle.read(MAX_DOCUMENT + 1)
    if len(data) > MAX_DOCUMENT:
        raise ContractError("Contract document exceeds 16 MiB; use bounded invocation artifacts")
    return loads(data)


@dataclass(frozen=True)
class Artifact:
    role: str
    path: str
    size: int
    sha256: str

    def __post_init__(self):
        text(self.role, "artifact role")
        safe_path(self.path)
        if type(self.size) is not int or self.size < 0:
            raise ContractError("Artifact size must be a nonnegative integer")
        sha(self.sha256)

    @classmethod
    def from_file(cls, root, role, path):
        with open_relative(root, path) as handle:
            size = os.fstat(handle.fileno()).st_size
            checksum = hashlib.file_digest(handle, "sha256").hexdigest()
        return cls(role, path, size, checksum)

    @classmethod
    def from_dict(cls, value):
        fields(value, "role path size sha256")
        return cls(**value)


@dataclass(frozen=True)
class Receipt:
    task: Task
    record_json: str
    artifacts: tuple[Artifact, ...]
    schema_version: int = SCHEMA

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != SCHEMA:
            raise ContractError("Unsupported receipt schema")
        if not isinstance(self.task, Task):
            raise ContractError("Receipt requires a Task")
        record = validate_record(self.task, loads(self.record_json))
        object.__setattr__(self, "record_json", canonical(record))
        if not isinstance(self.artifacts, tuple) or any(not isinstance(a, Artifact) for a in self.artifacts):
            raise ContractError("Receipt artifacts must be a tuple of Artifact declarations")
        roles = {a.role for a in self.artifacts}
        if len(roles) != len(self.artifacts) or len({a.path for a in self.artifacts}) != len(self.artifacts):
            raise ContractError("Duplicate artifact role or path")
        required = COMMON_ROLES
        if record["status"] == State.SUCCEEDED.value:
            required |= {"native_affinity", "structure"} if self.task.engine is Engine.BOLTZ else {"raw_psichic"}
        if not required <= roles:
            raise ContractError(f"Missing required artifacts: {sorted(required - roles)}")
        for artifact in self.artifacts:
            if artifact.role not in {"stdout", "stderr"} and artifact.size == 0:
                raise ContractError("Required evidence cannot be an empty file")
        native = next(a for a in self.artifacts if a.role == "native_input")
        if native.sha256 != self.task.fingerprint.native_input_sha256:
            raise ContractError("Native input artifact differs from fingerprint")
        object.__setattr__(self, "artifacts", tuple(sorted(self.artifacts, key=lambda a: a.role)))

    def to_dict(self):
        return {"schema_version": self.schema_version, "task": self.task.to_dict(),
                "record": loads(self.record_json), "artifacts": [asdict(a) for a in self.artifacts]}

    def to_json(self):
        return canonical(self.to_dict())

    @property
    def sha256(self):
        return digest(self.to_dict())

    @classmethod
    def from_dict(cls, value):
        fields(value, "schema_version task record artifacts")
        return cls(Task.from_dict(value["task"]), canonical(value["record"]),
                   tuple(Artifact.from_dict(a) for a in value["artifacts"]), value["schema_version"])

    @classmethod
    def from_json(cls, value):
        return cls.from_dict(loads(value))

    def publish(self, directory):
        """Worker-only publication; coordinator state is never touched."""
        directory = Path(directory)
        for artifact in self.artifacts:
            with open_relative(directory, artifact.path) as handle:
                if os.fstat(handle.fileno()).st_size != artifact.size or hashlib.file_digest(handle, "sha256").hexdigest() != artifact.sha256:
                    raise ContractError("Artifact digest/size mismatch")
                os.fsync(handle.fileno())
        temporary = directory / f".receipt.{uuid.uuid4().hex}.tmp"
        write_file(temporary, self.to_json() + "\n")
        # link is no-clobber publication, unlike replace, even with competing publishers.
        os.link(temporary, directory / "receipt.json")
        fsync_directory(directory)
        temporary.unlink()
        return directory / "receipt.json"


def validate_native(receipt, staging):
    paths = {a.role: a.sha256 for a in receipt.artifacts}
    provenance = read_document(staging, paths["provenance"])
    if canonical(provenance) != canonical(asdict(receipt.task.fingerprint)):
        raise ContractError("Provenance artifact differs from execution fingerprint")
    execution = read_document(staging, paths["execution"])
    fields(execution, "schema_version invocation_id command returncode")
    if type(execution["schema_version"]) is not int or execution["schema_version"] != SCHEMA or execution["invocation_id"] != receipt.task.invocation_id:
        raise ContractError("Execution evidence has wrong invocation identity")
    if not isinstance(execution["command"], list) or not execution["command"]:
        raise ContractError("Execution evidence requires a command")
    for argument in execution["command"]:
        text(argument, "command argument", empty=True)
    if type(execution["returncode"]) is not int:
        raise ContractError("Execution evidence requires an observed exit status")
    record = loads(receipt.record_json)
    if record["status"] == State.FAILED.value:
        return
    if receipt.task.engine is Engine.BOLTZ:
        read_document(staging, paths["native_affinity"])
        native = parse_affinity(Path(staging) / paths["native_affinity"],
                                Candidate(1, "", receipt.task.fingerprint.submitted_smiles))
        expected = {"status": "succeeded", "affinity_pred_value": native.affinity_pred_value,
                    "affinity_probability_binary": native.affinity_probability_binary,
                    "native": native.native, "submitted_smiles": native.submitted_smiles}
        if canonical(record) != canonical(expected):
            raise ContractError("Native Boltz JSON differs from parsed receipt fields")
        return
    raw = read_document(staging, paths["raw_psichic"])
    fields(raw, "schema_version task native")
    if type(raw["schema_version"]) is not int or raw["schema_version"] != SCHEMA or Task.from_dict(raw["task"]) != receipt.task:
        raise ContractError("Raw PSICHIC output has wrong task identity")
    native = raw["native"]
    if not isinstance(native, dict) or not finite_number(native.get("predicted_binding_affinity")) or native["predicted_binding_affinity"] != record["raw_affinity"]:
        raise ContractError("Native PSICHIC affinity differs from receipt")
    d = record["distribution"]
    if d["state"] == "not-requested":
        return
    alpha, beta = (_decode_native_parameter(native.get(name)) for name in ("reg_alpha", "reg_beta"))
    valid = all(finite_number(v) and v > 0 for v in (alpha, beta))
    if valid:
        try:
            parsed = PSICHICScore(record["raw_affinity"], record["adaptive_score"], record["submitted_smiles"],
                                   distribution=BetaDistribution(alpha, beta)).validated()
            valid = isinstance(parsed.distribution, BetaDistribution)
        except ValueError:
            valid = False
    if d["state"] == "valid":
        if not valid or alpha != d["alpha"] or beta != d["beta"]:
            raise ContractError("Native PSICHIC distribution differs from receipt")
    elif valid or d["raw_alpha"] != repr(alpha) or d["raw_beta"] != repr(beta):
        raise ContractError("Distribution failure does not match native diagnostics")


def stage_receipt(run_path, transfer):
    receipt = Receipt.from_dict(read_document(transfer, "receipt.json"))
    staging = run_path / "staging" / uuid.uuid4().hex
    staging.mkdir()
    write_file(staging / "receipt.json", receipt.to_json() + "\n")
    for artifact in receipt.artifacts:
        destination = staging / artifact.sha256
        owned = run_path / "blobs" / artifact.sha256
        if destination.exists() or owned.exists():
            with open_relative(transfer, artifact.path) as source:
                if os.fstat(source.fileno()).st_size != artifact.size or hashlib.file_digest(source, "sha256").hexdigest() != artifact.sha256:
                    raise ContractError("Transferred shared artifact digest/size mismatch")
            if not destination.exists():
                with open_relative(run_path, f"blobs/{artifact.sha256}") as stored:
                    if os.fstat(stored.fileno()).st_size != artifact.size or hashlib.file_digest(stored, "sha256").hexdigest() != artifact.sha256:
                        raise ContractError("Coordinator-owned immutable artifact was modified")
                os.link(owned, destination)
            continue
        checksum, size = hashlib.sha256(), 0
        with open_relative(transfer, artifact.path) as source, destination.open("xb") as out:
            while chunk := source.read(1024 * 1024):
                checksum.update(chunk)
                size += len(chunk)
                out.write(chunk)
            out.flush()
            os.fsync(out.fileno())
        if size != artifact.size or checksum.hexdigest() != artifact.sha256:
            raise ContractError("Transferred artifact digest/size mismatch")
    validate_native(receipt, staging)
    fsync_directory(staging)
    return receipt, staging


def publish_evidence(run_path, receipt, staging):
    for artifact in receipt.artifacts:
        source = staging / artifact.sha256
        destination = run_path / "blobs" / artifact.sha256
        if destination.exists():
            with destination.open("rb") as stored:
                if os.fstat(stored.fileno()).st_size != artifact.size or hashlib.file_digest(stored, "sha256").hexdigest() != artifact.sha256:
                    raise ContractError("Coordinator-owned immutable artifact was modified")
        else:
            source.chmod(0o444)
            os.link(source, destination)
    fsync_directory(run_path / "blobs")
    destination = run_path / "receipts" / f"{receipt.sha256}.json"
    if not destination.exists():
        (staging / "receipt.json").chmod(0o444)
        os.link(staging / "receipt.json", destination)
    fsync_directory(run_path / "receipts")
