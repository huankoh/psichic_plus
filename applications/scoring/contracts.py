"""Versioned declarations crossing the coordinator/worker boundary."""

from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
import re

from applications.scorer import BetaDistribution, DistributionFailure, DistributionNotRequested, Output
from applications.scorer.boltz2.models import (
    EnsembleMethod, Metric, Mode, OMITTED, PSICHICScore, Plan, finite_number,
)

SCHEMA = 1
MAX_BATCH = 4096


class ContractError(ValueError):
    pass


class StateError(RuntimeError):
    pass


class RunBusy(StateError):
    pass


class IncompleteRun(StateError):
    pass


class Engine(str, Enum):
    PSICHIC = "psichic"
    BOLTZ = "boltz"


class State(str, Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    UNKNOWN = "interrupted-or-unknown"


class Ownership(str, Enum):
    LOCAL = "task-owned-local"
    EXTERNAL = "external-job"


class Resolution(str, Enum):
    RUNNING = "still-running"
    UNKNOWN = "unresolved"
    STOPPED = "confirmed-stopped"


class Acceptance(str, Enum):
    ACCEPTED = "accepted"
    IDENTICAL = "identical"
    STALE = "superseded"
    INCOMPLETE = "incomplete-transfer"


class SelectionState(str, Enum):
    WAITING = "waiting"
    SHORTAGE = "shortage"
    COMMITTED = "committed"


class Completion(str, Enum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    SHORTAGE = "shortage"


def canonical(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ContractError("Expected JSON without non-finite values") from error


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def loads(value):
    def unique(pairs):
        result = {}
        for key, item in pairs:
            if key in result:
                raise ContractError(f"Duplicate JSON key {key!r}")
            result[key] = item
        return result

    try:
        result = json.loads(value, object_pairs_hook=unique)
        canonical(result)
        return result
    except (ValueError, TypeError) as error:
        raise ContractError("Invalid JSON contract") from error


def fields(value, names):
    if not isinstance(value, dict) or set(value) != set(names.split()):
        raise ContractError(f"Expected exact fields: {names}")


def text(value, name, *, empty=False):
    if not isinstance(value, str) or (not empty and not value.strip()):
        raise ContractError(f"{name} must be a string" + ("" if empty else " with content"))


def sha(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ContractError("Expected a lowercase SHA256 digest")


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{32}", value):
        raise ContractError("Expected a UUID hex identity")


def row_key(ordinal):
    if type(ordinal) is not int or not 1 <= ordinal <= 999999999999:
        raise ContractError("Manifest ordinal is outside the supported range")
    return f"row_{ordinal:012d}"


def key(value):
    if not isinstance(value, str) or not re.fullmatch(r"row_[0-9]{12}", value) or int(value[4:]) == 0:
        raise ContractError("Invalid manifest row key")


@dataclass(frozen=True)
class Asset:
    name: str
    sha256: str

    def __post_init__(self):
        text(self.name, "asset name")
        sha(self.sha256)

    @classmethod
    def from_dict(cls, value):
        fields(value, "name sha256")
        return cls(**value)


def assets(value, *, required=False):
    if not isinstance(value, tuple) or any(not isinstance(a, Asset) for a in value):
        raise ContractError("Assets must be a tuple of Asset declarations")
    if len({a.name for a in value}) != len(value) or (required and not value):
        raise ContractError("Asset names must be unique and required assets cannot be missing")


@dataclass(frozen=True)
class Target:
    identity: str
    sequence: str
    assets: tuple[Asset, ...]

    def __post_init__(self):
        text(self.identity, "target identity")
        if not isinstance(self.sequence, str) or not re.fullmatch(r"[A-Z]+", self.sequence):
            raise ContractError("Target sequence must already be normalized uppercase letters")
        assets(self.assets, required=True)

    @classmethod
    def from_dict(cls, value):
        fields(value, "identity sequence assets")
        return cls(value["identity"], value["sequence"], tuple(Asset.from_dict(a) for a in value["assets"]))


@dataclass(frozen=True)
class Entry:
    id: str
    smiles: str
    source: str
    metadata_json: str = "{}"

    def __post_init__(self):
        text(self.id, "entry ID", empty=True)
        text(self.smiles, "SMILES", empty=True)
        text(self.source, "source locator")
        metadata = loads(self.metadata_json)
        if not isinstance(metadata, dict):
            raise ContractError("Entry metadata must be an object")
        object.__setattr__(self, "metadata_json", canonical(metadata))


@dataclass(frozen=True)
class ManifestRow:
    ordinal: int
    entry: Entry

    @property
    def key(self):
        return row_key(self.ordinal)

    def to_dict(self):
        return {"ordinal": self.ordinal, "key": self.key, "id": self.entry.id, "smiles": self.entry.smiles,
                "source": self.entry.source, "metadata": loads(self.entry.metadata_json)}


@dataclass(frozen=True)
class Settings:
    mode: str
    shortlist_size: int | None = None
    ensemble: EnsembleMethod | None = None
    metric: Metric | None = None
    psichic_weight: float | None = None
    boltz_weight: float | None = None
    psichic_output: Output = Output.AFFINITY
    psichic_options: str | None = None
    boltz_options: str | None = None
    replicas: int = 1

    def __post_init__(self):
        if self.ensemble is not None and not isinstance(self.ensemble, EnsembleMethod):
            raise ContractError("Use a typed ensemble method")
        if self.metric is not None and not isinstance(self.metric, Metric):
            raise ContractError("Use a typed Boltz metric")
        if type(self.replicas) is not int or self.replicas != 1:
            raise ContractError("Only one replica is supported; replica aggregation needs a separate contract")
        object.__setattr__(self, "psichic_output", Output(self.psichic_output))
        if self.mode == "psichic-only":
            if any(v is not None for v in (self.shortlist_size, self.ensemble, self.metric,
                                           self.psichic_weight, self.boltz_weight, self.boltz_options)):
                raise ContractError("PSICHIC-only does not accept Boltz or ensemble settings")
        else:
            mode = Mode(self.mode)
            method = self.ensemble or (EnsembleMethod.CONCENTRATION if mode.ensemble else OMITTED)
            plan = Plan(mode, self.shortlist_size, self.metric or OMITTED,
                        self.psichic_weight if self.psichic_weight is not None else OMITTED,
                        self.boltz_weight if self.boltz_weight is not None else OMITTED, method,
                        self.psichic_output if mode is not Mode.BOLTZ else OMITTED)
            for name in ("ensemble", "metric", "psichic_weight", "boltz_weight"):
                object.__setattr__(self, name, getattr(plan, name))
        for engine in Engine:
            value = getattr(self, engine.value + "_options")
            if engine in self.engines:
                parsed = loads(value)
                if not isinstance(parsed, dict) or not parsed:
                    raise ContractError("Required resolved engine options cannot be missing or empty")
                object.__setattr__(self, engine.value + "_options", canonical(parsed))
            elif value is not None:
                raise ContractError("Options provided for an unused engine")

    @property
    def engines(self):
        if self.mode == "psichic-only":
            return (Engine.PSICHIC,)
        if self.mode == Mode.BOLTZ.value:
            return (Engine.BOLTZ,)
        return (Engine.PSICHIC, Engine.BOLTZ)

    @property
    def pyramid(self):
        return self.mode in (Mode.PYRAMID_ENSEMBLE.value, Mode.REFINEMENT.value)

    @property
    def required_output(self):
        return Output.DISTRIBUTION if self.ensemble is EnsembleMethod.CONCENTRATION else self.psichic_output

    @classmethod
    def from_dict(cls, value):
        fields(value, "mode shortlist_size ensemble metric psichic_weight boltz_weight psichic_output psichic_options boltz_options replicas")
        args = dict(value)
        args["ensemble"] = EnsembleMethod(args["ensemble"]) if args["ensemble"] else None
        args["metric"] = Metric(args["metric"]) if args["metric"] else None
        return cls(**args)


@dataclass(frozen=True)
class BatchContext:
    row_keys: tuple[str, ...]
    inputs_sha256: str
    batch_size: int
    seed: int
    replica: int = 0

    def __post_init__(self):
        if not isinstance(self.row_keys, tuple) or not 1 <= len(self.row_keys) <= MAX_BATCH:
            raise ContractError("Batch must contain 1 to 4096 rows")
        for value in self.row_keys:
            key(value)
        if len(set(self.row_keys)) != len(self.row_keys):
            raise ContractError("Duplicate batch row identity")
        sha(self.inputs_sha256)
        if type(self.batch_size) is not int or self.batch_size != len(self.row_keys):
            raise ContractError("Batch size must match actual submitted row context")
        if type(self.seed) is not int or type(self.replica) is not int or self.replica != 0:
            raise ContractError("Explicit integer seed and replica zero are required")

    @classmethod
    def from_dict(cls, value):
        fields(value, "row_keys inputs_sha256 batch_size seed replica")
        return cls(tuple(value["row_keys"]), value["inputs_sha256"], value["batch_size"], value["seed"], value["replica"])

    @classmethod
    def for_rows(cls, rows, *, seed):
        if not 1 <= len(rows) <= MAX_BATCH:
            raise ContractError("Bound the submitted batch to 4096 rows")
        return cls(tuple(r.key for r in rows), digest([r.to_dict() for r in rows]), len(rows), seed)


@dataclass(frozen=True)
class Fingerprint:
    engine: Engine
    engine_version: str
    submitted_smiles: str
    target: Target
    native_input_sha256: str
    weights: tuple[Asset, ...]
    loaded_source: tuple[Asset, ...]
    environment_json: str
    device: str
    precision: str
    parser_contract: str
    call_profile: str
    resolved_options: str
    batch: BatchContext
    output: Output
    msa_mode: str
    msa_assets: tuple[Asset, ...]
    schema_version: int = SCHEMA

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != SCHEMA:
            raise ContractError("Unsupported fingerprint schema")
        if not isinstance(self.engine, Engine) or not isinstance(self.target, Target) or not isinstance(self.batch, BatchContext):
            raise ContractError("Fingerprint requires typed engine, target and batch")
        if not isinstance(self.output, Output):
            raise ContractError("Fingerprint requires typed output")
        text(self.submitted_smiles, "submitted SMILES", empty=True)
        sha(self.native_input_sha256)
        assets(self.weights, required=True)
        assets(self.loaded_source, required=True)
        assets(self.msa_assets)
        for name in ("engine_version", "device", "precision", "parser_contract", "call_profile"):
            text(getattr(self, name), name)
        env = loads(self.environment_json)
        fields(env, "python packages runtime")
        text(env["python"], "Python version")
        text(env["runtime"], "runtime")
        if not isinstance(env["packages"], dict) or not env["packages"]:
            raise ContractError("Package versions cannot be missing")
        for name, version in env["packages"].items():
            text(name, "package name")
            text(version, "package version")
        options = loads(self.resolved_options)
        if not isinstance(options, dict) or not options:
            raise ContractError("Resolved options cannot be missing")
        object.__setattr__(self, "environment_json", canonical(env))
        object.__setattr__(self, "resolved_options", canonical(options))
        if self.msa_mode != "single" or self.msa_assets:
            raise ContractError("This contract permits only explicit single-sequence MSA with no MSA assets")
        if self.engine is Engine.BOLTZ and (self.engine_version != "2.2.1" or self.output is not Output.AFFINITY):
            raise ContractError("Boltz requires version 2.2.1 and affinity output")

    @property
    def sha256(self):
        return digest(asdict(self))

    @classmethod
    def from_dict(cls, value):
        fields(value, "engine engine_version submitted_smiles target native_input_sha256 weights loaded_source environment_json device precision parser_contract call_profile resolved_options batch output msa_mode msa_assets schema_version")
        args = dict(value)
        args.update(engine=Engine(args["engine"]), target=Target.from_dict(args["target"]),
                    batch=BatchContext.from_dict(args["batch"]), output=Output(args["output"]))
        for name in ("weights", "loaded_source", "msa_assets"):
            args[name] = tuple(Asset.from_dict(a) for a in args[name])
        return cls(**args)


@dataclass(frozen=True)
class Task:
    run_id: str
    population_digest: str
    entry_key: str
    engine: Engine
    attempt_id: str
    invocation_id: str
    ownership: Ownership
    fingerprint: Fingerprint
    schema_version: int = SCHEMA

    def __post_init__(self):
        identifier(self.run_id)
        identifier(self.attempt_id)
        identifier(self.invocation_id)
        sha(self.population_digest)
        key(self.entry_key)
        if type(self.schema_version) is not int or self.schema_version != SCHEMA:
            raise ContractError("Unsupported task schema")
        if not isinstance(self.ownership, Ownership) or not isinstance(self.fingerprint, Fingerprint):
            raise ContractError("Task requires typed ownership and fingerprint")
        if not isinstance(self.engine, Engine) or self.engine != self.fingerprint.engine:
            raise ContractError("Task and fingerprint engine differ")
        if self.entry_key not in self.fingerprint.batch.row_keys:
            raise ContractError("Task is absent from its batch context")

    def to_dict(self):
        return {**asdict(self), "fingerprint_sha256": self.fingerprint.sha256}

    def to_json(self):
        return canonical(self.to_dict())

    @classmethod
    def from_dict(cls, value):
        fields(value, "run_id population_digest entry_key engine attempt_id invocation_id ownership fingerprint schema_version fingerprint_sha256")
        args = dict(value)
        expected = args.pop("fingerprint_sha256")
        args.update(engine=Engine(args["engine"]), ownership=Ownership(args["ownership"]),
                    fingerprint=Fingerprint.from_dict(args["fingerprint"]))
        result = cls(**args)
        if result.fingerprint.sha256 != expected:
            raise ContractError("Fingerprint hash mismatch")
        return result

    @classmethod
    def from_json(cls, value):
        return cls.from_dict(loads(value))


def validate_record(task, value):
    """Return a canonical raw outcome; retain affinity when distribution failed."""
    if not isinstance(value, dict):
        raise ContractError("Outcome must be an object")
    if value.get("status") == State.FAILED.value:
        fields(value, "status stage reason")
        text(value["stage"], "failure stage")
        text(value["reason"], "failure reason")
        return value
    if value.get("status") != State.SUCCEEDED.value:
        raise ContractError("A receipt must contain success or explicit failure")
    if task.engine is Engine.BOLTZ:
        fields(value, "status affinity_pred_value affinity_probability_binary native submitted_smiles")
        for name in ("affinity_pred_value", "affinity_probability_binary"):
            if not finite_number(value[name]):
                raise ContractError("Boltz output must be finite")
        if not 0 <= value["affinity_probability_binary"] <= 1 or not isinstance(value["native"], dict):
            raise ContractError("Invalid Boltz output")
    else:
        fields(value, "status raw_affinity adaptive_score submitted_smiles normalized_smiles distribution")
        d = value["distribution"]
        if not isinstance(d, dict):
            raise ContractError("Distribution must be a tagged object")
        match d.get("state"):
            case "valid":
                fields(d, "state alpha beta concentration")
                if not all(finite_number(d[n]) for n in ("alpha", "beta", "concentration")):
                    raise ContractError("Distribution values must be finite numbers")
                distribution = BetaDistribution(d["alpha"], d["beta"])
                if distribution.concentration != d["concentration"]:
                    raise ContractError("Distribution concentration differs from alpha plus beta")
            case "not-requested":
                fields(d, "state")
                distribution = DistributionNotRequested()
                if task.fingerprint.output is Output.DISTRIBUTION:
                    raise ContractError("Distribution-required task cannot accept affinity-only evidence")
            case "distribution-failure":
                fields(d, "state reason raw_alpha raw_beta")
                for name in ("reason", "raw_alpha", "raw_beta"):
                    text(d[name], name)
                distribution = DistributionFailure(d["reason"], d["raw_alpha"], d["raw_beta"])
            case _:
                raise ContractError("Unknown distribution state")
        if value["normalized_smiles"] is not None:
            text(value["normalized_smiles"], "normalized SMILES", empty=True)
        checked = PSICHICScore(value["raw_affinity"], value["adaptive_score"], value["submitted_smiles"],
                               value["normalized_smiles"], distribution).validated()
        if checked.distribution != distribution:
            raise ContractError("Declared valid distribution is inconsistent with affinity")
        if value["adaptive_score"] != -value["raw_affinity"]:
            raise ContractError("Adaptive score must be negative raw affinity")
    if value["submitted_smiles"] != task.fingerprint.submitted_smiles:
        raise ContractError("Outcome SMILES differs from the submitted input")
    canonical(value)
    return value
