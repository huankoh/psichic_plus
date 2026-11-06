"""Block base class, JobArgs, and RuntimeArgs.

Mirrors Chaperone's BlockFunction, JobArgs, and RuntimeArgs shapes.
Blocks are dataclasses here so the skeleton stays stdlib only. The
manifest emitted per block is the contract for CLI and MCP generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import ClassVar

RESERVED_FIELDS = frozenset(
    {
        "run_args",
        "data",
        "job_args",
        "mode",
        "recompile",
        "allow_errors",
        "force_exec",
        "skip_cache",
        "skip_chunk_cache",
        "use_local_gpus",
        "max_retries",
    }
)


@dataclass
class JobArgs:
    """Portable resource request. Site config maps classes to a cluster."""

    partition: str | None = None
    gpu_class: str | None = None
    gres: str | None = None
    mem_gb: int = 8
    time_limit: str = "5:00:00"
    cpus_per_task: int = 1
    account: str | None = None
    qos: str | None = None
    rows_per_job: int | None = None
    job_name: str | None = None

    def merged_over(self, other: "JobArgs | None") -> "JobArgs":
        """Return self with unset fields filled from ``other``."""
        if other is None:
            return self
        defaults = JobArgs()
        out = {}
        for f in fields(self):
            mine, theirs, default = (
                getattr(self, f.name),
                getattr(other, f.name),
                getattr(defaults, f.name),
            )
            out[f.name] = mine if mine != default else theirs
        return JobArgs(**out)


@dataclass
class RuntimeArgs:
    """Execution mode and cache controls. Mirrors Chaperone RuntimeArgs."""

    mode: str = "local"  # local | slurm (slurm lands in a later phase)
    recompile: bool = False
    allow_errors: bool = False
    force_exec: bool = False
    skip_cache: bool = False
    skip_chunk_cache: bool = False
    use_local_gpus: bool = True
    max_retries: int = 0


class BlockFunction:
    """Base class for blocks. Subclasses set ClassVars and implement run.

    Table blocks implement ``execute(table, ctx) -> table``.
    Chunked scorers set ``_chunked = True`` and implement
    ``score_rows(rows, ctx) -> None``, mutating row dicts in place with
    namespaced metrics and an ``<instance>.status`` per row.
    """

    _name: ClassVar[str] = ""
    _role: ClassVar[str] = (
        "block"  # source | scorer | generator | combiner | selector | sink
    )
    _description: ClassVar[str] = ""
    _resources: ClassVar[JobArgs | None] = None
    _requires_input: ClassVar[bool] = True
    _chunked: ClassVar[bool] = False
    _environment: ClassVar[str] = "in-process"
    _produces: ClassVar[tuple[str, ...]] = ()
    _requires: ClassVar[tuple[str, ...]] = ()

    name: str = ""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        own = (
            set(getattr(cls, "__dataclass_fields__", {}))
            if hasattr(cls, "__dataclass_fields__")
            else set()
        )
        conflicts = RESERVED_FIELDS & own - {"name"}
        if conflicts:
            raise TypeError(
                f"Block {cls.__name__} uses reserved field names: {conflicts}"
            )

    @property
    def instance(self) -> str:
        return self.name or self._name

    def execute(self, table, ctx):  # table blocks override this
        raise NotImplementedError

    def score_rows(self, rows, ctx):  # chunked scorers override this
        raise NotImplementedError

    @classmethod
    def manifest(cls) -> dict:
        params = {}
        for f in fields(cls) if hasattr(cls, "__dataclass_fields__") else []:
            if f.name == "name":
                continue
            params[f.name] = {
                "default": repr(f.default) if f.default is not field else None
            }
        resources = cls._resources
        return {
            "name": cls._name,
            "role": cls._role,
            "description": cls._description,
            "environment": cls._environment,
            "requires_input": cls._requires_input,
            "chunked": cls._chunked,
            "requires": list(cls._requires),
            "produces": list(cls._produces),
            "resources": None
            if resources is None
            else {
                "partition": resources.partition,
                "gpu_class": resources.gpu_class,
                "mem_gb": resources.mem_gb,
                "rows_per_job": resources.rows_per_job,
            },
            "parameters": params,
        }
