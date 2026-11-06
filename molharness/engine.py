"""ExecutionEngine. Mirrors Chaperone's engine over candidate tables.

Local execution only in this skeleton phase. The SLURM executor plugs in
behind the same run, submit, gather, and cycle surface later.
"""

from __future__ import annotations

import json
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .block_config import BlockFunction, JobArgs, RuntimeArgs
from .data import CandidateTable
from .registry import Blocks


@dataclass
class ChunkContext:
    instance: str
    chunk_index: int
    rows_in_chunk: int
    job_args: JobArgs
    run_args: RuntimeArgs


class ExecutionEngine:
    def __init__(
        self,
        rundir: str | Path,
        reinit: bool = True,
        run_args_default: RuntimeArgs | None = None,
        job_args_default: JobArgs | None = None,
        _namespace: str = "",
    ):
        self.rundir = Path(rundir)
        self.rundir.mkdir(parents=True, exist_ok=True)
        self.run_args_default = run_args_default or RuntimeArgs()
        self.job_args_default = job_args_default
        self._namespace = _namespace
        self._manifest_path = self.rundir / "manifest.json"
        if reinit or not self._manifest_path.exists():
            self._steps: list[dict] = []
            self._save_manifest()
        else:
            self._steps = json.loads(self._manifest_path.read_text()).get("steps", [])

    def _save_manifest(self) -> None:
        self._manifest_path.write_text(
            json.dumps({"schema_version": 1, "steps": self._steps}, indent=1) + "\n"
        )

    def _resolve(self, block, kwargs) -> BlockFunction:
        if isinstance(block, BlockFunction):
            if kwargs:
                raise ValueError(
                    "kwargs require a block name or class, not an instance"
                )
            return block
        cls = Blocks.get(block) if isinstance(block, str) else block
        return cls(**kwargs)

    def run(
        self,
        block,
        data: CandidateTable | None = None,
        run_args: RuntimeArgs | None = None,
        job_args: JobArgs | None = None,
        **kwargs,
    ) -> CandidateTable:
        instance_block = self._resolve(block, kwargs)
        run_args = run_args or self.run_args_default
        if run_args.mode != "local":
            raise NotImplementedError(
                f"mode {run_args.mode!r} lands with the SLURM executor phase"
            )
        merged_jobs = (
            (job_args or JobArgs())
            .merged_over(self.job_args_default)
            .merged_over(instance_block._resources)
        )
        if instance_block._requires_input and data is None:
            raise ValueError(f"Block {instance_block.instance!r} requires data=")
        table = data.copy() if data is not None else CandidateTable()
        only_selected = getattr(instance_block, "only_selected", None)
        if instance_block._chunked:
            pending = table.pending(
                instance_block.instance, only_selected=only_selected
            )
            size = merged_jobs.rows_per_job or 512
            if size < 1:
                raise ValueError("rows_per_job must be positive")
            by_key = {r["row_key"]: r for r in table.rows}
            for start in range(0, len(pending), size):
                chunk = pending[start : start + size]
                ctx = ChunkContext(
                    instance_block.instance,
                    start // size,
                    len(chunk),
                    merged_jobs,
                    run_args,
                )
                scored = [dict(r) for r in chunk]
                instance_block.score_rows(scored, ctx)
                for row in scored:
                    by_key[row["row_key"]].update(row)
            done = len(pending)
        else:
            ctx = ChunkContext(
                instance_block.instance, 0, len(table), merged_jobs, run_args
            )
            table = instance_block.execute(table, ctx)
            done = len(table)
        step = f"{self._namespace}{instance_block.instance}"
        self._steps.append(
            {
                "step": step,
                "block": instance_block._name,
                "role": instance_block._role,
                "rows": done,
                "mode": run_args.mode,
            }
        )
        self._save_manifest()
        table.write_json(self.rundir / f"{step.replace('/', '_')}.json")
        return table

    def submit(self, block, data=None, **kwargs) -> Future:
        pool = ThreadPoolExecutor(max_workers=1)
        future = pool.submit(self.run, block, data, **kwargs)
        pool.shutdown(wait=False)
        return future

    @staticmethod
    def gather(*futures: Future) -> CandidateTable:
        """Merge tables by row_key. Later results win per column."""
        merged: dict[str, dict] = {}
        order: list[str] = []
        for future in futures:
            for row in future.result().rows:
                key = row["row_key"]
                if key not in merged:
                    merged[key] = dict(row)
                    order.append(key)
                else:
                    merged[key].update(row)
        return CandidateTable([merged[k] for k in order])

    @contextmanager
    def cycle(
        self, name: str, n_cycles: int
    ) -> Iterator[tuple[int, "ExecutionEngine"]]:
        """Yield (index, sub-engine) pairs. Use as a loop, not a with block.

        Example: ``for i, sub in engine.cycle("opt", 3): ...`` via
        ``yield from`` semantics provided by ``cycles`` below. This context
        form exists for symmetry with Chaperone and yields the list once.
        """
        yield list(enumerate(self._cycle_engines(name, n_cycles)))

    def cycles(
        self, name: str, n_cycles: int
    ) -> Iterator[tuple[int, "ExecutionEngine"]]:
        yield from enumerate(self._cycle_engines(name, n_cycles))

    def _cycle_engines(self, name: str, n_cycles: int) -> list["ExecutionEngine"]:
        return [
            ExecutionEngine(
                self.rundir,
                reinit=False,
                run_args_default=self.run_args_default,
                job_args_default=self.job_args_default,
                _namespace=f"{self._namespace}{name}/cycle_{i}/",
            )
            for i in range(n_cycles)
        ]

    def compile(self, table: CandidateTable) -> CandidateTable:
        table.write_json(self.rundir / "compiled.json")
        table.write_csv(self.rundir / "compiled.csv")
        return table
