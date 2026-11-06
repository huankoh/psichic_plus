# molharness skeleton

This package is phase 1 of the design in [the harness design doc](../docs/small-molecule-harness-design.md).
It mirrors the Chaperone backbone shape over a small-molecule data contract.
It contains no models, no GPU code, and no scheduler submission yet.
Fake blocks prove the composition rules end to end on CPU in seconds.

## Chaperone backbone mapping

Measured on Chaperone at commit `2d3ac1e` with a depth-1 clone.
Total `src` is 225 Python files and about 73.8k lines.
Tool blocks under `src/blocks` are 128 files and about 38.0k lines.
Evals under `src/chaperone-evals` are 46 files and about 16.5k lines.
The backbone under `src/chaperone` is 43 files and about 17.0k lines.
The CLI under `src/chaperone_cli` adds 8 files and about 2.3k lines.

| Chaperone backbone piece | Size | Here |
|---|---|---|
| `chaperone/engine` ExecutionEngine, chunking, resolve | 5 files, 2.7k lines | `molharness/engine.py` |
| `chaperone/data` PipelineDataHandler, schema | 6 files, 4.3k lines | `molharness/data.py` candidate and pair tables |
| `chaperone/block_config` BlockFunction base | 10 files, 2.2k lines | `molharness/block_config.py` |
| `chaperone/orchestration` JobArgs, SLURM runners | 11 files, 2.8k lines | `JobArgs` now, SLURM executor later |
| `blocks/__init__` lazy registry | part of tool blocks | `molharness/registry.py` |
| `chaperone/manifest.py`, `settings.py` | about 0.7k lines | Run manifest now, site settings later |
| `chaperone/containers.py`, `evals.py` | about 2.2k lines | Deferred |

The generic core worth mirroring is roughly 8k to 10k lines once structure-specific data handling, containers, evals, and cloud code are set aside.
This skeleton implements the smallest honest subset of that core in about 700 lines.
It keeps Chaperone spellings where they fit.
Examples are `ExecutionEngine`, `run`, `submit`, `gather`, `cycles`, `JobArgs`, `RuntimeArgs`, and string block names.
It uses dataclasses instead of pydantic so the skeleton stays stdlib only.
Manifests still emit per block for later CLI and MCP generation.

## What works now

Brute-force scoring and ranking run through CSV source, fake scorer, and rank combiner blocks.
Pyramid gating runs through a top-K selector plus `only_selected` on the second scorer.
Shortage marking, pending-only rescoring, and chunking by `rows_per_job` are covered by tests.
A proposer generator runs under engine cycles with per-round lineage.
Multi-target pair expansion covers full grids and sparse pair lists.
SLURM mode raises an explicit not-implemented error instead of silently running locally.

## Run the tests

```bash
python -m pytest molharness/tests -q
```

## Next phases

Phase 2 adds the real PSICHIC+ scorer block behind the same chunked scorer seam.
Phase 3 adds the SLURM executor behind `RuntimeArgs(mode="slurm")`.
Later phases add Boltz-2, SyntheMol and REINVENT generator blocks, the native adaptive strategy, and MCP generation from manifests.
Each phase keeps the pipeline scripts from earlier phases working unchanged.
