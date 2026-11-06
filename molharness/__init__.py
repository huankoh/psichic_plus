"""Small-molecule harness skeleton.

A Chaperone-shaped engine over candidate and pair tables. See
docs/small-molecule-harness-design.md for the full design and
molharness/README.md for the Chaperone-to-here mapping.
"""

from . import blocks as _blocks  # noqa: F401  (registers builtins)
from .block_config import BlockFunction, JobArgs, RuntimeArgs
from .data import CandidateTable, Target, expand_pairs
from .engine import ExecutionEngine
from .registry import Blocks

__all__ = [
    "BlockFunction",
    "Blocks",
    "CandidateTable",
    "ExecutionEngine",
    "JobArgs",
    "RuntimeArgs",
    "Target",
    "expand_pairs",
]
