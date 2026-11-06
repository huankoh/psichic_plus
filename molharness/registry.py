"""String-name block registry. Mirrors Chaperone's Blocks registry.

Builtins register with the ``@register`` decorator. Third-party blocks
register through the ``molharness.blocks`` entry-point group later.
"""

from __future__ import annotations

from .block_config import BlockFunction


class BlockRegistry:
    def __init__(self):
        self._blocks: dict[str, type[BlockFunction]] = {}

    def register(self, cls: type[BlockFunction]) -> type[BlockFunction]:
        name = cls._name
        if not name:
            raise ValueError(f"Block {cls.__name__} has no _name")
        if name in self._blocks:
            raise ValueError(
                f"Duplicate block {name!r}: {self._blocks[name].__module__} and {cls.__module__}"
            )
        self._blocks[name] = cls
        return cls

    def get(self, name: str) -> type[BlockFunction]:
        try:
            return self._blocks[name]
        except KeyError:
            raise KeyError(
                f"Block {name!r} not found. Available: {sorted(self._blocks)}"
            ) from None

    def __contains__(self, name: str) -> bool:
        return name in self._blocks

    def keys(self):
        return self._blocks.keys()

    def manifests(self) -> dict[str, dict]:
        return {name: cls.manifest() for name, cls in sorted(self._blocks.items())}


Blocks = BlockRegistry()
register = Blocks.register
