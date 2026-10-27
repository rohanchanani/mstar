"""Model-owned logical weight identities, independent of a local representation.

The source reads checkpoint tensors. Consumers choose partitioning, packing and
placement after these identities have been resolved. No reference module or
reference-specific parameter loader is required to obtain the logical values.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType

import torch


@dataclass(frozen=True)
class WeightSpec:
    name: str
    shape: tuple[int, ...]
    source: str | None = None
    alias_of: str | None = None
    default: Callable[[], torch.Tensor] | None = None


class LogicalWeightSource:
    """A repeatable checkpoint reader and its model-owned interpretation.

    ``default`` supplies an architectural value absent from the checkpoint,
    such as a projection's zero bias. ``alias_of`` means logical identity, so
    aliases resolve to the same Tensor instead of copying or re-reading it.
    Completeness and shape are checked before physical preparation starts.
    """

    def __init__(
        self,
        specs: Iterable[WeightSpec],
        tensors: Callable[[], Iterable[tuple[str, torch.Tensor]]],
    ):
        self.specs = tuple(specs)
        by_name = {spec.name: spec for spec in self.specs}
        if len(by_name) != len(self.specs):
            raise ValueError("logical weight names must be unique")
        for spec in self.specs:
            if spec.alias_of is not None and spec.alias_of not in by_name:
                raise ValueError(f"weight {spec.name!r} aliases unknown {spec.alias_of!r}")
        self.by_name: Mapping[str, WeightSpec] = MappingProxyType(by_name)
        self._tensors = tensors

    def load(self, names: Iterable[str] | None = None) -> dict[str, torch.Tensor]:
        requested = tuple(self.by_name) if names is None else tuple(names)
        needed = set(requested)
        visiting: set[str] = set()

        def include(name: str) -> None:
            if name in visiting:
                raise ValueError(f"logical weight alias cycle at {name!r}")
            spec = self.by_name[name]
            if spec.alias_of is not None:
                visiting.add(name)
                needed.add(spec.alias_of)
                include(spec.alias_of)
                visiting.remove(name)

        for name in requested:
            include(name)
        source_names: dict[str, list[str]] = {}
        for name in needed:
            spec = self.by_name[name]
            if spec.alias_of is None:
                source_names.setdefault(spec.source or name, []).append(name)
        values: dict[str, torch.Tensor] = {}
        for source_name, value in self._tensors():
            for name in source_names.get(source_name, ()):
                if name in values:
                    raise ValueError(f"checkpoint supplies weight {name!r} more than once")
                values[name] = value

        def resolve(name: str) -> torch.Tensor:
            spec = self.by_name[name]
            if name not in values:
                if spec.alias_of is not None:
                    values[name] = resolve(spec.alias_of)
                elif spec.default is not None:
                    values[name] = spec.default()
                else:
                    raise ValueError(f"checkpoint is missing logical weight {name!r} ({spec.source or name})")
            value = values[name]
            if tuple(value.shape) != spec.shape:
                raise ValueError(
                    f"logical weight {name!r}: expected shape {spec.shape}, got {tuple(value.shape)}"
                )
            return value

        return {name: resolve(name) for name in requested}

    def iter_weights(self, names: Iterable[str] | None = None) -> Iterator[tuple[str, torch.Tensor]]:
        yield from self.load(names).items()
