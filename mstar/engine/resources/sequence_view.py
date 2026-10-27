"""Logical use handles and explicitly selected physical sequence access."""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import pairwise
from typing import Any

import torch

from mstar.engine.resources.kv.config import ResolvedSequenceConfig
from mstar.engine.resources.kv.encoding import PlainKVEncoding
from mstar.engine.resources.sequence import DenseSequenceView, HeadSharded, PagedSequenceView


@dataclass(frozen=True)
class SequenceUse:
    request_ids: tuple[int | str, ...]
    labels: tuple[str, ...]
    lengths: tuple[int, ...]
    spans: tuple[int, ...]
    starts: tuple[int, ...]
    _plan: Any = field(repr=False, compare=False)

    @classmethod
    def from_plan(cls, plan):
        return cls(tuple(view.request_id for view in plan.views),
                   tuple(view.label for view in plan.views),
                   tuple(view.length for view in plan.views),
                   tuple(view.to_compute for view in plan.views),
                   tuple(view.start for view in plan.views), plan)


@dataclass(frozen=True)
class AttentionUse:
    request_ids: tuple[int | str, ...]
    query_lengths: tuple[int, ...]
    context_lengths: tuple[int, ...]
    causal: bool
    _plan: Any = field(repr=False, compare=False)

    @classmethod
    def from_plan(cls, plan):
        # Planned attention owns host indptrs; no device readback occurs here.
        lengths = getattr(plan, "query_lengths", None)
        if lengths is None:
            lengths = tuple(b - a for a, b in pairwise(plan.qo_indptr.tolist()))
        return cls(plan.request_ids, lengths,
                   tuple(view.length for view in plan.contexts), plan.causal, plan)


@dataclass(frozen=True)
class DenseWriteRow:
    request_id: int | str
    label: str
    positions: torch.Tensor
    keys: tuple[torch.Tensor, ...]
    values: tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class DenseReadRow:
    request_id: int | str
    keys: tuple[torch.Tensor, ...]
    values: tuple[torch.Tensor, ...]
    valid_length: int
    mask: torch.Tensor


class PagedSequenceAccess:
    def __init__(self, resource, request: PagedSequenceView):
        self.resource = resource
        self.config = resource.config
        self.encoding = _check_encoding(resource, request)
        if request.page_size is not None and request.page_size != self.config.page_size:
            raise ValueError("requested page size contradicts resolved sequence storage")
        _check_partition(self.config, request)

    def layer(self, index: int) -> torch.Tensor:
        return self.resource.kv_cache.tensor[index]

    def for_step(self, use: SequenceUse):
        return use._plan

    def for_attention(self, use: AttentionUse):
        return use._plan


class DenseSequenceAccess:
    def __init__(self, resource, request: DenseSequenceView):
        self.resource = resource
        self.config = resource.config
        self.encoding = _check_encoding(resource, request)
        if not isinstance(self.config, ResolvedSequenceConfig) or not self.config.dense:
            raise ValueError("contiguous sequence access requires resolved dense storage")
        _check_partition(self.config, request)

    def _layers(self, view):
        if len(view.page_idxs) != 1 or view.start != 0:
            raise ValueError("dense sequence access requires one complete request block")
        page = view.page_idxs[0]
        tensor = self.resource.kv_cache.tensor
        return (tuple(tensor[layer, page, 0] for layer in range(self.config.num_layers)),
                tuple(tensor[layer, page, 1] for layer in range(self.config.num_layers)))

    def for_step(self, use: SequenceUse) -> tuple[DenseWriteRow, ...]:
        rows = []
        for view in use._plan.views:
            keys, values = self._layers(view)
            positions = torch.arange(view.length - view.to_compute, view.length, dtype=torch.int64)
            rows.append(DenseWriteRow(view.request_id, view.label, positions, keys, values))
        return tuple(rows)

    def for_attention(self, use: AttentionUse) -> tuple[DenseReadRow, ...]:
        rows = []
        capacity = self.config.max_seq_len
        for rid, q, view in zip(use.request_ids, use.query_lengths, use._plan.contexts, strict=True):
            keys, values = self._layers(view)
            if use.causal:
                if q > view.length:
                    raise ValueError("causal queries exceed their sequence context")
                # Bottom-right alignment: row i sees columns <= length - q + i.
                mask = torch.full((q, capacity), float("-inf"), dtype=torch.float32)
                mask.triu_(diagonal=view.length - q + 1)
            else:
                mask = torch.full((q, capacity), float("-inf"), dtype=torch.float32)
                mask[:, :max(0, view.length)].zero_()
            rows.append(DenseReadRow(rid, keys, values, view.length, mask))
        return tuple(rows)


def _check_partition(config, request):
    if isinstance(config, ResolvedSequenceConfig):
        expected = "head_sharded" if isinstance(request.partition, HeadSharded) else "replicated"
        if config.partition != expected:
            raise ValueError("requested partition contradicts resolved sequence storage")


def _check_encoding(resource, request):
    encoding = resource.kv_cache.encoding
    config = resource.config
    inherited = (config.inherited_encoding if isinstance(config, ResolvedSequenceConfig)
                 else PlainKVEncoding(resource.kv_cache.dtype))
    requested = inherited if request.encoding is None else request.encoding
    if requested != encoding:
        raise ValueError("requested encoding contradicts resolved sequence storage")
    return encoding


def sequence_access(resource, request):
    if isinstance(request, DenseSequenceView):
        return DenseSequenceAccess(resource, request)
    if isinstance(request, PagedSequenceView):
        return PagedSequenceAccess(resource, request)
    raise TypeError(f"unsupported sequence view {request!r}")
