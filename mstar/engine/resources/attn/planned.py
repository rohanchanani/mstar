"""Logical attention plans consumed by a bound physical program.

The KV manager supplies page ownership and sequence generations. Attention adds
query order, the context mapping, and causality; it does not allocate cache
state or choose a kernel. A provider may supply ``run`` for retained reference
layers, or the selected node program may consume these plans directly.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import pairwise

import torch

from mstar.engine.resources.attn.base import AttentionManager
from mstar.engine.resources.attn.config import AttentionSpec, AttentionStep, CrossAttentionSpec, refuse_context_only
from mstar.engine.resources.base import EngineResourceInfo
from mstar.engine.resources.kv.plan import KVPlanOutput, KVPlanOutputs, SequenceView
from mstar.engine.resources.step import StepContext


@dataclass(frozen=True)
class AttentionPlan:
    request_ids: tuple[int, ...]
    qo_indptr: torch.Tensor | None
    contexts: tuple[SequenceView, ...]
    causal: bool
    page_size: int
    _query_source: KVPlanOutput | None = field(default=None, repr=False, compare=False)
    _query_lengths: tuple[int, ...] | None = field(default=None, repr=False, compare=False)

    def __getattribute__(self, name):
        # Keep the existing constructor/field API while deferring only this
        # representation. Exposed tensors retain their mutable identity.
        value = object.__getattribute__(self, name)
        if name == "qo_indptr" and value is None:
            source = object.__getattribute__(self, "_query_source")
            if source is not None:
                value = source.cpu_indptrs.qo_indptr
            else:
                lengths = object.__getattribute__(self, "_query_lengths")
                values = [0]
                for length in lengths:
                    values.append(values[-1] + length)
                value = torch.tensor(values, dtype=torch.int32)
            object.__setattr__(self, "qo_indptr", value)
        return value

    @property
    def query_lengths(self) -> tuple[int, ...]:
        # Access via object avoids constructing the tensor for logical uses.
        indptr = object.__getattribute__(self, "qo_indptr")
        if indptr is None:
            if self._query_source is not None:
                return self._query_source.query_lengths
            return self._query_lengths
        values = indptr.tolist()
        return tuple(b - a for a, b in pairwise(values))


AttentionProgram = Callable[[torch.Tensor, AttentionPlan, torch.Tensor], torch.Tensor]


class PlannedAttentionResource(AttentionManager):
    """An inline resource with per-invocation owned host plans."""

    def __init__(
        self, kv_cache: str, page_size: int,
        physical: AttentionProgram | None = None,
    ):
        self._kv_cache_name = kv_cache
        self._page_size = page_size
        self._physical = physical
        self._plans: dict[str, AttentionPlan] = {}
        self.reset_default_cursors()

    @classmethod
    def build(cls, spec: AttentionSpec, info: EngineResourceInfo):
        if spec.config.sliding_window is not None:
            raise ValueError("planned attention does not implement sliding_window")
        if spec.config.sm_scale is not None:
            raise ValueError("planned attention does not implement sm_scale")
        cfg = info.dependency(spec.config.kv_cache).config
        return cls(spec.config.kv_cache, cfg.page_size)

    def logical_use(self, plans):
        from mstar.engine.resources.sequence_view import AttentionUse

        return {label: AttentionUse.from_plan(plan) for label, plan in plans.items()}

    def depends_on(self) -> set[str]:
        return {self._kv_cache_name}

    def can_capture(self) -> bool:
        return False

    def plan(self, step: AttentionStep, ctx: StepContext) -> dict[str, AttentionPlan]:
        refuse_context_only(step, type(self).__name__)
        self.reset_default_cursors()
        outputs: KVPlanOutputs = ctx.plan_results[self._kv_cache_name]
        self._plans = {
            label: AttentionPlan(
                tuple(view.request_id for view in output.views),
                None, tuple(output.views), step.causal, self._page_size,
                _query_source=output,
            )
            for label, output in outputs.items()
        }
        return self._plans

    @torch.compiler.disable
    def run(
        self, q: torch.Tensor, label: str | None = None,
        kv_cache_layer: torch.Tensor | None = None, **kwargs,
    ) -> torch.Tensor:
        if self._physical is None:
            raise RuntimeError("attention is computed by the selected node program")
        return self._physical(q, self._plans[label or self._default_label], kv_cache_layer)

    @torch.compiler.disable
    def select_last_hidden(self, hidden: torch.Tensor, label: str | None = None) -> torch.Tensor:
        plan = self._plans[label or self._default_label]
        return hidden[plan.qo_indptr[1:].to(device=hidden.device, dtype=torch.long) - 1]


class PlannedCrossAttentionResource(PlannedAttentionResource):
    """Queries in decoder plan order, matched to context by request identity."""

    def __init__(
        self, kv_cache: str, page_size: int,
        query_kv_cache: str | None, context_label: str = "context",
        physical: AttentionProgram | None = None,
    ):
        super().__init__(kv_cache, page_size, physical)
        self._query_kv_cache_name = query_kv_cache
        self._context_label = context_label

    @classmethod
    def build(cls, spec: CrossAttentionSpec, info: EngineResourceInfo):
        cfg = info.dependency(spec.config.kv_cache).config
        return cls(spec.config.kv_cache, cfg.page_size,
                   spec.config.query_kv_cache, spec.config.context_label)

    def depends_on(self) -> set[str]:
        keys = {self._kv_cache_name}
        if self._query_kv_cache_name is not None:
            keys.add(self._query_kv_cache_name)
        return keys

    @property
    def context_cache_key(self) -> str:
        return self._kv_cache_name

    def plan(self, step: AttentionStep, ctx: StepContext) -> dict[str, AttentionPlan]:
        refuse_context_only(step, type(self).__name__)
        self.reset_default_cursors()
        contexts: KVPlanOutputs = ctx.plan_results[self._kv_cache_name]
        context_by_request = {v.request_id: v for v in contexts[self._context_label].views}
        queries: dict[str, tuple[tuple[int, ...], KVPlanOutput | None, tuple[int, ...] | None]] = {}
        if self._query_kv_cache_name is not None:
            for label, output in ctx.plan_results[self._query_kv_cache_name].items():
                if self._query_kv_cache_name == self._kv_cache_name and label == self._context_label:
                    continue
                queries[label] = (tuple(v.request_id for v in output.views), output, None)
        else:
            rows: dict[str, list[int]] = {}
            lengths: dict[str, list[int]] = {}
            for segment in step.segments or ():
                rows.setdefault(segment.label, []).append(segment.request_id)
                lengths.setdefault(segment.label, []).append(segment.span)
            queries = {label: (tuple(rids), None, tuple(lengths[label]))
                       for label, rids in rows.items()}
        self._plans = {
            label: AttentionPlan(
                rids, None, tuple(context_by_request[rid] for rid in rids),
                False, self._page_size, _query_source=source, _query_lengths=lengths,
            )
            for label, (rids, source, lengths) in queries.items()
        }
        return self._plans
