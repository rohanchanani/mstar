"""Planned attention preserves query order and refuses unsupported semantics."""
from types import SimpleNamespace

import pytest
import torch

from mstar.engine.resources.attn.config import AttentionConfig, AttentionSpec, AttentionStep
from mstar.engine.resources.attn.base import AttentionManager
from mstar.engine.resources.attn.config import AttnBackend
from mstar.engine.resources.attn.aiter import AiterPagedAttentionManager
from mstar.engine.resources.attn.planned import PlannedAttentionResource, PlannedCrossAttentionResource
from mstar.engine.resources.base import EngineResourceInfo
from mstar.engine.resources.kv.config import KVSpec, PagedKVConfig
from mstar.engine.resources.kv.plan import KVPlanOutput, SequenceView, build_paged_indptrs
from mstar.engine.resources.step import Segment


@pytest.mark.parametrize("option,value", [("sliding_window", 16), ("sm_scale", 0.125)])
def test_planned_build_refuses_unrepresented_options_before_dependency_lookup(option, value):
    spec = AttentionSpec("attn", {"node"}, AttentionConfig("kv", **{option: value}))
    with pytest.raises(ValueError, match=option):
        PlannedAttentionResource.build(spec, EngineResourceInfo(torch.device("cpu")))


@pytest.mark.parametrize("owner", [PlannedAttentionResource, PlannedCrossAttentionResource,
                                   AiterPagedAttentionManager])
def test_context_only_refused_before_plan_mutation(owner):
    resource = owner.__new__(owner)
    def cannot_mutate():
        pytest.fail("unsupported step reached plan mutation")
    resource.reset_default_cursors = cannot_mutate
    with pytest.raises(ValueError, match="context_only"):
        resource.plan(AttentionStep(context_only=True), None)


@pytest.mark.parametrize("option,value", [("sliding_window", 16), ("sm_scale", 0.125)])
def test_aiter_factory_keeps_upstream_option_validation(option, value):
    spec = AttentionSpec("attn", {"node"}, AttentionConfig("kv", backend=AttnBackend.AITER, **{option: value}))
    cfg = PagedKVConfig(num_layers=1, num_kv_heads=1, head_dim=64, max_seq_len=32)
    info = EngineResourceInfo(torch.device("cpu"), dependencies={"kv": KVSpec("kv", {"node"}, cfg)})
    with pytest.raises(ValueError, match=option):
        AttentionManager.build(spec, info)


def output(rows):
    return KVPlanOutput(None, rows,
                        indptr_builder=lambda: build_paged_indptrs(rows, 4),
                        query_lengths=tuple(row.to_compute for row in rows))


def test_default_plans_preserve_order_causality_and_lazy_lengths():
    query = output([SequenceView(10, "main", [1], 3, 3),
                    SequenceView(20, "main", [2], 2, 2)])
    context = output([SequenceView(20, "context", [3, 4], 5, 5),
                      SequenceView(10, "context", [5, 6], 6, 6)])
    ctx = SimpleNamespace(plan_results={"kv": {"main": query, "context": context}})
    self_attn = PlannedAttentionResource("kv", 4)
    plan = self_attn.plan(AttentionStep(), ctx)["main"]
    assert plan.request_ids == (10, 20)
    assert plan.query_lengths == (3, 2) and plan.causal
    cross = PlannedCrossAttentionResource("kv", 4, "kv")
    plan = cross.plan(AttentionStep(causal=False), ctx)["main"]
    assert plan.request_ids == (10, 20)
    assert plan.query_lengths == (3, 2) and not plan.causal
    assert [view.length for view in plan.contexts] == [6, 5]
    assert query._cpu_indptrs is None
    assert plan.qo_indptr.tolist() == [0, 3, 5]
    assert plan.qo_indptr is query.cpu_indptrs.qo_indptr
    plan.qo_indptr[1] = 2
    assert plan.query_lengths == (2, 3)


def test_cross_without_query_cache_uses_step_segments():
    context = output([SequenceView(20, "context", [3], 3, 3),
                      SequenceView(10, "context", [4], 4, 4)])
    ctx = SimpleNamespace(plan_results={"kv": {"context": context}})
    cross = PlannedCrossAttentionResource("kv", 4, None)
    plan = cross.plan(AttentionStep(causal=False, segments=(Segment(10, "main", 2), Segment(20, "main", 1))), ctx)["main"]
    assert plan.request_ids == (10, 20) and plan.query_lengths == (2, 1)
    assert [view.length for view in plan.contexts] == [4, 3]
    assert plan.qo_indptr.tolist() == [0, 2, 3]
