"""Real resource managers retain ownership when physical programs consume plans."""
from types import SimpleNamespace

import pytest
import torch

from mstar.distributed.communication import WorkerParallelGroups
from mstar.engine.engine import Engine
from mstar.model.binding import NodeAssembly
from mstar.engine.resources.attn.config import (
    AttentionConfig,
    AttentionSpec,
    AttentionStep,
    CrossAttentionConfig,
    CrossAttentionSpec,
)
from mstar.engine.resources.attn.planned import PlannedAttentionResource, PlannedCrossAttentionResource
from mstar.engine.resources.base import EngineResourceInfo
from mstar.engine.resources.binding import build_bound_resource
from mstar.engine.resources.kv.cache import KVCache
from mstar.engine.resources.kv.config import KVSpec, KVStep, PagedKVConfig
from mstar.engine.resources.kv.manager import KVManager
from mstar.engine.resources.position.config import PositionConfig, PositionStep
from mstar.engine.resources.position.manager import RopeManager
from mstar.engine.resources.runner import StepRunner
from mstar.engine.resources.sampler.config import SamplerStep, SamplingReqConfig
from mstar.engine.resources.sampler.plan import SamplingResult
from mstar.engine.resources.sampler.resource import SamplerResource
from mstar.engine.resources.spec import ResourceViewConflict
from mstar.engine.resources.step import Segment, StepContext, SubmoduleStep

CPU = torch.device("cpu")


def manager(pages=12, name="kv"):
    cfg = PagedKVConfig(num_layers=1, num_kv_heads=1, head_dim=2, max_seq_len=20,
                        page_size=4, max_num_pages=pages, prefix_cache=False)
    storage = torch.zeros((1, pages, 2, 4, 1, 2))
    cache = KVCache(cfg, CPU, torch.float32, storage=storage)
    resource = KVManager(cfg, name, None, None, CPU, torch.float32,
                         cache=cache, host_plan=True)
    assert resource.kv_cache.tensor is storage
    return resource


def context(*rids):
    return StepContext(request_ids=rids, graph_walk="decode", slot=0, capture=False)


def plan(kv, spans, ctx=None):
    step = KVStep(segments=tuple(Segment(rid, label, count) for rid, label, count in spans))
    ctx = ctx or context(*(dict.fromkeys(rid for rid, _, _ in spans)))
    assert kv.admit(step, ctx).ok
    result = kv.plan(step, ctx)
    ctx.plan_results[kv.name] = result
    return step, ctx, result


def test_host_plan_crosses_page_boundaries_and_keeps_old_invocation_storage():
    kv = manager()
    for rid in (10, 20):
        kv.ingest_request(rid)
    step, ctx, first = plan(kv, [(10, "main", 4), (20, "main", 5)])
    write = first["main"].write_locations
    assert write.token_to_cache.tolist() == [0, 1, 2, 3, 0, 1, 2, 3, 0]
    assert not torch.any(write.token_to_page == 0)
    keys = torch.arange(18, dtype=torch.float32).reshape(9, 1, 2)
    kv.write_kv(keys, keys + 100, label="main")
    assert torch.equal(kv.read_kv(plan_label="main")[:, 0], keys)
    kv.commit(step, ctx)
    old_pages = write.token_to_page.clone()
    step2, ctx2, second = plan(kv, [(10, "main", 1), (20, "main", 1)])
    assert second["main"].write_locations.token_to_cache.tolist() == [0, 1]
    assert [v.length for v in second["main"].views] == [5, 6]
    assert [v.length for v in first["main"].views] == [4, 5]
    assert torch.equal(write.token_to_page, old_pages)
    kv.commit(step2, ctx2)
    kv.remove_request(10)
    kv.remove_request(20)
    kv.assert_pages_conserved()
    assert kv._arena.allocator.num_free == kv.config.max_num_pages - 1
    kv.cleanup()


def test_logical_manager_preserves_page_admission():
    kv = manager(pages=3)
    kv.ingest_request(1)
    step, ctx, _ = plan(kv, [(1, "main", 8)])
    kv.commit(step, ctx)
    kv.ingest_request(2)
    pending = KVStep(segments=(Segment(2, "main", 1),))
    assert not kv.admit(pending, context(2)).ok
    kv.rollback_admit(pending, context(2))
    kv.remove_request(1)
    assert kv.admit(pending, context(2)).ok
    kv.rollback_admit(pending, context(2))
    kv.remove_request(2)
    kv.assert_pages_conserved()
    kv.cleanup()


@pytest.mark.parametrize(("spec_class", "config_class", "owner_class"), [
    (AttentionSpec, AttentionConfig, PlannedAttentionResource),
    (CrossAttentionSpec, CrossAttentionConfig, PlannedCrossAttentionResource),
])
def test_shared_attention_requires_an_explicit_physical_choice_at_engine_binding(spec_class, config_class, owner_class):
    spec = spec_class("attn", {"bound", "reference"}, config_class("kv"))
    kv = KVSpec("kv", spec.nodes, PagedKVConfig(
        num_layers=1, num_kv_heads=1, head_dim=2, max_seq_len=20,
    ))
    # Bound-only defaults supply logical plans without choosing a substrate.
    assert type(build_bound_resource(spec, EngineResourceInfo(CPU, dependencies={"kv": kv}))) is owner_class
    engine = Engine(None)
    with pytest.raises(ResourceViewConflict, match="explicit common physical attention"):
        engine.load_model({"bound": None, "reference": None}, [spec, kv],
                          WorkerParallelGroups(1, 0), CPU, None, assemblies={"bound": NodeAssembly(bind=lambda resources: None)})
    assert not engine._resources


def test_shared_attention_accepts_an_explicit_common_physical_owner():
    spec = AttentionSpec("attn", {"bound", "reference"}, AttentionConfig("kv"))
    owner = PlannedAttentionResource("kv", 4, physical=lambda q, plan, cache: q)
    calls = []
    def factory(given_spec, info):
        calls.append((given_spec, info.device))
        return owner
    actual = build_bound_resource(spec, EngineResourceInfo(CPU),
                                  (SimpleNamespace(build=factory),), default_kernel_required=True)
    assert actual is owner and calls == [(spec, CPU)]


def test_positions_and_attention_follow_authoritative_request_order():
    kv = manager()
    positions = RopeManager(PositionConfig("kv"), CPU, host_plan=True)
    for rid in (10, 20):
        kv.ingest_request(rid)
        positions.ingest_request(rid)
    step, ctx, first = plan(kv, [(10, "main", 3), (20, "main", 2),
                                 (20, "context", 5), (10, "context", 6)])
    pstep = PositionStep(segments=step.segments)
    old_positions = positions.plan(pstep, ctx)
    assert old_positions["main"].tolist() == [0, 1, 2, 0, 1]
    cross = PlannedCrossAttentionResource("kv", 4, "kv")
    cross_plan = cross.plan(AttentionStep(causal=False), ctx)["main"]
    assert cross_plan.request_ids == (10, 20)
    assert [v.length for v in cross_plan.contexts] == [6, 5]
    assert cross_plan.qo_indptr.tolist() == [0, 3, 5]
    assert not cross_plan.causal
    self_attn = PlannedAttentionResource("kv", 4)
    self_plan = self_attn.plan(AttentionStep(), ctx)["main"]
    assert self_plan.causal
    kv.commit(step, ctx)
    positions.commit(pstep, ctx)
    step2, ctx2, _ = plan(kv, [(20, "main", 1), (10, "main", 1)])
    next_positions = positions.plan(PositionStep(segments=step2.segments), ctx2)
    assert next_positions["main"].tolist() == [2, 3]
    assert old_positions["main"].tolist() == [0, 1, 2, 0, 1]
    kv.commit(step2, ctx2)
    for rid in (10, 20):
        kv.remove_request(rid)
    kv.cleanup()


def test_sampler_result_commits_once_and_only_kept_rows_through_runner():
    sampler = SamplerResource(32, False, CPU, planned=True)
    for rid in (10, 20):
        sampler.ingest_request(rid, SamplingReqConfig(temperature=0))
    runner = StepRunner({"sample": sampler})
    ctx = context(20, 10)
    step = SubmoduleStep({"sample": SamplerStep(kept_rids=frozenset({10}))})
    step.set_ctx(ctx)
    result = runner.plan(step)["sample"]
    assert result.request_ids == (20, 10)
    assert result.offsets == (0, 0)
    assert result.kept == (False, True)
    tokens = torch.tensor([7, 3])
    runner.accept_results(step, {"sample": SamplingResult(tokens)})
    assert sampler._sampler._step_offset == {10: 0, 20: 0}
    runner.commit(step)
    assert sampler._sampler._step_offset == {10: 1, 20: 0}
    with pytest.raises(RuntimeError, match="requires this invocation"):
        runner.commit(step)
    next_plan = runner.plan(step)["sample"]
    assert next_plan.offsets == (0, 1)
    assert result.offsets == (0, 0)
    assert torch.equal(tokens, torch.tensor([7, 3]))


def test_reference_sampler_callback_uses_identical_plan_commit():
    plans = []
    def greedy(logits, plan):
        plans.append(plan)
        return logits.argmax(-1)
    sampler = SamplerResource(3, False, CPU, planned=True, sample=greedy)
    sampler.ingest_request(7, SamplingReqConfig(temperature=0))
    step, ctx = SamplerStep(), context(7)
    expected = sampler.plan(step, ctx)
    assert sampler.sample([7], torch.tensor([[1., 3., 2.]])).tolist() == [1]
    assert plans == [expected]
    assert sampler._sampler._step_offset[7] == 0
    sampler.commit(step, ctx)
    assert sampler._sampler._step_offset[7] == 1
