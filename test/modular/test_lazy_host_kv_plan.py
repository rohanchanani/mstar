"""Deferred host representations keep the eager invocation's values and aliases."""
import sys
from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

import pytest
import torch
from test_planned_resources import manager, plan

from mstar.engine.resources.attn.config import AttentionStep
from mstar.engine.resources.attn.planned import (
    AttentionPlan,
    PlannedAttentionResource,
    PlannedCrossAttentionResource,
)
from mstar.engine.resources.kv import manager as manager_module
from mstar.engine.resources.kv import plan as plan_module
from mstar.engine.resources.kv.plan import KVPlanOutput, KVWritePlan, SequenceView
from mstar.engine.resources.position.config import PositionConfig, PositionStep
from mstar.engine.resources.position.manager import RopeManager
from mstar.engine.resources.sequence_view import AttentionUse


@pytest.fixture(autouse=True)
def cpu_staging_on_mac(monkeypatch):
    # This CPU-only fixture does not exercise unrelated MPS pinned staging.
    if sys.platform == "darwin":
        monkeypatch.setattr(torch.accelerator, "is_available", lambda: False)


@pytest.fixture
def builders(monkeypatch):
    calls = {"indptr": 0, "write": 0}
    for module, name, key in ((manager_module, "build_paged_indptrs", "indptr"),
                              (plan_module, "build_write_plan", "write")):
        original = getattr(module, name)
        def counted(*args, _original=original, _key=key, **kwargs):
            calls[_key] += 1
            return _original(*args, **kwargs)
        monkeypatch.setattr(module, name, counted)
    return calls


def test_logical_self_cross_and_positions_do_not_force_host_tensors(builders):
    kv = manager()
    positions = RopeManager(PositionConfig("kv"), torch.device("cpu"), host_plan=True)
    for rid in (10, 20):
        kv.ingest_request(rid)
        positions.ingest_request(rid)
    step, ctx, output = plan(kv, [(10, "main", 3), (20, "main", 2),
                                  (20, "context", 5), (10, "context", 6)])
    self_attn = PlannedAttentionResource("kv", 4)
    cross = PlannedCrossAttentionResource("kv", 4, "kv")
    self_plan = self_attn.plan(AttentionStep(), ctx)
    cross_plan = cross.plan(AttentionStep(causal=False), ctx)
    assert self_attn.logical_use(self_plan)["main"].query_lengths == (3, 2)
    use = cross.logical_use(cross_plan)["main"]
    assert use.request_ids == (10, 20) and use.context_lengths == (6, 5)
    assert use.query_lengths == (3, 2)
    assert kv.logical_use(output)["main"].spans == (3, 2)
    pos = positions.plan(PositionStep(segments=step.segments), ctx)
    assert pos["main"].tolist() == [0, 1, 2, 0, 1]
    assert output["main"].get_total_len() == 5
    assert builders == {"indptr": 0, "write": 0}
    kv.commit(step, ctx)
    for rid in (10, 20):
        kv.remove_request(rid)
    kv.cleanup()


def test_tensor_consumer_builds_once_and_exposed_mutation_is_authoritative(builders):
    kv = manager()
    out = kv._plan_output([SequenceView(7, "main", [1, 2], 6, 2)])
    attn = PlannedAttentionResource("kv", 4)
    ctx = SimpleNamespace(plan_results={"kv": {"main": out}})
    current = attn.plan(AttentionStep(), ctx)["main"]
    assert current.query_lengths == (2,) and builders["indptr"] == 0
    indptr = out.cpu_indptrs
    indptr.qo_indptr[1] = 4
    assert current.query_lengths == (4,)
    assert current.qo_indptr is indptr.qo_indptr
    current.qo_indptr[1] = 3
    assert AttentionUse.from_plan(current).query_lengths == (3,)
    assert out.query_lengths == (3,) and out.get_total_len() == 3
    assert out.cpu_indptrs is indptr and builders["indptr"] == 1
    addresses = out.write_locations
    assert addresses.token_to_page.tolist() == [2, 2]
    assert addresses.token_to_cache.tolist() == [0, 1]
    addresses.token_to_cache[0] = 2
    assert out.write_locations is addresses
    assert out.write_locations.token_to_cache.tolist() == [2, 1]
    assert builders["write"] == 1
    kv.cleanup()


def test_retained_unmaterialized_plan_survives_next_plan_commit_and_release(builders):
    kv = manager()
    kv.ingest_request(7)
    step, ctx, first = plan(kv, [(7, "main", 4)])
    old = first["main"]
    expected_page = old.views[0].page_idxs[0]
    kv.commit(step, ctx)
    step2, ctx2, second = plan(kv, [(7, "main", 1)])
    kv.commit(step2, ctx2)
    kv.remove_request(7)
    kv.cleanup()
    assert builders == {"indptr": 0, "write": 0}
    # Public view mutation never used to rewrite already-built metadata.
    old.views[0].page_idxs[0] = 999
    assert old.cpu_indptrs.paged_kv_indices.tolist() == [expected_page]
    assert old.write_locations.token_to_page.tolist() == [expected_page] * 4
    assert old.write_locations.token_to_cache.tolist() == [0, 1, 2, 3]
    assert old.query_lengths == (4,)
    assert second["main"].write_locations.token_to_cache.tolist() == [0]
    assert second["main"].query_lengths == (1,)


def test_legacy_write_and_read_resolve_original_shared_addresses_once(builders):
    kv = manager()
    kv.ingest_request(7)
    step, ctx, outputs = plan(kv, [(7, "main", 3)])
    out = outputs["main"]
    state = kv._current_plan_states["main"]
    assert state.total_tokens == 3 and builders["write"] == 0
    keys = torch.arange(6, dtype=torch.float32).reshape(3, 1, 2)
    kv.write_kv(keys, keys + 10, label="main")
    assert state.token_to_page is out.write_locations.token_to_page
    assert state.token_to_cache is out.write_locations.token_to_cache
    assert torch.equal(kv.read_kv(plan_label="main")[:, 0], keys)
    assert builders["write"] == 1 and builders["indptr"] == 0
    # Replacing the output field does not redirect the legacy planned state.
    original = state.token_to_page
    out.write_locations = KVWritePlan(torch.tensor([999]), torch.tensor([0]), 1)
    assert state.token_to_page is original and state.total_tokens == 3
    kv.commit(step, ctx)
    kv.remove_request(7)
    kv.cleanup()


@pytest.mark.parametrize("spans", [(), (0,), (1,), (4,), (0, 4, 1)])
def test_empty_zero_and_batch_metadata_matches_eager_builders(spans):
    kv = manager()
    views = [SequenceView(rid, "main", list(range(1, (span + 7) // 4 + 1)),
                          span + 3, span) for rid, span in enumerate(spans)]
    out = kv._plan_output(views)
    expected = plan_module.build_paged_indptrs(views, 4)
    for actual, wanted in zip(out.cpu_indptrs, expected, strict=True):
        assert torch.equal(actual, wanted)
    expected_write = plan_module.build_write_plan(views, 4)
    assert torch.equal(out.write_locations.token_to_page, expected_write.token_to_page)
    assert torch.equal(out.write_locations.token_to_cache, expected_write.token_to_cache)
    assert out.write_locations.total_tokens == sum(spans)
    kv.cleanup()


def test_cross_without_query_cache_defers_tensor_and_preserves_request_order(builders):
    kv = manager()
    contexts = kv._plan_output([SequenceView(20, "context", [2], 4, 0),
                               SequenceView(10, "context", [1], 3, 0)])
    cross = PlannedCrossAttentionResource("kv", 4, None)
    from mstar.engine.resources.step import Segment
    step = AttentionStep(causal=False, segments=(Segment(10, "main", 2), Segment(20, "main", 1)))
    current = cross.plan(step, SimpleNamespace(plan_results={"kv": {"context": contexts}}))["main"]
    assert current.query_lengths == (2, 1)
    assert [view.length for view in current.contexts] == [3, 4]
    assert builders == {"indptr": 0, "write": 0}
    assert current.qo_indptr.tolist() == [0, 2, 3]
    current.qo_indptr[1] = 1
    assert AttentionUse.from_plan(current).query_lengths == (1, 2)
    assert builders == {"indptr": 0, "write": 0}
    kv.cleanup()


def test_eager_constructor_frozen_fields_and_nonhost_plan_compatibility():
    view = SequenceView(7, "main", [1], 2, 2)
    indptr = torch.tensor([0, 2], dtype=torch.int32)
    current = AttentionPlan((7,), indptr, (view,), True, 4)
    assert current.qo_indptr is indptr and current.query_lengths == (2,)
    assert replace(current, causal=False).qo_indptr is indptr
    with pytest.raises(FrozenInstanceError):
        current.causal = False
    assert KVPlanOutput(None, [view]).cpu_indptrs is None
    nonhost = manager_module.KVManager.__new__(manager_module.KVManager)
    nonhost._host_plan = False
    nonhost._indptr_ring = None
    nonhost.kv_cache = SimpleNamespace(page_size=4)
    out = nonhost._plan_output([view])
    assert out._cpu_indptrs is not None and out.cpu_indptrs.qo_indptr.tolist() == [0, 2]
    assert out.write_locations is None


def test_invalid_write_address_still_rejected_during_plan_construction(builders):
    kv = manager()
    with pytest.raises(IndexError):
        kv._plan_output([SequenceView(7, "main", [1], 5, 1)])
    assert builders == {"indptr": 0, "write": 0}
    kv.cleanup()


def test_host_adapter_also_accepts_existing_eager_decode_metadata(builders):
    kv = manager()
    views = [SequenceView(7, "main", [1], 2, 1)]
    eager = KVPlanOutput(plan_module.build_paged_indptrs(views, 4), views)
    ctx = SimpleNamespace(is_preplan=False)
    kv._setup_plan_states({"main": eager}, ctx, None)
    assert builders["write"] == 0
    assert kv._current_plan_states["main"].token_to_cache.tolist() == [1]
    assert kv._current_plan_states["main"].token_to_page is eager.write_locations.token_to_page
    kv.cleanup()
