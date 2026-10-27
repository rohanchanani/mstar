"""Dense initialization follows execution, while cancellation follows row ownership."""
import pytest
import torch
from test_admit_rollback import _StubTransfer
from mstar.engine.resources.kv import manager as manager_module
from mstar.engine.resources.kv.manager import KVManager
from mstar.engine.resources.kv.config import KVSpec, KVStep, SequenceKVConfig
from mstar.engine.resources.sequence import (DenseSequenceView, PagedSequenceView,
    PlainKVEncoding, SequenceBudget, resolve_sequence_config)
from mstar.engine.resources.step import Segment, StepContext

CPU = torch.device("cpu")

@pytest.fixture(autouse=True)
def stub_transfer(monkeypatch):
    monkeypatch.setattr(manager_module, "KVTransferManager", _StubTransfer)


def owner(*, pages=3, encoding=None):
    logical = spec(pages=pages, storage_dtype=torch.float32 if encoding is None else encoding.dtype)
    cfg = resolve_sequence_config(logical, (DenseSequenceView(encoding=encoding),), dtype=torch.float32)
    return KVManager(cfg, "kv", None, None, CPU, host_plan=True)


def spec(*, pages=3, heads=1, storage_dtype=torch.float32):
    # A budget for this many full-capacity blocks, including the sink block.
    geometry = SequenceKVConfig(num_layers=2, num_kv_heads=heads, head_dim=2, max_seq_len=4)
    element_bytes = torch.empty((), dtype=storage_dtype).element_size()
    return KVSpec(resource_key="kv", nodes={"decoder"}, config=geometry,
                  budget=SequenceBudget(pages * 2 * 2 * 4 * heads * 2 * element_bytes))

def context(rid=1, *, is_preplan=False):
    return StepContext(request_ids=(rid,), graph_walk="decode", slot=0,
                       capture=False, is_preplan=is_preplan)

def step(span, rid=1):
    return KVStep(segments=(Segment(rid, "main", span),))

def test_half_owner_two_live_refusal_reassignment_preserves_actual_survivor_bytes():
    half = PlainKVEncoding(torch.float16)
    resource = owner(encoding=half)
    view = resource.view(DenseSequenceView(encoding=half))
    assert resource.config.max_num_pages == 3
    rows = {}
    try:
        for rid in (1, 2):
            resource.ingest_request(rid, None)
            current, ctx = step(1, rid), context(rid)
            assert resource.admit(current, ctx).ok
            plans = resource.plan(current, ctx)
            row, = view.for_step(resource.logical_use(plans)["main"])
            for tensor in (*row.keys, *row.values):
                tensor[0].fill_(rid + .125)
            row.keys[0][0, 0, 0] = -0.0
            rows[rid] = row
            resource.commit(current, ctx)
        survivor = tuple(t.view(torch.uint8).clone() for t in (*rows[2].keys, *rows[2].values))
        resource.ingest_request(3, None)
        refused, third_ctx = step(1, 3), context(3)
        outcome = resource.admit(refused, third_ctx)
        assert not outcome.ok and not outcome.reason.terminal
        resource.rollback_admit(refused, third_ctx)
        resource.remove_request(1)
        for tensor in (*rows[1].keys, *rows[1].values):
            tensor.fill_(float("nan"))
        assert resource.admit(refused, third_ctx).ok
        assert all(torch.isnan(t).all() for t in (*rows[1].keys, *rows[1].values))
        plans = resource.plan(refused, third_ctx)
        replacement, = view.for_step(resource.logical_use(plans)["main"])
        assert replacement.keys[0].data_ptr() == rows[1].keys[0].data_ptr()
        assert all(torch.count_nonzero(t) == 0 for t in (*replacement.keys, *replacement.values))
        for before, after in zip(survivor, (*rows[2].keys, *rows[2].values), strict=True):
            assert torch.equal(before, after.view(torch.uint8))
        resource.commit(refused, third_ctx)
        for rid in (2, 3):
            resource.remove_request(rid)
        resource.assert_pages_conserved()
        assert resource._arena.num_free == 2
    finally:
        resource.cleanup()

def test_dense_rollback_preserves_retained_bytes_and_reset_initialization():
    resource = owner()
    try:
        resource.ingest_request(1, None)
        current, ctx = step(1), context()
        assert resource.admit(current, ctx).ok
        resource.plan(current, ctx)
        resource.commit(current, ctx)
        stream = resource._streams[1]["main"]
        held = list(stream.page_indices)
        resource.kv_cache.tensor[:, held[0]].fill_(7)
        before = resource.kv_cache.tensor.clone()
        resource.reset_request(1)
        pending = list(resource._pending_initialization[(1, "main")])
        assert resource.admit(current, ctx).ok
        resource.rollback_admit(current, ctx)
        assert not stream.step_in_flight
        assert stream.page_indices == held
        assert resource._pending_initialization[(1, "main")] == pending
        assert torch.equal(resource.kv_cache.tensor, before)
        assert resource.admit(current, ctx).ok
        resource.plan(current, ctx)
        assert torch.count_nonzero(resource.kv_cache.tensor[:, held[0]]) == 0
        resource.commit(current, ctx)
    finally:
        resource.cleanup()


def test_dense_rows_unwind_independently_and_keep_unrelated_initialization():
    resource = owner()
    try:
        resource.kv_cache.tensor.fill_(13)
        for rid in (1, 2):
            resource.ingest_request(rid, None)
            assert resource.admit(step(1, rid), context(rid)).ok
        other = list(resource._pending_initialization[(2, "main")])
        resource.rollback_admit(step(1, 1), context(1))
        assert (1, "main") not in resource._pending_initialization
        assert resource._pending_initialization[(2, "main")] == other
        assert (2, "main") in resource._admit_reserved_pages
        assert resource._streams[2]["main"].step_in_flight
        assert torch.all(resource.kv_cache.tensor == 13)
        resource.rollback_admit(step(1, 2), context(2))
        assert not resource._pending_initialization
        assert not resource._admit_reserved_pages
        assert not resource._streams[2]["main"].step_in_flight
        resource.assert_pages_conserved()
        assert resource._arena.num_free == 2
    finally:
        resource.cleanup()


@pytest.mark.parametrize("cancel", [False, True])
def test_dense_preplan_defers_initialization_and_retains_it_when_unstaged(cancel):
    cfg = resolve_sequence_config(spec(), (DenseSequenceView(),), dtype=torch.float32)
    resource = KVManager(cfg, "kv", None, None, CPU, host_plan=False)
    try:
        resource.ingest_request(1, None)
        resource.kv_cache.tensor.fill_(17)
        current, pre_ctx = step(1), context(is_preplan=True)
        assert resource.admit(current, pre_ctx).ok
        resource.plan(current, pre_ctx)
        assert torch.all(resource.kv_cache.tensor == 17)
        page, = resource._pending_initialization[(1, "main")]
        if cancel:
            resource.clear_preplan()
            # Unstaging retains ordinary span reservations for a later step;
            # rollback_admit, not clear_preplan, releases those pages.
            assert resource._pending_initialization[(1, "main")] == [page]
            assert resource._streams[1]["main"].page_indices == [page]
            assert not resource._streams[1]["main"].step_in_flight
            assert torch.all(resource.kv_cache.tensor == 17)
        ctx = context()
        assert resource.admit(current, ctx).ok
        resource.plan(current, ctx)
        assert torch.count_nonzero(resource.kv_cache.tensor[:, page]) == 0
        assert not resource._pending_initialization
        resource.commit(current, ctx)
        assert not resource._streams[1]["main"].step_in_flight
        resource.assert_pages_conserved()
    finally:
        resource.cleanup()


def test_preplan_refusal_unwinds_reserved_dense_pages_on_execution_context():
    cfg = resolve_sequence_config(spec(), (DenseSequenceView(),), dtype=torch.float32)
    resource = KVManager(cfg, "kv", None, None, CPU, host_plan=False)
    try:
        resource.ingest_request(1, None)
        current, pre_ctx, ctx = step(1), context(is_preplan=True), context()
        assert resource.admit(current, pre_ctx).ok
        resource.plan(current, pre_ctx)
        assert resource.admit(current, ctx).ok
        resource.rollback_admit(current, ctx)
        assert resource._arena.num_free == 2
        assert not resource._pending_initialization
        assert not resource._streams[1]["main"].step_in_flight
        resource.clear_preplan()
        resource.assert_pages_conserved()
    finally:
        resource.cleanup()
