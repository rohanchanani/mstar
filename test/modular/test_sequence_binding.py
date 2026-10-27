"""Logical sequence binding and the shared owner of dense/paged views."""
from types import SimpleNamespace

import pytest
import torch

from mstar.engine.resources.attn.config import AttentionStep
from mstar.engine.resources.attn.planned import PlannedAttentionResource
from mstar.engine.resources.base import EngineResourceInfo
from mstar.engine.resources.binding import build_bound_resource, merge_resource_requests, resolve_bound_specs
from mstar.engine.resources.kv.cache import KVCache
from mstar.engine.resources.kv.config import KVSpec, KVStep, PagedKVConfig, SequenceKVConfig
from mstar.engine.resources.kv.encoding import KVEncoding
from mstar.engine.resources.sampler.config import SamplerSpec, SamplingReqConfig, sampling_policy_facts
from mstar.engine.resources.sampler.resource import SamplerResource
from mstar.engine.resources.sequence import (
    DenseSequenceView,
    HeadSharded,
    PagedSequenceView,
    PlainKVEncoding,
    Replicated,
    SequenceBudget,
    SequenceEncoding,
    resolve_sequence_config,
)
from mstar.engine.resources.spec import ResourceViewConflict, apply_yaml_overrides
from mstar.model.binding import NodeAssembly
from mstar.engine.resources.step import AdmitRuntimeError, Segment, StepContext

CPU = torch.device("cpu")


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


def owner(*, pages=3, encoding=None):
    logical = spec(pages=pages, storage_dtype=torch.float32 if encoding is None else encoding.dtype)
    requests = (DenseSequenceView(encoding=encoding), PagedSequenceView(encoding=encoding))
    physical, = resolve_bound_specs((logical,), {"kv": requests}, dtype=torch.float32)
    return build_bound_resource(physical, EngineResourceInfo(
        CPU, kv_dtype=torch.float32, needs_remote_transfer=False,
    ), requests)


def test_encoding_controls_capacity_allocation_and_views_independently_of_binding_dtype():
    logical = spec()
    half = PlainKVEncoding(torch.float16)
    requests = (DenseSequenceView(encoding=half), PagedSequenceView(encoding=half))
    physical, = resolve_bound_specs((logical,), {"kv": requests}, dtype=torch.float32)
    full = resolve_sequence_config(logical, (DenseSequenceView(),), dtype=torch.float32)
    assert physical.config.encoding == half
    assert physical.config.inherited_encoding == PlainKVEncoding(torch.float32)
    assert physical.config.page_bytes * 2 == full.page_bytes
    assert physical.config.max_num_pages == 2 * full.max_num_pages == 6
    resource = build_bound_resource(physical, EngineResourceInfo(
        CPU, kv_dtype=torch.float64, needs_remote_transfer=False,
    ), requests)
    try:
        assert resource.kv_cache.dtype is torch.float16
        assert resource.kv_cache.nbytes == physical.config.max_num_pages * physical.config.page_bytes
        assert resource.kv_cache.nbytes == logical.budget.max_bytes
        assert resource.view(requests[0]).encoding == resource.view(requests[1]).encoding == half
        assert resource.view(requests[1]).layer(0).data_ptr() == resource.kv_cache.tensor[0].data_ptr()
        with pytest.raises(ValueError, match="encoding contradicts"):
            resource.view(DenseSequenceView())
        with pytest.raises(ValueError, match="encoding contradicts"):
            resource.view(PagedSequenceView(encoding=PlainKVEncoding(torch.float32)))
    finally:
        resource.cleanup()


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16, torch.float64, None])
def test_omitted_encoding_preserves_inherited_legacy_dtype(dtype):
    expected = torch.get_default_dtype() if dtype is None else dtype
    config = resolve_sequence_config(spec(storage_dtype=expected), (DenseSequenceView(),), dtype=dtype)
    assert config.encoding == config.inherited_encoding == PlainKVEncoding(expected)
    cache = KVCache(config, CPU, torch.float32)
    assert cache.dtype is expected
    # Physical declarations outside the new logical resolver retain their old dtype path.
    legacy = PagedKVConfig(num_layers=1, num_kv_heads=1, head_dim=2, max_seq_len=4,
                           max_num_pages=2, page_size=4)
    assert KVCache(legacy, CPU, dtype).dtype is expected


def test_encoding_conflicts_and_unknown_formats_fail_during_resolution():
    half = PlainKVEncoding(torch.float16)
    for inherited in (DenseSequenceView(), SequenceEncoding()):
        with pytest.raises(ResourceViewConflict, match="incompatible sequence encodings"):
            resolve_sequence_config(spec(), (DenseSequenceView(encoding=half), inherited), dtype=torch.float32)
    # An omitted consumer is neutral about geometry/partition, but not encoding.
    config = resolve_sequence_config(spec(heads=4), (
        PagedSequenceView(2, HeadSharded(), PlainKVEncoding(torch.float32)), SequenceEncoding(),
    ), dtype=torch.float32, group_size=2)
    assert config.page_size == 2 and config.num_kv_heads == 2 and config.partition == "head_sharded"
    for encoding in (KVEncoding(), PlainKVEncoding(torch.bfloat16)):
        with pytest.raises(TypeError, match="unsupported explicit sequence encoding"):
            resolve_sequence_config(spec(), (DenseSequenceView(encoding=encoding),), dtype=torch.float32)


def test_retained_reference_encoding_conflicts_before_allocation():
    from mstar.engine.engine import Engine

    logical = spec(heads=4)
    logical.nodes = {"reference", "bound"}
    engine = Engine(None)
    groups = SimpleNamespace(get_joint_group_for_node=lambda _: SimpleNamespace(world_size=2))
    with pytest.raises(ResourceViewConflict, match="incompatible sequence encodings"):
        engine.load_model(
            {"reference": None, "bound": None}, [logical], groups, CPU, None,
            kv_cache_type=torch.float32, assemblies={"bound": NodeAssembly(bind=lambda resources: None)},
            resource_requests={"kv": (PagedSequenceView(
                partition=HeadSharded(), encoding=PlainKVEncoding(torch.float16),
            ),)},
        )
    assert not engine._resources


def test_unnegotiated_remote_encoding_override_fails_before_storage_allocation(monkeypatch):
    from mstar.engine.resources.kv import manager

    request = PagedSequenceView(2, encoding=PlainKVEncoding(torch.float16))
    physical, = resolve_bound_specs((spec(),), {"kv": (request,)}, dtype=torch.float32)
    monkeypatch.setattr(manager, "KVManager", lambda *args, **kwargs: pytest.fail("allocated storage"))
    with pytest.raises(ResourceViewConflict, match="remote transfer has no encoding agreement"):
        build_bound_resource(physical, EngineResourceInfo(CPU, kv_dtype=torch.float32,
                                                        needs_remote_transfer=True), (request,))


def test_half_owner_two_live_refusal_reassignment_preserves_actual_survivor_bytes():
    half = PlainKVEncoding(torch.float16)
    resource = owner(encoding=half)
    view = resource.view(DenseSequenceView(encoding=half))
    assert resource.config.max_num_pages == 3
    rows = {}
    try:
        for rid in (1, 2):
            resource.ingest_request(rid)
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
        resource.ingest_request(3)
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


def test_deployment_budget_preserves_geometry_and_views_choose_storage():
    logical = spec()
    geometry = logical.config
    apply_yaml_overrides((logical,), {"resources": {"kv": {"max_bytes": 512}}})
    assert logical.config is geometry
    paged, = resolve_bound_specs((logical,), {"kv": (PagedSequenceView(2),)}, dtype=torch.float32)
    dense, = resolve_bound_specs((logical,), {"kv": (DenseSequenceView(), PagedSequenceView())}, dtype=torch.float32)
    assert paged.config.page_size == 2
    assert dense.config.page_size == geometry.max_seq_len
    assert paged.config.max_num_pages * 2 == dense.config.max_num_pages * 4
    assert logical.config is geometry and not hasattr(geometry, "page_size")
    with pytest.raises(ValueError, match="incompatible sequence block sizes"):
        resolve_sequence_config(logical, (DenseSequenceView(), PagedSequenceView(2)), dtype=torch.float32)
    with pytest.raises(ValueError, match="incompatible sequence partitions"):
        resolve_sequence_config(logical, (PagedSequenceView(partition=Replicated()),
                                          PagedSequenceView(partition=HeadSharded())), dtype=torch.float32)


def test_resolved_replicated_and_head_sharded_geometry_is_not_sharded_again():
    logical = spec(heads=4)
    for partition, expected in ((Replicated(), 4), (HeadSharded(), 2)):
        request = PagedSequenceView(2, partition=partition)
        physical, = resolve_bound_specs((logical,), {"kv": (request,)}, dtype=torch.float32,
                                       group_sizes={"kv": 2})
        physical.config.shard = lambda _: pytest.fail("resolved geometry was sharded twice")
        resource = build_bound_resource(physical, EngineResourceInfo(
            CPU, joint_comm_group=SimpleNamespace(rank=0, world_size=2),
            kv_dtype=torch.float32, needs_remote_transfer=False,
        ), (request,))
        assert resource.config.num_kv_heads == expected
        resource.cleanup()
    assert logical.config.num_kv_heads == 4


def test_engine_collects_retained_reference_and_bound_consumer_views_before_allocation():
    from mstar.engine.engine import Engine

    logical = spec(heads=4)
    logical.nodes = {"reference", "bound"}
    engine = Engine(None)
    groups = SimpleNamespace(get_joint_group_for_node=lambda _: SimpleNamespace(world_size=2))
    with pytest.raises(ResourceViewConflict, match="incompatible sequence partitions"):
        engine.load_model(
            {"reference": None, "bound": None}, [logical], groups, CPU, None,
            kv_cache_type=torch.float32, assemblies={"bound": NodeAssembly(bind=lambda resources: None)},
            resource_requests={"kv": (DenseSequenceView(partition=Replicated()),)},
        )
    assert not engine._resources
    assert logical.config.num_kv_heads == 4


def test_nonsequence_request_union_is_checked_before_building_resources():
    logical = SamplerSpec(resource_key="sampler", nodes={"decoder"}, vocab_size=32)
    first, other = SimpleNamespace(layout="first"), SimpleNamespace(layout="other")
    requests = merge_resource_requests(
        {"sampler": (first,)}, {"sampler": (SimpleNamespace(layout="first"), other)},
    )
    assert requests == {"sampler": (first, other)}
    with pytest.raises(ResourceViewConflict, match="sampler: incompatible physical resource requests"):
        resolve_bound_specs((logical,), requests, dtype=torch.float32)


def test_dense_and_paged_views_alias_one_state_with_ordered_initialization():
    resource = owner(pages=2)
    resource.kv_cache.tensor.fill_(float("nan"))
    resource.ingest_request(1)
    initial, ctx = step(2), context()
    assert resource.admit(initial, ctx).ok
    assert torch.isnan(resource.kv_cache.tensor).all(), "admission must enqueue no storage writes"
    plans = resource.plan(initial, ctx)
    ctx.plan_results["kv"] = plans
    use = resource.logical_use(plans)["main"]
    dense, paged = resource.view(DenseSequenceView()), resource.view(PagedSequenceView())
    row, = dense.for_step(use)
    assert row.positions.tolist() == [0, 1]
    assert row.keys[0].shape == (4, 1, 2)
    assert all(torch.count_nonzero(tensor) == 0 for tensor in (*row.keys, *row.values))
    keys = torch.tensor([[[1., 2.]], [[3., 4.]]])
    for layer in range(2):
        row.keys[layer][row.positions] = keys + layer * 10
        row.values[layer][row.positions] = keys + layer * 20
        torch.testing.assert_close(resource.read_kv(layer, "main")[:, 0], keys + layer * 10)
    physical = paged.for_step(use)
    block = physical.views[0].page_idxs[0]
    assert paged.layer(0)[block, 0].data_ptr() == row.keys[0].data_ptr()
    attention = PlannedAttentionResource("kv", resource.config.page_size)
    logical_attention = attention.logical_use(attention.plan(AttentionStep(), ctx))["main"]
    read, = dense.for_attention(logical_attention)
    assert read.valid_length == 2
    torch.testing.assert_close(read.mask, torch.tensor([[0., -torch.inf, -torch.inf, -torch.inf],
                                                     [0., 0., -torch.inf, -torch.inf]]))
    # Dense full-capacity attention and the active paged prefix produce the same result.
    query = torch.tensor([[1., 0.], [0., 1.]])
    logits = query @ read.keys[0][:, 0].T
    dense_output = (logits + read.mask).softmax(-1) @ read.values[0][:, 0]
    active_mask = read.mask[:, :2]
    paged_output = (query @ keys[:, 0].T + active_mask).softmax(-1) @ keys[:, 0]
    torch.testing.assert_close(dense_output, paged_output)
    resource.commit(initial, ctx)

    append = step(1)
    assert resource.admit(append, ctx).ok
    resource.plan(append, ctx)
    torch.testing.assert_close(row.keys[0][:2], keys)
    resource.commit(append, ctx)
    too_long = resource.admit(step(2), ctx)
    assert not too_long.ok and isinstance(too_long.reason, AdmitRuntimeError)
    assert len(resource._streams[1]["main"].page_indices) == 1
    resource.rollback_admit(step(2), ctx)

    # A rewind reuses the same allocation but must reset its unread tail.
    resource.reset_request(1)
    assert resource.admit(step(1), ctx).ok
    resource.plan(step(1), ctx)
    assert torch.count_nonzero(row.keys[0]) == 0
    resource.commit(step(1), ctx)
    row.keys[0].fill_(float("nan"))
    resource.remove_request(1)
    resource.ingest_request(2)
    next_step, next_ctx = step(1, 2), context(2)
    assert resource.admit(next_step, next_ctx).ok
    assert torch.isnan(row.keys[0]).all()
    resource.plan(next_step, next_ctx)
    assert torch.count_nonzero(row.keys[0]) == 0
    resource.commit(next_step, next_ctx)
    resource.remove_request(2)
    resource.assert_pages_conserved()
    resource.cleanup()




def test_sampler_policy_facts_match_resolved_owner_and_are_detached():
    owner = SamplerResource(32, False, CPU, planned=True, enable_min_p=True)
    configs = (SamplingReqConfig(temperature=1e-8, top_k=7, top_p=.5, min_p=.2),
               SamplingReqConfig(temperature=.7, top_k=4, top_p=.9))
    for rid, config in enumerate(configs):
        owner.ingest_request(rid, config)
    before = tuple(sampling_policy_facts(config) for config in configs)
    assert owner.policy_facts((0, 1)) == before
    assert before[0]["temperature"] == 0 and before[0]["top_k"] == 0
    assert owner._sampler._step_offset == {0: 0, 1: 0}
    before[0]["temperature"] = 99
    assert owner.policy_facts((0,))[0]["temperature"] == 0
    assert not any("seed" in key or "offset" in key for key in before[1])
    for rid in (0, 1):
        owner.remove_request(rid)


@pytest.mark.parametrize("causal", [False, True])
def test_dense_masks_match_visibility_and_keep_independent_storage(causal):
    from mstar.engine.resources.sequence_view import AttentionUse

    resource = owner()
    dense = resource.view(DenseSequenceView())
    retained = []
    try:
        for length in range(5):
            for query_length in range(6):
                view = SimpleNamespace(page_idxs=[0], start=0, length=length)
                use = AttentionUse((1,), (query_length,), (length,), causal,
                                   SimpleNamespace(contexts=(view,)))
                if causal and query_length > length:
                    with pytest.raises(ValueError, match="causal queries exceed"):
                        dense.for_attention(use)
                    continue
                row, = dense.for_attention(use)
                expected = torch.tensor([
                    [0. if j < length and (not causal or j <= length - query_length + i)
                     else -torch.inf for j in range(4)]
                    for i in range(query_length)
                ], dtype=torch.float32).reshape(query_length, 4)
                assert torch.equal(row.mask, expected)
                assert row.mask.dtype == torch.float32
                assert row.mask.is_contiguous()
                retained.append((row.mask, expected))
        for mask, expected in retained:
            assert torch.equal(mask, expected)
        nonempty = [mask for mask, _ in retained if mask.numel()]
        assert len({mask.data_ptr() for mask in nonempty}) == len(nonempty)
    finally:
        resource.cleanup()


@pytest.mark.parametrize("requests,host_plan", [
    ((PagedSequenceView(4),), True),
    ((PagedSequenceView(4, device_plan=True),), False),
    ((PagedSequenceView(4), PagedSequenceView(4, device_plan=True)), False),
    ((DenseSequenceView(device_plan=True),), False),
])
def test_sequence_plan_placement_is_an_explicit_resource_requirement(requests, host_plan):
    logical = spec()
    owner = build_bound_resource(
        logical, EngineResourceInfo(CPU, kv_dtype=torch.float32, needs_remote_transfer=False), requests,
    )
    try:
        assert owner._host_plan is host_plan
        assert isinstance(logical.config, SequenceKVConfig)
    finally:
        owner.cleanup()
