"""Encoding, geometry and plan-location requirements compose independently."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from mstar.engine.resources.binding import resolve_bound_specs
from mstar.engine.resources.kv.config import KVSpec, SequenceKVConfig
from mstar.engine.resources.sequence import (
    DenseSequenceView, HeadSharded, PagedSequenceView, PlainKVEncoding,
    SequenceBudget, SequenceEncoding, build_sequence_resource, resolve_sequence_config,
)
from mstar.engine.resources.spec import ResourceViewConflict


def _spec():
    return KVSpec(resource_key="kv", nodes={"node"}, config=SequenceKVConfig(
        num_layers=2, num_kv_heads=4, head_dim=8, max_seq_len=16,
    ), budget=SequenceBudget(max_bytes=16384))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("device_plan", [False, True])
def test_device_plan_preserves_inherited_encoding_geometry_and_capacity(dtype, device_plan):
    spec = _spec()
    before = deepcopy(vars(spec.config))
    requests = (PagedSequenceView(4, partition=HeadSharded(), device_plan=device_plan),)
    physical, = resolve_bound_specs([spec], {"kv": requests}, dtype=dtype, group_sizes={"kv": 2})
    config = physical.config
    assert config.encoding == config.inherited_encoding == PlainKVEncoding(dtype)
    assert config.num_kv_heads == 2 and config.page_size == 4
    assert config.max_num_pages * config.page_bytes == spec.budget.max_bytes
    assert vars(spec.config) == before and spec.budget is physical.budget


def test_explicit_half_encoding_preserves_capacity_and_omitted_consumer_constraint():
    half = PlainKVEncoding(torch.float16)
    full = resolve_sequence_config(_spec(), (DenseSequenceView(),), dtype=torch.float32)
    # Positional encoding is part of the Pi view API; device_plan is orthogonal.
    request = PagedSequenceView(16, HeadSharded(), half, device_plan=True)
    selected = resolve_sequence_config(_spec(), (request,), dtype=torch.float32)
    assert selected.encoding == half
    assert selected.page_bytes * 2 == full.page_bytes
    assert selected.max_num_pages == full.max_num_pages * 2
    with pytest.raises(ResourceViewConflict, match="incompatible sequence encodings"):
        resolve_sequence_config(_spec(), (request, SequenceEncoding()), dtype=torch.float32)


@pytest.mark.parametrize("host_plan", [False, True])
def test_resource_construction_keeps_selected_storage_dtype_and_plan_location(monkeypatch, host_plan):
    from mstar.engine.resources.kv import manager

    seen = []
    monkeypatch.setattr(manager, "KVManager", lambda *args, **kwargs: seen.append((args, kwargs)))
    request = DenseSequenceView(encoding=PlainKVEncoding(torch.float16))
    info = SimpleNamespace(joint_comm_group=None, kv_dtype=torch.float32,
                           needs_remote_transfer=False, transfer_engine_info=None,
                           device=torch.device("cpu"))
    build_sequence_resource(_spec(), info, (request,), host_plan=host_plan)
    args, kwargs = seen.pop()
    assert args[5] is torch.float16
    assert kwargs["host_plan"] is host_plan
    assert args[0].encoding == PlainKVEncoding(torch.float16)
