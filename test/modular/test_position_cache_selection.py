"""Position-resource precision policy and host-plan lifetime, without GPU kernels."""
import math
from types import SimpleNamespace

import pytest
import torch

from mstar.engine.resources.kv.plan import KVPlanOutputs
from mstar.engine.resources.position import manager as position_manager
from mstar.engine.resources.position.config import PositionConfig, PositionStep
from mstar.engine.resources.position.manager import RopeManager
from mstar.engine.resources.step import Segment, StepContext


@pytest.fixture
def amd_manager(monkeypatch):
    # Exercise the actual CUDA/ROCm selection while allocating tables on CPU.
    monkeypatch.setattr(position_manager, "current_platform", SimpleNamespace(is_rocm=lambda: True))
    for name in ("_build_rope_cache", "_build_rocm_rope_cache"):
        build = getattr(RopeManager, name)

        def cpu_build(self, *args, _build=build, **kwargs):
            owner = SimpleNamespace(_max_seq_len=self._max_seq_len, _device=torch.device("cpu"))
            return _build(owner, *args, **kwargs)

        monkeypatch.setattr(RopeManager, name, cpu_build)

    def make(**kwargs):
        config = dict(kv_cache="kv", rotary_dim=8, rope_theta=1e6)
        config.update(kwargs)
        owner = RopeManager(PositionConfig(**config), torch.device("cuda"), max_seq_len=513, head_dim=8)
        owner._current_pos_ids["main"] = torch.arange(2)
        return owner

    return make


@pytest.fixture
def calls(monkeypatch):
    seen = []
    monkeypatch.setattr(torch.ops.mstar, "rope_apply_qk_inplace", lambda *args, **kwargs: seen.append(args))
    return seen


def apply(owner, **kwargs):
    q = torch.ones(2, 1, 8, dtype=torch.bfloat16)
    owner.apply_qk(q, q.clone(), "main", **kwargs)


@pytest.mark.parametrize("scale,llama", [(1., None), (4., None), (8., (1., 4., 8192))])
def test_amd_default_cache_uses_double_angles(amd_manager, scale, llama):
    config = {} if llama is None else dict(zip(
        ("low_freq_factor", "high_freq_factor", "old_context_len"), llama, strict=True,
    ))
    owner = amd_manager(rope_scale=scale, **config)
    pos = torch.arange(513, dtype=torch.float64)
    inv = 1.0 / (1e6 ** (torch.arange(0, 8, 2, dtype=torch.float64) / 8))
    if llama is None:
        pos = pos / scale
    else:
        low, high, old = llama
        smooth = (inv * (old / (2 * math.pi * (high - low))) - low / (high - low)).clamp(0, 1)
        inv = (1 - smooth) * inv / scale + smooth * inv
    angles = torch.outer(pos, inv)
    assert owner._rope_table is None
    assert torch.equal(owner._rope_cache, torch.cat((angles.cos(), angles.sin()), dim=-1).float())


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_explicit_table_takes_precedence_on_amd(amd_manager, calls, dtype):
    owner = amd_manager(rope_table_dtype=dtype, rope_scale=3.5)
    expected = RopeManager(PositionConfig("kv", rotary_dim=8, rope_theta=1e6,
                                         rope_scale=3.5, rope_table_dtype=dtype),
                           torch.device("cpu"), max_seq_len=513, head_dim=8)
    assert owner._rope_cache is None
    assert torch.equal(owner._rope_table, expected._rope_table)
    apply(owner)
    assert calls[-1][3] is owner._rope_table
    with pytest.raises(RuntimeError, match="do not match"):
        apply(owner, rope_theta=10000.)
    assert len(calls) == 1


@pytest.mark.parametrize("override", [dict(rope_theta=10000.), dict(rope_scale=2.), dict(rotary_dim=4),
                                    dict(rope_scale=8., low_freq_factor=1., high_freq_factor=4., old_context_len=8192)])
def test_default_cache_falls_back_for_other_angles(amd_manager, calls, override):
    owner = amd_manager()
    apply(owner)
    assert calls[-1][3] is owner._rope_cache
    apply(owner, low_freq_factor=1., high_freq_factor=1., old_context_len=8192)
    assert calls[-1][3] is owner._rope_cache  # scale-one blend is the identity
    apply(owner, **override)
    assert calls[-1][3] is None


def test_host_positions_keep_invocation_storage_and_counter_ownership():
    owner = RopeManager(PositionConfig("kv"), torch.device("cpu"), host_plan=True)
    assert not owner.can_capture() and not owner.supports_preplan and not owner.force_double_buffer
    for rid in (10, 20):
        owner.ingest_request(rid)

    def plan(rows):
        ctx = StepContext(request_ids=tuple(row[0] for row in rows), graph_walk="decode", slot=0, capture=False)
        ctx.plan_results["kv"] = KVPlanOutputs({"main": SimpleNamespace(rows=rows)})
        step = PositionStep(segments=tuple(Segment(*row) for row in rows))
        result = owner.plan(step, ctx)
        owner.commit(step, ctx)
        return result

    first = plan([(10, "main", 3), (20, "main", 2)])
    second = plan([(20, "main", 1), (10, "main", 1)])
    assert first["main"].tolist() == [0, 1, 2, 0, 1]
    assert second["main"].tolist() == [2, 3]
    assert owner.position(10, "main") == 4 and owner.position(20, "main") == 3
