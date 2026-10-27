"""Whisper's decoder caches hold the requests the deployment admits.

A request keeps its 1500-token encoder context (12 cross-attention pages) from
prefill to its last token. When more requests are in flight than the cache
holds, a prefill batch is refused and the worker offloads decoding requests to
admit it, over and over. The shipped configs cap ``max_concurrent_requests``
at what both explicitly sized caches hold, and the conductor queues the rest.
These tests drive the real conductor gate and
the real page allocator on the CPU; no weights are loaded.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, ".")

import pytest
import torch
import yaml

from mstar.conductor.conductor import Conductor
from mstar.engine.resources import AllocationFailed, apply_yaml_overrides
from mstar.engine.resources.binding import resolve_bound_specs
from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.kv.config import KVSpec, KVStep, PagedKVConfig
from mstar.engine.resources.kv.manager import KVManager
from mstar.engine.resources.sequence import HeadSharded, PagedSequenceView
from mstar.engine.resources.step import Segment, StepContext
from mstar.model.binding import resolve_autocast_dtype
from mstar.model.whisper.config import CONTEXT_LABEL, CROSS_KV_CACHE, KV_CACHE, WhisperModelConfig
from mstar.model.whisper.reference import WhisperDecoderReferenceSubmodule
from mstar.model.whisper.whisper_model import WhisperModel

CONFIGS = Path(__file__).resolve().parents[2] / "configs"
PROMPT_TOKENS = 4


class _StubTransfer:
    """No engine, no bytes moved."""

    def __init__(self, transfer_engine_info, kv_cache, **kwargs):
        del transfer_engine_info, kv_cache, kwargs

    def get_kv_transfer_info(self, **kwargs):
        del kwargs

    def owns_transfer_info(self, transfer_info, **kwargs):
        del kwargs
        return transfer_info == self.get_kv_transfer_info()

    def remove_request(self, request_id):
        del request_id

    def start_async_retrieve(self, **kwargs):
        del kwargs

    def cleanup(self):
        pass


@pytest.fixture(autouse=True)
def _stub_transfer(monkeypatch):
    monkeypatch.setattr(manager_mod, "KVTransferManager", _StubTransfer)


def _make_model() -> WhisperModel:
    model = object.__new__(WhisperModel)
    model.config = WhisperModelConfig()  # large-v3 dims
    return model


def _kv_specs(model: WhisperModel, config=None, *, requests=None) -> dict[str, KVSpec]:
    config = config or yaml.safe_load((CONFIGS / "whisper_large.yaml").read_text())
    specs = model.get_node_resources()
    apply_yaml_overrides(specs, config)
    decoder_group = next(group for group in config["node_groups"] if "decoder" in group["node_names"])
    group_size = decoder_group.get("tp_size", 1) * decoder_group.get("sp_size", 1)
    # The retained reference contributes this physical view at Engine binding.
    physical_requests = {s.resource_key: (PagedSequenceView(partition=HeadSharded()),)
                         for s in specs if isinstance(s, KVSpec)}
    physical_requests.update(requests or {})
    resolved = resolve_bound_specs(specs, physical_requests, dtype=resolve_autocast_dtype(model, config),
                                   group_sizes=dict.fromkeys(physical_requests, group_size))
    return {s.resource_key: s for s in resolved if isinstance(s, KVSpec)}


def _request_cap() -> int:
    return yaml.safe_load((CONFIGS / "whisper_large.yaml").read_text())["max_concurrent_requests"]


def _manager(config: PagedKVConfig, name: str) -> KVManager:
    """The pool a spec declares, page for page, with a token's storage shrunk
    to a few floats so it fits in a CPU test."""
    small = dataclasses.replace(
        config, num_layers=1, num_kv_heads=1, num_qo_heads=1, head_dim=4, cpu_offload_pages=0,
    )
    return KVManager(
        cfg=small, name=name, joint_comm_group=None, transfer_engine_info=None,
        device=torch.device("cpu"), dtype=torch.float32,
    )


def _conductor(max_concurrent_requests: int | None) -> Conductor:
    """Just the admission gate: ``_ingest_request`` and ``_try_admit_waiting``
    as they are, with dispatch reduced to recording the request in flight."""
    c = object.__new__(Conductor)
    c.requests = {}
    c.waiting_queue = []
    c.max_concurrent_requests = max_concurrent_requests
    c._early_abort_requests = set()
    c.enable_nvtx = False
    c._admit_request = lambda body: c.requests.__setitem__(body.request_id, body)
    return c


def _step(kv: KVManager, label: str, spans: dict[str, int], walk: str) -> bool:
    """One batched step over ``spans``; True if the cache admitted it."""
    step = KVStep(segments=tuple(Segment(rid, label, span) for rid, span in spans.items()))
    ctx = StepContext(request_ids=tuple(spans), graph_walk=walk, slot=0, capture=False)
    outcome = kv.admit(step, ctx)
    if not outcome.ok:
        assert isinstance(outcome.reason, AllocationFailed)
        return False
    kv.commit(step, ctx)
    return True


def _serve(model: WhisperModel, conductor: Conductor, arrivals: int) -> tuple[int, int]:
    """Every request arrives at once; the worker prefills the admitted ones in
    batches of the decoder's prefill cap, decodes each to the full 448 tokens,
    and finishes the oldest batch before the next round. Returns the refused
    prefill batches and the most requests ever in flight."""
    specs = _kv_specs(model)
    cross = _manager(specs[CROSS_KV_CACHE].config, CROSS_KV_CACHE)
    self_kv = _manager(specs[KV_CACHE].config, KV_CACHE)
    batch_size = WhisperDecoderReferenceSubmodule.MAX_BATCH_SIZE
    for i in range(arrivals):
        conductor._ingest_request(SimpleNamespace(
            request_id=f"r{i}", model_kwargs={}, client_request_id=None))

    running: list[str] = []
    refused = peak = finished = 0
    while finished < arrivals:
        peak = max(peak, len(conductor.requests))
        ready = [rid for rid in conductor.requests if rid not in running]
        for start in range(0, len(ready), batch_size):
            batch = ready[start:start + batch_size]
            for rid in batch:
                cross.ingest_request(rid)
                self_kv.ingest_request(rid)
            if not _step(cross, CONTEXT_LABEL, {rid: model.config.max_source_positions for rid in batch}, "prefill"):
                refused += 1
                for rid in batch:
                    cross.remove_request(rid)
                    self_kv.remove_request(rid)
                continue
            assert _step(self_kv, "main", dict.fromkeys(batch, PROMPT_TOKENS), "prefill")
            decode = model.config.max_target_positions - PROMPT_TOKENS
            assert _step(self_kv, "main", dict.fromkeys(batch, decode), "decode")
            running.extend(batch)
        if not running:
            break
        for rid in running[:batch_size]:
            cross.remove_request(rid)
            self_kv.remove_request(rid)
            del conductor.requests[rid]
            finished += 1
        del running[:batch_size]
        conductor._try_admit_waiting()
    return refused, peak


def test_default_caches_hold_the_request_cap():
    model = _make_model()
    specs = _kv_specs(model)
    cap = _request_cap()
    # 1500 encoder tokens and 448 decoder tokens in 128-token pages, plus the sink
    assert specs[CROSS_KV_CACHE].config.max_num_pages == 1 + 12 * cap
    assert specs[KV_CACHE].config.max_num_pages == 1 + 4 * cap


@pytest.mark.parametrize(("name", "group_size", "offload_pages"), [
    ("whisper_large", 1, 128), ("whisper_large_tp2", 2, 4096),
])
def test_shipped_yaml_caps_requests_at_what_the_caches_hold(name, group_size, offload_pages):
    path = CONFIGS / f"{name}.yaml"
    cfg = yaml.safe_load(path.read_text())
    model = _make_model()
    assert cfg["max_concurrent_requests"] == _request_cap()
    # Deployment supplies byte budgets; shared binding derives local storage.
    specs = _kv_specs(model, cfg)
    pages = {key: s.config.max_num_pages for key, s in specs.items()}
    assert pages == {
        key: s.config.max_num_pages for key, s in _kv_specs(model).items()
    }
    assert resolve_autocast_dtype(model, cfg) == torch.bfloat16
    for spec in specs.values():
        assert spec.config.partition == "head_sharded"
        assert spec.config.num_kv_heads == model.config.decoder_attention_heads // group_size
        assert spec.config.cpu_offload_pages == offload_pages
        assert spec.config.prefix_cache


def test_cache_budget_and_admission_cap_are_independent_deployment_choices():
    model = _make_model()
    cfg = yaml.safe_load((CONFIGS / "whisper_large.yaml").read_text())
    reference = _kv_specs(model, cfg)
    page_bytes = {key: s.budget.max_bytes // s.config.max_num_pages for key, s in reference.items()}
    cfg["max_concurrent_requests"] = 128
    cfg["resources"][KV_CACHE]["max_bytes"] = 64 * page_bytes[KV_CACHE] // 2
    cfg["resources"][CROSS_KV_CACHE]["max_bytes"] = 192 * page_bytes[CROSS_KV_CACHE]
    specs = _kv_specs(model, cfg, requests={KV_CACHE: (PagedSequenceView(64, HeadSharded()),)})
    physical = {key: s.config for key, s in specs.items()}
    assert physical[KV_CACHE].page_size == 64
    assert physical[KV_CACHE].max_num_pages == 64
    assert physical[CROSS_KV_CACHE].max_num_pages == 192


@pytest.mark.parametrize("cap", [0, -1, True, "64", 8.0])
def test_a_cap_that_is_not_a_positive_int_is_refused(cap, tmp_path):
    path = tmp_path / "bad-cap.yaml"
    config = yaml.safe_load((CONFIGS / "whisper_large.yaml").read_text())
    config.update(device_type="cpu", max_concurrent_requests=cap)
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="positive int"):
        Conductor(_make_model(), str(path))


def test_capped_arrivals_queue_instead_of_refusing_a_prefill():
    """Three times the cap arrive at once: the conductor holds the excess, and
    no prefill batch of the admitted ones is ever short of pages."""
    model = _make_model()
    cap = _request_cap()
    refused, peak = _serve(model, _conductor(cap), arrivals=3 * cap)
    assert refused == 0
    assert peak == cap


def test_uncapped_arrivals_refuse_prefills_once_the_cache_is_full():
    """The same arrivals with no cap: everything is dispatched, and the batches
    past the cache are refused, which is what sends the worker evicting."""
    model = _make_model()
    refused, peak = _serve(model, _conductor(None), arrivals=3 * _request_cap())
    assert refused > 0
    assert peak == 3 * _request_cap()


def test_old_context_cache_refused_a_full_prefill_batch_even_empty():
    """Why the old sizing collapsed from 16 in flight: 192 pages less the sink
    hold 15 contexts, and a full prefill batch is 16."""
    model = _make_model()
    config = dataclasses.replace(_kv_specs(model)[CROSS_KV_CACHE].config, max_num_pages=192)
    cross = _manager(config, CROSS_KV_CACHE)
    batch = [f"r{i}" for i in range(WhisperDecoderReferenceSubmodule.MAX_BATCH_SIZE)]
    for rid in batch:
        cross.ingest_request(rid)
    span = model.config.max_source_positions
    assert not _step(cross, CONTEXT_LABEL, dict.fromkeys(batch, span), "prefill")
    assert _step(cross, CONTEXT_LABEL, dict.fromkeys(batch[:-1], span), "prefill")
