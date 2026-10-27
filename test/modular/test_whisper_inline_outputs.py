"""Whisper adopts existing row-output and inline-client transport contracts."""

from __future__ import annotations

import queue
import threading
from types import SimpleNamespace

import pytest
import torch
from test_inline_tokens import _loop, _thread

from mstar.api_server.data_worker import InlineToken
from mstar.engine.engine import Engine, SubmoduleManagement
from mstar.engine.execution import LogicalInvocation
from mstar.model.whisper.config import SAMPLER, WhisperModelConfig
from mstar.model.whisper.results import DecoderResult
from mstar.model.whisper.submodules import WhisperDecoderSubmodule
from mstar.model.whisper.whisper_model import WhisperModel
from mstar.worker.worker import Worker


def _collected(tokens, rids):
    sub = WhisperDecoderSubmodule(WhisperModelConfig())
    invocation = LogicalInvocation("decode", tuple(rids), (), None)
    logical = sub.execution_result(DecoderResult(tokens), invocation)
    engine = Engine.__new__(Engine)
    collected = engine._collect_outputs(
        SubmoduleManagement(sub, sub.forward, sub.forward_batched, None, {}),
        None,
        logical.outputs,
        [],
        {},
        request_ids=list(rids),
        step_request_ids=tuple(rids),
    )
    worker = Worker.__new__(Worker)
    worker.device_capability = SimpleNamespace(is_available=False)
    cpu_outputs, host_rows = worker._prematerialize_for_check_stop(
        collected,
        None,
        request_ids=list(rids),
    )
    return sub, logical, collected, worker, cpu_outputs, host_rows


@pytest.mark.parametrize("rids,values", [([7], [42]), ([9, 3, 8], [50257, 12, 81])])
def test_real_engine_collect_owns_rows_in_forward_order_and_reuses_stop_clone(rids, values):
    tokens = torch.tensor(values, dtype=torch.int64)
    sub, logical, collected, worker, cpu_outputs, host_rows = _collected(tokens, rids)
    assert logical.outputs.row_outputs["new_token"] is tokens
    assert logical.outputs.check_stop_buffers["new_token"] is tokens
    assert logical.resource_results[SAMPLER].tokens is tokens
    assert collected.row_request_ids == tuple(rids)
    assert host_rows.request_ids == tuple(rids)
    stop = collected.check_stop_buffers["new_token"]
    assert stop.dtype == torch.int64 and stop.tolist() == values
    assert stop.data_ptr() != tokens.data_ptr()
    for i, rid in enumerate(rids):
        out = collected.get(rid)
        assert out["new_token"][0].data_ptr() == stop[i : i + 1].data_ptr()
        assert cpu_outputs[rid]["new_token"][0].item() == values[i]
        sub.postprocess(rid, None, out)
        assert out["text_inputs"] is out["new_token"]
    # The next primitive may reuse/overwrite its source. Published row clones
    # and the local loop route remain owned independently of that source.
    tokens.fill_(99)
    assert stop.tolist() == values
    assert [collected.get(r)["text_inputs"][0].item() for r in rids] == values


@pytest.mark.parametrize("retained", [[7], [9, 7], [8, 9, 7], []])
def test_inline_uses_only_retained_rows_and_maps_their_actual_forward_ids(retained):
    sub, _, collected, worker, _, host_rows = _collected(torch.tensor([70, 80, 90]), [7, 8, 9])
    worker._graph_runtime = SimpleNamespace(supports_inline_emit=True)
    engine = SimpleNamespace(inline_client_signals=lambda _node, walk: sub.inline_client_signals(walk))
    batch = SimpleNamespace(node_name="decoder", graph_walk="decode", batch=SimpleNamespace(request_walks={}))
    # Pending/stale speculative rows are removed before inline extraction;
    # their row buffers remain, so mapping must use engine-stamped identities.
    for rid in set([7, 8, 9]) - set(retained):
        collected.pop(rid)
    signal, values = worker._inline_emit_values(engine, batch, host_rows, retained, ["new_token", "text_inputs"])
    assert signal == "new_token"
    assert values == [{7: 70, 8: 80, 9: 90}[rid] for rid in retained]
    worker._graph_runtime = SimpleNamespace(supports_inline_emit=False)
    assert worker._inline_emit_values(engine, batch, host_rows, retained, ["new_token", "text_inputs"]) == (None, [])


def test_prefill_persistence_and_local_decode_tensor_edge_are_unchanged():
    sub = WhisperDecoderSubmodule(WhisperModelConfig())
    model = object.__new__(WhisperModel)
    model.config = sub.config
    graphs = model.get_graph_walk_graphs()
    prefill = graphs["prefill"].get_nodes()["decoder"]
    decode = graphs["decode"].get_nodes()["decoder"]
    assert [(e.name, e.persist) for e in prefill.outputs] == [("new_token", True)]
    assert [(e.name, e.persist) for e in decode.outputs] == [("new_token", False), ("text_inputs", False)]
    assert sub.inline_client_signals("prefill") == {}
    assert sub.inline_client_signals("decode") == {"new_token": "new_token"}
    assert sub.device_loopback_signals("decode") == frozenset()


@pytest.mark.parametrize(
    "hit_eos,ignore_eos,iteration,limit,stopped",
    [
        (True, False, 0, 8, True),
        (True, True, 0, 8, False),
        (False, False, 6, 8, False),
        (False, False, 7, 8, True),
    ],
)
def test_cpu_row_stop_input_preserves_eos_budget_and_ignore_eos(hit_eos, ignore_eos, iteration, limit, stopped):
    config = WhisperModelConfig()
    value = config.eos_token_id if hit_eos else 42
    sub, _, _, _, cpu_outputs, _ = _collected(torch.tensor([value]), [3])
    info = SimpleNamespace(
        resource_configs={SAMPLER: SimpleNamespace(ignore_eos=ignore_eos)},
        dynamic_loop_iter_counts={"decode_loop": iteration},
        max_tokens=limit,
    )
    assert sub.check_stop(3, info, cpu_outputs[3]) == ({"decode_loop"} if stopped else set())


def test_inline_client_backpressure_retains_order_without_reading_tensor_storage():
    worker, stop = _thread()
    entered = threading.Event()

    class Bounded(queue.Queue):
        def put(self, item, *args, **kwargs):
            entered.set()
            return super().put(item, *args, **kwargs)

    worker.out_queue = Bounded(maxsize=1)
    queue.Queue.put(worker.out_queue, "held")
    for i, value in enumerate([11, 12, 13]):
        worker.result_tensor_queue.put(InlineToken("req", value, _loop(i), "text"))
    thread = threading.Thread(target=worker.run)
    thread.start()
    try:
        assert entered.wait(5), "real data worker reached bounded client output"
        assert worker.out_queue.get(timeout=5) == "held"
        chunks = [worker.out_queue.get(timeout=5) for _ in range(3)]
        assert [c.data for c in chunks] == [b"<11>", b"<12>", b"<13>"]
        assert [c.request_id for c in chunks] == ["req"] * 3
    finally:
        stop.set()
        thread.join(timeout=5)
    assert not thread.is_alive()
