"""Proposal refinement uses real row walks and preserves interpreted inputs."""
from collections import Counter
from types import SimpleNamespace

import pytest
import torch

from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.engine import Engine, PreparedInputRow, SubmoduleManagement
from mstar.engine.execution import ExecutionCapabilities, ExecutionOption, Support
from mstar.graph.runtime.base import ColumnarEdgeSpecs
from mstar.model.submodule_base import NodeInputs, NodeSubmodule
from mstar.worker.batch_builder import BatchBuildRequest, FIFOBatchBuilder
from mstar.worker.micro_scheduler import MicroScheduler, ScheduledBatch
from mstar.worker.worker import Worker


class InterpretingShell(NodeSubmodule):
    def __init__(self):
        super().__init__()
        self.interpreted = Counter()

    def prepare_inputs(self, graph_walk, fwd_info, inputs, **kwargs):
        self.interpreted[fwd_info.rid_handle] += 1
        value = inputs["x"][0]
        return NodeInputs(tensor_inputs={"x": value}, input_seq_len=value.numel())

    def forward(self, *args, **kwargs):
        raise AssertionError("refinement must not execute")


def _fixture(*, singleton):
    shell = InterpretingShell()

    def supports(facts):
        if facts.graph_walk != "decode" or facts.batch_size not in (1, 2):
            return Support.NO
        if facts.batch_size == 1:
            return Support.YES if singleton else Support.NO
        if facts.total_tokens is None:
            return Support.UNKNOWN
        return Support.YES if facts.total_tokens <= 10 else Support.NO

    def no_effect(*args):
        raise AssertionError("proposal selection must precede execution effects")

    shell.execution_capabilities = ExecutionCapabilities((ExecutionOption(
        "bounded", supports, no_effect, no_effect, no_effect,
    ),), {"decode": 2})
    engine = Engine(graph_runtime=None)
    engine._loopback_steps = {}
    engine._submodules["node"] = SubmoduleManagement(shell, None, None, None, {})
    worker = Worker.__new__(Worker)
    worker._stream_buffer_rids = set()
    worker._graph_runtime = SimpleNamespace(set_in_flight=lambda *args: None)
    worker.engine_manager = SimpleNamespace(get_engine=lambda _: engine)
    worker.scheduler = MicroScheduler(worker.engine_manager)
    worker.scheduler.runtime = worker._graph_runtime
    infos = {
        rid: CurrentForwardPassInfo(f"r{rid}", "decode", 0, 0, 32, rid_handle=rid)
        for rid in (7, 8, 9)
    }
    worker.request_state = SimpleNamespace(
        get_partition_for_node=lambda _: "main",
        get_fwd_info=lambda rid, _: infos[rid],
    )
    worker._all_on_device = lambda *args: False
    reads = Counter()
    tensors = {7: torch.zeros(6), 8: torch.zeros(6), 9: torch.zeros(4)}

    def read(rid):
        reads[rid] += 1
        return tensors[rid]

    worker.tensor_manager = SimpleNamespace(get_tensor=read)
    return worker, engine, shell, reads


def _scheduled(rids):
    edges = ColumnarEdgeSpecs.empty()
    for rid in rids:
        edges.add(rid, "x", [rid], False)
    return ScheduledBatch("node", "decode", dict.fromkeys(rids, 0), edges)


@pytest.mark.parametrize("singleton", [False, True])
def test_late_infeasible_proposal_keeps_rows_without_reinterpreting(singleton):
    worker, engine, shell, reads = _fixture(singleton=singleton)
    batch = _scheduled((7, 8))
    executing = worker._build_executing_batch(batch)
    engine.prepare_inputs(executing)
    worker._refine_prepared_batch(batch, executing, engine)

    assert executing.admit_error is None
    assert executing.request_ids == ([7] if singleton else [])
    assert set(executing.deferred_rows) == ({8} if singleton else {7, 8})
    pending = SimpleNamespace(batch=batch, node_batch=executing,
                              node_name="node", graph_walk="decode")
    worker._park_deferred_rows(pending)
    waiting = worker.scheduler.backlog.pop(("node", "decode"))
    caps = shell.execution_capabilities

    if not singleton:
        rejected = FIFOBatchBuilder().build_batch(BatchBuildRequest(
            "node", "decode", backlog=waiting, execution_capabilities=caps,
        ))
        assert rejected.scheduled is None  # no repeated UNKNOWN submission
        waiting = rejected.backlog

    composed = FIFOBatchBuilder().build_batch(BatchBuildRequest(
        "node", "decode", backlog=waiting, fresh=_scheduled((9,)),
        execution_capabilities=caps,
    ))
    assert list(composed.scheduled.request_to_worker_graph) == ([8, 9] if singleton else [7, 9])
    retry = worker._build_executing_batch(composed.scheduled)
    engine.prepare_inputs(retry)
    worker._refine_prepared_batch(composed.scheduled, retry, engine)
    assert not retry.deferred_rows
    assert retry.inputs[0].input_seq_len == 6 and retry.inputs[1].input_seq_len == 4
    assert shell.interpreted == reads == {7: 1, 8: 1, 9: 1}

    if composed.backlog is not None:
        worker.scheduler.defer(composed.backlog)
        worker.scheduler.clear_rid(8, "r8")
        assert not worker.scheduler.backlog  # cancellation releases retained holds


def test_combined_queue_uses_actual_walk_and_finds_nonprefix_subset():
    worker, engine, shell, _ = _fixture(singleton=False)
    batch = _scheduled((7, 8, 9))
    executing = worker._build_executing_batch(batch)
    engine.prepare_inputs(executing)
    batch.prepared_rows = executing.retain_rows({7, 8, 9})
    batch.graph_walk = "prefill_decode"
    batch.request_walks = dict.fromkeys((7, 8, 9), "decode")
    batch.walk_output_signals = {"decode": ["token"]}
    result = FIFOBatchBuilder().build_batch(BatchBuildRequest(
        "node", "prefill_decode", fresh=batch,
        execution_capabilities=shell.execution_capabilities,
    ))
    result.scheduled.relabel_if_uniform()
    assert result.scheduled.graph_walk == "decode"
    assert list(result.scheduled.request_to_worker_graph) == [7, 9]
    assert list(result.backlog.request_to_worker_graph) == [8]


@pytest.mark.parametrize("mixed_supported", [True, False])
def test_mixed_proposal_keeps_row_walks_before_uniform_fallback(mixed_supported):
    worker, engine, shell, reads = _fixture(singleton=True)
    seen = []

    def supports(facts):
        seen.append(facts)
        if facts.graph_walk == "prefill_decode":
            assert facts.row_walks == ("prefill", "decode")
            return Support.YES if mixed_supported else Support.NO
        assert facts.row_walks == (facts.graph_walk,) * facts.batch_size
        return Support.YES if facts.batch_size == 1 else Support.NO

    shell.execution_capabilities = ExecutionCapabilities((ExecutionOption(
        "mixed", supports, lambda facts: None, lambda *args: None, lambda *args: None,
    ),), {"prefill": 2, "decode": 2})
    batch = _scheduled((7, 8))
    batch.graph_walk = "prefill_decode"
    batch.request_walks = {7: "prefill", 8: "decode"}
    batch.walk_output_signals = {"prefill": ["token"], "decode": ["token"]}
    proposed = FIFOBatchBuilder().build_batch(BatchBuildRequest(
        "node", "prefill_decode", fresh=batch,
        execution_capabilities=shell.execution_capabilities,
    ))
    assert list(proposed.scheduled.request_to_worker_graph) == ([7, 8] if mixed_supported else [7])
    seen.clear()
    executing = worker._build_executing_batch(batch)
    engine.prepare_inputs(executing)
    worker._refine_prepared_batch(batch, executing, engine)
    assert seen[0].graph_walk == "prefill_decode"
    assert list(executing.request_ids) == ([7, 8] if mixed_supported else [7])
    assert executing.graph_walk == ("prefill_decode" if mixed_supported else "prefill")
    assert executing.step_context.request_walks == (batch.request_walks if mixed_supported else {})
    assert set(executing.deferred_rows) == (set() if mixed_supported else {8})
    assert shell.interpreted == reads == {7: 1, 8: 1}


def test_exhausted_candidate_window_advances_without_rejecting_parked_rows():
    # The generic subset consumer may inspect exact-two support; this is not
    # an assertion that the registry admits exact-two-only deployments.
    def supports(facts):
        return Support.YES if facts.batch_size == 2 and facts.total_tokens <= 10 else Support.NO

    caps = ExecutionCapabilities((ExecutionOption(
        "two", supports, lambda facts: None, lambda *args: None, lambda *args: None,
    ),), {"decode": 2})
    waiting = _scheduled(tuple(range(caps.candidate_window + 1)))
    for rid in waiting.request_to_worker_graph:
        waiting.prepared_rows[rid] = PreparedInputRow(
            NodeInputs(input_seq_len=4 if rid == caps.candidate_window else 6), {}, None,
            CurrentForwardPassInfo(f"r{rid}", "decode", 0, 0, 32, rid_handle=rid), False, False,
        )
    builder = FIFOBatchBuilder()
    first = builder.build_batch(BatchBuildRequest(
        "node", "decode", backlog=waiting, execution_capabilities=caps,
    ))
    assert first.scheduled is None
    assert set(first.backlog.prepared_rows) == set(range(caps.candidate_window + 1))
    second = builder.build_batch(BatchBuildRequest(
        "node", "decode", backlog=first.backlog, execution_capabilities=caps,
    ))
    assert list(second.scheduled.request_to_worker_graph) == [1, caps.candidate_window]
    assert len(second.backlog) == caps.candidate_window - 1


def test_speculation_merge_keeps_policy_rows_aligned_with_continuing_requests():
    seen = []

    def supports(facts):
        seen.append(facts)
        assert facts.batch_size == 2 and facts.row_walks == ("decode", "decode")
        rows = facts.resource_facts["sampler"]
        assert len(rows) == facts.batch_size and rows[0]["temperature"] == 0
        return Support.YES if rows[1]["temperature"] == 0 else Support.NO

    caps = ExecutionCapabilities((ExecutionOption(
        "greedy", supports, lambda facts: None, lambda *args: None, lambda *args: None,
    ),), {"decode": 2})
    result = FIFOBatchBuilder().build_batch(BatchBuildRequest(
        "node", "decode", fresh=_scheduled((8, 9)),
        pre_existing_batch_size=1, pre_existing_walk="decode", pre_existing_rids=(7,),
        execution_capabilities=caps,
        resource_facts=lambda rids: {"sampler": tuple({"temperature": int(rid == 8)} for rid in rids)},
    ))
    assert list(result.scheduled.request_to_worker_graph) == [9]
    assert list(result.backlog.request_to_worker_graph) == [8]
    assert len(seen) == 2
