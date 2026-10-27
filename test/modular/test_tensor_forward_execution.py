"""The real execution consumer, with CPU numerical work and capture rehearsal.

Host replay replaces only the unavailable device capture boundary. It executes
the actual graph-runner staging, program, logical adapter and resource lifecycle;
it does not establish CUDA/HIP capture or numerical kernel support.
"""
from dataclasses import dataclass, replace
from types import SimpleNamespace

import pytest
import torch

from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.distributed.communication import WorkerParallelGroups
from mstar.engine.accelerator_graph_backend import AcceleratorGraphPool
from mstar.engine.accelerator_graph_config import BatchedAcceleratorGraphConfig
from mstar.engine.engine import ExecutingBatch
from mstar.engine.execution import ExecutionCapabilities, ExecutionResult, InvocationFacts, Support
from mstar.engine.resources.base import Resource
from mstar.engine.resources.spec import NodeResourceSpec
from mstar.engine.resources.step import (
    ADMIT_OK,
    AdmitOutcome,
    AdmitRuntimeError,
    ExecutionBusy,
    ResourceStep,
    StepContext,
    SubmoduleStep,
)
from mstar.engine.tensor_forward import CapturePolicy, CompilePolicy, TensorForward, bind_tensor_forward
from mstar.graph.base import GraphEdge, GraphNode
from mstar.graph.special_destinations import EMIT_TO_CLIENT
from mstar.model.base import Model
from mstar.model.binding import NodeAssembly
from mstar.model.submodule_base import BatchedModelOutput, NodeInputs, NodeSubmodule
from mstar.model.weights import LogicalWeightSource, WeightSpec
from mstar.worker.engine_manager import EngineManager


class Counter(Resource):
    force_double_buffer = True

    def __init__(self):
        self.counts, self.events = {}, []

    @classmethod
    def build(cls, spec, info):
        return cls()

    def ingest_request(self, rid, overrides):
        self.counts[rid] = 0

    def remove_request(self, rid):
        self.counts.pop(rid, None)

    def reset_request(self, rid, free=False):
        self.counts[rid] = 0

    def plan(self, step, ctx):
        self.events.append(("plan", tuple(ctx.request_ids), ctx.slot))

    def commit(self, step, ctx):
        self.events.append(("commit", tuple(ctx.request_ids), ctx.slot))
        for rid in ctx.request_ids:
            self.counts[rid] += 1


class CounterSpec(NodeResourceSpec):
    @property
    def resource_class(self):
        return Counter


@dataclass(frozen=True)
class ValueResult:
    values: torch.Tensor


class Shell(NodeSubmodule):
    def __init__(self):
        super().__init__()
        self.mapped_rows = []

    def prepare_inputs(self, graph_walk, fwd_info, inputs, **kwargs):
        return NodeInputs(tensor_inputs={"x": inputs["x"][0]}, input_seq_len=1)

    def declare_step(self, graph_walk, request_ids, inputs, **kwargs):
        return SubmoduleStep({"counter": ResourceStep()})

    def preprocess(self, *args, **kwargs):
        raise AssertionError("logical shell preprocessing is not a physical program")

    def forward(self, *args, **kwargs):
        raise AssertionError("logical shell forward is not a physical program")

    def execution_result(self, result, invocation):
        self.mapped_rows.append(tuple(invocation.request_ids))
        assert isinstance(result, ValueResult)
        return ExecutionResult(BatchedModelOutput(row_outputs={"edge_value": result.values}))


class Program(torch.nn.Module):
    def __init__(self, gain, *, batched=True):
        super().__init__()
        self.gain = torch.nn.Parameter(gain)
        self.batched = batched

    def preprocess(self, graph_walk, engine_inputs, inputs):
        return {"x": torch.stack([row.tensor_inputs["x"] for row in inputs])}

    def forward(self, graph_walk, engine_inputs, x):
        return BatchedModelOutput(row_outputs={"values": x * self.gain + 1})

    forward_batched = forward

    def can_batch(self, batch, model_inputs):
        return self.batched

    def get_accelerator_graph_configs(self, device, tp_world_size=1):
        if not self.batched:
            return []
        return [BatchedAcceleratorGraphConfig(
            "decode", NodeInputs(tensor_inputs={"x": torch.zeros(4, device=device)}, input_seq_len=1),
            capture_batch_sizes=[1, 2, 4], compile=False,
        )]


class OrdinaryNode(NodeSubmodule):
    disable_torch_compile = True
    disable_autocast = True

    def prepare_inputs(self, graph_walk, fwd_info, inputs, **kwargs):
        return NodeInputs(tensor_inputs={"x": inputs["x"][0]})

    def forward(self, graph_walk, engine_inputs, x):
        return {"edge_value": [x * 7]}


class TinyModel(Model):
    def __init__(self):
        self.shells, self.ordinary = [], []
        self.weight_reads = 0

    def get_node_resources(self):
        return [CounterSpec("counter", {"bound"})]

    def get_graph_walk_graphs(self):
        return {
            "decode": GraphNode("bound", {"x"}, [GraphEdge(EMIT_TO_CLIENT, "edge_value")]),
            "other": GraphNode("plain", {"x"}, [GraphEdge(EMIT_TO_CLIENT, "edge_value")]),
        }

    def create_submodule(self, node_name):
        assert node_name == "bound"
        shell = Shell()
        self.shells.append(shell)
        return shell

    def get_submodule(self, node_name, device, **kwargs):
        assert node_name == "plain"
        ordinary = OrdinaryNode()
        self.ordinary.append(ordinary)
        return ordinary

    def logical_weights(self):
        def read():
            self.weight_reads += 1
            yield "gain", torch.tensor(3.)
        return LogicalWeightSource([WeightSpec("gain", ())], read)

    def get_autocast_dtype(self):
        return None

    def process_prompt(self, *args, **kwargs):
        raise NotImplementedError

    def postprocess(self, *args, **kwargs):
        raise NotImplementedError

    def get_initial_forward_pass_args(self, *args, **kwargs):
        raise NotImplementedError

    def get_partition_forward_pass_args(self, *args, **kwargs):
        raise NotImplementedError


def supported(facts):
    if facts.graph_walk not in (None, "decode") or (
        facts.batch_size is not None and not 1 <= facts.batch_size <= 4
    ):
        return Support.NO
    if facts.graph_walk is None or facts.batch_size is None:
        return Support.UNKNOWN
    return Support.YES


def build(*, device="cpu", batched=True, capture=True, options=None, limit=4, fixed=False):
    model, programs, facts_seen, adapted = TinyModel(), [], [], []

    def assemble(binding):
        def bind(resources):
            gain = binding.model.logical_weights().load()["gain"].to(binding.device)
            result = []
            for name, support, scale in options or [("tensor", supported, 1)]:
                program_type = Program
                program = program_type(gain * scale, batched=batched)
                programs.append(program)

                def supports(facts, support=support):
                    facts_seen.append(facts)
                    return support(facts)

                def adapt(raw, invocation):
                    adapted.append((tuple(invocation.request_ids), raw))
                    values = (raw.row_outputs["values"] if raw.row_outputs is not None
                              else torch.cat([raw.per_rid_outputs[rid]["values"][0]
                                              for rid in invocation.request_ids]))
                    return ValueResult(values)

                result.append(bind_tensor_forward(
                    name, supports, binding, resources,
                    TensorForward(program, capture=CapturePolicy(enabled=capture), result=None if fixed else adapt),
                ))
            return ExecutionCapabilities(tuple(result), {"decode": limit},
                                         fixed_option=result[0] if fixed else None,
                                         fixed_admission=limit if fixed else 0)
        return NodeAssembly(bind, execution_names=tuple(name for name, _, _ in options or [("tensor", None, 0)]))

    manager = EngineManager()
    manager.load(
        {"bound", "plain"}, torch.device(device), {"runtime_extension": {"nodes": ["bound"]},
         "node_groups": [{"node_names": ["bound", "plain"]}]},
        WorkerParallelGroups(1, 0), None, model, None,
        runtime_extension=SimpleNamespace(assemble=assemble),
    )
    return manager, model, programs, facts_seen, adapted


def pending(manager, values, *, node="bound", walk="decode"):
    infos = {rid: CurrentForwardPassInfo(f"request-{rid}", walk, 0, 0, 32, rid_handle=rid) for rid in values}
    return ExecutingBatch(
        node, infos, StepContext(list(values), walk, 0, False),
        per_request_input_tensors={
            rid: {"x": [torch.tensor(row, device=manager.engine._device)]} for rid, row in values.items()
        },
    )


def execute(manager, values, **kwargs):
    batch = pending(manager, values, **kwargs)
    manager.engine.prepare_inputs(batch)
    return batch, manager.engine.exec(batch)


def host_replay(monkeypatch, engine):
    from mstar.engine import accelerator_graph_runner

    class HostStream:
        def __init__(self):
            self.waited = []

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def wait_event(self, event):
            self.waited.append(event)

    # Rehearsal is CPU-only: upstream warmup also queries peak allocation.
    monkeypatch.setattr(torch.accelerator, "memory_allocated", lambda *a: 0)
    monkeypatch.setattr(torch.accelerator, "reset_peak_memory_stats", lambda *a: None)
    monkeypatch.setattr(torch.accelerator, "max_memory_allocated", lambda *a: 0)
    monkeypatch.setattr(torch.accelerator, "empty_cache", lambda: None)
    stream = HostStream()
    backend = SimpleNamespace(
        is_available=lambda: True, graph_pool_handle=lambda: AcceleratorGraphPool(object()),
        memory_allocated=lambda: 0, set_device=lambda: None, synchronize=lambda: None,
        new_stream=HostStream, current_stream=lambda: stream,
    )
    engine._device_capability = SimpleNamespace(
        device=torch.device("cpu"), graph_backend=backend, is_available=False, is_accelerator=False,
    )
    captures = []

    class Replay:
        def __init__(self, run, outputs):
            self.run, self.outputs, self.replays = run, outputs, 0

        def replay(self):
            self.replays += 1
            fresh = self.run()
            for name, value in fresh.row_outputs.items():
                self.outputs.row_outputs[name].copy_(value)

    def capture(run, pool, device, autocast_dtype):
        outputs = run()
        replay = Replay(run, outputs)
        captures.append(replay)
        return replay, outputs

    monkeypatch.setattr(accelerator_graph_runner, "capture_into_graph", capture)
    return captures


@pytest.mark.parametrize("mode", ["cpu", "host_replay", "gpu"])
def test_separate_program_real_engine_and_stable_logical_outputs(monkeypatch, mode, caplog):
    if mode == "gpu" and not torch.cuda.is_available():
        pytest.skip("real accelerator graph execution requires CUDA/ROCm hardware")
    manager, model, programs, facts, adapted = build(device="cuda" if mode == "gpu" else "cpu")
    try:
        if mode == "host_replay":
            host_replay(monkeypatch, manager.engine)
        manager.engine.warmup()
        shell = model.shells[0]
        assert not tuple(shell.parameters())
        assert programs[0] is not shell and model.weight_reads == 1
        assert model.ordinary[0].execution_capabilities is None
        for rid in (7, 8, 9):
            manager.add_request(rid)
        facts.clear()
        values = {rid: [float(rid), 2., 3., 4.] for rid in (7, 8, 9)}
        with caplog.at_level("INFO"):
            first, output = execute(manager, values)
        assert first.admit_error is None
        saved = output.per_rid_outputs[7]["edge_value"][0]
        expected = torch.tensor([values[7]], device=saved.device) * 3 + 1
        torch.testing.assert_close(saved, expected)
        assert facts[0].batch_size == 3 and facts[0].token_lengths == (1, 1, 1)
        assert all(f.batch_size != 4 for f in facts)
        assert shell.mapped_rows[-1] == adapted[-1][0] == (7, 8, 9)
        assert adapted[-1][1].row_outputs["values"].shape[0] == 3
        if mode != "cpu":
            assert first.step_context.slot_lease.bucket.bs == 4
            assert "path=captured rows=3 runner=tensor_forward" in caplog.text
        second, result = execute(manager, {7: [1., 2., 3., 4.]})
        assert second.admit_error is None
        torch.testing.assert_close(result.per_rid_outputs[7]["edge_value"][0],
                                   torch.tensor([[4., 7., 10., 13.]], device=saved.device))
        torch.testing.assert_close(saved, expected)
        _, plain = execute(manager, {7: [1., 2., 3., 4.]}, node="plain", walk="other")
        torch.testing.assert_close(plain.per_rid_outputs[7]["edge_value"][0],
                                   torch.tensor([7., 14., 21., 28.], device=saved.device))
        counter = manager.engine._resources["counter"]
        assert {rid: counter.counts[rid] for rid in (7, 8, 9)} == {7: 2, 8: 1, 9: 1}
    finally:
        if mode == "gpu":
            torch.cuda.synchronize()
        manager.shutdown()


def test_uncaptured_unbatchable_program_keeps_per_row_resource_cycles():
    manager, _, _, facts, _ = build(batched=False)
    try:
        for rid in (7, 8, 9):
            manager.add_request(rid)
        facts.clear()
        batch, output = execute(manager, {rid: [float(rid), 2., 3., 4.] for rid in (7, 8, 9)})
        assert batch.admit_error is None and not batch.running_batched
        assert [f.batch_size for f in facts if f.token_lengths is not None] == [3, 1, 1, 1]
        assert set(output.per_rid_outputs) == {7, 8, 9}
        counter = manager.engine._resources["counter"]
        assert counter.counts == {7: 1, 8: 1, 9: 1}
        assert [slot for event, _, slot in counter.events if event == "commit"] == [0, 1, 0]
    finally:
        manager.shutdown()


@pytest.mark.parametrize("mode", ["eager", "host_replay", "per_request"])
def test_output_only_mapping_follows_commit_and_preserves_owned_outputs(monkeypatch, mode):
    monkeypatch.setattr(Shell, "execution_result_is_output_only", True)
    manager, model, _, _, _ = build(batched=mode != "per_request")
    try:
        if mode == "host_replay":
            host_replay(monkeypatch, manager.engine)
        manager.engine.warmup()
        for rid in (7, 8, 9):
            manager.add_request(rid)
        runner = model.shells[0].execution_capabilities.options[0].runner
        original_mapper = runner.implementation.result
        counter = manager.engine._resources["counter"]
        batch = pending(manager, {rid: [float(rid), 2., 3., 4.] for rid in (7, 8, 9)})
        mapped = []

        def map_after_commit(raw, invocation):
            assert all(counter.counts[rid] == 1 for rid in invocation.request_ids)
            assert batch.commit_done.is_set() == (mode != "per_request")
            assert not batch.outputs_ready.is_set()
            mapped.append(tuple(invocation.request_ids))
            return original_mapper(raw, invocation)

        runner.implementation = replace(runner.implementation, result=map_after_commit)
        manager.engine.prepare_inputs(batch)
        # Retain a reference to the selected batch workspace for lifetime checks.
        manager.engine._reserve_execution(batch)
        workspace = batch.selected_execution.workspace
        result = manager.engine.exec(batch)
        assert workspace.closed and batch.outputs_ready.is_set() and batch.commit_done.is_set()
        assert mapped == ([(7,), (8,), (9,)] if mode == "per_request" else [(7, 8, 9)])
        assert {rid: counter.counts[rid] for rid in (7, 8, 9)} == {7: 1, 8: 1, 9: 1}
        saved = result.per_rid_outputs[7]["edge_value"][0]
        expected = torch.tensor([[22., 7., 10., 13.]])
        torch.testing.assert_close(saved, expected)
        runner.implementation = replace(runner.implementation, result=original_mapper)
        # Cycle both slots so a subsequent replay overwrites the original buffers.
        for _ in range(2):
            execute(manager, {rid: [0., 0., 0., 0.] for rid in (7, 8, 9)})
        torch.testing.assert_close(saved, expected)
    finally:
        manager.shutdown()


@pytest.mark.parametrize("failure", ["mapping", "late_effects"])
def test_output_only_mapping_failure_releases_workspace_without_publishing(monkeypatch, failure):
    monkeypatch.setattr(Shell, "execution_result_is_output_only", True)
    manager, model, _, _, _ = build(capture=False)
    try:
        manager.add_request(7)
        runner = model.shells[0].execution_capabilities.options[0].runner

        def fail(raw, invocation):
            if failure == "mapping":
                raise ValueError("invalid logical output")
            return ExecutionResult(raw, {"counter": object()})

        runner.implementation = replace(runner.implementation, result=fail)
        batch = pending(manager, {7: [1., 2., 3., 4.]})
        manager.engine.prepare_inputs(batch)
        manager.engine._reserve_execution(batch)
        workspace = batch.selected_execution.workspace
        initial_outputs = batch.outputs
        with pytest.raises(ValueError, match="invalid logical output|returned resource effects"):
            manager.engine.exec(batch)
        assert workspace.closed and batch.selected_execution is None
        # Error cleanup wakes waiters too; it must not publish a result.
        assert batch.outputs_ready.is_set() and batch.outputs is initial_outputs
        assert manager.engine._resources["counter"].counts[7] == 1
    finally:
        manager.shutdown()


def test_unbatchable_program_rejects_unsupported_singletons_before_effects():
    def batch_only(facts):
        return Support.NO if facts.batch_size == 1 else supported(facts)
    manager, _, _, _, _ = build(batched=False, options=[("batch_only", batch_only, 1)])
    try:
        for rid in (7, 8):
            manager.add_request(rid)
        batch, _ = execute(manager, {7: [1., 2., 3., 4.], 8: [4., 3., 2., 1.]})
        assert isinstance(batch.admit_error, AdmitRuntimeError)
        assert not manager.engine._resources["counter"].events
    finally:
        manager.shutdown()


@pytest.mark.parametrize("batched", [False, True])
def test_resource_refusal_retains_interpreted_rows_and_runs_no_program(monkeypatch, batched):
    manager, _, _, _, adapted = build(batched=batched, capture=False)
    try:
        for rid in (7, 8):
            manager.add_request(rid)
        counter = manager.engine._resources["counter"]

        def refuse(step, ctx):
            return AdmitOutcome(False, reason=ExecutionBusy("busy")) if 8 in ctx.request_ids else ADMIT_OK

        monkeypatch.setattr(counter, "admit", refuse)
        batch, _ = execute(manager, {7: [1., 2., 3., 4.], 8: [4., 3., 2., 1.]})
        assert isinstance(batch.admit_error, ExecutionBusy)
        assert set(batch.prepared_rows) == {7, 8}
        assert not adapted and not counter.events
        assert batch.selected_execution is None
    finally:
        manager.shutdown()


def test_two_tensor_programs_select_before_their_capture_geometry(monkeypatch):
    def one(facts):
        return Support.NO if facts.batch_size not in (None, 1) else supported(facts)

    def many(facts):
        return Support.NO if facts.batch_size == 1 else supported(facts)

    manager, model, _, _, _ = build(options=[("one", one, 1), ("many", many, 2)])
    try:
        host_replay(monkeypatch, manager.engine)
        manager.engine.warmup()
        for rid in (7, 8, 9):
            manager.add_request(rid)
        first, one_out = execute(manager, {7: [1., 2., 3., 4.]})
        second, many_out = execute(manager, {rid: [1., 2., 3., 4.] for rid in (7, 8, 9)})
        assert first.step_context.slot_lease.bucket.execution_name == "one"
        assert second.step_context.slot_lease.bucket.execution_name == "many"
        assert second.step_context.slot_lease.bucket.bs == 4
        torch.testing.assert_close(one_out.per_rid_outputs[7]["edge_value"][0], torch.tensor([[4., 7., 10., 13.]]))
        torch.testing.assert_close(many_out.per_rid_outputs[7]["edge_value"][0], torch.tensor([[7., 13., 19., 25.]]))
        assert len(model.shells[0].execution_capabilities.options) == 2
    finally:
        manager.shutdown()


@pytest.mark.parametrize("platform_overrides", [False, True])
def test_compile_policy_explicitly_controls_platform_overrides(monkeypatch, platform_overrides):
    import mstar.platforms.compile as compile_module
    calls = []
    monkeypatch.setattr(torch, "compile", lambda fn, **kw: calls.append(("compile", kw)) or fn)
    monkeypatch.setattr(compile_module, "with_inductor_overrides",
                        lambda fn, *args: calls.append(("platform", args)) or fn)
    def fn(x):
        return x + 1
    policy = CompilePolicy(enabled=True, mode="reduce-overhead", dynamic=False,
                           platform_overrides=platform_overrides)
    assert policy.compile(fn, num_tokens=8, batch_size=2)(3) == 4
    assert calls[0] == ("compile", {"mode": "reduce-overhead", "fullgraph": False, "dynamic": False})
    assert (len(calls) == 2) is platform_overrides
    if platform_overrides:
        assert calls[1] == ("platform", (8, 2))


def test_required_capture_does_not_claim_success_on_cpu():
    manager, model, _, _, _ = build()
    try:
        runner = model.shells[0].execution_capabilities.options[0].runner
        runner.implementation = replace(runner.implementation, capture=CapturePolicy(enabled=True, required=True))
        with pytest.raises(RuntimeError, match="required accelerator graph capture failed"):
            manager.engine.warmup()
    finally:
        manager.shutdown()


def test_speculative_selection_pins_runner_before_preplan_and_rolls_back(monkeypatch):
    class HostEvent:
        def __init__(self, **kwargs):
            self.stream = None

        def record(self, stream=None):
            self.stream = stream

    monkeypatch.setattr(torch, "Event", HostEvent)
    monkeypatch.setattr(Counter, "supports_preplan", True)
    manager, model, _, _, _ = build()
    try:
        host_replay(monkeypatch, manager.engine)
        manager.engine.warmup()
        for rid in (7, 8):
            manager.add_request(rid)
        engine = manager.engine
        batch = pending(manager, {7: [1., 2., 3., 4.], 8: [4., 3., 2., 1.]})
        lease = engine.reserve_replay_slot(batch)
        selected = batch.selected_execution
        assert selected is not None and lease.bucket.execution_name == selected.option.name
        assert engine.pre_plan_for_batch(batch)
        event = batch.preplan_event
        assert event.stream is not None and engine._runner.staged
        engine.prepare_inputs(batch)
        result = engine.exec(batch)
        assert batch.admit_error is None and set(result.per_rid_outputs) == {7, 8}
        assert selected._closed and batch.selected_execution is None
        assert not engine._runner.staged
        runner = model.shells[0].execution_capabilities.options[0].runner.graph_runner
        assert event in runner._graph_backend.current_stream().waited

        abandoned = pending(manager, {7: [1., 2., 3., 4.], 8: [4., 3., 2., 1.]})
        engine.reserve_replay_slot(abandoned)
        reservation = abandoned.selected_execution
        assert engine.pre_plan_for_batch(abandoned)
        abandoned.drop_rids({8})
        assert engine.preplan_is_stale(abandoned)
        engine.reset_pre_plan_for_batch(abandoned)
        assert reservation._closed and abandoned.selected_execution is None
        assert abandoned.step_context.slot_lease is None
        assert abandoned.step_context.padded_request_ids == [7]
        assert not engine._runner.staged
    finally:
        manager.shutdown()


def test_unknown_program_constraints_cannot_preplan_or_lease(monkeypatch):
    def needs_inputs(facts):
        support = supported(facts)
        return Support.UNKNOWN if support is not Support.NO and facts.token_lengths is None else support

    monkeypatch.setattr(Counter, "supports_preplan", True)
    manager, _, _, _, _ = build(options=[("needs_inputs", needs_inputs, 1)])
    try:
        host_replay(monkeypatch, manager.engine)
        manager.engine.warmup()
        manager.add_request(7)
        counter = manager.engine._resources["counter"]
        counter.events.clear()
        batch = pending(manager, {7: [1., 2., 3., 4.]})
        assert manager.engine.reserve_replay_slot(batch) is None
        assert batch.slot is None and batch.selected_execution is None
        assert not manager.engine.pre_plan_for_batch(batch)
        assert not counter.events
        manager.engine.prepare_inputs(batch)
        manager.engine.exec(batch)
        assert batch.admit_error is None and counter.counts[7] == 1
    finally:
        manager.shutdown()


def test_selected_run_uses_the_warmed_runner_outside_engine_exec(monkeypatch):
    from mstar.engine.execution import LogicalInvocation
    from mstar.model.submodule_base import LazyRequestStates, ModelInputsFromEngine

    manager, model, _, _, _ = build()
    try:
        captures = host_replay(monkeypatch, manager.engine)
        manager.engine.warmup()
        manager.add_request(7)
        engine, shell = manager.engine, model.shells[0]
        option = shell.execution_capabilities.options[0]
        selected = shell.execution_capabilities.reserve(InvocationFacts("decode", 1, (1,)))
        batch = pending(manager, {7: [1., 2., 3., 4.]})
        engine.prepare_inputs(batch)
        logical = LogicalInvocation(
            "decode", [7], batch.inputs,
            ModelInputsFromEngine([7], batch.per_request_info, engine._resources,
                                  per_request_states=LazyRequestStates(shell, [7])),
        )
        physical = option.runner.prepare_invocation(selected.workspace, logical, slot=0)
        assert selected.workspace.lease is not None
        step = shell.declare_step("decode", physical.request_ids, physical.inputs)
        step.set_ctx(StepContext(
            [7], "decode", 0, False, slot_lease=selected.workspace.lease,
        ))
        step.ctx.set_padded_rids(physical.request_ids)
        physical.engine_inputs.step = step
        assert engine._runner.admit(step).ok
        physical = replace(physical, plan_results=engine._runner.plan(step))
        logical_result = selected.run(physical, None)
        outcome = shell.execution_result(logical_result, selected.workspace.original)
        engine._runner.accept_results(step, outcome.resource_results)
        engine._runner.commit(step)
        selected.close()
        assert sum(graph.replays for graph in captures) == 1
        assert engine._resources["counter"].counts[7] == 1
        torch.testing.assert_close(outcome.outputs.row_outputs["edge_value"], torch.tensor([[4., 7., 10., 13.]]))
    finally:
        manager.shutdown()


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("capture", [False, True])
def test_fixed_named_program_uses_common_lifecycle(monkeypatch, batched, capture):
    manager, model, programs, facts, adapted = build(fixed=True, batched=batched, capture=capture, limit=2)
    try:
        graphs = host_replay(monkeypatch, manager.engine)
        manager.engine.warmup()
        for rid in (7, 8):
            manager.add_request(rid)
        batch, output = execute(manager, {7: [1., 2., 3., 4.], 8: [4., 3., 2., 1.]})
        assert batch.admit_error is None
        assert not adapted and not model.shells[0].mapped_rows
        assert manager.engine._submodules["bound"].physical is programs[0]
        assert model.shells[0].execution_capabilities is not None
        assert manager.engine._resources["counter"].counts[7] == 1
        assert manager.engine._resources["counter"].counts[8] == 1
        torch.testing.assert_close(output.per_rid_outputs[7]["values"][0], torch.tensor([[4., 7., 10., 13.]]))
        if capture and batched:
            # Physical capture geometry is independent of the admission bound.
            assert manager.engine._submodules["bound"].accelerator_graph_runner.max_batch_size_for("decode") == 4
            assert sum(graph.replays for graph in graphs) == 1
    finally:
        manager.shutdown()


def test_fixed_unbatched_program_receives_each_rows_walk():
    manager, model, programs, facts, adapted = build(fixed=True, batched=False, capture=False)
    try:
        engine = manager.engine
        caps = engine.execution_capabilities("bound")
        caps.batch_limits.update({"prefill": 4})
        seen = []
        original = engine._submodules["bound"].forward
        def forward(walk, **kwargs):
            seen.append(walk)
            return original(walk, **kwargs)
        engine._submodules["bound"].forward = forward
        for rid in (1, 2):
            manager.add_request(rid, {})
        batch = pending(manager, {1: [1.], 2: [2.]})
        batch.step_context.graph_walk = "mixed"
        batch.step_context.request_walks = {1: "prefill", 2: "decode"}
        engine.prepare_inputs(batch)
        engine.exec(batch)
        assert seen == ["prefill", "decode"]
    finally:
        manager.shutdown()


@pytest.mark.parametrize("adapter", ["absent", "positional"])
@pytest.mark.parametrize("uniform", [False, True])
def test_fixed_program_preserves_optional_can_batch_interface(monkeypatch, adapter, uniform):
    if adapter == "absent":
        monkeypatch.delattr(Program, "can_batch")
    else:
        monkeypatch.setattr(Program, "can_batch", lambda self, batch, inputs: self.batched)
    if uniform:
        row = NodeInputs(tensor_inputs={"x": torch.tensor([1., 2., 3., 4.])}, input_seq_len=1)
        monkeypatch.setattr(Shell, "uniform_row_inputs", lambda self, *a, **kw: row)
    manager, _, _, _, _ = build(fixed=True, capture=False)
    try:
        manager.add_request(7)
        batch, output = execute(manager, {7: [1., 2., 3., 4.]})
        assert batch.running_batched
        torch.testing.assert_close(
            output.per_rid_outputs[7]["values"][0], torch.tensor([[4., 7., 10., 13.]]),
        )
    finally:
        manager.shutdown()


@pytest.mark.parametrize("output_only", [False, True])
def test_result_collection_preserves_implementation_autocast(monkeypatch, output_only):
    monkeypatch.setattr(Shell, "execution_result_is_output_only", output_only)
    manager, model, _, _, _ = build(capture=False)
    try:
        runner = model.shells[0].execution_capabilities.options[0].runner
        def mapped(raw, invocation):
            assert not torch.is_grad_enabled()
            assert torch.is_autocast_enabled("cpu")
            values = (raw.row_outputs["values"] if raw.row_outputs is not None else
                      torch.cat([raw.per_rid_outputs[rid]["values"][0]
                                 for rid in invocation.request_ids]))
            return ValueResult(values @ torch.eye(4))
        runner.implementation = replace(
            runner.implementation, result=mapped, autocast_dtype=torch.bfloat16,
        )
        manager.add_request(7)
        _, output = execute(manager, {7: [1., 2., 3., 4.]})
        result = output.per_rid_outputs[7]["edge_value"][0]
        assert result.dtype is torch.bfloat16
        torch.testing.assert_close(result, torch.tensor([[4., 7., 10., 13.]], dtype=torch.bfloat16))
    finally:
        manager.shutdown()


def test_fixed_required_capture_rejects_execution_before_warmup():
    manager, model, _, _, _ = build(fixed=True, capture=True)
    try:
        runner = model.shells[0].execution_capabilities.fixed_option.runner
        runner.implementation = replace(
            runner.implementation, capture=replace(runner.implementation.capture, required=True),
        )
        manager.add_request(7)
        with pytest.raises(RuntimeError, match="required capture has not been warmed"):
            execute(manager, {7: [1., 2., 3., 4.]})
        assert not manager.engine._resources["counter"].events
    finally:
        manager.shutdown()


@pytest.mark.parametrize("owner", ["selection", "capture_lease", "program"])
def test_failed_close_retains_ownership_for_retry(owner):
    from mstar.engine.execution import SelectedExecution
    from mstar.engine.tensor_forward import TensorForwardLease, TensorForwardRunner

    calls = []

    def release(*args):
        calls.append(args)
        if len(calls) == 1:
            raise RuntimeError("owner has not settled")

    backing = SimpleNamespace(release=release)
    if owner == "selection":
        value = SelectedExecution(None, SimpleNamespace(close=release))
        def closed():
            return value._closed
    elif owner == "capture_lease":
        value = TensorForwardLease(
            SimpleNamespace(graph_runner=backing),
            original=SimpleNamespace(request_ids=(7,)), lease=object(),
        )
        def closed():
            return value.closed
    else:
        value = TensorForwardRunner.__new__(TensorForwardRunner)
        value._closed = False
        value.implementation = SimpleNamespace(close=release)
        value.graph_runner = backing
        def closed():
            return value._closed

    with pytest.raises(RuntimeError, match="owner has not settled"):
        value.close()
    assert not closed()
    if owner == "program":
        assert value.graph_runner is backing
    value.close()
    assert closed()
    value.close()
    assert len(calls) == 2
