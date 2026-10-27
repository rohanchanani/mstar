"""Fresh assembly through the real engine and its resource lifecycle.

No worker, CUDA, or native-device execution is simulated. The test program is a
CPU implementation of a small stateful node, exercising the same binding and
plan/result contract the Pi program consumes.
"""
from dataclasses import dataclass

import pytest
import torch

from mstar.communication.tensors import LocalTransferEngine
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.distributed.communication import WorkerParallelGroups
from mstar.engine.engine import ExecutingBatch
from mstar.engine.execution import (
    ExecutionCapabilities,
    ExecutionOption,
    ExecutionResult,
    InvocationFacts,
    Support,
    UnsupportedInvocation,
)
from mstar.engine.resources.kv.config import KVSpec, KVStep, PagedKVConfig
from mstar.engine.resources.kv.transfer import TransferEngineInfo
from mstar.engine.resources.position.config import PositionConfig, PositionSpec, PositionStep
from mstar.engine.resources.sampler.config import SamplerSpec, SamplerStep, SamplingReqConfig
from mstar.engine.resources.sampler.plan import SamplingResult
from mstar.engine.resources.step import AdmitRuntimeError, ExecutionBusy, Segment, StepContext, SubmoduleStep
from mstar.graph.base import GraphEdge, GraphNode
from mstar.graph.special_destinations import EMIT_TO_CLIENT
from mstar.model.base import Model
from mstar.model.binding import NodeAssembly
from mstar.model.submodule_base import BatchedModelOutput, NodeInputs, NodeSubmodule
from mstar.model.weights import LogicalWeightSource, WeightSpec
from mstar.worker.engine_manager import EngineManager

CPU = torch.device("cpu")


@dataclass(frozen=True)
class NodeResult:
    tokens: torch.Tensor


class Shell(NodeSubmodule):
    def __init__(self):
        super().__init__()
        self.declarations = 0

    def prepare_inputs(self, graph_walk, fwd_info, inputs, **kwargs):
        x = inputs["x"][0]
        return NodeInputs(tensor_inputs={"x": x}, input_seq_len=x.numel())

    def declare_step(self, graph_walk, request_ids, inputs, **kwargs):
        self.declarations += 1
        segments = tuple(Segment(rid, "main", inp.input_seq_len)
                         for rid, inp in zip(request_ids, inputs, strict=True))
        return SubmoduleStep({"kv": KVStep(segments=segments),
                              "position": PositionStep(segments=segments),
                              "sampler": SamplerStep()})

    def forward(self, *args, **kwargs):
        raise AssertionError("the loaded reference forward must not be used")

    def preprocess(self, *args, **kwargs):
        raise AssertionError("physical reference preprocessing must not be used")

    def execution_result(self, result, invocation):
        return ExecutionResult(
            BatchedModelOutput(row_outputs={"token": result.tokens[:, None]},
                               check_stop_buffers={"token": result.tokens[:, None]}),
            {"sampler": SamplingResult(result.tokens)},
        )


class SmallModel(Model):
    def __init__(self, pages=8):
        self.pages = pages
        self.created = []
        self.weight_reads = 0

    def get_node_resources(self):
        cfg = PagedKVConfig(num_layers=1, num_kv_heads=1, head_dim=2,
                            max_seq_len=32, max_num_pages=self.pages,
                            page_size=4, prefix_cache=False)
        return [KVSpec("kv", {"node"}, cfg),
                PositionSpec("position", {"node"}, PositionConfig("kv")),
                SamplerSpec("sampler", {"node"}, 32, enable_repetion_penalty=False)]

    def get_graph_walk_graphs(self):
        return {"decode": GraphNode("node", {"x"}, [GraphEdge(EMIT_TO_CLIENT, "token", output_modality="text")])}

    def create_submodule(self, node_name):
        assert node_name == "node"
        shell = Shell()
        self.created.append(shell)
        return shell

    def logical_weights(self):
        def source():
            self.weight_reads += 1
            yield "bias", torch.tensor(2.)
        return LogicalWeightSource([WeightSpec("bias", ())], source)

    def get_submodule(self, *args, **kwargs):
        raise AssertionError("fresh construction must not load the reference")

    def get_autocast_dtype(self):
        return torch.float32

    def process_prompt(self, prompt, **kwargs):
        return {"x": [torch.tensor([float(prompt)])]}

    def postprocess(self, output, modality, **kwargs):
        return str(int(output[0])).encode()

    def get_initial_forward_pass_args(self, *args, **kwargs):
        raise AssertionError("this is a worker-engine integration test")

    def get_partition_forward_pass_args(self, *args, **kwargs):
        raise AssertionError("this is a worker-engine integration test")


class Lease:
    def __init__(self, owner):
        self.owner = owner
        self.closed = False

    def close(self):
        assert not self.closed
        self.closed = True
        self.owner.busy = False
        self.owner.releases += 1


class Extension:
    def __init__(self):
        self.busy = False
        self.acquires = 0
        self.releases = 0
        self.calls = []
        self.bindings = []
        self.plans = []

    def assemble(self, binding):
        self.bindings.append(binding)
        assert tuple(binding.submodule.parameters()) == ()
        assert binding.submodule.node_resources == {}

        def bind(resources):
            assert binding.submodule.node_resources == resources
            bias = binding.model.logical_weights().load()["bias"]

            def supports(facts):
                if facts.graph_walk not in (None, "decode") or facts.batch_size not in (None, 1):
                    return Support.NO
                if facts.graph_walk is None or facts.batch_size is None or facts.token_lengths is None:
                    return Support.UNKNOWN
                return Support.YES if 1 <= facts.token_lengths[0] <= 4 else Support.NO

            def acquire(facts):
                if self.busy:
                    return None
                self.busy = True
                self.acquires += 1
                return Lease(self)

            def materialize(invocation, execution):
                assert execution.stream is None  # CPU is already synchronous here.
                assert not torch.is_autocast_enabled("cpu")
                self.plans.append(invocation.plan_results)
                return invocation

            def primitive(invocation, execution):
                self.calls.append(invocation)
                x = invocation.inputs[0].tensor_inputs["x"]
                positions = invocation.plan_results["position"]["main"]
                k = x[:, None, None].expand(-1, 1, 2).contiguous()
                resources["kv"].write_kv(k, k + bias, label="main")
                return NodeResult((x[-1:] + positions[-1:] + bias).to(torch.long))

            return ExecutionCapabilities((ExecutionOption(
                "cpu-node", supports, acquire, materialize, primitive,
            ),), {"decode": 1})

        return NodeAssembly(bind)


def build(pages=8):
    model, extension = SmallModel(pages), Extension()
    manager = EngineManager()
    manager.load({"node"}, CPU, {"node_groups": [{"node_names": ["node"]}]}, WorkerParallelGroups(1, 0),
                 TransferEngineInfo("worker", "test", LocalTransferEngine("localhost")), model, None,
                     runtime_extension=extension)
    manager.add_request(7, {"sampler": SamplingReqConfig(temperature=0)})
    return manager, model, extension


def batch(values):
    info = CurrentForwardPassInfo("request-7", "decode", 0, 0, 32, rid_handle=7)
    return ExecutingBatch("node", {7: info}, StepContext([7], "decode", 0, False),
                          per_request_input_tensors={7: {"x": [torch.tensor(values)]}})


def execute(manager, values):
    pending = batch(values)
    manager.engine.prepare_inputs(pending)
    return pending, manager.engine.exec(pending)


def test_fresh_assembly_drives_real_resources_and_retains_previous_outputs():
    manager, model, extension = build()
    engine = manager.engine
    shell = model.created[0]
    assert extension.bindings[0].submodule is shell
    assert shell.binding.device == CPU
    assert model.weight_reads == 1
    assert not engine.can_pre_plan("node")
    first, first_output = execute(manager, [1., 2., 3.])
    assert first.admit_error is None
    assert first_output.per_rid_outputs[7]["token"][0].item() == 7
    assert first.commit_done.is_set() and first.outputs_ready.is_set()
    first_positions = extension.plans[0]["position"]["main"]
    first_views = extension.plans[0]["kv"]["main"].views
    second, second_output = execute(manager, [4.])
    assert second.admit_error is None
    assert second_output.per_rid_outputs[7]["token"][0].item() == 9
    assert first_output.per_rid_outputs[7]["token"][0].item() == 7
    assert first_positions.tolist() == [0, 1, 2]
    assert first_views[0].length == 3
    assert extension.plans[1]["position"]["main"].tolist() == [3]
    assert engine._resources["position"].position(7, "main") == 4
    assert engine._resources["sampler"]._sampler._step_offset[7] == 2
    assert extension.acquires == extension.releases == 2
    assert extension.calls[0].row_walks == ("decode",)
    assert shell.declarations == 2
    manager.remove_request(7)
    engine._resources["kv"].assert_pages_conserved()
    engine.shutdown()


def test_busy_and_unsupported_choices_have_no_admission_or_commit_effects():
    manager, model, extension = build()
    shell = model.created[0]
    initial_free = manager.engine._resources["kv"]._arena.allocator.num_free
    extension.busy = True
    waiting, output = execute(manager, [1.])
    assert isinstance(waiting.admit_error, ExecutionBusy)
    assert output.per_rid_outputs == {}
    extension.busy = False
    unsupported, _ = execute(manager, [1., 2., 3., 4., 5.])
    assert isinstance(unsupported.admit_error, AdmitRuntimeError)
    assert shell.declarations == 0
    assert extension.acquires == extension.releases == 0
    assert extension.calls == []
    assert manager.engine._resources["kv"]._arena.allocator.num_free == initial_free
    assert manager.engine._resources["position"].position(7, "main") == 0
    assert manager.engine._resources["sampler"]._sampler._step_offset[7] == 0
    _, output = execute(manager, [1.])
    assert output.per_rid_outputs[7]["token"][0].item() == 3
    manager.remove_request(7)
    manager.engine.shutdown()


def test_resource_rejection_releases_selected_workspace_without_running_program():
    manager, model, extension = build(pages=2)  # sink plus one usable page
    execute(manager, [1., 2., 3., 4.])
    failed, _ = execute(manager, [5.])
    assert failed.failed_resource == "kv"
    assert len(extension.calls) == 1
    assert extension.acquires == extension.releases == 2
    assert manager.engine._resources["position"].position(7, "main") == 4
    assert manager.engine._resources["sampler"]._sampler._step_offset[7] == 1
    manager.remove_request(7)
    manager.engine.shutdown()


def test_unique_best_available_execution_ignores_ties_in_unused_cost_tier():
    options = []
    for name, cost in (("first", 1), ("second", 2), ("third", 2)):
        owner = Extension()
        options.append(ExecutionOption(name, lambda _: Support.YES,
                                       lambda facts, owner=owner: Lease(owner), lambda *args: None,
                                       lambda *args: None,
                                       cost=lambda facts, cost=cost: cost))
    caps = ExecutionCapabilities(tuple(options), {"decode": 1})
    selected = caps.reserve(InvocationFacts("decode", 1, (1,)))
    assert selected.option.name == "first"
    selected.close()


def test_fresh_construction_separates_mutable_state_between_engines():
    model = SmallModel()
    engines = []
    for _ in range(2):
        manager = EngineManager()
        manager.load({"node"}, CPU, {"node_groups": [{"node_names": ["node"]}]}, WorkerParallelGroups(1, 0),
                     TransferEngineInfo("worker", "test", LocalTransferEngine("localhost")), model, None,
                     runtime_extension=Extension())
        engines.append(manager.engine)
    assert model.created[0] is not model.created[1]
    assert tuple(model.created[0].parameters()) == tuple(model.created[1].parameters()) == ()
    assert engines[0]._resources["kv"] is not engines[1]._resources["kv"]
    assert (
        engines[0]._resources["kv"].kv_cache.tensor.data_ptr()
        != engines[1]._resources["kv"].kv_cache.tensor.data_ptr()
    )
    assert model.weight_reads == 2
    for engine in engines:
        engine.shutdown()


def test_shutdown_releases_runtime_storage_while_closed_engine_remains_referenced():
    import gc
    import weakref

    manager, model, extension = build()
    storage_owner = weakref.ref(manager.engine._resources["kv"].kv_cache.tensor)
    manager.remove_request(7)
    manager.shutdown()
    gc.collect()
    assert storage_owner() is None
    assert model.created[0].node_resources == {}
    assert model.created[0].binding is None


def test_bound_program_owns_precision_and_restores_ambient_autocast():
    model, extension = SmallModel(), Extension()
    manager = EngineManager()
    manager.load({"node"}, CPU,
                 {"autocast_dtype": "bfloat16", "node_groups": [{"node_names": ["node"]}]},
                                  WorkerParallelGroups(1, 0),
                 TransferEngineInfo("worker", "test", LocalTransferEngine("localhost")), model, None,
                                  runtime_extension=extension)
    manager.add_request(7, {"sampler": SamplingReqConfig(temperature=0)})
    with torch.autocast("cpu", dtype=torch.bfloat16):
        _, output = execute(manager, [1.])
        assert torch.is_autocast_enabled("cpu")
    assert output.per_rid_outputs[7]["token"][0].item() == 3
    manager.remove_request(7)
    manager.shutdown()


def test_batch_builder_discovers_exact_two_even_though_singleton_is_unsupported():
    from mstar.worker.batch_builder import BatchBuildRequest, FIFOBatchBuilder
    from mstar.worker.micro_scheduler import ScheduledBatch

    option = ExecutionOption("exact-two", lambda facts: Support.YES if facts.batch_size == 2 else Support.NO,
                             lambda facts: None, lambda *args: None, lambda *args: None)
    capabilities = ExecutionCapabilities((option,), {"decode": 2})
    request = BatchBuildRequest("node", "decode",
                                fresh=ScheduledBatch("node", "decode", {7: 0, 8: 0, 9: 0}),
                                execution_capabilities=capabilities)
    result = FIFOBatchBuilder().build_batch(request)
    assert list(result.scheduled.request_to_worker_graph) == [7, 8]
    assert list(result.backlog.request_to_worker_graph) == [9]
    # A lone row remains parked. Admission policy must provide singleton
    # coverage if arbitrary individual requests are admitted by deployment.
    waiting = FIFOBatchBuilder().build_batch(BatchBuildRequest(
        "node", "decode", backlog=result.backlog, execution_capabilities=capabilities,
    ))
    assert waiting.scheduled is None
    assert list(waiting.backlog.request_to_worker_graph) == [9]


def test_policy_admission_uses_canonical_rows_without_inventing_a_walk_or_batch():
    manager, model, extension = build()
    seen = []

    def supports(facts):
        seen.append(facts)
        assert facts.graph_walk is facts.batch_size is facts.token_lengths is facts.row_walks is None
        policy, = facts.resource_facts["sampler"]
        return Support.NO if policy["temperature"] else Support.UNKNOWN

    def no_effect(*args):
        pytest.fail("policy admission must not acquire workspace or run code")

    model.created[0].execution_capabilities = ExecutionCapabilities((ExecutionOption(
        "greedy", supports, no_effect, no_effect, no_effect,
    ),), {"decode": 1})
    manager.add_request(8, {"sampler": SamplingReqConfig(temperature=1e-8, top_k=7)})
    assert seen[0].resource_facts["sampler"][0]["top_k"] == 0
    with pytest.raises(UnsupportedInvocation, match="request policy"):
        manager.add_request(9, {"sampler": SamplingReqConfig(temperature=.7)})
    sampler = manager.engine._resources["sampler"]._sampler
    assert 9 not in sampler._sampling_config
    assert sampler._step_offset[8] == 0
    manager.remove_request(7)
    manager.remove_request(8)
    manager.shutdown()


def test_partial_load_remains_owned_for_worker_drain_then_program_close():
    from types import SimpleNamespace

    from mstar.worker.worker import Worker

    events = []

    class TwoNodes(SmallModel):
        def get_node_resources(self):
            return []

        def get_graph_walk_graphs(self):
            return {name: GraphNode(name, {"x"}, []) for name in ("first", "second")}

        def create_submodule(self, name):
            shell = Shell()
            self.created.append(shell)
            return shell

    class FailingExtension:
        def assemble(self, binding):
            def bind(resources):
                if binding.node_name == "second":
                    raise ValueError("second program cannot prepare")
                return ExecutionCapabilities((ExecutionOption(
                    "first", lambda facts: Support.YES, lambda facts: None,
                    lambda *args: None, lambda *args: None,
                    close=lambda: events.append("close:first"),
                ),), {"first": 1})
            return NodeAssembly(bind)

    manager = EngineManager()
    model = TwoNodes()
    with pytest.raises(ValueError, match="second program"):
        manager.load(("first", "second"), CPU,
                     {"node_groups": [{"node_names": ["first", "second"]}]}, WorkerParallelGroups(1, 0),
                     TransferEngineInfo("worker", "test", LocalTransferEngine("localhost")), model, None,
                     runtime_extension=FailingExtension())
    assert manager.engine is not None
    assert model.created[0].execution_capabilities is not None
    worker = Worker.__new__(Worker)
    worker.engine_manager = manager
    worker.runtime_extension = SimpleNamespace(drain=lambda: events.append("drain"))
    worker.close()
    assert events == ["drain", "close:first"]
    assert all(shell.binding is None for shell in model.created)
    assert manager.engine._submodules == {}
