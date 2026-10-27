"""Physical runners share upstream's budget and retain their execution policy."""
from types import SimpleNamespace

import pytest
import torch

from mstar.engine.accelerator_graph_config import BatchedAcceleratorGraphConfig
from mstar.engine.accelerator_graph_runner import AcceleratorGraphRunner, CaptureCost
from mstar.engine.engine import Engine, SubmoduleManagement
from mstar.engine.execution import ExecutionCapabilities
from mstar.engine.fp32_precision import Fp32Precision
from mstar.engine.resources.step import BucketKey
from mstar.engine.tensor_forward import CapturePolicy, CompilePolicy, TensorForward, TensorForwardRunner
from mstar.model.submodule_base import NodeInputs, NodeSubmodule
from mstar.utils.device import DeviceCapability


class Shell(NodeSubmodule):
    def forward(self, *args, **kwargs):
        raise AssertionError("logical forward must not execute")

    def prepare_inputs(self, *args, **kwargs):
        raise AssertionError("not a request test")


class Program(torch.nn.Module):
    def forward(self, x):
        return x + 1

    forward_batched = forward

    def get_accelerator_graph_configs(self, device, tp_world_size):
        return [BatchedAcceleratorGraphConfig(
            "decode", NodeInputs(input_seq_len=1), capture_batch_sizes=[1, 2, 4, 8],
        )]


def adapter(name, shell, *, required=False):
    rank = SimpleNamespace(world_size=1)
    binding = SimpleNamespace(submodule=shell, node_name="bound", device=torch.device("cpu"),
                              tp_group=rank, sp_group=rank)
    return TensorForwardRunner(name, binding, {}, TensorForward(
        Program(), autocast_dtype=torch.bfloat16,
        capture=CapturePolicy(enabled=True, required=required, compile=CompilePolicy(
            enabled=True, mode="max-autotune-no-cudagraphs", dynamic=False,
        )),
    ))


def test_physical_config_namespace_limits_and_compile_policy(monkeypatch):
    shell = Shell()
    runner = adapter("fast", shell)
    runner.configure(SimpleNamespace(), num_slots=2, batch_limits={"decode": 3})
    graph = runner.graph_runner
    assert graph._submodule is shell
    assert graph._program is runner.program
    assert graph._autocast_dtype is torch.bfloat16
    specs = graph._get_slot_specs()
    assert {s.bs for s in specs} == {1, 2, 4}  # padding ceiling retained
    assert {s.slot for s in specs} == {0, 1}
    assert all(s.bucket.execution_name == "fast" for s in specs)
    calls = []
    monkeypatch.setattr(torch, "compile", lambda fn, **kwargs: calls.append(kwargs) or fn)
    graph._forward_for(specs[0])
    graph._forward_for(specs[1])
    assert calls == [{"mode": "max-autotune-no-cudagraphs", "fullgraph": False, "dynamic": False}]


@pytest.mark.parametrize("planned", [None, set(), {BucketKey(graph_walk="decode", bs=1, num_tokens=1)}])
def test_adapter_forwards_plan_without_substitution(planned):
    runner = adapter("fast", Shell())
    calls = []
    runner.graph_runner = SimpleNamespace(warmup_and_capture=calls.append)
    runner.warmup_and_capture(planned)
    assert calls[0] is planned
    assert runner._warmed


def test_required_capture_cannot_silently_accept_budget_exclusion():
    runner = adapter("fast", Shell(), required=True)
    bucket = BucketKey(graph_walk="decode", bs=1, num_tokens=1)
    runner.graph_runner = SimpleNamespace(
        warmup_and_capture=lambda plan: None, any_graphs=True, dropped_buckets=[bucket],
    )
    with pytest.raises(RuntimeError, match="required accelerator graph capture failed"):
        runner.warmup_and_capture(set())


def test_planner_accounts_for_each_physical_pool_with_node_precision(monkeypatch):
    bucket = BucketKey(graph_walk="decode", bs=1, num_tokens=1)
    calls = []

    class Sizer:
        def __init__(self, name, peak):
            self.name, self.peak = name, peak

        def size_captures(self):
            calls.append((self.name, torch.get_float32_matmul_precision()))
            return {bucket: CaptureCost(self.peak, 2**20, 0, 1)}

        def size_lent_captures(self, wanted):
            return self.size_captures()

    original = torch.get_float32_matmul_precision()
    engine = SimpleNamespace(
        _device=torch.device("cuda"), _gpu_memory_fraction=None,
        _submodules={"bound": SimpleNamespace(fp32_precision=Fp32Precision("highest", None))},
    )
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (2**30, 2**31))
    plan = Engine._plan_captures(engine, {"bound": Sizer("ordinary", 100)}, {}, {"bound": {
        "first": SimpleNamespace(graph_runner=Sizer("first", 100)),
        "second": SimpleNamespace(graph_runner=Sizer("second", 200)),
        "uncaptured": SimpleNamespace(graph_runner=None),
    }})
    assert plan.buckets["bound", "first"] == {bucket}
    assert plan.buckets["bound", "second"] == {bucket}
    assert ("bound", "uncaptured") not in plan.buckets
    assert plan.floor == 200
    assert calls == [("ordinary", "highest"), ("first", "highest"), ("second", "highest")] * 2
    assert torch.get_float32_matmul_precision() == original


def test_engine_prepares_all_before_planning_and_forwards_each_selection(monkeypatch):
    shell = Shell()
    runners = [adapter(name, shell) for name in ("first", "second")]
    shell.execution_capabilities = ExecutionCapabilities(tuple(
        SimpleNamespace(name=runner.name, runner=runner) for runner in runners
    ) + (SimpleNamespace(name="callable", runner=None),), {"decode": 3})
    engine = Engine.__new__(Engine)
    engine._submodules = {"bound": SubmoduleManagement(
        shell, shell.forward, shell.forward_batched, SimpleNamespace(world_size=1), {},
        fp32_precision=Fp32Precision("highest", None),
    )}
    engine._runner = SimpleNamespace()
    engine._device = torch.device("cpu")
    engine._device_capability = DeviceCapability.of("cpu")
    engine._enable_nvtx = False
    engine._resources = {}
    events = []
    monkeypatch.setattr(AcceleratorGraphRunner, "prepare_for_capture",
                        lambda self: events.append(("prepare", self._execution_name)))
    monkeypatch.setattr(Engine, "_compile_submodules", lambda self: events.append(("compile", "nodes")))
    monkeypatch.setattr(TensorForwardRunner, "compile_eager", lambda self: events.append(("compile", self.name)))
    choices = {"first": set(), "second": {BucketKey(graph_walk="decode", bs=2, num_tokens=2, execution_name="second")}}

    def plan(self, ordinary, piecewise, bound):
        assert not ordinary and not piecewise
        assert events == [("prepare", "first"), ("prepare", "second")]
        assert list(bound["bound"]) == ["first", "second"]
        events.append(("plan", "all"))
        return SimpleNamespace(floor=0, buckets={("bound", name): chosen for name, chosen in choices.items()})

    from contextlib import contextmanager

    @contextmanager
    def capture_into(self, plan, pool):
        yield plan.buckets[pool]

    def capture(self, chosen):
        assert chosen is choices[self._execution_name]
        assert torch.get_float32_matmul_precision() == "highest"
        events.append(("capture", self._execution_name))

    monkeypatch.setattr(Engine, "_plan_captures", plan)
    monkeypatch.setattr(Engine, "_capture_into", capture_into)
    monkeypatch.setattr(AcceleratorGraphRunner, "warmup_and_capture", capture)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (2**30, 2**31))
    engine.warmup()
    assert events == [("prepare", "first"), ("prepare", "second"), ("plan", "all"),
                      ("capture", "first"), ("capture", "second"),
                      ("compile", "nodes"), ("compile", "first"), ("compile", "second")]
    # A small capture ceiling does not redefine supported invocations.
    assert engine.get_max_batch_size("bound", "decode") == 3
