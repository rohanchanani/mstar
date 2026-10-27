"""Per-request eager fallback admits all rows before executing any of them."""
import pytest
import torch

from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.engine import Engine, ExecutingBatch, SubmoduleManagement
from mstar.engine.resources.base import Resource
from mstar.engine.resources.runner import StepRunner
from mstar.engine.resources.step import (
    ADMIT_OK,
    AdmitFailedReason,
    AdmitOutcome,
    ResourceStep,
    StepContext,
    SubmoduleStep,
)
from mstar.model.submodule_base import NodeInputs, NodeSubmodule


class Reservation(Resource):
    def __init__(self, failure, events):
        self.failure, self.events = failure, events
        self.live = set()
        self.error = ValueError("last-row admission failed")

    @classmethod
    def build(cls, spec, info):
        raise NotImplementedError

    def admit(self, step, ctx):
        rid = ctx.request_ids[0]
        self.live.add(rid)
        self.events.append(("admit", rid))
        if rid == 3:
            if self.failure == "raise":
                raise self.error
            if self.failure == "refuse":
                return AdmitOutcome(False, reason=AdmitFailedReason("controlled refusal"))
        return ADMIT_OK

    def rollback_admit(self, step, ctx):
        rid = ctx.request_ids[0]
        self.events.append(("rollback", rid))
        self.live.remove(rid)  # catches duplicate rollback

    def commit(self, step, ctx):
        rid = ctx.request_ids[0]
        self.events.append(("commit", rid))
        self.live.remove(rid)


class Shell(NodeSubmodule):
    def __init__(self, failure, events):
        super().__init__()
        self.failure, self.events = failure, events
        self.error = ValueError("last-row declaration failed")

    def prepare_inputs(self, *args, **kwargs):
        raise AssertionError("rows already prepared")

    def declare_step(self, graph_walk, request_ids, **kwargs):
        if request_ids == [3] and self.failure == "declare":
            raise self.error
        return SubmoduleStep({"reserve": ResourceStep()})

    def preprocess(self, graph_walk, engine_inputs, inputs):
        return {"x": inputs[0].tensor_inputs["x"]}

    def forward(self, graph_walk, engine_inputs, x):
        self.events.append(("forward", engine_inputs.request_ids[0]))
        return {"y": [x + 1]}


def run_case(failure):
    events = []
    resource = Reservation(failure, events)
    shell = Shell(failure, events)
    engine = Engine(None)
    engine._device = torch.device("cpu")
    engine._runner = StepRunner({"reserve": resource})
    engine._submodules = {"node": SubmoduleManagement(
        shell, shell.forward, shell.forward_batched, None, {"reserve": resource},
    )}
    batch = ExecutingBatch(
        node_name="node", per_request_info={rid: CurrentForwardPassInfo(
            request_id=f"r{rid}", rid_handle=rid, graph_walk="decode", fwd_index=0, random_seed=0, max_tokens=1,
        ) for rid in (1, 2, 3)},
        step_context=StepContext(request_ids=(1, 2, 3), graph_walk="decode", slot=0, capture=False),
        inputs=[NodeInputs(input_seq_len=1, tensor_inputs={"x": torch.tensor([rid])}) for rid in (1, 2, 3)],
    )
    return engine, batch, resource, shell, events


@pytest.mark.parametrize("failure", ["refuse", "raise", "declare"])
def test_preflight_failure_unwinds_prior_rows_once_in_reverse_order(failure):
    engine, batch, resource, shell, events = run_case(failure)
    if failure == "refuse":
        engine._exec_per_request(batch)
    else:
        with pytest.raises(ValueError) as caught:
            engine._exec_per_request(batch)
        assert caught.value is (shell.error if failure == "declare" else resource.error)
    assert not resource.live
    assert [rid for event, rid in events if event == "rollback"] == (
        [2, 1] if failure == "declare" else [3, 2, 1]
    )
    assert not any(event in ("forward", "commit") for event, _ in events)


def test_success_keeps_admit_all_then_plan_forward_commit_order():
    engine, batch, resource, shell, events = run_case(None)
    outputs = engine._exec_per_request(batch)
    assert events == [("admit", 1), ("admit", 2), ("admit", 3),
                      ("forward", 1), ("commit", 1), ("forward", 2), ("commit", 2),
                      ("forward", 3), ("commit", 3)]
    assert not resource.live
    for rid in (1, 2, 3):
        assert outputs.per_rid_outputs[rid]["y"][0].item() == rid + 1
