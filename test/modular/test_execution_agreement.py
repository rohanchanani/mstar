"""Real CPU process groups exercise common choice and reversible admission."""
import json
import socket
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.multiprocessing as mp

from mstar.communication.tensors import LocalTransferEngine
from mstar.distributed.communication import CommGroup, WorkerParallelGroups
from mstar.distributed.execution import ExecutionAgreement, InstanceFatalError
from mstar.engine.engine import Engine
from mstar.engine.execution import ExecutionCapabilities, ExecutionOption, InvocationFacts, Support
from mstar.engine.resources.base import EngineResourceInfo, Resource
from mstar.engine.resources.binding import build_bound_resource, resolve_bound_specs
from mstar.engine.resources.kv.config import KVSpec, KVStep, SequenceKVConfig
from mstar.engine.resources.kv.transfer import TransferEngineInfo
from mstar.engine.resources.runner import StepRunner
from mstar.engine.resources.sequence import DenseSequenceView, SequenceBudget
from mstar.engine.resources.step import (
    ADMIT_OK,
    FULL_ADMIT_OK,
    AdmitOutcome,
    AllocationFailed,
    ResourceStep,
    Segment,
    StepContext,
    SubmoduleStep,
)
from mstar.model.binding import NodeAssembly


class Lease:
    def __init__(self, released, name):
        self.released, self.name = released, name

    def close(self):
        self.released.append(self.name)


class RefusingResource(Resource):
    @classmethod
    def build(cls, spec, info):
        return cls()

    def depends_on(self):
        return {"kv"}

    def admit(self, step, ctx):
        return ADMIT_OK if self.rank == 0 else AdmitOutcome(False, reason=AllocationFailed(
            "controlled capacity refusal", 1, "main", ctx.request_ids[0],
        ))


def _spec():
    return KVSpec("kv", {"decoder"}, SequenceKVConfig(
        num_layers=1, num_kv_heads=2, head_dim=2, max_seq_len=4,
    ), budget=SequenceBudget(3 * 2 * 4 * 2 * 2 * 4))


def _participant(rank, port, directory, fail_after_collective):
    torch.set_num_threads(1)
    groups = WorkerParallelGroups(2, rank, any_parallelism=True, world_parallel_groups=[(0, 1)], dist_timeout_s=10)
    groups.add("decoder", CommGroup(rank, rank, [0, 1]))
    groups.init_dist(f"tcp://127.0.0.1:{port}", torch.device("cpu"))
    joint = groups.get_joint_group_for_node("decoder")
    agreement = ExecutionAgreement(joint, "decoder")
    record = {}
    try:
        # Speculative preparation must not enter a conditional collective on
        # only the participant whose local runner is eligible.
        assert not agreement.all(rank == 0, "speculative eligibility")
        assert agreement.all(True, "shared speculative eligibility")
        if fail_after_collective:
            try:
                with agreement.executing():
                    value = torch.tensor([float(rank + 1)])
                    joint.tp_group.all_reduce(value)
                    assert value.item() == 3
                    if rank == 1:
                        raise RuntimeError("controlled failure after actual collective")
                    agreement.complete()
                    record["committed"] = True
            except InstanceFatalError:
                groups.mark_failed()
                record["fatal"] = True
            assert record == {"fatal": True}
            return

        released = []
        refuse = [False]

        def option(name, cost):
            return ExecutionOption(
                name, lambda _: Support.YES,
                lambda _: None if name == "allreduce" and rank == 1 and refuse[0] else Lease(released, name),
                lambda invocation, execution: invocation, lambda call, execution: call,
                cost=lambda _: cost,
            )

        # The local winners differ, but the common summed cost selects allreduce.
        caps = ExecutionCapabilities((option("allreduce", (0, 2)[rank]),
                                      option("gather", (4, 0)[rank])), {"decode": 1})
        facts = InvocationFacts("decode", 1, (1,))
        selected = agreement.reserve(caps, facts, ("request-a",))
        assert selected.option.name == "allreduce"
        selected.close()
        released.clear()
        refuse[0] = True
        selected = agreement.reserve(caps, facts, ("request-a",))
        assert selected.option.name == "gather"
        assert released == (["allreduce"] if rank == 0 else [])
        selected.close()

        # Distinct local handles must not corrupt the peer's typed refusal.
        rid = 17 + rank * 100
        view = DenseSequenceView()
        physical, = resolve_bound_specs((_spec(),), {"kv": (view,)}, dtype=torch.float32,
                                        group_sizes={"kv": 2})
        kv = build_bound_resource(physical, EngineResourceInfo(
            torch.device("cpu"), joint_comm_group=joint, kv_dtype=torch.float32,
            transfer_engine_info=TransferEngineInfo(f"rank-{rank}", "agreement-test", LocalTransferEngine("localhost")),
            needs_remote_transfer=False,
        ), (view,))
        kv.ingest_request(rid)
        guard = RefusingResource()
        guard.rank = rank
        runner = StepRunner({"kv": kv, "refusal": guard})
        step = SubmoduleStep({"kv": KVStep((Segment(rid, "main", 1),)), "refusal": ResourceStep()})
        step.set_ctx(StepContext((rid,), "decode", 0, False))
        outcome = agreement.admit(runner, step, ("request-a",))
        assert not outcome.ok and outcome.failed_resource == "refusal"
        assert outcome.reason.request_id == rid
        stream = kv._streams[rid]["main"]
        assert not stream.step_in_flight and stream.page_indices == [] and stream.stored_len == 0
        kv.remove_request(rid)
        kv.cleanup()
        record = {"choice": "allreduce", "peer_refusal_choice": "gather", "rollback": True}
    finally:
        groups.close()
        record["process_group_closed"] = (
            not torch.distributed.is_initialized() if not groups._failed else "failed domain retired"
        )
        Path(directory, f"rank-{rank}.json").write_text(json.dumps(record))


@pytest.mark.parametrize("fail_after_collective", [False, True])
def test_real_cpu_agreement_and_failure_retirement(tmp_path, fail_after_collective):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    mp.spawn(_participant, args=(port, str(tmp_path), fail_after_collective), nprocs=2, join=True)
    records = [json.loads((tmp_path / f"rank-{rank}.json").read_text()) for rank in range(2)]
    if fail_after_collective:
        assert records == [{"fatal": True, "process_group_closed": "failed domain retired"}] * 2
    else:
        assert records == [{"choice": "allreduce", "peer_refusal_choice": "gather", "rollback": True,
                            "process_group_closed": True}] * 2




def test_admission_exception_releases_successful_peers_and_is_instance_fatal():
    def gathered(value):
        if isinstance(value, tuple):
            return [value, (None, "RuntimeError: rollback failed")]
        return [value, None]

    rolled_back = []
    runner = SimpleNamespace(admit=lambda _: FULL_ADMIT_OK, rollback_admit=rolled_back.append)
    step = SimpleNamespace(ctx=SimpleNamespace(request_ids=(17,)))
    agreement = ExecutionAgreement(SimpleNamespace(all_gather_object=gathered), "decoder")
    with pytest.raises(InstanceFatalError, match="rollback failed"):
        agreement.admit(runner, step, ("request-a",))
    assert rolled_back == [step]


def test_binding_disagreement_precedes_resource_allocation():
    joint = SimpleNamespace(world_size=2, all_gather_object=lambda value: [value, ("binding", "different plan")])
    groups = SimpleNamespace(get_joint_group_for_node=lambda _: joint)
    engine = Engine(None)
    with pytest.raises(InstanceFatalError, match="binding before allocation"):
        engine.load_model({"decoder": object()}, [_spec()], groups, torch.device("cpu"), None,
                          kv_cache_type=torch.float32,
                          assemblies={"decoder": NodeAssembly(lambda _: None, execution_names=("book",))},
                          resource_requests={"kv": (DenseSequenceView(),)})
    assert not engine._resources
