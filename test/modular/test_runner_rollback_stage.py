"""Unwound reservations cannot authorize promotion or bypass re-admission."""
import pytest
import torch

from mstar.engine.resources.base import Resource
from mstar.engine.resources.runner import StepRunner
from mstar.engine.resources.sampler.config import SamplerStep, SamplingReqConfig
from mstar.engine.resources.sampler.plan import SamplingResult
from mstar.engine.resources.sampler.resource import SamplerResource
from mstar.engine.resources.step import (
    ADMIT_OK,
    AdmitFailedReason,
    AdmitOutcome,
    ResourceStep,
    SlotLease,
    StepContext,
    SubmoduleStep,
)


class Reservation(Resource):
    @classmethod
    def build(cls, spec, info):
        raise NotImplementedError

    def __init__(self, name, events, *, preplan=True, deps=()):
        self.name, self.events = name, events
        self.preplan, self.deps = preplan, set(deps)
        self.reserved = False
        self.failure = None

    @property
    def supports_preplan(self):
        return self.preplan

    def depends_on(self):
        return self.deps

    def admit(self, step, ctx):
        self.events.append((self.name, "admit"))
        if self.failure == "refuse":
            return AdmitOutcome(False, reason=AdmitFailedReason("controlled refusal"))
        self.reserved = True
        if self.failure == "raise":
            raise ValueError("controlled exception")
        return ADMIT_OK

    def rollback_admit(self, step, ctx):
        self.events.append((self.name, "rollback"))
        self.reserved = False

    def clear_preplan(self):
        self.events.append((self.name, "clear"))


def step(*, preplan=False):
    s = SubmoduleStep({"reserve": ResourceStep(), "later": ResourceStep()})
    s.set_ctx(StepContext(request_ids=(1,), graph_walk="decode", slot=1, capture=False,
                         is_preplan=preplan, slot_lease=SlotLease(1, None)))
    return s


@pytest.mark.parametrize("failure", ["refuse", "raise", "rollback"])
def test_canceled_stage_reacquires_reservations_on_identical_retry(failure):
    events = []
    reserve = Reservation("reserve", events)
    later = Reservation("later", events, preplan=False, deps=("reserve",))
    runner = StepRunner({"reserve": reserve, "later": later})
    staged = step(preplan=True)
    assert runner.pre_admit(staged).ok
    runner.pre_plan(staged)
    later.failure = failure
    if failure == "raise":
        with pytest.raises(ValueError, match="controlled exception"):
            runner.admit(step())
    elif failure == "rollback":
        assert runner.admit(step()).ok
        runner.rollback_admit(step())
    else:
        assert not runner.admit(step()).ok
    assert not reserve.reserved and not later.reserved
    assert not runner.staged and runner._pre_admitted is None
    assert [e for e in events if e[1] == "rollback"] == [("later", "rollback"), ("reserve", "rollback")]
    later.failure = None
    assert runner.admit(step()).ok
    assert reserve.reserved and later.reserved
    assert events.count(("reserve", "admit")) == 2


def test_pre_admit_refusal_drops_previous_stage():
    events = []
    reserve = Reservation("reserve", events)
    later = Reservation("later", events, deps=("reserve",))
    runner = StepRunner({"reserve": reserve, "later": later})
    staged = step(preplan=True)
    assert runner.pre_admit(staged).ok
    runner.pre_plan(staged)
    later.failure = "refuse"
    assert not runner.pre_admit(step(preplan=True)).ok
    assert not runner.staged and runner._pre_admitted is None
    assert not reserve.reserved and not later.reserved


def test_physical_sampler_result_reaches_owner_and_commits_once():
    sampler = SamplerResource(32, False, torch.device("cpu"), planned=True)
    for rid in (10, 20):
        sampler.ingest_request(rid, SamplingReqConfig(temperature=0))
    runner = StepRunner({"sample": sampler})
    ctx = StepContext(request_ids=(20, 10), graph_walk="decode", slot=0, capture=False)
    s = SubmoduleStep({"sample": SamplerStep(kept_rids=frozenset({10}))})
    s.set_ctx(ctx)
    plan = runner.plan(s)["sample"]
    assert plan.offsets == (0, 0)
    result = SamplingResult(torch.tensor([7, 3]))
    with pytest.raises(KeyError):
        runner.accept_results(s, {"undeclared": result})
    runner.accept_results(s, {"sample": result})
    assert sampler._sampler._step_offset == {10: 0, 20: 0}
    runner.commit(s)
    assert sampler._sampler._step_offset == {10: 1, 20: 0}
    with pytest.raises(RuntimeError, match="requires this invocation"):
        runner.commit(s)
