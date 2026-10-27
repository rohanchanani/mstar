"""External computations obey the existing sampler policy and completion owner."""

from dataclasses import replace

import pytest
import torch
from test_sampler_generation_controls import RefRow, reference_probs

from mstar.engine.resources.sampler.config import SamplerStep, SamplingReqConfig, sampling_policy_facts
from mstar.engine.resources.sampler.cpu import can_use_native_greedy, categorical_token, probabilities, sample
from mstar.engine.resources.sampler.plan import SamplingResult
from mstar.engine.resources.sampler.resource import SamplerResource
from mstar.engine.resources.step import StepContext


def owner(configs, **kw):
    r = SamplerResource(32, True, torch.device("cpu"), planned=True, **kw)
    for i, c in enumerate(configs):
        r.ingest_request(i, c)
    return r


def invocation(r, n=1, **kw):
    ctx = StepContext(request_ids=tuple(range(n)), graph_walk="decode", slot=0, capture=False)
    step = SamplerStep(**kw)
    return step, ctx, r.plan(step, ctx)


@pytest.mark.parametrize("temperature", [0.3, 0.8, 1.0])
def test_existing_seeded_cpu_stream(temperature):
    # Independent frozen original algorithm, including double precision and zero plateaus.
    c = SamplingReqConfig(temperature=temperature, _seed=993)
    r = owner([c])
    step, ctx, plan = invocation(r)
    x = torch.linspace(-8, 8, 32)
    x[:7] = -float("inf")
    for offset in range(12):
        plan = replace(plan, offsets=(offset,))
        p = torch.softmax((x.double() - x.double().max()) / temperature, 0)
        cdf = p.cumsum(0)
        u = torch.rand(offset + 1, generator=torch.Generator().manual_seed(993), dtype=torch.float64)[-1]
        target = torch.minimum(u * cdf[-1], torch.nextafter(cdf[-1], torch.zeros_like(cdf[-1])))
        want = torch.searchsorted(cdf, target, right=True).reshape(1)
        assert torch.equal(categorical_token(x, plan), want)
        assert torch.equal(sample(x[None], plan), want)
    assert r._sampler._step_offset[0] == 0


def test_external_commit_history_kept_rows_reset_and_once():
    configs = [SamplingReqConfig(temperature=0.8, repetition_window=3, repetition_penalty=1.2, min_tokens=2)] * 2
    r = owner(configs, max_repetition_window=3, min_tokens_stop_ids=(31,))
    step, ctx, plan = invocation(r, 2, kept_rids=frozenset({1}))
    assert plan.history.count.tolist() == [0, 0]
    r.accept_result(SamplingResult(torch.tensor([2, 3])), ctx)
    assert r._sampler._step_offset == {0: 0, 1: 0}
    r.commit(step, ctx)
    assert r._sampler._step_offset == {0: 0, 1: 1}
    with pytest.raises(RuntimeError, match="requires this invocation"):
        r.commit(step, ctx)
    plan = r.plan(step, ctx)
    assert plan.history.count.tolist() == [0, 1]
    assert plan.history.tokens.tolist() == [[-1] * 3, [3, -1, -1]]
    r.remove_request(1)
    r.ingest_request(1, configs[1])
    plan = r.plan(step, ctx)
    assert plan.history.count.tolist() == [0, 0]
    assert plan.offsets == (0, 0)


@pytest.mark.parametrize("first", [False, True])
@pytest.mark.parametrize("greedy", [False, True])
def test_generation_distribution_matches_upstream_reference(first, greedy):
    row = RefRow(
        temperature=0 if greedy else 0.8,
        window=3,
        penalty=1.3,
        min_tokens=2,
        top_k=8,
        top_p=0.8,
        min_keep=1,
        top_p_first=first,
    )
    r = owner([row.config()], max_repetition_window=3, min_tokens_stop_ids=(31,), enable_top_p_first=True)
    x = torch.linspace(-2, 2, 32).reshape(1, -1)
    for token in (30, 29, 28, 27):
        step, ctx, plan = invocation(r)
        got = probabilities(x, plan)[0]
        want = reference_probs(x[0], row, stop_ids=(31,))
        torch.testing.assert_close(got.float(), want, atol=1e-6, rtol=1e-5)
        r.accept_result(SamplingResult(torch.tensor([token])), ctx)
        r.commit(step, ctx)
        row.generated.append(token)


def test_filter_bypass_and_native_greedy_eligibility():
    r = owner([SamplingReqConfig(temperature=0.8, top_k=1, top_p=0.5)])
    _, _, filtered = invocation(r)
    _, _, bypass = invocation(r, apply_filters=False)
    x = torch.arange(32).float()[None]
    assert (probabilities(x, filtered) > 0).sum() == 1
    assert (probabilities(x, bypass) > 0).sum() == 32
    g = owner([SamplingReqConfig(temperature=0)])
    assert can_use_native_greedy(invocation(g)[2])
    g = owner([SamplingReqConfig(temperature=0, min_tokens=3)], min_tokens_stop_ids=(31,))
    assert not can_use_native_greedy(invocation(g)[2])


def test_policy_facts_include_upstream_controls():
    f = sampling_policy_facts(SamplingReqConfig(repetition_window=3, min_tokens=5, top_p_first=True, top_p_min_keep=2))
    assert {k: f[k] for k in ("repetition_window", "min_tokens", "top_p_first", "top_p_min_keep")} == dict(
        repetition_window=3, min_tokens=5, top_p_first=True, top_p_min_keep=2
    )


def test_repeated_eager_draws_complete_each_time(monkeypatch):
    from mstar.engine.resources.sampler import utils

    r = SamplerResource(32, True, torch.device("cpu"))
    r.ingest_request(0, SamplingReqConfig(temperature=0.8))
    offsets = []

    def draw(**kw):
        offsets.append(kw["rand_offset"].clone())
        return torch.tensor([2])

    monkeypatch.setattr(utils, "sample_tokens", draw)
    ctx = StepContext(request_ids=(0,), graph_walk="decode", slot=0, capture=False)
    step = SamplerStep()
    r.plan(step, ctx)
    for _ in range(16):
        r.sample([0], torch.ones(1, 32))
    r.commit(step, ctx)
    assert [int(x[0]) for x in offsets] == list(range(16))
    assert r._sampler._step_offset[0] == 16
