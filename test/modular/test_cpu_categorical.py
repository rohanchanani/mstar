"""Physical CPU sampling and actual owner-commit controls, not a model oracle."""
import pytest
import torch

from mstar.engine.resources.sampler.config import SamplerStep, SamplingReqConfig
from mstar.engine.resources.sampler.cpu import _uniform, categorical_token
from mstar.engine.resources.sampler.plan import SamplingResult
from mstar.engine.resources.sampler.resource import SamplerResource
from mstar.engine.resources.step import StepContext


def test_local_uniform_replays_owner_offsets_without_global_rng_mutation():
    global_state = torch.random.get_rng_state().clone()
    generator = torch.Generator().manual_seed((1 << 62) + 73)
    expected = torch.rand(449, generator=generator, dtype=torch.float64)
    for offset in (0, 1, 16, 224, 448):
        assert _uniform((1 << 62) + 73, offset) == expected[offset]
        assert _uniform((1 << 62) + 73, offset) == _uniform((1 << 62) + 73, offset)
    assert torch.equal(global_state, torch.random.get_rng_state())


@pytest.mark.parametrize("kept", [False, True])
def test_actual_sampling_owner_alone_commits_retained_rows(kept):
    owner = SamplerResource(7, True, torch.device("cpu"), planned=True)
    owner.ingest_request(17, SamplingReqConfig(temperature=1., _seed=37))
    context = StepContext([17], "decode", slot=0, capture=False)
    step = SamplerStep(apply_penalty=False, kept_rids=frozenset({17}) if kept else frozenset())
    plan = owner.plan(step, context)
    logits = torch.tensor([-float("inf"), 0., 0., -float("inf"), 0., 0., -float("inf")])
    original = logits.clone()
    token = categorical_token(logits, plan)
    assert token.dtype == torch.int64 and token.shape == (1,)
    # Four equal-probability choices, with masked prefix/interior/tail entries.
    expected = (1, 2, 4, 5)[int(_uniform(37, 0) * 4)]
    assert token.item() == expected
    assert torch.equal(token, categorical_token(logits, plan))
    assert torch.equal(logits, original) and plan.offsets == (0,)
    assert owner._sampler._step_offset[17] == 0
    owner.accept_result(SamplingResult(token), context)
    assert owner._sampler._step_offset[17] == 0
    owner.commit(step, context)
    assert owner._sampler._step_offset[17] == int(kept)
    assert plan.offsets == (0,)
    logits.zero_()
    assert token.item() == expected
    owner.remove_request(17)


def test_categorical_rejects_invalid_rows_without_fallback():
    owner = SamplerResource(2, True, torch.device("cpu"), planned=True)
    owner.ingest_request(1, SamplingReqConfig(temperature=1., _seed=0))
    context = StepContext([1], "decode", slot=0, capture=False)
    plan = owner.plan(SamplerStep(apply_penalty=False), context)
    for logits in (torch.tensor([float("nan"), 0.]), torch.tensor([-float("inf"), -float("inf")])):
        with pytest.raises(ValueError, match="finite choice"):
            categorical_token(logits, plan)
    assert owner._sampler._step_offset[1] == 0
    owner.remove_request(1)
