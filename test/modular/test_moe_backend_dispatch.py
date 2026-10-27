"""Backend dispatch preserves upstream masks and explicit launch configuration."""
from types import SimpleNamespace

import pytest
import torch

from mstar.model.components import moe
from mstar.model.components.moe_parallel import MoeParallelConfig, instantiate_moe_block
from mstar.utils.fused_moe import runner


@pytest.mark.parametrize('skip', [False, True])
def test_selected_backend_receives_only_required_arguments(monkeypatch, skip):
    calls = []
    monkeypatch.setattr(moe, '_HAS_FUSED', True)
    x = SimpleNamespace(is_cuda=True)
    w1, w2, weights, ids = object(), object(), object(), object()
    result = object()
    def backend(*args, **kwargs):
        calls.append((args, kwargs))
        return result
    assert moe._dispatch(x, w1, w2, 4, ids, weights, backend, skip_invalid=skip) is result
    assert calls == [((x, w1, w2, weights, ids), {'skip_invalid': True} if skip else {})]


def test_default_dispatch_retains_upstream_mask_flag(monkeypatch):
    calls = []
    monkeypatch.setattr(moe, '_HAS_FUSED', True)
    monkeypatch.setattr(moe, '_fused_experts', lambda *args, **kw: calls.append(kw))
    moe._dispatch(SimpleNamespace(is_cuda=True), None, None, 4, None, None, skip_invalid=True)
    assert calls == [{'skip_invalid': True}]


@pytest.mark.parametrize('mask', [False, True])
def test_factory_forwards_backend_and_checks_mask_support(monkeypatch, mask):
    seen = []
    def make(kind, problem):
        seen.append(kind)
        return SimpleNamespace(supports_invalid_experts=kind == 'triton')
    monkeypatch.setattr(moe, 'make_moe_backend', make)
    args = dict(hidden_size=8, num_experts=4, num_experts_per_tok=2, moe_intermediate_size=8,
                moe_parallel=MoeParallelConfig(mask_padding=mask))
    block = instantiate_moe_block(**args, moe_backend='triton')
    assert block.experts_backend.supports_invalid_experts and seen == ['triton']
    if mask:
        with pytest.raises(ValueError, match='mask_padding'):
            instantiate_moe_block(**args, moe_backend='aiter')
    else:
        instantiate_moe_block(**args, moe_backend='aiter')


@pytest.mark.parametrize('explicit', [False, True])
def test_runner_preserves_configuration_and_alignment(monkeypatch, explicit):
    up = {'BLOCK_SIZE_M': 4, 'BLOCK_SIZE_K': 8}
    down = {**up, 'BLOCK_SIZE_K': 4}
    chosen = {**up, 'num_stages': 2}
    monkeypatch.setattr(runner, 'get_config', lambda **kw: (up, down))
    monkeypatch.setattr(runner, '_tl_compute_type', lambda dtype: dtype)
    monkeypatch.setattr(runner, 'act_and_mul_triton', lambda *a, **k: None)
    seen = []
    monkeypatch.setattr(runner, 'invoke_fused_moe_kernel', lambda **kw: seen.append(kw))
    # Precomputed alignment must be reused, not recomputed with another tile.
    monkeypatch.setattr(runner, 'moe_align_block_size', lambda *a: pytest.fail('alignment recomputed'))
    alignment = runner.MoeAlignment(torch.arange(8), torch.tensor([0, 1]), torch.tensor([8]), 4)
    x = torch.zeros(2, 8, dtype=torch.bfloat16)
    w1 = torch.zeros(4, 16, 8, dtype=x.dtype)
    w2 = torch.zeros(4, 8, 8, dtype=x.dtype)
    ids = torch.tensor([[0, 1], [4, 4]])
    result = runner.fused_experts(x, w1, w2, torch.ones(2, 2), ids,
                                 reduce_results=False, skip_invalid=True, alignment=alignment,
                                 config=chosen if explicit else None)
    assert torch.count_nonzero(result) == 0  # skipped slots are initialized, never uninitialized
    assert [k['config'] for k in seen] == ([chosen, chosen] if explicit else [up, down])
    assert all(k['sorted_token_ids'] is alignment.sorted_token_ids for k in seen)
    with pytest.raises(AssertionError, match='BLOCK_SIZE_M'):
        runner.fused_experts(x, w1, w2, torch.ones(2, 2), ids,
                            alignment=alignment._replace(block_m=16), config=chosen)
