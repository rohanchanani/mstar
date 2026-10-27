"""Arithmetic checks for the actual Triton kernels used by Qwen on AMD.

Run in a fresh process with TRITON_INTERPRET=1. The interpreter executes the
kernel bodies on CPU; it cannot validate HIP code generation, wave scheduling,
device memory ordering, or graph capture. GPU tests remain under test/rocm.
"""

import os

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(
    os.environ.get("TRITON_INTERPRET") != "1", reason="requires TRITON_INTERPRET=1",
)


@pytest.mark.parametrize("batch", [1, 2, 8])
def test_qk_norm_rope_strided_predictor_rows(batch):
    from mstar.utils.attention import fused_qk_norm_rope

    torch.manual_seed(31 + batch)
    heads, dim = 16, 128
    qkv = torch.randn(batch, 3 * heads * dim)
    x = qkv[:, :heads * dim].view(batch, heads, dim)
    untouched = qkv[:, heads * dim:].clone()
    original = x.clone()
    weight = torch.randn(dim)
    positions = torch.arange(batch, dtype=torch.int32) + 7
    eps, theta = 1e-6, 10000.0
    normed = original * torch.rsqrt(original.square().mean(-1, keepdim=True) + eps) * weight
    angle = positions.float()[:, None] / theta ** (torch.arange(0, dim, 2).float() / dim)
    cos, sin = angle.cos()[:, None], angle.sin()[:, None]
    first, second = normed.chunk(2, -1)
    expected = torch.cat((first * cos - second * sin, second * cos + first * sin), -1)
    got = fused_qk_norm_rope(x, weight, positions, eps, theta)
    torch.testing.assert_close(got, expected, rtol=2e-5, atol=2e-5)
    assert got.data_ptr() == x.data_ptr()
    assert torch.equal(qkv[:, heads * dim:], untouched)


@pytest.mark.parametrize("cache_length", [1, 15, 17])
def test_predictor_dense_decode_ignores_unwritten_tail(cache_length):
    from mstar.utils.attention import _decode_attn_nhd_kernel

    torch.manual_seed(20 + cache_length)
    batch, heads, kv_heads, dim = 2, 4, 2, 32
    q = torch.randn(batch, heads, dim)
    k = torch.full((batch, 32, kv_heads, dim), float("nan"))
    v = torch.full_like(k, float("nan"))
    k[:, :cache_length] = torch.randn(batch, cache_length, kv_heads, dim)
    v[:, :cache_length] = torch.randn(batch, cache_length, kv_heads, dim)
    got = torch.empty_like(q)
    # Bypass autotuning, which benchmarks on a device, but execute its real body.
    _decode_attn_nhd_kernel.fn[(batch, heads)](
        q, k, v, got, cache_length, dim ** -0.5,
        *q.stride(), *k.stride(), *v.stride(), *got.stride(),
        H_q=heads, H_kv=kv_heads, GROUP=heads // kv_heads, D=dim,
        BLOCK_N=16, num_warps=4, num_stages=2,
    )
    keys = k[:, :cache_length].transpose(1, 2).repeat_interleave(heads // kv_heads, 1)
    values = v[:, :cache_length].transpose(1, 2).repeat_interleave(heads // kv_heads, 1)
    expected = F.scaled_dot_product_attention(q.unsqueeze(2), keys, values).squeeze(2)
    torch.testing.assert_close(got, expected, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("penalize", [False, True])
def test_sampling_preparation_with_greedy_and_stochastic_rows(split, penalize):
    import triton

    from mstar.engine.resources.sampler import kernels

    torch.manual_seed(37)
    batch, vocab = 3, 129
    logits = torch.randn(batch, vocab)
    logits[:, :16] = -torch.inf
    temperature = torch.tensor([0.0, 0.6, 1.8])
    penalty = torch.tensor([1.0, 1.2, 1.4])
    seen = torch.rand(batch, vocab) < 0.2
    got = torch.empty_like(logits)
    options = dict(BLOCK_SIZE=64, APPLY_PENALTY=penalize, INCLUDE_GREEDY=True,
                   num_warps=4, num_stages=2)
    if not split:
        kernels._fused_sampling_prep_kernel.fn[(batch,)](
            logits, temperature, penalty, seen, got, vocab,
            *logits.stride(), *got.stride(), *seen.stride(), **options,
        )
    else:
        count, chunk = 3, 43
        maxima = torch.empty(batch, count)
        sums = torch.empty_like(maxima)
        args = torch.empty(batch, count, dtype=torch.int32)
        row_max, row_inv = torch.empty(batch), torch.empty(batch)
        row_arg = torch.empty(batch, dtype=torch.int32)
        kernels._split_partials_kernel[(batch, count)](
            logits, temperature, penalty, seen, maxima, sums, args,
            vocab, chunk, *logits.stride(), *seen.stride(), count, **options,
        )
        kernels._split_combine_kernel[(batch,)](
            maxima, sums, args, row_max, row_inv, row_arg, count, count,
            BLOCK_N=triton.next_power_of_2(count), INCLUDE_GREEDY=True, num_warps=4,
        )
        kernels._split_write_kernel[(batch, count)](
            logits, temperature, penalty, seen, row_max, row_inv, row_arg, got,
            vocab, chunk, *logits.stride(), *got.stride(), *seen.stride(), **options,
        )
    adjusted = torch.where(
        seen, torch.where(logits > 0, logits / penalty[:, None], logits * penalty[:, None]), logits,
    ) if penalize else logits
    expected = torch.softmax(adjusted / temperature.clamp_min(1e-30)[:, None], -1)
    expected[0] = F.one_hot(adjusted[0].argmax(), vocab).float()
    torch.testing.assert_close(got, expected, rtol=2e-5, atol=2e-6)
    assert torch.equal(got[0], expected[0])


@pytest.mark.parametrize("batch", [1, 3, 8])
def test_fused_moe_against_original_expert_loop(batch):
    from mstar.model.components.moe import dispatch_experts_fused
    from mstar.utils.fused_moe.runner import fused_experts

    torch.manual_seed(7 + batch)
    hidden, inter, experts, topk = 32, 16, 4, 2
    x = torch.randn(batch, hidden, dtype=torch.float16)
    w1 = torch.randn(experts, 2 * inter, hidden, dtype=torch.float16) * 0.1
    w2 = torch.randn(experts, hidden, inter, dtype=torch.float16) * 0.1
    weights, ids = torch.randn(batch, experts).softmax(-1).topk(topk, -1)
    weights /= weights.sum(-1, keepdim=True)
    got = fused_experts(
        x, w1, w2, weights, ids,
        config=dict(BLOCK_SIZE_M=16, BLOCK_SIZE_N=32, BLOCK_SIZE_K=16,
                    GROUP_SIZE_M=1, num_warps=4, num_stages=2),
    )
    expected = dispatch_experts_fused(x, w1, w2, experts, ids, weights)
    torch.testing.assert_close(got, expected, rtol=2e-3, atol=2e-4)
