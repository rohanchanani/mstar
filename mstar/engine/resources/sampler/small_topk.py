"""Small-vocabulary top-k probability filtering; RNG remains with FlashInfer."""

import torch
import triton
import triton.language as tl


@triton.jit
def _filter(P, K, Y, V: tl.constexpr, S: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    i = tl.arange(0, B)
    p = tl.load(P + row * S + i, i < V, other=-float("inf"))
    k = tl.load(K + row)
    ordered = tl.sort(p, descending=True)
    threshold = tl.sum(tl.where(i == tl.minimum(k, V) - 1, ordered, 0.0), 0)
    kept = tl.where((i < V) & (p >= threshold), p, 0.0)
    total = tl.sum(kept, 0)
    y = kept / tl.maximum(total, 1e-8)
    y = tl.where(k >= V, p, y)
    tl.store(Y + row * V + i, y, i < V)


def small_topk(probs, k, warps=8):
    assert probs.dtype == torch.float32 and probs.stride(1) == 1 and k.ndim == 1
    out = torch.empty(probs.shape, device=probs.device, dtype=probs.dtype)
    _filter[(probs.shape[0],)](
        probs,
        k,
        out,
        probs.shape[1],
        probs.stride(0),
        triton.next_power_of_2(probs.shape[1]),
        num_warps=warps,
        enable_fp_fusion=False,
    )
    return out
