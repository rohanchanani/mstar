"""CUDA KV scatter implementation, imported only on the CUDA fast path."""

import triton
import triton.language as tl

@triton.jit
def _kv_scatter_nhd_kernel(
    layer_ptr, k_ptr, v_ptr, page_idx_ptr, cache_idx_ptr,
    stride_page, stride_kv, stride_slot,
    stride_kt, stride_kh, stride_vt, stride_vh,
    num_pages, page_size,
    HEADS: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    page = tl.load(page_idx_ptr + token).to(tl.int64)
    slot = tl.load(cache_idx_ptr + token).to(tl.int64)
    # A bad index would overwrite another page's KV. Compiled in only with TRITON_DEBUG=1.
    tl.device_assert((page >= 0) & (page < num_pages), "KV page index out of range")
    tl.device_assert((slot >= 0) & (slot < page_size), "KV slot index out of range")
    offs = tl.arange(0, BLOCK)
    mask = offs < HEADS * HEAD_DIM
    head, dim = offs // HEAD_DIM, offs % HEAD_DIM
    dst = layer_ptr + page * stride_page + slot * stride_slot + offs
    k = tl.load(k_ptr + token * stride_kt + head * stride_kh + dim, mask=mask)
    v = tl.load(v_ptr + token * stride_vt + head * stride_vh + dim, mask=mask)
    tl.store(dst, k.to(layer_ptr.dtype.element_ty), mask=mask)
    tl.store(dst + stride_kv, v.to(layer_ptr.dtype.element_ty), mask=mask)
