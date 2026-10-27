"""MI300X small-codebook sampling preparation; vendor TopP owns the draw.

Fuse temperature/greedy softmax and sequential top-k renormalization. Preserve
BF16 input logits and the FP32 probability boundary, including ties. The fused
reduction can differ from the original padded reduction by small FP32 rounding
errors; it does not promise bitwise probability or universal token equality.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _small_vocabulary_prep_topk_kernel(
    logits_ptr, temperature_ptr, top_k_ptr, output_ptr,
    row_stride: tl.constexpr,
):
    row = tl.program_id(0)
    indices = tl.arange(0, 2048)
    logits = tl.load(logits_ptr + row * row_stride + indices).to(tl.float32)
    temperature = tl.load(temperature_ptr + row)
    inv_temperature = tl.where(
        temperature == 0, 1.0, 1.0 / tl.maximum(temperature, 1e-30),
    )
    maximum = tl.max(logits, 0)
    argmax = tl.argmax(logits, 0)
    exponentials = tl.exp(logits * inv_temperature - maximum * inv_temperature)
    probabilities = exponentials * (1.0 / tl.maximum(tl.sum(exponentials, 0), 1e-30))
    probabilities = tl.where(
        temperature == 0, tl.where(indices == argmax, 1.0, 0.0), probabilities,
    )
    top_k = tl.load(top_k_ptr + row)
    top_k = tl.where(top_k > 0, top_k, 2048)
    ordered = tl.sort(probabilities, descending=True)
    threshold = tl.sum(tl.where(indices == tl.minimum(top_k, 2048) - 1, ordered, 0.0), 0)
    kept = tl.where(probabilities >= threshold, probabilities, 0.0)
    filtered = kept / tl.maximum(tl.sum(kept, 0), 1e-8)
    filtered = tl.where(top_k >= 2048, probabilities, filtered)
    tl.store(output_ptr + row * 2048 + indices, filtered)


def prep(logits, temperature, top_k):
    output = torch.empty(logits.shape, device=logits.device, dtype=torch.float32)
    _small_vocabulary_prep_topk_kernel[(logits.shape[0],)](
        logits, temperature, top_k, output, logits.stride(0), num_warps=8,
    )
    return output


@torch.compiler.disable
def try_sample(logits, temperature, top_k, top_p, seed, offset):
    """Return None outside qualified preparation geometry; caller uses baseline.

    Caller excludes repetition-penalty and min-p policies. TopP still receives
    the original per-row policy and seed/offset buffers without mutation.
    """
    if not (
        torch.version.hip and logits.is_cuda and logits.ndim == 2
        and logits.shape[1] == 2048 and 1 <= logits.shape[0] <= 8
        and logits.dtype == torch.bfloat16 and logits.stride(1) == 1
        and torch.cuda.get_device_properties(logits.device).gcnArchName.split(':')[0] == 'gfx942'
    ):
        return None
    rows = logits.shape[0]
    for value, dtype in ((temperature, torch.float32), (top_k, torch.int32)):
        if (
            value.device != logits.device or value.dtype != dtype
            or value.shape != (rows,) or not value.is_contiguous()
        ):
            return None
    from mstar.utils.flashinfer_rocm import import_flashinfer_rocm

    filtered = prep(logits, temperature, top_k)
    return import_flashinfer_rocm('sampling').sampling.top_p_sampling_from_probs(
        filtered, top_p, deterministic=True, seed=seed, offset=offset,
    ).to(torch.int64)
