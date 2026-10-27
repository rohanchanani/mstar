"""Sampling kernels on AMD GPUs: FlashInfer's ROCm fork.

The fork keeps the per-row device ``seed``/``offset`` tensors that let a captured
graph draw fresh numbers on every replay, so the CUDA call sites carry over
unchanged. Imported only on ROCm.
"""

import inspect
from collections.abc import Callable

import torch

from mstar.utils.flashinfer_rocm import import_flashinfer_rocm


def sampling_from_probs(name: str) -> Callable:
    """The fork's ``flashinfer.sampling.<name>``, checked to take tensor seeds.

    The fork guards its ops against dynamo by raising when traced, where upstream
    graph-breaks; ``torch.compiler.disable`` restores the graph break, so a caller
    traced by dynamo runs the kernel eagerly rather than failing.
    """
    flashinfer = import_flashinfer_rocm("sampling")
    fn = getattr(flashinfer.sampling, name)
    params = inspect.signature(fn).parameters
    if "seed" not in params or "offset" not in params:
        raise RuntimeError(
            f"flashinfer.sampling.{name} in this amd-flashinfer build takes no "
            "seed/offset tensors, which captured sampling needs (0.6.18+amd.1 has them)"
        )
    if name == "top_k_top_p_sampling_from_probs":
        from mstar.engine.resources.sampler import small_topk as small_topk_module

        def small_vocabulary_sample(
            probs,
            top_k,
            top_p,
            indices=None,
            filter_apply_order="top_k_first",
            deterministic=True,
            generator=None,
            check_nan=False,
            seed=None,
            offset=None,
            return_valid=False,
        ):
            if (
                filter_apply_order == "top_k_first"
                and indices is None
                and not check_nan
                and probs.dtype == torch.float32
                and probs.ndim == 2
                and probs.shape[0] <= 8
                and probs.shape[1] == 2048
                and probs.stride(1) == 1
                and isinstance(top_k, torch.Tensor)
                and top_k.ndim == 1
                and top_k.shape[0] == probs.shape[0]
                and top_k.is_contiguous()
            ):
                filtered = small_topk_module.small_topk(probs, top_k, warps=8)
                return flashinfer.sampling.top_p_sampling_from_probs(
                    filtered,
                    top_p,
                    deterministic=deterministic,
                    generator=generator,
                    seed=seed,
                    offset=offset,
                    return_valid=return_valid,
                )
            return fn(
                probs,
                top_k,
                top_p,
                indices=indices,
                filter_apply_order=filter_apply_order,
                deterministic=deterministic,
                generator=generator,
                check_nan=check_nan,
                seed=seed,
                offset=offset,
                return_valid=return_valid,
            )

        return torch.compiler.disable(small_vocabulary_sample)
    return torch.compiler.disable(fn)
