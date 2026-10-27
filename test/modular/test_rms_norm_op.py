"""Backend-neutral RMSNorm operator tests."""

import torch

from mstar.engine.resources.rms_norm import run_rms_norm


def test_rms_norm_portable_kernel_matches_reference():
    x = torch.tensor([[3.0, 4.0], [1.0, -2.0]])
    weight = torch.tensor([0.5, 1.5])
    eps = 1e-6

    actual = run_rms_norm(x, weight, eps)
    expected = (
        x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps)
    ) * weight

    torch.testing.assert_close(actual, expected)


def test_rms_norm_operator_has_backend_neutral_name():
    assert hasattr(torch.ops.mstar, "rms_norm")


def test_fused_add_rms_norm_portable_kernel_preserves_rounding_and_mutation():
    for gemma in (False, True):
        x = torch.tensor([[0.12, 0.58, -1.3, 2.8]], dtype=torch.bfloat16)
        residual = torch.tensor([[0.9, -1.3, 2.1, 0.2]], dtype=torch.bfloat16)
        weight = torch.tensor([0.2, 0.4, 0.8, 1.2], dtype=torch.bfloat16)
        total = x.float() + residual.float()
        scale = weight.float() + 1 if gemma else weight.float()
        expected = (total * torch.rsqrt(total.square().mean(-1, keepdim=True) + 1e-6) * scale).bfloat16()
        x_ptr, residual_ptr = x.data_ptr(), residual.data_ptr()
        torch.ops.mstar.fused_add_rms_norm(x, residual, weight, 1e-6, gemma)
        assert x.data_ptr() == x_ptr and residual.data_ptr() == residual_ptr
        assert torch.equal(x, expected)
        assert torch.equal(residual, total.bfloat16())
