"""Host registration backend dispatch, without a GPU or Rust transport."""
from unittest.mock import Mock

import pytest

from mstar.communication import arena


@pytest.mark.parametrize("rocm", [False, True])
@pytest.mark.parametrize("register_rc, expected", [(0, True), (712, True), (1, False)])
def test_host_registration(monkeypatch, rocm, register_rc, expected):
    lib = Mock()
    lib.cudaHostRegister.return_value = register_rc
    lib.cudaHostUnregister.return_value = 0
    native = Mock(return_value=lib)
    binding = Mock(return_value=lib)
    monkeypatch.setattr(arena.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(arena, "current_platform", Mock(is_rocm=lambda: rocm))
    monkeypatch.setattr(arena, "_cudart", native)
    monkeypatch.setattr(arena.torch.cuda, "cudart", binding)

    assert arena._pin(4096, 8192) is expected
    lib.cudaHostRegister.assert_called_once_with(4096, 8192, 0)
    assert arena._unpin(4096)
    lib.cudaHostUnregister.assert_called_once_with(4096)
    assert native.call_count == (0 if rocm else 2)
    assert binding.call_count == (2 if rocm else 0)
    assert lib.cudaGetLastError.call_count == (int(register_rc != 0) if not rocm else 0)

    lib.cudaHostUnregister.return_value = 1
    assert not arena._unpin(4096)
    assert lib.cudaGetLastError.call_count == (int(register_rc != 0) + 1 if not rocm else 0)


def test_registration_without_device_or_cuda_library(monkeypatch):
    native = Mock(side_effect=OSError("missing CUDA runtime"))
    binding = Mock(side_effect=AssertionError("unexpected torch binding"))
    monkeypatch.setattr(arena, "_cudart", native)
    monkeypatch.setattr(arena.torch.cuda, "cudart", binding)
    monkeypatch.setattr(arena.torch.cuda, "is_available", lambda: False)
    assert not arena._pin(4096, 8192)
    assert not arena._unpin(4096)
    native.assert_not_called()

    monkeypatch.setattr(arena.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(arena, "current_platform", Mock(is_rocm=lambda: False))
    assert not arena._pin(4096, 8192)
    assert not arena._unpin(4096)
    binding.assert_not_called()
