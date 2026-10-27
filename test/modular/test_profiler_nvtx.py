"""NVTX is opt-in and only supported for CUDA deployments."""

import pytest
import torch

from mstar.utils.profiler import nvtx_enabled


@pytest.mark.parametrize("requested", [False, True])
@pytest.mark.parametrize("backend", ["cuda", "xpu", "cpu", None])
def test_nvtx_follows_the_selected_accelerator(monkeypatch, requested, backend):
    accelerator = torch.device(backend) if backend else None
    monkeypatch.setattr(
        torch.accelerator, "current_accelerator",
        lambda **kwargs: accelerator,
    )
    assert nvtx_enabled(requested) == (requested and backend == "cuda")


@pytest.mark.parametrize("backend", ["cuda", "xpu", "cpu"])
def test_explicit_worker_device_controls_nvtx(monkeypatch, backend):
    # A CUDA accelerator elsewhere must not enable NVTX for an XPU/CPU worker.
    monkeypatch.setattr(
        torch.accelerator, "current_accelerator",
        lambda **kwargs: torch.device("cuda"),
    )
    assert nvtx_enabled(True, torch.device(backend)) == (backend == "cuda")
    assert not nvtx_enabled(False, torch.device(backend))


@pytest.mark.parametrize("backend", ["cuda", "xpu", "cpu"])
def test_client_nvtx_is_only_enabled_on_cuda(monkeypatch, backend):
    from mstar.client.client import MStarClient

    monkeypatch.setattr(
        torch.accelerator, "current_accelerator",
        lambda **kwargs: torch.device(backend),
    )
    client = MStarClient(enable_nvtx=True)
    assert (client._nvtx is not None) == (backend == "cuda")


def test_disabled_phase_laps_are_shared_without_recording(monkeypatch):
    import mstar.utils.profiler as profiler

    monkeypatch.setattr(profiler, "PHASE_PERIOD", 0)
    monkeypatch.setattr(profiler, "_PHASE_BUF", {})
    first, second = profiler.PhaseLaps("first."), profiler.PhaseLaps("second.")
    assert first is second
    first.skip()
    second.lap("step")
    assert profiler.phase_buffer() == {}


def test_enabled_phase_laps_keep_independent_clocks(monkeypatch):
    from collections import defaultdict
    import mstar.utils.profiler as profiler

    monkeypatch.setattr(profiler, "PHASE_PERIOD", 1)
    monkeypatch.setattr(profiler, "_PHASE_BUF", defaultdict(list))
    wall, cpu = iter([0., 2., 5., 9.]), iter([0., 1., 3., 6.])
    monkeypatch.setattr(profiler, "perf_counter", lambda: next(wall))
    monkeypatch.setattr(profiler, "thread_time", lambda: next(cpu))
    first, second = profiler.PhaseLaps("first."), profiler.PhaseLaps("second.")
    first.lap("step")
    second.lap("step")
    assert dict(profiler.phase_buffer()) == {
        "first.step": [5.], "first.step.cpu": [3.],
        "second.step": [7.], "second.step.cpu": [5.],
    }
