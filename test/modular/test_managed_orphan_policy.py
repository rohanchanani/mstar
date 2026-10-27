"""Managed workers use the real upstream watchdog, without killing a test process."""

import signal
from types import SimpleNamespace

import pytest

from mstar import runtime_extensions
from mstar.utils import orphan


@pytest.mark.parametrize("owner_present", [False, True])
@pytest.mark.parametrize("record_fails", [False, True])
def test_managed_orphan_cleans_then_exits_without_acknowledging_release(
    monkeypatch, owner_present, record_fails,
):
    events = []
    watchdog = {}

    class Thread:
        def __init__(self, **kwargs):
            watchdog.update(kwargs)

        def start(self):
            pass

    class Owner:
        def unknown(self, reason):
            events.append("unknown")
            if record_fails:
                raise OSError("status write failed")

        def release(self):
            pytest.fail("forced exit must not acknowledge release")

    def hard_exit(code):
        events.append(("exit", code))
        raise SystemExit(code)

    monkeypatch.setattr(runtime_extensions.threading, "Thread", Thread)
    monkeypatch.setattr(signal, "signal", lambda *args: None)
    monkeypatch.setattr(orphan.mp, "parent_process", lambda: SimpleNamespace(is_alive=lambda: False))
    monkeypatch.setattr(orphan.os, "kill", lambda pid, sig: events.append(("signal", sig)))
    monkeypatch.setattr(orphan.time, "sleep", lambda seconds: events.append(("grace", seconds)))
    monkeypatch.setattr(orphan.os, "_exit", hard_exit)

    class Worker:
        def __init__(self, **kwargs):
            self.tensor_manager = object()

        def run(self):
            # Raise outside the managed error handler, as a real os._exit would
            # terminate the process rather than unwind its Python stack.
            try:
                watchdog["target"](*watchdog["args"], **watchdog["kwargs"])
            except SystemExit:
                pass

        def close(self):
            events.append("close")

    extension = SimpleNamespace(
        initialize=lambda *args: None,
        close=lambda: None,
        is_fatal=lambda: False,
    )
    runtime_extensions.run_managed_worker(
        Worker, {"worker_id": "w0", "device": "cpu"}, extension,
        Owner() if owner_present else None,
        cleanup_worker=lambda worker: events.append("unlink") if worker.tensor_manager else None,
    )
    expected = [("signal", signal.SIGTERM), ("grace", 5.0)]
    if owner_present:
        expected.append("unknown")
    assert events == expected + ["unlink", ("exit", 1), "close"]
    assert watchdog["target"] is orphan.exit_when_orphaned
