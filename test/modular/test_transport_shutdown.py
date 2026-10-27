"""Shutdown consumes upstream read-lifetime evidence before releasing owners."""
import weakref
from pathlib import Path

import pytest
from test_failed_tensor_read import _kept, _never_freed, _sources, _start
from test_failed_tensor_read import engine as engine  # noqa: F401, PLC0414 — shared fixture
from test_runtime_extension_lifecycle import Backend, HeldWitness, Owner, worker_class
from test_runtime_extension_lifecycle import managed_environment as managed_environment  # noqa: F401, PLC0414
from test_shm_shutdown_cleanup import _DeploymentCommunicator, _send_unacked

from mstar.communication.tensors import SharedMemoryCommunicationManager
from mstar.runtime_extensions import run_managed_worker


@pytest.mark.parametrize("polled", [False, True])
@pytest.mark.parametrize("terminal", [1, -1])
def test_shutdown_retains_running_read_then_releases_on_terminal_status(engine, polled, terminal):
    manager = engine.manager()
    sources = _sources(2)
    engine.plan = {sources[0].data_ptr(): [[-1]], sources[1].data_ptr(): [[0]]}
    future, pointers = _start(manager, "bad", sources)
    assert future.exception(timeout=5) is not None
    if polled:
        manager.get_ready_tensors()
        manager.force_cleanup_request("bad")
    with pytest.raises(RuntimeError, match="retained transfers"):
        manager.shutdown()
    assert _kept(manager) == pointers
    assert not engine.unregistered
    engine.report(sources[1].data_ptr(), terminal)
    manager.shutdown()
    assert not _kept(manager) and not _never_freed(manager)
    assert sorted(engine.unregistered) == sorted(pointers)
    assert manager.tensor_store.owned_uuids() == ()
    manager.shutdown()


def test_shutdown_retains_unknown_buffers_and_managed_release_is_not_acknowledged(engine, managed_environment):
    log, roots = managed_environment
    manager = engine.manager()
    sources = _sources(1)
    engine.plan = {sources[0].data_ptr(): [[9]]}
    future, pointers = _start(manager, "unknown", sources)
    assert future.exception(timeout=5) is not None
    backend = Backend(log)
    owner = Owner(log)
    BaseWorker = worker_class(log, roots)

    class Worker(BaseWorker):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.tensor_manager = manager
            roots.append(weakref.ref(manager))

    with pytest.raises(HeldWitness):
        run_managed_worker(Worker, {"worker_id": "w0", "device": "cpu"}, backend, owner)
    assert _never_freed(manager) == pointers
    assert not engine.unregistered
    assert owner.state == "held"
    assert "backend.close" not in log and "owner.release" not in log


@pytest.mark.parametrize("unique", [False, True])
def test_shutdown_releases_local_storage_but_unlinks_only_own_launch(tmp_path, monkeypatch, unique):
    if unique:
        monkeypatch.setenv("MSTAR_DEPLOYMENT_TOKEN", "ab" * 8)
    else:
        monkeypatch.delenv("MSTAR_DEPLOYMENT_TOKEN", raising=False)
    manager = SharedMemoryCommunicationManager(
        "worker", "host", "cpu", _DeploymentCommunicator(str(tmp_path / "endpoint")), shm_dir=str(tmp_path),
    )
    paths = _send_unacked(manager, 2)
    assert manager.names_unique_per_launch is unique
    if not unique:
        for path in paths:
            Path(path).write_bytes(b"new launch")
    manager.shutdown()
    assert manager.tensor_store.owned_uuids() == ()
    for path in paths:
        if unique:
            assert not Path(path).exists()
        else:
            assert Path(path).read_bytes() == b"new launch"
    manager.shutdown()
