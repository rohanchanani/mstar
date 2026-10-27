"""Host tests of the managed execution boundary, without device simulation.

The worker's real close/error methods drive explicit fake submitters and a
backend whose outstanding-use flag is cleared only by drain. Signal handling
and the permanent fatal wait are intercepted so no process is killed or hung.
"""
from __future__ import annotations

import weakref
from types import SimpleNamespace

import pytest
import torch

from mstar import runtime_extensions
from mstar.conductor import conductor as conductor_module
from mstar.conductor.conductor import Conductor
from mstar.runtime_extensions import BackendFatalError, run_managed_worker
from mstar.worker.micro_scheduler import ScheduledBatch
from mstar.worker.worker import Worker


class HeldWitness(BaseException):
    """The test reached the permanent hold with all relevant roots alive."""


class Owner:
    def __init__(self, log):
        self.log = log
        self.state = "reserved"

    def release(self):
        self.state = "released"
        self.log.append("owner.release")

    def unknown(self, reason):
        self.log.append("owner.unknown")


class Backend:
    def __init__(self, log, *, init_error=False, fatal=False, marker_error=False):
        self.log = log
        self.init_error = init_error
        self.fatal = fatal
        self.marker_error = marker_error
        self.pending = False
        self.owner = None

    def initialize(self, device, owner):
        self.log.append("backend.initialize")
        self.owner = owner
        owner.state = "active"
        self.pending = True
        if self.init_error:
            raise ValueError("initialization failed after accepted preparation")

    def is_fatal(self):
        return self.fatal

    def check_health(self):
        self.log.append("backend.health")
        if self.fatal:
            raise BackendFatalError("uncertain accepted work")

    def drain(self):
        self.log.append("backend.drain")
        self.check_health()
        self.pending = False

    def close(self):
        assert not self.pending
        self.log.append("backend.close")
        self.owner.release()

    def hold(self, reason):
        self.log.append("backend.hold")
        self.owner.state = "held"
        if self.marker_error:
            raise OSError("controlled durable marker failure")


class Executor:
    def __init__(self, name, log):
        self.name, self.log = name, log

    def shutdown(self, *, wait, cancel_futures=False):
        self.log.append(self.name + (".join" if wait else ".cancel"))
        if not wait:
            assert cancel_futures


@pytest.fixture
def managed_environment(monkeypatch):
    log, roots, handlers = [], [], {}

    class Watchdog:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def start(self):
            log.append("watchdog.start")

    class Event:
        def __init__(self):
            self.stopped = False

        def set(self):
            self.stopped = True

        def is_set(self):
            return self.stopped

        def wait(self, timeout=None):
            assert timeout is not None
            assert all(reference() is not None for reference in roots)
            assert "owner.release" not in log
            assert "backend.close" not in log
            assert all(handler is runtime_extensions.signal.SIG_IGN for handler in handlers.values())
            log.append("held.wait")
            raise HeldWitness()

    class Logger:
        def exception(self, *args):
            log.append("log.exception")

        def critical(self, *args):
            log.append("log.critical")

    monkeypatch.setattr(runtime_extensions, "threading", SimpleNamespace(Thread=Watchdog, Event=Event))
    monkeypatch.setattr(runtime_extensions, "logging", SimpleNamespace(getLogger=lambda _: Logger()))

    def install_signal(signum, handler):
        prior = handlers.get(signum)
        handlers[signum] = handler
        return prior

    monkeypatch.setattr(runtime_extensions.signal, "signal", install_signal)
    return log, roots


def worker_class(log, roots, *, run_error=False, construction_error=False):
    class LifecycleWorker:
        close = Worker.close
        request_stop = Worker.request_stop

        def __init__(self, *, runtime_extension, **kwargs):
            roots.append(weakref.ref(self))
            self.runtime_extension = runtime_extension
            self._executors = [Executor("gpu", log), Executor("plan", log)]

            def release(name):
                assert not runtime_extension.pending
                log.append(name)

            self.engine_manager = SimpleNamespace(shutdown=lambda: release("engine.cleanup"))
            if construction_error:
                raise ValueError("construction failed after engine binding")
            self.tensor_manager = SimpleNamespace(shutdown=lambda: release("tensor.cleanup"))

        def run(self):
            log.append("worker.run")
            if run_error:
                raise ValueError("submitted program raised")

    return LifecycleWorker


@pytest.mark.parametrize("run_error", [False, True])
def test_managed_normal_and_exception_exit_drain_before_releasing_owners(managed_environment, run_error):
    log, roots = managed_environment
    backend = Backend(log)
    owner = Owner(log)
    worker = worker_class(log, roots, run_error=run_error)
    if run_error:
        with pytest.raises(RuntimeError, match="submitted program raised"):
            run_managed_worker(worker, {"worker_id": "w0", "device": "cpu"}, backend, owner)
    else:
        run_managed_worker(worker, {"worker_id": "w0", "device": "cpu"}, backend, owner)
    order = ["gpu.cancel", "plan.cancel", "gpu.join", "plan.join", "backend.drain",
             "engine.cleanup", "tensor.cleanup", "backend.close", "owner.release"]
    assert [entry for entry in log if entry in order] == order
    assert owner.state == "released"
    assert roots[0]() is None


def test_synchronous_extension_uses_the_lifecycle_without_a_native_lease(managed_environment):
    log, roots = managed_environment

    class SynchronousBackend(Backend):
        def initialize(self, device, owner):
            assert device == torch.device("cpu") and owner is None
            self.pending = True
            log.append("backend.initialize")

        def close(self):
            assert not self.pending
            log.append("backend.close")

    backend = SynchronousBackend(log)
    run_managed_worker(worker_class(log, roots), {"worker_id": "w0", "device": "cpu"}, backend, None)
    assert log.index("backend.drain") < log.index("engine.cleanup") < log.index("backend.close")
    assert "owner.release" not in log and roots[0]() is None


@pytest.mark.parametrize("failure", ["initialization", "construction"])
def test_partial_construction_preserves_completion_before_cleanup(managed_environment, failure):
    log, roots = managed_environment
    backend = Backend(log, init_error=failure == "initialization")
    owner = Owner(log)
    worker = worker_class(log, roots, construction_error=failure == "construction")
    with pytest.raises(RuntimeError, match="failed after"):
        run_managed_worker(worker, {"worker_id": "w0", "device": "cpu"}, backend, owner)
    assert "worker.run" not in log
    assert log.index("backend.drain") < log.index("backend.close") < log.index("owner.release")
    if failure == "construction":
        assert log.index("backend.drain") < log.index("engine.cleanup")
    assert owner.state == "released"


@pytest.mark.parametrize("marker_error", [False, True])
def test_fatal_exit_retains_worker_and_native_roots_even_if_marker_write_fails(managed_environment, marker_error):
    log, roots = managed_environment
    backend = Backend(log, fatal=True, marker_error=marker_error)
    owner = Owner(log)
    worker = worker_class(log, roots, run_error=True)
    with pytest.raises(HeldWitness):
        run_managed_worker(worker, {"worker_id": "w0", "device": "cpu"}, backend, owner)
    assert "held.wait" in log
    assert "engine.cleanup" not in log
    assert "tensor.cleanup" not in log
    assert "backend.drain" not in log
    assert backend.pending


def test_main_loop_exception_joins_submitters_then_drains_before_request_cleanup():
    log = []
    worker = Worker.__new__(Worker)
    worker.worker_id = "w0"
    worker._in_flight_rids = {7}
    backend = Backend(log)
    backend.pending = True
    worker.runtime_extension = backend
    worker._clear_in_flight_flag = lambda batch: log.append("batch.clear")
    worker._reset_skip_plan_flags = lambda batch: log.append("plan.reset")
    worker._settle_speculation = lambda *args, **kwargs: log.append("speculation.release")
    worker._graph_runtime = SimpleNamespace(push_back_node=lambda *args: log.append("queue.restore"))

    def fail_requests(errors):
        assert not backend.pending
        assert set(errors) == {7}
        log.append("requests.fail")

    worker._fail_requests = fail_requests

    class Future:
        def __init__(self, name, fail=False):
            self.name, self.fail = name, fail

        def result(self):
            log.append(self.name)
            if self.fail:
                raise ValueError("submitted work reported an error")

    pending = SimpleNamespace(batch=ScheduledBatch("node", "decode", {7: 0}),
                              future=Future("gpu.result", fail=True), node_name="node")
    speculation = SimpleNamespace(
        plan_future=Future("plan.result"), submitted=False, node_batch=object(),
        scheduled_batch=ScheduledBatch("node", "decode", {8: 0}), continuing_rids=set(),
    )
    worker._handle_main_loop_error(ValueError("host exception"), (pending,), None, speculation)
    assert log.index("gpu.result") < log.index("backend.drain")
    assert log.index("plan.result") < log.index("backend.drain")
    assert log.index("backend.drain") < log.index("batch.clear") < log.index("requests.fail")
    assert log.index("backend.drain") < log.index("speculation.release")


@pytest.mark.parametrize("released", [False, True])
def test_conductor_never_kills_a_worker_without_native_release_ack(released):
    log = []

    class Process:
        pid = 42
        alive = True

        def is_alive(self): return self.alive
        def terminate(self): log.append("terminate")
        def join(self, timeout): log.append("join")
        def kill(self):
            log.append("kill")
            self.alive = False

    owner = SimpleNamespace(status=lambda: "released" if released else "active",
                            unknown=lambda reason: log.append("unknown"))
    conductor = Conductor.__new__(Conductor)
    conductor.worker_ids = ["w0"]
    conductor._worker_processes = [Process()]
    conductor._runtime_owners = {"w0": owner}
    conductor.shutdown()
    assert log[:2] == ["terminate", "join"]
    assert ("kill" in log) is released
    assert ("unknown" in log) is not released


@pytest.mark.parametrize("start_error", [False, True])
def test_spawn_reserves_before_start_and_failure_never_fabricates_release(monkeypatch, start_error):
    log = []
    owner = SimpleNamespace(
        unknown=lambda reason: log.append("unknown"),
    )
    processes = []

    class Process:
        # CPython can create an OS child and then fail while writing its
        # bootstrap pipe, before Process._popen is assigned. pid=None therefore
        # does not prove that there is no child or native ownership transfer.
        pid = None

        def __init__(self, *, target, kwargs, daemon):
            assert kwargs["runtime_owner"] is owner
            assert log == ["reserve"]
            assert not daemon
            processes.append(self)

        def start(self):
            log.append("start")
            if start_error:
                raise OSError("spawn failed before the process handle was returned")
            self.pid = 42

    def reserve(context, name):
        log.append("reserve")
        return owner

    monkeypatch.setattr(conductor_module.mp, "get_context", lambda _: SimpleNamespace(Process=Process))
    monkeypatch.setattr(conductor_module, "apply_process_env_defaults", lambda: None)
    monkeypatch.setattr(conductor_module, "keep_bootstrap_sockets_on_loopback", lambda: None)
    monkeypatch.setattr(conductor_module.atexit, "register", lambda callback: log.append("atexit"))
    conductor = Conductor.__new__(Conductor)
    conductor._sorted_ranks = [0]
    conductor.worker_ids = ["w0"]
    conductor.runtime_extension = SimpleNamespace(reserve_worker=reserve)
    conductor.runtime_admission = object()
    conductor._runtime_owners = {}
    conductor._worker_processes = []
    conductor._per_worker_graphs = {"w0": []}
    conductor.model_config = {}
    conductor._all_worker_graph_ids_to_graph_walks = {}
    conductor._all_worker_graph_ids_to_nodes = {}
    conductor._all_worker_graph_ids_to_dyn_loops = {}
    conductor.per_worker_sharding_config = {"w0": None}
    conductor.parallel_config = SimpleNamespace(per_worker_config={"w0": None})
    conductor.hostname = "localhost"
    conductor.socket_path_prefix = "/tmp/native-free-lifecycle-test"
    conductor._dist_init_method = None
    conductor.model = object()
    conductor.enable_nvtx = conductor.enable_prof = False
    conductor.device_type = "cpu"
    conductor.rank_devices = None
    conductor.log_level = "INFO"
    conductor.tensor_comm_protocol = None
    conductor.tcp_transfer_device = ""
    if start_error:
        with pytest.raises(OSError, match="spawn failed"):
            conductor._launch_workers()
        assert log == ["reserve", "start", "unknown"]
    else:
        conductor._launch_workers()
        assert log == ["reserve", "start", "atexit"]
        assert conductor._worker_processes == processes
    assert conductor._runtime_owners == {"w0": owner}


@pytest.mark.parametrize("read_error", [False, True])
def test_tensor_transport_waits_for_reads_before_releasing_real_storage(tmp_path, read_error):
    from concurrent.futures import Future

    from mstar.communication.tensors import FutureAndPointers, SharedMemoryCommunicationManager
    from mstar.graph.base import TensorPointerInfo

    manager = SharedMemoryCommunicationManager("worker_0", "localhost", "cpu",
                                                SimpleNamespace(send=lambda *args: None),
                                                shm_dir=str(tmp_path))
    tensor = torch.ones(4)
    host_copy = tensor.clone()
    reference, host_reference = weakref.ref(tensor), weakref.ref(host_copy)
    info = TensorPointerInfo(dims=[4], dtype=torch.float32, nbytes=16,
                             address=tensor.data_ptr(), stride=[1], uuid=101,
                             source_session_id="native-free", source_entity="worker_0")
    manager.tensor_store.put_tensor(7, 101, tensor, info, cpu_tensor=host_copy)
    manager.buffered_shards[7] = {"retained": [tensor]}
    path = tmp_path / "owned-tensor"
    path.write_bytes(b"retained until reads settle")
    manager._shm_files[101] = str(path)

    class ReadFuture(Future):
        def result(self, timeout=None):
            assert reference() is not None and host_reference() is not None
            assert manager.tensor_store.owned_uuids() == (101,)
            assert path.exists()
            return super().result(timeout)

    future = ReadFuture()
    if read_error:
        future.set_exception(OSError("read completion could not be established"))
    else:
        future.set_result(None)
    manager.pending.append(FutureAndPointers(
        future=future, request_id=7, graph_edges=[], transfer_uuids=[101],
    ))
    del tensor, host_copy
    if read_error:
        with pytest.raises(RuntimeError, match="retained transfers"):
            manager.shutdown()
        # No successful retirement is fabricated on an exceptional wait.
        assert reference() is not None
        assert path.exists()
    else:
        manager.shutdown()
        assert manager.tensor_store.owned_uuids() == ()
        assert manager.pending == [] and manager.buffered_shards == {}
        assert reference() is None and host_reference() is None
        assert path.exists()  # no launch-unique namespace: preserve upstream exit guard


def test_worker_target_forwards_upstream_settings_to_managed_lifecycle(monkeypatch):
    extension, owner = object(), object()
    got = {}
    monkeypatch.setattr(conductor_module, "apply_process_env_defaults", lambda: None)
    monkeypatch.setattr(conductor_module, "quiet_noisy_loggers", lambda: None)
    monkeypatch.setattr(conductor_module.logging, "basicConfig", lambda **kwargs: None)
    monkeypatch.setattr(conductor_module.signal, "signal", lambda *args: None)
    monkeypatch.setattr(conductor_module, "load_runtime_extension", lambda config: extension)

    def managed(cls, kwargs, selected, runtime_owner, *, cleanup_worker):
        assert cls is Worker and selected is extension and runtime_owner is owner
        got.update(kwargs)
        calls = []
        cleanup_worker(SimpleNamespace(tensor_manager=SimpleNamespace(
            names_unique_per_launch=True, unlink_owned_files=lambda: calls.append("unlink"),
        )))
        assert calls == ["unlink"]

    monkeypatch.setattr(conductor_module, "run_managed_worker", managed)
    conductor_module._worker_process_target(
        worker_id="w0", worker_ids=["w0"], my_worker_graphs=[], model_config={},
        all_worker_graph_ids_to_graph_walks={}, all_worker_graph_ids_to_nodes={},
        all_worker_graph_ids_to_dyn_loops={}, sharding_config=None, parallel_groups=None,
        hostname="localhost", socket_path_prefix="/tmp/lifecycle", dist_init_method="",
        device="cpu", status_interval_s=2.5, runtime_owner=owner,
    )
    assert got["status_interval_s"] == 2.5
    assert got["device"] == torch.device("cpu")
    assert got["worker_ids"] == ["w0"]
