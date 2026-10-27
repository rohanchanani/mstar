"""The API's one result drain waits for socket traffic or local publication."""
import os
import queue
import select
import threading
import time
from types import SimpleNamespace

import pytest
from api_server_helpers import make_test_api_server

from mstar.api_server.data_worker import PreprocessWorker, PreprocessWorkerThread
from mstar.api_server.entrypoint import PendingRequest
from mstar.api_server.request_types import PreprocessInput, ResultChunk
from mstar.communication.event import EventWakeup
from mstar.profile.format import RequestProfile

pytestmark = pytest.mark.skipif(not hasattr(os, "eventfd"), reason="Linux eventfd transport")


class PollingComm:
    def __init__(self):
        self.entered = threading.Event()
        self.polls = []

    def register_event_for_poll(self, event):
        self.event = event

    def get_all_new_messages(self):
        return []

    def poll_for_messages(self, timeout_ms):
        self.polls.append(timeout_ms)
        self.entered.set()
        ready, _, _ = select.select([self.event.fd], [], [], timeout_ms / 1000)
        if ready:
            self.event.drain()
        return False


class OutputFacade:
    def __init__(self):
        self.output_wakeup = EventWakeup()
        self.output_queue = queue.Queue()
        self.profile_queue = queue.Queue()
        self.pending = True
        self.cleaned = threading.Event()
        self.lifecycle = []

    def get_result_chunks(self):
        result = []
        while not self.output_queue.empty():
            result.append(self.output_queue.get_nowait())
            self.pending = False
        return result

    def get_profile_updates(self):
        return []

    def has_pending_tensors(self, rid):
        return self.pending

    def received_final_chunks(self, rid, outputs):
        return not self.pending

    def finished_reading(self, rid, drained):
        self.cleaned.set()

    def shutdown(self):
        self.lifecycle.append("producer_joined")

    def close_output_wakeup(self):
        self.lifecycle.append("descriptor_closed")
        PreprocessWorker.close_output_wakeup(self)


def start_server(facade=None, liveness=10):
    facade = facade or OutputFacade()
    comm = PollingComm()
    server = make_test_api_server(preprocess_worker=facade, communicator=comm,
                                  _liveness_interval_s=liveness)
    server._msg_thread = threading.Thread(target=server._process_messages, daemon=True)
    server._msg_thread.start()
    assert comm.entered.wait(1)
    return server, facade, comm


def test_idle_drain_parks_and_cleanup_joins_before_closing():
    server, facade, comm = start_server()
    fd = facade.output_wakeup.fd
    try:
        time.sleep(0.03)
        assert len(comm.polls) == 1
        assert comm.polls[0] > 9000
    finally:
        server.cleanup()
    assert not server._msg_thread.is_alive()
    assert facade.lifecycle == ["producer_joined", "descriptor_closed"]
    with pytest.raises(OSError):
        os.fstat(fd)
    server.cleanup()  # duplicate HTTP/finally cleanup does not reuse the fd


@pytest.mark.parametrize("unstarted", [False, True])
def test_cleanup_before_message_thread_starts(unstarted):
    facade = OutputFacade()
    server = make_test_api_server(preprocess_worker=facade)
    if unstarted:
        # finalize_setup has not started the real constructor's thread yet.
        server._msg_thread = threading.Thread(target=server._process_messages)
    server.cleanup()
    assert facade.output_wakeup is None
    assert facade.lifecycle == ["producer_joined", "descriptor_closed"]


def test_local_final_output_completes_without_another_socket_message():
    facade = OutputFacade()
    server, facade, comm = start_server(facade)
    req = PendingRequest(streaming=False, input_modalities=["text"],
                         output_modalities=["text"],
                         profile=RequestProfile(rid="r"))
    try:
        with server.request_lock:
            server.pending_requests["r"] = req
            server.recently_completed["r"] = time.time()
        facade.output_queue.put(ResultChunk(request_id="r", modality="text", data=b"answer"))
        facade.output_wakeup.signal()
        assert req.event.wait(1)
        assert facade.cleaned.wait(1)
        assert [chunk.data for chunk in req.chunks] == [b"answer"]
        assert "r" not in server.recently_completed
    finally:
        server.cleanup()


def test_publication_before_poll_is_not_lost():
    facade = OutputFacade()
    facade.output_queue.put(ResultChunk(request_id="gone", modality="text", data=b"late"))
    facade.output_wakeup.signal()
    server, facade, comm = start_server(facade)
    try:
        # The queued wake survives the empty-queue-to-wait boundary. A second
        # wait is reached immediately, rather than the ten-second deadline.
        deadline = time.monotonic() + 1
        while len(comm.polls) < 2 and time.monotonic() < deadline:
            time.sleep(0.001)
        assert len(comm.polls) == 2
    finally:
        server.cleanup()


def test_liveness_deadline_bounds_the_transport_wait():
    server, facade, comm = start_server(liveness=0.05)
    try:
        assert 0 <= comm.polls[0] <= 50
    finally:
        server.cleanup()


def test_silent_conductor_exit_still_fails_waiting_request():
    server, facade, comm = start_server(liveness=0.05)
    req = PendingRequest(streaming=False, input_modalities=["text"],
                         output_modalities=["text"], profile=RequestProfile(rid="r"))
    try:
        with server.request_lock:
            server.pending_requests["r"] = req
        server.conductor_proc = SimpleNamespace(is_alive=lambda: False, exitcode=1)
        assert req.event.wait(1)
        server._msg_thread.join(timeout=1)
        assert not server._msg_thread.is_alive()
        assert req.error_status == 503 and server.fatal_error
    finally:
        server.cleanup()


def test_lost_output_ttl_wakes_without_socket_or_local_output():
    server, facade, comm = start_server()
    req = PendingRequest(streaming=False, input_modalities=["text"],
                         output_modalities=["text"], profile=RequestProfile(rid="r"))
    try:
        with server.request_lock:
            server._recently_completed_ttl = 0.03
            server.pending_requests["r"] = req
            server.recently_completed["r"] = time.time()
        facade.output_wakeup.signal()
        assert req.event.wait(1)
        assert req.error_status == 500 and "incomplete" in req.error
        assert any(0 < timeout <= 30 for timeout in comm.polls)
    finally:
        server.cleanup()


def test_collector_release_wakes_post_completion_cleanup():
    server, facade, comm = start_server()
    req = PendingRequest(streaming=False, input_modalities=["text"],
                         output_modalities=["text"], profile=RequestProfile(rid="r"))
    try:
        with server.request_lock:
            req.event.set()
            server.pending_requests["r"] = req
            server.recently_completed["r"] = time.time()
            server._live["r"] = server._client_ids["r"] = "r"
        server.release_request("r")
        assert facade.cleaned.wait(1)
        assert "r" not in server.recently_completed
    finally:
        server.cleanup()


def producer(output_wakeup):
    return PreprocessWorkerThread(
        in_queue=queue.Queue(), result_tensor_queue=queue.Queue(),
        out_queue=queue.Queue(), profile_queue=queue.Queue(),
        cleanup_request_queue=queue.Queue(), abort_request_queue=queue.Queue(),
        reads_done_queue=queue.Queue(), discard_tensor_queue=queue.Queue(),
        stop_event=threading.Event(), communicator=SimpleNamespace(send=lambda *a: None),
        tensor_manager=SimpleNamespace(store_and_return_tensor_info=lambda **kw: {},
                                       register_for_send=lambda **kw: None),
        enable_prof=True, output_wakeup=output_wakeup,
    )


@pytest.mark.parametrize("kind", ["profile", "error", "ordered_output"])
def test_every_local_producer_signals_after_publication(kind):
    notifications = []
    worker = producer(SimpleNamespace(signal=lambda: notifications.append(
        (worker.profile_queue.qsize(), worker.out_queue.qsize()))))
    if kind == "profile":
        worker._process_input(PreprocessInput(request_id="r", text=None,
                              file_paths=None, input_modalities=[], output_modalities=[], model_kwargs={}))
        assert notifications == [(1, 0)]
    elif kind == "error":
        worker._fail_request("r", ValueError("bad input"), "preprocessing")
        assert notifications == [(0, 1)]
    else:
        second = ResultChunk(request_id="r", modality="text", data=b"second")
        first = ResultChunk(request_id="r", modality="text", data=b"first")
        worker._queue_completed_output("r", 1, second)
        assert not notifications  # held reorder entries are not yet readable
        worker._queue_completed_output("r", 0, first)
        assert notifications == [(0, 1), (0, 2)]
        assert worker.out_queue.get_nowait() is first
        assert worker.out_queue.get_nowait() is second
