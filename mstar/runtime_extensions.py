"""Optional implementation binding and native-worker lifecycle extension.

The core owns orchestration.  A backend supplies its actual health, drain and
release operations. Forced process exit does not acknowledge native release.
"""

import gc
import importlib
import logging
import signal
import threading

import torch

from mstar.utils.orphan import exit_when_orphaned


class BackendFatalError(RuntimeError):
    """The backend cannot establish ordinary request-level recovery."""


def load_runtime_extension(model_config):
    spec = model_config.get("runtime_extension")
    if spec is None:
        return None
    module, name = spec["factory"].split(":", 1)
    return getattr(importlib.import_module(module), name)(spec)


def extension_selects_node(model_config, node_name):
    """Explicit node selection; omitted selection preserves all-node binding."""
    spec = model_config.get("runtime_extension") or {}
    nodes = spec.get("nodes")
    if nodes is None:
        return True
    if not isinstance(nodes, list) or not nodes or any(not isinstance(name, str) for name in nodes):
        raise ValueError("runtime_extension.nodes must be a nonempty list of node names")
    if len(nodes) != len(set(nodes)):
        raise ValueError("runtime_extension.nodes contains duplicate node names")
    return node_name in nodes


def run_managed_worker(worker_class, kwargs, extension, owner, *, cleanup_worker=None):
    """Drain native work on normal shutdown; use upstream orphan escalation."""
    logger = logging.getLogger(__name__)
    stopped = threading.Event()
    worker = None

    def stop(_signum, _frame):
        stopped.set()
        if worker is not None:
            worker.request_stop()

    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, stop)

    def before_hard_exit():
        try:
            if owner is not None:
                owner.unknown("orphaned worker has not acknowledged native release")
        finally:
            if cleanup_worker is not None and worker is not None:
                cleanup_worker(worker)

    threading.Thread(
        target=exit_when_orphaned,
        args=(f"Worker {kwargs['worker_id']}", "conductor", signal.SIGTERM),
        kwargs={"before_hard_exit": before_hard_exit},
        daemon=True, name="parent-watch",
    ).start()

    def hold(reason):
        # Retain uncertain work while supervised. Parent death still follows
        # the upstream watchdog's hard-exit policy.
        for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(signum, signal.SIG_IGN)
        try:
            extension.hold(reason)
        except BaseException:
            # A failed status write cannot authorize destruction of storage
            # whose last physical use is unknown. The launcher's inflight
            # marker and inherited physical lease still cover this process.
            logger.exception("Could not persist backend-held status")
        logger.critical("BACKEND_HELD: %s; supervised recovery required", reason)
        retained = threading.Event()
        while True:
            retained.wait(30)

    failure = None
    try:
        device = torch.device(kwargs["device"])
        extension.initialize(device, owner)
        # Retain partial construction for ordered teardown if __init__ fails.
        worker = worker_class.__new__(worker_class)
        worker_class.__init__(worker, **{**kwargs, "device": device, "runtime_extension": extension})
        if not stopped.is_set():
            worker.run()
    except BaseException as error:
        logger.exception("Managed worker stopped with an error")
        if isinstance(error, BackendFatalError) and worker is not None and hasattr(worker, "parallel_groups"):
            worker.parallel_groups.mark_failed()
        if extension.is_fatal():
            hold(f"{type(error).__name__}: {error}")
        failure = f"{type(error).__name__}: {error}"

    try:
        if worker is not None:
            worker.close()
        else:
            extension.drain()
        worker = None
        # run() freezes its setup heap.  Drop closed programs/resources and
        # collect their cycles before asking the backend to prove zero owners.
        gc.unfreeze()
        gc.collect()
        extension.close()
    except BaseException as error:
        logger.exception("Managed worker could not establish native release")
        if owner is None:
            raise RuntimeError(f"worker shutdown failed: {error}") from error
        hold(f"shutdown: {type(error).__name__}: {error}")
    if failure is not None:
        raise RuntimeError(failure)
