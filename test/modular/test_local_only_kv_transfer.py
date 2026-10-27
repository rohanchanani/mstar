from types import SimpleNamespace

import pytest

from mstar.engine.resources.kv.transfer import KVTransferManager, LocalOnlyKVTransferEngine


def test_no_transport_needed_for_local_only_kv():
    manager = KVTransferManager(None, SimpleNamespace(page_size=128, num_layers=2), needs_remote_transfer=False)
    assert (manager._page_size, manager._num_layers) == (128, 2)
    engine = manager._kv_transfer_engine
    assert isinstance(engine, LocalOnlyKVTransferEngine)
    assert engine.read_batched_async(None, []) is None
    with pytest.raises(RuntimeError, match="Cross-worker KV migration"):
        engine.read_batched_async(None, [object()])
