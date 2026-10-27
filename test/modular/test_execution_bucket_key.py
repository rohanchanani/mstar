"""Distinct physical programs must not alias captured resource buffers."""
from dataclasses import replace

from mstar.engine.resources.base import CGSlotKey
from mstar.engine.resources.step import BucketKey, ExecutionBusy


def test_execution_identity_separates_capture_storage():
    ordinary = BucketKey("decode", 4, 4)
    a = replace(ordinary, execution_name="program-a")
    b = replace(ordinary, execution_name="program-b")
    assert ordinary == BucketKey("decode", 4, 4)
    assert len({CGSlotKey(k, 0, "main") for k in (ordinary, a, b)}) == 3
    assert str(ordinary) == "decode[bs=4,tokens=4]"
    assert "execution=program-a" in str(a)


def test_workspace_busy_is_retryable():
    assert not ExecutionBusy("workspace occupied").terminal
