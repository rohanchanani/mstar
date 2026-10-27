"""Partition registration must preserve the position of resident KV history."""
import torch

from mstar.engine.resources.position.config import PositionConfig, PositionStep
from mstar.engine.resources.position.manager import RopeManager
from mstar.engine.resources.step import Segment, StepContext


def manager():
    return RopeManager(PositionConfig(kv_cache="kv"), torch.device("cpu"), max_seq_len=64)


def advance(resource, span, label="main"):
    resource.commit(
        PositionStep(segments=(Segment("r0", label, span),)),
        StepContext(request_ids=("r0",), graph_walk="decode", slot=0, capture=False),
    )


def test_duplicate_partition_registration_preserves_live_positions():
    resource = manager()
    resource.ingest_request("r0")
    advance(resource, 16)
    advance(resource, 3, "other")
    resource.ingest_request("r0")
    assert resource.position("r0", "main") == 16
    assert resource.position("r0", "other") == 3
    assert resource._build_pos_ids([("r0", "main", 2)], "main", None).tolist() == [16, 17]
    advance(resource, 1)
    assert resource.position("r0", "main") == 17


def test_registration_before_history_leaves_positions_at_zero():
    resource = manager()
    resource.ingest_request("r0")
    resource.ingest_request("r0")
    assert resource.position("r0", "main") == 0


def test_explicit_reset_and_removed_request_start_fresh():
    resource = manager()
    resource.ingest_request("r0")
    advance(resource, 16)
    resource.reset_request("r0")
    assert resource.position("r0", "main") == 0
    advance(resource, 3)
    resource.remove_request("r0")
    resource.ingest_request("r0")
    assert resource.position("r0", "main") == 0
