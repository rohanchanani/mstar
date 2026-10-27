"""Resolve implementation sequence views against logical geometry and a budget."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import torch

from mstar.engine.resources.kv.config import (
    KVLayout,
    KVSpec,
    ResolvedSequenceConfig,
    SequenceKVConfig,
)
from mstar.engine.resources.kv.encoding import KVEncoding, PlainKVEncoding, plain_kv_dtype
from mstar.engine.resources.spec import ResourceViewConflict


@dataclass(frozen=True)
class SequenceBudget:
    """Deployment budget for tensor payload; allocator overhead is separate."""

    max_bytes: int
    prefix_cache: bool = False
    cpu_offload_bytes: int = 0
    prefix_cache_salt: str = ""

    def __post_init__(self):
        if self.max_bytes <= 0 or self.cpu_offload_bytes < 0:
            raise ValueError("sequence storage needs a positive byte budget")


@dataclass(frozen=True)
class Replicated:
    """Every participant holds the logical head geometry."""


@dataclass(frozen=True)
class HeadSharded:
    """Heads are partitioned over this resource's bound participant group."""


@dataclass(frozen=True)
class PagedSequenceView:
    """NHD access whose addresses are derived from the resolved block size."""

    page_size: int | None = None
    partition: Replicated | HeadSharded = Replicated()
    encoding: KVEncoding | None = None
    # A compatible consumer may require capturable device addressing.
    device_plan: bool = field(default=False, kw_only=True)


@dataclass(frozen=True)
class DenseSequenceView:
    """One contiguous, full-capacity K/V block for each resident sequence."""

    partition: Replicated | HeadSharded = Replicated()
    encoding: KVEncoding | None = None
    # A compatible consumer may require capturable device addressing.
    device_plan: bool = field(default=False, kw_only=True)


@dataclass(frozen=True)
class SequenceEncoding:
    """An encoding requirement without adding access geometry or a partition.

    A binder uses the inherited requirement for a consumer that omitted a
    physical declaration; omission must not disappear when merging consumers.
    """

    encoding: KVEncoding | None = None


SequenceViewRequest = PagedSequenceView | DenseSequenceView | SequenceEncoding


def _resolve_encoding(key, requests, inherited):
    encodings = set()
    for request in requests:
        if not isinstance(request, (PagedSequenceView, DenseSequenceView, SequenceEncoding)):
            raise TypeError(f"{key}: unsupported sequence view request {request!r}")
        encoding = request.encoding
        if encoding is not None and (
            type(encoding) is not PlainKVEncoding or encoding.dtype not in (torch.float32, torch.float16)
        ):
            raise TypeError(f"{key}: unsupported explicit sequence encoding {encoding!r}")
        encodings.add(inherited if encoding is None else encoding)
    if len(encodings) > 1:
        raise ResourceViewConflict(f"{key}: incompatible sequence encodings {encodings}")
    return next(iter(encodings), inherited)


def check_sequence_encoding(config, requests, *, dtype, key):
    """Validate requests against a previously resolved or legacy allocation."""
    if not requests:
        return
    inherited = (config.inherited_encoding if isinstance(config, ResolvedSequenceConfig)
                 else PlainKVEncoding(torch.get_default_dtype() if dtype is None else dtype))
    actual = config.encoding if isinstance(config, ResolvedSequenceConfig) else inherited
    if _resolve_encoding(key, requests, inherited) != actual:
        raise ResourceViewConflict(f"{key}: requested encoding contradicts resolved sequence storage")


def resolve_sequence_config(
    spec: KVSpec, requests: Sequence[SequenceViewRequest], *,
    dtype: torch.dtype, group_size: int = 1,
) -> ResolvedSequenceConfig:
    """Choose one physical representation satisfying every requested view."""
    logical = spec.config
    if not isinstance(logical, SequenceKVConfig):
        raise TypeError(f"{spec.resource_key}: expected logical sequence geometry")
    budget = spec.budget
    if budget is None:
        raise ValueError(f"{spec.resource_key}: deployment needs a max_bytes storage budget")
    requests = tuple(requests) or (PagedSequenceView(),)
    inherited = PlainKVEncoding(torch.get_default_dtype() if dtype is None else dtype)
    encoding = _resolve_encoding(spec.resource_key, requests, inherited)
    views = tuple(request for request in requests if not isinstance(request, SequenceEncoding))
    partitions = {request.partition for request in views} or {Replicated()}
    if len(partitions) != 1:
        raise ResourceViewConflict(f"{spec.resource_key}: incompatible sequence partitions {partitions}")
    partition = next(iter(partitions))
    if not isinstance(partition, (Replicated, HeadSharded)):
        raise TypeError(f"{spec.resource_key}: unsupported sequence partition {partition}")
    dense = any(isinstance(request, DenseSequenceView) for request in views)
    sizes = {request.page_size for request in views
             if isinstance(request, PagedSequenceView) and request.page_size is not None}
    if dense:
        sizes.add(logical.max_seq_len)
    if len(sizes) > 1:
        raise ResourceViewConflict(f"{spec.resource_key}: incompatible sequence block sizes {sorted(sizes)}")
    page_size = next(iter(sizes), 128)
    if page_size <= 0:
        raise ValueError(f"{spec.resource_key}: sequence block size must be positive")
    if dense and (budget.prefix_cache or budget.cpu_offload_bytes):
        raise ResourceViewConflict(
            f"{spec.resource_key}: dense sequence storage currently requires resident, unshared state"
        )
    config = ResolvedSequenceConfig(
        num_layers=logical.num_layers, num_kv_heads=logical.num_kv_heads,
        head_dim=logical.head_dim, num_qo_heads=logical.num_qo_heads,
        max_seq_len=logical.max_seq_len, page_size=page_size,
        layout=KVLayout.NHD, prefix_cache=budget.prefix_cache,
        prefix_cache_salt=budget.prefix_cache_salt, dense=dense,
        encoding=encoding, inherited_encoding=inherited,
    )
    if isinstance(partition, HeadSharded):
        config.shard(group_size)
        config.partition = "head_sharded"
    config.max_num_pages = budget.max_bytes // config.page_bytes
    config.cpu_offload_pages = budget.cpu_offload_bytes // config.page_bytes
    if config.max_num_pages < 2:
        raise ValueError(f"{spec.resource_key}: byte budget cannot hold a sink block and one live block")
    return config


def build_sequence_resource(spec: KVSpec, info, requests=(), *, host_plan=True):
    """Construct the existing state owner using a resolved local representation."""
    from mstar.engine.resources.kv.manager import KVManager

    config = spec.config
    if isinstance(config, SequenceKVConfig):
        group_size = info.joint_comm_group.world_size if info.joint_comm_group is not None else 1
        config = resolve_sequence_config(spec, requests, dtype=info.kv_dtype, group_size=group_size)
    check_sequence_encoding(config, requests, dtype=info.kv_dtype, key=spec.resource_key)
    if isinstance(config, ResolvedSequenceConfig) and config.dense and info.needs_remote_transfer:
        raise ResourceViewConflict(f"{spec.resource_key}: dense sequence storage currently requires local state")
    if (isinstance(config, ResolvedSequenceConfig) and info.needs_remote_transfer
            and config.encoding != config.inherited_encoding):
        raise ResourceViewConflict(
            f"{spec.resource_key}: a KV encoding override requires local state; "
            "remote transfer has no encoding agreement"
        )
    storage_dtype = plain_kv_dtype(config.encoding) if isinstance(config, ResolvedSequenceConfig) else info.kv_dtype
    return KVManager(
        config, spec.resource_key, info.joint_comm_group, info.transfer_engine_info,
        info.device, storage_dtype, needs_remote_transfer=info.needs_remote_transfer,
        host_plan=host_plan,
    )
