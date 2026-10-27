"""Shared resource construction for bound programs and their qualification."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

from mstar.engine.resources.kv.config import KVSpec, PagedKVConfig, SequenceKVConfig
from mstar.engine.resources.sequence import build_sequence_resource, check_sequence_encoding, resolve_sequence_config
from mstar.engine.resources.spec import ResourceViewConflict


def merge_resource_requests(*requests: Mapping[str, tuple[object, ...]]) -> dict[str, tuple[object, ...]]:
    """Collect every consumer's views; compatibility is resolved afterward."""
    merged = {}
    for requested in requests:
        for key, views in requested.items():
            uses = merged.setdefault(key, [])
            for view in views:
                if view not in uses:
                    uses.append(view)
    return {key: tuple(views) for key, views in merged.items()}


def resolve_bound_specs(specs, resource_requests, *, dtype, group_sizes=None):
    """Return physical spec copies; model declarations remain unchanged."""
    group_sizes = group_sizes or {}
    resolved = []
    for spec in specs:
        requests = resource_requests.get(spec.resource_key, ())
        if isinstance(spec, KVSpec) and isinstance(spec.config, SequenceKVConfig):
            spec = replace(spec, config=resolve_sequence_config(
                spec, requests, dtype=dtype,
                group_size=group_sizes.get(spec.resource_key, 1),
            ))
        elif isinstance(spec, KVSpec) and isinstance(spec.config, PagedKVConfig):
            check_sequence_encoding(spec.config, requests, dtype=dtype, key=spec.resource_key)
        elif not isinstance(spec, KVSpec):
            _common_request(spec.resource_key, requests)
        resolved.append(spec)
    return tuple(resolved)


def _common_request(key, requests):
    if not requests:
        return None
    request = requests[0]
    if any(other != request for other in requests[1:]):
        raise ResourceViewConflict(f"{key}: incompatible physical resource requests")
    return request


def build_bound_resource(spec, info, requests=(), *, device_plan_required=False,
                         default_kernel_required=False):
    """Build planned state owners independently of an implementation substrate.

    A new resource-specific representation may implement ``build(spec, info)``.
    The first built-in sequence resolver can expose several compatible views
    over one owner; unrelated custom representations must agree structurally.
    Shared attention also serving a retained reference needs an explicit common
    physical request: the default planned owner supplies no kernel.
    """
    from mstar.engine.resources.attn.config import AttentionSpec, CrossAttentionSpec
    from mstar.engine.resources.attn.planned import PlannedAttentionResource, PlannedCrossAttentionResource
    from mstar.engine.resources.position.config import PositionSpec
    from mstar.engine.resources.position.manager import RopeManager
    from mstar.engine.resources.sampler.config import SamplerSpec
    from mstar.engine.resources.sampler.resource import SamplerResource

    sequence = isinstance(spec, KVSpec) and isinstance(spec.config, (SequenceKVConfig, PagedKVConfig))
    if requests and not sequence:
        return _common_request(spec.resource_key, requests).build(spec, info)
    if sequence:
        device_plan_required = device_plan_required or any(
            getattr(request, "device_plan", False) for request in requests
        )
        return build_sequence_resource(spec, info, requests, host_plan=not device_plan_required)
    if default_kernel_required and isinstance(spec, (AttentionSpec, CrossAttentionSpec)):
        raise ResourceViewConflict(
            f"{spec.resource_key}: bound and retained reference consumers require an explicit common "
            "physical attention request"
        )
    if isinstance(spec, AttentionSpec):
        return PlannedAttentionResource.build(spec, info)
    if isinstance(spec, CrossAttentionSpec):
        return PlannedCrossAttentionResource.build(spec, info)
    if isinstance(spec, PositionSpec):
        config = info.dependency(spec.config.kv_cache).config
        return RopeManager(spec.config, info.device, config.max_seq_len,
                           config.head_dim, info.kv_dtype, host_plan=True)
    if isinstance(spec, SamplerSpec):
        return SamplerResource(spec.vocab_size, spec.enable_repetion_penalty,
                               info.device, info.joint_comm_group,
                               spec.enable_min_p, planned=True)
    return spec.resource_class.build(spec, info)
