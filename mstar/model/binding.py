"""Runtime facts supplied to an optional node implementation binder."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from mstar.engine.execution import ExecutionCapabilities

if TYPE_CHECKING:
    from mstar.engine.resources.base import NodeResourceSpec
    from mstar.model.base import Model
    from mstar.model.submodule_base import NodeSubmodule


def resolve_autocast_dtype(model: "Model", model_config: Mapping[str, Any]) -> torch.dtype | None:
    """Resolve the deployment override, including an explicit disabled policy."""
    dtype = model_config["autocast_dtype"] if "autocast_dtype" in model_config else model.get_autocast_dtype()
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype.removeprefix("torch."))
    return dtype


@dataclass(frozen=True)
class NodeBinding:
    model: "Model"
    submodule: "NodeSubmodule"
    node_name: str
    device: torch.device
    dtype: torch.dtype
    tp_group: Any
    sp_group: Any
    model_config: Mapping[str, Any]
    resource_specs: Sequence["NodeResourceSpec"]
    # Node-scoped physical preparation shared by programs, including owners
    # with leased workspace. Keys identify implementation-owned representations.
    prepared: dict[object, Any] = field(default_factory=dict, compare=False)
    # Compiler-owned preparation context, supplied by the implementation binder.
    services: Any = field(default=None, compare=False)
    # Hosted walks, determined by worker placement; None leaves placement unspecified.
    graph_walks: tuple[str, ...] | None = None


@dataclass(frozen=True)
class NodeAssembly:
    # M* constructs resources after assembly and before binding their views.
    bind: Callable[[dict[str, Any]], ExecutionCapabilities]
    resource_requests: Mapping[str, tuple[object, ...]] = field(default_factory=dict)
    # Binder-derived stable registration identities, available before allocation
    # or preparation callbacks. No optimizer-maintained protocol description.
    execution_names: tuple[str, ...] = ()
    # Derived from explicit resource requests before allocation; execution
    # runner choice does not imply a resource representation or plan location.
    device_plan_resources: frozenset[str] = frozenset()
    # Own early programs until bind transfers them to ExecutionCapabilities.
    close: Callable[[], None] | None = None
