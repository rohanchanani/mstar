"""Runtime consumer for independently prepared tensor-forward programs.

The program supplies computation and physical packing. The engine retains
logical request state, resource admission/commit, streams and completion.
"""
from __future__ import annotations

from collections.abc import Callable, Collection
from dataclasses import dataclass, field, replace
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import torch

from mstar.engine.execution import ExecutionOption, ExecutionResult, LogicalInvocation
from mstar.engine.resources.step import BucketKey

if TYPE_CHECKING:
    from mstar.model.submodule_base import BatchedModelOutput


@dataclass(frozen=True)
class CompilePolicy:
    enabled: bool = False
    mode: str | None = None
    fullgraph: bool = False
    dynamic: bool | None = None
    platform_overrides: bool = False
    fallback_to_eager: bool = False

    def compile(self, forward, *, num_tokens=None, batch_size=None):
        if not self.enabled:
            return forward
        try:
            compiled = torch.compile(
                forward, mode=self.mode, fullgraph=self.fullgraph, dynamic=self.dynamic,
            )
        except Exception:
            if not self.fallback_to_eager:
                raise
            import logging
            logging.getLogger(__name__).warning("torch.compile construction failed; using eager", exc_info=True)
            return forward
        return self.wrap(compiled, num_tokens=num_tokens, batch_size=batch_size)

    def wrap(self, compiled, *, num_tokens=None, batch_size=None):
        if self.platform_overrides:
            from mstar.platforms.compile import with_inductor_overrides
            return with_inductor_overrides(compiled, num_tokens, batch_size)
        return compiled


@dataclass(frozen=True)
class CapturePolicy:
    enabled: bool = False
    required: bool = False
    compile: CompilePolicy = field(default_factory=CompilePolicy)

    def __post_init__(self):
        if self.required and not self.enabled:
            raise ValueError("required capture must be enabled")


@dataclass(frozen=True)
class TensorForward:
    program: torch.nn.Module
    compile: CompilePolicy = field(default_factory=CompilePolicy)
    capture: CapturePolicy = field(default_factory=CapturePolicy)
    autocast_dtype: torch.dtype | None = None
    # Physical fields become a typed logical result only after padding has
    # been removed and graph-owned output storage has been copied.
    result: Callable[[BatchedModelOutput, LogicalInvocation], Any] | None = None
    close: Callable[[], None] | None = None


@dataclass
class TensorForwardLease:
    runner: TensorForwardRunner
    original: LogicalInvocation | None = None
    prepared_inputs: tuple | None = None
    lease: Any = None
    running_batched: bool = False
    plan_done_event: Any = None
    launch_started_event: Any = None
    closed: bool = False

    def close(self):
        if self.closed:
            return
        if self.lease is not None:
            self.runner.graph_runner.release(self.lease, len(self.original.request_ids))
        self.closed = True


class TensorForwardRunner:
    def __init__(self, name, binding, resources, implementation: TensorForward):
        if not isinstance(implementation.program, torch.nn.Module):
            raise TypeError("TensorForward.program must be an independent torch.nn.Module")
        if implementation.program is binding.submodule:
            raise ValueError("a tensor-forward program cannot be the logical lifecycle shell")
        self.name = name
        self.binding = binding
        self.resources = resources
        self.implementation = implementation
        self.program = implementation.program
        self.program.requires_grad_(False)
        self.forward = self.program.forward
        self.forward_batched = getattr(self.program, "forward_batched", None)
        self.graph_runner = None
        self._configured = False
        self._warmed = False
        self._closed = False

    def configure(self, step_runner, *, joint_comm_group=None, num_slots=1,
                  batch_limits=None, device_capability=None, enable_nvtx=False):
        from mstar.distributed.communication import JointGroups
        from mstar.engine.accelerator_graph_runner import AcceleratorGraphRunner
        from mstar.utils.device import DeviceCapability

        if self._configured:
            raise RuntimeError("tensor-forward runner has already been configured")
        self._configured = True
        if not self.implementation.capture.enabled:
            return
        group = joint_comm_group or JointGroups(self.binding.tp_group, self.binding.sp_group)
        self.graph_runner = AcceleratorGraphRunner(
            submodule_name=self.binding.node_name,
            submodule=self.binding.submodule,
            program=self.program,
            execution_name=self.name,
            compile_policy=self.implementation.capture.compile,
            resources=self.resources,
            step_runner=step_runner,
            device=self.binding.device,
            autocast_dtype=self.implementation.autocast_dtype,
            joint_comm_group=group,
            num_slots=num_slots,
            enable_nvtx=enable_nvtx,
            device_capability=device_capability or DeviceCapability.of(self.binding.device),
            batch_limits=batch_limits,
        )

    def prepare_for_capture(self):
        if self.graph_runner is not None:
            return self.graph_runner.prepare_for_capture()
        return []

    def warmup_and_capture(self, planned: Collection[BucketKey] | None):
        if self.graph_runner is not None:
            self.graph_runner.warmup_and_capture(planned)
        self._warmed = True
        if self.implementation.capture.required and (
            self.graph_runner is None or not self.graph_runner.any_graphs
            or self.graph_runner.dropped_buckets
        ):
            missing = [] if self.graph_runner is None else self.graph_runner.dropped_buckets
            raise RuntimeError(f"{self.name}: required accelerator graph capture failed: {missing}")

    def compile_eager(self):
        policy = self.implementation.compile
        self.forward = policy.compile(self.program.forward)
        if self.forward_batched is not None:
            self.forward_batched = policy.compile(self.program.forward_batched)

    def can_batch(self, batch, inputs):
        can_batch = getattr(self.program, "can_batch", None)
        return bool(can_batch(batch, inputs)) if can_batch is not None else self.forward_batched is not None

    def acquire(self, facts):
        del facts
        return TensorForwardLease(self)

    def prepare_invocation(self, workspace, invocation, *, slot=0, cg_key_info=None,
                           lease=None, plan_done_event=None, launch_started_event=None,
                           allow_capture=True):
        """Set physical geometry before the caller declares/admit/plans effects."""
        from mstar.model.submodule_base import LazyRequestStates

        if workspace.closed or workspace.runner is not self:
            raise ValueError("invalid tensor-forward workspace")
        if workspace.original is not None:
            raise RuntimeError("tensor-forward invocation was already prepared")
        if self.implementation.capture.required and not self._warmed:
            raise RuntimeError("required capture has not been warmed")
        workspace.original = invocation
        prepare_rows = getattr(self.program, "prepare_rows", None)
        if prepare_rows is not None:
            from mstar.engine.accelerator_graph_runner import autocast_scope
            with torch.no_grad(), autocast_scope(
                self.implementation.autocast_dtype, self.binding.device.type,
            ):
                rows = tuple(prepare_rows(invocation.inputs))
            if len(rows) != len(invocation.inputs) or any(
                row.input_seq_len != original.input_seq_len
                or row.graph_walk != original.graph_walk
                or row.resource_step_info is not original.resource_step_info
                for row, original in zip(rows, invocation.inputs, strict=False)
            ):
                raise ValueError("physical row preparation changed logical geometry or effects")
            workspace.prepared_inputs = rows
            invocation = replace(invocation, inputs=rows)
        runner = self.graph_runner
        if allow_capture and lease is None and runner is not None and runner.any_graphs:
            if cg_key_info is None:
                key = getattr(self.program, "cg_key_info", self.binding.submodule.cg_key_info)
                cg_key_info = key(
                    invocation.graph_walk, invocation.engine_inputs.per_request_info,
                    per_request_input_metadata=invocation.engine_inputs.per_request_input_metadata,
                )
            lease = runner.lease_slot(
                invocation.graph_walk, len(invocation.request_ids), slot,
                sum(inp.input_seq_len for inp in invocation.inputs), cg_key_info,
            )
        workspace.lease = lease
        workspace.plan_done_event = plan_done_event
        workspace.launch_started_event = launch_started_event
        workspace.running_batched = self.can_batch(
            SimpleNamespace(graph_walk=invocation.graph_walk, request_ids=invocation.request_ids),
            invocation.inputs,
        )
        if lease is not None:
            request_ids = runner.step_ids(lease, invocation.request_ids)
            engine_inputs = replace(
                invocation.engine_inputs, request_ids=list(request_ids), captured=True,
                per_request_info=runner.step_metadata(
                    lease, invocation.request_ids, invocation.engine_inputs.per_request_info,
                ),
                per_request_states=LazyRequestStates(self.binding.submodule, request_ids),
            )
            return replace(
                invocation, request_ids=request_ids,
                inputs=runner.pad_inputs(lease, list(invocation.inputs)), engine_inputs=engine_inputs,
                row_walks=tuple(invocation.graph_walk for _ in request_ids),
            )
        if not workspace.running_batched and len(invocation.request_ids) != 1:
            raise ValueError(
                f"{self.name}: the selected tensor-forward program cannot execute this "
                "multi-row invocation eagerly; use the engine's per-request dispatch "
                "or implement batching"
            )
        return invocation

    def materialize(self, invocation, execution):
        from mstar.engine.accelerator_graph_runner import autocast_scope

        workspace = execution.workspace
        if workspace.original is None:
            if invocation.engine_inputs.step is not None and self.graph_runner is not None:
                raise ValueError("prepare tensor-forward geometry before resource declaration and plan")
            invocation = self.prepare_invocation(workspace, invocation)
        with torch.no_grad(), autocast_scope(
            self.implementation.autocast_dtype, self.binding.device.type,
        ):
            values = self.program.preprocess(
                invocation.graph_walk, engine_inputs=invocation.engine_inputs,
                inputs=invocation.inputs,
            )
        if not isinstance(values, dict):
            raise TypeError("tensor-forward preprocess must return keyword arguments")
        return invocation, values

    def run(self, call, execution):
        from mstar.engine.accelerator_graph_runner import autocast_scope

        workspace = execution.workspace
        with torch.no_grad(), autocast_scope(
            self.implementation.autocast_dtype, self.binding.device.type,
        ):
            raw = self._submit(call, execution)
            outputs = self.collect_outputs(raw, workspace.original, workspace.lease)
            if self.implementation.result is None:
                return ExecutionResult(outputs)
            return self.implementation.result(outputs, workspace.original)

    def submit_step(self, invocation, workspace, stream):
        """Submit tensor work; the engine commits before collecting outputs."""
        from mstar.engine.execution import Execution

        if not self.binding.submodule.execution_result_is_output_only:
            raise RuntimeError("output mapping may return resource effects")
        if workspace.closed:
            raise RuntimeError("tensor-forward workspace is closed")
        execution = Execution(stream=stream, workspace=workspace)
        return self.submit(self.materialize(invocation, execution), execution)

    def map_outputs(self, outputs, workspace):
        """Validate and interpret owned outputs without changing resource state."""
        result = (ExecutionResult(outputs) if self.implementation.result is None
                  else self.implementation.result(outputs, workspace.original))
        if not isinstance(result, ExecutionResult):
            result = self.binding.submodule.execution_result(result, workspace.original)
        if result.resource_results:
            raise ValueError("output-only result mapping returned resource effects")
        return result.outputs

    def submit(self, call, execution):
        from mstar.engine.accelerator_graph_runner import autocast_scope

        with torch.no_grad(), autocast_scope(
            self.implementation.autocast_dtype, self.binding.device.type,
        ):
            return self._submit(call, execution)

    def _submit(self, call, execution):
        from mstar.model.submodule_base import BatchedModelOutput

        invocation, values = call
        workspace = execution.workspace
        if workspace.lease is not None:
            raw = self.graph_runner.run_forward(
                workspace.lease, values,
                plan_done_event=workspace.plan_done_event,
                launch_started_event=workspace.launch_started_event,
            )
        else:
            if workspace.launch_started_event is not None:
                workspace.launch_started_event.set()
            forward = self.forward_batched if workspace.running_batched else self.forward
            raw = forward(invocation.graph_walk, engine_inputs=invocation.engine_inputs, **values)
            if not workspace.running_batched and not isinstance(raw, BatchedModelOutput):
                raw = {invocation.request_ids[0]: raw}
        raw = BatchedModelOutput.coerce(raw)
        original = workspace.original
        original = replace(
            original, plan_results=invocation.plan_results,
            engine_inputs=replace(
                invocation.engine_inputs, request_ids=list(original.request_ids),
                per_request_info=original.engine_inputs.per_request_info,
                per_request_states=original.engine_inputs.per_request_states,
            ),
        )
        workspace.original = original
        return raw

    def collect_outputs(self, raw, invocation, lease):
        """Own the real rows before returning graph buffers to the next replay."""
        from mstar.model.submodule_base import BatchedModelOutput

        if self.binding.submodule.execution_result_is_output_only:
            from mstar.engine.output_collection import collect_outputs
            return collect_outputs(
                self.program, self.graph_runner, lease, raw, invocation.inputs,
                invocation.engine_inputs.per_request_info,
                list(invocation.request_ids), tuple(invocation.request_ids),
            )
        request_ids = tuple(invocation.request_ids)
        out_ids = request_ids if lease is None else self.graph_runner.slot_for(lease).dummy_rids
        rows = raw.clone_row_outputs(len(request_ids))
        rows = {name: value.clone() if isinstance(value, torch.Tensor) and value.ndim == 0 else value
                for name, value in rows.items()}
        per_rid = {}
        filter_output = getattr(self.program, "filter_batched_output", None)
        for i, (rid, out_id) in enumerate(zip(request_ids, out_ids, strict=False)):
            values = {name: [tensor[i:i + 1]] for name, tensor in rows.items()
                      if isinstance(tensor, torch.Tensor) and tensor.ndim and i < tensor.shape[0]}
            values.update(_clone_tensors(raw.per_rid_outputs.get(out_id, {})))
            if filter_output is not None:
                values = filter_output(invocation.engine_inputs.per_request_info.get(rid), values)
            if values:
                per_rid[rid] = values
        unpack = getattr(self.program, "unpack_packed_outputs", None)
        if raw.packed_outputs and unpack is not None:
            unpacked = unpack(
                static_output=raw.packed_outputs, request_ids=list(request_ids),
                real_seq_lens=[inp.input_seq_len for inp in invocation.inputs],
                inputs=list(invocation.inputs), per_request_info=invocation.engine_inputs.per_request_info,
            )
            for rid, values in unpacked.items():
                per_rid.setdefault(rid, {}).update(_clone_tensors(values))
        return BatchedModelOutput(
            per_rid_outputs=per_rid,
            row_outputs=rows if filter_output is None else None,
            packed_outputs=_clone_tensors(raw.packed_outputs) if unpack is None else {},
            check_stop_buffers=raw.clone_check_stop_buffers(reuse=rows),
            row_request_ids=request_ids,
        )

    def close(self):
        if self._closed:
            return
        if self.implementation.close is not None:
            self.implementation.close()
        self.graph_runner = None
        self._closed = True


def _clone_tensors(value):
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, dict):
        return {key: _clone_tensors(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_tensors(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_tensors(item) for item in value)
    return value


def bind_tensor_forward(name, supports, binding, resources, implementation, *, cost=None):
    runner = TensorForwardRunner(name, binding, resources, implementation)
    return ExecutionOption(
        name, supports, runner.acquire, runner.materialize, runner.run,
        cost=cost, close=runner.close, runner=runner,
    )
