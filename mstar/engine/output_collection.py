"""Runtime output ownership, row mapping and packed-output collection."""
from typing import Any, Mapping

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.resources import SlotLease
from mstar.model.submodule_base import BatchedModelOutput, NodeInputs, NodeSubmodule


def collect_outputs(
    submodule,
    graph_runner,
    lease: SlotLease | None,
    raw_outputs: BatchedModelOutput,
    inputs: list[NodeInputs],
    req_info: Mapping[str, CurrentForwardPassInfo],
    request_ids: list[str],
    step_request_ids: tuple[str, ...],
    rows_only: bool = False,
    copy_outputs: bool = True,
    logical=None,
) -> BatchedModelOutput:
    """Per-rid outputs for the real requests: drop the padding rows and map
    a captured graph's keys back to real ids.

    ``rows_only`` (see ``ExecutingBatch.rows_only``) keeps the row views
    and the stop buffers but builds no per-rid dicts: nothing downstream
    reads them on such a step, and at 128 rows they were a few hundred
    allocations on the gpu thread.

    A captured forward emits its per-rid entries under the slot's padding
    ids (those were the batch at capture time), so entry ``i`` belongs to
    ``request_ids[i]`` on either path.
    """
    logical = submodule if logical is None else logical
    if (
        rows_only
        and raw_outputs.check_stop_buffers is not None
        and not raw_outputs.per_rid_outputs
        and not raw_outputs.packed_outputs
        and getattr(type(logical), "filter_batched_output", NodeSubmodule.filter_batched_output)
        is NodeSubmodule.filter_batched_output
    ):
        # No clone either: the worker takes the host copy on this thread,
        # on this stream, right behind the step (``_stage_host_rows``), so
        # a later replay cannot overwrite the rows first, and nothing else
        # reads them on the device. Padded rows past the real ones ride
        # along and are never read.
        return BatchedModelOutput(
            check_stop_buffers=raw_outputs.check_stop_buffers,
            row_request_ids=tuple(request_ids),
            rows_only=True,
        )
    out_ids = (
        step_request_ids if lease is None
        else graph_runner.slot_for(lease).dummy_rids
    )
    outputs: dict[str, NameToTensorList] = {}

    row_clones, row_views = _merge_per_rid(
        outputs, raw_outputs, request_ids, out_ids, logical, req_info,
        rows_only=rows_only, copy_outputs=copy_outputs,
    )
    _merge_unpacked(
        outputs, raw_outputs, request_ids, submodule,
        inputs[:len(request_ids)], req_info,
    )
    return BatchedModelOutput(
        per_rid_outputs=outputs,
        rows_only=rows_only and not outputs,
        # the stop buffer that is a row output shares its clone: one copy
        # of the sampled tokens per step, not two
        check_stop_buffers=(raw_outputs.clone_check_stop_buffers(reuse=row_clones)
                            if copy_outputs else raw_outputs.check_stop_buffers),
        # row i is request_ids[i] as the forward ran them, not the
        # worker's later-rewritten request list
        row_request_ids=tuple(request_ids),
        row_views=row_views,
    )

def _merge_per_rid(
    outputs: dict[str, NameToTensorList],
    raw_outputs: BatchedModelOutput,
    request_ids: list[str],
    out_ids: list[str],
    submodule: NodeSubmodule,
    req_info: Mapping[str, CurrentForwardPassInfo],
    rows_only: bool = False,
    copy_outputs: bool = True,
) -> tuple[dict[str, torch.Tensor], dict[str, tuple[torch.Tensor, ...]] | None]:
    """Fold the forward's per-rid entries into ``outputs``.

    ``row_outputs`` are cloned once for the batch and handed out as row
    views; per-rid entries are cloned individually after
    ``filter_batched_output``. A name in both resolves to the per-rid one.
    Returns the row clones so the caller can share them, and on the
    rows-only path the views themselves (name -> views, row i for
    ``request_ids[i]``), which the store describes without reading each.
    """
    # Tensor-forward has already taken ownership before adapting logical
    # outputs. Other consumers may still return leased workspace buffers.
    row_clones = (raw_outputs.clone_row_outputs(len(request_ids)) if copy_outputs
                  else raw_outputs.row_outputs or {})
    # The common decode case: only row outputs, and a submodule that does
    # not filter per request. One ``split`` per output makes every row
    # view in a single call, the same ``tensor[i:i+1]`` views the loop
    # below would slice one at a time, and the per-rid dicts are built
    # without the candidate/from_rows bookkeeping.
    if (
        not raw_outputs.per_rid_outputs
        and getattr(type(submodule), "filter_batched_output", NodeSubmodule.filter_batched_output)
        is NodeSubmodule.filter_batched_output
    ):
        row_views = {
            name: tensor.split(1)
            for name, tensor in row_clones.items()
            if isinstance(tensor, torch.Tensor) and tensor.dim()
        }
        if row_views and not rows_only:
            for i, rid in enumerate(request_ids):
                merged = outputs.setdefault(rid, {})
                for name, views in row_views.items():
                    if i < len(views):
                        merged[name] = [views[i]]
        return row_clones, (row_views or None)
    for i, (rid, out_id) in enumerate(
        zip(request_ids, out_ids, strict=False)
    ):
        rid_out = raw_outputs.get(out_id)
        # names served by a view of the batch clone: don't clone again
        from_rows: set[str] = set()
        candidates: dict[str, Any] = {}
        for name, tensor in row_clones.items():
            if isinstance(tensor, torch.Tensor) and tensor.dim() and (
                i < tensor.shape[0]
            ):
                candidates[name] = [tensor[i : i + 1]]
                from_rows.add(name)
        if isinstance(rid_out, dict):
            for name, value in rid_out.items():
                candidates[name] = value
                from_rows.discard(name)
        if not candidates:
            continue
        # captured output keys are fixed for graph compat; the submodule
        # decides which of them this real request should receive
        filter_output = getattr(submodule, "filter_batched_output", None)
        rid_out = candidates if filter_output is None else filter_output(req_info.get(rid), candidates)
        merged = outputs.setdefault(rid, {})
        for key, value in rid_out.items():
            if key in from_rows:
                merged[key] = value
            elif isinstance(value, list):
                merged[key] = [t.clone() for t in value] if copy_outputs else value
            elif isinstance(value, torch.Tensor):
                merged[key] = [value.clone() if copy_outputs else value]
            else:
                merged[key] = value
    return row_clones, None

def _merge_unpacked(
    outputs: dict[str, NameToTensorList],
    raw_outputs: BatchedModelOutput,
    request_ids: list[str],
    submodule: NodeSubmodule,
    real_inputs: list[NodeInputs],
    req_info: Mapping[str, CurrentForwardPassInfo],
) -> None:
    """Let the submodule slice any batch-wide packed sentinels.

    A captured region can't do this itself: the per-request slice ends
    depend on the real seq_lens, which only reach it through the plan.
    """
    unpack = getattr(submodule, "unpack_packed_outputs", None)
    if unpack is None or (
        not raw_outputs.packed_outputs
        and type(submodule).unpack_packed_outputs is NodeSubmodule.unpack_packed_outputs
    ):
        # nothing packed and the base no-op: skip the per-rid lengths
        return
    unpacked = unpack(
        static_output=raw_outputs.packed_outputs,
        request_ids=request_ids,
        real_seq_lens=[inp.input_seq_len for inp in real_inputs],
        inputs=real_inputs,
        per_request_info=req_info,
    )
    for rid, rid_out in unpacked.items():
        outputs.setdefault(rid, {}).update(rid_out)
