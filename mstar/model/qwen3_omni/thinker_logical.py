"""Logical Thinker request interpretation and transitions."""
from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Protocol

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.execution import ExecutionResult
from mstar.engine.resources import AttentionStep, KVStep, PositionStep, SamplerStep, Segment, SlotLease, SubmoduleStep
from mstar.model.contract import NodeContract, TensorInput, TensorOutput
from mstar.model.qwen3_omni.components.rope import get_rope_index_audio, get_rope_index_text, get_rope_index_vision
from mstar.model.qwen3_omni.config import THINKER_ATTN, THINKER_KV, THINKER_POS, THINKER_SAMPLER, Qwen3OmniModelConfig
from mstar.model.qwen3_omni.results import ThinkerResult
from mstar.model.submodule_base import ARNodeInputs, ARNodeSubmodule, BatchedModelOutput

logger = logging.getLogger(__name__)

class ThinkerInputPreparation(Protocol):
    """Physical operations requested in model-defined interpretation order."""
    def embed_tokens(self, token_ids: torch.Tensor) -> torch.Tensor: ...
    def wrap_media(self, values: torch.Tensor, *, start_token_id: int,
                   end_token_id: int) -> torch.Tensor: ...


def talker_text_mask(input_ids: torch.Tensor, config: Qwen3OmniModelConfig) -> torch.Tensor:
    """
    Cut system prompt and previous assistant parts out of the talker input
    """
    im_start_indexes = (
        input_ids == config.im_start_token_id
    ).nonzero(as_tuple=True)[0]
    mask = torch.ones(input_ids.shape, dtype=torch.bool, device=input_ids.device)

    for i in range(len(im_start_indexes) - 1):
        im_start_index = im_start_indexes[i]
        segment_end_index = im_start_indexes[i + 1]
        role_token = input_ids[im_start_index + 1]
        # Talker should ignore thinker system prompt
        if role_token == config.system_token_id:
            mask[im_start_index:segment_end_index] = 0
        elif role_token == config.assistant_token_id:
            mask[im_start_index:segment_end_index] = 0
    return mask


class ThinkerLogical(ARNodeSubmodule):
    """Logical request interpretation and state transitions; no numerical program."""
    DEEPSTACK_KEY = "deepstack"
    execution_result_is_output_only = True

    def __init__(self, config):
        super().__init__()
        self.config = config
        self._decode_thinker_mask = None

    @staticmethod
    def _token_present(logical, walk, info):
        return walk == "thinker_decode" or info.step_metadata.get("is_last_prefill", False)

    @staticmethod
    def _conditioning_present(logical, walk, info):
        return info.step_metadata.get("audio_output", True)

    @staticmethod
    def _dimensions(logical, walk, info):
        return {"media_tokens": max(0, logical.input_seq_len - 2)}

    def logical_contract(self):
        h = self.config.thinker_hidden_size
        return NodeContract(
            inputs=(TensorInput("input_ids", "input_ids", ("tokens",), dtype=torch.long,
                                walks=("prefill_text", "thinker_decode")),
                    TensorInput("media_embeds", "input_embeds", ("media_tokens", h),
                                walks=("prefill_audio", "prefill_vision")),
                    TensorInput("positions", "custom_pos_ids", (3, "tokens"), dtype=torch.float32),
                    TensorInput("masks", "tensor_inputs.masks_for_talker", (2, "tokens"), dtype=torch.bool),
                    *(TensorInput(f"deepstack_{i}", f"tensor_inputs.deepstack_{i}", ("tokens", h),
                                  walks=("prefill_vision",))
                      for i in range(len(self.config.vision.deepstack_visual_indexes)))),
            outputs=(TensorOutput("new_token", (1,), dtype=(torch.int64, torch.int32), per_request=True,
                                  present_when=self._token_present),
                     TensorOutput("thinker_states", ("tokens", 2*h), per_request=True,
                                  present_when=self._conditioning_present),
                     TensorOutput("thinker_mask", (2, "tokens"), dtype=torch.bool, per_request=True,
                                  present_when=self._conditioning_present)),
            dimensions=self._dimensions, result_type=ThinkerResult, weight_prefix="thinker.",
            input_preparation=ThinkerInputPreparation)

    def preprocess(self, *args, **kwargs):
        raise RuntimeError("physical preprocessing belongs to the selected implementation")

    def forward(self, *args, **kwargs):
        raise RuntimeError("select a physical Thinker implementation")

    def execution_result(self, result, invocation):
        if not isinstance(result, ThinkerResult):
            return super().execution_result(result, invocation)
        return ExecutionResult(BatchedModelOutput(per_rid_outputs={
            rid: {name: [getattr(result, name)[i]]
                  for name in ("new_token", "thinker_states", "thinker_mask")
                  if getattr(result, name)[i] is not None}
            for i, rid in enumerate(invocation.request_ids)}))

    def _get_decode_thinker_mask(self, device: torch.device) -> torch.Tensor:
        """Return the constant ``[[0], [1]]`` decode mask (multimodal row =
        0, text-inclusion row = 1), lazily allocated per device."""
        if (
            self._decode_thinker_mask is None
            or self._decode_thinker_mask.device != device
        ):
            self._decode_thinker_mask = torch.tensor(
                [[0], [1]], dtype=torch.bool, device=device,
            )
        return self._decode_thinker_mask


    def _get_talker_text_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        return talker_text_mask(input_ids, self.config)


    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        preparation: ThinkerInputPreparation | None = None,
        **kwargs
    ) -> ARNodeInputs:
        device = self.get_device()
        start_pos = self.node_resources[THINKER_POS].position(
            rid=fwd_info.rid_handle, label="main"
        )
        if graph_walk == "thinker_decode":
            # Get previous token ID from text_inputs
            token_id = inputs["text_inputs"][0].to(device)  # (1,) or scalar
            if token_id.dim() == 0:
                token_id = token_id.unsqueeze(0)
            if preparation is not None:
                preparation.embed_tokens(token_id)

            # Next MRoPE position for all 3 components: read from the
            # stream's position counter (advanced by the runner's commit
            # of each declared step).
            # Fill kernel, not ``torch.tensor(..., device=cuda)``: a pageable H2D
            # syncs the stream, stalling prepare(N+1) until step N drains.
            pos_ids = torch.full(
                (3, 1), float(start_pos), dtype=torch.float, device=device,
            )  # (3, 1)

            return ARNodeInputs(
                input_seq_len=1,
                input_ids=token_id,
                input_embeds=None,
                custom_pos_ids=pos_ids,
                tensor_inputs={
                    "embedding_kind": "text",
                    "masks_for_talker": self._get_decode_thinker_mask(device)
                }  # no additional tensors for decode step
            )

        if graph_walk == "prefill_text":
            text_ids = inputs["text_inputs"][0].to(device)  # (seq_len,)
            if preparation is not None:
                preparation.embed_tokens(text_ids)
            seq_len = text_ids.shape[0]

            # Compute 3D MRoPE position IDs for a pure-text span.  Each
            # prefill graph walk is single-modality so we use the simple
            # per-modality helper instead of the full HF parser.
            #
            # ``start_pos`` is the next MRoPE position for this request,
            # carried forward across walks by the stream's position counter
            # (advanced by the runner's commit of each declared step).
            pos_ids = get_rope_index_text(seq_len, start_pos, device)
            masks_for_talker = torch.stack([
                torch.zeros(text_ids.shape, dtype=torch.bool, device=device), # multimodal
                self._get_talker_text_mask(text_ids) # text inclusion
            ])
            return ARNodeInputs(
                input_seq_len=seq_len,
                input_embeds=None,
                input_ids=text_ids,
                custom_pos_ids=pos_ids,
                tensor_inputs={
                    "embedding_kind": "text",
                    "masks_for_talker": masks_for_talker
                }
            )

        if graph_walk == "prefill_audio":
            audio_embeds = inputs["audio_embeds"][0].to(device)  # (audio_tokens, hidden)
            audio_len = audio_embeds.shape[0]

            mm_mask = torch.ones(audio_len + 2, dtype=torch.bool, device=device)
            mm_mask[[0, -1]] = 0
            masks_for_talker = torch.stack([
                mm_mask,
                ~mm_mask
            ])

            wrapped_embeds = (audio_embeds if preparation is None else preparation.wrap_media(
                audio_embeds, start_token_id=self.config.thinker.audio_start_token_id,
                end_token_id=self.config.thinker.audio_end_token_id))
            seq_len = audio_len + 2
            # Position IDs:
            #   - audio_start_token: text-like position at start_pos
            #   - audio tokens:      temporal increments per frame,
            #                        h/w = start_pos (handled by helper)
            #   - audio_end_token:   text-like position right after
            start_pos_ids = get_rope_index_text(1, start_pos, device)
            audio_pos_ids = get_rope_index_audio(
                audio_len,
                start_pos + 1,
                device,
                self.config.thinker.position_id_per_seconds,
            )
            end_pos_ids = get_rope_index_text(
                1, start_pos + 1 + audio_len, device
            )
            pos_ids = torch.cat(
                [start_pos_ids, audio_pos_ids, end_pos_ids], dim=1
            )
            return ARNodeInputs(
                input_seq_len=seq_len,
                input_embeds=audio_embeds,
                custom_pos_ids=pos_ids,
                tensor_inputs={
                    "embedding_kind": "audio",
                    "masks_for_talker": masks_for_talker
                }
            )

        if graph_walk == "prefill_vision":
            vision_embeds = inputs["vision_embeds"][0].to(device)
            vision_len = vision_embeds.shape[0]

            mm_mask = torch.ones(vision_len + 2, dtype=torch.bool, device=device)
            mm_mask[[0, -1]] = 0
            masks_for_talker = torch.stack([
                mm_mask,
                ~mm_mask
            ])

            wrapped_embeds = (vision_embeds if preparation is None else preparation.wrap_media(
                vision_embeds, start_token_id=self.config.thinker.vision_start_token_id,
                end_token_id=self.config.thinker.vision_end_token_id))
            total_len = vision_len + 2
            # Vision tokens use spatial 3D positions (temporal constant,
            # h/w from the spatial grid after merging).  If a proper
            # ``image_grid_thw`` is available, use ``get_rope_index_vision``;
            # otherwise fall back to a 1-D sequence (test path)
            grid_thw = inputs.get("image_grid_thw", [None])[0]
            assert vision_len == int(grid_thw.prod()) // self.config.vision.spatial_merge_size ** 2, (
                f"the vision encoder gave {vision_len} rows, where the layout counts the grid's merged patches"
            )
            seconds_per_grid = inputs.get("video_second_per_grid", [])
            seconds_per_grid = seconds_per_grid[0].item() if seconds_per_grid else None
            vision_pos_ids = get_rope_index_vision(
                grid_thw.to(device),
                start_pos + 1,  # leave room for the BOS token
                position_id_per_seconds=self.config.thinker.position_id_per_seconds,
                device=device,
                spatial_merge_size=self.config.vision.spatial_merge_size,
                seconds_per_grid=seconds_per_grid
            )

            # Sentinel token positions (text-like).
            start_pos_ids = get_rope_index_text(1, start_pos, device)
            end_pos_base = float(vision_pos_ids.max().item()) + 1
            end_pos_ids = get_rope_index_text(1, end_pos_base, device)

            pos_ids = torch.cat(
                [start_pos_ids, vision_pos_ids, end_pos_ids], dim=1
            )

            # Next MRoPE position after this vision block is ``end_pos_base
            # + 1`` (one past the EOS token). A commit advances positions by
            # the span by default, which for vision (= vision_len + 2) is
            # typically smaller than the 3D-grid span. Emit the correct
            # per-request advance so ``declare_step`` can declare it.
            mrope_pos_advance = int(end_pos_base + 1 - start_pos)
            deepstack = []
            for deepstack_inp in inputs["deepstack"]:
                full_deepstack = (torch.zeros_like(wrapped_embeds) if preparation is not None
                                  else vision_embeds.new_zeros((total_len, self.config.thinker_hidden_size)))
                full_deepstack[mm_mask, :] = deepstack_inp
                deepstack.append(full_deepstack)

            return ARNodeInputs(
                input_seq_len=total_len,
                input_embeds=vision_embeds,
                custom_pos_ids=pos_ids,
                tensor_inputs={
                    "embedding_kind": "vision",
                    "masks_for_talker": masks_for_talker,
                    "mrope_pos_advance": mrope_pos_advance,
                    "deepstack": deepstack,
                    **{f"deepstack_{i}": value for i, value in enumerate(deepstack)},
                }
            )


    def declare_step(
        self, graph_walk: str,
        request_ids: list[str],
        inputs: list[ARNodeInputs],
        slot_lease: SlotLease | None = None,
        piecewise_leases: Mapping[str, SlotLease] | None = None,
        per_request_info: Mapping[int, CurrentForwardPassInfo] | None = None,
        **kwargs,
    ) -> SubmoduleStep:
        row_walks = [inp.graph_walk or graph_walk for inp in inputs]
        kept_rids = None
        if graph_walk != "thinker_decode":
            # decode rows, and the last prefill, keep their token through `postprocess`
            info = per_request_info or {}
            kept_rids = frozenset(
                rid for rid, walk in zip(request_ids, row_walks, strict=True)
                if walk == "thinker_decode" or (
                    rid in info and info[rid].step_metadata.get("is_last_prefill", False)
                )
            )
        prefill_tokens = {
            rid: inp.input_ids
            for rid, inp, walk in zip(request_ids, inputs, row_walks, strict=True)
            if walk == "prefill_text"
        }

        pos_advance = None
        if "prefill_vision" in row_walks:
            # A vision row's 3D-grid MRoPE span is larger than its token count,
            # so it advances the counter by its own; other rows by their span.
            pos_advance = tuple(
                int(inp.tensor_inputs.get("mrope_pos_advance", 0))
                if walk == "prefill_vision" else inp.input_seq_len
                for inp, walk in zip(inputs, row_walks, strict=True)
            )
        return SubmoduleStep(
            # must match `cg_key_info`, which picked the leased capture
            cg_key_info=self.DEEPSTACK_KEY if pos_advance is not None else None,
            segments=[
                Segment(
                    request_id=rid,
                    label="main",
                    span=inp.input_seq_len,
                ) for rid, inp in zip(request_ids, inputs, strict=True)
            ],
            steps={
                THINKER_KV: KVStep(),
                THINKER_ATTN: AttentionStep(causal=True),
                THINKER_SAMPLER: SamplerStep(
                    apply_penalty=True,
                    prefill_tracked_tokens=prefill_tokens,
                    kept_rids=kept_rids,
                ),
                THINKER_POS: PositionStep(
                    advance=pos_advance
                )
            },

        )


    def postprocess(
        self, request_id: str,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
        **kwargs
    ):
        return_token = request_info.graph_walk == "thinker_decode" or \
            request_info.step_metadata.get("is_last_prefill", False)
        if not return_token:
            outputs.pop("new_token", None)

        if not request_info.step_metadata.get("audio_output", True):
            # drop thinker_states and thinker_match
            outputs.pop("thinker_states", None)
            outputs.pop("thinker_mask", None)
        else:
            # Pick layer-0 for text positions and layer-N for multimodal,
            # drop positions the Talker won't consume (system prompt /
            # previous-assistant), and emit a 1D multimodal mask aligned to
            # the kept tokens. Doing this here (not in the Talker) avoids
            # shipping both hidden states for every token.
            ts_list = outputs.get("thinker_states")
            mask_list = outputs.get("thinker_mask")
            if ts_list and mask_list:
                ts = ts_list[0]                # (seq_len, 2*hidden)
                mask = mask_list[0]            # (2, seq_len)
                thinker_hidden = self.config.thinker_hidden_size
                layer_0 = ts[..., :thinker_hidden]
                layer_n = ts[..., thinker_hidden:]
                multimodal_mask = mask[0].bool()
                text_inclusion_mask = mask[1].bool()
                text_mask = text_inclusion_mask & ~multimodal_mask
                inclusion_mask = text_mask | multimodal_mask

                selected = torch.where(
                    multimodal_mask.unsqueeze(-1), layer_n, layer_0,
                )                              # (seq_len, hidden)
                outputs["thinker_states"] = [selected[inclusion_mask]]
                outputs["thinker_mask"] = [multimodal_mask[inclusion_mask]]

        if "new_token" not in outputs:
            return
        outputs["text_inputs"] = outputs["new_token"]


    def check_stop(
        self, request_id: str,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
    ) -> set[str]:
        if "new_token" not in outputs or request_info.graph_walk != "thinker_decode":
            return set()
        token = outputs["new_token"][0].item()
        ignore_eos= request_info.resource_configs[THINKER_SAMPLER].ignore_eos
        eos_token_id = self.config.im_end_token_id
        if (not ignore_eos and eos_token_id == token) or \
                (request_info.dynamic_loop_iter_counts.get("thinker_decode_loop", 0) + 1 >= request_info.max_tokens):
            return {"thinker_decode_loop"}
        return set()


    def filter_batched_output(
        self,
        request_info: CurrentForwardPassInfo,
        rid_output: dict[str, list[torch.Tensor]],
    ) -> dict[str, list[torch.Tensor]]:
        """Drop ``thinker_states`` + ``thinker_mask`` for text-only requests.

        ``forward_batched`` always emits these keys so the captured CUDA
        graph's output shape is static.  Here, outside the captured
        region, we gate them on the real request's ``audio_output`` flag
        so the Talker edge stays unrouted for text-only requests (matches
        the pre-capture eager-mode behaviour).
        """
        if request_info is None:
            return rid_output
        if request_info.step_metadata.get("audio_output", True):
            return rid_output
        return {
            k: v for k, v in rid_output.items()
            if k not in ("thinker_states", "thinker_mask")
        }

