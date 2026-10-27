"""Logical Talker interpretation, state effects and output routing.

This module is importable without selecting a numerical implementation. Raw
conditioning constants are bound independently from the physical projections.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.resources import AttentionStep, KVStep, PositionStep, SamplerStep, Segment, SlotLease, SubmoduleStep
from mstar.model.contract import NodeContract, TensorInput, TensorOutput
from mstar.model.qwen3_omni.config import (
    CODE_PRED_SAMPLER,
    TALKER_ATTN,
    TALKER_KV,
    TALKER_POS,
    TALKER_SAMPLER,
    Qwen3OmniModelConfig,
)
from mstar.model.qwen3_omni.results import TalkerResult
from mstar.model.submodule_base import ARNodeInputs, ARNodeSubmodule, BatchedModelOutput

LAST_PREFILL_TOKENS_PER_REQUEST = 6


def codec_prefix_ids(config, speaker: str, device: torch.device) -> torch.Tensor:
    tc = config.talker
    speaker_id = tc.speaker_id.get(speaker.lower(), tc.codec_pad_id)
    return torch.tensor([
        tc.codec_nothink_id, tc.codec_think_bos_id, tc.codec_think_eos_id,
        speaker_id, tc.codec_pad_id, tc.codec_bos_id,
    ], device=device, dtype=torch.long)


class TalkerSubmodule(ARNodeSubmodule):
    TALKER_LAST_PREFILL_TOKENS_PER_REQ = LAST_PREFILL_TOKENS_PER_REQUEST

    def __init__(self, config: Qwen3OmniModelConfig,
                 conditioning_source: Callable[[], torch.Tensor] | None = None):
        super().__init__()
        self.config = config
        self.cp_cfg = config.code_predictor
        self.num_codes = self.cp_cfg.num_code_groups
        self._conditioning_source = conditioning_source
        self.register_buffer("_tts_raw_embeddings", None, persistent=False)
        self._eos_embed_sent: set[int] = set()

    def bind_execution(self, binding, capabilities) -> None:
        super().bind_execution(binding, capabilities)
        if self._tts_raw_embeddings is None:
            if self._conditioning_source is None:
                raise ValueError("Talker logical binding requires its conditioning weight source")
            raw = self._conditioning_source()
            if tuple(raw.shape) != (3, self.config.thinker_hidden_size):
                raise ValueError("TTS embeddings must contain pad, bos and eos rows")
            self._tts_raw_embeddings = raw
        self._tts_raw_embeddings = self._tts_raw_embeddings.to(device=binding.device)

    def _get_last_prefill_codec_ids(self, speaker: str) -> torch.Tensor:
        return codec_prefix_ids(self.config, speaker, self.get_device())

    def forward(self, *args, **kwargs):
        raise RuntimeError("The logical Talker has no numerical implementation")

    def preprocess(self, *args, **kwargs):
        raise RuntimeError("Talker physical input packing belongs to its implementation")

    def logical_contract(self) -> NodeContract:
        return NodeContract(
            inputs=(
                TensorInput("thinker_states", "tensor_inputs.thinker_states",
                            ("tokens", self.config.thinker_hidden_size)),
                TensorInput("multimodal_mask", "tensor_inputs.multimodal_mask", ("tokens",),
                            dtype=torch.bool, walks=("talker_prefill",)),
                TensorInput("codec_ids", "tensor_inputs.codec_ids", ("tokens",),
                            dtype=torch.long, walks=("talker_last_prefill",)),
                TensorInput("codec_embeds", "input_embeds", ("tokens", self.config.talker_hidden_size),
                            dtype=torch.float32, walks=("talker_decode",)),
            ),
            outputs=(
                TensorOutput("codec_tokens", ("batch", self.num_codes), dtype=torch.long),
                TensorOutput("codec_embeds", ("batch", self.cp_cfg.hidden_size), dtype=torch.float32),
                TensorOutput("tokens", ("batch",), dtype=torch.long),
            ),
            result_type=TalkerResult,
            weight_prefix="talker.",
            suppressed_tokens={TALKER_SAMPLER: {
                walk: tuple(i for i in range(max(0, self.config.talker_text.vocab_size - 1024),
                                            self.config.talker_text.vocab_size)
                            if i != self.config.talker.codec_eos_token_id)
                for walk in ("talker_last_prefill", "talker_decode")
            }},
        )


    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs,
    ) -> ARNodeInputs:
        device = self.get_device()
        if graph_walk == "talker_prefill":
            selected_hidden = inputs["thinker_states"][0].to(device)
            return ARNodeInputs(
                input_seq_len=selected_hidden.shape[0],
                tensor_inputs={"thinker_states": selected_hidden,
                               "multimodal_mask": inputs["thinker_mask"][0].to(device)},
            )
        if graph_walk == "talker_last_prefill":
            last_hidden = inputs["thinker_states"][0].to(device)
            raw = self._tts_raw_embeddings
            return ARNodeInputs(
                input_seq_len=self.TALKER_LAST_PREFILL_TOKENS_PER_REQ,
                tensor_inputs={
                    "codec_ids": self._get_last_prefill_codec_ids(fwd_info.step_metadata.get("voice", "Ethan")),
                    "thinker_states": torch.cat([raw[0:1].expand(4, -1), raw[1:2], last_hidden]),
                },
            )
        if graph_walk != "talker_decode":
            raise ValueError(f"Unknown Talker walk: {graph_walk}")
        thinker_states = inputs.get("thinker_states", [])
        rid = fwd_info.rid_handle
        if thinker_states:
            raw = thinker_states[0].to(device)
            kind = "text"
        elif rid not in self._eos_embed_sent:
            raw = self._tts_raw_embeddings[2:3]
            kind = "eos"
            self._eos_embed_sent.add(rid)
        else:
            raw = self._tts_raw_embeddings[0:1]
            kind = "pad"
        return ARNodeInputs(
            input_embeds=inputs["talker_input_embeds"][0],
            input_seq_len=1,
            tensor_inputs={"thinker_states": raw},
            kwargs={"conditioning_kind": kind},
        )


    def declare_step(
        self, graph_walk: str,
        request_ids: list[str],
        inputs: list[ARNodeInputs],
        slot_lease: SlotLease | None = None,
        piecewise_leases: Mapping[str, SlotLease] | None = None,
        **kwargs,
    ) ->SubmoduleStep:
        steps = {
            TALKER_KV: KVStep(),
            TALKER_ATTN: AttentionStep(causal=True),
            TALKER_POS: PositionStep()
        }
        if graph_walk in ("talker_last_prefill", "talker_decode"):
            steps.update(
                **{
                    TALKER_SAMPLER: SamplerStep(
                        apply_penalty=True,
                    ),
                    CODE_PRED_SAMPLER: SamplerStep(
                        apply_penalty=False
                    )
                }
            )
        return SubmoduleStep(
            segments=[
                Segment(
                    request_id=rid,
                    label="main",
                    span=inp.input_seq_len,
                ) for rid, inp in zip(request_ids, inputs, strict=True)
            ],
            steps=steps
        )


    def postprocess(
        self, request_id: str,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
        **kwargs
    ):
        if "new_token" not in outputs:
            return
        # Rename new_token → layer0_codes for downstream graph routing.
        # check_stop reads layer0_codes (same tensor) on the worker's slow path.
        codes = outputs.pop("new_token")[0]
        outputs["layer0_codes"] = [codes]


    def check_stop(
        self, request_id: str,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
    ) -> set[str]:
        if "layer0_codes" not in outputs:
            return set()
        token = outputs["layer0_codes"][0].item()
        eos_token_id = self.config.talker.codec_eos_token_id
        max_tokens = request_info.step_metadata.get(
            "talker_max_tokens", request_info.max_tokens
        )
        if (eos_token_id is not None and eos_token_id == token) or \
                (request_info.dynamic_loop_iter_counts.get("talker_decode_loop", 0) + 1 >= max_tokens):
            return {"talker_decode_loop"}
        return set()


    def cleanup_request(self, request_id: str) -> None:
        """Remove per-request state when a request completes."""
        super().cleanup_request(request_id)
        self._eos_embed_sent.discard(request_id)


    def execution_result(self, result, invocation):
        from mstar.engine.execution import ExecutionResult
        if isinstance(result, ExecutionResult):
            return result
        if invocation.graph_walk == "talker_prefill":
            return ExecutionResult(BatchedModelOutput(
                per_rid_outputs={rid: {} for rid in invocation.request_ids},
            ))
        if not isinstance(result, TalkerResult):
            return super().execution_result(result, invocation)
        return ExecutionResult(BatchedModelOutput(per_rid_outputs={
            rid: {"talker_input_embeds": [result.codec_embeds[i:i + 1]],
                  "codec_tokens": [result.codec_tokens[i]],
                  "new_token": [result.tokens[i]]}
            for i, rid in enumerate(invocation.request_ids)
        }, check_stop_buffers={"layer0_codes": result.tokens},
            row_request_ids=tuple(invocation.request_ids)))
