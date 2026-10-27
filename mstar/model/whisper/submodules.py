# ---------------------------------------------------------------------------
# NodeSubmodule wrappers for Whisper ASR
# ---------------------------------------------------------------------------
#
# Two submodules covering the encoder-decoder pipeline:
#   1. WhisperEncoderSubmodule  (stateless)  — HF WhisperEncoder, one-shot
#   2. WhisperDecoderSubmodule  (KV/attention/cross-attention/positions/sampler)
# ---------------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.resources import AttentionStep, KVStep, PositionStep, SamplerStep, Segment, SlotLease, SubmoduleStep
from mstar.model.contract import NodeContract, TensorInput, TensorOutput
from mstar.model.submodule_base import (
    ARNodeInputs,
    ARNodeSubmodule,
    ModelInputsFromEngine,
    NodeInputs,
    NodeSubmodule,
)
from mstar.model.whisper.config import (
    ATTN,
    CONTEXT_LABEL,
    CROSS_ATTN,
    CROSS_KV_CACHE,
    KV_CACHE,
    POS,
    SAMPLER,
    WhisperModelConfig,
)
from mstar.model.whisper.results import DecoderResult, EncoderResult

# ===================================================================
# 1. WhisperEncoderSubmodule (stateless)
# ===================================================================


class WhisperEncoderSubmodule(NodeSubmodule):
    """Thin wrapper around the HF Whisper audio encoder.

    Consumes the log-mel spectrogram produced by ``process_prompt``
    (a fixed 30 s window: ``(num_mel_bins, 3000)``) and emits
    ``encoder_states`` of shape ``(max_source_positions, d_model)``
    for the decoder's cross-attention. Runs once per request.
    """

    def __init__(self, config: WhisperModelConfig, reference_program: nn.Module | None = None):
        super().__init__()
        self.config = config
        self.reference_program = reference_program

    def make_reference_program(self, binding):
        from mstar.model.whisper.reference import build_encoder
        return build_encoder(self.config, binding.model.logical_weights().load(),
                             device=binding.device, dtype=binding.dtype)

    def logical_contract(self) -> NodeContract:
        return NodeContract(
            inputs=(TensorInput("audio_features", "tensor_inputs.audio_features",
                                (self.config.num_mel_bins, self.config.max_source_positions * 2)),),
            outputs=(TensorOutput("encoder_states",
                                  (self.config.max_source_positions, self.config.d_model),
                                  per_request=True),),
            result_type=EncoderResult,
            weight_prefix="encoder.",
        )

    def execution_result(self, result, invocation):
        from mstar.engine.execution import ExecutionResult
        from mstar.model.submodule_base import BatchedModelOutput
        return ExecutionResult(outputs=BatchedModelOutput(per_rid_outputs={
            rid: {"encoder_states": [states]}
            for rid, states in zip(invocation.request_ids, result.encoder_states, strict=True)
        }, check_stop_buffers={}))

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs,
    ) -> NodeInputs:
        return NodeInputs(
            tensor_inputs={"audio_features": inputs["audio_features"][0]}
        )

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        audio_features: torch.Tensor,
        **kwargs,
    ) -> NameToTensorList:
        if self.reference_program is None:
            raise RuntimeError("The fresh encoder must be bound to an execution program")
        return {"encoder_states": [self.reference_program.run(audio_features=audio_features)]}


# ===================================================================
# 2. WhisperDecoderSubmodule
# ===================================================================


class WhisperDecoderSubmodule(ARNodeSubmodule):
    """Autoregressive Whisper decoder.

    Dispatches on graph_walk:
      - prefill: embed the forced decoder prompt
        (``<|startoftranscript|><|lang|><|task|><|notimestamps|>``),
        project ``encoder_states`` to per-layer cross-attention K/V and
        write them into the context stream, fill the self-attention KV
        cache, and sample the first transcript token.
      - decode: embed the previous token, single-step decode.

    Both cache streams belong to the step: ``main`` grows by the token
    count, ``context`` grows by the encoder output at prefill and is
    declared zero-span (read-only) thereafter.
    """

    def __init__(self, config: WhisperModelConfig, reference_program: nn.Module | None = None):
        super().__init__()
        self.config = config
        self.reference_program = reference_program

    def make_reference_program(self, binding):
        from mstar.model.whisper.reference import build_decoder
        return build_decoder(self.config, binding.model.logical_weights().load(),
                             device=binding.device, dtype=binding.dtype,
                             tp_group=binding.tp_group)

    def logical_contract(self) -> NodeContract:
        return NodeContract(
            inputs=(
                TensorInput("input_ids", "input_ids", ("tokens",), dtype=torch.long),
                TensorInput("encoder_states", "tensor_inputs.encoder_states",
                            (self.config.max_source_positions, self.config.d_model),
                            walks=("prefill",)),
            ),
            outputs=(TensorOutput("tokens", ("batch",), dtype=torch.long),),
            result_type=DecoderResult,
            weight_prefix="decoder.",
            suppressed_tokens={
                SAMPLER: {
                    walk: tuple(self.config.suppressed_tokens(walk))
                    for walk in ("prefill", "decode")
                },
            },
        )

    def execution_result(self, result, invocation):
        from mstar.engine.execution import ExecutionResult
        from mstar.engine.resources.sampler.resource import SamplingResult
        from mstar.model.submodule_base import BatchedModelOutput
        return ExecutionResult(
            outputs=BatchedModelOutput(
                row_outputs={"new_token": result.tokens},
                check_stop_buffers={"new_token": result.tokens},
            ),
            resource_results={SAMPLER: SamplingResult(result.tokens)},
        )

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs,
    ) -> ARNodeInputs:
        token_ids = inputs["text_inputs"][0].reshape(-1)

        tensor_inputs = {}
        if graph_walk == "prefill":
            tensor_inputs["encoder_states"] = inputs["encoder_states"][0]

        return ARNodeInputs(
            input_seq_len=token_ids.shape[0],
            input_ids=token_ids,
            tensor_inputs=tensor_inputs,
        )

    @staticmethod
    def _context_span(inp: ARNodeInputs) -> int:
        """How far this step grows the request's encoder-context stream.

        Non-zero only on the prefill that writes it; every later step reads
        the same pages without extending them.
        """
        encoder_states = inp.tensor_inputs.get("encoder_states")
        return 0 if encoder_states is None else encoder_states.shape[0]

    def declare_step(
        self,
        graph_walk: str,
        request_ids: list[str],
        inputs: list[ARNodeInputs],
        slot_lease: SlotLease | None = None,
        piecewise_leases: Mapping[str, SlotLease] | None = None,
        **kwargs,
    ) -> SubmoduleStep:
        context_segments = tuple(
            Segment(
                request_id=rid,
                label=CONTEXT_LABEL,
                span=self._context_span(inp),
            ) for rid, inp in zip(request_ids, inputs, strict=True)
        )
        return SubmoduleStep(
            # the default, for the resources over the self-attention cache
            segments=[
                Segment(
                    request_id=rid,
                    label="main",
                    span=inp.input_seq_len,
                ) for rid, inp in zip(request_ids, inputs, strict=True)
            ],
            steps={
                KV_CACHE: KVStep(),
                ATTN: AttentionStep(causal=True),
                # The context stream: a real span on the prefill that writes
                # it, zero afterwards. `commit` is what turns the prefill's
                # reservation into resident pages the later steps read.
                CROSS_KV_CACHE: KVStep(segments=context_segments),
                CROSS_ATTN: AttentionStep(
                    segments=context_segments, causal=False,
                ),
                SAMPLER: SamplerStep(apply_penalty=False),
                POS: PositionStep(),
            },
        )

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[ARNodeInputs],
    ) -> dict[str, torch.Tensor | Any]:
        preprocessed: dict[str, torch.Tensor | Any] = {
            "input_ids": torch.cat([inp.input_ids for inp in inputs]),
        }
        if graph_walk == "prefill":
            # Concatenated in segment order, which is how the context plan
            # laid the requests' pages out.
            preprocessed["encoder_states"] = torch.cat(
                [inp.tensor_inputs["encoder_states"] for inp in inputs], dim=0,
            )
        return preprocessed

    def _forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_ids: torch.Tensor,
        encoder_states: torch.Tensor | None,
    ) -> torch.Tensor:
        """The explicit legacy reference path; bound programs bypass this body."""
        if self.reference_program is None:
            raise RuntimeError("The fresh decoder must be bound to an execution program")
        return self.reference_program.run(graph_walk, engine_inputs, input_ids, encoder_states)

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_ids: torch.Tensor,
        encoder_states: torch.Tensor | None = None,
        **kwargs,
    ) -> NameToTensorList:
        return {
            "new_token": [self._forward(
                graph_walk=graph_walk,
                engine_inputs=engine_inputs,
                input_ids=input_ids,
                encoder_states=encoder_states,
            )]
        }

    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_ids: torch.Tensor,
        encoder_states: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, NameToTensorList]:
        new_tokens = self._forward(
            graph_walk=graph_walk,
            engine_inputs=engine_inputs,
            input_ids=input_ids,
            encoder_states=encoder_states,
        )
        return {
            rid: {"new_token": [new_tokens[i:i + 1]]}
            for i, rid in enumerate(engine_inputs.request_ids)
        }

    def postprocess(
        self,
        request_id: str,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
        **kwargs,
    ):
        # Metadata-only: rebind output name so the decode loop feeds the
        # sampled token back in as the next step's text_inputs.
        if "new_token" not in outputs:
            return
        outputs["text_inputs"] = outputs["new_token"]

    def inline_client_signals(self, graph_walk: str) -> dict[str, str]:
        # Only decode's client edge is non-persisted; its local tensor loop
        # still follows the ordinary route/refcount path.
        return {"new_token": "new_token"} if graph_walk == "decode" else {}

    def check_stop(
        self,
        request_id: str,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
    ) -> set[str]:
        if "new_token" not in outputs:
            return set()
        token = outputs["new_token"][0].item()
        ignore_eos = request_info.resource_configs[SAMPLER].ignore_eos
        decoded_tokens = request_info.dynamic_loop_iter_counts.get("decode_loop", 0) + 1
        if (not ignore_eos and token == self.config.eos_token_id) or \
                decoded_tokens >= request_info.max_tokens:
            return {"decode_loop"}
        return set()
