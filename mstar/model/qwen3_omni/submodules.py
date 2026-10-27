# ---------------------------------------------------------------------------
# NodeSubmodule wrappers for Qwen3-Omni
# ---------------------------------------------------------------------------
#
# Five submodules covering the full Thinker-Talker dual-AR pipeline:
#   1. AudioEncoderSubmodule
#   2. VisionEncoderSubmodule
#   3. ThinkerSubmodule        (3D MRoPE, MoE, layer captures)
#   4. TalkerSubmodule         (streaming decode, Code Predictor)
#   5. Code2WavSubmodule
# ---------------------------------------------------------------------------

from __future__ import annotations

import functools
import inspect
import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Optional

import torch
from torch import nn

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.accelerator_graph_config import (
    AcceleratorGraphConfig,
    BatchedAcceleratorGraphConfig,
    PackedAcceleratorGraphConfig,
)
from mstar.engine.engine import ExecutingBatch
from mstar.engine.resources import AttentionStep, KVStep, PositionStep, SamplerStep, Segment, SlotLease, SubmoduleStep
from mstar.engine.resources.attn.flashinfer import FlashInferManager
from mstar.engine.resources.sampler.resource import SamplerResource
from mstar.model.qwen3_omni.components.rope import (
    compute_3d_cos_sin,
    compute_rope_freqs,
)
from mstar.model.qwen3_omni.config import (
    CODE_PRED_SAMPLER,
    TALKER_ATTN,
    TALKER_KV,
    TALKER_POS,
    TALKER_SAMPLER,
    THINKER_ATTN,
    THINKER_MIXED,
    THINKER_SAMPLER,
    Qwen3OmniModelConfig,
)
from mstar.model.qwen3_omni.logical import TalkerSubmodule as TalkerSubmodule  # noqa: PLC0414 — public logical identity
from mstar.model.qwen3_omni.thinker_logical import ThinkerLogical
from mstar.model.qwen3_omni.thinker_logical import (
    talker_text_mask as talker_text_mask,  # noqa: PLC0414 — upstream public helper
)
from mstar.model.submodule_base import (
    ARNodeInputs,
    ARNodeSubmodule,
    InputMetadata,
    ModelInputsFromEngine,
    NodeInputs,
    NodeSubmodule,
)

if TYPE_CHECKING:
    from mstar.model.qwen3_omni.components.code2wav import Qwen3OmniMoeCode2Wav
    from mstar.model.qwen3_omni.components.talker import Qwen3OmniCodePredictor, Qwen3OmniTalkerModel

logger = logging.getLogger(__name__)


def _accepts_kwarg(fn: Any, name: str) -> bool:
    """True if fn takes `name` as a keyword (explicitly or via **kwargs)."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return True
    param = params.get(name)
    return param is not None and param.kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    )




# ===================================================================
# 1. AudioEncoderSubmodule (enc_dec engine)
# ===================================================================


class AudioEncoderSubmodule(NodeSubmodule):
    """Thin wrapper around the HF Whisper-style audio encoder.

    Extracts mel spectrograms from raw audio inputs, pads for batching,
    and runs the encoder to produce audio embeddings that the Thinker
    will splice into its input sequence.

    Runs once per request (not batched across requests).
    """

    def __init__(self, audio_encoder: nn.Module, config: Qwen3OmniModelConfig):
        super().__init__()
        self.audio_encoder = audio_encoder
        self.config = config
        # Older transformers take return_dict; newer ones dropped it.
        self._encoder_takes_return_dict = _accepts_kwarg(audio_encoder.forward, "return_dict")

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs
    ) -> NodeInputs:
        # Edge name from graph walk is "audio_features"
        audio_features = inputs["audio_features"][0]
        audio_seqlens = inputs.get("audio_seqlens", [None])[0]

        return NodeInputs(
            tensor_inputs={
                "audio_features": audio_features,
                "audio_seqlens": audio_seqlens,
            }
        )

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        audio_features: torch.Tensor,
        audio_seqlens: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> NameToTensorList:
        """Run the audio encoder and return embeddings.

        Returns:
            {"audio_embeds": [tensor of shape (audio_tokens, hidden_size)]}
        """
        logger.debug(
            "Running AudioEncoder with audio_features shape=%s",
            audio_features.shape,
        )
        extra_kwargs = {"return_dict": True} if self._encoder_takes_return_dict else {}
        audio_out = self.audio_encoder(
            audio_features,
            feature_lens=audio_seqlens,
            **extra_kwargs,
        )
        audio_embeds = getattr(audio_out, "last_hidden_state", audio_out)

        # Flatten to (num_audio_tokens, hidden_size) if needed
        if audio_embeds.dim() == 3:
            audio_embeds = audio_embeds.squeeze(0)

        return {"audio_embeds": [audio_embeds]}


# ===================================================================
# 2. VisionEncoderSubmodule (enc_dec engine)
# ===================================================================


class VisionEncoderSubmodule(NodeSubmodule):
    """Thin wrapper around the HF vision encoder (ViT + spatial merge).

    Extracts pixel_values and grid_thw from inputs, computes cu_seqlens
    for FlashAttention, runs the encoder, and returns vision embeddings
    plus DeepStack intermediate features for the Thinker.

    Runs once per request (not batched across requests).
    """

    def __init__(self, vision_encoder: nn.Module, config: Qwen3OmniModelConfig):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.config = config

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs
    ) -> NodeInputs:
        """Extract pixel_values, grid_thw, and compute cu_seqlens.

        ``pixel_values`` and ``image_grid_thw`` are produced by
        ``Qwen3OmniModel.process_prompt`` from the raw ``image_inputs``
        loaded by the data worker.
        """
        # Edge name from graph walk is "pixel_values"
        pixel_values = inputs["pixel_values"][0]       # (N_patches, C, patch_H, patch_W)
        grid_thw = inputs.get("image_grid_thw", inputs.get("grid_thw", [None]))[0]

        # Normalize grid_thw to shape (num_images, 3).  Single-image requests
        # store grid_thw as a 1-D tensor [T, H, W] (after process_prompt
        # indexes proc_out["image_grid_thw"][0] to strip the batch dim);
        # the per-image iteration logic below requires 2-D.
        if grid_thw is None:
            raise ValueError(
                "VisionEncoder: 'image_grid_thw' input is None. "
                "Make sure process_prompt is producing image_grid_thw via the "
                "HF AutoImageProcessor."
            )
        if grid_thw.dim() == 1:
            grid_thw = grid_thw.unsqueeze(0)  # (1, 3)

        return NodeInputs(
            tensor_inputs={
                "pixel_values": pixel_values,
                "grid_thw": grid_thw,
            }
        )

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
        **kwargs,
    ) -> NameToTensorList:
        """Run vision encoder, return embeddings and DeepStack features.

        Returns:
            {
                "vision_embeds": [tensor of shape (vision_tokens, hidden_size)],
                "deepstack": [list of intermediate layer features],
            }
        """
        logger.debug(
            "Running VisionEncoder with pixel_values shape=%s, grid_thw shape=%s",
            pixel_values.shape, grid_thw.shape,
        )
        # HF vision encoder returns (hidden_states, deepstack_features)
        # depending on the model variant; handle both cases
        encoder_output = self.vision_encoder(
            pixel_values,
            grid_thw=grid_thw,
        )

        if isinstance(encoder_output, tuple):
            vision_embeds, deepstack = encoder_output
        else:
            vision_embeds = encoder_output.pooler_output
            deepstack = encoder_output.deepstack_features

        if isinstance(deepstack, torch.Tensor):
            deepstack = [deepstack]

        return {
            "vision_embeds": [vision_embeds],
            "deepstack": deepstack if deepstack is not None else [torch.tensor([])],
        }


# ===================================================================
# 3. ThinkerSubmodule (ar engine) -- MOST COMPLEX
# ===================================================================




class ThinkerSubmodule(ThinkerLogical):
    """Wraps the FlashInfer-based Thinker MoE transformer.

    Dispatches on graph_walk:
      - prefill_text:   embed text tokens, compute 3D MRoPE, fill KV cache
      - prefill_audio:  splice audio embeddings, extend KV cache
      - prefill_vision: splice vision embeddings, extend KV cache
      - thinker_decode: embed previous token, single-step decode

    All walks produce ``thinker_states`` (layer-0 + layer-N concat) that
    stream to the Talker partition.  ``thinker_decode`` additionally
    produces ``logits`` for text token sampling.
    """

    # Default MRoPE section for head_dim=128: [24, 20, 20]
    MROPE_SECTION = [24, 20, 20]

    def __init__(
        self,
        thinker_model: nn.Module,
        config: Qwen3OmniModelConfig,
    ):
        super().__init__(config)
        self.model = thinker_model  # Qwen3OmniThinkerModel
        self.config = config

        # Pre-compute inverse frequencies for 3D MRoPE
        self._inv_freq: torch.Tensor | None = None

        # Lazily-cached constant mask used by ``_preprocess_decode`` for the
        # Talker partition.  Every decode-step mask is the same constant
        # ``[[0], [1]]`` so we allocate it once per device instead of per
        # request per step.  Helps keep the captured graph's output-dict
        # contents self-evidently constant, too.
        self._decode_thinker_mask: torch.Tensor | None = None

        self._audio_bos_embed: torch.Tensor | None = None
        self._audio_eos_embed: torch.Tensor | None = None

        self._vision_bos_embed: torch.Tensor | None = None
        self._vision_eos_embed: torch.Tensor | None = None

    def _get_inv_freq(self, device: torch.device) -> torch.Tensor:
        """Lazy-initialize and cache inverse frequencies."""
        if self._inv_freq is None or self._inv_freq.device != device:
            self._inv_freq = compute_rope_freqs(
                self.config.thinker_head_dim,
                rope_theta=self.config.thinker_text.rope_theta,
                device=device,
            )
        return self._inv_freq


    def _wrap_audio_input(self, audio_embeds: torch.Tensor):
        # Wrap the audio span in ``<|audio_bos|>`` / ``<|audio_eos|>``
        # sentinel token embeddings so the Thinker sees the same
        # prompt layout the HF processor produces.
        device = self.get_device()
        if self._audio_bos_embed is None or self._audio_eos_embed is None:
            audio_start_id = self.config.thinker.audio_start_token_id
            audio_end_id = self.config.thinker.audio_end_token_id
            start_tok = torch.tensor(
                [audio_start_id], dtype=torch.long, device=device
            )
            end_tok = torch.tensor(
                [audio_end_id], dtype=torch.long, device=device
            )
            self._audio_bos_embed = self.model.model.embed_tokens(start_tok)
            self._audio_eos_embed = self.model.model.embed_tokens(end_tok)

        return torch.cat([
            self._audio_bos_embed,
            audio_embeds,
            self._audio_eos_embed
        ], dim=0)

    def _wrap_vision_input(self, vision_embeds: torch.Tensor):
        # Wrap the vision span in ``<|vision_bos|>`` / ``<|vision_eos|>``
        # sentinel token embeddings.
        if self._vision_bos_embed is None or self._vision_eos_embed is None:
            device = vision_embeds.device
            vision_start_id = self.config.thinker.vision_start_token_id
            vision_end_id = self.config.thinker.vision_end_token_id
            start_tok = torch.tensor(
                [vision_start_id], dtype=torch.long, device=device
            )
            end_tok = torch.tensor(
                [vision_end_id], dtype=torch.long, device=device
            )
            self._vision_bos_embed = self.model.model.embed_tokens(start_tok)
            self._vision_eos_embed = self.model.model.embed_tokens(end_tok)

        return torch.cat([
            self._vision_bos_embed,
            vision_embeds,
            self._vision_eos_embed
        ], dim=0)

    def prepare_inputs(self, graph_walk, fwd_info, inputs, **kwargs):
        from dataclasses import replace
        logical = super().prepare_inputs(graph_walk, fwd_info, inputs, **kwargs)
        if logical is None:
            return None
        kind = logical.tensor_inputs["embedding_kind"]
        if kind == "text":
            embeddings = self.model.model.embed_tokens(logical.input_ids)
        elif kind == "audio":
            embeddings = self._wrap_audio_input(logical.input_embeds)
        else:
            embeddings = self._wrap_vision_input(logical.input_embeds)
        return replace(logical, input_embeds=embeddings,
            tensor_inputs={key: value for key, value in logical.tensor_inputs.items()
                           if key != "embedding_kind" and not key.startswith("deepstack_")})


    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[ARNodeInputs],
    ) -> dict[str, torch.Tensor | Any]: # input name to tensor
        self.model.maybe_log_expert_load(synthetic=engine_inputs.synthetic)
        device = self.get_device()
        # Concatenate across requests
        input_embeds = torch.cat([
            inp.input_embeds for inp in inputs
        ], dim=0)
        position_ids_3d = torch.cat([
            inp.custom_pos_ids for inp in inputs
        ], dim=1)  # (3, total_tokens)
        seq_lens = [
            inp.input_seq_len for inp in inputs
        ]

        # Compute cos/sin for 3D MRoPE.  Returned as separate tensor keys
        # (not a tuple) so the CUDA graph runner can detect them as static
        # inputs and copy them into the captured buffers at replay.
        inv_freq = self._get_inv_freq(device)
        cos_3d, sin_3d = compute_3d_cos_sin(
            position_ids_3d, inv_freq,
            mrope_section=self.MROPE_SECTION,
            target_dtype=input_embeds.dtype,
        )

        extra_inputs = {}
        if graph_walk == "prefill_vision" or any("deepstack" in inp.tensor_inputs for inp in inputs):
            # Q3(b): emit deepstack as separate keys ``deepstack_<i>`` so each
            # tensor gets its own static buffer in the captured config (the
            # runner's static-buffer interning is per-key and tensor-typed;
            # passing a list-of-tensors under one key would not get interned
            # and addresses captured into the graph would be stale at replay).
            # ``forward_batched``/``forward`` reassemble the list from kwargs.
            extra_inputs = self._deepstack_inputs(inputs, input_embeds.dtype, device)

        if self.model.mask_padding:
            # Real rows lead the bucket; the captured MoE skips the rest.
            # A fill kernel, so no H2D copy.
            num_valid = engine_inputs.real_seq_len
            if num_valid is None:
                num_valid = input_embeds.shape[0]
            extra_inputs["num_valid_tokens"] = torch.full(
                (1,), num_valid, dtype=torch.int64, device=device,
            )

        return {
            "input_embeds": input_embeds,
            "cos_3d": cos_3d,
            "sin_3d": sin_3d,
            "mrope_section": self.MROPE_SECTION,
            "seq_lens": seq_lens,
            "masks_for_talker": {
                rid: inp.tensor_inputs.get("masks_for_talker") \
                    for (rid, inp) in zip(engine_inputs.request_ids, inputs, strict=True)
            },
            **extra_inputs
        }

    def _deepstack_inputs(
        self, inputs: list[ARNodeInputs], dtype: torch.dtype, device: torch.device,
    ) -> dict[str, torch.Tensor]:
        """Each deepstack layer packed over the step's rows; a row with no
        vision features (a text, audio or decode row beside a vision one)
        contributes zeros, which the splice adds as nothing."""
        num_deepstack = len(self.config.vision.deepstack_visual_indexes)
        layers: list[list[torch.Tensor]] = [[] for _ in range(num_deepstack)]
        for inp in inputs:
            rows = inp.tensor_inputs.get("deepstack")
            if rows is None or (isinstance(rows, torch.Tensor) and rows.numel() == 0):
                zeros = torch.zeros(
                    (inp.input_seq_len, self.config.thinker_hidden_size), dtype=dtype, device=device,
                )
                rows = [zeros] * num_deepstack
            assert len(rows) == num_deepstack, (
                f"deepstack list length ({len(rows)}) does not match "
                f"vision.deepstack_visual_indexes ({num_deepstack})."
            )
            for layer, t in zip(layers, rows, strict=True):
                layer.append(t)
        return {
            f"deepstack_{i}": layer[0] if len(layer) == 1 else torch.cat(layer)
            for i, layer in enumerate(layers)
        }

    # ---- forward ----

    def _collect_deepstack_kwargs(
        self, kwargs: dict[str, Any]
    ) -> list[torch.Tensor] | None:
        """Reassemble the deepstack list from per-layer ``deepstack_<i>`` keys.

        ``preprocess`` emits one key per ``vision.deepstack_visual_indexes``
        entry so each layer's visual feature gets its own static buffer in
        the captured prefill_vision config. Both the eager ``forward`` and
        the captured ``forward_batched`` call this helper to put the list
        back together before invoking ``Qwen3OmniThinkerModel.forward``.

        Returns None when no ``deepstack_*`` keys are present (e.g. for
        prefill_text / prefill_audio / thinker_decode), so the inner model
        receives ``deepstack_visual_embeds=None`` and skips the deepstack
        splice entirely.
        """
        num_deepstack = len(self.config.vision.deepstack_visual_indexes)
        if not any(f"deepstack_{i}" in kwargs for i in range(num_deepstack)):
            return None
        out: list[torch.Tensor] = []
        for i in range(num_deepstack):
            t = kwargs.get(f"deepstack_{i}")
            if t is None:
                return None
            out.append(t)
        return out

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_embeds: torch.Tensor | None = None,
        cos_3d: torch.Tensor | None = None,
        sin_3d: torch.Tensor | None = None,
        mrope_section: list[int] | None = None,
        masks_for_talker: dict[str, torch.Tensor] | None = None,
        **kwargs,
    ) -> NameToTensorList:
        """Run Thinker transformer, produce logits (decode) and thinker_states.

        ``thinker_states`` is only emitted when audio output is requested
        (checked via ``request_info.step_metadata["audio_output"]``). This
        saves cross-partition bandwidth for text-only requests. Defaults to
        ``True`` for backwards compatibility with callers that do not set
        the flag (e.g. unit tests).

        ``deepstack`` (used by prefill_vision) is reassembled from per-layer
        ``deepstack_<i>`` kwargs — see ``_collect_deepstack_kwargs``.
        """
        deepstack = self._collect_deepstack_kwargs(kwargs)
        request_info: CurrentForwardPassInfo = engine_inputs.single_request_info
        audio_output = request_info.step_metadata.get(
            "audio_output", True,
        )

        cos_sin_3d = (cos_3d, sin_3d) if cos_3d is not None else None

        hidden, layer_0_embed, layer_n_hidden = self.model(
            input_embeds=input_embeds,
            cos_sin_3d=cos_sin_3d,
            mrope_section=mrope_section,
            deepstack_visual_embeds=deepstack,
            label="main",
            num_valid_tokens=kwargs.get("num_valid_tokens"),
        )

        result: NameToTensorList = {}

        # Decode: produce logits for text token sampling
        if graph_walk == "thinker_decode" or request_info.step_metadata.get("is_last_prefill", False):
            logits = self.model.lm_head(hidden[-1:, :])
            result["new_token"] = [engine_inputs.resources[THINKER_SAMPLER].sample(
                engine_inputs.request_ids, logits=logits
            )]

        # Pack thinker_states for Talker conditioning ONLY when audio output
        # is requested.  For text-only requests we skip this entirely to
        # avoid sending hidden states the Talker will never consume.
        if audio_output:
            # Concatenate layer-0 embeddings and layer-N hidden states along
            # last dim -> (tokens, 2 * hidden_size)
            if layer_n_hidden is not None:
                thinker_states = torch.cat(
                    [layer_0_embed, layer_n_hidden], dim=-1,
                )
            else:
                # Fallback: use layer_0_embed doubled (shouldn't happen in
                # practice)
                thinker_states = torch.cat(
                    [layer_0_embed, layer_0_embed], dim=-1,
                )
            result["thinker_states"] = [thinker_states]
            result["thinker_mask"] = [next(iter(masks_for_talker.values()))] \
                if masks_for_talker else []
        return result

    # ---- batching ----
    def can_batch(self, batch: ExecutingBatch, model_inputs: list[NodeInputs]) -> bool:
        return len(model_inputs) > 1

    PREFILL_TOKEN_BUCKETS = [128, 256, 512, 1024, 2048]
    # past 4 for thinker_mixed, whose decode rows ride in these captures
    PREFILL_CAPTURE_BATCH_SIZES = [1, 2, 4, 8, 16, 32]

    # prefill_vision buckets are larger than text/audio because video
    # produces many vision tokens per request
    PREFILL_VISION_TOKEN_BUCKETS = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    # coarse batch buckets, padded host-side; past 1 for mixed steps
    PREFILL_VISION_CAPTURE_BATCH_SIZES = [1, 4, 16, 32]
    DEEPSTACK_KEY = "deepstack"

    def get_accelerator_graph_configs(
        self, device: torch.device, tp_world_size: int = 1,
    ) -> list[AcceleratorGraphConfig]:
        vision_input = functools.partial(self._vision_capture_input, device)
        return [
            BatchedAcceleratorGraphConfig(
                capture_graph_walk="thinker_decode",
                single_request_inputs=ARNodeInputs(
                    input_seq_len=1,
                    input_embeds=torch.zeros(
                        (1, self.config.thinker_hidden_size),
                        device=device, dtype=torch.bfloat16
                    ),
                    custom_pos_ids=torch.tensor(
                        [[0], [0], [0]],
                        dtype=torch.float,
                        device=device,
                    ),
                    tensor_inputs={
                        "masks_for_talker": self._get_decode_thinker_mask(device)
                    }
                ),
                capture_batch_sizes=[1, 2, 4, 8, 16, 32],
            ),
            PackedAcceleratorGraphConfig(
                capture_graph_walk="prefill_text",
                replay_graph_walks=["prefill_text", "prefill_audio", THINKER_MIXED],
                make_node_input=lambda n: ARNodeInputs(
                    input_seq_len=n,
                    input_ids=torch.zeros((n,), dtype=torch.long, device=device),
                    input_embeds=torch.zeros(
                        (n, self.config.thinker_hidden_size),
                        device=device, dtype=torch.bfloat16
                    ),
                    custom_pos_ids=torch.zeros(
                        (3, n),
                        dtype=torch.float,
                        device=device,
                    ),
                    tensor_inputs={
                        "masks_for_talker": torch.zeros(
                            (2, n),
                            dtype=torch.float,
                            device=device,
                        ),
                    }
                ),
                capture_token_lengths=self.PREFILL_TOKEN_BUCKETS,
                capture_batch_sizes=self.PREFILL_CAPTURE_BATCH_SIZES,
            ),
            # any step holding a vision row: the packed forward plus deepstack
            PackedAcceleratorGraphConfig(
                capture_graph_walk="prefill_vision",
                replay_graph_walks=["prefill_vision", THINKER_MIXED],
                additional_key_info=self.DEEPSTACK_KEY,
                capture_token_lengths=self.PREFILL_VISION_TOKEN_BUCKETS,
                capture_batch_sizes=self.PREFILL_VISION_CAPTURE_BATCH_SIZES,
                make_node_input=vision_input,
            ),
        ]

    def cg_key_info(
        self, graph_walk: str,
        per_request_info: dict[str, CurrentForwardPassInfo],
        per_request_input_metadata=None,
        **kwargs,
    ):
        """A step holding a vision row replays the deepstack captures."""
        del per_request_input_metadata, kwargs
        if graph_walk == "prefill_vision" or graph_walk == THINKER_MIXED and any(
            info.graph_walk == "prefill_vision" for info in per_request_info.values()
        ):
            return self.DEEPSTACK_KEY
        return None

    def _vision_capture_input(self, device: torch.device, n: int) -> ARNodeInputs:
        num_deepstack = len(self.config.vision.deepstack_visual_indexes)
        return ARNodeInputs(
            input_seq_len=n,
            input_embeds=torch.zeros(
                (n, self.config.thinker_hidden_size),
                device=device, dtype=torch.bfloat16,
            ),
            custom_pos_ids=torch.zeros(
                (3, n),
                dtype=torch.float,
                device=device,
            ),
            tensor_inputs={
                "masks_for_talker": torch.zeros(
                    (2, n), dtype=torch.float, device=device,
                ),
                # Use a list (matches eager prepare_inputs) — preprocess turns it
                # into per-layer ``deepstack_<i>`` keys.
                "deepstack": [
                    torch.zeros(
                        (n, self.config.thinker_hidden_size),
                        dtype=torch.bfloat16, device=device,
                    )
                    for _ in range(num_deepstack)
                ],
                "mrope_pos_advance": n,
            },
        )

    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_embeds: torch.Tensor | None = None,
        cos_3d: torch.Tensor | None = None,
        sin_3d: torch.Tensor | None = None,
        mrope_section: list[int] | None = None,
        masks_for_talker: dict[str, torch.Tensor] | None = None,
        **kwargs,
    ) -> dict[str, NameToTensorList]:
        """Batched Thinker forward shared between ``thinker_decode`` and the prefill walks.

        Decode path (1 token per request, ``hidden`` shape ``(bs, hidden)``):
        Always packs ``thinker_states`` + ``thinker_mask`` in every per-rid
        output dict so the captured CUDA graph has a static output shape
        regardless of request metadata. Per-rid filtering happens in
        ``unpack_packed_output``.
        """

        is_prefill = graph_walk in (
            "prefill_text", "prefill_audio", "prefill_vision", THINKER_MIXED,
        )

        # prefill_vision: reassemble per-layer deepstack tensors from
        # ``deepstack_<i>`` kwargs into the list shape ``model.forward``
        # expects. None for non-vision walks → model skips the splice.
        deepstack = self._collect_deepstack_kwargs(kwargs)

        cos_sin_3d = (cos_3d, sin_3d) if cos_3d is not None else None
        attn_manager: FlashInferManager = engine_inputs.resources[THINKER_ATTN]
        hidden, layer_0_embed, layer_n_hidden = self.model(
            input_embeds=input_embeds,
            cos_sin_3d=cos_sin_3d,
            mrope_section=mrope_section,
            deepstack_visual_embeds=deepstack,
            label="main",
            num_valid_tokens=kwargs.get("num_valid_tokens"),
        )

        request_ids = engine_inputs.request_ids
        sampler: SamplerResource = engine_inputs.resources[THINKER_SAMPLER]

        if is_prefill:
            last_hidden = attn_manager.select_last_hidden(hidden)
            logits = self.model.lm_head(last_hidden)  # (padded_bs, vocab)
            new_tokens = sampler.sample(request_ids, logits)
            if layer_n_hidden is not None:
                thinker_states = torch.cat(
                    [layer_0_embed, layer_n_hidden], dim=-1,
                )  # (total_tokens, 2*hidden)
            else:
                thinker_states = torch.cat(
                    [layer_0_embed, layer_0_embed], dim=-1,
                )

            return {
                "__batched_thinker_states__": thinker_states,
                **{
                    rid: {
                        "new_token": [new_tokens[i:i + 1]]
                    } for i, rid in enumerate(request_ids)
                }
            }

        # thinker_decode (existing behavior)
        logits = self.model.lm_head(hidden)  # (batch, vocab)
        new_tokens = sampler.sample(request_ids, logits)

        if layer_n_hidden is not None:
            thinker_states = torch.cat(
                [layer_0_embed, layer_n_hidden], dim=-1,
            )
        else:
            thinker_states = torch.cat(
                [layer_0_embed, layer_0_embed], dim=-1,
            )

        outputs: dict[str, NameToTensorList] = {}
        for i, rid in enumerate(request_ids):
            req_out: NameToTensorList = {
                "thinker_states": [thinker_states[i : i + 1]],
                "new_token": [new_tokens[i:i+1]]
            }
            if masks_for_talker is not None and rid in masks_for_talker:
                req_out["thinker_mask"] = [masks_for_talker[rid]]
            outputs[rid] = req_out

        return outputs

    def unpack_packed_outputs(
        self,
        static_output: dict,
        request_ids: list[str],
        real_seq_lens: list[int],
        inputs: list[ARNodeInputs],
        per_request_info: dict[str, "CurrentForwardPassInfo"],
    ) -> dict[str, dict[str, list[torch.Tensor]]]:
        """Slice the packed ``__batched_thinker_states__`` per real seq_len.

        Captured forward emits the full ``(total_tokens, 2*hidden)`` packed
        tensor; here we cut it at the real per-request token boundaries and
        reattach the per-request talker masks, which live on the original
        ARNodeInputs (the captured graph never saw them — masks vary in
        shape with text content). Drops per-rid emission for requests with
        ``audio_output=False``, mirroring ``filter_batched_output``'s gating
        for the decode path.
        """
        packed_states = static_output.get("__batched_thinker_states__")
        if packed_states is None:
            return {}

        out: dict[str, dict[str, list[torch.Tensor]]] = {}
        cum = 0
        for i, rid in enumerate(request_ids):
            sl = real_seq_lens[i]
            slice_start, slice_end = cum, cum + sl
            cum = slice_end

            info = per_request_info.get(rid) if per_request_info else None
            if info is not None and not info.step_metadata.get("audio_output", True):
                continue

            ts_slice = packed_states[slice_start:slice_end].clone()
            rid_out: dict[str, list[torch.Tensor]] = {
                "thinker_states": [ts_slice],
            }
            mask = inputs[i].tensor_inputs.get("masks_for_talker")
            if mask is not None:
                rid_out["thinker_mask"] = [mask]
            out[rid] = rid_out
        return out





# ===================================================================
# 4. TalkerSubmodule (ar engine) -- SECOND MOST COMPLEX
# ===================================================================

class LegacyTalkerSubmodule(ARNodeSubmodule):
    """Retained default numerical implementation; not an acceleration shell."""
    MAX_BATCH_SIZE = 32

    def __init__(
        self,
        talker_model: Qwen3OmniTalkerModel,
        code_predictor: Qwen3OmniCodePredictor,
        config: Qwen3OmniModelConfig,
    ):
        super().__init__()
        self.model = talker_model
        self.code_predictor = code_predictor
        self.talker_code_emb = talker_model.model.codec_embedding
        self.config = config
        self.cp_cfg = config.code_predictor
        self.num_codes = self.cp_cfg.num_code_groups

        # Pre-computed TTS special inputs.
        self._tts_pad_embed_cached: torch.Tensor | None = None
        self._tts_bos_embed_cached: torch.Tensor | None = None
        self._tts_eos_embed_cached: torch.Tensor | None = None
        self.register_buffer("_tts_raw_embeddings", None, persistent=False)

        # Lazy-built suppress mask for layer-0 logits.  Shape (vocab_size,)
        # with True at positions to suppress.  Cached on first forward.
        self._suppress_mask: torch.Tensor | None = None

        # Per-request flag: whether we've already sent tts_eos_embed as
        # the text conditioning for this request. We use this flag to
        # inject tts_eos_embed for ONE step before falling back to pad.
        self._eos_embed_sent: set[int] = set()

        # Lazy-built CodePredictor scratch KV cache, overwritten every step.
        self._cp_kv_cache: torch.Tensor | None = None

    @torch.no_grad()
    def _init_tts_raw_embeddings(self, raw: torch.Tensor) -> None:
        if tuple(raw.shape) != (3, self.config.thinker_hidden_size):
            raise ValueError("TTS embeddings must contain pad, bos and eos rows")
        dtype = self.model.text_projection.linear_fc1.weight.dtype
        self._tts_raw_embeddings = raw.to(device=next(self.model.parameters()).device, dtype=dtype)
        # Preserve the reference's single-row projection arithmetic.
        for i, name in enumerate(("pad", "bos", "eos")):
            projected = self.model.text_projection(self._tts_raw_embeddings[i:i + 1]).squeeze(0)
            setattr(self, f"_tts_{name}_embed_cached", projected)

    def _get_cp_kv_cache(self):
        # TODO: make this a resource
        if self._cp_kv_cache is None:
            self._cp_kv_cache = torch.zeros((
                    self.cp_cfg.num_hidden_layers,
                    self.MAX_BATCH_SIZE, 2, self.num_codes,
                    self.cp_cfg.num_key_value_heads,
                    self.cp_cfg.head_dim
                ), dtype=self.talker_code_emb.weight.dtype,
                device=self.get_device(),
            )
        return self._cp_kv_cache

    def init_tts_embeds(self, thinker_embed_tokens: nn.Embedding) -> None:
        """Pre-compute TTS pad/bos/eos hidden states using the Thinker's
        embedding table + Talker text_projection

        Must be called after both the Thinker and Talker weights are loaded
        (only applicable when both reside on the same worker).  When they
        are on different workers, these embeddings should be transferred as
        constant tensors during model init.
        """
        device = thinker_embed_tokens.weight.device
        with torch.no_grad():
            self._init_tts_raw_embeddings(thinker_embed_tokens(torch.tensor([
                self.config.tts_pad_token_id, self.config.tts_bos_token_id, self.config.tts_eos_token_id,
            ], device=device)))

        logger.info(
            "TalkerSubmodule: pre-computed TTS special embeddings via "
            "Thinker embed_tokens + Talker text_projection"
        )

    def _get_suppress_mask(self) -> torch.Tensor:
        """Return the bool mask of layer-0 logits to set to -inf.

        Matches HF's ``talker_supppressed_tokens`` list: suppress the top
        1024 IDs of the Talker vocab (the "special token" region) EXCEPT
        ``codec_eos_token_id``, which is the valid stop signal.
        """
        if  self._suppress_mask is None:
            device = next(self.model.parameters()).device
            vocab_size = self.config.talker_text.vocab_size
            mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
            start = vocab_size - 1024
            start = max(start, 0)
            mask[start:vocab_size] = True
            # Do not suppress codec_eos (the valid stop signal).
            eos = self.config.talker.codec_eos_token_id
            if 0 <= eos < vocab_size:
                mask[eos] = False
            self._suppress_mask = mask
        return self._suppress_mask

    def _get_talker_embeds(
        self, selected_hidden: torch.Tensor, multimodal_mask: torch.Tensor,
    ):
        # ``selected_hidden`` is layer-0 embed for text positions and
        # layer-N hidden for multimodal positions (the Thinker's
        # postprocess already did that selection and dropped positions
        # the Talker won't consume). ``multimodal_mask`` is 1D, aligned
        # to the kept tokens, and tells us which projection to apply.
        text_mask = ~multimodal_mask
        projected = torch.empty(
            selected_hidden.shape[0], self.config.talker_hidden_size,
            device=selected_hidden.device, dtype=selected_hidden.dtype,
        )
        if text_mask.any():
            projected[text_mask] = self.model.text_projection(
                selected_hidden[text_mask]
            )
        if multimodal_mask.any():
            projected[multimodal_mask] = self.model.hidden_projection(
                selected_hidden[multimodal_mask]
            )
        return projected

    def _get_last_prefill_talker_hidden(
        self, thinker_hidden: torch.Tensor
    ):
        # Six-token assistant prefix: pad*4, bos, first Thinker token.
        # Special values preserve the startup single-row projections.
        pad = self._tts_pad_embed_cached.expand(4, -1)  # 4 pad tokens
        bos_text = self._tts_bos_embed_cached.unsqueeze(0) # 1 bos token

        return torch.cat([
            pad,          # pad * 4   (4 tokens)
            bos_text,     # bos       (1 token)
            self.model.text_projection(thinker_hidden), #  (1 token)
        ], dim=0)  # (6, talker_hidden)

    def _get_last_prefill_codec_ids(self, speaker: str) -> torch.Tensor:
        tc = self.config.talker
        speaker_id = tc.speaker_id.get(speaker.lower())
        if speaker_id is None:
            logger.warning("Speaker %s not implemented", speaker)
            speaker_id = tc.codec_pad_id
        return torch.tensor([
            tc.codec_nothink_id, tc.codec_think_bos_id, tc.codec_think_eos_id,
            speaker_id, tc.codec_pad_id, tc.codec_bos_id,
        ], device=self.get_device(), dtype=torch.long)

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

    def _materialize_input_embeds(self, graph_walk: str, inp: ARNodeInputs) -> torch.Tensor:
        raw = inp.tensor_inputs["thinker_states"]
        if graph_walk == "talker_prefill":
            return self._get_talker_embeds(raw, inp.tensor_inputs["multimodal_mask"])
        if graph_walk == "talker_last_prefill":
            return self.talker_code_emb(inp.tensor_inputs["codec_ids"]) + self._get_last_prefill_talker_hidden(raw[-1:])
        dtype = self.model.text_projection.linear_fc1.weight.dtype
        kind = inp.kwargs.get("conditioning_kind", "text")
        if kind == "text":
            text_hidden = self.model.text_projection(raw.to(dtype))
        else:
            text_hidden = getattr(self, f"_tts_{kind}_embed_cached")
        # The incoming codec sum may be a shared graph output. Repeated input
        # preparation must not add conditioning into that source in place.
        return inp.input_embeds.to(dtype) + text_hidden

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

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[ARNodeInputs],
    ) -> dict[str, torch.Tensor | Any]:
        seq_lens = [
            inp.input_seq_len for inp in inputs
        ]
        input_embeds = torch.cat([
            self._materialize_input_embeds(graph_walk, inp) for inp in inputs
        ], dim=0)
        device = self.get_device()

        extra_args = {}
        # codec scratch is decode-only; talker_prefill runs the LLM backbone
        # alone and interning these would clash (codec_emb_sum's hidden dim
        # aliases the token-bucket seq dim).
        if graph_walk != "talker_prefill":
            extra_args["suppress_mask"] = self._get_suppress_mask()
            extra_args["all_codes"] = torch.zeros(
                (len(inputs), self.num_codes), device=device, dtype=torch.long
            )
            extra_args["codec_emb_sum"] = torch.zeros(
                (len(inputs), self.cp_cfg.hidden_size), device=device
            )
            extra_args["pos_buf"] = torch.zeros(
                (len(inputs), 1), device=device, dtype=torch.long
            )
        if graph_walk == "talker_last_prefill":
            extra_args["last_token_indices"] = (
                torch.tensor(seq_lens, device=device, dtype=torch.long).cumsum(0) - 1
            )
        return {
            "input_embeds": input_embeds,
            **extra_args
        }

    def _forward_prefill(
        self, input_embeds: torch.Tensor,
    ):
        self.model(
            input_embeds=input_embeds, label="main",
        )
        return {}

    def _forward_decode_like(
        self, request_ids: list[str],
        talker_sampler: SamplerResource,
        code_pred_sampler: SamplerResource,
        suppress_mask: torch.Tensor,
        all_codes: torch.Tensor,
        codec_emb_sum: torch.Tensor,
        pos_buf: torch.Tensor,
        is_batched_decode: bool,
        input_embeds: torch.Tensor,
        last_token_indices: torch.Tensor | None = None,
        **kwargs
    ):
        """
        Runs the Talker LLM for stages that grpoh walks that sample a token
        and feed into the code predictor.

        ``last_token_indices`` (when provided) is used to ``index_select`` the
        per-request last hidden out of a packed multi-token-per-request hidden
        — the batched ``talker_last_prefill`` path computes it in ``preprocess``
        as ``cumsum(seq_lens) - 1`` and passes it through here so the
        codec_head only sees one hidden per request. Mutually exclusive with
        the batched-decode (``hidden`` is already ``(bs, hidden)``) and
        non-batched (``hidden[-1:, :]``) branches.
        """
        hidden = self.model(
            input_embeds=input_embeds, label="main",
        )
        if last_token_indices is not None:
            last_hidden = hidden.index_select(0, last_token_indices)
        elif not is_batched_decode:
            last_hidden = hidden[-1:, :]
        else:
            last_hidden = hidden
        logits = self.model.codec_head(last_hidden)
        logits = logits.masked_fill(suppress_mask.unsqueeze(0), float("-inf"))
        # Apply the repetition penalty to the Talker's layer-0 codes (the code
        # predictor depth loop below samples through its own aux config and
        # stays penalty-free). The penalty is baked into the captured graph.
        layer0_codes = talker_sampler.sample(
            request_ids, logits
        )

        # code predictor section
        embed = self.talker_code_emb(layer0_codes)  # [bs, hidden]
        codec_emb_sum.add_(embed)
        all_codes[:, 0] = layer0_codes
        bs = all_codes.shape[0]

        kv_cache = self._get_cp_kv_cache()[:, :bs]

        # forward over [last_hidden] to update kv cache with the Talker's final hidden
        # state as context for the code prediction. This returns nothing because the
        # layer 0 code is already provided by the talker LLM
        cp = self.code_predictor
        codec_embedding = cp.model.codec_embedding
        cp.forward_depth_unrolled(
            last_hidden.unsqueeze(1), pos_buf, kv_cache, cache_pos=0,
        )
        pos_buf += 1

        for group_idx in range(1, self.num_codes):
            hidden = cp.forward_depth_unrolled(
                embed.unsqueeze(1), pos_buf, kv_cache, cache_pos=group_idx,
            ).squeeze(1)
            pos_buf += 1

            logits = torch.matmul(
                hidden, cp.lm_head_weight[group_idx - 1].t()
            )

            # Params come from the "code_predictor" aux sampler's static
            # buffers, so per-request values survive graph replay.
            tokens = code_pred_sampler.sample(
                request_ids, logits,
            )
            all_codes[:, group_idx] = tokens
            embed = codec_embedding[group_idx - 1](tokens)
            codec_emb_sum.add_(embed)

        return {
            "talker_input_embeds": [codec_emb_sum],
            "codec_tokens": [all_codes],
            "new_token": [layer0_codes]
        }

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> NameToTensorList:
        if graph_walk == "talker_prefill":
            return self._forward_prefill(
                input_embeds=input_embeds
            )
        return self._forward_decode_like(
            request_ids=engine_inputs.request_ids,
            talker_sampler=engine_inputs.resources[TALKER_SAMPLER],
            code_pred_sampler=engine_inputs.resources[CODE_PRED_SAMPLER],
            input_embeds=input_embeds,
            is_batched_decode=(graph_walk == "talker_decode"),
            **kwargs
        )

    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_embeds: torch.Tensor | None = None,
        **kwargs,
    ):
        """Batched Talker forward shared between ``talker_decode``, ``talker_prefill``, and ``talker_last_prefill``.

        Decode path (1 token per request, ``hidden`` shape ``(bs, hidden)``):
          Runs the full LLM + codec_head + suppress_mask via _forward_decode_like,
          samples inline through the injected samplers, and emits per-rid
          {talker_input_embeds, codec_tokens, new_token} entries.

        Prefill path (``talker_prefill``; multi-token-per-request, ``hidden``
        shape ``(total_tokens, hidden)``): Runs only the LLM backbone, and
        populates the KV cache.

        Last-prefill path (``talker_last_prefill``; fixed 6 tokens per request,
        ``hidden`` shape ``(bs * 6, hidden)``)
        """
        assert graph_walk in (
            "talker_decode", "talker_prefill", "talker_last_prefill",
        )

        if graph_walk == "talker_prefill":
            hidden = self.model(
                input_embeds=input_embeds,
                label="main",
            )
            return {
                "__batched_talker_prefill_hidden__": hidden,
            }

        last_token_indices = kwargs.pop("last_token_indices", None)
        if graph_walk == "talker_last_prefill":
            assert last_token_indices is not None, (
                "talker_last_prefill forward_batched requires "
                "last_token_indices from preprocess; got None."
            )

        fwd_out = self._forward_decode_like(
            request_ids=engine_inputs.request_ids,
            talker_sampler=engine_inputs.resources[TALKER_SAMPLER],
            code_pred_sampler=engine_inputs.resources[CODE_PRED_SAMPLER],
            is_batched_decode=True,
            last_token_indices=last_token_indices,
            input_embeds=input_embeds,
            **kwargs
        )

        outputs = {
            rid: {
                "talker_input_embeds": [fwd_out["talker_input_embeds"][0][i:i+1]],
                "codec_tokens": [fwd_out["codec_tokens"][0][i]],
                "new_token": [fwd_out["new_token"][0][i]],
            } for i, rid in enumerate(engine_inputs.request_ids)
        }
        return outputs

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

    def can_batch(self, batch: ExecutingBatch, model_inputs: list[NodeInputs]) -> bool:
        return batch.graph_walk in [
            "talker_decode", "talker_last_prefill"
        ] and len(model_inputs) <= self.MAX_BATCH_SIZE

    def max_batch_size(self, graph_walk):
        return self.MAX_BATCH_SIZE

    TALKER_PREFILL_TOKEN_BUCKETS = [128, 256, 512, 1024]
    TALKER_PREFILL_CAPTURE_BATCH_SIZES = [1, 2, 4]

    # Fixed assistant prefix per request: pad*4 + bos + projected_thinker = 6.
    TALKER_LAST_PREFILL_TOKENS_PER_REQ = 6
    TALKER_LAST_PREFILL_CAPTURE_BATCH_SIZES = [1, 2, 4, 8, 16, 32]

    def get_accelerator_graph_configs(
        self, device: torch.device, tp_world_size: int = 1,
    ) -> list[AcceleratorGraphConfig]:
        dtype = self.model.text_projection.linear_fc1.weight.dtype
        return [
            BatchedAcceleratorGraphConfig(
                capture_graph_walk="talker_decode",
                single_request_inputs=ARNodeInputs(
                    input_embeds=torch.zeros(
                        (1, self.config.talker_hidden_size),
                        device=device, dtype=torch.float32,
                    ),
                    tensor_inputs={"thinker_states": torch.zeros(
                        (1, self.config.thinker_hidden_size), device=device, dtype=dtype,
                    )},
                    input_seq_len=1,
                ),
                capture_batch_sizes=[1, 2, 4, 8, 16, 32],
                compile=True
            ),
            PackedAcceleratorGraphConfig(
                capture_graph_walk="talker_prefill",
                replay_graph_walks=["talker_prefill"],
                make_node_input=lambda n: ARNodeInputs(
                    tensor_inputs={
                        "thinker_states": torch.zeros(
                            (n, self.config.thinker_hidden_size), device=device, dtype=dtype,
                        ),
                        "multimodal_mask": torch.zeros(n, device=device, dtype=torch.bool),
                    },
                    input_seq_len=n,
                ),
                capture_token_lengths=self.TALKER_PREFILL_TOKEN_BUCKETS,
                capture_batch_sizes=self.TALKER_PREFILL_CAPTURE_BATCH_SIZES,
                compile=True
            ),
            BatchedAcceleratorGraphConfig(
                capture_graph_walk="talker_last_prefill",
                single_request_inputs=ARNodeInputs(
                    tensor_inputs={
                        "codec_ids": self._get_last_prefill_codec_ids("Ethan"),
                        "thinker_states": torch.cat([
                            self._tts_raw_embeddings[0:1].expand(4, -1),
                            self._tts_raw_embeddings[1:2],
                            torch.zeros((1, self.config.thinker_hidden_size), device=device, dtype=dtype),
                        ]),
                    },
                    input_seq_len=self.TALKER_LAST_PREFILL_TOKENS_PER_REQ,
                ),
                capture_batch_sizes=self.TALKER_LAST_PREFILL_CAPTURE_BATCH_SIZES,
                compile=True
            ),
        ]


# ===================================================================
# 5. Code2WavSubmodule
# ===================================================================


class Code2WavSubmodule(NodeSubmodule):
    """Wraps the HF Code2Wav vocoder for streaming chunk decode.

    Receives codec_tokens from the Talker (via StreamBuffer), selects
    the first ``num_quantizers`` codebook layers, runs the ConvNet
    vocoder, trims overlap context, and returns the PCM audio chunk.
    """

    # Runs in the checkpoint's bf16 without autocast. ``consolidate()``
    # compiles the forward itself, so the engine must not compile it again.
    disable_torch_compile = True
    disable_autocast = True

    # No capture carries this key, so a batch of terminal flushes runs eager
    # and skips the decode when none of its windows has a new frame.
    _CONTEXT_ONLY_KEY = "context_only"

    def __init__(self, code2wav_model: Qwen3OmniMoeCode2Wav, config: Qwen3OmniModelConfig):
        super().__init__()
        self.code2wav = code2wav_model
        self.config = config

        # Pre-compute the total upsample factor. HF defines this as
        # ``np.prod(upsample_rates + upsampling_ratios)`` — both tuples
        # contribute (upsample_rates via the decoder blocks, upsampling_ratios
        # via the upsample stack). For Qwen3-Omni this is 8*5*4*3*2*2 = 1920.
        total_upsample = 1
        for r in self.config.code2wav.upsample_rates:
            total_upsample *= r
        for r in self.config.code2wav.upsampling_ratios:
            total_upsample *= r
        self.total_upsample = total_upsample

        # Every window, from the ramp's first few frames to a steady
        # context + chunk window, is zero-padded to this one shape. The
        # vocoder is causal, so the padding changes only the look-ahead
        # samples that ``postprocess`` drops, and the convs keep the input
        # lengths of a full window (bf16 cuDNN convs go wrong at some other
        # odd lengths on B200).
        self.full_seqlen = self.config.code2wav.codec_max_window

    @staticmethod
    def _chunk(rid, per_request_input_metadata: Mapping[str, InputMetadata] | None):
        meta = (per_request_input_metadata or {}).get(rid)
        return meta.stream_chunks.get("codec_tokens") if meta is not None else None

    @classmethod
    def _maybe_context_only(
        cls, request_ids, per_request_input_metadata: Mapping[str, InputMetadata] | None,
    ) -> bool:
        """Whether every window may be a terminal flush that emits no audio:
        its new items are at most the Talker's codec_eos frame, which ends
        every stream the Talker stops itself, so a reply whose frames end on a
        chunk boundary ends with such a flush. The counts can't tell that
        frame from a real one, so ``preprocess`` checks the decoded frames."""
        for rid in request_ids:
            chunk = cls._chunk(rid, per_request_input_metadata)
            if chunk is None or chunk.num_items - chunk.context_items > int(chunk.is_final):
                return False
        return bool(request_ids)

    def cg_key_info(
        self,
        graph_walk: str,
        per_request_info: Mapping[str, CurrentForwardPassInfo],
        per_request_input_metadata: Mapping[str, InputMetadata] | None = None,
        **kwargs: Any,
    ) -> Any:
        if self._maybe_context_only(list(per_request_info), per_request_input_metadata):
            return self._CONTEXT_ONLY_KEY
        return None

    def get_accelerator_graph_configs(self, device, tp_world_size: int = 1):
        num_quantizers = self.config.code2wav.num_quantizers
        return [
            BatchedAcceleratorGraphConfig(
                capture_graph_walk="code2wav_chunk",
                single_request_inputs=ARNodeInputs(
                    tensor_inputs={
                        "codec_tokens": torch.zeros((
                            num_quantizers, self.full_seqlen,
                        ), dtype=torch.long, device=device),
                        "position_ids": torch.arange(self.full_seqlen, device=device)
                    },
                ),
                capture_batch_sizes=[1, 2, 4, 8, 16, 32],
                compile=False
            ),
        ]

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        input_metadata: InputMetadata | None = None,
        **kwargs
    ) -> NodeInputs:
        num_quantizers = self.config.code2wav.num_quantizers  # 16
        codec_eos = self.config.talker.codec_eos_token_id
        codec_tokens = inputs["codec_tokens"][0]

        # Reshape to (num_frames, num_code_groups) if flat
        if codec_tokens.dim() == 1:
            num_groups = self.config.num_code_groups  # 16 (Qwen3-Omni)
            if codec_tokens.shape[0] % num_groups == 0:
                codec_tokens = codec_tokens.view(-1, num_groups)
            else:
                codec_tokens = codec_tokens.unsqueeze(0)

        # Filter out codec_eos frames
        if codec_tokens.dim() == 2 and codec_tokens.shape[0] > 0:
            eos_mask = codec_tokens[:, 0] == codec_eos
            if eos_mask.any():
                codec_tokens = codec_tokens[~eos_mask]

        # Select first num_quantizers codebook layers
        if codec_tokens.shape[-1] > num_quantizers:
            codec_tokens = codec_tokens[..., :num_quantizers]

        # The window's leading frames that an earlier window already decoded.
        # EOS is only ever a stream's last item, but it can be a context item
        # of the final flush (the previous window's last new item). The filter
        # above dropped it, so the clamp keeps the context within the frames.
        chunk = (input_metadata.stream_chunks.get("codec_tokens") if input_metadata is not None else None)
        orig_seq_len = codec_tokens.shape[0]
        context = min(chunk.context_items if chunk is not None else 0, orig_seq_len)

        # pad sequence to full_seqlen for batching
        pad_len = self.full_seqlen - codec_tokens.shape[0]
        if pad_len > 0:
            pad = torch.zeros((
                pad_len, codec_tokens.shape[1]
            ),dtype=codec_tokens.dtype,
            device=codec_tokens.device)
            codec_tokens = torch.cat([codec_tokens, pad], dim=0)

        # Transpose to (Q, T)
        codec_tokens = codec_tokens.T  # (Q, T)
        # The window's geometry rides with this pass rather than in request
        # state: the engine may prepare a request's next window before it
        # postprocesses this one.
        return NodeInputs(
            tensor_inputs={
                "codec_tokens": codec_tokens,
                "position_ids": torch.arange(codec_tokens.shape[1], device=codec_tokens.device)
            },
            kwargs={"frames": orig_seq_len, "context": context},
        )

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[NodeInputs],
    ) -> dict[str, torch.Tensor | Any]:
        all_codec_tokens = [
            inp.tensor_inputs["codec_tokens"] for inp in inputs
        ]
        # Assert all requests have the same numel so they can be batched
        assert all(t.numel() == all_codec_tokens[0].numel() for t in all_codec_tokens), (
            f"All codec token inputs must have the same numel for batching, "
            f"got: {[t.numel() for t in all_codec_tokens]}"
        )
        # Stack into (bs, Q, T)
        batched_codec_tokens = torch.stack(all_codec_tokens, dim=0)
        position_ids = torch.stack([
            inp.tensor_inputs["position_ids"] for inp in inputs
        ], dim=0)
        metadata = engine_inputs.per_request_input_metadata
        if not engine_inputs.captured and self._maybe_context_only(engine_inputs.request_ids, metadata) and all(
            inp.kwargs["frames"] <= self._chunk(rid, metadata).context_items
            for rid, inp in zip(engine_inputs.request_ids, inputs, strict=True)
        ):
            return {"codec_tokens": batched_codec_tokens[..., :0], "position_ids": position_ids[..., :0]}
        return {"codec_tokens": batched_codec_tokens, "position_ids": position_ids}

    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        codec_tokens: torch.Tensor,
        position_ids: torch.Tensor,
        **kwargs
    ) -> dict[str, NameToTensorList]:
        """Decode a batch of padded windows to untrimmed int16 PCM.

        The Talker→Code2Wav StreamBuffer uses ``ScheduledLeftContextChunkPolicy``:
        chunk k for a request carries ``codec_chunk_schedule[k]`` new frames
        (``codec_chunk_frames`` once the schedule runs out) behind up to
        ``codec_left_context_frames`` frames an earlier chunk delivered, so
        the first chunk has no overlap. The overlap lets the vocoder warm up
        at chunk boundaries; ``postprocess`` cuts each request's new audio
        out of the window.

        A zero-frame batch (terminal flushes with no new frames, see
        ``preprocess``) decodes nothing and emits empty chunks.
        """
        request_ids = engine_inputs.request_ids
        if codec_tokens is None:
            return {rid: {} for rid in request_ids}
        if codec_tokens.numel() == 0:
            return {
                rid: {"audio_chunk": [torch.empty(0, dtype=torch.int16, device=codec_tokens.device)]}
                for rid in request_ids
            }

        wavs = self.code2wav(
            codec_tokens, position_ids
        )

        results: dict[str, NameToTensorList] = {}
        for i, rid in enumerate(request_ids):
            wav = wavs[i]
            audio_int16 = (wav.clamp(-1, 1) * 32767).to(torch.int16).squeeze()
            results[rid] = {"audio_chunk": [audio_int16]}
        return results

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        codec_tokens: torch.Tensor,
        position_ids: torch.Tensor,
        **kwargs
    ) -> NameToTensorList:
        """Raw vocoder forward -- returns int16 PCM without any trim.

        Prefer ``forward_batched`` for the streaming path; this method exists
        for callers that need a non-streaming, single-shot decode (e.g.
        debugging or offline batch use).
        """
        if codec_tokens is None or codec_tokens.numel() == 0:
            return {}

        wav = self.code2wav(codec_tokens, position_ids)
        audio_int16 = (wav.clamp(-1, 1) * 32767).to(torch.int16)
        return {"audio_chunk": [audio_int16]}

    def can_batch(self, batch: ExecutingBatch, model_inputs: list[NodeInputs]) -> bool:
        return len({
            inputs.tensor_inputs["codec_tokens"].numel() \
                for inputs in model_inputs
        }) == 1

    def can_use_accelerator_graphs(self, batch, model_inputs: list[NodeInputs]):
        res = super().can_use_accelerator_graphs(batch, model_inputs) \
            and self.can_batch(batch, model_inputs) \
                and model_inputs[0].tensor_inputs["codec_tokens"].shape[1] == self.full_seqlen
        return res

    def postprocess(
        self, request_id: str,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
        inputs: NodeInputs | None = None,
        **kwargs
    ):
        if "audio_chunk" not in outputs:
            return
        if inputs is None:
            raise ValueError("Code2Wav postprocess needs the pass's inputs for its window geometry")

        orig_seq_len = int(inputs.kwargs["frames"])
        left_context_size = int(inputs.kwargs["context"])
        audio = outputs["audio_chunk"][0].reshape(-1)
        if audio.numel() == 0:
            outputs["audio_chunk"][0] = audio
            return
        # Each decoder block's ConvTranspose1d(2r, stride r, padding r) needs
        # the next frame, so a window decodes to ``tail`` samples (555 for
        # Qwen3-Omni) short of window * total_upsample. The next window
        # re-decodes those frames as context and emits the missing samples, so
        # both ends of the slice move back by ``tail``. Moving the end also
        # drops the samples that looked ahead into the zero padding.
        window = max(orig_seq_len, self.full_seqlen)
        tail = max(window * self.total_upsample - audio.numel(), 0)
        start = max(left_context_size * self.total_upsample - tail, 0)
        end = max(orig_seq_len * self.total_upsample - tail, start)
        outputs["audio_chunk"][0] = audio[start:end]

