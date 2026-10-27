"""Explicit executable references; never constructed by fresh shell creation.

The encoder is Hugging Face's WhisperEncoder.  The decoder is M*'s original
WhisperDecoderModel and consumes the same resource plans as physical programs.
"""
from __future__ import annotations

from dataclasses import asdict

import torch
from torch import nn

from mstar.engine.accelerator_graph_config import AcceleratorGraphConfig, BatchedAcceleratorGraphConfig
from mstar.model.submodule_base import ARNodeInputs, NodeInputs
from mstar.model.whisper.config import ATTN, POS, SAMPLER, WhisperModelConfig
from mstar.model.whisper.submodules import WhisperDecoderSubmodule


class EncoderReference(nn.Module):
    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder

    def run(self, audio_features: torch.Tensor) -> torch.Tensor:
        parameter = next(self.encoder.parameters())
        features = audio_features.to(device=parameter.device, dtype=parameter.dtype)
        if features.ndim == 2:
            features = features.unsqueeze(0)
        return self.encoder(features).last_hidden_state.squeeze(0)


class DecoderReference(nn.Module):
    def __init__(self, decoder: nn.Module, config: WhisperModelConfig):
        super().__init__()
        self.decoder = decoder
        self.config = config
        self._suppress_ids = None
        self._begin_suppress_ids = None

    def _apply_suppression(self, logits, *, first):
        # Retain the original reference's once-per-device preparation: a
        # captured legacy decode must not allocate/copy host constants anew.
        if self._suppress_ids is None or self._suppress_ids.device != logits.device:
            self._suppress_ids = torch.tensor(self.config.suppress_tokens, dtype=torch.long, device=logits.device)
            self._begin_suppress_ids = torch.tensor(
                self.config.begin_suppress_tokens, dtype=torch.long, device=logits.device,
            )
        if self._suppress_ids.numel():
            logits.index_fill_(-1, self._suppress_ids, float("-inf"))
        if first and self._begin_suppress_ids.numel():
            logits.index_fill_(-1, self._begin_suppress_ids, float("-inf"))
        return logits

    def run(self, graph_walk, engine_inputs, input_ids, encoder_states=None,
            *, return_logits=False):
        parameter = next(self.decoder.parameters())
        input_ids = input_ids.to(parameter.device)
        if encoder_states is not None:
            encoder_states = encoder_states.to(device=parameter.device, dtype=parameter.dtype)
            self.decoder.write_cross_kv(encoder_states)
        positions = engine_inputs.resources[POS].pos_ids("main")
        hidden = self.decoder(input_embeds=self.decoder.embed(input_ids, positions), label="main")
        if graph_walk == "prefill":
            hidden = engine_inputs.resources[ATTN].select_last_hidden(hidden)
        logits = self._apply_suppression(self.decoder.lm_head(hidden), first=graph_walk == "prefill")
        if return_logits:
            return logits
        return engine_inputs.resources[SAMPLER].sample(engine_inputs.request_ids, logits=logits)


def build_encoder(config, weights, *, device="cpu", dtype=torch.float32):
    from transformers import WhisperConfig
    from transformers.models.whisper.modeling_whisper import WhisperEncoder

    hf = WhisperConfig(**asdict(config), pad_token_id=config.eos_token_id,
                       bos_token_id=config.decoder_start_token_id)
    encoder = WhisperEncoder._from_config(hf, attn_implementation="sdpa")
    state = {name: weights["encoder." + name] for name in encoder.state_dict()}
    encoder.load_state_dict(state, strict=True)
    return EncoderReference(encoder.to(device=device, dtype=dtype).eval())


def build_decoder(config, weights, *, device="cpu", dtype=torch.float32, tp_group=None):
    from mstar.model.loader import WHISPER_STACKED_PARAMS, load_hf_weights
    from mstar.model.whisper.components.decoder import WhisperDecoderModel

    with torch.device("meta"):
        decoder = WhisperDecoderModel(config, comm_group=tp_group)
    decoder.to(dtype=dtype).to_empty(device=device)
    tensors = ((name.removeprefix("decoder."), tensor.to(device=device, dtype=dtype))
               for name, tensor in weights.items() if name.startswith("decoder."))
    load_hf_weights(decoder, tensors, stacked_params=WHISPER_STACKED_PARAMS,
                    name_remapper=lambda name: name.replace("self_attn.out_proj", "self_attn.o_proj"))
    decoder.zero_missing_biases()
    return DecoderReference(decoder.eval(), config)

class WhisperDecoderReferenceSubmodule(WhisperDecoderSubmodule):
    """The retained Torch implementation's batching and capture strategy."""

    DECODE_CAPTURE_BATCH_SIZES = [1, 2, 4, 8, 16]

    def get_accelerator_graph_configs(
        self, device: torch.device, tp_world_size: int = 1,
    ) -> list[AcceleratorGraphConfig]:
        return [
            BatchedAcceleratorGraphConfig(
                capture_graph_walk="decode",
                single_request_inputs=ARNodeInputs(
                    input_ids=torch.zeros(1, dtype=torch.long, device=device),
                    input_seq_len=1,
                ),
                capture_batch_sizes=self.DECODE_CAPTURE_BATCH_SIZES,
            ),
        ]

    # Prefill isn't captured, so the engine's capture cap doesn't bound it and
    # `can_batch` would take the whole ready set. A set too big for the context
    # cache can't be admitted; with offload configured it evicts and retries,
    # without it there is nothing to evict and the batch reforms identically.
    MAX_BATCH_SIZE = 16

    def can_batch(
        self, batch,
        model_inputs: list[NodeInputs],
    ) -> bool:
        return True

    def max_batch_size(self, graph_walk: str) -> int | None:
        del graph_walk
        return self.MAX_BATCH_SIZE
