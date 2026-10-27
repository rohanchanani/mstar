"""Whisper's logical parameters, independently of a compute representation.

The identities deliberately retain separate Q/K/V projections.  Fused QKV,
packing, sharding and conversion belong to the program that consumes them.
"""
from __future__ import annotations

from collections.abc import Iterable

import torch

from mstar.model.weights import LogicalWeightSource, WeightSpec
from mstar.model.whisper.config import WhisperModelConfig


def weight_specs(config: WhisperModelConfig) -> tuple[WeightSpec, ...]:
    specs: list[WeightSpec] = []
    h = config.d_model

    def weight(name: str, shape: tuple[int, ...], *, zero: bool = False):
        specs.append(WeightSpec(
            name=name, shape=shape, source="model." + name,
            default=(lambda shape=shape: torch.zeros(shape, dtype=torch.float32))
            if zero else None,
        ))

    def norm(prefix: str):
        weight(prefix + ".weight", (h,))
        weight(prefix + ".bias", (h,))

    def attention(prefix: str):
        for projection in ("q_proj", "k_proj", "v_proj", "out_proj"):
            weight(f"{prefix}.{projection}.weight", (h, h))
            # Whisper's logical K bias is zero, absent from HF checkpoints.
            weight(f"{prefix}.{projection}.bias", (h,), zero=projection == "k_proj")

    weight("encoder.conv1.weight", (h, config.num_mel_bins, 3))
    weight("encoder.conv1.bias", (h,))
    weight("encoder.conv2.weight", (h, h, 3))
    weight("encoder.conv2.bias", (h,))
    weight("encoder.embed_positions.weight", (config.max_source_positions, h))
    for i in range(config.encoder_layers):
        prefix = f"encoder.layers.{i}"
        norm(prefix + ".self_attn_layer_norm")
        attention(prefix + ".self_attn")
        norm(prefix + ".final_layer_norm")
        weight(prefix + ".fc1.weight", (config.encoder_ffn_dim, h))
        weight(prefix + ".fc1.bias", (config.encoder_ffn_dim,))
        weight(prefix + ".fc2.weight", (h, config.encoder_ffn_dim))
        weight(prefix + ".fc2.bias", (h,))
    norm("encoder.layer_norm")

    weight("decoder.embed_tokens.weight", (config.vocab_size, h))
    weight("decoder.embed_positions.weight", (config.max_target_positions, h))
    for i in range(config.decoder_layers):
        prefix = f"decoder.layers.{i}"
        norm(prefix + ".self_attn_layer_norm")
        attention(prefix + ".self_attn")
        norm(prefix + ".encoder_attn_layer_norm")
        attention(prefix + ".encoder_attn")
        norm(prefix + ".final_layer_norm")
        weight(prefix + ".fc1.weight", (config.decoder_ffn_dim, h))
        weight(prefix + ".fc1.bias", (config.decoder_ffn_dim,))
        weight(prefix + ".fc2.weight", (h, config.decoder_ffn_dim))
        weight(prefix + ".fc2.bias", (h,))
    norm("decoder.layer_norm")
    specs.append(WeightSpec(
        "decoder.proj_out.weight", (config.vocab_size, h),
        alias_of="decoder.embed_tokens.weight",
    ))
    return tuple(specs)


def logical_weights(local_dir: str, config: WhisperModelConfig) -> LogicalWeightSource:
    def checkpoint_tensors() -> Iterable[tuple[str, torch.Tensor]]:
        from mstar.model.loader.iterators import iter_safetensors_shards
        yield from iter_safetensors_shards(local_dir, device="cpu")

    return LogicalWeightSource(weight_specs(config), checkpoint_tensors)
