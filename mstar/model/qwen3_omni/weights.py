"""Qwen logical checkpoint tensors, before packing or device placement.

Thinker and audio tensors retain their checkpoint identities. TTS conditioning
uses three rows of the Thinker's embedding table; exposing those model-owned
values avoids retaining a second full text embedding table on the Talker worker.
"""
from __future__ import annotations

import torch

from mstar.model.qwen3_omni.config import Qwen3OmniModelConfig
from mstar.model.weights import LogicalWeightSource, WeightSpec


def talker_weight_specs(config: Qwen3OmniModelConfig) -> tuple[WeightSpec, ...]:
    specs: list[WeightSpec] = []

    def weight(name: str, shape: tuple[int, ...]):
        specs.append(WeightSpec(name=name, shape=shape))

    def attention(prefix, hidden, heads, kv_heads, head_dim):
        weight(prefix + ".q_proj.weight", (heads * head_dim, hidden))
        for name in ("k_proj", "v_proj"):
            weight(f"{prefix}.{name}.weight", (kv_heads * head_dim, hidden))
        weight(prefix + ".o_proj.weight", (hidden, heads * head_dim))
        weight(prefix + ".q_norm.weight", (head_dim,))
        weight(prefix + ".k_norm.weight", (head_dim,))

    def mlp(prefix, hidden, intermediate):
        for name in ("gate_proj", "up_proj"):
            weight(f"{prefix}.{name}.weight", (intermediate, hidden))
        weight(prefix + ".down_proj.weight", (hidden, intermediate))

    tc, cp = config.talker_text, config.code_predictor
    h = tc.hidden_size
    weight("talker.model.codec_embedding.weight", (tc.vocab_size, h))
    weight("talker.codec_head.weight", (tc.vocab_size, h))
    weight("talker.model.norm.weight", (h,))
    for name in ("text_projection", "hidden_projection"):
        prefix = "talker." + name
        weight(prefix + ".linear_fc1.weight", (tc.intermediate_size, config.thinker_hidden_size))
        weight(prefix + ".linear_fc1.bias", (tc.intermediate_size,))
        weight(prefix + ".linear_fc2.weight", (h, tc.intermediate_size))
        weight(prefix + ".linear_fc2.bias", (h,))
    for i in range(tc.num_hidden_layers):
        prefix = f"talker.model.layers.{i}"
        weight(prefix + ".input_layernorm.weight", (h,))
        weight(prefix + ".post_attention_layernorm.weight", (h,))
        attention(prefix + ".self_attn", h, tc.num_attention_heads, tc.num_key_value_heads, tc.head_dim)
        weight(prefix + ".mlp.gate.weight", (tc.num_experts, h))
        for expert in range(tc.num_experts):
            mlp(f"{prefix}.mlp.experts.{expert}", h, tc.moe_intermediate_size)
        mlp(prefix + ".mlp.shared_expert", h, tc.shared_expert_intermediate_size)
        weight(prefix + ".mlp.shared_expert_gate.weight", (1, h))
    prefix = "talker.code_predictor"
    weight(prefix + ".model.norm.weight", (cp.hidden_size,))
    for group in range(config.num_code_groups - 1):
        weight(f"{prefix}.model.codec_embedding.{group}.weight", (cp.vocab_size, cp.hidden_size))
        weight(f"{prefix}.lm_head.{group}.weight", (cp.vocab_size, cp.hidden_size))
    for i in range(cp.num_hidden_layers):
        layer = f"{prefix}.model.layers.{i}"
        weight(layer + ".input_layernorm.weight", (cp.hidden_size,))
        weight(layer + ".post_attention_layernorm.weight", (cp.hidden_size,))
        attention(layer + ".self_attn", cp.hidden_size, cp.num_attention_heads,
                  cp.num_key_value_heads, cp.head_dim)
        mlp(layer + ".mlp", cp.hidden_size, cp.intermediate_size)
    specs.extend(conditioning_weight_specs(config))
    return tuple(specs)


def conditioning_weight_specs(config: Qwen3OmniModelConfig) -> tuple[WeightSpec, ...]:
    return tuple(WeightSpec(f"talker.conditioning.tts_{name}", (config.thinker_hidden_size,))
                 for name in ("pad", "bos", "eos"))


def _conditioning_tensors(local_dir: str, config: Qwen3OmniModelConfig):
    from mstar.model.loader.iterators import TensorSlice, iter_safetensors_shards

    ids = {name: getattr(config, f"tts_{name}_token_id") for name in ("pad", "bos", "eos")}
    start, stop = min(ids.values()), max(ids.values()) + 1
    for _, value in iter_safetensors_shards(
        local_dir, device="cpu", keys={"thinker.model.embed_tokens.weight"},
        slice_spec=lambda _: TensorSlice(0, start, stop),
    ):
        for name, token_id in ids.items():
            yield f"talker.conditioning.tts_{name}", value[token_id - start].clone()


def load_conditioning_embeddings(local_dir: str, config: Qwen3OmniModelConfig) -> torch.Tensor:
    """Read the three logical constants without loading Talker parameters."""
    source = LogicalWeightSource(conditioning_weight_specs(config),
                                 lambda: _conditioning_tensors(local_dir, config))
    values = source.load()
    return torch.stack([values[f"talker.conditioning.tts_{name}"] for name in ("pad", "bos", "eos")])


class QwenLogicalWeights(LogicalWeightSource):
    def __init__(self, local_dir, config):
        self.local_dir, self.config = local_dir, config
        super().__init__((*talker_weight_specs(config), *encoder_thinker_weight_specs(local_dir)), lambda: ())

    def load(self, names=None):
        from mstar.model.loader.iterators import iter_safetensors_shards
        names = tuple(self.by_name) if names is None else tuple(names)
        def tensors():
            yield from iter_safetensors_shards(self.local_dir, device="cpu", keys=set(names))
            if any(name.startswith("talker.conditioning.") for name in names):
                yield from _conditioning_tensors(self.local_dir, self.config)
        return LogicalWeightSource(self.specs, tensors).load(names)


def logical_weights(local_dir, config):
    return QwenLogicalWeights(local_dir, config)


def encoder_thinker_weight_specs(local_dir):
    import json
    from pathlib import Path

    from safetensors import safe_open
    root = Path(local_dir)
    index_path = root / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())["weight_map"]
    else:
        index = {}
        for shard in root.glob("*.safetensors"):
            with safe_open(shard, framework="pt", device="cpu") as source:
                index.update({name: shard.name for name in source.keys()})
    selected = {name: shard for name, shard in index.items()
                if name.startswith(("thinker.model.", "thinker.lm_head.", "thinker.audio_tower."))}
    for shard in sorted(set(selected.values())):
        with safe_open(root / shard, framework="pt", device="cpu") as source:
            for name in sorted(name for name, file in selected.items() if file == shard):
                yield WeightSpec(name, tuple(source.get_slice(name).get_shape()))
