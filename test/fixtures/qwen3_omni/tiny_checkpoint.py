"""Deterministic small logical checkpoint for Qwen integration tests.

Test code may reuse this helper. Production modules must not import it.
"""
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from mstar.model.qwen3_omni.config import Qwen3OmniModelConfig
from mstar.model.qwen3_omni.weights import talker_weight_specs


def write_tiny_checkpoint(path: str | Path, *, num_code_groups: int = 16):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    raw = {
        "tts_pad_token_id": 1, "tts_bos_token_id": 2, "tts_eos_token_id": 3,
        "thinker_config": {"text_config": {
            "vocab_size": 16, "hidden_size": 48, "num_attention_heads": 3,
            "num_key_value_heads": 1, "head_dim": 16, "num_hidden_layers": 2,
            "num_experts": 2,
        }},
        "talker_config": {
            "num_code_groups": num_code_groups, "thinker_hidden_size": 48,
            "codec_nothink_id": 2, "codec_think_bos_id": 3, "codec_think_eos_id": 4,
            "codec_pad_id": 5, "codec_bos_id": 6, "codec_eos_token_id": 7,
            "speaker_id": {"ethan": 8},
            "text_config": {
                "hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 2,
                "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 16,
                "num_experts": 2, "num_experts_per_tok": 2, "moe_intermediate_size": 32,
                "shared_expert_intermediate_size": 32, "vocab_size": 1100,
            },
            "code_predictor_config": {
                "hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 2,
                "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 16,
                "vocab_size": 64, "num_code_groups": num_code_groups,
            },
        },
    }
    (path / "config.json").write_text(json.dumps(raw))
    cfg = Qwen3OmniModelConfig.from_pretrained(path)
    rng = torch.Generator().manual_seed(3107)
    values = {
        spec.name: torch.randn(spec.shape, generator=rng) * 0.05
        for spec in talker_weight_specs(cfg) if ".conditioning." not in spec.name
    }
    for name, value in values.items():
        if "norm.weight" in name:
            value.fill_(1)
    values["thinker.model.embed_tokens.weight"] = torch.randn(16, 48, generator=rng)
    save_file(values, path / "model.safetensors")
    return path, values
