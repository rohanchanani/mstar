"""Physical preparation for the retained Talker Torch implementation.

This constructor belongs to the retained default deployment path. External
accelerations supply their own physical program from logical checkpoint tensors.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from mstar.model.loader.base import StackedParamRule


def stacked_parameter_rules(num_experts: int) -> list[StackedParamRule]:
    rules = []
    for i in range(num_experts):
        for source, target, shard in (
            ("gate", "gate_up", "gate"),
            ("up", "gate_up", "up"),
            ("down", "down", "down"),
        ):
            rules.append(StackedParamRule(
                target_suffix=f".experts.{target}_proj",
                source_suffix=f".experts.{source}_proj.__expert{i}__.weight",
                shard_id=f"{shard}:{i}",
            ))
    # Longer expert suffixes must precede the dense projection suffixes.
    rules.extend([
        StackedParamRule(".gate_up_proj", ".gate_proj", 0),
        StackedParamRule(".gate_up_proj", ".up_proj", 1),
        StackedParamRule(".qkv_proj", ".q_proj", "q"),
        StackedParamRule(".qkv_proj", ".k_proj", "k"),
        StackedParamRule(".qkv_proj", ".v_proj", "v"),
    ])
    return rules


@dataclass(frozen=True)
class TalkerProgram:
    model: torch.nn.Module
    code_predictor: torch.nn.Module
    tts_embeddings: torch.Tensor


def build_talker_program(model, *, device, dtype, tp_group=None) -> TalkerProgram:
    from mstar.model.components.moe import prepare_moe_weights
    from mstar.model.loader import load_hf_weights
    from mstar.model.qwen3_omni.components.talker import Qwen3OmniCodePredictor, Qwen3OmniTalkerModel

    config = model.config
    source = model.logical_weights()
    values = source.load(name for name in source.by_name if name.startswith("talker."))
    rules = stacked_parameter_rules(config.talker_text.num_experts)
    with torch.device("meta"):
        talker = Qwen3OmniTalkerModel(config, comm_group=tp_group)
        predictor = Qwen3OmniCodePredictor(config)
    for module, prefix, exclude in (
        (talker, "talker.", ("talker.code_predictor.", "talker.conditioning.")),
        (predictor, "talker.code_predictor.", ()),
    ):
        module.to(dtype=dtype).to_empty(device=device)
        weights = (
            (name.removeprefix(prefix), value.to(device=device, dtype=dtype))
            for name, value in values.items()
            if name.startswith(prefix) and not name.startswith(exclude)
        )
        loaded = load_hf_weights(
            module, weights, stacked_params=rules,
            name_remapper=model._thinker_remap,
        )
        missing = set(dict(module.named_parameters())) - loaded
        if missing:
            raise ValueError(f"Talker reference did not initialize parameters: {sorted(missing)}")
        module.eval()
    prepare_moe_weights(talker)
    predictor.consolidate_stacked_weights()
    tts_embeddings = torch.stack([
        values[f"talker.conditioning.tts_{name}"] for name in ("pad", "bos", "eos")
    ]).to(device=device, dtype=dtype)
    return TalkerProgram(talker, predictor, tts_embeddings)
