"""Optional CPU arithmetic check of the real Qwen code predictor against HF.

Set QWEN_OMNI_TEST_CHECKPOINT to a local snapshot with predictor shards. No
download occurs. This uses explicit CPU RoPE/dense-attention oracles in the
retained predictor loop; it does not qualify ROCm kernels or graph capture.
"""
import json
import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("triton")

from mstar.model.loader import load_hf_weights
from mstar.model.loader.iterators import iter_safetensors_shards
from mstar.model.qwen3_omni.components import talker
from mstar.model.qwen3_omni.config import Qwen3OmniModelConfig
from mstar.model.qwen3_omni.reference import stacked_parameter_rules
from mstar.model.qwen3_omni.weights import talker_weight_specs
from mstar.model.weights import LogicalWeightSource, WeightSpec


@pytest.fixture(scope="module")
def predictors():
    location = os.environ.get("QWEN_OMNI_TEST_CHECKPOINT")
    if location is None:
        pytest.skip("set QWEN_OMNI_TEST_CHECKPOINT for the optional real-weight host check")
    snapshot = Path(location)
    config = Qwen3OmniModelConfig.from_pretrained(snapshot)
    prefix = "talker.code_predictor."
    specs = [s for s in talker_weight_specs(config) if s.name.startswith(prefix)]
    specs.append(WeightSpec("talker.model.codec_embedding.weight",
                            (config.talker_text.vocab_size, config.talker_hidden_size)))
    keys = {spec.name for spec in specs}
    values = LogicalWeightSource(
        specs, lambda: iter_safetensors_shards(snapshot, device="cpu", keys=keys),
    ).load()
    with torch.device("meta"):
        mstar = talker.Qwen3OmniCodePredictor(config)
    mstar.to(dtype=torch.float32).to_empty(device="cpu")
    loaded = load_hf_weights(
        mstar, ((name.removeprefix(prefix), value.float()) for name, value in values.items()
                if name.startswith(prefix)), stacked_params=stacked_parameter_rules(0),
    )
    assert loaded == set(dict(mstar.named_parameters()))
    mstar.consolidate_stacked_weights()
    mstar.eval()

    from transformers import AutoConfig
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeTalkerCodePredictorModel
    hf_config = AutoConfig.from_pretrained(snapshot, local_files_only=True).talker_config.code_predictor_config
    with torch.device("meta"):
        hf = Qwen3OmniMoeTalkerCodePredictorModel._from_config(hf_config, attn_implementation="eager")
    hf.load_state_dict({name.removeprefix(prefix + "model."): value.float()
                        for name, value in values.items() if name.startswith(prefix + "model.")},
                       strict=True, assign=True)
    # Nonpersistent frequency buffers are not checkpoint weights.
    hf.rotary_emb = type(hf.rotary_emb)(hf_config, device="cpu")
    hf.eval()
    yield config, mstar, hf, values["talker.model.codec_embedding.weight"].float()


def _qk_norm_rope(x, weight, *, pos, eps, rope_theta):
    normed = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps)
    normed = normed * weight.float()
    half = x.shape[-1] // 2
    inv = 1.0 / (rope_theta ** (torch.arange(half).float() / half))
    angles = pos.float()[:, None] * inv[None, :]
    cos, sin = angles.cos()[:, None], angles.sin()[:, None]
    first, second = normed.chunk(2, -1)
    return torch.cat([first * cos - second * sin, second * cos + first * sin], -1).to(x.dtype)


def _decode_attention(q, k, v, length):
    return F.scaled_dot_product_attention(
        q.transpose(1, 2), k[:, :length].transpose(1, 2), v[:, :length].transpose(1, 2),
        enable_gqa=True,
    ).transpose(1, 2)


def _eager_decode_attention(q, k, v, length):
    # Same elementary arithmetic order as HF eager attention; this is an
    # independent host oracle, not a serving fallback or a GPU-kernel test.
    q = q.transpose(1, 2)
    repeats = q.shape[1] // k.shape[2]
    k = k[:, :length].transpose(1, 2).repeat_interleave(repeats, dim=1)
    v = v[:, :length].transpose(1, 2).repeat_interleave(repeats, dim=1)
    scores = (q @ k.transpose(-2, -1)) * q.shape[-1] ** -0.5
    return (scores.softmax(-1, dtype=torch.float32) @ v).transpose(1, 2)


@pytest.mark.parametrize("seed", [407, 829])
@pytest.mark.parametrize("attention_oracle", ["sdpa", "eager"])
@torch.no_grad()
def test_all_predictor_depths_match_hf_for_fresh_teacher_forced_history(
    predictors, monkeypatch, seed, attention_oracle,
):
    from transformers.cache_utils import DynamicCache

    config, mstar, hf, layer0_embedding = predictors
    monkeypatch.setattr(talker, "fused_qk_norm_rope", _qk_norm_rope)
    monkeypatch.setattr(talker, "decode_attn_nhd", {
        "sdpa": _decode_attention, "eager": _eager_decode_attention,
    }[attention_oracle])
    rng = torch.Generator().manual_seed(seed)
    cp, groups = config.code_predictor, config.num_code_groups
    history = torch.randint(cp.vocab_size, (groups,), generator=rng)
    context = torch.randn(1, 1, cp.hidden_size, generator=rng)
    inputs = [context, layer0_embedding[history[0]].reshape(1, 1, -1)]
    inputs.extend(mstar.model.codec_embedding[i - 1](history[i:i + 1]).reshape(1, 1, -1)
                  for i in range(1, groups - 1))
    # Compare matching sequential execution. A whole-sequence HF call changes
    # FP32 GEMM reduction order and is not the numerical decode reference.
    hf_cache = DynamicCache(config=hf.config)
    expected = torch.cat([
        hf(inputs_embeds=value, past_key_values=hf_cache, use_cache=True).last_hidden_state
        for value in inputs
    ], dim=1)
    cache = torch.full((cp.num_hidden_layers, 1, 2, groups, cp.num_key_value_heads, cp.head_dim), float("nan"))
    actual = torch.cat([
        mstar.forward_depth_unrolled(value, torch.tensor([[position]]), cache, position)
        for position, value in enumerate(inputs)
    ], dim=1)
    assert torch.isfinite(cache).all()
    logit_errors = []
    logit_pairs = []
    for group in range(1, groups):
        weight = mstar.lm_head_weight[group - 1]
        reference = F.linear(expected[:, group], weight)
        observed = F.linear(actual[:, group], weight)
        logit_pairs.append((observed, reference))
        logit_errors.append(float((observed - reference).abs().max()))
    print(json.dumps({"scope": "real-weight CPU teacher-forced predictor arithmetic",
                      "seed": seed, "attention_oracle": attention_oracle, "depths": groups,
                      "weight_dtype": "float32 from checkpoint",
                      "hidden_max_abs_error": float((actual - expected).abs().max()),
                      "logits_max_abs_error": max(logit_errors),
                      "gpu_execution": False}))
    tolerance = 0 if attention_oracle == "eager" else 1e-4
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    for observed, reference in logit_pairs:
        torch.testing.assert_close(observed, reference, rtol=tolerance, atol=tolerance)
