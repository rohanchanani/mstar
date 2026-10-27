"""Compile actual Qwen kernels to gfx942 HSACO without a GPU or a launch.

Opt in with MSTAR_TEST_AMD_CODEGEN=1; do not set TRITON_INTERPRET. The real
Triton AMD compiler consumes the existing JIT kernels, their runtime pointer
types, and the actual launch configurations. This checks compiler acceptance
and emitted code, not GPU execution, output accuracy, occupancy, or capture.

Use --basetemp to retain assembly/HSACO/metadata in a chosen evidence directory.
The same checks run with an isolated production Triton-ROCm on PYTHONPATH.
"""

import json
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("MSTAR_TEST_AMD_CODEGEN") != "1",
    reason="opt-in offline gfx942 compilation: MSTAR_TEST_AMD_CODEGEN=1",
)


def _compile(tmp_path, fn, pointers, constants, options, scalar_types=None):
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    assert os.environ.get("TRITON_INTERPRET") != "1", "offline codegen requires actual JIT functions"
    signature = dict.fromkeys(fn.arg_names, "i32")
    signature.update(pointers)
    signature.update(scalar_types or {})
    signature.update(dict.fromkeys(constants, "constexpr"))
    assert set(signature) == set(fn.arg_names)
    result = triton.compile(
        ASTSource(fn, signature, constexprs=constants),
        target=GPUTarget("hip", "gfx942", 64), options=options,
    )
    assembly, binary = result.asm["amdgcn"], result.asm["hsaco"]
    assert "gfx942" in assembly
    assert binary[:4] == b"\x7fELF" and len(binary) > 256
    assert result.metadata.warp_size == 64
    assert result.metadata.num_warps * result.metadata.warp_size <= 1024
    (tmp_path / "kernel.amdgcn").write_text(assembly)
    (tmp_path / "kernel.hsaco").write_bytes(binary)
    (tmp_path / "metadata.json").write_text(json.dumps({
        "kernel": fn.__name__, "triton": triton.__version__,
        "signature": signature, "constants": constants, "options": options,
        "metadata": result.metadata._asdict(), "hsaco_bytes": len(binary),
        "gpu_execution": False,
    }, indent=2, default=str) + "\n")


@pytest.mark.parametrize("dtype", ["bf16", "fp32"])
def test_predictor_qk_norm_rope_codegen(tmp_path, dtype):
    from mstar.utils.attention import _qk_norm_rope_kernel

    _compile(tmp_path, _qk_norm_rope_kernel,
             {"X_ptr": f"*{dtype}", "W_norm_ptr": f"*{dtype}", "Pos_ptr": "*i32"},
             {"D": 128, "HALF_D": 64}, {"num_warps": 4, "num_stages": 1},
             {"eps": "fp32", "rope_theta": "fp32"})


@pytest.mark.parametrize("dtype", ["bf16", "fp32"])
@pytest.mark.parametrize("config_index", range(5))
def test_predictor_attention_autotune_codegen(tmp_path, dtype, config_index):
    from mstar.utils.attention import _decode_attn_nhd_kernel

    cfg = _decode_attn_nhd_kernel.configs[config_index]
    _compile(tmp_path, _decode_attn_nhd_kernel.fn,
             dict.fromkeys(("Q_ptr", "K_ptr", "V_ptr", "O_ptr"), f"*{dtype}"),
             {"H_q": 16, "H_kv": 8, "GROUP": 2, "D": 128, **cfg.kwargs},
             {"num_warps": cfg.num_warps, "num_stages": cfg.num_stages},
             {"sm_scale": "fp32"})


@pytest.mark.parametrize("config_index", range(6))
def test_launchable_fused_sampling_autotune_codegen(tmp_path, config_index):
    from mstar.engine.resources.sampler.kernels import _fused_sampling_prep_kernel

    # Same wave64 launchability rule used by the actual runtime autotuner.
    configs = [cfg for cfg in _fused_sampling_prep_kernel.configs if cfg.num_warps * 64 <= 1024]
    assert len(configs) == 6
    cfg = configs[config_index]
    _compile(tmp_path, _fused_sampling_prep_kernel.fn,
             {"logits_ptr": "*fp32", "temperature_ptr": "*fp32", "penalty_ptr": "*fp32",
              "seen_mask_ptr": "*i1", "probs_ptr": "*fp32"},
             {"APPLY_PENALTY": True, "INCLUDE_GREEDY": True, **cfg.kwargs},
             {"num_warps": cfg.num_warps, "num_stages": cfg.num_stages})


@pytest.mark.parametrize("stage", ["partials", "combine", "write"])
def test_split_sampling_codegen(tmp_path, stage):
    from mstar.engine.resources.sampler import kernels

    fn = getattr(kernels, f"_split_{stage}_kernel")
    pointers = {name: "*fp32" for name in fn.arg_names if name.endswith("_ptr")}
    for name in pointers:
        if "arg_ptr" in name:
            pointers[name] = "*i32"
        elif name == "seen_mask_ptr":
            pointers[name] = "*i1"
    constants = {"INCLUDE_GREEDY": True}
    if stage == "combine":
        constants["BLOCK_N"] = 32
    else:
        constants.update(BLOCK_SIZE=4096, APPLY_PENALTY=True)
    _compile(tmp_path, fn, pointers, constants,
             {"num_warps": 4 if stage == "combine" else 8, "num_stages": 2})


@pytest.mark.parametrize("model,intermediate,hidden,topk", [
    ("talker", 384, 1024, 6), ("thinker", 768, 2048, 8),
])
@pytest.mark.parametrize("batch", [1, 2, 8])
@pytest.mark.parametrize("stage", ["gate_up", "down"])
def test_tuned_bf16_moe_codegen(tmp_path, model, intermediate, hidden, topk, batch, stage):
    import triton.language as tl

    from mstar.utils.fused_moe.config import CONFIG_DIR, MoEProblem, load_tuned_configs, nearest_config
    from mstar.utils.fused_moe.kernels import fused_moe_kernel

    assert model in ("talker", "thinker")
    table = load_tuned_configs(MoEProblem(128, intermediate, hidden, topk),
                               "AMD_Instinct_MI300X", Path(CONFIG_DIR))
    assert table is not None
    cfg = nearest_config(table, batch)
    constants = {name: value for name, value in cfg.items() if name in fused_moe_kernel.arg_names}
    constants.update(MUL_ROUTED_WEIGHT=stage == "down", top_k=1 if stage == "down" else topk,
                     compute_type=tl.bfloat16,
                     even_Ks=(intermediate if stage == "down" else hidden) % cfg["BLOCK_SIZE_K"] == 0)
    options = {name: value for name, value in cfg.items() if name not in fused_moe_kernel.arg_names}
    _compile(tmp_path, fused_moe_kernel,
             {"a_ptr": "*bf16", "b_ptr": "*bf16", "c_ptr": "*bf16", "topk_weights_ptr": "*fp32",
              "sorted_token_ids_ptr": "*i32", "expert_ids_ptr": "*i32", "num_tokens_post_padded_ptr": "*i32"},
             constants, options)
