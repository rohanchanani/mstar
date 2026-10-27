"""Qwen's logical boundary and real small-weight Torch preparation on the host.

The lifecycle rehearsal uses explicit CPU attention/RoPE/sampling oracles.
It is not a ROCm service, graph-capture or numerical-qualification verdict.
"""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from mstar.engine.resources.kv.config import SequenceKVConfig
from mstar.model.binding import NodeBinding
from mstar.model.qwen3_omni.config import (
    CODE_PRED_SAMPLER,
    TALKER_ATTN,
    TALKER_KV,
    TALKER_POS,
    TALKER_SAMPLER,
    Qwen3OmniModelConfig,
)
from mstar.model.qwen3_omni.qwen3_omni_model import Qwen3OmniModel
from mstar.model.qwen3_omni.weights import talker_weight_specs
from mstar.model.submodule_base import ModelInputsFromEngine


@pytest.fixture
def tiny_checkpoint(tmp_path, request):
    import runpy
    fixture = Path(__file__).parents[1] / "fixtures/qwen3_omni/tiny_checkpoint.py"
    write = runpy.run_path(str(fixture))["write_tiny_checkpoint"]
    return write(tmp_path, num_code_groups=getattr(request, "param", 4))

@pytest.fixture
def prepared_talker(tiny_checkpoint, request):
    path, values = tiny_checkpoint
    model = Qwen3OmniModel.describe(model_path_hf=str(path))
    shell = model.create_submodule("Talker")
    binding = NodeBinding(model=model, submodule=shell, node_name="Talker",
                          device=torch.device("cpu"), dtype=getattr(request, "param", torch.float32),
                          tp_group=None, sp_group=None, model_config={}, resource_specs=())
    prepare = pytest.importorskip(
        "mstar_accelerators.implementations.qwen3_omni.amd.preparation"
    ).prepare_program
    program = prepare(binding, {})
    shell.bind_execution(binding, None)
    return model, shell, program, binding, values


def test_description_reads_configuration_only_and_fresh_shells_do_not_alias(monkeypatch, tiny_checkpoint):
    import huggingface_hub
    import transformers

    def forbidden(*args, **kwargs):
        pytest.fail("description constructed or fetched serving assets")
    monkeypatch.setattr(huggingface_hub, "snapshot_download", forbidden)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", forbidden)
    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", forbidden)
    model = Qwen3OmniModel.describe(model_path_hf=str(tiny_checkpoint[0]))
    first, second = model.create_submodule("Talker"), model.create_submodule("Talker")
    assert first is not second and first.request_states is not second.request_states
    assert list(first.parameters()) == [] and not hasattr(first, "model")
    assert model._submodule_cache == {} and model._tokenizer is None
    assert {"talker_prefill", "talker_last_prefill", "talker_decode"} <= model.get_graph_walk_graphs().keys()
    assert {"Thinker", "Talker"} <= {node for spec in model.get_node_resources() for node in spec.nodes}
    assert all(isinstance(spec.config, SequenceKVConfig) for spec in model.get_node_resources()
               if spec.resource_key in ("thinker_kv", "talker_kv"))
    assert not list(model.create_submodule("Thinker").parameters())
    with pytest.raises(FileNotFoundError, match="local configuration"):
        Qwen3OmniModel.describe(model_path_hf="not-a-local-model")


def test_logical_binding_reads_only_conditioning_rows_once(monkeypatch, tiny_checkpoint):
    from mstar.model.loader import iterators

    model = Qwen3OmniModel.describe(model_path_hf=str(tiny_checkpoint[0]))
    calls = []
    original = iterators.iter_safetensors_shards

    def conditioning_only(*args, **kwargs):
        assert kwargs.get("keys") == {"thinker.model.embed_tokens.weight"}
        assert "prefix" not in kwargs
        calls.append(kwargs)
        yield from original(*args, **kwargs)

    monkeypatch.setattr(iterators, "iter_safetensors_shards", conditioning_only)
    monkeypatch.setattr(model, "logical_weights",
                        lambda: pytest.fail("logical binding loaded physical Talker weights"))
    shell = model.create_submodule("Talker")
    assert calls == []
    binding = NodeBinding(model=model, submodule=shell, node_name="Talker",
                          device=torch.device("cpu"), dtype=torch.bfloat16,
                          tp_group=None, sp_group=None, model_config={}, resource_specs=())
    shell.bind_execution(binding, None)
    shell.bind_execution(binding, None)
    assert len(calls) == 1
    assert shell._tts_raw_embeddings.dtype == torch.float32
    assert torch.equal(shell._tts_raw_embeddings,
                       tiny_checkpoint[1]["thinker.model.embed_tokens.weight"][1:4])
    assert shell.prepare_inputs("talker_last_prefill", _info(10), {
        "thinker_states": [torch.zeros(1, 48, dtype=torch.bfloat16)],
    }).input_seq_len == 6


def test_external_preparation_owns_the_numerical_constructor(monkeypatch, tiny_checkpoint):
    from mstar.model.qwen3_omni import reference

    prepare = pytest.importorskip(
        "mstar_accelerators.implementations.qwen3_omni.amd.preparation"
    ).prepare_program
    model = Qwen3OmniModel.describe(model_path_hf=str(tiny_checkpoint[0]))
    shell = model.create_submodule("Talker")

    def forbidden(*args, **kwargs):
        pytest.fail("external program borrowed the model's numerical implementation")

    monkeypatch.setattr(reference, "build_talker_program", forbidden)
    monkeypatch.setattr(model, "_thinker_remap", forbidden)
    binding = NodeBinding(model=model, submodule=shell, node_name="Talker",
                          device=torch.device("cpu"), dtype=torch.float32,
                          tp_group=None, sp_group=None, model_config={}, resource_specs=())
    program = prepare(binding, {})
    assert type(program.model).__module__.startswith("mstar_accelerators.implementations.")
    assert type(program.code_predictor).__module__.startswith("mstar_accelerators.implementations.")
    assert program.config.talker_text.moe_backend == "triton"
    assert model.config.talker_text.moe_backend == "auto"
    assert not hasattr(shell, "model") and list(shell.parameters()) == []


def test_published_checkpoint_namespace_matches_every_logical_talker_weight():
    fixture = Path(__file__).parents[1] / "fixtures" / "qwen3_omni"
    meta = json.loads((fixture / "checkpoint_metadata.json").read_text())
    assert hashlib.sha256((fixture / "config.json").read_bytes()).hexdigest() == meta["config_sha256"]
    config = Qwen3OmniModelConfig.from_pretrained(fixture)
    names = sorted(spec.name for spec in talker_weight_specs(config) if ".conditioning." not in spec.name)
    assert len(names) == meta["talker_tensor_count"] == 8037
    assert hashlib.sha256(("\n".join(names) + "\n").encode()).hexdigest() == meta["sorted_newline_talker_names_sha256"]
    shapes = {spec.name: list(spec.shape) for spec in talker_weight_specs(config) if ".conditioning." not in spec.name}
    shape_digest = hashlib.sha256(json.dumps(shapes, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert shape_digest == meta["sorted_talker_shapes_json_sha256"]


@pytest.mark.parametrize("name", ["qwen3omni", "qwen3omni_2gpu", "qwen3omni_2gpu_large_kv",
                                  "qwen3omni_colocated", "qwen3omni_full_tp2",
                                  "qwen3omni_pd_disaggregated", "qwen3omni_thinker_tp2", "qwen3omni_thinker_ep2"])
def test_native_deployment_byte_budgets_keep_the_original_page_capacity(name):
    import yaml

    from mstar.engine.resources.sequence import HeadSharded, PagedSequenceView, resolve_sequence_config
    from mstar.engine.resources.spec import apply_yaml_overrides

    root = Path(__file__).parents[2]
    model = Qwen3OmniModel.describe(model_path_hf=str(root / "test" / "fixtures" / "qwen3_omni"))
    deployment = yaml.safe_load((root / "configs" / (name + ".yaml")).read_text())
    specs = model.get_node_resources()
    apply_yaml_overrides(specs, deployment)
    for spec in specs:
        if not isinstance(getattr(spec, "config", None), SequenceKVConfig):
            continue
        tp = max(g.get("tp_size", 1) for g in deployment["node_groups"] if set(g["node_names"]) & spec.nodes)
        resolved = resolve_sequence_config(spec, [PagedSequenceView(partition=HeadSharded())],
                                           dtype=torch.bfloat16, group_size=tp)
        pages = 256 if name == "qwen3omni_colocated" else 1024 if (
            name == "qwen3omni_2gpu" and spec.resource_key == "thinker_kv") else 2048
        assert resolved.max_num_pages == pages
        assert resolved.num_kv_heads == max(1, spec.config.num_kv_heads // tp)


def test_bf16_preparation_keeps_the_fp32_codec_accumulator_boundary(tiny_checkpoint):
    model = Qwen3OmniModel.describe(model_path_hf=str(tiny_checkpoint[0]))
    shell = model.create_submodule("Talker")
    binding = NodeBinding(model=model, submodule=shell, node_name="Talker",
                          device=torch.device("cpu"), dtype=torch.bfloat16,
                          tp_group=None, sp_group=None, model_config={}, resource_specs=())
    prepare = pytest.importorskip(
        "mstar_accelerators.implementations.qwen3_omni.amd.preparation"
    ).prepare_program
    program = prepare(binding, {})
    shell.bind_execution(binding, None)
    assert list(shell.parameters()) == []
    assert all(p.dtype == torch.bfloat16 for p in program.parameters())
    inp = shell.prepare_inputs("talker_decode", _info(10), {
        "thinker_states": [torch.randn(1, 48, dtype=torch.bfloat16)],
        "talker_input_embeds": [torch.randn(1, 32)],
    })
    prepared = program.preprocess("talker_decode", ModelInputsFromEngine([10], {}), [inp])
    assert inp.input_embeds.dtype == torch.float32
    assert prepared["input_embeds"].dtype == torch.bfloat16
    assert prepared["codec_emb_sum"].dtype == torch.float32


@pytest.mark.parametrize("defect", ["missing", "shape"])
def test_incomplete_or_wrong_checkpoint_fails_before_physical_preparation(tiny_checkpoint, defect):
    path, values = tiny_checkpoint
    key = "talker.code_predictor.model.layers.0.self_attn.k_proj.weight"
    if defect == "missing":
        del values[key]
    else:
        values[key] = values[key][:-1]
    save_file(values, path / "model.safetensors")
    model = Qwen3OmniModel.describe(model_path_hf=str(path))
    with pytest.raises(ValueError, match="missing logical weight|expected shape"):
        model.logical_weights().load()


def test_external_program_packing_is_separate_from_the_logical_shell(prepared_talker):
    model, shell, program, binding, checkpoint = prepared_talker
    weights = model.logical_weights().load()
    for i, name in enumerate(("pad", "bos", "eos"), 1):
        expected = checkpoint["thinker.model.embed_tokens.weight"][i]
        assert torch.equal(weights[f"talker.conditioning.tts_{name}"], expected)
    assert torch.equal(weights["thinker.model.embed_tokens.weight"], checkpoint["thinker.model.embed_tokens.weight"])
    for module, prefix in ((program.model, "talker"), (program.code_predictor, "talker.code_predictor")):
        attn = module.model.layers[0].self_attn
        expected = torch.cat([checkpoint[f"{prefix}.model.layers.0.self_attn.{name}_proj.weight"]
                              for name in ("q", "k", "v")])
        assert torch.equal(attn.qkv_proj.weight, expected)
        assert not any(p.is_meta for p in module.parameters())
    expert = program.model.model.layers[0].mlp.experts
    assert torch.equal(expert.gate_up_proj[1], torch.cat([
        checkpoint[f"talker.model.layers.0.mlp.experts.1.{name}_proj.weight"] for name in ("gate", "up")
    ]))
    assert torch.equal(program.code_predictor.lm_head_weight, torch.stack([
        checkpoint[f"talker.code_predictor.lm_head.{i}.weight"] for i in range(3)
    ]))
    assert shell.request_states == {} and shell._eos_embed_sent == set()
    assert not hasattr(shell, "model") and not hasattr(shell, "install_reference_program")
    assert not hasattr(program, "request_states") and not hasattr(program, "_eos_embed_sent")


def _info(rid):
    return SimpleNamespace(rid_handle=rid, step_metadata={}, max_tokens=100,
                           dynamic_loop_iter_counts={})


def _project(x, projection):
    return F.linear(F.silu(F.linear(x, projection.linear_fc1.weight, projection.linear_fc1.bias)),
                    projection.linear_fc2.weight, projection.linear_fc2.bias)


def test_raw_conditioning_and_eos_are_model_owned_and_inputs_are_not_mutated(prepared_talker, monkeypatch):
    _, shell, program, _, _ = prepared_talker
    codec = torch.randn(1, shell.config.talker_hidden_size)
    original = codec.clone()
    thinker = torch.randn(1, shell.config.thinker_hidden_size)
    with monkeypatch.context() as patch:
        patch.setattr(program.model.text_projection, "forward",
                      lambda *a: pytest.fail("logical preparation projected inputs"))
        text = shell.prepare_inputs("talker_decode", _info(10), {
            "talker_input_embeds": [codec], "thinker_states": [thinker],
        })
        eos = shell.prepare_inputs("talker_decode", _info(10), {"talker_input_embeds": [codec]})
        pad = shell.prepare_inputs("talker_decode", _info(10), {"talker_input_embeds": [codec]})
    assert text.tensor_inputs["thinker_states"] is thinker
    assert torch.equal(eos.tensor_inputs["thinker_states"], shell._tts_raw_embeddings[2:3])
    assert torch.equal(pad.tensor_inputs["thinker_states"], shell._tts_raw_embeddings[0:1])
    engine = ModelInputsFromEngine([10], {})
    for inp, conditioning in ((text, _project(thinker, program.model.text_projection)),
                              (eos, program._tts_eos_embed_cached), (pad, program._tts_pad_embed_cached)):
        first = program.preprocess("talker_decode", engine, [inp])
        second = program.preprocess("talker_decode", engine, [inp])
        one = program._add_text_conditioning(first["input_embeds"], first["text_raw"])
        two = program._add_text_conditioning(second["input_embeds"], second["text_raw"])
        assert torch.equal(one, codec + conditioning) and torch.equal(one, two)
        assert torch.equal(codec, original)
    shell.cleanup_request(10)
    assert 10 not in shell._eos_embed_sent
    restarted = shell.prepare_inputs("talker_decode", _info(10), {"talker_input_embeds": [codec]})
    assert restarted.kwargs["conditioning_kind"] == "eos"


@pytest.mark.parametrize("walk", ["talker_prefill", "talker_last_prefill"])
def test_prefill_keeps_projection_choice_and_assistant_prefix(prepared_talker, walk):
    _, shell, program, _, _ = prepared_talker
    n = 3 if walk == "talker_prefill" else 1
    raw = torch.randn(n, shell.config.thinker_hidden_size)
    mask = torch.tensor([False, True, False])
    inp = shell.prepare_inputs(walk, _info(10), {"thinker_states": [raw], "thinker_mask": [mask]})
    if walk == "talker_prefill":
        expected = torch.empty(n, shell.config.talker_hidden_size)
        expected[~mask] = _project(raw[~mask], program.model.text_projection)
        expected[mask] = _project(raw[mask], program.model.hidden_projection)
    else:
        assert inp.input_seq_len == 6 and inp.input_ids is None
        assert inp.tensor_inputs["codec_ids"].tolist() == [2, 3, 4, 8, 5, 6]
        assert torch.equal(inp.tensor_inputs["thinker_states"][-1:], raw)
        expected = program.talker_code_emb(inp.tensor_inputs["codec_ids"]) + torch.cat([
            program._tts_pad_embed_cached.expand(4, -1), program._tts_bos_embed_cached[None],
            _project(raw, program.model.text_projection),
        ])
    actual = program.preprocess(walk, ModelInputsFromEngine([10], {}), [inp])
    assert torch.equal(actual["input_embeds"], expected)
    if walk == "talker_last_prefill":
        assert actual["last_token_indices"].tolist() == [5]


def test_capture_templates_use_the_logical_inputs_and_reset_predictor_scratch(prepared_talker):
    _, shell, program, _, _ = prepared_talker
    configs = program.get_accelerator_graph_configs(torch.device("cpu"))
    for cfg in configs:
        inp = cfg.make_node_input(13) if cfg.capture_graph_walk == "talker_prefill" else cfg.single_request_inputs
        if cfg.capture_graph_walk == "talker_decode":
            assert inp.input_embeds.dtype == torch.float32
        if cfg.capture_graph_walk == "talker_last_prefill":
            assert torch.equal(inp.tensor_inputs["thinker_states"][:4], shell._tts_raw_embeddings[0:1].expand(4, -1))
            assert torch.equal(inp.tensor_inputs["thinker_states"][4], shell._tts_raw_embeddings[1])
        rows = [inp.clone(), inp.clone()]
        prepared = program.preprocess(cfg.capture_graph_walk, ModelInputsFromEngine([-1, -2], {}), rows)
        assert prepared["input_embeds"].shape[0] == 2 * inp.input_seq_len
        assert rows[0].tensor_inputs["thinker_states"].data_ptr() != rows[1].tensor_inputs["thinker_states"].data_ptr()
        if cfg.capture_graph_walk != "talker_prefill":
            assert not prepared["pos_buf"].any() and not prepared["codec_emb_sum"].any()
            assert prepared["codec_emb_sum"].dtype == torch.float32


def test_result_mapping_uses_request_order_and_prefill_has_no_sampling(prepared_talker):
    from mstar.model.qwen3_omni.results import TalkerResult
    _, shell, program, _, _ = prepared_talker
    result = TalkerResult(torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]), torch.randn(2, 32), torch.tensor([1, 5]))
    invocation = SimpleNamespace(graph_walk="talker_decode", request_ids=(20, 10))
    adapted = shell.execution_result(result, invocation)
    assert adapted.outputs.per_rid_outputs[20]["new_token"][0].item() == 1
    assert adapted.outputs.per_rid_outputs[10]["codec_tokens"][0].tolist() == [5, 6, 7, 8]
    assert adapted.resource_results == {}
    invocation.graph_walk = "talker_prefill"
    adapted = shell.execution_result(None, invocation)
    assert adapted.resource_results == {} and adapted.outputs.per_rid_outputs == {20: {}, 10: {}}

    from mstar.engine.execution import ExecutionResult
    explicit = ExecutionResult(adapted.outputs, resource_results={"future_owner": object()})
    assert shell.execution_result(explicit, invocation) is explicit


def _rotate(x, positions, theta):
    half = x.shape[-1] // 2
    angles = positions.float()[:, None] * theta ** (-torch.arange(half).float()[None, :] / half)
    cos, sin = angles.cos()[:, None], angles.sin()[:, None]
    first, second = x.float().chunk(2, -1)
    return torch.cat([first * cos - second * sin, second * cos + first * sin], -1).to(x.dtype)


def _install_cpu_oracles(monkeypatch):
    from mstar_accelerators.implementations.qwen3_omni.amd import model as talker

    from mstar.engine.resources.position.manager import RopeManager
    from mstar.engine.resources.sampler import utils as sampling

    def qk_norm_rope(x, weight, *, pos, eps, rope_theta):
        normed = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps)
        return _rotate((normed * weight).to(x.dtype), pos, rope_theta)

    def dense_attention(q, k, v, length):
        return F.scaled_dot_product_attention(
            q.transpose(1, 2), k[:, :length].transpose(1, 2), v[:, :length].transpose(1, 2),
            enable_gqa=True,
        ).transpose(1, 2)

    def position(self, q, k, label, rope_theta, **kwargs):
        positions = self._current_pos_ids[label]
        return _rotate(q, positions, rope_theta), _rotate(k, positions, rope_theta)

    def sample(logits, temperature, top_k, top_p, repetition_penalty, seen_token_mask,
               seed, rand_offset, **kwargs):
        # An explicit host oracle, using each real owner's seed/offset/policy.
        tokens = []
        for row, values in enumerate(logits.float()):
            if seen_token_mask is not None:
                penalty = repetition_penalty[row]
                values = torch.where(seen_token_mask[row],
                                     torch.where(values > 0, values / penalty, values * penalty), values)
            if temperature[row] == 0:
                tokens.append(values.argmax())
                continue
            assert top_k[row] == 0 and top_p[row] == 1
            generator = torch.Generator().manual_seed(int(seed[row]) + int(rand_offset[row]) * 104729)
            tokens.append(torch.multinomial((values / temperature[row]).softmax(-1), 1, generator=generator)[0])
        return torch.stack(tokens)

    from mstar_accelerators.implementations.qwen3_omni.amd import program as physical
    from mstar_accelerators.implementations.qwen3_omni.amd import router_consistency

    def cp_qk(qkv, qw, kw, pos, kv, cache_pos, seq_len, eps_q, eps_k, theta, nq, nk, dim):
        q, k, v = qkv.split((nq * dim, nk * dim, nk * dim), dim=-1)
        q.copy_(qk_norm_rope(q.reshape(-1, nq, dim), qw, pos=pos, eps=eps_q, rope_theta=theta).reshape_as(q))
        key = qk_norm_rope(k.reshape(-1, nk, dim), kw, pos=pos, eps=eps_k, rope_theta=theta)
        kv[:, 0, cache_pos:cache_pos + seq_len] = key.reshape(-1, seq_len, nk, dim)
        kv[:, 1, cache_pos:cache_pos + seq_len] = v.reshape(-1, seq_len, nk, dim)

    def embed(tokens, weight, total, codes, group):
        values = F.embedding(tokens.long(), weight)
        total.add_(values)
        codes[:, group] = tokens
        return values

    def rounded_norm(x, delta, weight, eps):
        residual = x + delta
        out = residual.float() * torch.rsqrt(residual.float().square().mean(-1, keepdim=True) + eps)
        return (out * weight).to(x.dtype), residual

    def attention_projection(q, k, v, weight, residual, norm_weight, length, eps):
        attended = dense_attention(q, k, v, length).flatten(-2)
        return rounded_norm(residual, F.linear(attended, weight), norm_weight, eps)

    # Explicit host arithmetic oracles, never production CPU fallbacks.
    monkeypatch.setattr(router_consistency, "router_mm", F.linear)
    monkeypatch.setattr(talker, "cp_qk_norm_rope_kv_write", cp_qk)
    monkeypatch.setattr(talker, "short_decode_attn_nhd", dense_attention)
    monkeypatch.setattr(talker, "rounded_add_norm", rounded_norm)
    monkeypatch.setattr(talker, "attention_projection_norm", attention_projection)
    monkeypatch.setattr(physical, "cp_embed_accumulate", embed)
    monkeypatch.setattr(talker, "fused_qk_norm_rope", qk_norm_rope)
    monkeypatch.setattr(talker, "decode_attn_nhd", dense_attention)
    monkeypatch.setattr(RopeManager, "apply_qk", position)
    monkeypatch.setattr(sampling, "sample_tokens", sample)


def _cpu_resources(shell, *, dtype=torch.float32, joint_comm_group=None):
    from mstar.engine.resources.attn.planned import PlannedAttentionResource
    from mstar.engine.resources.kv.config import PagedKVConfig
    from mstar.engine.resources.kv.manager import KVManager
    from mstar.engine.resources.position.config import PositionConfig
    from mstar.engine.resources.position.manager import RopeManager
    from mstar.engine.resources.sampler.resource import SamplerResource

    cfg = shell.config.talker_text
    kv_config = PagedKVConfig(num_layers=cfg.num_hidden_layers, num_kv_heads=cfg.num_key_value_heads,
                             num_qo_heads=cfg.num_attention_heads, head_dim=cfg.head_dim,
                             max_seq_len=64, page_size=4, max_num_pages=64, prefix_cache=False)

    def attention(q, plan, cache):
        results = []
        for row, view in enumerate(plan.contexts):
            begin, end = int(plan.qo_indptr[row]), int(plan.qo_indptr[row + 1])
            positions = torch.arange(view.start, view.start + view.length)
            pages = torch.tensor(view.page_idxs)[positions // plan.page_size]
            k, v = cache[pages, 0, positions % plan.page_size], cache[pages, 1, positions % plan.page_size]
            q_positions = view.length - (end - begin) + torch.arange(end - begin)
            mask = torch.arange(view.length)[None, :] <= q_positions[:, None]
            results.append(F.scaled_dot_product_attention(
                q[begin:end].transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1),
                attn_mask=mask, enable_gqa=True,
            ).transpose(0, 1))
        return torch.cat(results)

    from mstar.communication.tensors import LocalTransferEngine
    from mstar.engine.resources.kv.transfer import TransferEngineInfo

    cpu = torch.device("cpu")
    transfer = TransferEngineInfo("worker", "test", LocalTransferEngine("localhost"))
    return {
        TALKER_KV: KVManager(kv_config, TALKER_KV, joint_comm_group, transfer, cpu, dtype, host_plan=True),
        TALKER_ATTN: PlannedAttentionResource(TALKER_KV, 4, attention),
        TALKER_POS: RopeManager(PositionConfig(TALKER_KV), cpu, host_plan=True),
        TALKER_SAMPLER: SamplerResource(cfg.vocab_size, True, cpu),
        CODE_PRED_SAMPLER: SamplerResource(shell.cp_cfg.vocab_size, False, cpu),
    }


@pytest.mark.parametrize("batch_size", [1, 2, 8])
@pytest.mark.parametrize("tiny_checkpoint", [4, 16], indirect=True)
@pytest.mark.parametrize("prepared_talker", [torch.float32, torch.bfloat16], indirect=True)
@torch.no_grad()
def test_complete_talker_cpu_lifecycle_with_real_resource_owners(prepared_talker, monkeypatch, batch_size):
    from mstar.engine.resources.runner import StepRunner
    from mstar.engine.resources.sampler.config import SamplingReqConfig
    from mstar.engine.resources.step import StepContext

    _, shell, program, _, _ = prepared_talker
    _install_cpu_oracles(monkeypatch)
    resources = _cpu_resources(shell, dtype=next(program.parameters()).dtype)
    shell.bind_node_resources(resources)
    program.bind_resources(resources)
    runner = StepRunner(resources)
    rids = list(range(10, 10 + batch_size))
    rng = torch.Generator().manual_seed(101)
    for rid in rids:
        policies = {
            TALKER_SAMPLER: SamplingReqConfig(temperature=0.0 if rid % 2 else 0.8,
                                              repetition_penalty=1.1),
            CODE_PRED_SAMPLER: SamplingReqConfig(temperature=0.9),
        }
        for name, policy in policies.items():
            policy.apply_conductor_config(seed=rid, resource_key=name)
        runner.ingest_request(rid, overrides=policies)

    def run(walk, active, raw):
        infos = {rid: _info(rid) for rid in active}
        inputs = [shell.prepare_inputs(walk, infos[rid], raw[rid]) for rid in active]
        step = shell.declare_step(walk, active, inputs)
        ctx = StepContext(request_ids=tuple(active), graph_walk=walk, slot=0, capture=False)
        step.set_ctx(ctx)
        assert runner.admit(step).ok
        runner.plan(step)
        engine_inputs = ModelInputsFromEngine(active, infos, resources=resources)
        materialized = program.preprocess(walk, engine_inputs, inputs)
        physical = program.forward_batched(walk, engine_inputs, **materialized)
        from mstar.engine.execution import LogicalInvocation
        from mstar.model.qwen3_omni.results import TalkerResult
        logical = None if walk == "talker_prefill" else TalkerResult(**physical.row_outputs)
        invocation = LogicalInvocation(walk, active, inputs, engine_inputs)
        result = shell.execution_result(logical, invocation).outputs.per_rid_outputs
        runner.commit(step)
        return result

    prefill = {rid: {"thinker_states": [torch.randn(2 + rid % 2, 48, generator=rng)],
                     "thinker_mask": [torch.zeros(2 + rid % 2, dtype=torch.bool)]} for rid in rids}
    run("talker_prefill", rids, prefill)
    assert all(resources[TALKER_SAMPLER]._sampler._step_offset[rid] == 0 for rid in rids)
    first = run("talker_last_prefill", rids, {
        rid: {"thinker_states": [torch.randn(1, 48, generator=rng)]} for rid in rids
    })
    retained = {rid: first[rid]["codec_tokens"][0].clone() for rid in rids}
    previous = first
    for iteration in range(3):
        active = rids[::-1] if iteration % 2 else rids
        prior_embeds = {rid: previous[rid]["talker_input_embeds"][0].clone() for rid in active}
        raw = {rid: {"talker_input_embeds": previous[rid]["talker_input_embeds"]} for rid in active}
        if iteration == 0:
            for row in raw.values():
                row["thinker_states"] = [torch.randn(1, 48, generator=rng)]
        program._get_cp_kv_cache().fill_(float("nan"))
        result = run("talker_decode", active, raw)
        for rid in rids:
            assert torch.isfinite(result[rid]["talker_input_embeds"][0]).all()
            assert torch.equal(first[rid]["codec_tokens"][0], retained[rid])
            assert torch.equal(previous[rid]["talker_input_embeds"][0], prior_embeds[rid])
            assert resources[TALKER_SAMPLER]._sampler._step_offset[rid] == iteration + 2
            assert resources[CODE_PRED_SAMPLER]._sampler._step_offset[rid] == (iteration + 2) * (shell.num_codes - 1)
            assert resources[TALKER_POS]._counters[rid]["main"] == 2 + rid % 2 + 6 + iteration + 1
        assert shell._eos_embed_sent == (set() if iteration == 0 else set(rids))
        previous = result
    # Cancellation and handle reuse release every authoritative owner.
    for rid in rids:
        runner.remove_request(rid)
        shell.cleanup_request(rid)
    assert shell._eos_embed_sent == set()
    assert resources[TALKER_SAMPLER]._sampler._step_offset == {}
    assert resources[CODE_PRED_SAMPLER]._sampler._step_offset == {}
    resources[TALKER_KV].assert_pages_conserved()
    assert resources[TALKER_KV]._arena.allocator.num_free == 63
    resources[TALKER_KV].cleanup()


@pytest.mark.parametrize("tiny_checkpoint", [16], indirect=True)
@torch.no_grad()
def test_greedy_talker_matches_independent_logical_weight_oracle(prepared_talker, monkeypatch):
    from mstar_accelerators.testing.models.qwen3_omni_oracle import TalkerOracle

    from mstar.engine.resources.runner import StepRunner
    from mstar.engine.resources.sampler.config import SamplingReqConfig
    from mstar.engine.resources.step import StepContext

    model, shell, program, _, _ = prepared_talker
    _install_cpu_oracles(monkeypatch)
    oracle = TalkerOracle(model)
    resources = _cpu_resources(shell)
    program.bind_resources(resources)
    runner = StepRunner(resources)
    rids = [71, 19]
    rng = torch.Generator().manual_seed(947)
    for rid in rids:
        policies = {
            TALKER_SAMPLER: SamplingReqConfig(temperature=0.0, repetition_penalty=1.1),
            CODE_PRED_SAMPLER: SamplingReqConfig(temperature=0.0),
        }
        for name, policy in policies.items():
            policy.apply_conductor_config(seed=rid, resource_key=name)
        runner.ingest_request(rid, overrides=policies)

    previous = {}
    walks = ["talker_prefill", "talker_last_prefill", "talker_decode", "talker_decode", "talker_decode"]
    for iteration, walk in enumerate(walks):
        active = rids[::-1] if iteration % 2 else rids
        infos = {rid: _info(rid) for rid in active}
        raw = {}
        for row, rid in enumerate(active):
            if walk == "talker_prefill":
                raw[rid] = {"thinker_states": [torch.randn(2 + row, 48, generator=rng)],
                            "thinker_mask": [torch.tensor([False, True] + ([False] if row else []))]}
            elif walk == "talker_last_prefill":
                raw[rid] = {"thinker_states": [torch.randn(1, 48, generator=rng)]}
            else:
                raw[rid] = {"talker_input_embeds": [previous[rid]]}
                if iteration == 2:
                    raw[rid]["thinker_states"] = [torch.randn(1, 48, generator=rng)]
        inputs = [shell.prepare_inputs(walk, infos[rid], raw[rid]) for rid in active]
        expected = [oracle.step(walk, rid, inp, repetition_penalty=1.1)
                    for rid, inp in zip(active, inputs, strict=True)]
        step = shell.declare_step(walk, active, inputs)
        step.set_ctx(StepContext(request_ids=tuple(active), graph_walk=walk, slot=0, capture=False))
        assert runner.admit(step).ok
        runner.plan(step)
        engine_inputs = ModelInputsFromEngine(active, infos, resources=resources)
        prepared = program.preprocess(walk, engine_inputs, inputs)
        actual = program.forward_batched(walk, engine_inputs, **prepared)
        runner.commit(step)
        for row, rid in enumerate(active):
            ref = expected[row]
            assert resources[TALKER_POS]._counters[rid]["main"] == ref.position
            assert resources[TALKER_SAMPLER]._sampler._step_offset[rid] == ref.sampler_offsets[0]
            assert resources[CODE_PRED_SAMPLER]._sampler._step_offset[rid] == ref.sampler_offsets[1]
            manager = resources[TALKER_KV]
            stream = manager._streams[rid]["main"]
            positions = torch.arange(ref.kv_length)
            pages = torch.tensor(stream.page_indices)[positions // 4]
            for layer, (key, value) in enumerate(oracle.kv(rid)):
                cache = manager.layer_view(layer)
                torch.testing.assert_close(cache[pages, 0, positions % 4], key, atol=2e-6, rtol=2e-5)
                torch.testing.assert_close(cache[pages, 1, positions % 4], value, atol=2e-6, rtol=2e-5)
            if walk == "talker_prefill":
                assert ref.result is None
                continue
            assert torch.equal(actual.row_outputs["codec_tokens"][row], ref.result.codec_tokens[0])
            assert torch.equal(actual.row_outputs["tokens"][row], ref.result.tokens[0])
            torch.testing.assert_close(actual.row_outputs["codec_embeds"][row],
                                       ref.result.codec_embeds[0], atol=2e-6, rtol=2e-5)
            previous[rid] = actual.row_outputs["codec_embeds"][row:row + 1].clone()
            seen = resources[TALKER_SAMPLER]._sampler.get_token_mask(rid)._seen_token_mask
            assert set(seen.nonzero().flatten().tolist()) == set(ref.seen_tokens)

    for rid in rids:
        runner.remove_request(rid)
        shell.cleanup_request(rid)
        oracle.reset(rid)
    assert oracle.requests == {}
    resources[TALKER_KV].assert_pages_conserved()
    resources[TALKER_KV].cleanup()


def test_talker_stop_rows_preserve_request_mapping_and_stop_decisions(tiny_checkpoint):
    from mstar.model.qwen3_omni.results import TalkerResult
    from mstar.worker.worker import Worker

    shell = Qwen3OmniModel.describe(model_path_hf=str(tiny_checkpoint[0])).create_submodule("Talker")
    rids = (71, 19, 42)
    eos = shell.config.talker.codec_eos_token_id
    tokens = torch.tensor([eos, 0, 0])
    result = TalkerResult(torch.zeros(3, shell.num_codes, dtype=torch.long),
                         torch.zeros(3, shell.cp_cfg.hidden_size), tokens)
    invocation = SimpleNamespace(graph_walk="talker_decode", request_ids=rids)
    output = shell.execution_result(result, invocation).outputs
    assert output.row_request_ids == rids
    assert output.check_stop_buffers == {"layer0_codes": tokens}
    host = Worker._rows_to_per_rid(output.check_stop_buffers, list(rids))
    for i, rid in enumerate(rids):
        info = SimpleNamespace(step_metadata={}, max_tokens=4,
                               dynamic_loop_iter_counts={"talker_decode_loop": 3 if i == 2 else 0})
        legacy = dict(output.per_rid_outputs[rid])
        shell.postprocess(rid, info, legacy)
        assert torch.equal(legacy["layer0_codes"][0], host[rid]["layer0_codes"][0].squeeze(0))
        expected = {"talker_decode_loop"} if i in (0, 2) else set()
        assert shell.check_stop(rid, info, legacy) == expected
        assert shell.check_stop(rid, info, host[rid]) == expected
