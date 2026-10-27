"""The authoring frontend and runtime consume one logical model description."""

import json
from types import SimpleNamespace

import pytest
import torch
import yaml

from mstar.engine.resources import KVSpec, SequenceKVConfig, apply_yaml_overrides
from mstar.engine.resources.binding import resolve_bound_specs
from mstar.engine.resources.sequence import DenseSequenceView, PagedSequenceView
from mstar.model.whisper.whisper_model import WhisperModel


@pytest.fixture
def described_model(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "d_model": 8,
                "encoder_layers": 1,
                "decoder_layers": 1,
                "encoder_attention_heads": 2,
                "decoder_attention_heads": 2,
                "encoder_ffn_dim": 16,
                "decoder_ffn_dim": 16,
                "num_mel_bins": 4,
                "max_source_positions": 6,
                "max_target_positions": 8,
                "vocab_size": 32,
            }
        )
    )
    (tmp_path / "generation_config.json").write_text(
        json.dumps(
            {
                "suppress_tokens": [1, 2],
                "begin_suppress_tokens": [3],
            }
        )
    )
    return WhisperModel.describe(model_path_hf=str(tmp_path))


def test_description_needs_only_configuration_and_constructs_fresh_shells(described_model):
    model = described_model
    assert model.nodes == ["audio_encoder", "decoder"]
    assert set(model.node_types()) == set(model.nodes)
    assert model._tokenizer is model._feature_extractor is model._detokenizer_cache is None
    for name in model.nodes:
        first, second = model.create_submodule(name), model.create_submodule(name)
        assert first is not second
        assert type(first) is model.node_types()[name]
        assert list(first.parameters()) == []
        assert first.logical_contract() == second.logical_contract()


def test_description_does_not_resolve_remote_assets():
    with pytest.raises(FileNotFoundError, match="local configuration directory"):
        WhisperModel.describe(model_path_hf="not-a-local-checkpoint")


def test_contract_sources_match_real_logical_input_preparation(described_model):
    encoder = described_model.create_submodule("audio_encoder")
    features = torch.randn(4, 12)
    encoded = encoder.prepare_inputs("prefill", None, {"audio_features": [features]})
    field = encoder.logical_contract().inputs[0]
    assert field.source == "tensor_inputs.audio_features"
    assert encoded.tensor_inputs[field.name] is features
    assert field.shape == tuple(features.shape)

    decoder = described_model.create_submodule("decoder")
    ids, states = torch.tensor([4, 5]), torch.randn(6, 8)
    prefill = decoder.prepare_inputs("prefill", None, {"text_inputs": [ids], "encoder_states": [states]})
    decode = decoder.prepare_inputs("decode", None, {"text_inputs": [ids[:1]]})
    contract = decoder.logical_contract()
    assert contract.inputs[0].source == "input_ids"
    assert prefill.input_ids.data_ptr() == ids.data_ptr()
    assert prefill.tensor_inputs["encoder_states"] is states
    assert "encoder_states" not in decode.tensor_inputs
    assert contract.inputs[1].walks == ("prefill",)
    assert contract.outputs[0].shape == ("batch",)
    assert contract.suppressed_tokens == {"sampler": {"prefill": (1, 2, 3), "decode": (1, 2)}}


def test_deployment_resolves_physical_storage_without_model_page_budgets(described_model):
    specs = described_model.get_node_resources()
    logical = {s.resource_key: s.config for s in specs if isinstance(s, KVSpec)}
    assert all(isinstance(cfg, SequenceKVConfig) for cfg in logical.values())
    assert all(not hasattr(cfg, "max_num_pages") for cfg in logical.values())
    apply_yaml_overrides(
        specs,
        {
            "max_concurrent_requests": 100,
            "resources": {
                "kv_cache": {"max_bytes": 4096, "prefix_cache": False},
                "cross_kv_cache": {"max_bytes": 8192, "prefix_cache": False},
            },
        },
    )
    assert all(s.config is logical[s.resource_key] for s in specs if isinstance(s, KVSpec))
    resolved = resolve_bound_specs(
        specs,
        {
            "kv_cache": (DenseSequenceView(), PagedSequenceView()),
            "cross_kv_cache": (PagedSequenceView(3),),
        },
        dtype=torch.float32,
    )
    physical = {s.resource_key: s.config for s in resolved if isinstance(s, KVSpec)}
    assert physical["kv_cache"].page_size == logical["kv_cache"].max_seq_len
    assert physical["cross_kv_cache"].page_size == 3
    assert physical["kv_cache"].max_seq_len == logical["kv_cache"].max_seq_len
    assert physical["kv_cache"].num_layers == logical["kv_cache"].num_layers


def test_logical_storage_requires_a_deployment_budget(described_model):
    specs = described_model.get_node_resources()
    apply_yaml_overrides(specs, {})
    with pytest.raises(ValueError, match="deployment needs a max_bytes storage budget"):
        resolve_bound_specs(specs, {}, dtype=torch.float32)


@pytest.mark.parametrize("sequence_bound", [4, 9])
def test_storage_override_cannot_redefine_logical_sequence_bound(described_model, sequence_bound):
    with pytest.raises(TypeError, match="max_seq_len"):
        apply_yaml_overrides(
            described_model.get_node_resources(),
            {
                "resources": {
                    "kv_cache": {"max_bytes": 4096, "max_seq_len": sequence_bound},
                    "cross_kv_cache": {"max_bytes": 8192},
                }
            },
        )


def test_bound_deployment_uses_requested_groups_not_reference_capabilities(
    described_model,
    tmp_path,
    monkeypatch,
):
    from mstar.distributed.base import ShardingConfig
    from mstar.validation.placement import check_placement

    monkeypatch.setattr(
        described_model,
        "get_default_sharding_config",
        lambda: ShardingConfig(
            groups=[],
            tp_enabled_nodes={"decoder"},
            shard_dim={"activation": 1, "preserved_shard": 2, "encoder_states": None},
        ),
    )
    config = {
        "node_groups": [
            {"node_names": ["audio_encoder"], "ranks": [0, 1, 2, 3], "tp_size": 2, "sp_size": 2},
            {"node_names": ["decoder"], "ranks": [4]},
        ],
        "runtime_extension": {"factory": "external:Runtime"},
        "sharding_config": {"shard_dim": {"activation": 0}},
    }
    path = tmp_path / "deployment.yaml"
    path.write_text(yaml.safe_dump(config))
    sharding = described_model.get_sharding_config(str(path))
    assert check_placement(described_model, config) == []
    assert sharding.tp_enabled_nodes == {"decoder"}
    assert sharding.sp_enabled_nodes == set()
    assert len(sharding.groups) == 1
    assert sharding.groups[0].tp_size == 4
    assert sharding.groups[0].nodes == {"audio_encoder"}
    assert sharding.shard_dim == {"activation": 0, "preserved_shard": 2, "encoder_states": None}
    del config["runtime_extension"]
    path.write_text(yaml.safe_dump(config))
    problems = check_placement(described_model, config)
    assert any("not in tp_enabled_nodes" in p for p in problems)
    assert any("not in sp_enabled_nodes" in p for p in problems)


@pytest.mark.parametrize("walk,tp_size", [("prefill", 1), ("decode", 2)])
def test_engine_binding_preserves_hosted_walks_and_explicit_disabled_autocast(
    described_model,
    monkeypatch,
    walk,
    tp_size,
):
    from mstar.distributed.communication import CommGroup, WorkerParallelGroups
    from mstar.engine.execution import ExecutionCapabilities
    from mstar.model.binding import NodeAssembly
    from mstar.worker.engine_manager import EngineManager

    # This test checks construction and placement, before executing any program.
    monkeypatch.setattr(described_model, "get_node_resources", lambda: [])
    monkeypatch.setattr(described_model, "get_autocast_dtype", lambda: torch.bfloat16)
    bindings = []

    def assemble(binding):
        bindings.append(binding)
        return NodeAssembly(bind=lambda resources: ExecutionCapabilities((), {}))

    groups = WorkerParallelGroups(num_workers=tp_size, global_rank=0)
    tp_group = CommGroup(0, 0, list(range(tp_size)))
    control_values = []

    def construction_control_gather(value):
        # Construction-only peers agree on metadata; no program or real
        # distributed transport is executed by this placement test.
        control_values.append(value)
        return [value] * tp_size

    if tp_size > 1:
        monkeypatch.setattr(tp_group, "all_gather_object", construction_control_gather)
    groups.add("decoder", tp_group)
    manager = EngineManager()
    manager.load(
        {"decoder"},
        device=torch.device("cpu"),
        model_config={
            "autocast_dtype": None,
            "node_groups": [{"node_names": ["decoder"], "ranks": list(range(tp_size)), "tp_size": tp_size}],
        },
        parallel_groups=groups,
        transfer_engine_info=None,
        model=described_model,
        graph_runtime=None,
        runtime_extension=SimpleNamespace(assemble=assemble),
        node_walks={("decoder", walk)},
    )
    try:
        assert len(bindings) == 1
        assert bindings[0].graph_walks == (walk,)
        assert bindings[0].tp_group.world_size == tp_size
        assert bindings[0].dtype is torch.float32
        assert set(manager.engine._submodules) == {"decoder"}
        assert [value[0] for value in control_values] == (
            ["binding before allocation", "execution completion"] if tp_size > 1 else []
        )
    finally:
        manager.shutdown()


@pytest.mark.parametrize("selected", [None, ["audio_encoder"], ["decoder"]])
def test_mixed_group_retains_native_support_checks(described_model, selected):
    from mstar.validation.placement import check_placement
    extension = {"factory": "external:Runtime"}
    if selected is not None:
        extension["nodes"] = selected
    config = {"runtime_extension": extension, "node_groups": [{
        "node_names": ["audio_encoder", "decoder"], "ranks": [0, 1], "tp_size": 2,
    }]}
    problems = check_placement(described_model, config)
    assert bool(problems) == (selected == ["decoder"])
    if problems:
        assert all("not in tp_enabled_nodes" in p for p in problems)


@pytest.mark.parametrize("mutation,fragment", [
    ({"ranks": [0, 1, 2], "tp_size": 2}, "not divisible"),
    ({"ranks": [0], "tp_size": 0}, "integer >= 1"),
    ({"ranks": [-1]}, "integers >= 0"),
    ({"node_names": ["audio_encoder"]}, "decoder"),
])
def test_registry_selection_keeps_structural_refusals(described_model, mutation, fragment):
    from mstar.validation.placement import check_placement
    config = {"runtime_extension": {"factory": "external:Runtime"}, "node_groups": [{
        "node_names": ["audio_encoder", "decoder"], "ranks": [0], **mutation,
    }]}
    assert any(fragment in p for p in check_placement(described_model, config))


def test_valid_placement_still_requires_a_matching_acceleration(described_model):
    from mstar.distributed.communication import WorkerParallelGroups
    from mstar.worker.engine_manager import EngineManager
    from mstar.validation.placement import check_placement
    from mstar_accelerators.registry import Registry

    config = {"runtime_extension": {"factory": "external:Runtime"}, "node_groups": [{
        "node_names": ["audio_encoder", "decoder"], "ranks": [0],
    }]}
    assert check_placement(described_model, config) == []
    with pytest.raises(ValueError, match="no acceleration binds audio_encoder"):
        EngineManager().load(
            {"audio_encoder"}, torch.device("cpu"), config,
            WorkerParallelGroups(num_workers=1, global_rank=0), None,
            described_model, None, runtime_extension=Registry({}),
            node_walks={("audio_encoder", "prefill")},
        )
