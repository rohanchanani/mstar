"""Legacy physical options stay with their consumer, before worker allocation."""
from types import SimpleNamespace

import pytest
import torch
from test_validate_deployment import _cfg, _Model, _problems

from mstar.distributed.communication import WorkerParallelGroups
from mstar.model.components.moe_parallel import MoeParallelConfig
from mstar.model.submodule_base import NodeSubmodule
from mstar.runtime_extensions import validate_implementation_options
from mstar.worker.engine_manager import EngineManager


@pytest.mark.parametrize("ep", [None, {}, {"parallel": "tp"}, {"parallel": "ep", "mask_padding": True}])
@pytest.mark.parametrize("nodes", [None, ["llm"]])
def test_explicit_legacy_option_on_registry_node_fails_before_loading(ep, nodes):
    extension = {"factory": "not_imported:Runtime"}
    if nodes is not None:
        extension["nodes"] = nodes
    cfg = {"runtime_extension": extension,
           "node_groups": [{"node_names": ["llm"], "ep_config": ep}]}
    with pytest.raises(ValueError, match="upstream physical implementation.*llm"):
        EngineManager().load({"llm"}, torch.device("cpu"), cfg, None, None, None, None)


def test_omitted_legacy_options_do_not_constrain_registry_algorithms():
    validate_implementation_options({
        "runtime_extension": {"factory": "not_imported:Runtime"},
        "node_groups": [{"node_names": ["llm"], "tp_size": 2}],
    })


def test_conductor_deployment_validation_reports_option_scope(tmp_path):
    cfg = _cfg(tmp_path, node_groups=[{
        "node_names": ["llm"], "ranks": [0], "ep_config": {"parallel": "ep"},
    }], runtime_extension={"factory": "not_imported:Runtime"})
    problems = _problems(_Model(), cfg)
    assert any("ep_config configures the upstream physical implementation" in p for p in problems)


class _Shell(NodeSubmodule):
    def prepare_inputs(self, *args, **kwargs):
        raise AssertionError("not an invocation test")

    def forward(self, *args, **kwargs):
        raise AssertionError("not an invocation test")


@pytest.mark.parametrize("use_extension", [False, True])
def test_manager_keeps_upstream_ep_and_registry_construction_separate(monkeypatch, use_extension):
    import mstar.worker.engine_manager as module

    loaded = {}
    monkeypatch.setattr(module, "Engine", lambda **kw: SimpleNamespace(
        load_model=lambda submodules, **kwargs: loaded.update(submodules=submodules, **kwargs),
    ))
    upstream_calls, bindings = [], []
    assembly = SimpleNamespace(resource_requests={})

    def get_submodule(name, device, **kwargs):
        upstream_calls.append((name, kwargs))
        return _Shell()

    def assemble(binding):
        bindings.append(binding)
        return assembly

    model = SimpleNamespace(
        get_node_resources=lambda: [], get_autocast_dtype=lambda: torch.bfloat16,
        get_submodule=get_submodule, create_submodule=lambda name: _Shell(),
        combined_walk_of=lambda: {},
        get_graph_walk_graphs=lambda: {"walk": SimpleNamespace(get_nodes=lambda: ["legacy", "optimized"])},
    )
    ep = {"parallel": "ep", "mask_padding": True, "log_expert_load_every": 7}
    config = {"node_groups": [
        {"node_names": ["legacy"], "ep_config": ep},
        {"node_names": ["optimized"]},
    ]}
    if use_extension:
        config["runtime_extension"] = {"factory": "not_imported:Runtime", "nodes": ["optimized"]}
    manager = EngineManager()
    manager.load(
        {"legacy", "optimized"}, torch.device("cpu"), config,
        WorkerParallelGroups(num_workers=1, global_rank=0), None, model, SimpleNamespace(),
        runtime_extension=SimpleNamespace(assemble=assemble) if use_extension else None,
        node_walks={("optimized", "walk")},
    )
    assert upstream_calls[0][0] == "legacy"
    assert upstream_calls[0][1]["moe_parallel"] == MoeParallelConfig(**ep)
    assert upstream_calls[0][1]["autocast_dtype"] is torch.bfloat16
    if use_extension:
        assert len(upstream_calls) == len(bindings) == 1
        assert bindings[0].node_name == "optimized"
        assert bindings[0].graph_walks == ("walk",)
        assert loaded["assemblies"] == {"optimized": assembly}
        assert loaded["submodules"]["optimized"].binding is bindings[0]
    else:
        assert upstream_calls[1][0] == "optimized"
        assert "moe_parallel" not in upstream_calls[1][1]
        assert not bindings and loaded["assemblies"] == {}
