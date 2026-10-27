"""The same logical values are available before physical representation choice."""
import pytest
import torch

from mstar.model.loader.base import StackedParamRule, iter_routed_weights, load_weights_into
from mstar.model.weights import LogicalWeightSource, WeightSpec


def test_logical_values_defaults_and_aliases_precede_partitioning():
    full = torch.arange(24, dtype=torch.float32).reshape(3, 8)
    source = LogicalWeightSource([
        WeightSpec("experts", (3, 8), source="model.experts"),
        WeightSpec("bias", (3,), default=lambda: torch.zeros(3)),
        WeightSpec("tied", (3, 8), alias_of="experts"),
    ], lambda: iter([("model.experts", full), ("unrelated", torch.ones(1))]))
    values = source.load()
    assert values["experts"] is full
    assert values["tied"] is full
    assert torch.equal(values["bias"], torch.zeros(3))
    # Two physical preparations can partition different axes of the same source.
    assert torch.equal(torch.cat(full.chunk(2, dim=1), dim=1), full)
    assert torch.equal(torch.cat(full.split(1, dim=0), dim=0), full)
    assert source.load(["tied"])["tied"] is full


@pytest.mark.parametrize("values,error", [([], "missing"), ([("w", torch.ones(3))], "shape"),
                                          ([("w", torch.ones(2)), ("w", torch.ones(2))], "more than once")])
def test_incomplete_or_ambiguous_checkpoint_is_rejected_before_preparation(values, error):
    source = LogicalWeightSource([WeightSpec("w", (2,))], lambda: iter(values))
    with pytest.raises(ValueError, match=error):
        source.load()


def test_alias_cycle_is_reported():
    source = LogicalWeightSource([WeightSpec("a", (1,), alias_of="b"),
                                  WeightSpec("b", (1,), alias_of="a")], lambda: iter(()))
    with pytest.raises(ValueError, match="cycle"):
        source.load()


def test_routing_is_separate_from_reference_physical_loader():
    q, k = torch.ones(2, 2), torch.full((2, 2), 3.0)
    weights = [("model.q_proj", q), ("model.k_proj", k), ("skip", torch.ones(1))]
    rules = [StackedParamRule("qkv", "q_proj", "q"), StackedParamRule("qkv", "k_proj", "k")]
    remap = lambda name: name.removeprefix("model.")
    skip = lambda name: name == "skip"
    routed = list(iter_routed_weights(weights, rules, remap, skip))
    assert [(r.name, r.component) for r in routed] == [("qkv", "q"), ("qkv", "k")]
    assert routed[0].value is q and routed[1].value is k
    reference = torch.nn.Module()
    reference.qkv = torch.nn.Parameter(torch.zeros(4, 2), requires_grad=False)
    def physical_loader(parameter, tensor, component):
        parameter.data[(0 if component == "q" else 2): (2 if component == "q" else 4)].copy_(tensor)
    reference.qkv.weight_loader = physical_loader
    assert load_weights_into(reference, weights, rules, remap, skip) == {"qkv"}
    assert torch.equal(reference.qkv, torch.cat([q, k]))
