"""Selection owns ambiguity, admission ordering and workspace release."""
from types import SimpleNamespace

import pytest

from mstar.engine.execution import (
    AmbiguousExecution,
    ExecutionCapabilities,
    ExecutionOption,
    InvocationFacts,
    SelectedExecution,
    Support,
    UnsupportedInvocation,
)

FACTS = InvocationFacts("decode", 1, (1,))


def option(name, cost, calls, *, support=Support.YES, busy=False):
    def acquire(facts):
        assert facts is FACTS
        calls.append((name, "acquire"))
        return None if busy else SimpleNamespace(close=lambda: calls.append((name, "close")))
    return ExecutionOption(name, lambda _: support, acquire, lambda *args: None,
                           lambda *args: None, cost=None if cost is None else lambda _: cost)


@pytest.mark.parametrize("cost", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_cost_is_rejected_before_acquisition_even_for_one_option(cost):
    calls = []
    caps = ExecutionCapabilities((option("only", cost, calls),), {"decode": 1})
    with pytest.raises(ValueError, match="finite"):
        caps.reserve(FACTS)
    assert calls == []


@pytest.mark.parametrize("support", [Support.NO, Support.UNKNOWN])
def test_unsupported_or_unresolved_invocations_have_no_effects(support):
    calls = []
    caps = ExecutionCapabilities((option("only", None, calls, support=support),), {"decode": 1})
    with pytest.raises(UnsupportedInvocation):
        caps.reserve(FACTS)
    assert calls == []


def test_busy_best_falls_through_without_acquiring_unused_ties():
    calls = []
    caps = ExecutionCapabilities((option("busy", 0, calls, busy=True),
                                  option("best", 1, calls), option("unused_a", 2, calls),
                                  option("unused_b", 2, calls)), {"decode": 1})
    selected = caps.reserve(FACTS)
    assert selected.option.name == "best"
    selected.close()
    selected.close()
    assert calls == [("busy", "acquire"), ("best", "acquire"), ("best", "close")]


def test_ambiguous_tier_releases_every_acquisition():
    calls = []
    caps = ExecutionCapabilities((option("a", 1, calls), option("b", 1, calls)), {"decode": 1})
    with pytest.raises(AmbiguousExecution):
        caps.reserve(FACTS)
    assert calls == [("a", "acquire"), ("b", "acquire"), ("a", "close"), ("b", "close")]


def test_failed_release_remains_retryable_and_success_prevents_execution():
    attempts = []
    def close():
        attempts.append(None)
        if len(attempts) == 1:
            raise RuntimeError("still in use")
    selected = SelectedExecution(option("only", None, []), SimpleNamespace(close=close))
    with pytest.raises(RuntimeError, match="still in use"):
        selected.close()
    selected.close()
    selected.close()
    assert len(attempts) == 2
    with pytest.raises(RuntimeError, match="already been released"):
        selected.run(None, None)
