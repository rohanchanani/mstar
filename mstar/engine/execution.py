"""Bound execution choices at the node/scheduler boundary.

This module is independent of any acceleration declaration language.  A binder
supplies executable options; the scheduler asks about a proposed batch, and the
engine refines that query with logical inputs before acquiring one option.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from itertools import groupby
import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mstar.model.submodule_base import BatchedModelOutput, ModelInputsFromEngine, NodeInputs


class Support(Enum):
    NO = 0
    YES = 1
    UNKNOWN = 2


@dataclass(frozen=True)
class InvocationFacts:
    graph_walk: str | None
    batch_size: int | None
    # None means inputs have not been interpreted yet.  It never means zero.
    token_lengths: tuple[int, ...] | None = None
    # Canonical request policy, keyed by its resource owner.  Missing facts
    # remain unknown; the selector does not normalize resource configuration.
    resource_facts: Mapping[str, tuple[Mapping[str, Any], ...]] = field(default_factory=dict)
    # Present for a complete proposal, in its request order. A combined queue
    # name alone does not describe which walks the invocation actually runs.
    row_walks: tuple[str, ...] | None = None

    @property
    def total_tokens(self) -> int | None:
        return None if self.token_lengths is None else sum(self.token_lengths)


@dataclass(frozen=True)
class LogicalInvocation:
    graph_walk: str
    request_ids: Sequence[int]
    inputs: Sequence["NodeInputs"]
    engine_inputs: "ModelInputsFromEngine"
    plan_results: Mapping[str, Any] = field(default_factory=dict)
    row_walks: tuple[str, ...] | None = None


@dataclass(frozen=True)
class Execution:
    """The worker's current stream and one implementation's leased workspace.

    A primitive joins any extra work to this stream before returning.  Returning
    does not mean device completion; M* records and consumes its ordinary event.
    """

    stream: Any
    workspace: Any


@dataclass(frozen=True)
class ExecutionResult:
    outputs: "BatchedModelOutput"
    resource_results: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExecutionOption:
    name: str
    supports: Callable[[InvocationFacts], Support]
    acquire: Callable[[InvocationFacts], Any | None]
    materialize: Callable[[LogicalInvocation, Execution], Any]
    primitive: Callable[[Any, Execution], Any]
    # Comparable costs within this bound node.  Missing costs do not confer
    # priority, and registration order is deliberately not a selection policy.
    cost: Callable[[InvocationFacts], float] | None = None
    close: Callable[[], None] | None = None
    # Optional runtime consumer. Its callbacks above remain the executable
    # interface; this exposes preparation/geometry to the engine before plan.
    runner: Any = None


@dataclass
class SelectedExecution:
    option: ExecutionOption
    workspace: Any
    _closed: bool = field(default=False, init=False, repr=False)

    def run(self, invocation: LogicalInvocation, stream: Any) -> Any:
        if self._closed:
            raise RuntimeError("selected execution has already been released")
        execution = Execution(stream=stream, workspace=self.workspace)
        call = self.option.materialize(invocation, execution)
        return self.option.primitive(call, execution)

    def close(self) -> None:
        # The backend lease transfers ownership to submitted work.  Closing a
        # transferred lease cannot free storage still used by that work.
        if not self._closed:
            self.workspace.close()
            self._closed = True


class UnsupportedInvocation(ValueError):
    pass


class AmbiguousExecution(ValueError):
    pass


@dataclass(frozen=True)
class ExecutionCapabilities:
    options: tuple[ExecutionOption, ...]
    # Search bounds derived by the binder.  They bound scheduler exploration;
    # they do not assert that every smaller batch is supported.
    batch_limits: Mapping[str, int]
    # Proven independent of invocation policy/length/walk inside this domain.
    fixed_option: ExecutionOption | None = None
    fixed_admission: int = 0

    @property
    def candidate_window(self) -> int:
        # Look past one executable batch to find supported nonprefix subsets,
        # without repeatedly interpreting an unbounded ready queue.
        return max(16, 2 * max(self.batch_limits.values(), default=1))

    def possibilities(self, facts: InvocationFacts) -> tuple[ExecutionOption, ...]:
        return tuple(option for option in self.options if option.supports(facts) is not Support.NO)

    def reserve(self, facts: InvocationFacts) -> SelectedExecution | None:
        """Select before effects; None means supported but temporarily busy."""
        supported = []
        costs = {}
        unresolved = []
        for option in self.options:
            support = option.supports(facts)
            if support is Support.YES:
                supported.append(option)
                if option.cost is not None:
                    cost = float(option.cost(facts))
                    if not math.isfinite(cost):
                        raise ValueError(f"{option.name}: execution cost must be finite")
                    costs[option.name] = cost
            elif support is Support.UNKNOWN:
                unresolved.append(option.name)
        if unresolved:
            raise UnsupportedInvocation(f"unresolved invocation facts for {unresolved}")
        if not supported:
            raise UnsupportedInvocation(f"no execution supports {facts}")
        if len(supported) > 1:
            if any(option.cost is None for option in supported):
                raise AmbiguousExecution(f"overlapping executions need comparable costs: {[o.name for o in supported]}")
            scored = sorted(((costs[option.name], option) for option in supported), key=lambda item: item[0])
            tiers = ([option for _, option in items] for _, items in groupby(scored, key=lambda item: item[0]))
        else:
            tiers = [supported]
        for tier in tiers:
            acquired = []
            try:
                for option in tier:
                    workspace = option.acquire(facts)
                    if workspace is not None:
                        acquired.append(SelectedExecution(option, workspace))
                if len(acquired) > 1:
                    raise AmbiguousExecution(f"equal execution costs for {facts}")
            except BaseException:
                for selected in acquired:
                    selected.close()
                raise
            if acquired:
                return acquired[0]
        return None

    def close(self) -> None:
        for option in self.options:
            if option.close is not None:
                option.close()
