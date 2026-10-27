"""Batch composition: which ready rows of one (node, walk) run this step.

The ``MicroScheduler`` picks the (node, walk) and owns the mechanics (popping
ready queues, the backlog, holds, TP follow); a ``BaseBatchBuilder`` decides
which of the popped rows go into the step, which stay parked in the backlog
and which go back to their ready queues.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from itertools import combinations, islice
from typing import TYPE_CHECKING, Any, NamedTuple

from mstar.engine.execution import ExecutionCapabilities, InvocationFacts

if TYPE_CHECKING:
    from mstar.worker.micro_scheduler import ScheduledBatch


@dataclass
class BatchBuildRequest:
    node_name: str
    graph_walk: str
    # Both popped: `backlog` out of the scheduler's backlog, `fresh` off the
    # ready queues. Whatever the builder neither schedules nor returns must come
    # back as `BatchBuildResult.backlog`.
    backlog: "ScheduledBatch | None" = None
    fresh: "ScheduledBatch | None" = None
    # Rows of `backlog` that cannot run this step; they stay parked.
    blocked_rids: set[int] = field(default_factory=set)
    # The rid whose capture group the step must share, else its first row's.
    capture_group_of: int | None = None
    # Each real walk's cap (None or absent: uncapped); under `graph_walk`, the
    # cap of a step mixing walks.
    walk_caps: dict[str, int | None] = field(default_factory=dict)
    # Under a combined walk: how many rows of each real walk a step mixing
    # walks may hold (absent is uncapped). A step of one walk is not bound.
    walk_row_caps: dict[str, int] = field(default_factory=dict)
    # The caller's own rows (a speculation merge's continuing ones) and their walk.
    pre_existing_batch_size: int = 0
    pre_existing_walk: str | None = None
    pre_existing_rids: tuple[int, ...] = ()
    # `Engine.capture_group` for one rid of this (node, walk).
    capture_group: Callable[[int], Any] = lambda _: None
    execution_capabilities: ExecutionCapabilities | None = None
    resource_facts: Callable[[Sequence[int]], Mapping] = lambda _: {}


class BatchBuildResult(NamedTuple):
    scheduled: "ScheduledBatch | None"
    # Parked under this (node, walk) for a later step.
    backlog: "ScheduledBatch | None" = None
    # Popped rows handed back to their ready queues: rid -> worker graph id.
    returned: dict[int, int] | None = None


def off_group_rids(
    batch: "ScheduledBatch", capture_group: Callable[[int], Any],
    exclude_rids: set[int] = frozenset(), anchor: int | None = None,
) -> set[int]:
    """The rids of ``batch`` outside ``anchor``'s capture group, or outside
    the first eligible rid's when no anchor is given."""
    rids = [
        rid for rid in batch.request_to_worker_graph
        if rid not in exclude_rids
    ]
    if not rids:
        return set()
    # `is not None`: handle 0 is a real rid
    wanted = capture_group(rids[0] if anchor is None else anchor)
    return {rid for rid in rids if capture_group(rid) != wanted}


def _merged(
    backlog: "ScheduledBatch | None", fresh: "ScheduledBatch | None",
) -> "ScheduledBatch | None":
    """``backlog`` with ``fresh`` folded in behind it."""
    if not fresh:
        return backlog
    if backlog is None:
        return fresh
    backlog.merge(fresh)
    return backlog


class CapabilityChoice(NamedTuple):
    request_ids: tuple[int, ...]
    search_exhausted: bool = False


def choose_capability_rows(
    rids: Sequence[int], *, capabilities: ExecutionCapabilities,
    graph_walk: str,
    walk_of: Callable[[int], str], facts_for: Callable[[str, tuple[int, ...]], InvocationFacts],
    pre_existing_batch_size: int = 0, pre_existing_walk: str | None = None,
    possibilities: Callable[[InvocationFacts], Any] | None = None,
) -> CapabilityChoice:
    """Bounded FIFO-priority subset search, shared by proposal and refinement.

    Uniform proposals use their actual walk; mixed proposals keep the combined
    label and row composition. A failed composition does not exclude smaller
    or nonprefix subsets. Exhausting this search leaves rows waiting.
    """
    rows = tuple(islice(rids, capabilities.candidate_window))
    upper = min(len(rows), max(capabilities.batch_limits.values(), default=0) - pre_existing_batch_size)

    def supported(proposed):
        walks = {walk_of(rid) for rid in proposed}
        if pre_existing_walk is not None:
            walks.add(pre_existing_walk)
        walk = next(iter(walks)) if len(walks) == 1 else graph_walk
        limit = capabilities.batch_limits.get(walk)
        if limit is None:
            # The condition still decides support for a mixed invocation. A
            # per-walk bound limits search without inventing a new walk limit.
            limit = max((capabilities.batch_limits.get(w, 0) for w in walks), default=0)
        return (
            len(proposed) + pre_existing_batch_size <= limit
            and bool((possibilities or capabilities.possibilities)(facts_for(walk, proposed)))
        )

    budget = 256
    for size in range(upper, 1, -1):
        for proposed in islice(combinations(rows, size), budget):
            budget -= 1
            if supported(proposed):
                return CapabilityChoice(proposed)
        if budget == 0:
            break
    # Keep ordinary singleton progress even if the combination budget ran out.
    if upper >= 1:
        for rid in rows:
            if supported((rid,)):
                return CapabilityChoice((rid,))
    return CapabilityChoice((), budget == 0 or len(rids) > len(rows))


class BaseBatchBuilder(ABC):
    @abstractmethod
    def build_batch(self, request: BatchBuildRequest) -> BatchBuildResult:
        """One step's rows out of ``request``'s backlog and entries."""

    def room_for_continuing(
        self, cap: int | None, backlog: "ScheduledBatch | None",
    ) -> int | None:
        """How many of a speculation chain's own rids fit in a step capped at
        ``cap`` when ``backlog`` waits under the same (node, walk).

        The chain only ever continues its own rids, so at the cap a backlogged
        chunk would never be reached. Giving the backlog first claim costs the
        displaced rids one step — their nodes go ready again when the in-flight
        batch lands. None for an uncapped node: nothing is displaced.
        """
        if cap is None:
            return None
        return max(0, cap - (0 if backlog is None else len(backlog)))

    def chain_must_yield(
        self, backlog: "ScheduledBatch", live_rids: set[int], anchor: int,
        capture_group: Callable[[int], Any],
    ) -> bool:
        """Whether a speculation chain anchored at ``anchor`` has to stop so
        ``backlog``'s live rows can run: the chain's merge (a build with
        ``capture_group_of=anchor``) would never take them."""
        exclude = set(backlog.request_to_worker_graph) - live_rids
        return bool(off_group_rids(
            backlog, capture_group, exclude_rids=exclude, anchor=anchor,
        ))


class FIFOBatchBuilder(BaseBatchBuilder):
    """Backlog first, then fresh rows, in arrival order, up to the cap; one
    capture group per step.

    Under a combined walk a step of one walk takes that walk's cap and a mixed
    step the combined walk's own, or with none the smallest of the caps of the
    walks present in it; a walk with no row in the step does not cap it. Rows join in
    order; one whose walk would cap the step at or below the rows already in
    it is left out, and is first in line next step.

    Where this differs from the scheduler it replaces, and why. main had two
    paths: with a backlog it drained that alone, never touching the ready
    queues, so the backlog strictly shrank; with none it popped a fresh set,
    capped it and parked the remainder. This builder always does both, which
    is better -- a backlog one row short of the cap gets topped up instead of
    running short -- but it means a step that drains a backlog also pops fresh
    rows, and parking their overflow would refill the backlog from the queues
    on every pass. It would then never drain, and since a key holding a
    backlog is served before any other node, that node starves the rest: on
    ``wan22.yaml`` with ``dit`` capped at 1 a fifth request's
    ``text_encoder`` waited 48.7 s for its first step against 0.2 s on main.

    So a fresh row the cap pushed out goes back to its ready queue **only on
    a step that served a backlog**, which is the case main never created. With
    no backlog it parks, exactly as main did, and the remainder keeps its
    arrival order instead of being re-popped and reordered. A row held back
    for any other reason -- OOM backoff, pending remove, a foreign capture
    group, a walk the combined cap left out -- always parks: it cannot run
    this step whoever holds it.
    """

    def build_batch(self, request: BatchBuildRequest) -> BatchBuildResult:
        # both read before `_merged`, which folds `fresh` into `backlog` in place
        had_backlog = request.backlog is not None
        fresh_rids = (
            set(request.fresh.request_to_worker_graph) if request.fresh else set()
        )
        batch = _merged(request.backlog, request.fresh)
        if batch is None:
            return BatchBuildResult(None)

        exclude = request.blocked_rids | off_group_rids(
            batch, request.capture_group,
            exclude_rids=request.blocked_rids, anchor=request.capture_group_of,
        )
        if request.execution_capabilities is not None:
            if request.pre_existing_batch_size:
                # A speculation merge gathers fresh input edges itself. Park
                # interpreted rows for the ordinary path that owns their holds.
                exclude |= batch.prepared_rows.keys()
            def facts_for(walk, rids):
                lengths = (
                    tuple(batch.prepared_rows[rid].inputs.input_seq_len for rid in rids)
                    if not request.pre_existing_batch_size
                    and all(rid in batch.prepared_rows for rid in rids) else None
                )
                return InvocationFacts(
                    walk, len(rids) + request.pre_existing_batch_size, lengths,
                    request.resource_facts((*request.pre_existing_rids, *rids))
                    if len(request.pre_existing_rids) == request.pre_existing_batch_size else {},
                    ((request.pre_existing_walk,) * request.pre_existing_batch_size
                     + tuple(batch.walk_of(rid) for rid in rids))
                    if not request.pre_existing_batch_size or request.pre_existing_walk is not None else None,
                )

            candidates = [rid for rid in batch.request_to_worker_graph if rid not in exclude]
            choice = choose_capability_rows(
                candidates,
                capabilities=request.execution_capabilities,
                graph_walk=request.graph_walk,
                walk_of=batch.walk_of, facts_for=facts_for,
                pre_existing_batch_size=request.pre_existing_batch_size,
                pre_existing_walk=request.pre_existing_walk,
            )
            selected = choice.request_ids
            if not selected:
                exhausted = (
                    choice.search_exhausted
                    or len(candidates) >= request.execution_capabilities.candidate_window
                )
                if candidates and exhausted:
                    # A bounded search failure is not an unsupported request.
                    # Advance its window rather than retrying the same subset.
                    rid = candidates[0]
                    batch.request_to_worker_graph[rid] = batch.request_to_worker_graph.pop(rid)
                    batch._by_walk = None
                return BatchBuildResult(None, backlog=batch)
            scheduled, remainder = batch.split_off_first(
                None, exclude_rids=set(batch.request_to_worker_graph) - set(selected),
            )
            # Interpreted rows must stay with their holds. Fresh overflow may
            # return to the ready queues, preserving the existing fairness rule.
            give_back = {
                rid for rid in remainder.request_to_worker_graph
                if rid in fresh_rids and rid not in exclude and rid not in remainder.prepared_rows
            } if had_backlog and remainder else set()
            if give_back:
                parked, returned = remainder.split_off_first(None, exclude_rids=give_back)
                return BatchBuildResult(scheduled, parked, dict(returned.request_to_worker_graph))
            return BatchBuildResult(scheduled, remainder)
        walk = _single_walk(batch, request)
        if walk is None:
            max_bs, left_out = _compose_walks(batch, exclude, request)
            exclude = exclude | left_out
        else:
            cap = request.walk_caps.get(walk)
            max_bs = None if cap is None else cap - request.pre_existing_batch_size
        if max_bs is not None and max_bs <= 0:
            return BatchBuildResult(None, backlog=batch)  # the caller's rows fill it
        scheduled, remainder = batch.split_off_first(max_bs, exclude_rids=exclude)
        if remainder is None:
            return BatchBuildResult(scheduled)
        give_back = {
            rid for rid in remainder.request_to_worker_graph
            if rid in fresh_rids and rid not in exclude
        } if had_backlog else set()
        if not give_back:
            return BatchBuildResult(scheduled, backlog=remainder)
        parked, returned = remainder.split_off_first(None, exclude_rids=give_back)
        return BatchBuildResult(
            scheduled, backlog=parked,
            returned=dict(returned.request_to_worker_graph) if returned else None,
        )


def _single_walk(batch: "ScheduledBatch", request: BatchBuildRequest) -> str | None:
    """The one walk of every row, the caller's included, else None."""
    if not batch.request_walks:
        return request.graph_walk
    walks = set(batch.request_walks.values())  # rows and request_walks share keys
    if request.pre_existing_walk is not None:
        walks.add(request.pre_existing_walk)
    return walks.pop() if len(walks) == 1 else None


def _min_cap(*caps: int | None) -> int | None:
    return min((cap for cap in caps if cap is not None), default=None)


def _compose_walks(
    batch: "ScheduledBatch", exclude: set[int], request: BatchBuildRequest,
) -> tuple[int | None, set[int]]:
    """Rows left for the step after the caller's, and the rows left out."""
    caps = request.walk_caps
    row_caps = request.walk_row_caps
    n = request.pre_existing_batch_size
    walks = {request.pre_existing_walk} if request.pre_existing_walk else set()
    counts = {request.pre_existing_walk: n} if request.pre_existing_walk else {}
    cap = _min_cap(*(caps.get(walk) for walk in walks))
    left_out = set()
    for rid in batch.request_to_worker_graph:
        if rid in exclude:
            continue
        walk = batch.walk_of(rid)
        new_cap = cap
        if walk not in walks:
            mixed_cap = caps.get(request.graph_walk) if walks else None
            # a mixed step: the combined walk's own cap where it has one,
            # else the smallest of the walks present
            new_cap = mixed_cap if mixed_cap is not None else _min_cap(cap, caps.get(walk))
        if new_cap is not None and new_cap <= n:
            left_out.add(rid)
            continue
        if row_caps and walks:
            if walk not in walks:
                # the row makes the step mix walks: every walk's rows count
                over = any(
                    counts.get(w, 0) + (w == walk) > row_caps[w]
                    for w in (*walks, walk) if w in row_caps
                )
            else:
                # already mixed or not, only this walk's count moves
                cap_w = row_caps.get(walk)
                over = len(walks) > 1 and cap_w is not None and counts[walk] + 1 > cap_w
            if over:
                # the step would mix walks past one's row cap
                left_out.add(rid)
                continue
        walks.add(walk)
        counts[walk] = counts.get(walk, 0) + 1
        cap = new_cap
        n += 1
    return (None if cap is None else cap - request.pre_existing_batch_size), left_out


class BatchBuilderType(Enum):
    FIFO = "fifo"


def make_batch_builder(kind: BatchBuilderType) -> BaseBatchBuilder:
    return {BatchBuilderType.FIFO: FIFOBatchBuilder}[kind]()
