"""Agreement for one bound execution on a lockstep worker instance."""

import logging
import math
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import fields, is_dataclass, replace
from enum import Enum
from itertools import groupby

import torch

from mstar.engine.execution import (
    AmbiguousExecution,
    SelectedExecution,
    Support,
    UnsupportedInvocation,
)
from mstar.engine.resources.step import (
    FULL_ADMIT_NOT_READY,
    FULL_ADMIT_OK,
)
from mstar.runtime_extensions import BackendFatalError

logger = logging.getLogger(__name__)


def structural_resource_identity(value):
    """Immutable declared resource geometry and policy, without object addresses."""
    if isinstance(value, Enum):
        return (type(value).__module__, type(value).__qualname__, value.value)
    if is_dataclass(value) and not isinstance(value, type):
        return (type(value).__module__, type(value).__qualname__, tuple(
            (item.name, structural_resource_identity(getattr(value, item.name)))
            for item in fields(value)
        ))
    if isinstance(value, Mapping):
        return tuple(sorted((key, structural_resource_identity(item)) for key, item in value.items()))
    if isinstance(value, (tuple, list)):
        return tuple(structural_resource_identity(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted(structural_resource_identity(item) for item in value))
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, type):
        return (value.__module__, value.__qualname__)
    return value


class InstanceFatalError(BackendFatalError):
    """A lockstep instance cannot resume request-level execution."""

    def __init__(self, node_name, message):
        self.node_name = node_name
        super().__init__(f"{node_name}: {message}")


class ExecutionAgreement:
    """Use the existing control groups before entering an opaque program.

    Acquire and admission are reversible local reservations. Planning,
    materialization and computation may submit or communicate; failures after
    that boundary are instance-fatal rather than local selection retries.
    """

    def __init__(self, group, node_name):
        self.group = group
        self.node_name = node_name

    def _gather(self, value):
        try:
            return self.group.all_gather_object(value)
        except Exception as error:
            raise InstanceFatalError(self.node_name, f"control agreement failed: {error}") from error

    def agree(self, value, stage):
        values = self._gather((stage, value))
        if any(peer != values[0] for peer in values[1:]):
            raise InstanceFatalError(self.node_name, f"participants disagree at {stage}: {values!r}")

    def check(self, error, stage):
        errors = self._gather((stage, None if error is None else f"{type(error).__name__}: {error}"))
        if any(peer_stage != stage or error is not None for peer_stage, error in errors):
            raise InstanceFatalError(self.node_name, f"participant failed at {stage}: {errors}")

    def all(self, value: bool, stage: str) -> bool:
        """A conservative common decision before a conditional collective."""
        values = self._gather((stage, bool(value)))
        if any(peer_stage != stage for peer_stage, _ in values):
            raise InstanceFatalError(self.node_name, f"participants disagree at {stage}: {values!r}")
        return all(decision for _, decision in values)

    def refinement_capabilities(self, capabilities):
        limits = self._gather(dict(capabilities.batch_limits))
        walks = set.intersection(*(set(limit) for limit in limits))
        return replace(capabilities, batch_limits={walk: min(limit[walk] for limit in limits) for walk in walks})

    def possibilities(self, capabilities, facts):
        error = None
        local = ()
        try:
            local = tuple(option.name for option in capabilities.possibilities(facts))
        except Exception as caught:
            error = caught
        self.check(error, "proposal support")
        return set.intersection(*(set(names) for names in self._gather(local)))

    def reserve(self, capabilities, facts, identity):
        """Choose a common supported option and acquire all local workspaces.

        Costs may differ by role; their sum determines the instance cost.
        Refusal on any rank releases every successful peer reservation before
        the next common candidate is considered. Equal available costs remain
        an ambiguity, as on one rank.
        """
        self.agree((identity, facts), "invocation")
        offers = {}
        failure = None
        try:
            for option in capabilities.options:
                support = option.supports(facts)
                cost = None if support is not Support.YES or option.cost is None else float(option.cost(facts))
                if cost is not None and not math.isfinite(cost):
                    raise ValueError(f"{option.name}: execution cost must be finite")
                offers[option.name] = (support.value, cost)
        except Exception as error:
            failure = f"{type(error).__name__}: {error}"
        gathered = self._gather((offers, failure))
        failures = [error for _, error in gathered if error is not None]
        if failures:
            raise RuntimeError(f"participant support evaluation failed: {failures}")
        common = set.intersection(*(set(offer) for offer, _ in gathered))
        unresolved = [name for name in sorted(common)
                      if all(offer[name][0] != Support.NO.value for offer, _ in gathered)
                      and any(offer[name][0] == Support.UNKNOWN.value for offer, _ in gathered)]
        if unresolved:
            raise UnsupportedInvocation(f"unresolved invocation facts for common executions: {unresolved}")
        supported = [name for name in sorted(common)
                     if all(offer[name][0] == Support.YES.value for offer, _ in gathered)]
        if not supported:
            raise UnsupportedInvocation("no common execution supports the agreed invocation")
        if len(supported) > 1 and any(
            offer[name][1] is None for name in supported for offer, _ in gathered
        ):
            raise AmbiguousExecution(f"overlapping executions need comparable costs: {supported}")
        costs = sorted((sum(offer[name][1] or 0.0 for offer, _ in gathered), name)
                       for name in supported)
        by_name = {option.name: option for option in capabilities.options}
        for _, tier in groupby(costs, key=lambda item: item[0]):
            acquired = []
            try:
                for _, name in tier:
                    workspace = None
                    failure = None
                    try:
                        workspace = by_name[name].acquire(facts)
                    except Exception as error:
                        failure = f"{type(error).__name__}: {error}"
                    try:
                        outcomes = self._gather((workspace is not None, failure))
                    except BaseException:
                        if workspace is not None:
                            workspace.close()
                        raise
                    if any(error is not None for _, error in outcomes):
                        if workspace is not None:
                            workspace.close()
                        raise RuntimeError(f"participant workspace acquisition failed: {outcomes}")
                    if all(ok for ok, _ in outcomes):
                        acquired.append(SelectedExecution(by_name[name], workspace))
                    else:
                        if workspace is not None:
                            workspace.close()
                        logger.info(
                            "Execution agreement: node=%s stage=workspace implementation=%s "
                            "outcome=refused released=%s refused_ranks=%s",
                            self.node_name, name, workspace is not None,
                            [rank for rank, (ok, _) in enumerate(outcomes) if not ok],
                        )
                if len(acquired) > 1:
                    raise AmbiguousExecution(f"equal available instance execution costs: {costs}")
                if acquired:
                    logger.info(
                        "Execution agreement: node=%s stage=selection implementation=%s requests=%s outcome=accepted",
                        self.node_name, acquired[0].option.name, identity,
                    )
                    return acquired[0]
            except BaseException:
                for selected in acquired:
                    selected.close()
                raise
        return None

    def admit(self, runner, step, request_ids=None):
        """Admit the same step everywhere, rolling back successful peers."""
        local_ids = () if step is None else tuple(step.ctx.request_ids)
        request_ids = local_ids if request_ids is None else request_ids
        to_wire = dict(zip(local_ids, request_ids, strict=True))
        from_wire = dict(zip(request_ids, local_ids, strict=True))

        def translated(outcome, mapping):
            reason = outcome.reason
            fields = {}
            if hasattr(reason, "request_id"):
                fields["request_id"] = mapping[reason.request_id]
            if hasattr(reason, "request_ids"):
                fields["request_ids"] = frozenset(mapping[rid] for rid in reason.request_ids)
            if fields:
                return outcome._replace(outcome=outcome.outcome._replace(reason=replace(reason, **fields)))
            return outcome

        local = None
        failure = None
        try:
            local = FULL_ADMIT_OK if step is None else runner.admit(step)
            offered = translated(local, to_wire)
        except Exception as error:
            offered = None
            failure = f"{type(error).__name__}: {error}"
        try:
            outcomes = self._gather((offered, failure))
        except BaseException:
            if step is not None and local is not None and local.ok:
                runner.rollback_admit(step)
            raise
        failures = [error for _, error in outcomes if error is not None]
        refusal = next((outcome for outcome, _ in outcomes if outcome is not None and not outcome.ok), None)
        if failures or refusal is not None:
            rollback_error = None
            if step is not None and local is not None and local.ok:
                try:
                    runner.rollback_admit(step)
                except Exception as error:
                    rollback_error = f"{type(error).__name__}: {error}"
            rollback_errors = self._gather(rollback_error)
            if failures or any(error is not None for error in rollback_errors):
                raise InstanceFatalError(
                    self.node_name, f"resource admission or rollback failed: {failures}; {rollback_errors}",
                )
            logger.info("Execution agreement: node=%s stage=resources outcome=rolled_back resource=%s",
                        self.node_name, refusal.failed_resource)
            return translated(refusal, from_wire)
        return FULL_ADMIT_OK if all(outcome.ready for outcome, _ in outcomes) else FULL_ADMIT_NOT_READY

    @contextmanager
    def executing(self):
        try:
            yield
        except InstanceFatalError:
            raise
        except Exception as error:
            raise InstanceFatalError(self.node_name, f"execution failed: {error}") from error

    def complete(self):
        """All callbacks returned before any participant accepts their effects."""
        self.agree("returned", "execution completion")
