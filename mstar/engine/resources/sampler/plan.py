"""Sampling inputs and returned physical output for a planned realization."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mstar.engine.resources.sampler.utils import HistoryRows, SamplingConfig


@dataclass(frozen=True)
class SamplingPlan:
    request_ids: tuple[int, ...]
    configs: tuple[SamplingConfig, ...]
    offsets: tuple[int, ...]
    kept: tuple[bool, ...]
    apply_penalty: bool
    offset_stride: int
    seen_tokens: tuple[torch.Tensor | None, ...]
    apply_filters: bool = True
    history: HistoryRows | None = None


@dataclass(frozen=True)
class SamplingResult:
    """Owned tokens in plan row order; returning them does not commit RNG."""
    tokens: torch.Tensor
