"""Deployment budgets for logical sequence storage."""
from dataclasses import dataclass


@dataclass(frozen=True)
class SequenceBudget:
    """Deployment budget for tensor payload; allocator overhead is separate."""

    max_bytes: int
    prefix_cache: bool = False
    cpu_offload_bytes: int = 0
    prefix_cache_salt: str = ""

    def __post_init__(self):
        if self.max_bytes <= 0 or self.cpu_offload_bytes < 0:
            raise ValueError("sequence storage needs a positive byte budget")
