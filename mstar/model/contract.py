"""Typed logical node interfaces, before implementation materialization."""
from collections.abc import Mapping, Callable
from dataclasses import dataclass, field

import torch


@dataclass(frozen=True)
class TensorInput:
    name: str
    source: str
    shape: tuple[int | str, ...]
    dtype: torch.dtype | None = None
    walks: tuple[str, ...] = ()


@dataclass(frozen=True)
class TensorOutput:
    name: str
    shape: tuple[int | str, ...]
    # First alternative is used when allocating output storage; returned
    # tensors may use any explicitly declared alternative without conversion.
    dtype: torch.dtype | tuple[torch.dtype, ...] | None = None
    per_request: bool = False
    present_when: Callable | None = None

    def __post_init__(self):
        if isinstance(self.dtype, tuple) and (
            not self.dtype or any(not isinstance(dtype, torch.dtype) for dtype in self.dtype)
        ):
            raise ValueError("output dtype alternatives must be a nonempty tuple of torch dtypes")

    def dtypes(self, fallback):
        return self.dtype if isinstance(self.dtype, tuple) else (self.dtype or fallback,)


@dataclass(frozen=True)
class NodeContract:
    """Information not recoverable from the graph's named edges alone.

    Input sources address already interpreted ``NodeInputs`` fields. Shapes
    describe one input row; ``tokens`` is that row's sequence length and
    ``batch`` is the invocation's row count. A missing dtype uses the bound
    numerical policy. Resource relationships remain in the resource specs.
    """

    inputs: tuple[TensorInput, ...]
    outputs: tuple[TensorOutput, ...]
    result_type: type
    weight_prefix: str
    suppressed_tokens: Mapping[str, Mapping[str, tuple[int, ...]]] = field(default_factory=dict)
    dimensions: Callable | None = None
    input_preparation: type | None = None


def row_dimensions(contract, logical, walk, info, batch):
    values = {"batch": batch}
    if hasattr(logical, "input_seq_len"):
        values["tokens"] = logical.input_seq_len
    if contract.dimensions is not None:
        extra = contract.dimensions(logical, walk, info)
        if values.keys() & extra.keys():
            raise ValueError("model dimensions cannot replace tokens or batch")
        values.update(extra)
    if any(type(n) is not int or n < 0 for n in values.values()):
        raise ValueError("logical dimensions must be nonnegative integers")
    return values


def tensor_shape(spec, dimensions):
    return tuple(dimensions[n] if isinstance(n, str) else n for n in spec.shape)
