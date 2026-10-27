"""Physical KV encoding identities, independent of logical tensor precision."""
from dataclasses import dataclass

import torch


class KVEncoding:
    """An encoding description, not a promise of one scalar dtype or plane.

    Future packed/scaled encodings need their own allocation and access paths.
    They must not inherit plain storage's addressing or interpretation.
    """


@dataclass(frozen=True)
class PlainKVEncoding(KVEncoding):
    """Unscaled scalar K/V values, with no auxiliary metadata."""

    dtype: torch.dtype

    def __post_init__(self):
        if not isinstance(self.dtype, torch.dtype):
            raise TypeError("plain KV encoding requires a torch dtype")


def plain_kv_dtype(encoding: KVEncoding) -> torch.dtype:
    """Check the concrete format at a plain tensor-storage boundary."""
    if type(encoding) is not PlainKVEncoding:
        raise TypeError(f"plain KV storage does not implement encoding {encoding!r}")
    return encoding.dtype
