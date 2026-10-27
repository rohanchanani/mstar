"""Logical results of Whisper node transitions.

Tensors carry normal Torch storage ownership.  Results do not transfer an
implementation's reusable workspace to the model lifecycle.
"""
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class EncoderResult:
    encoder_states: tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class DecoderResult:
    tokens: torch.Tensor
