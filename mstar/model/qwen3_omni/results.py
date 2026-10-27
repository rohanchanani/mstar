"""Logical results of one Talker transition."""
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TalkerResult:
    codec_tokens: torch.Tensor
    codec_embeds: torch.Tensor
    tokens: torch.Tensor


@dataclass(frozen=True)
class ThinkerResult:
    new_token: tuple[torch.Tensor | None, ...]
    thinker_states: tuple[torch.Tensor | None, ...]
    thinker_mask: tuple[torch.Tensor | None, ...]


@dataclass(frozen=True)
class AudioResult:
    audio_embeds: tuple[torch.Tensor, ...]
