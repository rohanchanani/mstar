"""Whisper log-mel extraction with exact-zero FFT work elision on CPU."""

import numpy as np
import torch
from transformers import WhisperFeatureExtractor as HFFeatureExtractor


class WhisperFeatureExtractor(HFFeatureExtractor):
    """Keep HF input/configuration semantics and the complete feature context."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cpu_window = torch.hann_window(self.n_fft, dtype=torch.float32, device="cpu").numpy()
        self._cpu_filters = np.ascontiguousarray(self.mel_filters.T, dtype=np.float32)
        self._cpu_window.flags.writeable = self._cpu_filters.flags.writeable = False

    def to_dict(self):
        config = super().to_dict()
        config.pop("_cpu_window", None)
        config.pop("_cpu_filters", None)
        return config

    def _torch_extract_fbank_features(self, waveform, device="cpu"):
        if (device != "cpu" or self.dither or waveform.ndim != 2
                or waveform.shape[-1] <= self.n_fft // 2 or torch.get_default_dtype() != torch.float32):
            return super()._torch_extract_fbank_features(waveform, device)
        waveform = waveform.astype(np.float32, copy=False)
        half = self.n_fft // 2
        frame_count = (waveform.shape[-1] + 2 * half - self.n_fft) // self.hop_length
        nonzero = np.flatnonzero(np.any(waveform != 0, axis=0))
        active = (min(frame_count, (int(nonzero[-1]) + 1 + half + self.hop_length - 1) // self.hop_length)
                  if len(nonzero) else 0)
        if active == frame_count:
            return super()._torch_extract_fbank_features(waveform, device)

        # HF has already padded/truncated the waveform. A frame after this
        # support bound contains only zeros, including center-reflection edges.
        # Its mel energy is zero, but its final log-mel value is not zero.
        mel = np.empty((len(waveform), self.feature_size, active), dtype=np.float32)
        if active:
            centered = np.pad(waveform, ((0, 0), (half, half)), mode="reflect")
            frames = np.lib.stride_tricks.sliding_window_view(centered, self.n_fft, axis=-1)
            frames = frames[:, ::self.hop_length, :][:, :active, :]
            stft = np.fft.rfft(frames * self._cpu_window, axis=-1).astype(np.complex64)
            magnitudes = np.abs(stft) ** 2
            mel[:] = self._cpu_filters @ magnitudes.transpose(0, 2, 1)
        # Inactive frames have the same pre-log energy. Include their floor
        # in the per-clip maximum, then fill their final value once.
        tail = np.log10(np.float32(1e-10))
        log_spec = np.log10(np.maximum(mel, np.float32(1e-10)))
        maximum = (np.maximum(log_spec.max(axis=(-2, -1), keepdims=True), tail)
                   if active else np.full((len(waveform), 1, 1), tail, dtype=np.float32))
        floor = maximum - 8.0
        np.maximum(log_spec, floor, out=log_spec)
        np.add(log_spec, 4.0, out=log_spec)
        np.divide(log_spec, 4.0, out=log_spec)
        tail = np.maximum(tail, floor)
        np.add(tail, 4.0, out=tail)
        np.divide(tail, 4.0, out=tail)
        features = np.empty((len(waveform), self.feature_size, frame_count), dtype=np.float32)
        features[:, :, :active] = log_spec
        features[:, :, active:] = tail
        return features
