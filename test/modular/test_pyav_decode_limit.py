"""Portable FFmpeg decoding obeys the shared refusal boundary."""
from contextlib import contextmanager
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest
import torch

av = pytest.importorskip("av")

from mstar.api_server import media_io as api  # noqa: E402
from mstar.model.components import media_io as portable  # noqa: E402


@pytest.mark.parametrize("container_format", ["WAV", "FLAC"])
def test_valid_samples_match_portable_decoder(tmp_path, container_format):
    sf = pytest.importorskip("soundfile")
    path = str(tmp_path / ("clip." + container_format.lower()))
    tone = np.sin(np.arange(4000) * 0.1).astype(np.float32)
    sf.write(path, np.stack([tone, tone * 0.5], axis=1), 8000, format=container_format)
    expected = portable._PyAVAudio().decode(path, 16000, 1).data[0]
    actual = api._decode_pyav_capped(path, 16000, 0.5)
    assert torch.equal(actual, expected)
    with pytest.raises(api.AudioTooLongError):
        api._decode_pyav_capped(path, 16000, 0.25)


@pytest.mark.parametrize("kind", ["samples", "elapsed", "flush", "valid_offset"])
def test_bounds_use_decoded_content_not_header(monkeypatch, kind):
    rate, start = 16000, 30
    visited = []
    converted = []
    def frame(n, offset):
        def array():
            converted.append(n)
            return np.zeros((1, n), dtype=np.float32)
        return SimpleNamespace(samples=n, pts=start * rate + offset,
                               time_base=Fraction(1, rate), to_ndarray=array)
    first = frame(rate // 2, 0)
    second = frame(rate // 2, rate // 2)
    extra = frame(1, 0)  # compressed timestamps: sample count still refuses
    frames = [first, second]
    flush = []
    if kind == "samples":
        frames += [extra, frame(1, 0)]
    elif kind == "elapsed":
        frames[1] = frame(rate // 2, rate)
    elif kind == "flush":
        flush = [extra]
    stream = SimpleNamespace(start_time=start * rate, time_base=Fraction(1, rate), duration=1)
    def decode(_stream):
        for f in frames:
            visited.append(f)
            yield f
    @contextmanager
    def opened(_):
        yield SimpleNamespace(streams=SimpleNamespace(audio=[stream]), decode=decode)
    monkeypatch.setattr(av, "open", opened)
    monkeypatch.setattr(av, "AudioResampler", lambda **kw: SimpleNamespace(
        resample=lambda f: flush if f is None else [f]))
    if kind == "valid_offset":
        assert api._decode_pyav_capped("x", rate, 1).numel() == rate
    else:
        with pytest.raises(api.AudioTooLongError):
            api._decode_pyav_capped("x", rate, 1)
        assert len(visited) <= 3
        assert 1 not in converted  # over-limit frame never retained


@pytest.mark.parametrize("loader", ["api", "model"])
@pytest.mark.parametrize("soundfile_refuses", [False, True])
def test_refusal_propagates_through_loaders(monkeypatch, loader, soundfile_refuses):
    from mstar.model.components.audio_features import load_audio_file
    calls = []
    monkeypatch.setattr(portable, "uses_torchcodec", lambda: False)
    def soundfile(*args, **kwargs):
        if soundfile_refuses:
            raise api.AudioTooLongError(api.MAX_AUDIO_SECONDS)
        raise RuntimeError("unsupported container")
    def pyav(*args, **kwargs):
        calls.append("pyav")
        raise api.AudioTooLongError(api.MAX_AUDIO_SECONDS)
    monkeypatch.setattr(api, "read_soundfile_capped", soundfile)
    monkeypatch.setattr(api, "_decode_pyav_capped", pyav)
    monkeypatch.setattr(api, "decode_torchcodec_capped", lambda *a: pytest.fail("retried refusal"))
    with pytest.raises(api.AudioTooLongError):
        if loader == "api":
            api.decode_audio("x")
        else:
            load_audio_file("x")
    assert calls == ([] if soundfile_refuses else ["pyav"])
