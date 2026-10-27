"""The model frontend retains HF's full-context input and feature contract.

The float comparison uses the existing audio-frontend tolerance from
test_audio_features.py, not the decoder's quantization policy. Target campaign
checks additionally reuse its frozen HF features and component comparator.
"""

import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

transformers = pytest.importorskip("transformers")
from mstar.model.whisper.feature_extraction import WhisperFeatureExtractor  # noqa: E402


@pytest.fixture(autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def extract(extractor, waveform, **kwargs):
    return extractor(waveform, sampling_rate=16000, return_tensors="pt", **kwargs)


def compare(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype == torch.float32
    assert torch.isfinite(actual).all()
    assert torch.allclose(actual, expected, atol=2e-4), (actual - expected).abs().max()


@pytest.mark.parametrize("feature_size", [80, 128])
@pytest.mark.parametrize("length", [0, 1, 159, 160, 199, 200, 399, 400, 401, 16000, 479999, 480000, 480001])
def test_zero_short_full_and_truncated_inputs_match_hf(feature_size, length):
    rng = np.random.default_rng(231)
    waveform = rng.uniform(-.1, .1, length).astype(np.float32)
    original = transformers.WhisperFeatureExtractor(feature_size=feature_size)
    candidate = WhisperFeatureExtractor(feature_size=feature_size)
    actual = extract(candidate, waveform)["input_features"]
    assert actual.shape == (1, feature_size, 3000)
    compare(actual, extract(original, waveform)["input_features"])


@pytest.mark.parametrize("position", [0, 1, 119, 120, 159, 160, 199, 200, 399, 400,
                                     479638, 479639, 479640, 479799, 479800, 479998, 479999])
def test_impulse_support_includes_hop_and_reflection_boundaries(position):
    waveform = np.zeros(480000, dtype=np.float32)
    waveform[position] = .1
    candidate = WhisperFeatureExtractor()
    expected = extract(transformers.WhisperFeatureExtractor(), waveform)["input_features"]
    compare(extract(candidate, waveform)["input_features"], expected)


def test_silence_keeps_complete_nonzero_log_mel_floor():
    candidate = WhisperFeatureExtractor()
    with patch("numpy.fft.rfft", side_effect=AssertionError("silence needs no FFT")):
        features = extract(candidate, np.zeros(16000, dtype=np.float32))["input_features"]
    assert features.shape == (1, 80, 3000)
    assert not bool((features == 0).any())
    original = extract(transformers.WhisperFeatureExtractor(), np.zeros(16000, dtype=np.float32))
    compare(features, original["input_features"])


def test_short_audio_really_elides_only_zero_fft_frames():
    waveform = np.ones(16000, dtype=np.float32) * .05
    fft = np.fft.rfft
    with patch("numpy.fft.rfft", wraps=fft) as call:
        features = extract(WhisperFeatureExtractor(), waveform)["input_features"]
    assert features.shape == (1, 80, 3000)
    assert call.call_count == 1
    assert 0 < call.call_args.args[0].shape[1] < 3000


def test_nonzero_suffix_uses_unchanged_hf_path():
    waveform = np.zeros(480000, dtype=np.float32)
    waveform[-1] = .1
    with patch("numpy.fft.rfft", side_effect=AssertionError("full frame set delegates to HF")):
        actual = extract(WhisperFeatureExtractor(), waveform)["input_features"]
    compare(actual, extract(transformers.WhisperFeatureExtractor(), waveform)["input_features"])


@pytest.mark.parametrize("normalize", [False, True])
def test_batch_padding_normalization_and_per_clip_clamp_stay_hf_owned(normalize):
    rng = np.random.default_rng(231)
    quiet = rng.uniform(-.001, .001, 16000).astype(np.float32)
    loud = rng.uniform(-.5, .5, 32000).astype(np.float32)
    candidate = WhisperFeatureExtractor()
    original = transformers.WhisperFeatureExtractor()
    actual = extract(candidate, [quiet, loud], do_normalize=normalize, return_attention_mask=True)
    expected = extract(original, [quiet, loud], do_normalize=normalize, return_attention_mask=True)
    compare(actual["input_features"], expected["input_features"])
    assert torch.equal(actual["attention_mask"], expected["attention_mask"])
    compare(actual["input_features"][:1], extract(candidate, quiet, do_normalize=normalize)["input_features"])


def test_nonzero_padding_and_serialized_configuration_stay_hf_owned():
    original = transformers.WhisperFeatureExtractor(padding_value=.01)
    candidate = WhisperFeatureExtractor(padding_value=.01)
    waveform = np.ones(16000, dtype=np.float32) * .05
    compare(extract(candidate, waveform)["input_features"], extract(original, waveform)["input_features"])
    assert candidate.to_dict() == original.to_dict()


@pytest.mark.parametrize("device,dither", [("cuda", 0.), ("cpu", .1)])
def test_unsupported_paths_delegate_to_hf(device, dither):
    candidate = WhisperFeatureExtractor(dither=dither)
    waveform = np.zeros((1, 480000), dtype=np.float32)
    marker = object()
    with patch.object(transformers.WhisperFeatureExtractor, "_torch_extract_fbank_features",
                      return_value=marker) as parent:
        assert candidate._torch_extract_fbank_features(waveform, device) is marker
    parent.assert_called_once_with(waveform, device)


@pytest.mark.skipif(not os.environ.get("MSTAR_TEST_WHISPER_CHECKPOINT"),
                    reason="requires explicit local checkpoint; never downloads")
def test_actual_checkpoint_and_process_prompt_keep_full_context():
    from mstar.model.whisper.whisper_model import WhisperModel

    checkpoint = Path(os.environ["MSTAR_TEST_WHISPER_CHECKPOINT"]).resolve()
    model = WhisperModel(str(checkpoint), local_files_only=True)
    original = transformers.WhisperFeatureExtractor.from_pretrained(checkpoint, local_files_only=True)
    assert isinstance(model.feature_extractor, WhisperFeatureExtractor)
    assert model.feature_extractor is model.feature_extractor
    assert model.feature_extractor.to_dict() == original.to_dict()
    waveform = torch.ones(16000, dtype=torch.float32) * .05
    result = model.process_prompt(None, ["audio"], ["text"], tensors={"audio_inputs": [waveform]},
                                  language="en", task="transcribe")
    features, = result["audio_features"]
    assert features.shape == (model.config.num_mel_bins, original.nb_max_frames)
    compare(features, extract(original, waveform.numpy())["input_features"][0])
    assert result["text_inputs"][0].tolist() == model.config.decoder_prompt_ids("en", "transcribe")


@pytest.mark.parametrize("position", [0, 159, 160, 32000])
@pytest.mark.parametrize("special", ["finite", "zero", "infinity", "nan"])
def test_active_tail_matches_original_full_array_elementwise_arithmetic(position, special):
    candidate = WhisperFeatureExtractor()
    waveform = np.zeros((2, 480000), dtype=np.float32)
    waveform[:, position] = .1
    active = (position + 1 + 200 + 159) // 160
    rng = np.random.default_rng(712)
    stft = (rng.uniform(-.5, .5, (2, active, 201))
            + 1j * rng.uniform(-.5, .5, (2, active, 201))).astype(np.complex64)
    stft[0] *= np.float32(.001)
    if special == "zero":
        stft[:] = 0
    elif special == "infinity":
        stft[1, 0, 1] = complex(float("inf"), 0.)
    elif special == "nan":
        stft[1, 0, 1] = complex(float("nan"), 0.)
    original_stft = stft.copy()
    # The old implementation applied these elementwise operations to the full
    # matrix, with an exact-zero mel tail and one maximum per clip.
    with np.errstate(all="ignore"):
        mel = np.zeros((2, 80, 3000), dtype=np.float32)
        mel[:, :, :active] = candidate._cpu_filters @ (np.abs(stft) ** 2).transpose(0, 2, 1)
        expected = np.log10(np.maximum(mel, np.float32(1e-10)))
        expected = np.maximum(expected, expected.max(axis=(-2, -1), keepdims=True) - 8.0)
        expected = (expected + 4.0) / 4.0
        with patch("numpy.fft.rfft", return_value=stft):
            actual = candidate._torch_extract_fbank_features(waveform)
    assert actual.dtype == np.float32 and actual.shape == (2, 80, 3000)
    assert np.array_equal(actual.view(np.uint32), expected.view(np.uint32))
    assert np.array_equal(stft.view(np.uint32), original_stft.view(np.uint32))


def test_constant_tail_does_not_run_full_context_logarithm():
    candidate = WhisperFeatureExtractor()
    waveform = np.ones(16000, dtype=np.float32) * .05
    with patch("numpy.log10", wraps=np.log10) as logarithm:
        features = extract(candidate, waveform)["input_features"]
    assert features.shape == (1, 80, 3000)
    assert logarithm.call_count == 2  # one scalar floor and the active mel block
    arrays = [call.args[0] for call in logarithm.call_args_list if isinstance(call.args[0], np.ndarray)]
    assert len(arrays) == 1 and 0 < arrays[0].shape[-1] < 3000
