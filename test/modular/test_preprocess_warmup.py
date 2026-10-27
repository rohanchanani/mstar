"""The preprocess worker warms the model's prompt-side state once at start."""

import logging

from mstar.api_server.data_worker import warm_up_model
from mstar.model.base import Model


def test_default_warmup_is_a_no_op():
    assert Model.warmup_preprocess.__doc__
    assert Model.warmup_preprocess(object()) is None


def test_warm_up_model_calls_and_survives_failures(caplog):
    calls = []

    class Fine:
        def warmup_preprocess(self):
            calls.append("fine")

    class Broken:
        def warmup_preprocess(self):
            raise RuntimeError("no spaCy model")

    warm_up_model(Fine())
    assert calls == ["fine"]
    warm_up_model(None)
    warm_up_model(object())  # a model without the hook
    with caplog.at_level(logging.ERROR):
        warm_up_model(Broken())
    assert "warmup_preprocess failed" in caplog.text


def test_omnivoice_warms_its_data_worker_assets():
    """OmniVoice loads its tokenizer and duration estimator on first use; the
    preprocess warm-up pulls that forward to start-up."""
    from mstar.model.omnivoice.omnivoice_model import OmniVoiceModel

    model = OmniVoiceModel.__new__(OmniVoiceModel)
    calls = []
    model._ensure_data_worker_assets = lambda: calls.append("loaded")
    model.warmup_preprocess()
    assert calls == ["loaded"]


def test_whisper_warms_the_actual_prompt_and_output_property_caches(tmp_path, monkeypatch):
    """Only asset factories are intercepted; model properties and consumers run."""
    import json

    import torch
    from tokenizers import Tokenizer, models
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    from mstar.model.whisper.feature_extraction import WhisperFeatureExtractor
    from mstar.model.whisper.whisper_model import WhisperModel

    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "d_model": 8,
                "encoder_layers": 1,
                "decoder_layers": 1,
                "encoder_attention_heads": 2,
                "decoder_attention_heads": 2,
                "encoder_ffn_dim": 16,
                "decoder_ffn_dim": 16,
                "num_mel_bins": 4,
                "max_source_positions": 1500,
                "max_target_positions": 8,
                "vocab_size": 8,
                "decoder_start_token_id": 3,
            }
        )
    )
    (tmp_path / "generation_config.json").write_text(
        json.dumps(
            {
                "lang_to_id": {"<|en|>": 4},
                "task_to_id": {"transcribe": 5},
                "no_timestamps_token_id": 6,
            }
        )
    )
    model = WhisperModel.describe(model_path_hf=str(tmp_path))
    features = WhisperFeatureExtractor(feature_size=4)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(models.WordLevel({"!": 0, "a": 1, "<unk>": 2}, unk_token="<unk>")),
        unk_token="<unk>",
    )
    calls = []

    def factory(kind, value):
        def load(path, **kwargs):
            assert path == str(tmp_path.resolve())
            assert kwargs == {"cache_dir": None, "local_files_only": True}
            calls.append(kind)
            return value

        return load

    monkeypatch.setattr(WhisperFeatureExtractor, "from_pretrained", factory("features", features))
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", factory("tokenizer", tokenizer))
    assert model._feature_extractor is model._tokenizer is model._detokenizer_cache is None
    warm_up_model(model)
    assert calls == ["features", "tokenizer"]
    assert model._feature_extractor is features and model._tokenizer is tokenizer
    detokenizer = model._detokenizer_cache
    assert detokenizer.tokenizer is tokenizer
    warm_up_model(model)
    inputs = model.process_prompt(
        None, ["audio"], ["text"], tensors={"audio_inputs": [torch.zeros(1600)]}
    ).new_input_tensors
    assert inputs["audio_features"][0].shape == (4, 3000)
    assert inputs["text_inputs"][0].tolist() == [3, 4, 5, 6]
    assert model.postprocess(torch.tensor([1, 0, 2]), "text") == b"a!"
    assert model._detokenizer_cache is detokenizer
    assert model._submodule_cache == {}
    assert calls == ["features", "tokenizer"]
