"""Actual pinned tokenizer/media preprocessing, without loading model weights.

Set MSTAR_QWEN3_OMNI_PATH to a local snapshot with the tokenizer and processor
files. These tests execute the public model loaders and prompt builder on CPU;
they are not GPU model-inference evidence.
"""
import os
from pathlib import Path

import pytest
import torch
import yaml

from mstar.engine.resources.spec import apply_yaml_overrides
from mstar.engine.resources.sequence import HeadSharded, PagedSequenceView, resolve_sequence_config
from mstar.engine.resources.kv.config import KVSpec
from mstar.model.qwen3_omni.qwen3_omni_model import Qwen3OmniModel


ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = os.environ.get("MSTAR_QWEN3_OMNI_PATH")
pytestmark = pytest.mark.skipif(not CHECKPOINT, reason="set MSTAR_QWEN3_OMNI_PATH for actual tokenizer/processor assets")


@pytest.fixture(scope="module")
def model():
    value = Qwen3OmniModel(model_path_hf=CHECKPOINT)
    assert value._processor is not None
    return value


@pytest.mark.parametrize("modality", ["image", "audio", "video"])
def test_public_media_loader_and_actual_processor(model, tmp_path, modality):
    from PIL import Image

    if modality == "image":
        path = tmp_path / "image.png"
        Image.new("RGB", (224, 224), "red").save(path)
    else:
        path = ROOT / "test/qwen3-omni" / {"audio": "audio.wav", "video": "video.webm"}[modality]
    loaded = getattr(model, f"load_{modality}")(str(path), "cpu")
    assert loaded.data.numel() > 0
    assert torch.isfinite(loaded.data).all()
    key = f"{modality}_inputs"
    outputs = model.process_prompt(
        "Describe the input briefly.", [modality, "text"], ["text"],
        tensors={key: [loaded.data]}, input_metadata={key: [loaded.metadata]},
    )
    assert len(outputs["text_inputs"]) == 2
    assert all(value.dtype == torch.long and value.numel() for value in outputs["text_inputs"])
    feature_key = {"image": "pixel_values", "audio": "audio_features", "video": "pixel_values_videos"}[modality]
    features = outputs[feature_key][0]
    assert features.numel() > 0 and torch.isfinite(features).all()
    if modality == "audio":
        assert loaded.metadata["sample_rate"] == 16000
        assert outputs["audio_seqlens"][0].item() == features.shape[-1]


@pytest.mark.parametrize("filename", sorted(path.name for path in (ROOT / "configs").glob("qwen3omni*.yaml")))
def test_published_geometry_and_every_reference_deployment_budget(model, filename):
    config = yaml.safe_load((ROOT / "configs" / filename).read_text())
    specs = model.get_node_resources()
    apply_yaml_overrides(specs, config)
    kv_specs = [spec for spec in specs if isinstance(spec, KVSpec)]
    assert len(kv_specs) == 2
    for spec in kv_specs:
        # Each physical consumer resolves the actual model geometry; the YAML
        # supplies bytes, never a duplicated logical shape.
        resolved = resolve_sequence_config(
            spec, [PagedSequenceView(partition=HeadSharded())], dtype=torch.bfloat16,
        )
        assert resolved.max_num_pages >= 256
        assert resolved.page_size == 128
        assert resolved.head_dim == 128
