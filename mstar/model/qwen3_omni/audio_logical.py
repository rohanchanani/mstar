"""Logical audio-encoder interface, independent of its numerical implementation."""
import torch

from mstar.engine.execution import ExecutionResult
from mstar.model.contract import NodeContract, TensorInput, TensorOutput
from mstar.model.qwen3_omni.results import AudioResult
from mstar.model.submodule_base import BatchedModelOutput, NodeInputs, NodeSubmodule


class AudioEncoderLogical(NodeSubmodule):
    execution_result_is_output_only = True
    def __init__(self, config):
        super().__init__()
        self.config = config

    def prepare_inputs(self, graph_walk, fwd_info, inputs, **kwargs):
        return NodeInputs(tensor_inputs={"audio_features": inputs["audio_features"][0],
                          "audio_seqlens": inputs.get("audio_seqlens", [None])[0]})

    def preprocess(self, *args, **kwargs):
        raise RuntimeError("physical preprocessing belongs to the selected implementation")

    def forward(self, *args, **kwargs):
        raise RuntimeError("select a physical audio implementation")

    @staticmethod
    def _dimensions(logical, walk, info):
        features = logical.tensor_inputs["audio_features"]
        lengths = logical.tensor_inputs["audio_seqlens"]
        # Same three stride-2 convolution lengths as the model's audio encoder.
        lengths = lengths if lengths is not None else torch.tensor([features.shape[-1]])
        remainder = lengths % 100
        first = (remainder - 1) // 2 + 1
        output_lengths = ((first - 1) // 2 + 1 - 1) // 2 + 1 + (lengths // 100) * 13
        return {"mel_bins": features.shape[-2], "frames": features.shape[-1],
                "segments": lengths.numel(), "audio_tokens": int(output_lengths.sum().item())}

    def logical_contract(self):
        return NodeContract(inputs=(
                    TensorInput("audio_features", "tensor_inputs.audio_features",
                                ("mel_bins", "frames"), dtype=torch.float32),
                    TensorInput("audio_seqlens", "tensor_inputs.audio_seqlens", ("segments",), dtype=torch.long)),
            outputs=(TensorOutput("audio_embeds", ("audio_tokens", self.config.thinker_hidden_size),
                                  per_request=True),),
            dimensions=self._dimensions, result_type=AudioResult, weight_prefix="thinker.audio_tower.")

    def execution_result(self, result, invocation):
        if not isinstance(result, AudioResult):
            return super().execution_result(result, invocation)
        return ExecutionResult(BatchedModelOutput(per_rid_outputs={
            rid: {"audio_embeds": [result.audio_embeds[i]]}
            for i, rid in enumerate(invocation.request_ids)}))
