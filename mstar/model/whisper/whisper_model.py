"""
WhisperModel: encoder-decoder ASR model (openai/whisper-large-v3).

Whisper transcribes speech: a log-mel spectrogram runs through a
32-layer audio encoder once, and a 32-layer text decoder generates the
transcript autoregressively, attending to the encoder output through
cross-attention at every step.

Architecture (2 nodes, single partition):
    audio_encoder  (enc_dec) — HF WhisperEncoder, one-shot at prefill
    decoder        (ar)      — mstar-native decoder; paged self-attn KV
                               cache + per-request static cross-attn K/V

Graph walks:
    prefill — audio_encoder -> decoder (forced decoder prompt; the
              engine samples the first transcript token from the logits)
    decode  — decoder loop; each step feeds the sampled token back

Cross-attention K/V depend only on the encoder output, so the decoder
projects them once at prefill and writes them into a context KV stream;
every later step declares a zero-span segment on that label and runs the
engine's pre-planned cross-attention resource.
"""

import logging
from pathlib import Path
from time import perf_counter

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import (
    CurrentForwardConductorMetadata,
    StreamingConnectionState,
)
from mstar.engine.resources import (
    AttentionConfig,
    AttentionSpec,
    CrossAttentionConfig,
    CrossAttentionSpec,
    KVSpec,
    NodeResourceSpec,
    PositionConfig,
    PositionSpec,
    ResourceReqConfig,
    SamplerSpec,
    SamplingReqConfig,
    SequenceKVConfig,
)
from mstar.graph.base import GraphEdge, GraphNode, GraphSection, Loop, Sequential, TensorPointerInfo
from mstar.graph.special_destinations import EMIT_TO_CLIENT
from mstar.model.base import ForwardPassArgs, KVLabel, LabelLayout, Model, ProcessPromptOutput, Span
from mstar.model.submodule_base import NodeSubmodule
from mstar.model.whisper.config import (
    ATTN,
    CONTEXT_LABEL,
    CROSS_ATTN,
    CROSS_KV_CACHE,
    KV_CACHE,
    POS,
    SAMPLER,
    WhisperModelConfig,
)

logger = logging.getLogger(__name__)


def _resolve_local_hf_snapshot(
    repo_id: str, cache_dir: str | None = None, *, local_files_only: bool = False,
) -> str:
    if Path(repo_id).is_dir():
        return str(Path(repo_id).resolve())
    if local_files_only:
        raise FileNotFoundError(f"model description requires a local configuration directory: {repo_id}")

    from huggingface_hub import snapshot_download

    local_dir = snapshot_download(
        repo_id=repo_id,
        cache_dir=cache_dir,
        local_files_only=False,
    )
    return str(Path(local_dir))


class WhisperModel(Model):
    """Whisper ASR: HF audio encoder + mstar-native AR decoder."""

    def __init__(
        self,
        model_path_hf: str,
        cache_dir: str | None = None,
        local_files_only: bool = False,
        **kwargs,
    ):
        self.cache_dir = cache_dir
        self.model_path_hf = model_path_hf
        self.local_dir = _resolve_local_hf_snapshot(
            model_path_hf, cache_dir=cache_dir, local_files_only=local_files_only,
        )
        self.config = WhisperModelConfig.from_pretrained(self.local_dir)
        self._tokenizer = None
        self._feature_extractor = None
        self._detokenizer_cache = None
        self._submodule_cache: dict[str, NodeSubmodule | None] = {}

    @classmethod
    def describe(cls, **model_kwargs):
        """Read configuration and expose the ordinary model declarations."""
        return cls(**(model_kwargs | {"local_files_only": True}))

    def warmup_preprocess(self) -> None:
        """Load the cached prompt and text-output assets in their owning thread."""
        logger.info("Whisper preprocessing asset warmup started")
        started = perf_counter()
        _ = self.feature_extractor
        features_ready = perf_counter()
        _ = self._detokenizer
        ready = perf_counter()
        logger.info(
            "Whisper preprocessing assets ready: feature_extractor_seconds=%.6f "
            "detokenizer_seconds=%.6f total_seconds=%.6f",
            features_ready - started, ready - features_ready, ready - started,
        )

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.local_dir, cache_dir=self.cache_dir, local_files_only=True,
            )
        return self._tokenizer

    @property
    def feature_extractor(self):
        if self._feature_extractor is None:
            from mstar.model.whisper.feature_extraction import WhisperFeatureExtractor
            self._feature_extractor = WhisperFeatureExtractor.from_pretrained(
                self.local_dir, cache_dir=self.cache_dir, local_files_only=True,
            )
        return self._feature_extractor

    @property
    def _detokenizer(self):
        if self._detokenizer_cache is None:
            from mstar.model.utils import ByteLevelDetokenizer
            self._detokenizer_cache = ByteLevelDetokenizer(self.tokenizer)
        return self._detokenizer_cache

    # -------------------------------------------------------------------
    # Model ABC: resources
    # -------------------------------------------------------------------

    def get_node_resources(self) -> list[NodeResourceSpec]:
        """Decoder self-attention KV + a separate encoder-context KV.

        Two caches rather than two labels on one: the self-attention resource
        plans a wrapper per label of the cache it names, so a shared cache
        would have it planning the 1500-token context every step for nothing.
        ``audio_encoder`` is stateless.
        """
        kv_config = SequenceKVConfig(
            num_layers=self.config.decoder_layers,
            num_kv_heads=self.config.decoder_attention_heads,
            head_dim=self.config.head_dim,
            max_seq_len=self.config.max_target_positions,
            num_qo_heads=self.config.decoder_attention_heads,
        )
        context_kv_config = SequenceKVConfig(
            num_layers=self.config.decoder_layers,
            num_kv_heads=self.config.decoder_attention_heads,
            head_dim=self.config.head_dim,
            max_seq_len=self.config.max_source_positions,
            num_qo_heads=self.config.decoder_attention_heads,
        )
        return [
            KVSpec(resource_key=KV_CACHE, nodes={"decoder"}, config=kv_config),
            AttentionSpec(
                resource_key=ATTN, nodes={"decoder"},
                config=AttentionConfig(kv_cache=KV_CACHE),
            ),
            KVSpec(
                resource_key=CROSS_KV_CACHE, nodes={"decoder"},
                config=context_kv_config,
            ),
            CrossAttentionSpec(
                resource_key=CROSS_ATTN, nodes={"decoder"},
                config=CrossAttentionConfig(
                    kv_cache=CROSS_KV_CACHE,
                    query_kv_cache=KV_CACHE,
                    context_label=CONTEXT_LABEL,
                ),
            ),
            PositionSpec(
                resource_key=POS, nodes={"decoder"},
                # No RoPE: this exists for the position counter, whose planned
                # ids drive the learned ``embed_positions`` lookup.
                config=PositionConfig(kv_cache=KV_CACHE),
            ),
            SamplerSpec(
                resource_key=SAMPLER, nodes={"decoder"},
                vocab_size=self.config.vocab_size,
                # ASR transcription decodes greedily; no seen-token buffers.
                enable_repetion_penalty=False,
            ),
        ]

    def get_request_resource_configs(
        self, partition_fwd_args: dict[str, ForwardPassArgs],
        model_kwargs: dict | None = None,
    ) -> dict[str, ResourceReqConfig]:
        del partition_fwd_args
        model_kwargs = model_kwargs or {}
        return {
            SAMPLER: SamplingReqConfig(
                # ASR default is greedy (temperature 0 -> argmax).
                temperature=model_kwargs.get("temperature", 0.0),
                top_p=model_kwargs.get("top_p", 1.0),
                ignore_eos=model_kwargs.get("ignore_eos", False),
            )
        }

    # -------------------------------------------------------------------
    # Model ABC: graph walk definitions
    # -------------------------------------------------------------------

    def get_max_output_tokens(self, **model_kwargs):
        # The learned position table caps prompt + generated tokens at
        # max_target_positions (448); the forced decoder prompt takes 4.
        limit = self.config.max_target_positions - 4
        return min(model_kwargs.get("max_output_tokens", limit), limit)

    def get_graph_walk_graphs(self) -> dict[str, GraphSection]:
        prefill = Sequential([
            GraphNode(
                name="audio_encoder",
                input_names=["audio_features"],
                outputs=[GraphEdge(next_node="decoder", name="encoder_states")],
            ),
            GraphNode(
                name="decoder",
                input_names=["encoder_states", "text_inputs"],
                outputs=[
                    GraphEdge(
                        next_node=EMIT_TO_CLIENT,
                        name="new_token",
                        output_modality="text",
                        persist=True,
                    ),
                ],
            ),
        ])

        decode = Loop(
            name="decode_loop",
            section=GraphNode(
                name="decoder",
                input_names=["text_inputs"],
                outputs=[
                    GraphEdge(
                        next_node=EMIT_TO_CLIENT,
                        name="new_token",
                        output_modality="text",
                    ),
                    GraphEdge(
                        next_node="decoder",
                        name="text_inputs",
                    ),
                ],
            ),
            max_iters=self.get_max_output_tokens(),
            outputs=[],
        )

        return dict(prefill=prefill, decode=decode)

    # -------------------------------------------------------------------
    # Model ABC: forward pass args
    # -------------------------------------------------------------------

    def get_initial_forward_pass_args(
        self,
        partition_name: str,
        input_modalities: list[str],
        output_modalities: list[str],
        input_signals: dict[str, list[TensorPointerInfo]],
        model_kwargs: dict | None = None,
    ) -> ForwardPassArgs:
        full_metadata = CurrentForwardConductorMetadata(
            input_modalities=input_modalities,
            output_modalities=output_modalities,
            graph_walk="prefill",
            is_prefill=True,
        )

        audio_edge = GraphEdge(next_node="audio_encoder", name="audio_features")
        audio_edge.tensor_info = input_signals.get("audio_features", [])
        text_edge = GraphEdge(next_node="decoder", name="text_inputs")
        text_edge.tensor_info = input_signals.get("text_inputs", [])
        inputs = [audio_edge, text_edge]

        unpersist_tensors = sum([inp.tensor_info for inp in inputs], start=[])

        return ForwardPassArgs(
            full_metadata=full_metadata,
            inputs=inputs,
            unpersist_tensors=unpersist_tensors,
            step_metadata={"is_prefill": True},
        )

    def get_partition_forward_pass_args(
        self,
        partition_name: str,
        partition_metadata: CurrentForwardConductorMetadata,
        persist_signals: dict[str, list[TensorPointerInfo]],
        incoming_connections: list[StreamingConnectionState] | None = None,
    ) -> ForwardPassArgs:
        """Single-partition state machine: prefill -> decode loop -> done."""
        metadata = partition_metadata

        if metadata.is_prefill:
            metadata.is_prefill = False
            metadata.graph_walk = "decode"
        elif metadata.graph_walk == "decode":
            # The decode dynamic loop returned to the conductor: EOS or
            # max tokens was hit, so the request is complete.
            return ForwardPassArgs(
                full_metadata=metadata,
                inputs=[],
                unpersist_tensors=[],
                request_done=True,
            )

        edge = GraphEdge(next_node="decoder", name="text_inputs")
        edge.tensor_info = persist_signals.get("new_token", [])
        inputs = [edge]
        unpersist_tensors = sum([inp.tensor_info for inp in inputs], start=[])

        return ForwardPassArgs(
            full_metadata=metadata,
            inputs=inputs,
            unpersist_tensors=unpersist_tensors,
            step_metadata={"is_prefill": False},
        )

    # -------------------------------------------------------------------
    # Model ABC: prompt processing
    # -------------------------------------------------------------------

    def max_input_files(self) -> dict[str, int]:
        return {"audio": 1}


    def process_prompt(
        self,
        prompt: str | None,
        input_modalities: list[str],
        output_modalities: list[str],
        tensors: NameToTensorList | None = None,
        **kwargs,
    ) -> ProcessPromptOutput:
        """Extract the log-mel spectrogram and build the forced decoder prompt.

        The text ``prompt`` is unused — Whisper is conditioned via the
        forced token sequence (language / task / timestamps), which is
        controlled by ``language`` and ``task`` in model kwargs.
        """
        raw_audio_inputs = (tensors or {}).get("audio_inputs", [])
        if len(raw_audio_inputs) != 1:
            raise ValueError(
                f"Whisper expects exactly one audio input per request; "
                f"got {len(raw_audio_inputs)}."
            )

        # The extractor pads/truncates to the fixed 30 s window; audio
        # beyond 30 s is dropped (no long-form chunking yet).
        feat = self.feature_extractor(
            raw_audio_inputs[0].cpu().numpy(),
            sampling_rate=self.feature_extractor.sampling_rate,
            return_tensors="pt",
        )
        audio_features = feat["input_features"][0]  # (num_mel_bins, 3000)

        prompt_ids = self.config.decoder_prompt_ids(
            language=kwargs.get("language", "en"),
            task=kwargs.get("task", "transcribe"),
        )

        # the encoder takes the padded window whole, so its context is always this long
        context = self.config.max_source_positions
        kv_layout = {
            KV_CACHE: {"main": LabelLayout(
                spans=(Span("ids", "prefill", len(prompt_ids), len(prompt_ids), "text_inputs"),),
                growth=(("decode_loop", 1),),
            )},
            CROSS_KV_CACHE: {CONTEXT_LABEL: LabelLayout(
                spans=(Span("rows", "prefill", context, context, "encoder_states"),),
            )},
        }
        return ProcessPromptOutput({
            "audio_features": [audio_features],
            "text_inputs": [torch.tensor(prompt_ids, dtype=torch.long)],
        }, {"kv_layout": kv_layout})


    def get_loop_caps(self, **model_kwargs) -> dict[str, int]:
        return {"decode_loop": self.get_max_output_tokens(**model_kwargs)}


    def kv_labels(self) -> dict[str, dict[str, KVLabel]]:
        return {
            KV_CACHE: {"main": KVLabel(growth=(("decode", "decode_loop"),))},
            CROSS_KV_CACHE: {CONTEXT_LABEL: KVLabel()},
        }

    # -------------------------------------------------------------------
    # Model ABC: postprocess
    # -------------------------------------------------------------------

    def postprocess(
        self,
        output: torch.Tensor,
        modality: str,
        **kwargs,
    ) -> bytes:
        if modality == "text":
            return self._detokenizer.to_bytes(output.reshape(-1).tolist())
        raise ValueError(f"Unsupported modality for Whisper: {modality!r}")

    # -------------------------------------------------------------------
    # Model ABC: sharding
    # -------------------------------------------------------------------

    def get_default_sharding_config(self):
        from mstar.distributed.base import ShardingConfig

        return ShardingConfig(groups=[], tp_enabled_nodes={"decoder"}, shard_dim={})

    # -------------------------------------------------------------------
    # Model ABC: submodule loading
    # -------------------------------------------------------------------

    @classmethod
    def node_types(cls):
        from mstar.model.whisper.submodules import WhisperDecoderSubmodule, WhisperEncoderSubmodule
        return {"audio_encoder": WhisperEncoderSubmodule, "decoder": WhisperDecoderSubmodule}

    def create_submodule(self, node_name: str) -> NodeSubmodule | None:
        """Fresh semantic/lifecycle shell, with no reference weights or compute."""
        node_type = self.node_types().get(node_name)
        return None if node_type is None else node_type(self.config)

    def logical_weights(self):
        from mstar.model.whisper.weights import logical_weights
        if not hasattr(self, "_logical_weight_source"):
            self._logical_weight_source = logical_weights(self.local_dir, self.config)
        return self._logical_weight_source

    def quality_baselines(self) -> dict:
        import json
        from pathlib import Path
        descriptor = Path(__file__).with_name("public_q8_quality.json")
        return {"public-q8-kleidiai-v1": json.loads(descriptor.read_text())}


    def get_submodule(
        self, node_name: str, device: str = "cpu", tp_group=None,
        autocast_dtype: torch.dtype | None = None,
    ) -> NodeSubmodule | None:
        if node_name in self._submodule_cache:
            return self._submodule_cache[node_name]
        submodule = self._create_submodule(
            node_name, device, tp_group=tp_group, autocast_dtype=autocast_dtype,
        )
        logger.info("Successfully loaded Whisper submodule for %s", node_name)
        self._submodule_cache[node_name] = submodule
        return submodule

    def _create_submodule(
        self, node_name: str, device: str, tp_group=None,
        autocast_dtype: torch.dtype | None = None,
    ) -> NodeSubmodule | None:
        if node_name == "audio_encoder":
            return self._create_encoder_submodule(device)
        elif node_name == "decoder":
            return self._create_decoder_submodule(
                device, autocast_dtype=autocast_dtype, tp_group=tp_group,
            )
        return None

    def _create_encoder_submodule(self, device: str) -> NodeSubmodule:
        from transformers import WhisperConfig
        from transformers.models.whisper.modeling_whisper import WhisperEncoder

        from mstar.model.utils import (
            ModuleAndPrefix,
            load_weights_from_hf_shards,
        )

        hf_config = WhisperConfig.from_pretrained(self.local_dir)
        audio_encoder = WhisperEncoder._from_config(
            hf_config, attn_implementation="sdpa",
        )
        modules = [ModuleAndPrefix(audio_encoder, prefix="model.encoder")]
        load_weights_from_hf_shards(
            repo_dir=self.local_dir, modules=modules, device=device,
        )
        audio_encoder.eval()

        from mstar.model.whisper.reference import EncoderReference
        from mstar.model.whisper.submodules import WhisperEncoderSubmodule
        return WhisperEncoderSubmodule(config=self.config, reference_program=EncoderReference(audio_encoder))

    @staticmethod
    def _decoder_remap(name: str) -> str:
        # The shared Attention component names its output projection
        # ``o_proj``; the cross-attn module keeps HF's ``out_proj``.
        return name.replace("self_attn.out_proj", "self_attn.o_proj")

    def _create_decoder_submodule(
        self, device: str, autocast_dtype: torch.dtype | None = None, tp_group=None
    ) -> NodeSubmodule:
        from mstar.model.loader import WHISPER_STACKED_PARAMS, load_hf_weights
        from mstar.model.loader.iterators import iter_safetensors_shards
        from mstar.model.whisper.components.decoder import WhisperDecoderModel

        with torch.device("meta"):
            decoder = WhisperDecoderModel(self.config, comm_group=tp_group)
        # Cast on meta (no allocation) so to_empty allocates directly in the
        # target dtype instead of fp32-then-downcast.
        if autocast_dtype is not None:
            decoder = decoder.to(autocast_dtype)
        decoder.to_empty(device=device)

        weights = iter_safetensors_shards(
            self.local_dir, device=device, prefix="model.decoder.",
        )
        weights = ((k.removeprefix("model.decoder."), v) for k, v in weights)
        load_hf_weights(
            decoder,
            weights,
            stacked_params=WHISPER_STACKED_PARAMS,
            name_remapper=self._decoder_remap,
        )
        decoder.zero_missing_biases()
        decoder.eval()

        from mstar.model.whisper.reference import DecoderReference, WhisperDecoderReferenceSubmodule
        return WhisperDecoderReferenceSubmodule(
            config=self.config, reference_program=DecoderReference(decoder, self.config),
        )
