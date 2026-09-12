# SPDX-License-Identifier: MIT
"""HF normalization and immutable CSA2 topology, without GPU dependencies."""

from copy import deepcopy
from dataclasses import dataclass
from enum import Enum

from transformers import PretrainedConfig


class IndexTieBreak(str, Enum):
    SMALL_POSITION = "small_position"
    LARGE_POSITION = "large_position"


class AttentionMode(str, Enum):
    WINDOW = "window"
    FULL = "full"
    REINDEX = "reindex"
    REUSE = "reuse"


@dataclass(frozen=True)
class LayerAttentionSpec:
    layer_id: int
    ratio: int
    mode: AttentionMode
    kv_owner: int | None = None
    index_key_owner: int | None = None
    topk_owner: int | None = None
    candidate_owner: int | None = None


@dataclass(frozen=True)
class NativeQuantization:
    weight_block_size: tuple[int, int]
    activation_block_size: int
    scale_format: str
    expert_dtype: str

    @classmethod
    def from_dict(cls, config):
        if config.get("quant_method") != "fp8":
            raise ValueError("DeepSeek-V4.1 requires the native FP8 checkpoint format")
        if config.get("weight_block_size") not in ([32, 32], (32, 32)):
            raise ValueError("DeepSeek-V4.1 weight_block_size must be [32, 32]")
        if config.get("scale_fmt") != "ue8m0":
            raise ValueError("DeepSeek-V4.1 dense scales must be ue8m0")
        if config.get("activation_scheme") != "dynamic":
            raise ValueError("DeepSeek-V4.1 requires dynamic per-32 FP8 activations")
        if config.get("expert_dtype") != "fp4":
            raise ValueError("DeepSeek-V4.1 routed experts must use native fp4 weights")
        return cls((32, 32), 32, "ue8m0", "fp4")


def _layer_ids(config, field, stop):
    values = tuple(getattr(config, field))
    if any(type(value) is not int or not 0 <= value < stop for value in values):
        raise ValueError(f"{field} must contain layer IDs in [0, {stop})")
    if tuple(sorted(set(values))) != values:
        raise ValueError(f"{field} must be sorted and contain no duplicates")
    return values


def build_attention_topology(config) -> tuple[LayerAttentionSpec, ...]:
    """Resolve each consumer to its physical KV and query-index owners once."""
    layers = config.num_hidden_layers
    total_layers = layers + config.num_nextn_predict_layers
    ratios = tuple(config.compress_ratios)
    if len(ratios) != total_layers:
        raise ValueError(
            f"compress_ratios must cover {total_layers} backbone/draft layers"
        )
    if any(type(ratio) is not int or ratio not in (0, 1, 2) for ratio in ratios):
        raise ValueError("CSA2 compress_ratios must contain only 0, 1 or 2")
    if any(ratios[layers:]):
        raise ValueError("V4.1 draft layers must use window-only attention")
    kv_sources = _layer_ids(config, "kv_source_layer_ids", layers)
    index_sources = _layer_ids(config, "index_source_layer_ids", layers)
    if not set(kv_sources).issubset(index_sources):
        raise ValueError("Every KV source must also be an index source")
    if any(ratios[layer] == 0 for layer in index_sources):
        raise ValueError("A window-only layer cannot own global KV or indices")
    candidate = config.candidate_source_layer_id
    if candidate != -1 and candidate not in index_sources:
        raise ValueError("candidate_source_layer_id must be an index source or -1")
    if candidate >= 0 and (
        config.candidate_topk_blocks <= 0 or config.candidate_block_size <= 0
    ):
        raise ValueError("Candidate block count and size must be positive")

    result = []
    kv_owner = topk_owner = candidate_kv_owner = None
    for layer_id, ratio in enumerate(ratios):
        if ratio == 0:
            result.append(LayerAttentionSpec(layer_id, ratio, AttentionMode.WINDOW))
            continue
        if layer_id in kv_sources:
            kv_owner = layer_id
        if kv_owner is None or ratios[kv_owner] != ratio:
            raise ValueError(
                f"Layer {layer_id} has no preceding KV owner with ratio {ratio}"
            )
        if layer_id in index_sources:
            topk_owner = layer_id
        if topk_owner is None or topk_owner < kv_owner:
            raise ValueError(f"Layer {layer_id} has no index source for its KV owner")
        if layer_id == candidate:
            candidate_kv_owner = kv_owner
        uses_candidates = candidate >= 0 and layer_id >= candidate
        if uses_candidates and candidate_kv_owner != kv_owner:
            raise ValueError(
                f"Layer {layer_id} cannot reuse candidates from another KV owner"
            )
        mode = (
            AttentionMode.FULL
            if layer_id == kv_owner
            else (
                AttentionMode.REINDEX if layer_id == topk_owner else AttentionMode.REUSE
            )
        )
        result.append(
            LayerAttentionSpec(
                layer_id,
                ratio,
                mode,
                kv_owner,
                kv_owner,
                topk_owner,
                candidate if uses_candidates else None,
            )
        )
    return tuple(result)


class DeepseekV41TextConfig(PretrainedConfig):
    """The published text schema, with root token IDs and quantization preserved."""

    model_type = "deepseek_v41_text"

    def __init__(self, index_topk_tie_break="small_position", **kwargs):
        super().__init__(**kwargs)
        try:
            self.index_topk_tie_break = IndexTieBreak(index_topk_tie_break).value
        except ValueError as error:
            raise ValueError(
                "index_topk_tie_break must be small_position or large_position"
            ) from error

    def validate_parallelism(self, tensor_parallel_size, expert_parallel_size=1):
        if tensor_parallel_size <= 0 or expert_parallel_size <= 0:
            raise ValueError("Parallel sizes must be positive")
        for field in ("num_attention_heads", "index_n_heads", "o_groups"):
            if getattr(self, field) % tensor_parallel_size:
                raise ValueError(f"{field} must be divisible by tensor parallel size")
        if self.n_routed_experts % expert_parallel_size:
            raise ValueError(
                "n_routed_experts must be divisible by expert parallel size"
            )
        if self.moe_intermediate_size % tensor_parallel_size:
            raise ValueError(
                "moe_intermediate_size must be divisible by tensor parallel size"
            )


class DeepseekV41VisionConfig(PretrainedConfig):
    model_type = "deepseek_v41_vision"


class DeepseekV41Config(PretrainedConfig):
    model_type = "deepseek_v41"


def normalize_hf_config(raw: dict) -> DeepseekV41TextConfig:
    """Keep ATOM's flat text interface and an independent full multimodal config.

    The full config uses separate objects, avoiding a text->root->text cycle in
    HF serialization. GPU topology objects are not stored inside the HF config.
    """
    if raw.get("model_type") != "deepseek_v41":
        raise ValueError("Expected model_type=deepseek_v41")
    text = deepcopy(raw.get("text_config", {}))
    vision = deepcopy(raw.get("vision_config", {}))
    if text.get("model_type") != "deepseek_v41_text" or not vision:
        raise ValueError("DeepSeek-V4.1 requires text_config and vision_config")
    if raw.get("architectures") != ["DeepseekV41ForCausalLM"]:
        raise ValueError("Unexpected DeepSeek-V4.1 model architecture")
    for field in (
        "architectures",
        "dtype",
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
        "image_token_id",
        "quantization_config",
    ):
        if field in raw:
            text.setdefault(field, deepcopy(raw[field]))
    NativeQuantization.from_dict(text["quantization_config"])
    text.pop("model_type")
    config = DeepseekV41TextConfig(**text)
    required_positive = (
        "hidden_size",
        "vocab_size",
        "num_hidden_layers",
        "num_attention_heads",
        "head_dim",
        "q_lora_rank",
        "o_lora_rank",
        "o_groups",
        "index_n_heads",
        "index_head_dim",
        "index_topk",
        "sliding_window",
        "moe_intermediate_size",
        "n_routed_experts",
        "num_experts_per_tok",
        "hc_mult",
        "hc_sinkhorn_iters",
        "max_position_embeddings",
        "rms_norm_eps",
        "hc_eps",
    )
    for field in required_positive:
        if getattr(config, field, 0) <= 0:
            raise ValueError(f"{field} must be positive")
    if config.num_key_value_heads != 1 or config.n_shared_experts != 1:
        raise ValueError("V4.1 requires one KV head and one shared expert")
    if config.num_attention_heads % config.o_groups:
        raise ValueError("num_attention_heads must be divisible by o_groups")
    if config.num_experts_per_tok > config.n_routed_experts:
        raise ValueError("num_experts_per_tok exceeds the routed expert count")
    rope_dim = config.qk_rope_head_dim
    if (
        rope_dim <= 0
        or rope_dim % 2
        or rope_dim > min(config.head_dim, config.index_head_dim)
    ):
        raise ValueError(
            "qk_rope_head_dim must be even and fit both attention and index heads"
        )
    if config.num_nextn_predict_layers < 0:
        raise ValueError("num_nextn_predict_layers must be nonnegative")
    engram_layers = _layer_ids(config, "engram_layer_ids", config.num_hidden_layers)
    if len(engram_layers) != len(config.engram_num_embeddings):
        raise ValueError("Engram layer IDs and table sizes must have equal lengths")
    if any(rows <= 0 for rows in config.engram_num_embeddings):
        raise ValueError("Engram table sizes must be positive")
    build_attention_topology(config)

    # The full model config is data, not a fallback AutoConfig lookup. The
    # installed Transformers version need not know this new architecture.
    root_data = deepcopy(raw)
    root_data["text_config"] = DeepseekV41TextConfig(**deepcopy(text))
    root_data["vision_config"] = DeepseekV41VisionConfig(**vision)
    config._multimodal_config = DeepseekV41Config(**root_data)
    return config
