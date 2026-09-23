# SPDX-License-Identifier: MIT
"""HF normalization and immutable CSA2 topology, without GPU dependencies."""

from copy import deepcopy
from dataclasses import dataclass, replace
from enum import Enum
from itertools import groupby

from transformers import PretrainedConfig


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
    topk_owner: int | None = None
    candidate_owner: int | None = None
    # The contiguous run of layers whose attention indices one launch writes.
    # `indices.py` owns why it has to be contiguous.
    index_group_start: int = 0
    index_group_size: int = 1

    @property
    def shares_attention_input(self):
        """Whether anything but `wqkv_a` reads this layer's normed input.

        A compressor and an indexer both project the same tensor in BF16, and
        only a FULL layer builds the first or a FULL/REINDEX layer the second
        -- so everywhere else the norm has exactly one reader and can hand it
        the quantized pair instead of a tensor to quantize again.
        """
        return self.mode in (AttentionMode.FULL, AttentionMode.REINDEX)

    @property
    def produces_candidates(self):
        return self.candidate_owner == self.layer_id

    @property
    def candidate_source(self):
        """The layer whose blocks bound this one, or None if none do.

        None covers a layer with no candidate owner and the owner itself, so
        it answers consumption only -- production is the property above, and
        conflating the two has the first kind emitting blocks nobody asked for.
        """
        return None if self.produces_candidates else self.candidate_owner


def validate_native_quantization(config):
    """Reject a checkpoint whose quantization is not the one V4.1 reads.

    There is no second format to select between, so this returns nothing: the
    five shapes below are the only ones the kernels accept, and a checkpoint
    that declares anything else has no reader to fall back to.
    """
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
    if candidate >= 0 and (
        config.candidate_topk_blocks * config.candidate_block_size < config.index_topk
    ):
        # A row's selection count is read off its position rather than counted
        # (`_indptr_scan`), which holds only while the candidates can hold a
        # whole top-k. Below that a consumer scores fewer rows than it reserves
        # room for and leaves a hole in its own slice.
        raise ValueError("Candidate blocks must be able to hold a whole top-k")

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
                topk_owner,
                candidate if uses_candidates else None,
            )
        )
    # Backbone and draft are grouped apart. A run straddling the boundary would
    # hand a backbone layer a size counting draft layers the model never builds.
    return _tag_index_groups(result[:layers]) + _tag_index_groups(result[layers:])


def _tag_index_groups(specs):
    """Number each maximal run of layers that share an index build."""
    tagged = []
    for _, members in groupby(specs, key=lambda s: (s.ratio, s.kv_owner, s.topk_owner)):
        run = tuple(members)
        tagged += [
            replace(spec, index_group_start=run[0].layer_id, index_group_size=len(run))
            for spec in run
        ]
    return tuple(tagged)


class DeepseekV41TextConfig(PretrainedConfig):
    """The published text schema, with root token IDs and quantization preserved."""

    model_type = "deepseek_v41_text"

    def __init__(self, **kwargs):
        # transformers >= 5.13 runs RoPE standardization inside the base
        # __post_init__ (fired by super().__init__). It reads
        # self.max_position_embeddings eagerly -- it is the default argument to
        # rope_parameters.setdefault("original_max_position_embeddings", ...),
        # which Python evaluates even when the key is already present -- before the
        # base class has assigned it from kwargs, raising AttributeError. Seed it
        # first so standardization finds it; transformers 5.12 has no such
        # __post_init__ hook and simply reassigns the same value in super().
        if "max_position_embeddings" in kwargs:
            self.max_position_embeddings = kwargs["max_position_embeddings"]
        super().__init__(**kwargs)
        # transformers >= 5.13 folds rope_theta into rope_parameters when
        # rope_scaling is present and drops the top-level attribute the model
        # reads as config.rope_theta; restore it from the folded params (or raw
        # kwargs). transformers 5.12 keeps the attribute, so this is a no-op there.
        if not hasattr(self, "rope_theta"):
            rope_params = getattr(self, "rope_parameters", None) or {}
            self.rope_theta = kwargs.get("rope_theta", rope_params.get("rope_theta"))

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

    def validate_request(self, *, num_draft_tokens, multimodal_data):
        if num_draft_tokens and multimodal_data:
            raise ValueError(
                "DeepSeek-V4.1 DSpark currently supports text requests only"
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
    validate_native_quantization(text["quantization_config"])
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


def validate_speculative_config(config):
    """Supported DSpark deployment; model math and shared scheduling stay separate."""
    speculative = config.speculative_config
    if speculative is None:
        return
    if speculative.method != "dspark" or speculative.num_speculative_tokens != 5:
        raise ValueError("DeepSeek-V4.1 requires native DSpark with five draft tokens")
    if speculative.model is not None:
        from pathlib import Path

        if Path(speculative.model).resolve() != Path(config.model).resolve():
            raise ValueError("DeepSeek-V4.1 DSpark must use the target checkpoint")
    if config.tensor_parallel_size != 4:
        raise ValueError("DeepSeek-V4.1 DSpark is validated on TP4")
    if config.kv_cache_dtype != "bf16":
        raise ValueError("DeepSeek-V4.1 DSpark requires a BF16 KV cache")
    from atom.utils import envs

    if envs.ATOM_ENABLE_RELAXED_MTP:
        raise ValueError("DeepSeek-V4.1 DSpark requires strict target verification")
    if speculative.synthetic_acceptance_rates is not None:
        raise ValueError("DeepSeek-V4.1 DSpark requires real target verification")
    if config.dspark.confidence_schedule and (
        not config.dspark.ragged or not config.dspark.calibration_profile
    ):
        raise ValueError(
            "DeepSeek-V4.1 dynamic DSpark requires ragged verification and a calibration_profile"
        )


def validate_runtime_config(config):
    """Gate unimplemented execution modes before weights or pools are loaded."""
    from atom.config import CUDAGraphMode

    unsupported = []
    graph_mode = getattr(config.compilation_config, "cudagraph_mode", None)
    for name, enabled in (
        (
            "CUDAGraph mode (use FULL, PIECEWISE or enforce_eager=True)",
            not config.enforce_eager
            and graph_mode not in (CUDAGraphMode.FULL, CUDAGraphMode.PIECEWISE),
        ),
        (
            "torch.compile level (use 0 or 3)",
            config.compilation_config.level not in (0, 3),
        ),
        (
            "level 3 CUDA Graph mode (use FULL or enforce_eager=True)",
            config.compilation_config.level == 3
            and not config.enforce_eager
            and graph_mode != CUDAGraphMode.FULL,
        ),
        ("pipeline parallel", config.pipeline_parallel_size != 1),
        (
            "context parallel",
            config.prefill_context_parallel_size != 1
            or config.decode_context_parallel_size != 1,
        ),
        (
            "data parallel",
            config.parallel_config.data_parallel_size != 1
            or config.enable_dp_attention,
        ),
        ("TBO", config.enable_tbo or config.enable_tbo_decode),
        ("KV transfer", bool(config.kv_transfer_config) or config.enable_rapidserve),
        ("plugin mode", config.plugin_config is not None),
        ("online quantization", config.online_quant_config is not None),
        ("EPLB", config.eplb_enable),
        (
            # The plane a paged scorer reads. Only that scorer is left, so the
            # format is the runtime's rather than a choice; the main pool stays
            # independent of it and takes bf16 or fp4 either way.
            "an index plane other than fp8",
            config.index_cache_dtype != "fp8",
        ),
        (
            "a KV cache other than bf16 or fp4",
            config.kv_cache_dtype not in ("bf16", "fp4"),
        ),
    ):
        if enabled:
            unsupported.append(name)
    if unsupported:
        raise ValueError(
            "DeepSeek-V4.1 runtime does not support " + ", ".join(unsupported)
        )
    if config.kv_cache_block_size % 2:
        raise ValueError("DeepSeek-V4.1 PAGE token count must be even")
    validate_speculative_config(config)
