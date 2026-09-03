# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import enum
import fnmatch
import hashlib
import logging
import os
import re
from contextlib import contextmanager
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any, ClassVar

import torch
from torch.distributed import ProcessGroup, ReduceOp
from transformers import AutoConfig, GenerationConfig, PretrainedConfig

# plugin-related utilities
from atom.plugin import is_plugin_mode, is_vllm
from atom.plugin.config import PluginConfig
from atom.quant_spec import (
    LayerQuantConfig,
    get_quant_parser,
)
from atom.utils import envs, get_open_port
from atom.utils.distributed.utils import stateless_init_torch_distributed_process_group

if TYPE_CHECKING:
    # Annotation only. Importing AITER here would put a GPU kernel build behind
    # `import atom.config`, which is what `atom.quant_spec` defers on purpose.
    from aiter import QuantType

logger = logging.getLogger("atom")


@dataclass
class KVCacheTensor:
    """
    A class for specifying how the workers should initialize the KV cache.
    """

    layer_num: int
    k_cache: torch.Tensor = field(default_factory=lambda: torch.tensor([]))
    v_cache: torch.Tensor = field(default_factory=lambda: torch.tensor([]))
    k_scale: torch.Tensor = None
    v_scale: torch.Tensor = None
    # DSA sparse layers (GLM-5.2 / DeepSeek-V3.2): indexer key cache, block-major
    # ``(num_blocks, block_size, aligned_index_dim)``. Omitted for non-DSA layers.
    index_cache: torch.Tensor | None = None
    # ReplaySSM record buffers for linear-attention layers: this layer's slice
    # of the (k, u, g) pools.  None for every other attention type.  Carried
    # here because the layer-id -> linear-attn-index mapping already lives in
    # the builder's `build_kv_cache_tensor`.
    replay_buf_k: torch.Tensor = None
    replay_buf_u: torch.Tensor = None
    replay_buf_g: torch.Tensor = None
    # True when the tensors above are a hybrid's PER-REQUEST state (GDN/KDA
    # recurrent state) rather than paged KV. Only the GDN and KDA linear-attn
    # forwards set it (`gdn_attn.py`, `kimi_mla_gdn_attn.py`, and the rtpllm/
    # sglang plugin twins); no DeepSeek-V4 path does. They belong
    # in ``kv_cache_data`` -- the linear-attention forward reads them from there
    # -- but are addressed by request slot, so no block-addressed mover may
    # touch them. The dense codec skips them; the state tier reaches the same
    # bytes through ``state_entry_views``.
    per_request_state: bool = False


@dataclass
class KVCacheConfig:
    """
    The KV cache configuration of a model.
    """

    kv_cache_tensors: list[KVCacheTensor]


class CUDAGraphMode(enum.Enum):
    """Constants for the cudagraph mode in CompilationConfig.
    Meanwhile, the subset enum `NONE`, `PIECEWISE` and `FULL` are also
    treated as concrete runtime mode for cudagraph runtime dispatching.
    """

    NONE = 0
    PIECEWISE = 1
    FULL = 2
    FULL_DECODE_ONLY = (FULL, NONE)
    FULL_AND_PIECEWISE = (FULL, PIECEWISE)
    # AF_PIECEWISE ("attention/FFN-wise"): PIECEWISE + attention core in its own
    # cudagraph (DSpark). Tuple so decode/mixed_mode resolve to PIECEWISE (existing
    # == PIECEWISE checks treat it as such); extra capture gated by is_attn_ffn_piecewise().
    AF_PIECEWISE = (PIECEWISE, PIECEWISE)

    def decode_mode(self) -> "CUDAGraphMode":
        return CUDAGraphMode(self.value[0]) if self.separate_routine() else self

    def mixed_mode(self) -> "CUDAGraphMode":
        return CUDAGraphMode(self.value[1]) if self.separate_routine() else self

    def is_attn_ffn_piecewise(self) -> bool:
        """True only for AF_PIECEWISE — gates the extra attention-core cudagraph
        (attention/FFN-wise capture) on top of the standard piecewise pieces."""
        return self is CUDAGraphMode.AF_PIECEWISE

    def requires_piecewise_compilation(self) -> bool:
        return (
            self.decode_mode() == CUDAGraphMode.PIECEWISE
            or self.mixed_mode() == CUDAGraphMode.PIECEWISE
        )

    def max_cudagraph_mode(self) -> "CUDAGraphMode":
        return CUDAGraphMode(max(self.value)) if self.separate_routine() else self

    def has_full_cudagraphs(self) -> bool:
        return self.max_cudagraph_mode() == CUDAGraphMode.FULL

    def separate_routine(self) -> bool:
        return isinstance(self.value, tuple)


class CompilationLevel:
    # constants for the levels of the compilation process
    NO_COMPILATION = 0
    DYNAMO_AS_IS = 1
    DYNAMO_ONCE = 2
    PIECEWISE = 3


@dataclass
class CompilationConfig:
    level: int = 0
    """The level of compilation:

    - 0: no compilation.
    - 1: dynamo as is.
    - 2: dynamo once.
    - 3: piecewise compilation."""
    # use_cudagraph: bool = field(default_factory=lambda: 0)

    use_cudagraph: bool = True

    local_cache_dir: str = field(default=None, init=False)  # type: ignore
    # cudagraph_capture_sizes: Optional[list[int]] = [1,2,4,8]
    cudagraph_capture_sizes: list[int] | None = None

    cuda_graph_sizes: list[int] = field(default_factory=list)
    """Cuda graph capture sizes
    1. if none provided, then default set to [min(max_num_seqs * 2, 512)]
    2. if one value is provided, then the capture list would follow the
    pattern: [1, 2, 4] + [i for i in range(8, cuda_graph_sizes + 1, 8)]
    3. more than one value (e.g. 1 2 128) is provided, then the capture list
    will follow the provided list."""
    debug_dump_path: str = ""
    """The path to dump the debug information."""

    """custom ops that are disabled"""
    traced_files: set[str] = field(default_factory=set, init=False)

    cache_dir: str = ""

    use_inductor: bool = True

    # CudaGraph compilation
    cudagraph_mode: CUDAGraphMode | None = None
    """
    The mode of the cudagraph:

    - NONE, no cudagraph capture.
    - PIECEWISE. (v1 default)
    - FULL.
    - FULL_DECODE_ONLY.
    - FULL_AND_PIECEWISE.
    - AF_PIECEWISE.

    AF_PIECEWISE ("attention/FFN-wise") mode: PIECEWISE where the attention
    core is ALSO captured into its own cudagraph with zero-copy public buffers
    (DeepSeek-V4 DSpark), so small-batch decode is all-replay with no eager
    attention gap between the dense pieces. Falls back to plain PIECEWISE
    behavior for models that don't implement the attention-core capture.

    PIECEWISE mode build piecewise cudagraph only, keeping the cudagraph
    incompatiable ops (i.e. some attention ops) outside the cudagraph
    for general flexibility.
    This is the default mode.

    FULL mode: Capture full cudagraph for all batches. Can be good for small
    models or workloads with small prompts; not supported by many backends.
    Generally for performance FULL_AND_PIECEWISE is better.

    FULL_DECODE_ONLY mode: Capture full cudagraph for decode batches only.
    Mixed prefill-decode batches are run without cudagraphs. Can be good for
    decode instances in a P/D setup where prefill is not as important so we
    can save some memory.

    FULL_AND_PIECEWISE mode: Capture full cudagraph for decode batches and
    piecewise cudagraph for prefill and mixed prefill-decode batches.
    This is like the most performant mode for most models.

    Currently, the cudagraph mode is only used for the v1 engine.
    Note that the cudagraph logic is generally orthogonal to the
    compilation logic. While piecewise cudagraphs require piecewise
    compilation (level=PIECEWISE and non-empty splitting_ops), full
    cudagraphs are supported with and without compilation.

    Warning: This flag is new and subject to change in addition
    more modes may be added.
    """

    compilation_time: float = field(default=0.0, init=False)

    splitting_ops: list[str] | None = None
    """A list of ops to split the full graph into subgraphs, used in piecewise
    compilation."""

    # splitting_ops: Optional[list[str]] = field(default_factory=list)

    cudagraph_copy_inputs: bool = False
    """Whether to copy input tensors for
    cudagraph. If the caller can guarantee that the same input buffers
    are always used, it can set this to False. Otherwise, it should
    set this to True, and the compiler will copy the input to an
    internally managed buffer. Default is False.
    Note that this flag is only effective when cudagraph_mode is PIECEWISE.
    """

    inductor_compile_config: dict = field(default_factory=dict)
    """Additional configurations for inductor.
    - None: use default configurations."""

    compile_sizes: list[int | str] | None = None
    """Sizes to compile for inductor. In addition
    to integers, it also supports "cudagraph_capture_sizes" to
    specify the sizes for cudagraph capture."""

    static_forward_context: dict[str, Any] = field(default_factory=dict, init=False)

    def init_with_cudagraph_sizes(self) -> None:
        """To complete the initialization of config,
        we need to know the cudagraph sizes."""
        computed_compile_sizes = []
        if self.compile_sizes is not None:
            # de-duplicate the sizes provided by the config
            self.compile_sizes = list(set(self.compile_sizes))
            for x in self.compile_sizes:
                if isinstance(x, str):
                    assert x == "cudagraph_capture_sizes", (
                        "Unrecognized size type in compile_sizes, "
                        f"expect 'cudagraph_capture_sizes', got {x}"
                    )
                    computed_compile_sizes.extend(self.cudagraph_capture_sizes)
                else:
                    assert isinstance(x, int)
                    computed_compile_sizes.append(x)
        self.compile_sizes = computed_compile_sizes  # type: ignore

    def __post_init__(self):
        if self.level not in {0, 1, 2, 3}:
            raise ValueError("level must in 0-3")
        if not self.cuda_graph_sizes:
            self.cuda_graph_sizes = [512]

    def compute_hash(self) -> str:
        """
        WARNING: Whenever a new field is added to this config,
        ensure that it is included in the factors list if
        it affects the computation graph.

        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        """
        factors: list[Any] = []
        factors.append(self.level)
        factors.append(self.use_cudagraph)
        factors.append(self.local_cache_dir)
        factors.append(self.cudagraph_capture_sizes)
        factors.append(self.cuda_graph_sizes)

        return hashlib.sha256(str(factors).encode()).hexdigest()

    def set_splitting_ops_for_v1(self):
        # NOTE: this function needs to be called only when level is
        # CompilationLevel.PIECEWISE
        assert self.level == CompilationLevel.PIECEWISE, (
            "set_splitting_ops_for_v1 should only be called when "
            "level is CompilationLevel.PIECEWISE"
        )

        if self.splitting_ops is None:
            self.splitting_ops = [
                "aiter.unified_attention_with_output",
                "aiter.mla_attention",
                "aiter.atom_vllm_mha_attention",
                "aiter.atom_vllm_mla_attention",
            ]


class QuantizationConfig:
    """Model-wide quantization configuration.

    API:
    - ``get_layer_quant_config(prefix)`` -> :class:`LayerQuantConfig`
    - ``global_quant_config`` property -> :class:`LayerQuantConfig`
    - ``quant_type``, ``quant_dtype``, ``is_dynamic`` convenience properties
    """

    def __init__(
        self,
        config: PretrainedConfig = None,
        online_quant_config: dict | None = None,
    ):
        if config is None:
            self.torch_dtype = torch.bfloat16
            self.hf_quant_config = None
            self.global_spec: LayerQuantConfig = LayerQuantConfig()
            self.layer_pattern_specs: list[tuple[str, LayerQuantConfig]] = []
            self.exclude_layers: list[str] = []
            self.quant_method = ""
            self.online_quant = False
            self.online_quant_config_raw = None
            self.online_global_spec: LayerQuantConfig = LayerQuantConfig()
            self.online_layer_pattern_specs: list[tuple[str, LayerQuantConfig]] = []
            self.online_exclude_layers: list[str] = []
            return

        # Some HF configs set torch_dtype=None; normalize to bf16 default.
        self.torch_dtype = getattr(config, "torch_dtype", None) or torch.bfloat16
        self.hf_quant_config = getattr(config, "quantization_config", None)

        if self.hf_quant_config is None:
            self.global_spec = LayerQuantConfig.no_quant(self.torch_dtype)
            self.layer_pattern_specs = []
            self.exclude_layers = []
            self.quant_method = ""
        else:
            self.quant_method = self.hf_quant_config.get("quant_method", "")

        # Online quantization: re-quantize float / FP8 / MXFP4 / MXFP8 / Quark
        # models at load time.
        self.online_quant = False
        self.online_quant_config_raw = online_quant_config
        self.online_global_spec: LayerQuantConfig = LayerQuantConfig()
        self.online_layer_pattern_specs: list[tuple[str, LayerQuantConfig]] = []
        self.online_exclude_layers: list[str] = []
        if online_quant_config and self.quant_method in [
            "",
            "fp8",
            "mxfp4",
            "mxfp8",
            "quark",
            "compressed-tensors",
        ]:
            self.online_quant = True
            if self.quant_method == "compressed-tensors":
                logger.warning(
                    "Online quant with compressed-tensors is not fully supported. "
                    "Be careful about the online quant config setting when launching "
                    "the server."
                )
            online_parser = get_quant_parser("online_quant")
            online_parsed_quant_config = online_parser.parse(online_quant_config)
            self.online_global_spec = online_parsed_quant_config.global_spec
            self.online_layer_pattern_specs = (
                online_parsed_quant_config.layer_pattern_specs
            )
            self.online_exclude_layers = list(online_parsed_quant_config.exclude_layers)

        if self.quant_method == "":
            return
        # Use the parser registry to build a structured ParsedQuantConfig
        parser = get_quant_parser(self.quant_method)
        parsed_quant_config = parser.parse(self.hf_quant_config)
        self.global_spec = parsed_quant_config.global_spec
        self.layer_pattern_specs = parsed_quant_config.layer_pattern_specs
        self.exclude_layers = list(parsed_quant_config.exclude_layers)

    # -- typed API (preferred) ----------------------------------------------

    @property
    def global_quant_config(self) -> LayerQuantConfig:
        """Alias for ``global_spec``."""
        return self.global_spec

    def get_layer_quant_config(
        self,
        layer_name: str,
        use_online_quant: bool = False,
        *,
        check_children: bool = False,
    ) -> LayerQuantConfig:
        """Return the :class:`LayerQuantConfig` for *layer_name*.

        Resolution order:
        1. Check exclude list -> ``LayerQuantConfig.no_quant()``.
        2. fnmatch-style pattern match in ``layer_pattern_specs``.
        3. Fall back to ``global_spec``.
        """
        if use_online_quant:
            layer_pattern_specs = self.online_layer_pattern_specs
            global_spec = self.online_global_spec
            exclude_layers = self.online_exclude_layers
        else:
            layer_pattern_specs = self.layer_pattern_specs
            global_spec = self.global_spec
            exclude_layers = self.exclude_layers

        # 1. Exclude list
        if self._is_excluded(layer_name, exclude_layers, check_children=check_children):
            return LayerQuantConfig(quant_dtype=self.torch_dtype)

        # 2. Pattern match
        for pattern, spec in layer_pattern_specs:
            if "*" not in pattern:
                if layer_name in pattern:
                    return spec
            elif fnmatch.fnmatch(layer_name, pattern):
                return spec

        # 3. Global default
        return global_spec

    # -- convenience properties (delegate to global_spec) ---------------------

    @property
    def quant_type(self) -> "QuantType":
        return self.global_spec.quant_type

    @property
    def quant_dtype(self) -> torch.dtype:
        return self.global_spec.quant_dtype

    @property
    def is_dynamic(self) -> bool:
        return self.global_spec.is_dynamic

    # -- other methods ------------------------------------------------------

    def compute_hash(self) -> str:
        """
        WARNING: Whenever a new field is added to this config,
        ensure that it is included in the factors list if
        it affects the computation graph.

        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        """
        factors: list[Any] = []
        factors.append(self.global_spec)
        factors.append(self.layer_pattern_specs)
        factors.append(self.exclude_layers)
        if self.online_quant:
            factors.append(self.online_layer_pattern_specs)
            factors.append(self.online_global_spec)
            factors.append(self.online_exclude_layers)
        hash_value = hashlib.sha256(str(factors).encode()).hexdigest()
        return hash_value

    def get_name(self):
        """Returns the quantization method name."""
        return self.quant_method

    # -- internal helpers ---------------------------------------------------

    def _is_excluded(
        self,
        layer_name: str,
        exclude_layers: list[str] | None = None,
        *,
        check_children: bool = False,
    ) -> bool:
        if exclude_layers is None:
            exclude_layers = self.exclude_layers
        if layer_name is None or not exclude_layers:
            return False
        prefix = layer_name + "."
        for ignore_str in exclude_layers:
            if self._matches_exclude(layer_name, ignore_str):
                return True
            # When check_children is True, also match if any exclude entry
            # is a child of layer_name.  This is needed by container modules
            # like FusedMoE whose prefix (e.g. "mtp.layers.60.mlp.experts")
            # is a parent of the leaf-level exclude entries (e.g.
            # "mtp.layers.60.mlp.experts.0.gate_up_proj").
            if check_children and ignore_str.startswith(prefix):
                return True
        return False

    @staticmethod
    def _matches_exclude(
        layer_name: str, ignore_str: str, check_contains: bool = False
    ) -> bool:
        """Match the target string or regular expression.

        Supports exact match, prefix match (layer under an excluded module),
        fnmatch glob patterns (``*`` / ``?``), and ``re:`` regex patterns.
        """
        if ignore_str.startswith("re:"):
            pattern = ignore_str[3:]
            if re.search(pattern, layer_name):
                return True
        elif "*" in ignore_str or "?" in ignore_str:
            # Glob pattern: match exact or as prefix of deeper sub-modules
            if fnmatch.fnmatch(layer_name, ignore_str):
                return True
            if fnmatch.fnmatch(layer_name, ignore_str + ".*"):
                return True
        elif check_contains:
            return layer_name.lower() in ignore_str.lower()
        else:
            # Exact match or prefix match (e.g. "lm_head" excludes "lm_head.weight")
            if layer_name == ignore_str or layer_name.startswith(ignore_str + "."):
                return True
        return False

    def apply_exclude_name_mapping(self, mapping: dict[str, str]):
        if not mapping or not self.exclude_layers:
            return
        new_excludes = []
        for name in self.exclude_layers:
            for old, new in mapping.items():
                name = name.replace(old, new)
            new_excludes.append(name)
        self.exclude_layers = list(dict.fromkeys(new_excludes))

    def apply_default_exclude_layers(self, excludes: list[str]):
        if not excludes:
            return
        for exclude in excludes:
            if exclude not in self.exclude_layers:
                self.exclude_layers.append(exclude)

    def remap_layer_name(
        self,
        hf_config: PretrainedConfig,
        packed_modules_mapping: dict | None = None,
        weights_mapper={},
        quant_exclude_name_mapping: dict[str, str] | None = None,
    ):
        model_type = hf_config.model_type
        self.packed_modules_mapping = (
            packed_modules_mapping if packed_modules_mapping is not None else {}
        )
        # for special models
        if model_type in ("deepseek_mtp", "deepseek_v3", "kimi_k2", "glm_moe_dsa"):
            if hasattr(hf_config, "q_lora_rank") and hf_config.q_lora_rank is not None:
                self.packed_modules_mapping = {
                    "q_a_proj": ("fused_qkv_a_proj", 0),
                    "kv_a_proj_with_mqa": ("fused_qkv_a_proj", 1),
                    "gate_proj": ("gate_up_proj", 0),
                    "up_proj": ("gate_up_proj", 1),
                }
            else:
                self.packed_modules_mapping = {
                    "gate_proj": ("gate_up_proj", 0),
                    "up_proj": ("gate_up_proj", 1),
                }
        elif model_type == "qwen3_moe" or model_type == "qwen3_next":
            if getattr(hf_config, "mlp_only_layers", []):
                self.packed_modules_mapping["gate_up_proj"] = ["gate_proj", "up_proj"]

        if weights_mapper:
            self.exclude_layers = [
                weights_mapper._map_name(name) for name in self.exclude_layers
            ]

        # remap
        def _remap_layer_name(name: str) -> list[str]:
            for packed_key, packed_value in self.packed_modules_mapping.items():
                # for self_attn.up_proj and self_attn.gate_up_proj
                # up_proj in gate_up_proj, so add prefix .
                match_key = (
                    packed_key if packed_key.startswith(".") else f".{packed_key}"
                )
                if match_key in name:
                    if isinstance(packed_value, list):
                        # "gate_up_proj" → ["gate_proj", "up_proj"]
                        return [
                            name.replace(packed_key, part, 1) for part in packed_value
                        ]
                    else:
                        # "gate_proj" → ("gate_up_proj", 0)
                        packed_remap_part, _ = packed_value
                        return [name.replace(packed_key, packed_remap_part, 1)]
            return [name]

        new_pattern_specs = []
        for pattern, spec in self.layer_pattern_specs:
            for remapped in _remap_layer_name(pattern):
                new_pattern_specs.append((remapped, spec))
        self.layer_pattern_specs = new_pattern_specs

        new_exclude = []
        for name in self.exclude_layers:
            new_exclude.extend(_remap_layer_name(name))
        self.exclude_layers = list(dict.fromkeys(new_exclude))
        if self.online_quant:
            new_online_pattern_specs = []
            for pattern, spec in self.online_layer_pattern_specs:
                for remapped in _remap_layer_name(pattern):
                    new_online_pattern_specs.append((remapped, spec))
            self.online_layer_pattern_specs = new_online_pattern_specs

            new_online_exclude = []
            for name in self.online_exclude_layers:
                new_online_exclude.extend(_remap_layer_name(name))
            self.online_exclude_layers = list(dict.fromkeys(new_online_exclude))

        # Apply model-declared HF-name to ATOM-path translations for exclude entries.
        # Models that have a mismatch between their HF quant config names and ATOM
        # module paths declare `quant_exclude_name_mapping` as a class attribute.
        if quant_exclude_name_mapping:
            self.apply_exclude_name_mapping(quant_exclude_name_mapping)


# Rows per block that `deepgemm_fp8_paged_mqa_logits` requires to stay in its
# preshuffled layout, which is the only layout it computes correctly -- with
# `Preshuffle=False` it disagrees with the flat `fp8_mqa_logits` kernel by ~100%
# at every block size, and aiter's assert guards only the preshuffle side. Any
# cache that kernel pages over must therefore hold a multiple of this many rows
# per block. Sizing constants belong to whoever enforces them, and the block
# size is set here.
_MQA_LOGITS_PRESHUFFLE_ROWS = 16


def glm5_kpool_block_size(index_kpool: int) -> int:
    """Tokens per KV block that lets GLM-5.3's pooled index cache be exact.

    A block of B tokens needs ``B // index_kpool`` index rows, and that count
    must be a multiple of `_MQA_LOGITS_PRESHUFFLE_ROWS`. The smallest B that
    satisfies it is the product, and the smallest is what we want: a larger
    block only adds paging waste, while a smaller one forces the cache to be
    padded back up to one row per token -- which is the whole cost being
    removed here.
    """
    return index_kpool * _MQA_LOGITS_PRESHUFFLE_ROWS


_CONFIG_REGISTRY: dict[str, str] = {
    "deepseek_v32": "deepseek_v3",
    "deepseek_v4": "deepseek_v3",  # V4 reuses V3 schema; V4-specific fields
    # (compress_ratios, num_hash_layers, hc_mult, swiglu_limit, ...) flow
    # through as extra config attrs and are read in DeepseekV4Args.from_hf_config.
    "glm_moe_dsa": "deepseek_v3",  # GLM 5.0 MoE, structure similar to DeepSeek v3.2
    "kimi_k2": "deepseek_v3",
    "qwen3_next": "qwen3_next",
}


# model_types that exist only as speculative-draft checkpoints. transformers
# has no config class for them, so AutoConfig.from_pretrained raises; load them
# as a bare PretrainedConfig instead (see get_hf_config).
_PLAIN_CONFIG_MODEL_TYPES: frozenset[str] = frozenset({"k3_dspark"})

_MULTIMODAL_MODEL_TYPES: dict[str, str] = {
    # Maps multimodal model_type -> key in config_dict for the text sub-config
    "kimi_k3": "text_config",
    "kimi_k25": "text_config",
    "qwen3_5": "text_config",
    "qwen3_5_moe": "text_config",
    "mistral3": "text_config",
    "glm5_next": "text_config",  # GLM-5.3-Flash: text-only hybrid KDA/DSA runtime
}

# Text sub-config model_types that this image's transformers has no class for.
# Loaded as a bare PretrainedConfig; the ATOM model normalizes the aliases it
# needs at construction time.
_PLAIN_TEXT_CONFIG_MODEL_TYPES: frozenset[str] = frozenset(
    {"kimi_linear", "glm5_next_text"}
)

# multimodal models fully supported by plugin mode
_PLUGIN_SUPPORTED_MULTIMODAL_MODELS: set[str] = {
    "kimi_k25",
    "qwen3_5",
    "qwen3_5_moe",
}


def get_hf_config(model: str, trust_remote_code: bool = False) -> PretrainedConfig:
    config_dict, _ = PretrainedConfig.get_config_dict(
        model,
    )
    model_type = config_dict.get("model_type")

    def _get_hf_token() -> str | None:
        token = os.getenv("HF_TOKEN")
        if token and token.strip():
            return token
        return None

    multimodal_model_types = _MULTIMODAL_MODEL_TYPES
    if is_vllm():
        # Avoid mutating module-level state
        multimodal_model_types = {
            name: text_key
            for name, text_key in _MULTIMODAL_MODEL_TYPES.items()
            if name not in _PLUGIN_SUPPORTED_MULTIMODAL_MODELS
        }
    # For multimodal models, extract the text sub-config so the rest of ATOM
    # (which is text-only today) works transparently.
    if model_type in multimodal_model_types:
        text_config_key = multimodal_model_types[model_type]
        text_config_dict = config_dict.get(text_config_key, {}).copy()
        # Remove auto_map to avoid trust_remote_code issues
        text_config_dict.pop("auto_map", None)
        # Propagate quantization_config from root level into text config
        # (quantization_config lives alongside text_config, not inside it).
        if (
            "quantization_config" not in text_config_dict
            and "quantization_config" in config_dict
        ):
            text_config_dict["quantization_config"] = config_dict["quantization_config"]
        text_model_type = text_config_dict.get("model_type", "deepseek_v3")
        if text_model_type in _PLAIN_TEXT_CONFIG_MODEL_TYPES:
            # Transformers does not ship a config class for these in this image
            # (KimiLinearConfig, Glm5NextTextConfig). Keep the fields as plain
            # PretrainedConfig attrs; the ATOM model normalizes the aliases it
            # needs at construction time.
            hf_config = PretrainedConfig.from_dict(text_config_dict)
        else:
            mapped_type = _CONFIG_REGISTRY.get(text_model_type, text_model_type)
            config_class = AutoConfig.for_model(mapped_type)
            hf_config = config_class.from_dict(text_config_dict)
        # Override architectures so that ATOM selects the correct model class
        # which can handle the multimodal weight prefix during loading.
        original_arch = config_dict.get("architectures", [])
        if original_arch:
            hf_config.architectures = original_arch
        # Propagate top-level token IDs if missing in text config
        for field in ("bos_token_id", "eos_token_id", "pad_token_id"):
            if getattr(hf_config, field, None) is None and field in config_dict:
                setattr(hf_config, field, config_dict[field])
        if not hasattr(hf_config, "text_config"):
            hf_config.text_config = hf_config
        # Store full multimodal config (with vision_config) for vision encoder init
        try:
            full_config = AutoConfig.from_pretrained(
                model, trust_remote_code=trust_remote_code
            )
            hf_config._multimodal_config = full_config
        except Exception:  # noqa: BLE001 - transformers raises anything here
            # Consumers use None to report that their full multimodal config
            # could not be loaded. A partial generic object can bypass those
            # checks and run with default token IDs. GLM-5.3 is text-only here,
            # so it has no reason to weaken that contract.
            hf_config._multimodal_config = None
        return hf_config

    if model_type in _PLAIN_CONFIG_MODEL_TYPES:
        # Speculative-draft checkpoints ship their own model_type that
        # transformers has never heard of (and never will -- they are not
        # standalone LMs). There is no schema to map onto, and nothing here
        # needs one: the draft model reads plain attributes off the config.
        # Keep every field verbatim as a bare PretrainedConfig.
        return PretrainedConfig.from_dict(config_dict)

    if model_type in _CONFIG_REGISTRY:
        config_class = AutoConfig.for_model(_CONFIG_REGISTRY[model_type])
        hf_config = config_class.from_pretrained(
            model,
            token=_get_hf_token(),
            trust_remote_code=trust_remote_code,
        )
        # transformers' from_pretrained strips fields that aren't in the target
        # config schema. For mapped types (e.g. deepseek_v4 → deepseek_v3) the
        # source-specific fields would be dropped. Re-inject them so V4-only
        # attrs (compress_ratios, num_hash_layers, hc_mult, swiglu_limit, ...)
        # remain accessible via getattr(hf_config, field) downstream.
        for field, value in config_dict.items():
            if not hasattr(hf_config, field):
                setattr(hf_config, field, value)
        return hf_config
    try:
        hf_config = AutoConfig.from_pretrained(
            model, trust_remote_code=trust_remote_code
        )
    except ValueError as e:
        # For the unsupported model in current transformers, try vllm if in plugin mode
        if is_vllm():
            from vllm.transformers_utils.config import get_config
            from vllm.transformers_utils.gguf_utils import (
                maybe_patch_hf_config_from_gguf,
            )

            hf_config = get_config(model, trust_remote_code=trust_remote_code)
            hf_config = maybe_patch_hf_config_from_gguf(model, hf_config)
        else:
            raise e
    return hf_config


def get_generation_config(model: str) -> GenerationConfig:
    try:
        return GenerationConfig.from_pretrained(
            model,
        )
    except OSError:  # Not found
        return None


def _is_minimax_m3_config(hf_config: PretrainedConfig) -> bool:
    architectures = getattr(hf_config, "architectures", None) or ()
    if any("MiniMaxM3" in arch for arch in architectures):
        return True
    text_config = getattr(hf_config, "text_config", None)
    return any(
        "minimax_m3" in str(model_type).lower()
        for model_type in (
            getattr(hf_config, "model_type", ""),
            getattr(text_config, "model_type", ""),
        )
    )


def _normalize_minimax_m3_text_config(hf_config: PretrainedConfig) -> None:
    if not _is_minimax_m3_config(hf_config):
        return
    text_config = getattr(hf_config, "text_config", None)
    if text_config is None or text_config is hf_config:
        return

    if getattr(text_config, "hidden_act", None) == "swigluoai":
        if getattr(text_config, "swiglu_beta", None) is None:
            text_config.swiglu_beta = 1.0

    for attr_name in (
        "use_index_cache",
        "index_topk_freq",
        "index_topk_pattern",
        "index_skip_topk_offset",
    ):
        attr_value = getattr(hf_config, attr_name, None)
        if attr_value is not None:
            setattr(text_config, attr_name, attr_value)

    for attr_name, attr_value in vars(text_config).items():
        if attr_name.startswith("_") or getattr(hf_config, attr_name, None) is not None:
            continue
        setattr(hf_config, attr_name, attr_value)


@dataclass
class ParallelConfig:
    data_parallel_size: int = 1
    """Number of data parallel groups. MoE layers will be sharded according to
    the product of the tensor parallel size and data parallel size."""
    data_parallel_size_local: int | None = None
    """DP ranks this node runs. Defaults to data_parallel_size, i.e. the
    single-node case where every global rank is local. Set it below the global
    size to give a node one slice of a multi-node run; it also reaches MoRI as
    `gpu_per_node` (see model_ops/moe.py), so it must describe real hardware."""
    data_parallel_rank: int = 0
    """Rank of the data parallel group."""
    data_parallel_rank_local: int | None = None
    """Local rank of the data parallel group,
    set only in SPMD mode."""
    decode_context_parallel_size: int = 1
    """DCP group size. tp_size must be divisible by dcp_size.
    DCP does not increase world_size; it reuses TP GPUs."""
    pipeline_parallel_rank: int = 0
    """Pipeline stage index of this EngineCore (0 = first stage). Each PP stage
    runs as an independent EngineCore process; this identifies which stage."""
    pp_meta_addrs: list = field(default_factory=list)
    """ZMQ endpoints (len == pp_size) where each downstream stage receives the
    scheduled batch from the head. Populated by CoreManager for pp_size > 1."""
    pp_token_addr: str = ""
    """ZMQ endpoint where the head receives sampled tokens back from the last
    stage. Populated by CoreManager for pp_size > 1."""
    control_address: str = ""
    """ZMQ endpoint carrying control traffic (utility commands, abort, shutdown)
    for this EngineCore, separate from the request endpoint. Keeping the two
    apart leaves the request socket with a single writer thread, so admitting a
    request needs no synchronization. Populated by launch_engine_core."""
    pp_kv_status_addr: str = ""
    """ZMQ endpoint where the head receives KV offload status from downstream
    PP stages. All downstream stages PUSH; the head PULLs."""
    data_parallel_master_port: int = 29500
    """Port of the data parallel master."""

    data_parallel_base_port: int = get_open_port()

    data_parallel_master_ip: str = "127.0.0.1"

    @property
    def is_multinode_dp(self) -> bool:
        """Whether this node owns only part of the global DP group.

        Inferred from the topology rather than a separate flag: either this
        node runs fewer ranks than exist globally, or its slice starts at a
        non-zero global rank.
        """
        # data_parallel_size_local is int | None in the declaration, but
        # __post_init__ always resolves it before any caller can reach here.
        assert self.data_parallel_size_local is not None
        return (
            self.data_parallel_size_local < self.data_parallel_size
            or self.data_parallel_rank > 0
        )

    def get_next_dp_init_port(self) -> int:
        """
        We might need to initialize process groups in multiple
        processes that is related to data parallelism,
        e.g. both in the worker and in the engine, which
        can live in different processes. To avoid port conflicts, we
        pop a new port from the prepared port list each time we need to
        initialize a new process group related to data parallelism.
        """
        answer = self.data_parallel_master_port
        self.data_parallel_master_port += self.data_parallel_rank

        return answer

    def stateless_init_dp_group(self):
        # NOTE: In high-concurrency scenarios multiple processes
        # can pick the same (currently free) port through a race
        # condition when calling `get_open_port()`. When the first
        # process binds the port the others will subsequently fail
        # with `torch.distributed.DistNetworkError: EADDRINUSE`.
        # To make the initialization more robust we retry a few times
        # with a fresh port whenever this specific error is observed.
        dp_group = stateless_init_torch_distributed_process_group(
            self.data_parallel_master_ip,
            self.get_next_dp_init_port(),
            self.data_parallel_rank,
            self.data_parallel_size,
            backend="gloo",
        )
        return dp_group

    @staticmethod
    def has_unfinished_dp(dp_group: ProcessGroup, has_unfinished: bool) -> bool:
        tensor = torch.tensor([has_unfinished], dtype=torch.int32, device="cpu")
        # dp rank 0: has_unfinished_seqs=True
        # dp rank 1: has_unfinished_seqs=False
        # aggregated: has_unfinished_seqs=True
        # so this is an OR operation, i.e. MAX in integers
        torch.distributed.all_reduce(tensor, op=ReduceOp.MAX, group=dp_group)
        aggregated_has_unfinished = bool(tensor.item())
        return aggregated_has_unfinished

    def compute_hash(self):
        """
        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        """
        factors: list[Any] = []
        factors.append(self.data_parallel_size)
        factors.append(self.data_parallel_size_local)
        factors.append(self.data_parallel_rank)
        factors.append(self.data_parallel_rank_local)
        factors.append(self.data_parallel_master_ip)
        factors.append(self.data_parallel_master_port)
        return hashlib.sha256(str(factors).encode()).hexdigest()

    def __post_init__(self) -> None:
        # Only override with env vars if explicitly set.
        # This allows programmatic configuration to take precedence.
        if envs.is_set("ATOM_DP_SIZE"):
            self.data_parallel_size = envs.ATOM_DP_SIZE
        if envs.is_set("ATOM_DP_RANK"):
            self.data_parallel_rank = envs.ATOM_DP_RANK
        if envs.is_set("ATOM_DP_RANK_LOCAL"):
            self.data_parallel_rank_local = envs.ATOM_DP_RANK_LOCAL
        if envs.is_set("ATOM_DP_MASTER_IP"):
            self.data_parallel_master_ip = envs.ATOM_DP_MASTER_IP
        if envs.is_set("ATOM_DP_MASTER_PORT"):
            self.data_parallel_master_port = envs.ATOM_DP_MASTER_PORT
        if envs.is_set("ATOM_DP_BASE_PORT"):
            self.data_parallel_base_port = envs.ATOM_DP_BASE_PORT

        if self.data_parallel_size < 1:
            raise ValueError("data_parallel_size must be at least 1")

        if envs.is_set("ATOM_DP_SIZE_LOCAL"):
            self.data_parallel_size_local = envs.ATOM_DP_SIZE_LOCAL

        # Default the local slice to the whole group: on one node every global
        # rank is local, and that is the overwhelmingly common case.
        if self.data_parallel_size_local is None:
            self.data_parallel_size_local = self.data_parallel_size

        if self.data_parallel_size_local < 1:
            raise ValueError("data_parallel_size_local must be at least 1")
        if self.data_parallel_rank < 0:
            raise ValueError("data_parallel_rank must be non-negative")
        if (
            self.data_parallel_rank + self.data_parallel_size_local
            > self.data_parallel_size
        ):
            raise ValueError(
                f"data_parallel_rank ({self.data_parallel_rank}) + "
                f"data_parallel_size_local ({self.data_parallel_size_local}) "
                f"must not exceed data_parallel_size "
                f"({self.data_parallel_size}): this node's slice would run off "
                f"the end of the global DP group"
            )


_DSPARK_DEFAULT_MAX_BLOCK = 16
_DSPARK_DEFAULT_ROLLING_WINDOW = 128


def _normalize_draft_dspark_config(hf_config: PretrainedConfig) -> None:
    """Map a standalone DSpark draft config onto ATOM's canonical names.

    DSpark checkpoints come in two shapes and name the same quantities
    differently:

    - INLINE (e.g. V4-Pro-DSpark) ships inside the target checkpoint and already
      uses the ``dspark_*`` names the rest of ATOM reads. It never comes here.
    - STANDALONE drafts are their OWN checkpoint (``architectures:
      ["*DSparkModel"]``, e.g. Kimi-K3-DSpark's ``K3DSparkModel``) and carry
      their DSpark fields at the config top level: ``target_layer_ids``,
      ``mask_token_id``, ``markov_rank``, and optionally a training block width.
    """
    target_layer_ids = getattr(hf_config, "target_layer_ids", None)
    if not target_layer_ids:
        raise ValueError(
            "K3DSparkModel config is missing `target_layer_ids` (the target "
            "decoder layers whose hidden states the draft consumes). Without "
            "it the draft has no context input."
        )
    # 0-based target decoder-layer indices, matching ATOM's `layers[i]` aux-tap
    # convention: the reference indexes `hidden_states[layer_id + 1]`, i.e. the
    # OUTPUT of layer `layer_id`, which is the layer this convention taps.
    hf_config.dspark_target_layer_ids = [int(i) for i in target_layer_ids]

    mask_token_id = getattr(hf_config, "mask_token_id", None)
    if mask_token_id is None:
        raise ValueError(
            "K3DSparkModel config is missing `mask_token_id` (the noise token "
            "the draft block is seeded with)."
        )
    hf_config.dspark_noise_token_id = int(mask_token_id)

    hf_config.dspark_markov_rank = int(getattr(hf_config, "markov_rank", 0) or 0)
    # Synthesized marker, not a checkpoint field: the flavor discriminator the
    # rest of the stack branches on (SpeculativeConfig.use_dspark_with_draft).
    hf_config.dspark_with_draft = True

    # NOTE: no `dspark_block_size` here. Unlike V4-Pro-DSpark and the SpecForge
    # SpecForge DFlash checkpoints, this config carries no block width: the draft is
    # width-agnostic in its weights and the block is sized by
    # --num-speculative-tokens (7 in the checkpoint's own serving recipe).
    # DSparkProposer._resolve_mtp_k falls back to that.

    logger.info(
        "Detected MLA DSpark drafter with a separate draft model "
        f"(markov_rank={hf_config.dspark_markov_rank}, "
        f"target_layers={hf_config.dspark_target_layer_ids}, "
        f"mask_token_id={hf_config.dspark_noise_token_id})"
    )


def _normalize_moe_config_fields(
    hf_config: PretrainedConfig,
    model_path: str | None = None,
) -> None:
    """Normalize common MoE config field names across model families."""
    moe_config = getattr(hf_config, "text_config", hf_config)
    updates: dict[str, Any] = {}

    n_routed = getattr(
        moe_config,
        "n_routed_experts",
        getattr(moe_config, "num_experts", None),
    )
    if n_routed is not None:
        updates["n_routed_experts"] = n_routed

    existing_n_shared = getattr(moe_config, "n_shared_experts", None)
    if existing_n_shared is not None:
        updates["n_shared_experts"] = existing_n_shared
    elif n_routed is not None and model_path is not None:
        from atom.models.utils import ckpt_shared_expert_count

        n_shared = ckpt_shared_expert_count(model_path)
        if n_shared > 0:
            updates["n_shared_experts"] = n_shared

    if not updates:
        return

    moe_config.update(updates)
    if moe_config is not hf_config:
        hf_config.update(updates)


@dataclass
class SpeculativeConfig:
    method: str | None = ""
    model: str | None = None
    num_speculative_tokens: int | None = None
    draft_model_hf_config: PretrainedConfig | None = None
    use_aux_hidden_state: bool = False
    eagle3_aux_layer_ids: list[int] = field(default_factory=list)
    # Debug/benchmark knobs: force a speculative acceptance curve independent of
    # the real draft/target agreement, so a run can replay a published
    # acceptance-length figure (an InferenceX golden AL, say) while the draft head
    # is still training. Set at most one; both resolve into
    # `synthetic_acceptance_rates`. See ROCm/ATOM#555.
    #
    # Mean acceptance length in [1, num_speculative_tokens + 1], counting the
    # target's own guaranteed token -- the same unit as vLLM's
    # synthetic_acceptance_length and SGLang's SGLANG_SIMULATE_ACC_LEN, so a
    # published AL can be pasted in without conversion.
    synthetic_acceptance_length: float | None = None
    # The same target expressed as a mean acceptance RATE in [0, 1]
    # (accepted_draft / total_draft), i.e. (length - 1) / num_speculative_tokens.
    synthetic_acceptance_rate: float | None = None
    # Resolved per-position *unconditional* acceptance rates (entry i = marginal
    # probability that the first i+1 draft tokens are all accepted), filled in by
    # __post_init__. None => real draft/target rejection sampling.
    synthetic_acceptance_rates: list[float] | None = None

    # model_type → mtp_model_type mapping
    _MTP_TYPE_MAP: ClassVar[dict[str, str]] = {
        "deepseek_v3": "deepseek_mtp",
        "deepseek_v32": "deepseek_mtp",
        "deepseek_v4": "deepseek_v4_mtp",
        "glm_moe_dsa": "deepseek_mtp",
        "qwen3_next": "qwen3_next_mtp",
        "qwen3_5": "qwen3_5_mtp",
        "qwen3_5_moe": "qwen3_5_mtp",
        "qwen3_5_text": "qwen3_5_mtp",
        "qwen3_5_moe_text": "qwen3_5_mtp",
        "mimo_v2": "mimo_v2_mtp",
        "mimo_v2_flash": "mimo_v2_mtp",
    }

    # mtp_model_type → (n_predict_attr, architecture)
    _MTP_CONFIG: ClassVar[dict[str, tuple[str, str]]] = {
        "deepseek_mtp": ("num_nextn_predict_layers", "DeepSeekMTPModel"),
        "deepseek_v4_mtp": ("num_nextn_predict_layers", "DeepseekV4MTPModel"),
        "qwen3_next_mtp": ("num_nextn_predict_layers", "Qwen3NextMTPModel"),
        "qwen3_5_mtp": ("mtp_num_hidden_layers", "Qwen3_5MTPModel"),
    }

    def use_dspark(self) -> bool:
        """DSpark semi-autoregressive block drafter (either flavor).

        DSpark is a parallel block drafter (parallel backbone + Markov
        sequential head + confidence head), NOT serial MTP. Two checkpoint
        flavors reach here, and both normalize to `dspark_block_size` in
        `hf_config_override`:

        - V4-Pro-DSpark: ships INSIDE the V4 target checkpoint under the same
          `mtp.*` namespace serial MTP uses, so only the DSpark-only
          `dspark_block_size` field distinguishes the two.
        - Kimi-K3-DSpark: a standalone MLA-backbone checkpoint with its own
          `architectures: ["K3DSparkModel"]`.

        We intentionally never silently fall back to MTP: a wrong fallback
        loads cleanly but measures the wrong algorithm.
        """
        cfg = self.draft_model_hf_config
        return (
            self.method == "dspark"
            or bool(getattr(cfg, "dspark_block_size", None))
            or bool(getattr(cfg, "dspark_with_draft", False))
        )

    def use_dspark_with_draft(self) -> bool:
        """True when DSpark was given a separate draft model (--draft-model).

        ``dspark_with_draft`` is NOT a checkpoint field -- do not go looking for
        it in config.json. It is synthesized by
        :func:`_normalize_draft_dspark_config`, which runs from
        :meth:`hf_config_override` when the draft's ``architectures`` is
        ``["K3DSparkModel"]``, so it is set for exactly the configs that went
        through that normalization.
        """
        cfg = self.draft_model_hf_config
        return bool(getattr(cfg, "dspark_with_draft", False))

    def _resolve_synthetic_acceptance(self) -> None:
        """Validate the forced-acceptance knobs and resolve to per-position rates.

        Both knobs describe the same curve, so exactly one may be set; the rate
        form is converted to a length and everything downstream reads only
        ``synthetic_acceptance_rates``.
        """
        # Local import: the schedule lives next to the kernel that consumes it,
        # and that module pulls in triton, which has no business loading just
        # because someone imported a config.
        from atom.model_ops.rejection_sampler import acceptance_length_to_rates

        if (
            self.synthetic_acceptance_length is not None
            and self.synthetic_acceptance_rate is not None
        ):
            raise ValueError(
                "--spec-decode-acceptance-length and --spec-decode-acceptance-rate "
                "describe the same curve; set at most one."
            )
        length = self.synthetic_acceptance_length
        if self.synthetic_acceptance_rate is None and length is None:
            return

        n = self.num_speculative_tokens
        if not n:
            raise ValueError(
                "Forced speculative acceptance needs --num-speculative-tokens, "
                f"but it is {n!r}."
            )
        if self.synthetic_acceptance_rate is not None:
            rate = self.synthetic_acceptance_rate
            if not 0.0 <= rate <= 1.0:
                raise ValueError(
                    "synthetic_acceptance_rate (--spec-decode-acceptance-rate) "
                    f"must be in [0, 1], but got {rate}."
                )
            length = 1.0 + n * rate
        if not 1.0 <= length <= float(n + 1):
            raise ValueError(
                "synthetic_acceptance_length (--spec-decode-acceptance-length) "
                f"must be in [1, {n + 1}] for num_speculative_tokens={n}, but got "
                f"{length}."
            )
        self.synthetic_acceptance_length = length
        self.synthetic_acceptance_rates = acceptance_length_to_rates(length, n)
        logger.info(
            "Forced speculative acceptance ON: mean acceptance length %.4f over "
            "%d draft positions (per-position rates %s). Throughput numbers from "
            "this run are synthetic; output text and accuracy are meaningless.",
            length,
            n,
            [round(r, 4) for r in self.synthetic_acceptance_rates],
        )

    def __post_init__(self):
        self._resolve_synthetic_acceptance()
        if self.draft_model_hf_config is None:
            self.draft_model_hf_config = get_hf_config(
                self.model, trust_remote_code=True
            )
        # For multimodal models, extract text_config
        if hasattr(self.draft_model_hf_config, "text_config"):
            self.draft_model_hf_config = self.draft_model_hf_config.text_config
        self.hf_config_override(self.draft_model_hf_config, self.model)

        if self.method == "eagle3":
            # MLA drafts (kv_lora_rank set) route to Eagle3DeepseekMLAModel
            # via the arch rewrite in hf_config_override; no early reject.
            # Aux hidden state layers: prefer the draft checkpoint's
            # eagle_config; if absent or the list is empty, ModelRunner
            # falls back to model.get_eagle3_aux_hidden_state_layers(),
            # which defaults to 3 layers — early / middle / late
            # (see DeepseekV2ForCausalLM.get_eagle3_aux_hidden_state_layers,
            # returns `(2, num_layers // 2, num_layers - 3)`, aligned with vLLM).
            eagle_cfg = getattr(self.draft_model_hf_config, "eagle_config", None)
            if eagle_cfg:
                self.use_aux_hidden_state = eagle_cfg.get("use_aux_hidden_state", False)
                if self.use_aux_hidden_state and not self.eagle3_aux_layer_ids:
                    self.eagle3_aux_layer_ids = eagle_cfg.get(
                        "eagle_aux_hidden_state_layer_ids", []
                    )
            else:
                self.use_aux_hidden_state = True

    @staticmethod
    def hf_config_override(
        hf_config: PretrainedConfig, model_path: str | None = None
    ) -> None:
        # Eagle3 architecture mapping (architecture-level, not model_type)
        arch = (getattr(hf_config, "architectures", None) or [""])[0]
        if arch.endswith("DSparkModel"):
            _normalize_draft_dspark_config(hf_config)
            return
        if arch == "LlamaForCausalLMEagle3":
            hf_config.architectures = ["Eagle3LlamaModel"]
        elif arch == "Eagle3DeepseekV2ForCausalLM":
            hf_config.architectures = ["Eagle3DeepseekMLAModel"]

        # DSpark detection (before MTP rewrite): the V4 DSpark checkpoint has
        # model_type=deepseek_v4 just like serial MTP, but carries DSpark-only
        # config fields. Route it to the DSpark draft model and skip the MTP
        # n_predict=1 rewrite (DSpark uses dspark_block_size, not n_predict).
        if getattr(hf_config, "dspark_block_size", None):
            hf_config.model_type = "deepseek_v4_dspark"
            hf_config.architectures = ["DeepseekV4DSparkModel"]
            logger.info(
                "Detected DeepSeek-V4 DSpark drafter "
                f"(block_size={hf_config.dspark_block_size}, "
                f"markov_rank={getattr(hf_config, 'dspark_markov_rank', None)}, "
                f"target_layers={getattr(hf_config, 'dspark_target_layer_ids', None)})"
            )
            _normalize_moe_config_fields(hf_config, model_path)
            return

        # Step 1: resolve model_type → mtp model_type
        mtp_type = SpeculativeConfig._MTP_TYPE_MAP.get(hf_config.model_type)
        if mtp_type is not None:
            hf_config.model_type = mtp_type

        # Step 2: apply MTP-specific config overrides
        entry = SpeculativeConfig._MTP_CONFIG.get(hf_config.model_type)
        if entry is not None:
            n_predict_attr, arch = entry
            n_predict = getattr(hf_config, n_predict_attr, 1)
            if n_predict != 1:
                logger.warning(
                    f"Overriding {n_predict_attr} from {n_predict} to 1 "
                    "(MTP typically uses 1 layer that gets reused)"
                )
                n_predict = 1

            updates: dict[str, Any] = {
                "n_predict": n_predict,
                "num_nextn_predict_layers": n_predict,
                "architectures": [arch],
            }
            hf_config.update(updates)

        # MiMo-V2 has not MTP related information in HF config.json,
        # override n_predict with the actual layer count (default 3).
        if hf_config.model_type == "mimo_v2_mtp":
            n_predict = getattr(hf_config, "num_nextn_predict_layers", 3)
            hf_config.update(
                {
                    "n_predict": n_predict,
                    "num_nextn_predict_layers": n_predict,
                    "architectures": ["MiMoV2FlashMTPModel"],
                }
            )

        _normalize_moe_config_fields(hf_config, model_path)
        logger.info(f"hf config is: {hf_config}")

    def __repr__(self) -> str:
        method = self.method
        num_spec_tokens = self.num_speculative_tokens
        return f"SpeculativeConfig({method=}, {num_spec_tokens=})"


@dataclass
class KVEventsConfig:
    """Configuration for KV cache event publishing."""

    enable: bool = False
    publisher: str = "zmq"  # "null" | "zmq"
    endpoint: str = "tcp://127.0.0.1:5557"
    topic: str = ""
    # ZMQ high-water-mark on the PUB socket (0 = unlimited).
    hwm: int = 0
    # Bounded in-process queue between scheduler and sender thread. When full,
    # oldest batch is dropped — KV events are advisory, never stall inference.
    buffer_steps: int = 10_000

    @classmethod
    def from_env(cls) -> "KVEventsConfig":
        """Build a config from `ATOM_KV_EVENTS_*` env vars. Provides an env-only
        opt-in path so containerized deployments can enable events without a
        CLI flag (see `atom/utils/envs.py`)."""
        return cls(
            enable=envs.ATOM_KV_EVENTS_ENABLE,
            publisher=envs.ATOM_KV_EVENTS_PUBLISHER,
            endpoint=envs.ATOM_KV_EVENTS_ENDPOINT,
            topic=envs.ATOM_KV_EVENTS_TOPIC,
            hwm=envs.ATOM_KV_EVENTS_HWM,
            buffer_steps=envs.ATOM_KV_EVENTS_BUFFER_STEPS,
        )


@dataclass
class DSparkConfig:
    """Runtime configuration for DSpark speculative verification.

    Single source of truth for the DSpark knobs read across the model runner,
    the V4 attention op, and the Eagle proposer. It is built ONCE in the parent
    process from the ``--dspark-config`` JSON dict (see :meth:`from_dict`), then
    pickled into every engine-core worker subprocess as part of :class:`Config`,
    so all read sites observe the same resolved values via ``config.dspark.*``
    (no ``os.environ`` lookups).

    Fields:
      - confidence_schedule: use the DSpark confidence head to pick a per-request
        verify length ell_r (paper Algorithm 1) + variable-length verification.
      - ragged: per-request ragged verify (§5.2 avoid-padding); each decode seq
        forwards its own ell_r+1 tokens (no batch-level q padding).
      - ragged_graph_sizes: comma-separated per-seq CUDA-graph query-length
        buckets to capture for the ragged path (e.g. "1,3,6" or "8").
      - q_buckets: CUDA-graph query-length buckets for the (older) batch-uniform
        q-bucket verify path (independent of the ragged path).
      - disable_sps_calib: skip SPS calibration (replays captured graphs at
        warmup); fall back to the synthetic SPS stub.
    """

    confidence_schedule: bool = False
    ragged: bool = False
    ragged_graph_sizes: str = ""
    q_buckets: str = ""
    disable_sps_calib: bool = False

    @classmethod
    def from_dict(cls, cfg: dict | None) -> "DSparkConfig":
        """Build from the ``--dspark-config`` JSON dict.

        ``cfg`` maps directly onto this dataclass' fields; unknown keys raise so
        typos fail fast."""
        cfg = cfg or {}
        allowed = {f.name for f in fields(cls)}
        unknown = set(cfg) - allowed
        if unknown:
            raise ValueError(
                f"Unknown --dspark-config key(s): {sorted(unknown)}. "
                f"Supported keys: {sorted(allowed)}"
            )
        return cls(**cfg)


@dataclass
class EPLBConfig:
    """EPLB sub-config (vLLM-style: enable + config object)."""

    load_window_size: int = 1000
    rebalance_interval: int = 3000
    rebalance_layers_per_chunk: int = 64
    num_redundant_experts: int = 0
    rebalance_min_balancedness: float = 2.0
    rebalance_balancedness_agg: str = "min"
    p2p_batch_chunk_size: int = 32
    # Placement policy for spending the redundant-expert budget:
    #   "naive"  -> greedy replicate + balanced_packing (spread thinly)
    #   "biased" -> fully replicate top-K hottest experts to all GPUs
    #               (K = num_redundant // num_gpus, per-node in multi-node)
    placement_policy: str = "naive"

    def __post_init__(self):
        self.load_window_size = int(self.load_window_size)
        assert self.load_window_size > 0, "eplb.load_window_size must be > 0"
        self.rebalance_interval = int(self.rebalance_interval)
        assert self.rebalance_interval > 0, "eplb.rebalance_interval must be > 0"
        assert (
            self.rebalance_interval >= self.load_window_size
        ), "eplb.rebalance_interval must be >= eplb.load_window_size"
        self.rebalance_layers_per_chunk = int(self.rebalance_layers_per_chunk)
        assert (
            self.rebalance_layers_per_chunk > 0
        ), "eplb.rebalance_layers_per_chunk must be > 0"
        self.num_redundant_experts = int(self.num_redundant_experts)
        assert (
            self.num_redundant_experts >= 0
        ), "eplb.num_redundant_experts must be >= 0"
        self.rebalance_min_balancedness = float(self.rebalance_min_balancedness)
        self.rebalance_balancedness_agg = (
            str(self.rebalance_balancedness_agg).lower().strip()
        )
        assert self.rebalance_balancedness_agg in {
            "min",
            "mean",
        }, "eplb.rebalance_balancedness_agg must be one of {'min','mean'}"
        self.p2p_batch_chunk_size = int(self.p2p_batch_chunk_size)
        assert self.p2p_batch_chunk_size > 0, "eplb.p2p_batch_chunk_size must be > 0"
        self.placement_policy = str(self.placement_policy).lower().strip()
        assert self.placement_policy in {
            "naive",
            "biased",
        }, "eplb.placement_policy must be one of {'naive','biased'}"

    @classmethod
    def from_dict(cls, cfg: dict | None) -> "EPLBConfig":
        """Build from the ``--eplb-config`` JSON dict.

        ``cfg`` maps directly onto this dataclass' fields; unknown keys raise so
        typos fail fast."""
        cfg = cfg or {}
        allowed = {f.name for f in fields(cls)}
        unknown = set(cfg) - allowed
        if unknown:
            raise ValueError(
                f"Unknown --eplb-config key(s): {sorted(unknown)}. "
                f"Supported keys: {sorted(allowed)}"
            )
        return cls(**cfg)


DCP_COMM_BACKENDS = ("ag_rs", "a2a")


@dataclass
class DCPConfig:
    """DCP (Decode Context Parallel) sub-config: interleave granularity, query
    replication, output-merge placement and the merge collective backend --
    knobs that would otherwise each grow the top-level CLI surface."""

    interleave_size: int = 1
    enable_query_replication: bool = True
    enable_project_before_merge: bool = True
    comm_backend: str = "a2a"

    def __post_init__(self):
        self.interleave_size = int(self.interleave_size)
        assert self.interleave_size >= 1, "dcp.interleave_size must be >= 1"
        self.enable_query_replication = bool(self.enable_query_replication)
        self.enable_project_before_merge = bool(self.enable_project_before_merge)
        self.comm_backend = str(self.comm_backend)
        assert self.comm_backend in DCP_COMM_BACKENDS, (
            f"dcp.comm_backend must be one of {list(DCP_COMM_BACKENDS)}; "
            f"got {self.comm_backend!r}"
        )

    @classmethod
    def from_dict(cls, cfg: dict | None) -> "DCPConfig":
        """Build from the ``--dcp-config`` JSON dict.

        ``cfg`` maps directly onto this dataclass' fields; unknown keys raise so
        typos fail fast."""
        cfg = cfg or {}
        allowed = {f.name for f in fields(cls)}
        unknown = set(cfg) - allowed
        if unknown:
            raise ValueError(
                f"Unknown --dcp-config key(s): {sorted(unknown)}. "
                f"Supported keys: {sorted(allowed)}"
            )
        return cls(**cfg)


def qrep_unsupported_reason(
    dcp_size: int, speculative_config, mxfp4_bmm: bool
) -> str | None:
    """Why DCP query replication cannot run here, or None if it can.

    Kept a module-level pure function so it is unit-testable: the alternative,
    exercising it through ``Config.__post_init__``, needs a real model directory
    and an HF config. ``Config.__post_init__`` is its only production caller.
    """
    if dcp_size <= 1:
        # No DCP group means there is no AllGather Q to remove.
        return "decode_context_parallel_size <= 1 (no DCP group)"
    if speculative_config is not None:
        # MTP / eagle3 / dspark run a qlen>1 verify on the cprr kernel.
        return "speculative decode (qlen>1 cprr path)"
    if mxfp4_bmm:
        # fp4 (mxfp4) absorbed BMM has a different scale structure.
        return "fp4 (mxfp4) BMM weights"
    return None


@dataclass
class Config:
    model: str
    trust_remote_code: bool = False
    max_num_batched_tokens: int = 16384
    long_prefill_token_threshold: int = 0
    attn_prefill_chunk_size: int = 16384
    # Tokens between rungs of the state-checkpoint ladder, shared by every
    # Pool.STATE class. Must be a multiple of the prefix-cache hash block size
    # (snapped, with a warning, in BlockManager).
    #   >0  a rung every N tokens
    #    0  state checkpointing off entirely
    #   -1  no interval rungs, but the demand rung and the prompt-end anchor
    #       still place checkpoints
    # See BlockManager.checkpointers_at.
    state_checkpoint_interval_tokens: int = 8192
    # Whether a refused hit may place a rung of its own. Off leaves the
    # prompt-end anchor as the only placement; the rung reads back far less
    # often than the anchor, so its worth is an open question — see
    # `StateSlotPool.mark_speculative` for the measurement and
    # `BlockManager._record_checkpoint_demand` for the placement.
    state_checkpoint_demand: bool = True
    scheduler_delay_factor: float = 0.0
    max_num_seqs: int = 512
    max_model_len: int | None = None
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    decode_context_parallel_size: int = 1
    dcp_config: DCPConfig = field(default_factory=DCPConfig)
    pipeline_parallel_size: int = 1
    prefill_context_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: PretrainedConfig = field(init=False)
    generation_config: GenerationConfig = field(init=False)
    parallel_config: ParallelConfig = field(default_factory=ParallelConfig)
    bos_token_id: int = -1
    eos_token_id: int = -1
    stop_token_ids: list[int] = field(default_factory=list)
    kv_cache_block_size: int = 16
    num_kvcache_blocks: int = -1
    kv_cache_dtype: str = "bf16"
    index_cache_dtype: str | None = None
    enable_prefix_caching: bool = True
    enable_chunked_prefill: bool = True
    enable_log_stats: bool = True
    # Seconds between engine-status lines. Validated > 0 by EngineStats.
    throughput_log_interval: float = 10.0
    # Requests in the sliding window behind the status line's prefix-cache hit
    # rate. Validated > 0 by EngineStats.
    cache_hit_rate_window: int = 1000
    port: int = 8006
    torch_profiler_dir: str | None = field(
        default_factory=lambda: envs.ATOM_TORCH_PROFILER_DIR
    )
    # Engine steps to skip after /start_profile before the profiler records,
    # and steps to record before it stops itself. 0 = no delay / no limit.
    profiler_delay_iters: int = 0
    profiler_max_iters: int = 0
    compilation_config: CompilationConfig = field(default_factory=CompilationConfig)
    quant_config: QuantizationConfig = field(init=False)
    asyncio_mode: bool = False
    mark_trace: bool = False
    load_dummy: str | None = None
    enable_expert_parallel: bool = False
    fake_eplb: bool = False
    # Width the MoE shards experts for when DP-attention simulates a deployment
    # wider than the box (set by CoreManager); 0 = not simulating.
    # `parallel_config.data_parallel_size` stays the real rank count, since it
    # sizes the process group and the token collectives.
    dp_logical_size: int = 0
    master_addr: str = "127.0.0.1"
    enable_dp_attention: bool = False
    # DP request-routing strategy used by CoreManager to pick an engine rank:
    # "round_robin" | "least_requests" (default) | "least_tokens". Only has an
    # effect when more than one DP engine rank is launched. See
    # atom/model_engine/engine_core_mgr.py:_select_dp_rank_locked. The literal
    # default must stay in sync with engine_core_mgr.DP_LB_DEFAULT (config.py
    # cannot import it without a cycle: engine_core_mgr imports Config).
    dp_load_balance: str = "least_requests"
    # MoE expert-parallel layout policy. When True, MoE EP computes ranks in the
    # flattened DP x TP device space (and shared-expert fusion is disabled,
    # because the fused shared expert assumes the per-DP MoE layout). The vLLM
    # plugin sets this when EP is enabled; native ATOM and other plugins use the
    # per-DP MoE layout and leave it False. Set by the frontend in
    # atom/plugin/config.py, not queried via is_vllm() at the call site.
    moe_ep_flatten_tp_across_dp: bool = False
    torch_dtype: torch.dtype = field(init=False)
    speculative_config: SpeculativeConfig | None = None
    kv_transfer_config: dict = field(default_factory=dict)
    kv_events_config: KVEventsConfig = field(default_factory=KVEventsConfig.from_env)
    # DSpark runtime knobs. Built once in the parent from --dspark-config (see
    # EngineArgs) and pickled into every worker. Read sites use `config.dspark.*`
    # (no os.environ lookups). Defaults to all-off.
    dspark: DSparkConfig = field(default_factory=DSparkConfig)

    enable_tbo: bool = False
    enable_tbo_decode: bool = False
    enable_low_latency: bool = False
    # Post-routing routed-MoE implementation. This is deliberately separate
    # from all2all backend/mode: Mega owns dispatch, both GEMMs, and combine.
    moe_backend: str = "standard"
    runner_qualname: str = "atom.model_engine.model_runner.ModelRunner"
    # EPLB master switch + sub-config
    eplb_enable: bool = False
    eplb_config: EPLBConfig = field(default_factory=EPLBConfig)

    # only use for plugin mode
    plugin_config: PluginConfig | None = None
    # only for quark_online_quantization
    online_quant_config: dict | None = None
    hf_overrides: dict[str, Any] | None = None

    # Intra-GPU prefill/decode disaggregation
    enable_rapidserve: bool = False
    # ZMQ IPC address: decode PUSH → prefill PULL (BlockAssignment messages)
    disagg_d2p_addr: str = ""
    # ZMQ IPC address: prefill PUSH → decode PULL (PrefillDone messages)
    disagg_p2d_addr: str = ""
    # Bootstrap round 1: prefill PUSH → decode PULL (weight IPC handles)
    disagg_weight_ipc_addr: str = ""
    # Bootstrap round 1 ACK: decode PUSH → prefill PULL (signals weights freed)
    disagg_weight_ack_addr: str = ""
    # Bootstrap round 2: prefill PUSH → decode PULL (kvcache_args + num_blocks)
    disagg_kvcache_ipc_addr: str = ""
    # True for the decode process in disagg mode: skip GPU weight/kvcache allocation.
    disagg_is_decode: bool = False
    # Name of the shared-memory region used for dynamic CU partitioning.
    # Both prefill and decode processes open this to exchange batch sizes.
    disagg_cu_shm_name: str = ""
    # Override max_num_seqs for the prefill process in disagg mode.
    # When None, prefill inherits the base max_num_seqs.
    disagg_prefill_max_num_seqs: int | None = None
    # When True (and enable_rapidserve=True), use CU-masked streams + shm
    # coordination between prefill and decode. When False (default),
    # use plain separate streams with no CU masking.
    disagg_constrained: bool = False

    @property
    def tp_world_size(self) -> int:
        """Number of TP worker processes actually launched.

        `tensor_parallel_size` is the *logical* width -- how many shards every
        weight is cut into. Under `--fake-eplb` on a box with fewer visible
        devices than `-tp`, only the first `tp_world_size` of those shards get
        a process, reproducing the first N devices of the larger deployment.
        See `atom/distributed/simulated_tp.py`.

        Gated on `fake_eplb` because such a run's output is garbage anyway;
        without it, an oversized `-tp` keeps raising in ModelRunner instead of
        silently running smaller. A property, not a field: `enable_dp_attention`
        rewrites `tensor_parallel_size` after Config is built.
        """
        tp = self.tensor_parallel_size
        if not self.fake_eplb:
            return tp
        # Does not create a CUDA context, so it is safe in the parent process.
        visible = torch.cuda.device_count()
        return visible if 0 < visible < tp else tp

    @property
    def capture_sizes(self) -> list[int]:
        """The declared CUDAGraph capture ladder, in batch sizes.

        Declared, not schedulable -- `ModelRunner` drops what its own token
        budget can never produce, a bound needing the drafter's resolved
        `mtp_k` that config cannot see. Returns a copy: the runner sorts and
        filters in place.
        """
        declared = self.compilation_config.cudagraph_capture_sizes
        if declared:
            return list(declared)
        sizes = self.compilation_config.cuda_graph_sizes
        if len(sizes) == 1:
            return [1, 2, 4, 8] + list(range(16, sizes[0] + 1, 16))
        return list(sizes)

    def __post_init__(self):
        self.moe_backend = self.moe_backend.strip().lower()
        if self.moe_backend not in ("standard", "mega"):
            raise ValueError(
                "moe_backend must be one of {'standard', 'mega'}, "
                f"got {self.moe_backend!r}"
            )
        if self.moe_backend == "mega" and not self.enable_expert_parallel:
            raise ValueError(
                "moe_backend='mega' requires expert parallelism; "
                "pass --enable-expert-parallel."
            )

        if isinstance(self.compilation_config, dict):
            self.compilation_config = CompilationConfig(**self.compilation_config)
        if isinstance(self.eplb_config, dict):
            self.eplb_config = EPLBConfig(**self.eplb_config)
        elif isinstance(self.eplb_config, EPLBConfig):
            # Normalize/validate even when constructed programmatically.
            self.eplb_config = EPLBConfig(**self.eplb_config.__dict__)
        else:
            raise TypeError("eplb_config must be EPLBConfig or dict")
        if isinstance(self.dcp_config, dict):
            self.dcp_config = DCPConfig(**self.dcp_config)
        elif isinstance(self.dcp_config, DCPConfig):
            # Normalize/validate even when constructed programmatically.
            self.dcp_config = DCPConfig(**self.dcp_config.__dict__)
        else:
            raise TypeError("dcp_config must be DCPConfig or dict")
        # assert os.path.isdir(self.model)

        # The forced-acceptance schedule spends its whole budget on the first
        # ceil(length - 1) positions, and the sampler can only accept what was
        # actually drafted. The DSpark confidence scheduler hands each request
        # its own verify length ell_r, so whenever ell_r falls below that many
        # positions the run quietly lands under the acceptance length it was
        # asked to reproduce (measured at length 3.78 over 7 positions: exact
        # while ell_r >= 3, 3.39 at ell_r = 2, 2.89 at ell_r = 1). ell_r is
        # chosen at runtime from the confidence head, so there is no upfront
        # check that would catch it -- and a benchmark reporting a number it
        # never hit is worse than one that refuses to start.
        if (
            self.speculative_config is not None
            and self.speculative_config.synthetic_acceptance_rates is not None
            and self.dspark.confidence_schedule
        ):
            raise ValueError(
                "Forced speculative acceptance (--spec-decode-acceptance-length "
                "/ --spec-decode-acceptance-rate) cannot be combined with the "
                "DSpark confidence scheduler (--dspark-config "
                "'{\"confidence_schedule\": true}'): it sizes each request's "
                "verify length at runtime, and a short one caps acceptance below "
                "the requested length with no way to detect it upfront. Drop "
                "confidence_schedule (and ragged, which needs it) for "
                "forced-acceptance runs."
            )

        # RapidServe (intra-GPU prefill/decode disagg) needs a specialized
        # runner in both the prefill and decode processes. Select it unless the
        # user explicitly overrode runner_qualname.
        if (
            self.enable_rapidserve
            and self.runner_qualname == "atom.model_engine.model_runner.ModelRunner"
        ):
            self.runner_qualname = (
                "atom.model_engine.model_runner.RapidServeModelRunner"
            )

        assert 1 <= self.tensor_parallel_size <= 8
        if self.decode_context_parallel_size > 1:
            assert self.tensor_parallel_size % self.decode_context_parallel_size == 0, (
                f"tp_size ({self.tensor_parallel_size}) must be divisible by "
                f"dcp_size ({self.decode_context_parallel_size})"
            )
            # Spec-decode + DCP arch gating. Any speculative method (mtp /
            # eagle3 / dspark) runs a q>1 verify pass, and DCP decode with q>1
            # uses the round-robin CP (cprr) MLA kernel, which is persistent-only
            # and ships only on gfx950; on gfx942 the non-persistent fallback
            # ignores the cprr masking and silently produces WRONG output.
            if self.speculative_config is not None:
                from aiter.jit.utils.chip_info import get_gfx

                gfx = get_gfx()
                assert gfx == "gfx950", (
                    f"Speculative decode + DCP is only supported on gfx950 (needs "
                    f"the persistent cprr MLA kernel); got {gfx}. Disable DCP or "
                    f"speculative decode on this GPU."
                )
                # TBO slices its ubatch work buffers by request, while the
                # sparse DCP indexer indexes them by query token. The two agree
                # only at one query per sequence.
                assert not self.enable_tbo_decode, (
                    "Decode TBO (--enable-tbo all) combined with speculative "
                    "decode and DCP is unverified. Use --enable-tbo (prefill "
                    "only), or drop DCP or speculative decode."
                )
        # DCP KV-cache interleave granularity S. S=1 (default) = token-level
        # round-robin (unchanged). S>1 = block-level interleave; must divide the
        # KV block so each physical block holds an integer number of S-groups
        # (the (i//(S*W))*S + i%S local-index math relies on block_size % S == 0),
        # and only makes sense under DCP.
        assert 1 <= self.dcp_config.interleave_size <= self.kv_cache_block_size, (
            f"dcp_config.interleave_size ({self.dcp_config.interleave_size}) must "
            f"be in [1, kv_cache_block_size={self.kv_cache_block_size}]"
        )
        if self.dcp_config.interleave_size > 1:
            assert self.kv_cache_block_size % self.dcp_config.interleave_size == 0, (
                f"kv_cache_block_size ({self.kv_cache_block_size}) must be divisible "
                f"by dcp_config.interleave_size ({self.dcp_config.interleave_size})"
            )
            assert self.decode_context_parallel_size > 1, (
                "dcp_config.interleave_size > 1 only applies under DCP "
                f"(decode_context_parallel_size={self.decode_context_parallel_size})"
            )
            assert self.speculative_config is None, (
                "dcp_config.interleave_size > 1 (block-level DCP interleave) is "
                "incompatible with speculative decode (MTP/eagle/dspark): the q>1 "
                "verify cprr MLA kernel assumes token-level interleave. Use "
                "dcp_config.interleave_size=1 with speculative decode, or disable "
                "speculative decode for block-level interleave."
            )

        # DCP Query Replication (QREP) first-cut gating: turn the flag OFF
        # (warn, not error) for combinations not yet wired, so it can default to
        # on without breaking mixed runs.
        if self.dcp_config.enable_query_replication:
            qrep_off = qrep_unsupported_reason(
                self.decode_context_parallel_size,
                self.speculative_config,
                envs.ATOM_USE_TRITON_MXFP4_BMM,
            )
            if qrep_off is not None:
                logger.warning(
                    "dcp_config.enable_query_replication disabled: %s not "
                    "supported in the first cut.",
                    qrep_off,
                )
                self.dcp_config.enable_query_replication = False
        assert 1 <= self.pipeline_parallel_size
        self.hf_config = get_hf_config(
            self.model, trust_remote_code=self.trust_remote_code
        )
        num_hidden_layers = getattr(self.hf_config, "num_hidden_layers", None)
        if num_hidden_layers is not None:
            assert num_hidden_layers >= self.pipeline_parallel_size, (
                f"num_hidden_layers ({num_hidden_layers}) must be >= "
                f"pipeline_parallel_size ({self.pipeline_parallel_size})"
            )
        if self.hf_overrides:
            self.hf_config.update(self.hf_overrides)
            logger.info("Applied HF config overrides: %s", self.hf_overrides)
        _normalize_minimax_m3_text_config(self.hf_config)
        # Multimodal config (full config with vision_config) for vision encoder init
        self.multimodal_config = getattr(self.hf_config, "_multimodal_config", None)
        _normalize_moe_config_fields(self.hf_config, self.model)
        # transformers 5+ exposes rope_parameters; <5 often only rope_scaling + rope_theta.
        # Synthesize when missing or None so GPT-OSS YaRN (rope_type in rope_scaling) is preserved.
        if getattr(self.hf_config, "rope_parameters", None) is None:
            # Compatible with transformers < 5
            rope_params = getattr(self.hf_config, "rope_scaling", None) or {}
            rope_params = dict(rope_params)
            # rope_theta: GPT-OSS / LLaMA-style configs keep it on the root in <5
            rope_params["rope_theta"] = getattr(self.hf_config, "rope_theta", None)
            # rope_type: must NOT overwrite rope_scaling["rope_type"] (e.g. GPT-OSS YaRN).
            # transformers 4.x has no top-level rope_type; getattr(..., "default") was wrong.
            if "rope_type" not in rope_params and "type" in rope_params:
                rope_params["rope_type"] = rope_params["type"]
            if "rope_type" not in rope_params:
                rope_params["rope_type"] = getattr(
                    self.hf_config, "rope_type", "default"
                )
            self.hf_config.rope_parameters = rope_params

        self.generation_config = get_generation_config(self.model)
        if self.generation_config is not None:
            if (
                eos_ids := getattr(self.generation_config, "eos_token_id", None)
            ) is not None:
                self.stop_token_ids = [eos_ids] if isinstance(eos_ids, int) else eos_ids
        self.quant_config = QuantizationConfig(
            self.hf_config,
            self.online_quant_config,
        )
        # In plugin mode, supplement exclude_layers with vLLM's ignored_layers when
        # the HF quant config didn't produce any exclusions (non-quark quant methods).
        if (
            self.plugin_config is not None
            and self.plugin_config.vllm_config is not None
            and len(self.quant_config.exclude_layers) == 0
        ):
            vllm_ignored = getattr(
                self.plugin_config.vllm_config.quant_config, "ignored_layers", []
            )
            self.quant_config.exclude_layers = list(vllm_ignored)
        hf_config_max_position_embeddings = getattr(
            self.hf_config, "max_position_embeddings", 8192
        )
        if self.max_model_len is None:
            self.max_model_len = hf_config_max_position_embeddings
        else:
            self.max_model_len = min(
                self.max_model_len, hf_config_max_position_embeddings
            )
        # assert self.max_num_batched_tokens >= self.max_model_len
        if self.long_prefill_token_threshold > 0:
            if self.long_prefill_token_threshold > self.max_model_len:
                raise ValueError(
                    f"long_prefill_token_threshold "
                    f"({self.long_prefill_token_threshold}) cannot be greater "
                    f"than max_model_len ({self.max_model_len})."
                )
            if self.long_prefill_token_threshold < self.kv_cache_block_size:
                raise ValueError(
                    f"long_prefill_token_threshold "
                    f"({self.long_prefill_token_threshold}) must be >= "
                    f"kv_cache_block_size ({self.kv_cache_block_size})."
                )
        if min(self.profiler_delay_iters, self.profiler_max_iters) < 0:
            raise ValueError(
                "profiler_delay_iters and profiler_max_iters must be >= 0, "
                f"got {self.profiler_delay_iters} and {self.profiler_max_iters}"
            )
        if not is_plugin_mode():
            if self.torch_profiler_dir is not None:
                os.makedirs(self.torch_profiler_dir, exist_ok=True)
            assert self.torch_profiler_dir is None or os.path.isdir(
                self.torch_profiler_dir
            ), f"torch_profiler_dir {self.torch_profiler_dir} is not a valid directory"

        # only for server mode or plugin mode(vllm)
        # for torch compile policy, plugin mode(vllm) uses the ATOM compile policy
        # for cuda graph capture, plugin mode(vllm) uses the vLLM's cuda graph capture policy
        if (
            not is_plugin_mode()
            or (self.plugin_config is not None and self.plugin_config.is_vllm)
        ) and self.compilation_config.level == CompilationLevel.PIECEWISE:
            self.compilation_config.set_splitting_ops_for_v1()
            # Keep an explicit cudagraph_mode (e.g. FULL); default to
            # PIECEWISE only when unset. splitting_ops/sizes are set either
            # way so the model is still piece-split-compiled at level 3.
            if self.compilation_config.cudagraph_mode is None:
                self.compilation_config.cudagraph_mode = CUDAGraphMode.PIECEWISE
            self.compilation_config.init_with_cudagraph_sizes()

        self.torch_dtype = (
            self.hf_config.dtype
            if getattr(self.hf_config, "dtype", None) is not None
            else torch.bfloat16
        )

        if hasattr(self, "kv_transfer_config") and isinstance(
            self.kv_transfer_config, str
        ):
            import json

            try:
                self.kv_transfer_config = json.loads(self.kv_transfer_config)
            except json.JSONDecodeError:
                import ast

                self.kv_transfer_config = ast.literal_eval(self.kv_transfer_config)

        if self.speculative_config is not None:
            num_spec = self.speculative_config.num_speculative_tokens
            is_dspark = getattr(self.speculative_config, "use_dspark", lambda: False)()
            draft_cfg = self.speculative_config.draft_model_hf_config
            if not is_dspark:
                # Sequential drafters (MTP / Eagle): one drafted token per
                # backbone pass, so the horizon is a small fixed depth.
                max_spec = 4
            else:
                # DSpark is a PARALLEL block drafter: all flavors
                # (inline V4, standalone K3 / Qwen3 / ...) share this path with
                # no per-model constants.
                train_block = getattr(draft_cfg, "dspark_block_size", None)
                if getattr(draft_cfg, "dspark_with_draft", False):
                    # Standalone draft: attends a paged sibling KV holding the
                    # full context, so the hard ceiling is a block cap the
                    # checkpoint may set, else a generous default.
                    max_spec = int(
                        getattr(draft_cfg, "dspark_max_block", None)
                        or _DSPARK_DEFAULT_MAX_BLOCK
                    )
                else:
                    # Inline draft (V4): attends a ROLLING target-KV window, so
                    # the hard ceiling is that window -- beyond it the
                    # [window ++ draft] block attention no longer fits its
                    # context. `sliding_window` may be present-but-None.
                    max_spec = int(
                        getattr(draft_cfg, "sliding_window", None)
                        or _DSPARK_DEFAULT_ROLLING_WINDOW
                    )
                if train_block and num_spec is not None and num_spec > train_block:
                    logger.warning(
                        "num_speculative_tokens=%d exceeds the DSpark draft's "
                        "training block size (%d): accepted, but expect a lower "
                        "accepted length.",
                        num_spec,
                        train_block,
                    )
            if num_spec is None or num_spec < 1 or num_spec > max_spec:
                raise ValueError(
                    f"num_speculative_tokens must be between 1 and {max_spec}, "
                    f"got {num_spec}."
                )

        # DeepSeek V4: paper §3.6.1 mandates classical KV cache block_size =
        # a multiple of lcm(m, m'). For V4-Pro / V4-Flash lcm(4, 128) = 128;
        # we use 2*lcm = 256 so each block holds k1=256/4=64 CSA entries — the
        # FP4 paged-MQA-logits indexer kernels require kv_block_size=64 (so
        # NTPW=4 N-tiles share one physical block, N_PHYS=1). ATOM's
        # BlockManager + slot_mapping math assume one global block_size, so we
        # override `kv_cache_block_size` here when V4 is detected; the V4
        # attention builder enforces the same value.
        #
        # NOTE: cannot use `hf_config.model_type` for detection — `_CONFIG_REGISTRY`
        # maps "deepseek_v4" → "deepseek_v3" so model_type reads as "deepseek_v3".
        # Use the preserved `architectures` field (re-injected by get_hf_config,
        # line 567) which keeps the original "DeepseekV4ForCausalLM[NextN]" name.
        arches = getattr(self.hf_config, "architectures", None) or []
        is_deepseek_v4 = any("DeepseekV4" in str(a) for a in arches)
        if is_deepseek_v4:
            v4_block_size = 256
            if self.kv_cache_block_size != v4_block_size:
                self.kv_cache_block_size = v4_block_size

        # GLM-5.3-Flash's indexer caches one pooled key per `index_kpool`
        # tokens, so a KV block of B tokens needs only B // index_kpool index
        # rows. Those rows are what `deepgemm_fp8_paged_mqa_logits` pages over,
        # and it is correct only in its preshuffled layout, which requires the
        # row count per block to be a multiple of 16. B = 16 would give 4 rows
        # and force the index cache to be padded to one row per TOKEN, wasting
        # `index_kpool - 1` of every `index_kpool` rows. B = 64 gives exactly
        # 16. Same reasoning and same mechanism as the V4 override above: the
        # BlockManager and slot_mapping assume one global block size, so it is
        # set here and the attention builder sizes the index cache from it.
        is_glm5_next = any("Glm5Next" in str(a) for a in arches)
        if is_glm5_next:
            unsupported_features = []
            if self.prefill_context_parallel_size > 1:
                unsupported_features.append("PCP")
            if self.decode_context_parallel_size > 1:
                unsupported_features.append("DCP")
            if self.speculative_config is not None:
                unsupported_features.append("speculative decoding")
            if self.enable_tbo or self.enable_tbo_decode:
                unsupported_features.append("TBO")
            if unsupported_features:
                raise ValueError(
                    "GLM-5.3-Flash text serving does not yet support "
                    f"{', '.join(unsupported_features)}"
                )
            index_kpool = int(getattr(self.hf_config, "index_kpool", 1) or 1)
            if index_kpool > 1:
                glm5_block_size = glm5_kpool_block_size(index_kpool)
                if self.kv_cache_block_size != glm5_block_size:
                    self.kv_cache_block_size = glm5_block_size

                # Turning sparsity off is an exact A/B only at or below
                # `index_topk`; past it the dense/token-granular fallback is a
                # DIFFERENT answer, not a slower one. Validate that at startup
                # rather than per forward: `max_model_len` already bounds
                # `max_seqlen_k`, so one check here replaces a branch on a
                # 45-layer hot path, and it fails the launch cleanly instead of
                # raising inside the model and taking the engine down with it
                # mid-batch -- which is what both `envs.py` and
                # `docs/environment_variables.md` describe as refusing the
                # request.
                index_topk = int(getattr(self.hf_config, "index_topk", 0) or 0)
                sparsity_off = []
                if not envs.ATOM_GLM5_KPOOL:
                    sparsity_off.append("ATOM_GLM5_KPOOL=0")
                if envs.ATOM_GLM5_FORCE_DENSE_MLA:
                    sparsity_off.append("ATOM_GLM5_FORCE_DENSE_MLA=1")
                if sparsity_off and index_topk and self.max_model_len > index_topk:
                    raise ValueError(
                        f"GLM-5.3-Flash: {', '.join(sparsity_off)} turns the "
                        "pooled sparse selection off, which matches the pooled "
                        f"path only while sequences stay at or below "
                        f"index_topk={index_topk}. max_model_len is "
                        f"{self.max_model_len}. Lower --max-model-len to "
                        f"{index_topk} for the comparison, or drop the "
                        "override."
                    )

        # Keep ``None`` intact until the model architecture is known so an
        # omitted index-cache option remains distinguishable from an explicit
        # fp8/bf16 override. Native single-node V4 defaults to the FP4 indexer
        # except on gfx942. Plugin proxy pools and KV-transfer region maps do
        # not yet describe the separate FP4 scale pool, so those integrations
        # retain FP8 until their layouts support it. Every other model keeps
        # the historical KV-cache-dtype default.
        if self.index_cache_dtype is None and is_deepseek_v4:
            if self.plugin_config is None and not self.kv_transfer_config:
                from aiter.jit.utils.chip_info import get_gfx

                self.index_cache_dtype = "fp8" if get_gfx() == "gfx942" else "fp4"
            else:
                self.index_cache_dtype = "fp8"
        elif self.index_cache_dtype is None:
            self.index_cache_dtype = self.kv_cache_dtype

    def compute_hash(self) -> str:
        """
        WARNING: Whenever a new field is added to this config,
        ensure that it is included in the factors list if
        it affects the computation graph.

        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        """
        factors: list[Any] = []

        # summarize vllm config
        vllm_factors: list[Any] = []
        if self.quant_config:
            vllm_factors.append(self.quant_config.compute_hash())

        if self.compilation_config:
            vllm_factors.append(self.compilation_config.compute_hash())

        if self.parallel_config:
            vllm_factors.append(self.parallel_config.compute_hash())

        factors.append(vllm_factors)
        factors.append(self.tensor_parallel_size)
        # PCP changes the compiled graph: when pcp>1 the indexer runs through the
        # opaque `indexer_with_output` op (whose identity output is fed as the MLA
        # query) and the indexer takes the round-robin all-gather / separate-rope
        # path. A pcp1 vs pcp2 run over the same model+source otherwise hashes
        # identically, so without this factor pcp2 loads pcp1's cached artifact
        # (no indexer op) and trips copy_misaligned_inputs / assert_size_stride at
        # runtime — the same stale-artifact hazard documented for the vocab-embed
        # flag below.
        factors.append(self.prefill_context_parallel_size)
        factors.append(self.enable_dp_attention)
        factors.append(self.index_cache_dtype)
        text_config = getattr(self.hf_config, "text_config", self.hf_config)
        factors.append(
            (
                getattr(
                    text_config,
                    "use_index_cache",
                    getattr(self.hf_config, "use_index_cache", False),
                ),
                getattr(
                    text_config,
                    "index_topk_freq",
                    getattr(self.hf_config, "index_topk_freq", None),
                ),
                getattr(
                    text_config,
                    "index_topk_pattern",
                    getattr(self.hf_config, "index_topk_pattern", None),
                ),
                getattr(
                    text_config,
                    "index_skip_topk_offset",
                    getattr(self.hf_config, "index_skip_topk_offset", None),
                ),
            )
        )
        # Vocab-embedding replication (ATOM_REPLICATE_VOCAB_EMBED) changes both the
        # embed weight shape ([vocab] vs [vocab/tp]) and the embed op (local
        # F.embedding vs masked-embedding + all-reduce), so it alters the compiled
        # graph and MUST be part of its key. Without this, toggling the flag — or
        # deploying it on top of a cache built with the other setting — reuses a
        # stale artifact and trips assert_size_stride at runtime.
        factors.append(bool(envs.ATOM_REPLICATE_VOCAB_EMBED))

        hash_str = hashlib.md5(
            str(factors).encode(), usedforsecurity=False
        ).hexdigest()[:10]
        return hash_str


_current_atom_config: Config | None = None


def set_current_atom_config(atom_config: Config):
    global _current_atom_config
    _current_atom_config = atom_config


def _get_current_atom_config_from_vllm_forward_context() -> Config | None:
    # In vLLM plugin mode (especially speculative decode), main/draft models
    # can coexist in one process. Resolve per-forward config first to avoid
    # reading a stale global singleton.
    try:
        from vllm.forward_context import (
            get_forward_context as get_vllm_forward_context,
        )
        from vllm.forward_context import (
            is_forward_context_available,
        )
    except (ImportError, AttributeError):
        return None
    if not is_forward_context_available():
        return None
    try:
        return get_vllm_forward_context().additional_kwargs.get("atom_config")
    except (ImportError, AttributeError):
        return None


def get_current_atom_config() -> Config:
    # Try to get the atom config from forward context first in vLLM plugin mode.
    if is_vllm():
        forward_atom_config = _get_current_atom_config_from_vllm_forward_context()
        if forward_atom_config is not None:
            return forward_atom_config
    assert _current_atom_config is not None, "Current atom config is not set"
    return _current_atom_config


@contextmanager
def use_custom_atom_config(custom_atom_config: Config):
    # Temporarily masquerade the custom atom_config as the current atom_config
    # for the current context and restore upon exit
    global _current_atom_config
    prev = _current_atom_config
    _current_atom_config = custom_atom_config
    try:
        yield custom_atom_config
    finally:
        _current_atom_config = prev
