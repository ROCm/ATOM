# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import enum
import fnmatch
import hashlib
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Optional, Union

import torch
from aiter import QuantType
from atom.quant_spec import (
    LayerQuantConfig,
    get_quant_parser,
)
from atom.utils import envs, get_open_port
from atom.utils.distributed.utils import stateless_init_torch_distributed_process_group
from torch.distributed import ProcessGroup, ReduceOp
from transformers import AutoConfig, GenerationConfig, PretrainedConfig

# plugin-related utilities
from atom.plugin import is_plugin_mode, is_vllm
from atom.plugin.config import PluginConfig
from atom.utils import resolve_obj_by_qualname

logger = logging.getLogger("atom")


@dataclass
class KVCacheTensor:
    """
    A class for specifying how the workers should initialize the KV cache.
    """

    layer_num: int
    k_cache: torch.Tensor = torch.tensor([])
    v_cache: torch.Tensor = torch.tensor([])
    k_scale: torch.Tensor = None
    v_scale: torch.Tensor = None


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

    def decode_mode(self) -> "CUDAGraphMode":
        return CUDAGraphMode(self.value[0]) if self.separate_routine() else self

    def mixed_mode(self) -> "CUDAGraphMode":
        return CUDAGraphMode(self.value[1]) if self.separate_routine() else self

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
    cudagraph_capture_sizes: Optional[list[int]] = None

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
    cudagraph_mode: Optional[CUDAGraphMode] = None
    """
    The mode of the cudagraph:

    - NONE, no cudagraph capture.
    - PIECEWISE. (v1 default)
    - FULL.
    - FULL_DECODE_ONLY.
    - FULL_AND_PIECEWISE.

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

    splitting_ops: Optional[list[str]] = None
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

    compile_sizes: Optional[list[Union[int, str]]] = None
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
            ]


class QuantizationConfig:
    """Model-wide quantization configuration.

    API:
    - ``get_layer_quant_config(prefix)`` -> :class:`LayerQuantConfig`
    - ``global_quant_config`` property -> :class:`LayerQuantConfig`
    - ``quant_type``, ``quant_dtype``, ``is_dynamic`` convenience properties
    """

    def __init__(self, config: PretrainedConfig = None):
        if config is None:
            self.torch_dtype = torch.bfloat16
            self.hf_quant_config = None
            self.global_spec: LayerQuantConfig = LayerQuantConfig()
            self.layer_pattern_specs: list[tuple[str, LayerQuantConfig]] = []
            self.exclude_layers: list[str] = []
            self.quant_method = ""
            return

        # Some HF configs set torch_dtype=None; normalize to bf16 default.
        self.torch_dtype = getattr(config, "torch_dtype", None) or torch.bfloat16
        self.hf_quant_config = getattr(config, "quantization_config", None)

        if self.hf_quant_config is None:
            self.global_spec = LayerQuantConfig(
                quant_type=QuantType.No, quant_dtype=self.torch_dtype
            )
            self.layer_pattern_specs = []
            self.exclude_layers = []
            self.quant_method = ""
            return

        self.quant_method = self.hf_quant_config.get("quant_method", "")

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

    def get_layer_quant_config(self, layer_name: str) -> LayerQuantConfig:
        """Return the :class:`LayerQuantConfig` for *layer_name*.

        Resolution order:
        1. Check exclude list -> ``LayerQuantConfig.no_quant()``.
        2. fnmatch-style pattern match in ``layer_pattern_specs``.
        3. Fall back to ``global_spec``.
        """
        # 1. Exclude list
        if self._is_excluded(layer_name):
            return LayerQuantConfig(quant_dtype=self.torch_dtype)

        # 2. Pattern match
        for pattern, spec in self.layer_pattern_specs:
            if "*" not in pattern:
                if layer_name in pattern:
                    return spec
            elif fnmatch.fnmatch(layer_name, pattern):
                return spec

        # 3. Global default
        return self.global_spec

    # -- convenience properties (delegate to global_spec) ---------------------

    @property
    def quant_type(self) -> QuantType:
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
        hash_value = hashlib.sha256(str(factors).encode()).hexdigest()
        return hash_value

    def get_name(self):
        """Returns the quantization method name."""
        return self.quant_method

    # -- internal helpers ---------------------------------------------------

    def _is_excluded(self, layer_name: str) -> bool:
        if layer_name is None or not self.exclude_layers:
            return False
        return any(
            self._matches_exclude(layer_name, ignore_str)
            for ignore_str in self.exclude_layers
        )

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
                if f".{packed_key}" in name:
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

        # Apply model-declared HF-name to ATOM-path translations for exclude entries.
        # Models that have a mismatch between their HF quant config names and ATOM
        # module paths declare `quant_exclude_name_mapping` as a class attribute.
        if quant_exclude_name_mapping:
            self.apply_exclude_name_mapping(quant_exclude_name_mapping)


_CONFIG_REGISTRY: dict[str, str] = {
    "deepseek_v32": "deepseek_v3",
    "glm_moe_dsa": "deepseek_v3",  # GLM 5.0 MoE, structure similar to DeepSeek v3.2
    "kimi_k2": "deepseek_v3",
}


_MULTIMODAL_MODEL_TYPES: dict[str, str] = {
    # Maps multimodal model_type -> key in config_dict for the text sub-config
    "kimi_k25": "text_config",
    "qwen3_5": "text_config",
    "qwen3_5_moe": "text_config",
}

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
        return hf_config

    if model_type in _CONFIG_REGISTRY:
        config_class = AutoConfig.for_model(_CONFIG_REGISTRY[model_type])
        return config_class.from_pretrained(
            model,
            # revision=revision,
            # code_revision=code_revision,
            token=_get_hf_token(),
            trust_remote_code=trust_remote_code,
        )
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


@dataclass
class ParallelConfig:
    data_parallel_size: int = 1
    """Number of data parallel groups. MoE layers will be sharded according to
    the product of the tensor parallel size and data parallel size."""
    data_parallel_size_local: int = 1
    """Number of local data parallel groups."""
    data_parallel_rank: int = 0
    """Rank of the data parallel group."""
    data_parallel_rank_local: Optional[int] = None
    """Local rank of the data parallel group,
    set only in SPMD mode."""
    world_size: int = field(init=False)
    """world_size is TPxPP, it affects the number of workers we create."""
    data_parallel_master_port: int = 29500
    """Port of the data parallel master."""

    data_parallel_base_port: int = get_open_port()

    data_parallel_master_ip: str = "127.0.0.1"

    @property
    def world_size_across_dp(self) -> int:
        """world_size_across_dp is TPxPPxDP, it is the size of the world
        including data parallelism."""
        return self.world_size * self.data_parallel_size

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
        # torch.distributed.all_reduce(tensor, op=ReduceOp.MAX, group=dp_group)
        # from aiter.dist.parallel_state import get_dp_group
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
        # self.data_parallel_master_ip = envs.ATOM_DP_MASTER_IP
        # self.data_parallel_master_port = get_open_port()


@dataclass
class SpeculativeConfig:
    method: Optional[str] = ""
    model: Optional[str] = None
    num_speculative_tokens: Optional[int] = None
    draft_model_hf_config: Optional[PretrainedConfig] = None

    def __post_init__(self):
        if self.draft_model_hf_config is None:
            self.draft_model_hf_config = AutoConfig.from_pretrained(self.model)
        self.hf_config_override(self.draft_model_hf_config)

    @staticmethod
    def hf_config_override(hf_config: PretrainedConfig) -> PretrainedConfig:
        if hf_config.model_type in ("deepseek_v3", "glm_moe_dsa"):
            hf_config.model_type = "deepseek_mtp"
        if hf_config.model_type == "qwen3_next":
            hf_config.model_type = "qwen3_next_mtp"

        if hf_config.model_type == "deepseek_mtp":
            # DeepSeek MTP typically uses only 1 layer that gets reused
            n_predict = getattr(hf_config, "num_nextn_predict_layers", 1)
            # Override to use only 1 layer if config says otherwise
            if n_predict != 1:
                logger.warning(
                    f"Overriding num_nextn_predict_layers from {n_predict} to 1 "
                    "(MTP typically uses 1 layer that gets reused)"
                )
                n_predict = 1
            hf_config.update(
                {
                    "n_predict": n_predict,
                    "num_nextn_predict_layers": n_predict,
                    "architectures": ["DeepSeekMTPModel"],
                }
            )
        if hf_config.model_type == "qwen3_next_mtp":
            n_predict = getattr(hf_config, "num_nextn_predict_layers", 1)
            if n_predict != 1:
                logger.warning(
                    f"Overriding num_nextn_predict_layers from {n_predict} to 1 "
                    "(MTP typically uses 1 layer that gets reused)"
                )
                n_predict = 1
            hf_config.update(
                {"n_predict": n_predict, "architectures": ["Qwen3NextMTPModel"]}
            )
        logger.info(f"hf config is: {hf_config}")

    def __repr__(self) -> str:
        method = self.method
        num_spec_tokens = self.num_speculative_tokens
        return f"SpeculativeConfig({method=}, {num_spec_tokens=})"


@dataclass
class Config:
    model: str
    trust_remote_code: bool = False
    max_num_batched_tokens: int = 16384
    scheduler_delay_factor: float = 0.0
    max_num_seqs: int = 512
    max_model_len: int | None = None
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
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
    enable_prefix_caching: bool = False
    port: int = 8006
    torch_profiler_dir: str | None = field(
        default_factory=lambda: envs.ATOM_TORCH_PROFILER_DIR
    )
    compilation_config: CompilationConfig = field(default_factory=CompilationConfig)
    quant_config: QuantizationConfig = field(init=False)
    asyncio_mode: bool = False
    mark_trace: bool = False
    load_dummy: bool = False
    enable_expert_parallel: bool = False
    master_addr: str = "127.0.0.1"
    graph_bs: Optional[list[int]] = None
    enable_dp_attention: bool = False
    torch_dtype: torch.dtype = field(init=False)
    speculative_config: Optional[SpeculativeConfig] = None

    # only use for plugin mode
    plugin_config: Optional[PluginConfig] = None

    def _set_cudagraph_sizes(self):
        if self.compilation_config.cudagraph_capture_sizes:
            self.graph_bs = self.compilation_config.cudagraph_capture_sizes
        else:
            cuda_graph_sizes = self.compilation_config.cuda_graph_sizes
            if len(cuda_graph_sizes) == 1:
                self.graph_bs = [1, 2, 4, 8] + [
                    i for i in range(16, cuda_graph_sizes[0] + 1, 16)
                ]
            elif len(cuda_graph_sizes) > 1:
                self.graph_bs = cuda_graph_sizes

    def __post_init__(self):
        # assert os.path.isdir(self.model)

        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = get_hf_config(
            self.model, trust_remote_code=self.trust_remote_code
        )
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
        self.quant_config = QuantizationConfig(self.hf_config)
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
        if not is_plugin_mode():
            if self.torch_profiler_dir is not None:
                os.makedirs(self.torch_profiler_dir, exist_ok=True)
            assert self.torch_profiler_dir is None or os.path.isdir(
                self.torch_profiler_dir
            ), f"torch_profiler_dir {self.torch_profiler_dir} is not a valid directory"

        # only for server mode or plugin mode(vllm)
        # for torch compile policy, plugin mode(vllm) uses the ATOM compile policy
        # for cuda graph capture, plugin mode(vllm) uses the vLLM's cuda graph capture policy
        if not is_plugin_mode() or (
            self.plugin_config is not None and self.plugin_config.is_vllm
        ):
            if self.compilation_config.level == CompilationLevel.PIECEWISE:
                self.compilation_config.set_splitting_ops_for_v1()
                self._set_cudagraph_sizes()
                self.compilation_config.cudagraph_mode = CUDAGraphMode.PIECEWISE
                self.compilation_config.init_with_cudagraph_sizes()

        self.torch_dtype = (
            self.hf_config.dtype
            if getattr(self.hf_config, "dtype", None) is not None
            else torch.bfloat16
        )

        if self.speculative_config is not None:
            if self.speculative_config.num_speculative_tokens > 4:
                raise ValueError(
                    f"num_speculative_tokens must be between 1 and 4,, got {self.speculative_config.num_speculative_tokens}. "
                )

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
        factors.append(self.enable_dp_attention)

        hash_str = hashlib.md5(
            str(factors).encode(), usedforsecurity=False
        ).hexdigest()[:10]
        return hash_str


_current_atom_config: Optional[Config] = None


def set_current_atom_config(atom_config: Config):
    global _current_atom_config
    _current_atom_config = atom_config


def get_current_atom_config() -> Config:
    assert _current_atom_config is not None, "Current atom config is not set"
    return _current_atom_config
