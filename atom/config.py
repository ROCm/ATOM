# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import enum
import hashlib
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, ClassVar, Optional, Union

import torch
from atom.utils import envs, get_open_port
from atom.utils.distributed.utils import stateless_init_torch_distributed_process_group
from torch.distributed import ProcessGroup, ReduceOp
from transformers import AutoConfig, PretrainedConfig, GenerationConfig

from aiter import QuantType
from aiter.dist.parallel_state import get_dp_group
from aiter.utility.dtypes import d_dtypes

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


class QuantizationConfig(dict):
    def __init__(
        self,
        quant_type=QuantType.No,
        quant_dtype=torch.bfloat16,
        is_dynamic=True,
        quant_name="",
        quant_method=None,
    ):
        super().__init__()
        self["quant_type"] = quant_type if quant_type is not None else QuantType.No
        self["quant_dtype"] = quant_dtype if quant_dtype is not None else torch.bfloat16
        self["quant_name"] = quant_name
        self["is_dynamic"] = is_dynamic
        self["quant_method"] = quant_method

    def get_name(self):
        return self["quant_name"]

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
        factors.append(self["quant_type"])
        factors.append(self["quant_dtype"])
        factors.append(self["quant_name"])
        factors.append(self["is_dynamic"])
        factors.append(self["quant_method"])
        str_factors = str(factors)
        # assert_hashable(str_factors)
        return hashlib.sha256(str(factors).encode()).hexdigest()


def get_quant_config(config: PretrainedConfig) -> QuantizationConfig:
    torch_dtype = getattr(config, "torch_dtype", "bf16")
    orig_quant_config = getattr(config, "quantization_config", None)
    if orig_quant_config is None:
        return QuantizationConfig(
            quant_type=QuantType.No,
            quant_dtype=torch_dtype,
        )

    quant_method = orig_quant_config.get("quant_method", None)
    RE_QUANT_BLOCKSIZE = r"\'(?:group_size|weight_block_size)\'\:\s*(?:\[\n*)\s*(\d+),"
    orig_quant_config_str = str(orig_quant_config)
    if quant_method == "compressed-tensors" or "channel'," in orig_quant_config_str:
        quant_type = QuantType.per_Token
    elif group_size := re.search(RE_QUANT_BLOCKSIZE, orig_quant_config_str):
        group_size = int(group_size.group(1))
        assert group_size in (32, 128), f"Unsupported group size {group_size}"
        if group_size == 128:
            quant_type = QuantType.per_1x128
        elif group_size == 32:
            quant_type = QuantType.per_1x32
    else:
        quant_type = QuantType.per_Tensor

    RE_QUANT_DTYPE = r"\'(?:d?type|weight_dtype|quant_method)\'\:\s*\'(\w+)\'"
    quant_dtype = None
    m = re.search(RE_QUANT_DTYPE, orig_quant_config_str)
    if m and m.group(1).lower() in ["fp8", "fp4", "int8", "int4", "fp8_e4m3", "mxfp4"]:
        dtype = m.group(1).lower().split("_")[0]
        if dtype == "mxfp4":
            dtype = "fp4"
        if dtype.endswith("4"):
            dtype += "x2"
        quant_dtype = d_dtypes[dtype]
    else:
        bit_match = re.search(r"\'(?:num_)?bits\'\:\s*(\d+)", orig_quant_config_str)
        if bit_match:
            bit = int(bit_match.group(1))
            dtype_match = re.search(RE_QUANT_DTYPE, orig_quant_config_str)
            if dtype_match:
                dtype = dtype_match.group(1).lower()
                dtype_prefix = "i" if dtype.startswith("int") else "fp"
            else:
                dtype_prefix = "i"
            quant_dtype_str = (
                f"{dtype_prefix}{bit}" if bit != 4 else f"{dtype_prefix}{bit}x2"
            )
            quant_dtype = d_dtypes.get(quant_dtype_str, None)
    assert (
        quant_dtype is not None
    ), f"Cannot parse quant dtype from {orig_quant_config_str}"
    if quant_dtype == d_dtypes["fp4x2"]:
        quant_type = QuantType.per_1x32

    RE_STATIC_QUANT = r"\'(?:activation_scheme)\'\:\s*\'(static)\'"
    if re.search(RE_STATIC_QUANT, orig_quant_config_str):
        is_dynamic = False
    else:
        is_dynamic = True
    return QuantizationConfig(
        quant_type, quant_dtype, is_dynamic, quant_method=quant_method
    )


_CONFIG_REGISTRY: dict[str, str] = {
    "deepseek_v32": "deepseek_v3",
}


def get_hf_config(model: str) -> PretrainedConfig:
    config_dict, _ = PretrainedConfig.get_config_dict(
        model,
    )
    model_type = config_dict.get("model_type")

    def _get_hf_token() -> str | None:
        token = os.getenv("HF_TOKEN")
        if token and token.strip():
            return token
        return None

    if model_type in _CONFIG_REGISTRY:
        config_class = AutoConfig.for_model(_CONFIG_REGISTRY[model_type])
        return config_class.from_pretrained(
            model,
            # revision=revision,
            # code_revision=code_revision,
            token=_get_hf_token(),
        )
    return AutoConfig.from_pretrained(model)


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
        # Only override with env vars if not already set to non-default values
        # This allows programmatic configuration to take precedence
        import os

        if os.getenv("ATOM_DP_SIZE") is not None:
            self.data_parallel_size = envs.ATOM_DP_SIZE
        if os.getenv("ATOM_DP_RANK") is not None:
            self.data_parallel_rank = envs.ATOM_DP_RANK
        if os.getenv("ATOM_DP_RANK_LOCAL") is not None:
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
        if hf_config.model_type == "deepseek_v3":
            hf_config.model_type = "deepseek_mtp"
        if hf_config.model_type == "deepseek_mtp":
            # DeepSeek MTP typically uses only 1 layer that gets reused
            n_predict = getattr(hf_config, "num_nextn_predict_layers", 1)
            # Override to use only 1 layer if config says otherwise
            if n_predict != 1:
                logger.warning(f"Overriding num_nextn_predict_layers from {n_predict} to 1 "
                             "(MTP typically uses 1 layer that gets reused)")
                n_predict = 1
            hf_config.update({
                "n_predict": n_predict,
                "num_nextn_predict_layers": n_predict,
                "architectures": ["DeepSeekMTPModel"]
            })

    def __repr__(self) -> str:
        method = self.method
        num_spec_tokens = self.num_speculative_tokens
        return f"SpeculativeConfig({method=}, {num_spec_tokens=})"

@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 16384
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
    torch_profiler_dir: str | None = os.getenv("ATOM_TORCH_PROFILER_DIR", None)
    compilation_config: CompilationConfig = field(default_factory=CompilationConfig)
    quant_config: QuantizationConfig = field(
        default_factory=lambda: QuantizationConfig()
    )
    asyncio_mode: bool = False
    load_dummy: bool = False
    enable_expert_parallel: bool = False
    master_addr: str = "127.0.0.1"
    graph_bs: Optional[list[int]] = None
    enable_dp_attention: bool = False
    torch_dtype: torch.dtype = field(init=False)
    speculative_config: Optional[SpeculativeConfig] = None

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
        assert (
            self.kv_cache_block_size % 16 == 0 or self.kv_cache_block_size == 1
        ), f"kv_cache_block_size ({self.kv_cache_block_size}) must be a multiple of 16 or 1"
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = get_hf_config(self.model)
        self.generation_config = get_generation_config(self.model)
        if self.generation_config is not None:
            if (eos_ids := getattr(self.generation_config, "eos_token_id", None)) is not None:
                self.stop_token_ids = [eos_ids] if isinstance(eos_ids, int) else eos_ids
        self.quant_config = get_quant_config(self.hf_config)
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
        if self.torch_profiler_dir is not None:
            os.makedirs(self.torch_profiler_dir, exist_ok=True)
        assert self.torch_profiler_dir is None or os.path.isdir(
            self.torch_profiler_dir
        ), f"torch_profiler_dir {self.torch_profiler_dir} is not a valid directory"
        if self.compilation_config.level == CompilationLevel.PIECEWISE:
            self.compilation_config.set_splitting_ops_for_v1()
            self._set_cudagraph_sizes()
            self.compilation_config.cudagraph_mode = CUDAGraphMode.PIECEWISE
            self.compilation_config.init_with_cudagraph_sizes()
        self.torch_dtype = (
            self.hf_config.torch_dtype
            if getattr(self.hf_config, "torch_dtype", None) is not None
            else torch.bfloat16
        )

        if self.speculative_config is not None:
            if self.speculative_config.num_speculative_tokens != 1:
                raise ValueError(
                    f"num_speculative_tokens must be 1, got {self.speculative_config.num_speculative_tokens}. "
                    "Only num_speculative_tokens=1 is currently supported."
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
    # for MoE to check
    import os

    os.environ["ATOM_ENFORCE_EAGER"] = "1" if atom_config.enforce_eager else "0"


def get_current_atom_config() -> Config:
    assert _current_atom_config is not None, "Current atom config is not set"
    return _current_atom_config
