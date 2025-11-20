import enum
import hashlib
import os
import re
from dataclasses import dataclass, field
from typing import Any, ClassVar, Optional, Union

import torch
from aiter import QuantType
from aiter.utility.dtypes import d_dtypes
from transformers import AutoConfig, PretrainedConfig

@dataclass
class KVCacheTensor:
    """
    A class for specifying how the workers should initialize the KV cache.
    """
    layer_num: int
    k_cache: torch.Tensor=torch.tensor([])
    v_cache: torch.Tensor=torch.tensor([])
    k_scale: torch.Tensor=None
    v_scale: torch.Tensor=None

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

    static_forward_context: dict[str, Any] = field(default_factory=dict,
                                                   init=False)

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
    ):
        super().__init__()
        self["quant_type"] = quant_type if quant_type is not None else QuantType.No
        self["quant_dtype"] = quant_dtype if quant_dtype is not None else torch.bfloat16
        self["quant_name"] = quant_name
        self["is_dynamic"] = is_dynamic

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
    return QuantizationConfig(quant_type, quant_dtype, is_dynamic)


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


@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: PretrainedConfig = field(init=False)
    bos_token_id: int = -1
    eos_token_id: int = -1
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
        self.quant_config = get_quant_config(self.hf_config)
        self.max_model_len = min(
            self.max_model_len, self.hf_config.max_position_embeddings
        )
        assert self.max_num_batched_tokens >= self.max_model_len
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

        factors.append(vllm_factors)
        factors.append(self.tensor_parallel_size)

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
