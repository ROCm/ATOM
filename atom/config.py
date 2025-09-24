import os
import re
from dataclasses import dataclass, field
from typing import Optional

import torch
from aiter import QuantType
from aiter.utility.dtypes import d_dtypes
from transformers import AutoConfig, PretrainedConfig


@dataclass
class CompilationConfig:
    level: int = 0
    """The level of compilation:

    - 0: no compilation.
    - 1: dynamo as is.
    - 2: dynamo once.
    - 3: piecewise compilation."""
    use_cudagraph: bool = field(default_factory=lambda: 0)
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


    def __post_init__(self):
        if self.level not in {0, 1, 2, 3}:
            raise ValueError("level must in 0-3")
        if not self.cuda_graph_sizes:
            self.cuda_graph_sizes = [512]


class QuantizationConfig(dict):
    def __init__(
        self, quant_type=QuantType.No, quant_dtype=torch.bfloat16, is_dynamic=True, quant_name=""
    ):
        super().__init__()
        self["quant_type"] = quant_type if quant_type is not None else QuantType.No
        self["quant_dtype"] = quant_dtype if quant_dtype is not None else torch.bfloat16
        self["quant_name"] = quant_name
        self["is_dynamic"] = is_dynamic

    def get_name(self):
        return self["quant_name"]


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
    if m and m.group(1).lower() in ["fp8", "fp4", "int8", "int4", "fp8_e4m3"]:
        dtype = m.group(1).lower().split("_")[0]
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

    RE_STATIC_QUANT = r"\'(?:activation_scheme)\'\:\s*\'(static)\'"
    if re.search(RE_STATIC_QUANT, orig_quant_config_str):
        is_dynamic = False
    else:
        is_dynamic = True
    return QuantizationConfig(quant_type, quant_dtype, is_dynamic)


@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos_token_id: int = -1
    kvcache_block_size: int = 16
    num_kvcache_blocks: int = -1
    kv_cache_dtype: str = "bf16"
    enable_prefix_caching: bool = False
    port: int = 8006
    torch_profiler_dir: str | None = os.getenv("ATOM_TORCH_PROFILER_DIR", None)
    compilation_config: CompilationConfig = field(default_factory=CompilationConfig)
    quant_config: QuantizationConfig = field(default_factory=lambda: QuantizationConfig())
    asyncio_mode: bool = False
    master_addr: str = "127.0.0.1"

    def __post_init__(self):
        # assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 16 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.quant_config = get_quant_config(self.hf_config)
        self.max_model_len = min(
            self.max_model_len, self.hf_config.max_position_embeddings
        )
        assert self.max_num_batched_tokens >= self.max_model_len
        assert self.torch_profiler_dir is None or os.path.isdir(
            self.torch_profiler_dir
        ), f"torch_profiler_dir {self.torch_profiler_dir} is not a valid directory"
