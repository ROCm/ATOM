import argparse
from typing import List

from atom import AsyncLLMEngine, LLMEngine
from atom.config import CompilationConfig


def parse_size_list(size_str: str) -> List[int]:
    """Parse a string representation of a list into a Python list."""
    import ast

    try:
        return ast.literal_eval(size_str)
    except (ValueError, SyntaxError) as e:
        raise ValueError(f"Error parsing size list: {size_str}") from e


class EngineArgs:
    """Arguments for configuring the LLM Engine."""

    def __init__(
        self,
        model: str = "Qwen/Qwen3-0.6B",
        tensor_parallel_size: int = 1,
        data_parallel_size: int = 1,
        enforce_eager: bool = False,
        enable_prefix_caching: bool = False,
        port: int = 8006,
        kv_cache_dtype: str = "bf16",
        block_size: int = 16,
        max_model_len: int = 8192,
        cudagraph_capture_sizes: str = "[1,2,4,8,16]",
        level: int = 3,
        load_dummy: bool = False,
        enable_expert_parallel: bool = False,
        torch_profiler_dir: str = None,
        enable_dp_attention: bool = False,
    ):
        self.model = model
        self.tensor_parallel_size = tensor_parallel_size
        self.enforce_eager = enforce_eager
        self.enable_prefix_caching = enable_prefix_caching
        self.port = port
        self.kv_cache_dtype = kv_cache_dtype
        self.block_size = block_size
        self.max_model_len = max_model_len
        self.cudagraph_capture_sizes = cudagraph_capture_sizes
        self.level = level
        self.load_dummy = load_dummy
        self.enable_expert_parallel = enable_expert_parallel
        self.torch_profiler_dir = torch_profiler_dir
        self.enable_dp_attention = enable_dp_attention
        self.data_parallel_size = data_parallel_size

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Add engine arguments to an argument parser."""
        # Model configuration
        parser.add_argument(
            "--model", type=str, default="Qwen/Qwen3-0.6B", help="Model name or path."
        )

        parser.add_argument(
            "--tensor-parallel-size",
            "-tp",
            type=int,
            default=1,
            help="Tensor parallel size.",
        )

        parser.add_argument(
            "--enforce-eager",
            action="store_true",
            help="Enforce eager mode execution.",
        )

        parser.add_argument(
            "--enable_prefix_caching",
            action="store_true",
            help="Enable prefix caching.",
        )

        parser.add_argument("--port", type=int, default=8006, help="API server port")

        # KV cache configuration
        parser.add_argument(
            "--kv_cache_dtype",
            choices=["bf16", "fp8"],
            type=str,
            default="bf16",
            help="KV cache type. Default is 'bf16'.",
        )

        parser.add_argument(
            "--block-size", type=int, default=16, help="KV cache block size."
        )

        parser.add_argument(
            "--max-model-len",
            type=int,
            default=8192,
            help="Maximum model context length.",
        )

        # Compilation configuration
        parser.add_argument(
            "--cudagraph-capture-sizes",
            type=str,
            default="[1,2,4,8,16,32,48,64,128,256]",
            help="Sizes to capture cudagraph. Example: [1,2,4,8,16]",
        )

        parser.add_argument(
            "--level", type=int, default=3, help="The level of compilation (0-3)."
        )

        # Advanced options
        parser.add_argument(
            "--load_dummy", action="store_true", help="Skip loading model weights."
        )

        parser.add_argument(
            "--enable-expert-parallel",
            action="store_true",
            help="Enable expert parallel(EP MoE).",
        )

        parser.add_argument(
            "--torch-profiler-dir",
            type=str,
            default=None,
            help="Directory to save torch profiler traces",
        )

        parser.add_argument(
            "--data-parallel-size",
            "-dp",
            type=int,
            default=1,
            help="Data parallel size.",
        )

        parser.add_argument(
            "--enable-dp-attention",
            action="store_true",
            help="Enable DP attention.",
        )
        return parser

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> "EngineArgs":
        """Create an EngineArgs instance from parsed command-line arguments."""
        return cls(
            model=args.model,
            tensor_parallel_size=args.tensor_parallel_size,
            data_parallel_size=getattr(args, "data_parallel_size", 1),
            enforce_eager=args.enforce_eager,
            enable_prefix_caching=args.enable_prefix_caching,
            port=args.port,
            kv_cache_dtype=args.kv_cache_dtype,
            block_size=args.block_size,
            max_model_len=getattr(args, "max_model_len", 8192),
            cudagraph_capture_sizes=args.cudagraph_capture_sizes,
            level=args.level,
            load_dummy=args.load_dummy,
            enable_expert_parallel=args.enable_expert_parallel,
            torch_profiler_dir=args.torch_profiler_dir,
            enable_dp_attention=args.enable_dp_attention,
        )

    def create_engine(self) -> LLMEngine:
        """Create and return an LLMEngine instance with the configured parameters."""
        return LLMEngine(
            self.model,
            enforce_eager=self.enforce_eager,
            tensor_parallel_size=self.tensor_parallel_size,
            kv_cache_dtype=self.kv_cache_dtype,
            kv_cache_block_size=self.block_size,
            max_model_len=self.max_model_len,
            port=self.port,
            load_dummy=self.load_dummy,
            enable_expert_parallel=self.enable_expert_parallel,
            torch_profiler_dir=self.torch_profiler_dir,
            compilation_config=CompilationConfig(
                level=self.level,
                cudagraph_capture_sizes=parse_size_list(self.cudagraph_capture_sizes) if self.cudagraph_capture_sizes else None,
            ),
            data_parallel_size=self.data_parallel_size,
            enable_dp_attention=self.enable_dp_attention,
        )

    def create_async_engine(self) -> AsyncLLMEngine:
        """Create and return an AsyncLLMEngine instance with the configured parameters."""
        return AsyncLLMEngine(
            self.model,
            enforce_eager=self.enforce_eager,
            tensor_parallel_size=self.tensor_parallel_size,
            kv_cache_dtype=self.kv_cache_dtype,
            kv_cache_block_size=self.block_size,
            max_model_len=self.max_model_len,
            port=self.port,
            load_dummy=self.load_dummy,
            enable_expert_parallel=self.enable_expert_parallel,
            torch_profiler_dir=self.torch_profiler_dir,
            asyncio_mode=True,
            compilation_config=CompilationConfig(
                level=self.level,
                cudagraph_capture_sizes=parse_size_list(self.cudagraph_capture_sizes),
            ),
            data_parallel_size=self.data_parallel_size,
            enable_dp_attention=self.enable_dp_attention,
        )
