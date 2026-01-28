# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import argparse
from dataclasses import dataclass, fields
from typing import List, Optional

from atom import LLMEngine
from atom.config import CompilationConfig, SpeculativeConfig


def parse_size_list(size_str: str) -> List[int]:
    """Parse a string representation of a list into a Python list."""
    import ast

    try:
        return ast.literal_eval(size_str)
    except (ValueError, SyntaxError) as e:
        raise ValueError(f"Error parsing size list: {size_str}") from e


@dataclass
class EngineArgs:
    """Arguments for configuring the LLM Engine."""

    model: str = "Qwen/Qwen3-0.6B"
    trust_remote_code: bool = False
    tensor_parallel_size: int = 1
    data_parallel_size: int = 1
    enforce_eager: bool = False
    enable_prefix_caching: bool = False
    port: int = 8006
    kv_cache_dtype: str = "bf16"
    block_size: int = 16
    max_model_len: Optional[int] = None
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    gpu_memory_utilization: float = 0.9
    cudagraph_capture_sizes: str = "[1,2,4,8,16,32,48,64,128,256]"
    level: int = 3
    load_dummy: bool = False
    enable_expert_parallel: bool = False
    torch_profiler_dir: Optional[str] = None
    enable_dp_attention: bool = False
    method: Optional[str] = None
    num_speculative_tokens: int = 1

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Add engine arguments to an argument parser."""
        # Model configuration
        parser.add_argument(
            "--model", type=str, default="Qwen/Qwen3-0.6B", help="Model name or path."
        )
        parser.add_argument(
            "--trust-remote-code",
            action="store_true",
            help="Trust remote code when loading model.",
        )
        parser.add_argument(
            "--tensor-parallel-size",
            "-tp",
            type=int,
            default=1,
            help="Tensor parallel size.",
        )
        parser.add_argument(
            "--data-parallel-size",
            "-dp",
            type=int,
            default=1,
            help="Data parallel size.",
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
        parser.add_argument(
            "--port", 
            type=int, 
            default=8006, 
            help="Engine internal port",
        )
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
            default=None,
            help="Maximum model context length, the default is set to hf_config.max_position_embeddings.",
        )
        parser.add_argument(
            "--cudagraph-capture-sizes",
            type=str,
            default="[1,2,4,8,16,32,48,64,128,256]",
            help="Sizes to capture cudagraph. Example: [1,2,4,8,16]",
        )
        parser.add_argument(
            "--level", type=int, default=3, help="The level of compilation (0-3)."
        )
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
            "--enable-dp-attention",
            action="store_true",
            help="Enable DP attention.",
        )
        parser.add_argument(
            "--method",
            type=str,
            default=None,
            choices=["mtp"],
            help="Speculative method",
        )
        parser.add_argument(
            "--num-speculative-tokens",
            type=int,
            default=1,
            help="Number of speculative tokens to generate per iteration (draft model runs this many times autoregressively)",
        )
        parser.add_argument(
            "--max-num-batched-tokens",
            type=int,
            default=16384,
            help="Maximum number of tokens to batch together in async engine",
        )
        parser.add_argument(
            "--max-num-seqs",
            type=int,
            default=512,
            help="Maximum number of sequences to batch together",
        )
        parser.add_argument(
            "--gpu-memory-utilization",
            type=float,
            default=0.9,
            help="GPU memory utilization (0.0 to 1.0)",
        )

        return parser

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> "EngineArgs":
        """Create an EngineArgs instance from parsed command-line arguments."""
        attrs = [attr.name for attr in fields(cls)]
        engine_args = cls(
            **{attr: getattr(args, attr) for attr in attrs if hasattr(args, attr)}
        )
        return engine_args

    def _get_engine_kwargs(self) -> dict:
        """Get common engine initialization kwargs.
        
        Most fields are directly passed through with the same name.
        Only handles special cases that need transformation.
        """
        kwargs = {f.name: getattr(self, f.name) for f in fields(self) if f.name != "model"}
        
        # Handle special transformations
        kwargs["kv_cache_block_size"] = kwargs.pop("block_size")
        kwargs["compilation_config"] = CompilationConfig(
            level=kwargs.pop("level"),
            cudagraph_capture_sizes=parse_size_list(kwargs.pop("cudagraph_capture_sizes")) if self.cudagraph_capture_sizes else None,
        )
        if self.method:
            kwargs["speculative_config"] = SpeculativeConfig(
                method=kwargs.pop("method"),
                model=self.model,
                num_speculative_tokens=kwargs.pop("num_speculative_tokens"),
            )
        else:
            kwargs.pop("method")
            kwargs.pop("num_speculative_tokens")
            kwargs["speculative_config"] = None
        
        return kwargs

    def create_engine(self) -> LLMEngine:
        """Create and return an LLMEngine instance with the configured parameters."""
        return LLMEngine(self.model, **self._get_engine_kwargs())
