# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import json
import logging
from dataclasses import dataclass, fields

from atom import LLMEngine
from atom.config import (
    CompilationConfig,
    CUDAGraphMode,
    DCPConfig,
    DSparkConfig,
    EPLBConfig,
    ParallelConfig,
    SpeculativeConfig,
)
from atom.model_engine.engine_core_mgr import DP_LB_DEFAULT, DP_LB_STRATEGIES

logger = logging.getLogger("atom")


def parse_size_list(size_str: str) -> list[int]:
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
    decode_context_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    prefill_context_parallel_size: int = 1
    data_parallel_size: int = 1
    data_parallel_size_local: int | None = None
    data_parallel_rank: int = 0
    data_parallel_master_ip: str = "127.0.0.1"
    data_parallel_master_port: int = 29500
    data_parallel_base_port: int | None = None
    enforce_eager: bool = False
    enable_prefix_caching: bool = True
    port: int = 8006
    kv_cache_dtype: str = "bf16"
    index_cache_dtype: str | None = None
    block_size: int = 16
    max_model_len: int | None = None
    max_num_batched_tokens: int = 16384
    long_prefill_token_threshold: int = 0
    attn_prefill_chunk_size: int = 16384
    state_checkpoint_interval_tokens: int = 8192
    state_checkpoint_demand: bool = True
    enable_chunked_prefill: bool = True
    enable_log_stats: bool = True
    throughput_log_interval: float = 10.0
    cache_hit_rate_window: int = 1000
    scheduler_delay_factor: float = 0.0
    max_num_seqs: int = 512
    gpu_memory_utilization: float = 0.9
    cudagraph_capture_sizes: str = "[1,2,4,8,16,32,48,64,128,256]"
    level: int = 3
    cudagraph_mode: str = "FULL"
    load_dummy: str | None = None
    enable_expert_parallel: bool = False
    fake_eplb: bool = False
    torch_profiler_dir: str | None = None
    enable_dp_attention: bool = False
    dp_load_balance: str = DP_LB_DEFAULT
    enable_tbo: str | None = None
    all2all_backend: str | None = None
    moe_backend: str = "standard"
    method: str | None = None
    num_speculative_tokens: int = 1
    kv_transfer_config: str = "{}"
    draft_model: str | None = None
    spec_decode_acceptance_rate: float | None = None
    spec_decode_acceptance_length: float | None = None
    mark_trace: bool = False
    enable_rapidserve: bool = False
    disagg_prefill_max_num_seqs: int | None = None
    disagg_constrained: bool = False
    online_quant_config: dict | None = None
    hf_overrides: dict | None = None
    dspark_config: dict | None = None

    eplb_enable: bool = False
    eplb_config: dict | None = None
    dcp_config: dict | None = None

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Add engine arguments to an argument parser."""
        # Model configuration
        parser.add_argument(
            "--model", type=str, default="Qwen/Qwen3-0.6B", help="Model name or path."
        )
        parser.add_argument(
            "--served-model-name",
            type=str,
            default=None,
            help="Override the model name returned by the API. "
            "If not specified, defaults to the --model value.",
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
            "--pipeline-parallel-size",
            "-pp",
            type=int,
            default=1,
            help="Pipeline parallel size. Splits the model's layers across "
            "stages (world = tp x pp x pcp).",
        )
        parser.add_argument(
            "--prefill-context-parallel-size",
            "-pcp",
            type=int,
            default=1,
            help="Prefill context parallel size. Independent dimension "
            "(world = tp x pcp); splits the sequence during prefill.",
        )
        parser.add_argument(
            "--data-parallel-size",
            "-dp",
            type=int,
            default=1,
            help="Data parallel size.",
        )
        parser.add_argument(
            "--data-parallel-size-local",
            type=int,
            default=None,
            help=(
                "Number of data-parallel ranks to run on THIS node. Defaults "
                "to --data-parallel-size (single-node). Set it lower to give "
                "this node one slice of a multi-node run."
            ),
        )
        parser.add_argument(
            "--data-parallel-rank",
            type=int,
            default=0,
            help=(
                "First GLOBAL data-parallel rank owned by this node. Node 0 "
                "uses 0; the second node of a 2x4 run uses 4."
            ),
        )
        parser.add_argument(
            "--data-parallel-master-ip",
            type=str,
            default="127.0.0.1",
            help="IP of the coordinator node (global DP rank 0).",
        )
        parser.add_argument(
            "--data-parallel-master-port",
            type=int,
            default=29500,
            help=(
                "Rendezvous port for the DP process group. Engine sockets are "
                "derived from it (base = port + 100, 3 ports per DP rank)."
            ),
        )
        parser.add_argument(
            "--data-parallel-base-port",
            type=int,
            default=None,
            help=(
                "Rendezvous port for model-runner distributed init. Set "
                "explicitly for multi-node launches."
            ),
        )
        parser.add_argument(
            "--decode-context-parallel-size",
            "-dcp",
            type=int,
            default=1,
            help="Decode context parallel size. Must divide tensor_parallel_size.",
        )
        parser.add_argument(
            "--enforce-eager",
            action="store_true",
            help="Enforce eager mode execution.",
        )
        parser.add_argument(
            "--enable_prefix_caching",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Enable prefix caching (default: enabled). "
            "Use --no-enable_prefix_caching to disable.",
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
            "--index-cache-dtype",
            "--index_cache_dtype",
            choices=["bf16", "fp8", "fp4"],
            type=str,
            default=None,
            help="Index cache type. Native single-node DeepSeek-V4 defaults to "
            "'fp4' except on gfx942, which defaults to 'fp8'; other models "
            "default to --kv_cache_dtype.",
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
            default="[1,2,4,8,16,32,48,64,128,256,512]",
            help="Sizes to capture cudagraph. Example: [1,2,4,8,16]",
        )
        parser.add_argument(
            "--level", type=int, default=3, help="The level of compilation (0-3)."
        )
        parser.add_argument(
            "--cudagraph-mode",
            type=str,
            default="FULL",
            choices=["NONE", "PIECEWISE", "FULL", "FULL_AND_PIECEWISE", "AF_PIECEWISE"],
            help="CUDA graph runtime mode. FULL = manual whole-forward capture "
            "(default, existing behavior). PIECEWISE = per-piece cudagraph with "
            "attention eager (requires --level 3). AF_PIECEWISE = PIECEWISE where "
            "the attention core is also captured into its own cudagraph with "
            "zero-copy buffers (DeepSeek-V4 DSpark).",
        )
        parser.add_argument(
            "--load_dummy",
            nargs="?",
            const="empty",
            default=None,
            choices=["empty", "zero", "xavier"],
            help="Use dummy weights instead of reading the checkpoint. Bare flag "
            "or '=empty': skip loading (uninitialized, legacy behavior). '=zero': "
            "all weights 0. '=xavier': xavier_uniform_ for bf16 weights and a "
            "constant target magnitude for fp4/fp8 packed weights (finite, "
            "roughly real-scale; fp4 is the validated path).",
        )
        parser.add_argument(
            "--enable-expert-parallel",
            action="store_true",
            help="Enable expert parallel(EP MoE).",
        )
        parser.add_argument(
            "--fake-eplb",
            action="store_true",
            help="Replace MoE router logits with a synthetic uniform "
            "distribution so every expert is selected equally. For "
            "benchmarking the balanced-load upper bound only.",
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
            "--dp-load-balance",
            type=str,
            default=DP_LB_DEFAULT,
            choices=list(DP_LB_STRATEGIES),
            help="Strategy the CoreManager uses to route a request to a DP "
            "engine rank. 'round_robin': legacy request-count-agnostic "
            "rotation. 'least_requests' (default): route to the rank with the "
            "fewest in-flight requests, breaking ties by the lighter in-flight "
            "prompt-token load. 'least_tokens': route to the rank with "
            "the lowest combined in-flight token load (prompt tokens + "
            "per-request token-equivalent, tunable via ATOM_DP_LB_REQ_EQUIV). "
            "Has no effect when data_parallel_size == 1.",
        )
        parser.add_argument(
            "--enable-tbo",
            nargs="?",
            const="prefill",
            default=None,
            choices=["prefill", "all"],
            help="Enable TBO (Two-Batch Overlap) for comm/compute overlap. "
            "'--enable-tbo' or '--enable-tbo prefill': TBO for prefill only. "
            "'--enable-tbo all': TBO for both prefill and decode.",
        )
        parser.add_argument(
            "--all2all-backend",
            nargs="?",
            const="high-throughput",
            default=None,
            choices=["high-throughput", "low-latency"],
            help="All2all backend mode for MORI. "
            "Default is 'high-throughput'. "
            "Use '--all2all-backend low-latency' for AsyncLL MORI kernel overlap.",
        )
        parser.add_argument(
            "--moe-backend",
            type=str,
            default="standard",
            choices=["standard", "mega"],
            help="MoE implementation. 'standard' uses the existing "
            "path and tuned communication-fused kernels when available; "
            "'mega' uses fused FlyDSL MegaMoE.",
        )
        parser.add_argument(
            "--method",
            type=str,
            default=None,
            choices=["mtp", "eagle3", "dspark"],
            help="Speculative method",
        )
        parser.add_argument(
            "--num-speculative-tokens",
            type=int,
            default=1,
            help="Number of speculative tokens to generate per iteration (draft model runs this many times autoregressively)",
        )
        parser.add_argument(
            "--draft-model",
            type=str,
            default=None,
            help="Path to a standalone draft-model checkpoint. Required when "
            "--method eagle3; optional for --method dspark (needed for the "
            "DFlash-backbone drafts such as Kimi-K3-DSpark, omitted for "
            "V4-Pro-DSpark which ships inside the target checkpoint).",
        )
        parser.add_argument(
            "--spec-decode-acceptance-length",
            type=float,
            default=None,
            help="Debug/benchmark knob: force a fixed speculative-decoding mean "
            "acceptance length (AL) in [1, num_speculative_tokens + 1]. When "
            "set, the rejection sampler ignores the real draft/target agreement "
            "and force-accepts draft tokens so the measured accept length "
            "converges to this value. AL counts the target's own guaranteed "
            "token, so it is the same unit as vLLM's synthetic_acceptance_length "
            "and SGLang's SGLANG_SIMULATE_ACC_LEN and a published golden AL can "
            "be passed through unchanged. Only meaningful with a speculative "
            "method; leave unset to disable.",
        )
        parser.add_argument(
            "--spec-decode-acceptance-rate",
            type=float,
            default=None,
            help="The same knob as --spec-decode-acceptance-length, expressed as "
            "a mean acceptance rate in [0, 1] (accepted_draft/total_draft), i.e. "
            "(length - 1) / num_speculative_tokens. Mutually exclusive with "
            "--spec-decode-acceptance-length.",
        )
        parser.add_argument(
            "--max-num-batched-tokens",
            type=int,
            default=16384,
            help="Maximum number of tokens to batch together in async engine",
        )
        parser.add_argument(
            "--long-prefill-token-threshold",
            type=int,
            default=0,
            help=(
                "For chunked prefill, cap a single request's per-step prefill "
                "size at this many tokens. 0 disables the cap (request is only "
                "bounded by max_num_batched_tokens). Useful to interleave long "
                "prefills with decode for lower ITL."
            ),
        )
        parser.add_argument(
            "--attn-prefill-chunk-size",
            type=int,
            default=16384,
            help=(
                "MLA chunked-prefill budget in tokens. Default uses "
                "max_num_batched_tokens."
            ),
        )
        parser.add_argument(
            "--state-checkpoint-interval-tokens",
            type=int,
            default=8192,
            help=(
                "For models with per-request state (DeepSeek-V4 compressor "
                "ring and sliding window, GDN recurrent state), keep a state "
                "checkpoint every N tokens of context so a later prefix hit "
                "can resume there. "
                "A prompt shorter than N publishes nothing, which is what keeps "
                "the feature free on workloads that never reuse a prefix. Must "
                "be a multiple of the prefix-cache hash block size. Prefill "
                "chunks are aligned to these positions, so this also quantizes "
                "chunk boundaries. "
                "0 disables state checkpointing entirely. -1 keeps it on but "
                "places no interval rungs: checkpoints are then taken only "
                "where a request is seen to want one and at each prompt's own "
                "end, which is where agentic traffic actually resumes — every "
                "rung costs the prompt that keeps it an extra prefill chunk, "
                "and on measured traces the interval ladder is ~30x the writes "
                "for reuse the other two placements already reach."
            ),
        )
        parser.add_argument(
            "--state-checkpoint-demand",
            action=argparse.BooleanOptionalAction,
            default=True,
            help=(
                "Let a hit that was refused for want of a checkpoint place a "
                "rung of its own. --no-state-checkpoint-demand leaves the "
                "prompt-end anchor as the only placement. On measured traces a "
                "demand is 47% of all checkpoint writes but reads back 2.8% of "
                "the time, against 85.2% for an anchor, so the rung's write "
                "traffic may cost more in evictions than its reuse is worth."
            ),
        )
        parser.add_argument(
            "--enable_chunked_prefill",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Enable chunked prefill (default: enabled). "
            "Use --no-enable_chunked_prefill to disable.",
        )
        parser.add_argument(
            "--enable-log-stats",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Log the periodic engine-status line (running/waiting reqs, "
            "KV cache usage, prefix cache hit rate, prompt/generation "
            "throughput; default: enabled). Use --no-enable-log-stats to "
            "disable. Applies to offline LLM(...) as well as to the server. "
            "Scoped to that line only: the [MTP Stats] and "
            "[Cache Stats] lines have their own gates (--method mtp and "
            "--enable-prefix-caching) and keep their own cadences.",
        )
        parser.add_argument(
            "--throughput-log-interval",
            type=float,
            default=10.0,
            help="Seconds between engine-status lines (default: 10, matching "
            "vLLM). Must be > 0. Ignored when --no-enable-log-stats.",
        )
        parser.add_argument(
            "--cache-hit-rate-window",
            type=int,
            default=1000,
            help="Requests in the sliding window behind the engine-status "
            "line's prefix cache hit rate (default: 1000, matching vLLM). "
            "Must be > 0. Only the status line is windowed; /metrics and "
            "[Cache Stats] stay cumulative.",
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

        parser.add_argument(
            "--kv-transfer-config",
            type=str,
            default="{}",
            help="KV transfer config as JSON string.",
        )

        parser.add_argument(
            "--scheduler-delay-factor",
            type=float,
            default=0.0,
            help="Apply a delay (of delay factor multiplied by previous"
            "prompt latency) before scheduling next prompt.",
        )
        parser.add_argument(
            "--mark-trace",
            action="store_true",
            help="Enable graph_marker nodes for tracing/profile instrumentation.",
        )
        parser.add_argument(
            "--enable-rapidserve",
            action="store_true",
            help="Enable intra-GPU prefill/decode disaggregation. "
            "Defaults to unconstrained mode (plain separate streams, "
            "no CU masking). Pass --disagg-constrained to enable "
            "CU-masked streams + shm coordination.",
        )
        parser.add_argument(
            "--disagg-prefill-max-num-seqs",
            type=int,
            default=None,
            help="Max sequences per prefill batch in disagg mode. "
            "Defaults to --max-num-seqs when not set.",
        )
        parser.add_argument(
            "--disagg-constrained",
            action="store_true",
            help="With --enable-rapidserve, enable CU-masked streams and "
            "shm-based prefill/decode coordination. Default (off) "
            "uses plain separate streams with no CU masking.",
        )
        parser.add_argument(
            "--online_quant_config",
            type=json.loads,
            default=None,
            help=(
                "Online quantization config as a JSON string. "
                "Supported quantization formats: ptpc_fp8, mxfp4. "
                "The JSON object has three fields "
                "(at least one must be provided):\n"
                '  - "global_quant_config": str, default quantization '
                "format applied to all layers.\n"
                '  - "layer_quant_config": dict, per-layer overrides '
                "using glob patterns as keys. "
                "Overrides global_quant_config for matched layers.\n"
                '  - "exclude_layer": str or list[str], layer name '
                "patterns to exclude from quantization.\n"
                "Example:\n"
                """  '{"global_quant_config": "ptpc_fp8", """
                """"layer_quant_config": {"*expert*": "mxfp4"}, """
                """"exclude_layer": "lm_head"}'"""
            ),
        )
        parser.add_argument(
            "--hf-overrides",
            type=json.loads,
            default=None,
            help=(
                "JSON object of HF config attributes to override after loading "
                "the model config. Example: "
                '\'{"use_index_cache": true, "index_topk_freq": 4}\''
            ),
        )
        parser.add_argument(
            "--dspark-config",
            type=json.loads,
            default=None,
            help=(
                "DSpark dynamic config as a JSON dict, parsed straight into a "
                "DSparkConfig object (no env vars). Supported keys:\n"
                '  - "confidence_schedule": bool, enable confidence-scheduled '
                "verification (per-request verify length ell_r).\n"
                '  - "ragged": bool, enable per-request ragged verify '
                "(no batch-level q padding).\n"
                '  - "ragged_graph_sizes": str, comma-separated per-seq CUDA-graph '
                'query-length buckets to capture, e.g. "1,3,6" or "8".\n'
                '  - "q_buckets": str, CUDA-graph query-length buckets for the '
                "older batch-uniform q-bucket verify path.\n"
                '  - "disable_sps_calib": bool, skip SPS calibration and use the '
                "synthetic stub.\n"
                "Example:\n"
                """  '{"confidence_schedule": true, "ragged": true, """
                """"ragged_graph_sizes": "8"}'"""
            ),
        )
        dcp_group = parser.add_argument_group("DCP options")
        dcp_group.add_argument(
            "--dcp-config",
            type=json.loads,
            default=None,
            help=(
                "DCP (Decode Context Parallel) knobs as one JSON dict, parsed "
                "straight into DCPConfig (no per-field flags); unknown keys "
                "raise. Details and constraints: "
                "docs/context_parallel_guide.md.\n"
                '  "interleave_size" (int, 1): KV interleave granularity S -- '
                "token i lives on rank (i // S) %% W; 1 = round-robin.\n"
                '  "enable_query_replication" (bool, TRUE): drop the per-step '
                "decode AllGather Q by replicating q_proj at load time.\n"
                '  "enable_project_before_merge" (bool, TRUE): project V '
                "before the output merge, shrinking it by "
                "kv_lora_rank/v_head_dim.\n"
                "  \"comm_backend\" (str, a2a): 'a2a' = one all-to-all; "
                "'ag_rs' = AllGather LSE + ReduceScatter output.\n"
                "The last three default to the NEW behaviour (and the middle "
                "two auto-disable where unsupported), so a control run must "
                "pass the old values explicitly -- passing nothing re-runs the "
                "new path.\n"
                'Example: \'{"interleave_size": 16, "enable_query_replication": true}\''
            ),
        )
        eplb_group = parser.add_argument_group("EPLB options")
        eplb_group.add_argument(
            "--eplb-enable",
            "--enable-eplb",
            action="store_true",
            help="Enable EPLB runtime load monitoring and expert rebalance.",
        )
        eplb_group.add_argument(
            "--eplb-config",
            type=json.loads,
            default=None,
            help=(
                "EPLB config as a JSON dict, parsed straight into an EPLBConfig "
                "object (no per-field flags). --eplb-enable turns EPLB on; "
                "--eplb-config only tunes it. Supported keys:\n"
                '  - "load_window_size": int, non-dummy forwards accumulated '
                "for EPLB load stats.\n"
                '  - "rebalance_interval": int, forward-pass interval for '
                "EPLB rebalance gating.\n"
                '  - "rebalance_layers_per_chunk": int, MoE layers migrated '
                "per EPLB rebalance chunk.\n"
                '  - "num_redundant_experts": int, extra physical expert '
                "slots per MoE layer for EPLB replicas.\n"
                '  - "rebalance_min_balancedness": float, skip EPLB '
                "rebalance when balancedness is at least this value.\n"
                '  - "rebalance_balancedness_agg": "min"|"mean", layer '
                "aggregation used by the EPLB balancedness gate.\n"
                '  - "p2p_batch_chunk_size": int, P2P batch chunk size used '
                "while migrating expert weights.\n"
                '  - "placement_policy": "naive"|"biased", how to spend the '
                "redundant budget: 'naive' (spread) or 'biased' (fully "
                "replicate top-K hottest experts to all GPUs).\n"
                "Example:\n"
                """  '{"num_redundant_experts": 8, "placement_policy": """
                """"biased"}'"""
            ),
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
        kwargs = {
            f.name: getattr(self, f.name) for f in fields(self) if f.name != "model"
        }

        # Handle special transformations
        kwargs["kv_cache_block_size"] = kwargs.pop("block_size")
        kwargs["compilation_config"] = CompilationConfig(
            level=kwargs.pop("level"),
            cudagraph_mode=CUDAGraphMode[kwargs.pop("cudagraph_mode")],
            cudagraph_capture_sizes=(
                parse_size_list(kwargs.pop("cudagraph_capture_sizes"))
                if self.cudagraph_capture_sizes
                else None
            ),
        )
        if self.method and self.num_speculative_tokens > 0:
            method = kwargs.pop("method")
            num_spec_tokens = kwargs.pop("num_speculative_tokens")
            draft_model = kwargs.pop("draft_model")
            synthetic_acceptance_rate = kwargs.pop("spec_decode_acceptance_rate")
            synthetic_acceptance_length = kwargs.pop("spec_decode_acceptance_length")
            if method == "eagle3" and not draft_model:
                raise ValueError("--draft-model is required when --method eagle3.")
            if draft_model and method == "mtp":
                raise ValueError(
                    "--draft-model is not supported with --method mtp: the MTP "
                    "draft is loaded from the target checkpoint."
                )
            kwargs["speculative_config"] = SpeculativeConfig(
                method=method,
                model=draft_model or self.model,
                num_speculative_tokens=num_spec_tokens,
                synthetic_acceptance_rate=synthetic_acceptance_rate,
                synthetic_acceptance_length=synthetic_acceptance_length,
            )
        else:
            kwargs.pop("method")
            kwargs.pop("num_speculative_tokens")
            kwargs.pop("draft_model")
            kwargs.pop("spec_decode_acceptance_rate")
            kwargs.pop("spec_decode_acceptance_length")
            kwargs["speculative_config"] = None

        # --enable-tbo [prefill|all] → enable_tbo + enable_tbo_decode
        tbo_mode = kwargs.pop("enable_tbo", None)
        kwargs["enable_tbo"] = tbo_mode is not None
        kwargs["enable_tbo_decode"] = tbo_mode == "all"

        all2all_backend = kwargs.pop("all2all_backend", None)
        kwargs["enable_low_latency"] = all2all_backend == "low-latency"

        # --dspark-config (JSON dict) → DSparkConfig object, passed through as
        # Config.dspark (no env vars).
        kwargs["dspark"] = DSparkConfig.from_dict(kwargs.pop("dspark_config", None))
        # --eplb-config (JSON dict) → EPLBConfig object (--eplb-enable
        # is the master switch, --eplb-config only tunes it).
        kwargs["eplb_config"] = EPLBConfig.from_dict(kwargs.pop("eplb_config"))
        # --dcp-config (JSON dict) → DCPConfig object, passed through as
        # Config.dcp_config.
        kwargs["dcp_config"] = DCPConfig.from_dict(kwargs.pop("dcp_config"))

        # DP topology -> ParallelConfig. `data_parallel_size` stays in kwargs
        # too: LLMEngine still reads the loose kwarg on the legacy path.
        parallel_config_kwargs = {
            "data_parallel_size": kwargs["data_parallel_size"],
            "data_parallel_size_local": kwargs.pop("data_parallel_size_local"),
            "data_parallel_rank": kwargs.pop("data_parallel_rank"),
            "data_parallel_master_ip": kwargs.pop("data_parallel_master_ip"),
            "data_parallel_master_port": kwargs.pop("data_parallel_master_port"),
        }
        base_port = kwargs.pop("data_parallel_base_port")
        if base_port is not None:
            parallel_config_kwargs["data_parallel_base_port"] = base_port
        kwargs["parallel_config"] = ParallelConfig(**parallel_config_kwargs)

        logger.info(f"Engine kwargs: {kwargs}")

        return kwargs

    def create_engine(self, tokenizer=None) -> LLMEngine:
        """Create and return an LLMEngine instance with the configured parameters."""
        return LLMEngine(self.model, tokenizer=tokenizer, **self._get_engine_kwargs())
