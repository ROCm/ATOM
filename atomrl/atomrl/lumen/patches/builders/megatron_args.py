# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Shared Megatron ARGS-phase CLI registrations."""

from __future__ import annotations

from atomrl.lumen.models.training_contract import (
    add_shared_checkpoint_args,
    add_shared_experiment_args,
)
from atomrl.lumen.models.utils import safe_add_argument
from atomrl.lumen.patches.registry import PatchPhase, register_patch

TE_FORCE_OVERRIDES = {
    "transformer_impl": "local",
    "fp8_param_gather": False,
    "keep_fp8_weight_transpose_cache": False,
    "deprecated_keep_fp8_weight_transpose_cache": False,
    "fp4": None,
    "fp4_param": False,
    "te_rng_tracker": False,
    "inference_rng_tracker": False,
}


def add_common_megatron_args(parser):
    """Register CLI argument groups shared by all Megatron model scripts.

    Registers: ``--backend``, lumen, mxfp8-block-config, lora,
    fp8-training, and warmup/early-stop groups.

    Uses :func:`safe_add_argument` so that model-specific scripts can
    pre-register any of these flags with different defaults **before**
    calling this function.
    """
    safe_add_argument(
        parser,
        "--backend",
        type=str,
        default="megatron",
        choices=["megatron", "fsdp"],
        help="Training backend.",
    )

    lumen = parser.add_argument_group(title="Lumen")
    safe_add_argument(
        lumen,
        "--lumen-attn-backend",
        type=str,
        default="auto",
        choices=["auto", "triton", "csrc", "asm"],
        help="Lumen attention kernel backend. 'auto' prefers csrc with triton fallback. "
        "'asm' uses ASM kernels with fallback chain: asm -> csrc -> triton.",
    )
    safe_add_argument(
        lumen,
        "--lumen-fp8-attn",
        type=str,
        default="none",
        choices=["none", "dpa", "mha"],
        help="FP8 attention scope: 'none' = BF16 attention, "
        "'dpa' = FP8 dot-product attention only, "
        "'mha' = FP8 for full Multi-Head Attention block "
        "(QKV projection + attention + output projection).",
    )
    safe_add_argument(
        lumen,
        "--lumen-fp8-quant-type",
        type=str,
        default="blockwise",
        choices=[
            "dynamic",
            "delayed",
            "blockwise",
            "blockwise2d",
            "per_token",
            "none",
            "mxfp8",
        ],
        help="FP8 quantisation type for FP8 attention backends.",
    )
    safe_add_argument(
        lumen,
        "--lumen-rmsnorm",
        action="store_true",
        default=False,
        help="Replace RMSNorm with Lumen Triton-accelerated RMSNorm.",
    )
    safe_add_argument(
        lumen,
        "--lumen-norm",
        action="store_true",
        default=False,
        help="Replace all norm modules (RMSNorm and LayerNorm) with Lumen implementations.",
    )
    safe_add_argument(
        lumen,
        "--lumen-linear",
        action="store_true",
        default=False,
        help="Use Lumen parallel linear modules (LumenColumnParallelLinear, "
        "LumenRowParallelLinear, LumenLayerNormLinear) via the Lumen spec provider.",
    )
    safe_add_argument(
        lumen,
        "--lumen-cross-entropy",
        action="store_true",
        default=False,
        help="Compute loss using Lumen's Triton parallel cross-entropy kernel.",
    )
    safe_add_argument(
        lumen,
        "--lumen-ce-chunk-rows",
        type=int,
        default=0,
        help=(
            "Row chunk size for chunked cross-entropy (0 = disabled). "
            "Splits B*SQ into chunks of this size so each chunk's allgather "
            "transfers chunk_rows*3 floats instead of B*SQ*3, reducing peak "
            "activation memory. Effective only when --lumen-cross-entropy is set. "
            "Typical value: 2048 (= 2 chunks for MBS=1 seq_len=4096)."
        ),
    )
    safe_add_argument(
        lumen,
        "--lumen-cpu-offload",
        action="store_true",
        default=False,
        help="Offload activations to CPU pinned memory during forward, prefetch in backward.",
    )
    safe_add_argument(
        lumen,
        "--lumen-fp8-checkpoint",
        action="store_true",
        default=False,
        help="Use FP8-aware activation checkpointing that preserves scaling state.",
    )
    safe_add_argument(
        lumen,
        "--lumen-hip-graphs",
        action="store_true",
        default=False,
        help="Graph-capture training steps to reduce kernel launch overhead.",
    )
    safe_add_argument(
        lumen,
        "--lumen-fp8-activation-store",
        action="store_true",
        default=False,
        help="Store MLP activations in FP8 during forward for reduced memory in backward.",
    )
    safe_add_argument(
        lumen,
        "--lumen-cp-comm-type",
        type=str,
        default="a2a",
        choices=["a2a", "p2p"],
        help="Context parallelism communication type: 'a2a' (all-to-all) or 'p2p' (ring).",
    )
    safe_add_argument(
        lumen,
        "--lumen-delay-wgrad",
        action="store_true",
        default=False,
        help="Defer weight gradient computation to overlap with next layer comm.",
    )
    safe_add_argument(
        lumen,
        "--lumen-fp8-param-gather",
        action="store_true",
        default=False,
        help="Store and all-gather parameters in FP8 for reduced communication volume.",
    )
    safe_add_argument(
        lumen,
        "--fp8-param-storage",
        action="store_true",
        default=False,
        help="Store frozen base-model weights in FP8 after checkpoint loading. "
        "Halves model weight memory (~140GB→~70GB for 70B) enabling TP=1 on "
        "192GB GPUs. Weights are dequantized on-the-fly during forward pass.",
    )
    safe_add_argument(
        lumen,
        "--lumen-tp-comm-overlap",
        action="store_true",
        default=False,
        help="Overlap TP communication with GEMM computation. "
        "Mode is set by --lumen-tp-comm-overlap-mode (default: none, which uses "
        "SDMA async overlap when --use-sdma is set). Use 'pipeline' for chunked "
        "NCCL fused pipelining (requires sequence_parallel, BF16/scaling_type=none).",
    )
    safe_add_argument(
        lumen,
        "--lumen-tp-comm-overlap-mode",
        type=str,
        default="none",
        choices=["none", "pipeline"],
        help="TP comm-GEMM overlap mode. 'none': legacy SDMA async overlap (requires "
        "--use-sdma). 'pipeline': chunked NCCL fused pipelining with user-buffer "
        "double-buffering (requires sequence_parallel, BF16).",
    )
    safe_add_argument(
        lumen,
        "--lumen-tp-comm-overlap-chunks",
        type=int,
        default=4,
        help="Number of pipeline chunks for 'pipeline' overlap mode. Sequence length "
        "must be divisible by this value. More chunks = finer overlap granularity "
        "but higher scheduling overhead.",
    )
    safe_add_argument(
        lumen,
        "--lumen-tp-comm-overlap-method",
        type=str,
        default="nccl",
        choices=["nccl"],
        help="Communication backend for 'pipeline' overlap mode. Currently only 'nccl' "
        "is supported.",
    )
    safe_add_argument(
        lumen,
        "--use-sdma",
        action="store_true",
        default=False,
        help="Use mori SDMA instead of torch.distributed for supported collectives "
        "(TP comm, amax all-reduce, CP all-to-all) when available.",
    )
    safe_add_argument(
        lumen,
        "--lumen-fused-rope",
        action="store_true",
        default=False,
        help="Use AITER fused RoPE kernel for rotary positional embeddings.",
    )
    safe_add_argument(
        lumen,
        "--lumen-gradient-accumulation-fusion",
        action="store_true",
        default=False,
        help="Fuse weight gradient accumulation into GEMM backward (accumulate into main_grad).",
    )
    safe_add_argument(
        lumen,
        "--lumen-fused-mlp",
        action="store_true",
        default=False,
        help="Use fused MLP modules (LumenFusedMLP / LumenGatedMLP) for reduced kernel launch overhead.",
    )
    mxfp8 = parser.add_argument_group(title="mxfp8-block-config")
    safe_add_argument(mxfp8, "--mxfp8-block-m-fwd", type=int, default=128)
    safe_add_argument(mxfp8, "--mxfp8-block-n-fwd", type=int, default=128)
    safe_add_argument(mxfp8, "--mxfp8-block-m-dq-bwd", type=int, default=128)
    safe_add_argument(mxfp8, "--mxfp8-block-n-dq-bwd", type=int, default=128)
    safe_add_argument(mxfp8, "--mxfp8-block-m-dkv-bwd", type=int, default=128)
    safe_add_argument(mxfp8, "--mxfp8-block-n-dkv-bwd", type=int, default=128)
    safe_add_argument(mxfp8, "--mxfp8-quant-block-size", type=int, default=128)

    lora = parser.add_argument_group(title="lora")
    safe_add_argument(
        lora, "--lora-rank", type=int, default=0, help="LoRA rank. 0 = disabled."
    )
    safe_add_argument(lora, "--lora-alpha", type=float, default=32.0)
    safe_add_argument(lora, "--lora-dropout", type=float, default=0.1)
    safe_add_argument(
        lora,
        "--lora-target-modules",
        type=str,
        default="all",
        choices=["attention", "attention_mlp", "all"],
        help="LoRA target scope: 'attention' (QKV+proj, NeMo reference), "
        "'attention_mlp' (attention+MLP), 'all' (attention+MLP+emb+output).",
    )
    safe_add_argument(
        lora,
        "--lora-a2a",
        action="store_true",
        default=False,
        help="Enable LoRA all-to-all communication optimisation.",
    )

    lfp8 = parser.add_argument_group(title="linear-fp8")
    # The two gates are mutually exclusive at the parser, not just inside
    # LumenConfig.from_args: the FSDP and RL call sites read args.linear_fp4 off
    # the namespace without going through it, so a config asking for both used
    # to get as far as building a model.
    _linear_quant_gate = parser.add_mutually_exclusive_group()
    safe_add_argument(
        _linear_quant_gate,
        "--linear-fp8",
        action="store_true",
        default=False,
        help="Enable FP8 quantised training for Linear layers. Excludes --linear-fp4.",
    )
    safe_add_argument(
        lfp8,
        "--linear-fp8-format",
        type=str,
        default=None,
        choices=["fp8_e4m3", "fp8_e5m2", "hybrid", "mxfp8"],
        help="FP8 quantisation format for Linear layers. Takes precedence over "
        "Megatron's --fp8-format. MXFP4 is enabled separately with --linear-fp4. "
        "Default None means --fp8-format decides; do not give this a "
        "non-None default or it would silently override --fp8-format.",
    )
    safe_add_argument(
        lfp8,
        "--linear-fp8-scaling",
        type=str,
        default="delayed",
        choices=["dynamic", "delayed", "blockwise", "blockwise2d", "per_token", "none"],
    )
    safe_add_argument(lfp8, "--linear-fp8-block-size", type=int, default=128)
    safe_add_argument(
        lfp8,
        "--linear-fp8-amax-algo",
        type=str,
        default="max",
        choices=["max", "most_recent"],
    )
    safe_add_argument(
        lfp8, "--linear-fp8-reduce-amax", action="store_true", default=False
    )
    safe_add_argument(lfp8, "--linear-fp8-amax-history", type=int, default=16)
    safe_add_argument(
        lfp8,
        "--linear-fp8-margin",
        type=int,
        default=0,
        help="Margin for FP8 scaling factor computation.",
    )
    safe_add_argument(
        lfp8, "--linear-fp8-activation", action="store_true", default=True
    )
    safe_add_argument(
        lfp8,
        "--no-linear-fp8-activation",
        dest="linear_fp8_activation",
        action="store_false",
    )
    safe_add_argument(lfp8, "--linear-fp8-wgrad", action="store_true", default=True)
    safe_add_argument(
        lfp8,
        "--no-linear-fp8-wgrad",
        dest="linear_fp8_wgrad",
        action="store_false",
        help="Execute weight gradient GEMM in higher precision (BF16) even for FP8 runs.",
    )
    safe_add_argument(
        lfp8,
        "--linear-fp8-cache-frozen-weight",
        dest="linear_fp8_cache_frozen_weight",
        action="store_true",
        default=False,
        help="Cache FP8-quantised frozen base weights to avoid re-quantisation on every forward/recompute.",
    )

    safe_add_argument(
        _linear_quant_gate,
        "--linear-fp4",
        action="store_true",
        default=False,
        help=(
            "Enable MXFP4 quantised training for Linear layers (32-element blocks). "
            "Excludes --linear-fp8."
        ),
    )

    grad_quant = parser.add_argument_group(title="gradient-quantization")
    safe_add_argument(
        grad_quant,
        "--grad-quant-type",
        type=str,
        default=None,
        # "fp4" stays accepted so scripts written against it still parse. It has
        # never computed anything -- ScalingManager raises NotImplementedError
        # for it -- so this is not an alias for mxfp4: aliasing would quietly
        # start quantizing gradients to a format the flag does not name. The
        # failure just keeps arriving from the place that can explain it.
        choices=["fp8", "mxfp8", "mxfp4", "fp4"],
        help="Gradient quantization type (None=disabled). Applies to Linear, Attention, and RMSNorm.",
    )

    wes = parser.add_argument_group(title="warmup-early-stop")
    safe_add_argument(wes, "--warmup-steps", type=int, default=0)
    safe_add_argument(wes, "--val-loss-target", type=float, default=None)

    ckpt = parser.add_argument_group(title="checkpoint")
    add_shared_checkpoint_args(ckpt)

    experiment = parser.add_argument_group(title="experiment")
    add_shared_experiment_args(experiment)

    parser.set_defaults(**TE_FORCE_OVERRIDES)

    return parser


register_patch(
    "common_megatron_args",
    PatchPhase.ARGS,
    description="Shared Lumen Megatron CLI groups (backend, lora, fp8, checkpoint, experiment)",
    tags=frozenset({"megatron", "lumen"}),
    default=False,
)(add_common_megatron_args)
