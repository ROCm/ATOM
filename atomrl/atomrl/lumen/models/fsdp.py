# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

###############################################################################
#
# See LICENSE for license information.
###############################################################################

"""Shared FSDP training helpers for Lumen models.

This module consolidates FP8 and LoRA helper functions **and common CLI
argument groups** that are shared by all PyTorch-FSDP-based training scripts
(LLaMA2 SFT, LLaMA 3.1 pretraining, etc.).  Model-specific code (model
building, trainer class, model-specific CLI args) remains in the per-model
subpackages.
"""

import logging
import math
import os

import torch
import torch.distributed as dist
from torch import nn

from atomrl.lumen.models.training_contract import (
    add_fsdp_fp8_contract_args,
    add_fsdp_runtime_contract_args,
    add_shared_checkpoint_args,
    add_shared_experiment_args,
)

logger = logging.getLogger(__name__)


def _rank0_print(msg: str) -> None:
    if not dist.is_initialized() or dist.get_rank() == 0:
        logger.info(msg)


# ---------------------------------------------------------------------------
# Common CLI argument groups
# ---------------------------------------------------------------------------


def add_common_fsdp_args(parser):
    """Register CLI argument groups shared by all FSDP model scripts.

    Covers: backend, training basics, data, FSDP sharding, LoRA,
    linear-fp8 training, warmup / early-stop.

    Model-specific arguments (architecture sizes, checkpoint management,
    etc.) should be registered **after** calling this function.
    """
    parser.add_argument(
        "--backend",
        type=str,
        default="fsdp",
        choices=["megatron", "fsdp"],
        help="Training backend.",
    )

    # -- Training --
    t = parser.add_argument_group("training")
    t.add_argument("--micro-batch-size", type=int, default=1)
    t.add_argument("--gradient-accumulation-steps", type=int, default=8)
    t.add_argument("--max-steps", type=int, default=800)
    t.add_argument("--lr", type=float, default=4e-4)
    t.add_argument("--min-lr", type=float, default=0.0)
    t.add_argument("--weight-decay", type=float, default=0.01)
    t.add_argument("--max-grad-norm", type=float, default=1.0)
    t.add_argument("--log-interval", type=int, default=10)
    t.add_argument(
        "--save-interval",
        type=int,
        default=0,
        help="Save checkpoint every N steps. 0 = disabled.",
    )
    t.add_argument("--save-dir", type=str, default="./checkpoints")
    t.add_argument("--num-workers", type=int, default=4)
    add_fsdp_runtime_contract_args(t)

    # -- Data --
    d = parser.add_argument_group("data")
    d.add_argument("--train-data-path", type=str, default=None)
    d.add_argument("--val-data-path", type=str, default=None)
    d.add_argument("--train-samples", type=int, default=10000)
    d.add_argument("--val-samples", type=int, default=500)

    # -- FSDP --
    f = parser.add_argument_group("fsdp")
    f.add_argument(
        "--sharding-strategy",
        type=str,
        default="full_shard",
        choices=["full_shard", "shard_grad_op", "no_shard"],
    )
    f.add_argument(
        "--fsdp-version",
        type=int,
        default=1,
        choices=[1, 2],
        help="FSDP version: 1 (legacy FullyShardedDataParallel) or 2 (fully_shard API).",
    )
    f.add_argument(
        "--use-sdma",
        action="store_true",
        default=False,
        help="Use mori SDMA instead of torch.distributed for supported collectives "
        "(TP comm, amax all-reduce, CP all-to-all) when available.",
    )

    # -- LoRA --
    lora = parser.add_argument_group("lora")
    lora.add_argument(
        "--lora-rank",
        type=int,
        default=0,
        help="LoRA rank. 0 = disabled (full fine-tuning).",
    )
    lora.add_argument("--lora-alpha", type=float, default=32.0)
    lora.add_argument("--lora-dropout", type=float, default=0.1)

    # -- Linear FP8 training --
    lfp8 = parser.add_argument_group("linear-fp8")
    add_fsdp_fp8_contract_args(lfp8)
    lfp8.add_argument("--linear-fp8-wgrad", action="store_true", default=True)
    lfp8.add_argument(
        "--no-linear-fp8-wgrad",
        dest="linear_fp8_wgrad",
        action="store_false",
        help="Execute weight gradient GEMM in higher precision (BF16) even for FP8 runs.",
    )
    lfp8.add_argument(
        "--grad-quant-type",
        type=str,
        default=None,
        choices=["fp8", "mxfp8", "mxfp4", "fp4"],
        help="Gradient quantization type (None=disabled). "
        "Applies to Linear, Attention, and RMSNorm.",
    )
    lfp8.add_argument(
        "--first-last-layers-bf16",
        action="store_true",
        default=False,
        help="Keep first and last N transformer layers in BF16 during FP8 training.",
    )
    lfp8.add_argument("--num-layers-at-start-in-bf16", type=int, default=1)
    lfp8.add_argument("--num-layers-at-end-in-bf16", type=int, default=1)

    # -- Attention FP8 --
    afp8 = parser.add_argument_group("attention-fp8")
    afp8.add_argument(
        "--lumen-attn-backend",
        type=str,
        default="auto",
        choices=["auto", "triton", "csrc", "asm"],
        help="Lumen attention kernel backend. 'auto' prefers csrc with triton fallback.",
    )
    afp8.add_argument(
        "--lumen-fp8-attn",
        type=str,
        default="none",
        choices=["none", "dpa", "mha"],
        help="FP8 attention scope: 'none' = BF16, 'dpa' = FP8 dot-product, "
        "'mha' = FP8 for full MHA block.",
    )
    afp8.add_argument(
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

    # -- Context parallelism --
    cp = parser.add_argument_group("context-parallelism")
    cp.add_argument(
        "--lumen-cp-comm-type",
        type=str,
        default="a2a",
        choices=["a2a", "p2p"],
        help="Context parallelism comm type: 'a2a' or 'p2p' (ring).",
    )

    # -- Comm overlap --
    overlap = parser.add_argument_group("comm-overlap")
    overlap.add_argument(
        "--lumen-tp-comm-overlap",
        action="store_true",
        default=False,
        help="Overlap TP communication with GEMM computation. "
        "Mode is set by --lumen-tp-comm-overlap-mode (default: none, which uses "
        "SDMA async overlap when --use-sdma is set). Use 'pipeline' for chunked "
        "NCCL fused pipelining (requires sequence_parallel, BF16/scaling_type=none).",
    )
    overlap.add_argument(
        "--lumen-tp-comm-overlap-mode",
        type=str,
        default="none",
        choices=["none", "pipeline"],
        help="TP comm-GEMM overlap mode. 'none': legacy SDMA async overlap (requires "
        "--use-sdma). 'pipeline': chunked NCCL fused pipelining with user-buffer "
        "double-buffering (requires sequence_parallel, BF16).",
    )
    overlap.add_argument(
        "--lumen-tp-comm-overlap-chunks",
        type=int,
        default=4,
        help="Number of pipeline chunks for 'pipeline' overlap mode.",
    )
    overlap.add_argument(
        "--lumen-tp-comm-overlap-method",
        type=str,
        default="nccl",
        choices=["nccl"],
        help="Communication backend for 'pipeline' overlap mode. Only 'nccl' supported.",
    )

    # -- Fused MLP --
    fmlp = parser.add_argument_group("fused-mlp")
    fmlp.add_argument(
        "--lumen-fused-mlp",
        action="store_true",
        default=False,
        help="Use fused MLP modules for reduced kernel launch overhead.",
    )

    # -- FP8 activation store --
    fp8act = parser.add_argument_group("fp8-activation-store")
    fp8act.add_argument(
        "--lumen-fp8-activation-store",
        action="store_true",
        default=False,
        help="Store MLP activations in FP8 during forward.",
    )

    # -- CPU offload --
    cpuoff = parser.add_argument_group("cpu-offload")
    cpuoff.add_argument(
        "--lumen-cpu-offload",
        action="store_true",
        default=False,
        help="Offload activations to CPU during forward, prefetch in backward.",
    )

    # -- Delay wgrad --
    dwg = parser.add_argument_group("delay-wgrad")
    dwg.add_argument(
        "--lumen-delay-wgrad",
        action="store_true",
        default=False,
        help="Defer weight gradient to overlap with next layer comm.",
    )

    # -- Gradient accumulation fusion --
    gaf = parser.add_argument_group("grad-accum-fusion")
    gaf.add_argument(
        "--lumen-gradient-accumulation-fusion",
        action="store_true",
        default=False,
        help="Fuse weight gradient accumulation into GEMM backward.",
    )

    # -- FP8 param all-gather --
    fp8p = parser.add_argument_group("fp8-params")
    fp8p.add_argument(
        "--lumen-fp8-param-gather",
        action="store_true",
        default=False,
        help="Store and all-gather parameters in FP8 for reduced comm volume.",
    )
    fp8p.add_argument(
        "--fsdp-fp8-param-storage",
        action="store_true",
        default=False,
        help="FSDP2 only: store the LoRA-frozen blockwise2d base weight as FP8 and "
        "all-gather it as FP8 (no per-step re-quant). Requires --fsdp-version 2 + "
        "--linear-fp8-scaling blockwise2d.",
    )
    fp8p.add_argument(
        "--fsdp-mxfp4-comm",
        action="store_true",
        default=False,
        help="FSDP2 only: all-gather MXFP4-patched weights as packed FP4 "
        "(~4x fewer bytes on the wire). Requires --fsdp-version 2 + --linear-fp4.",
    )

    # -- Fused RoPE --
    rope = parser.add_argument_group("fused-rope")
    rope.add_argument(
        "--lumen-fused-rope",
        action="store_true",
        default=False,
        help="Use AITER fused RoPE kernel for rotary positional embeddings.",
    )

    # -- HIP graphs --
    hg = parser.add_argument_group("hip-graphs")
    hg.add_argument(
        "--lumen-hip-graphs",
        action="store_true",
        default=False,
        help="Graph-capture training steps to reduce kernel launch overhead.",
    )

    # -- FP8 checkpoint --
    fp8_ckpt = parser.add_argument_group("fp8-checkpoint")
    fp8_ckpt.add_argument(
        "--lumen-fp8-checkpoint",
        action="store_true",
        default=False,
        help="Use FP8-aware activation checkpointing.",
    )

    # -- Norm replacement --
    norm = parser.add_argument_group("norm")
    norm.add_argument(
        "--lumen-norm",
        action="store_true",
        default=False,
        help="Replace all norm modules (RMSNorm and LayerNorm) with Lumen implementations.",
    )

    # -- Warmup + Early stopping --
    wes = parser.add_argument_group("warmup-early-stop")
    wes.add_argument("--warmup-steps", type=int, default=0)
    wes.add_argument("--val-loss-target", type=float, default=None)

    ckpt = parser.add_argument_group("checkpoint")
    add_shared_checkpoint_args(ckpt)

    experiment = parser.add_argument_group("experiment")
    add_shared_experiment_args(experiment)

    return parser


# ---------------------------------------------------------------------------
# Scheduler helpers
# ---------------------------------------------------------------------------


def build_cosine_warmup_scheduler(optimizer, args):
    """Build a cosine scheduler with optional linear warmup."""
    warmup = getattr(args, "lr_warmup_steps", 0)
    max_steps = getattr(args, "max_steps", 0)
    max_lr = getattr(args, "lr", 0.0)
    min_lr = getattr(args, "min_lr", 0.0)

    def _lr_lambda(step):
        if step < warmup:
            return float(step) / max(warmup, 1)
        progress = float(step - warmup) / max(max_steps - warmup, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        min_ratio = min_lr / max_lr if max_lr > 0 else 0.0
        return min_ratio + (1.0 - min_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)


# ---------------------------------------------------------------------------
# Runtime helpers
# ---------------------------------------------------------------------------


def sync_scheduler_to_ckpt_step(scheduler, args) -> None:
    """Advance a scheduler to the checkpoint start step, if any."""
    ckpt_start_step = getattr(args, "ckpt_start_step", 0)
    if ckpt_start_step > 0:
        scheduler.step(ckpt_start_step)


def should_run_eval_step(global_step: int, args) -> bool:
    """Return whether validation should run at *global_step*."""
    eval_interval = getattr(args, "eval_interval", 0)
    eval_every = getattr(args, "eval_every", 0)
    start_eval_at = getattr(args, "start_eval_at", 0)

    interval = eval_interval if eval_interval > 0 else eval_every
    if interval <= 0 or global_step < start_eval_at:
        return False
    return (global_step - start_eval_at) % interval == 0


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def save_fsdp_checkpoint(model, path: str) -> None:
    """Save an unwrapped FSDP/HF model to *path*."""
    os.makedirs(path, exist_ok=True)
    unwrapped = model
    while hasattr(unwrapped, "module"):
        unwrapped = unwrapped.module

    if hasattr(unwrapped, "save_pretrained"):
        unwrapped.save_pretrained(path)
    else:
        torch.save(unwrapped.state_dict(), os.path.join(path, "model.pt"))


# ---------------------------------------------------------------------------
# FP8 quantised training
# ---------------------------------------------------------------------------


def patch_norms(model: nn.Module, args) -> None:
    """Replace norm modules in the model with Lumen implementations.

    Works for both HuggingFace and other FSDP-compatible models.

    .. deprecated:: Prefer :meth:`LumenConfig.enable` which handles norm
       patching automatically.
    """
    from atomrl.lumen.config import LumenConfig

    cfg = LumenConfig.from_args(args)
    if cfg.lumen_norm:
        cfg._patch_norms(model)


def apply_fp8_training(model: nn.Module, args, dp_group=None) -> None:
    """Enable FP8 quantised training via Lumen.

    Args:
        dp_group: Data-parallel process group used for amax reduction.
            If ``None`` and ``args.linear_fp8_reduce_amax`` is true, falls back
            to ``dist.group.WORLD``.

    .. deprecated:: Prefer :meth:`LumenConfig.enable` directly.
    """
    from atomrl.lumen.config import LumenConfig

    cfg = LumenConfig.from_args(args)
    _manager, model = cfg.enable(model, dp_group=dp_group)


def register_quant_optimizer_hooks(model: nn.Module, optimizer, args) -> bool:
    """Wire quantization caches to ``optimizer.step()`` for FSDP training.

    MXFP4 caches each layer's quantized weight so the micro-batches of one
    gradient-accumulation step do not re-quantize an unchanged weight. The cache
    falls back to keying on the parameter's ``_version``, which is enough when
    forward reads the parameter the optimizer updates. Under FSDP it does not:
    forward reads an all-gather buffer, and that buffer's version says nothing
    about the sharded parameter behind it. Without this hook every quantized
    layer keeps training against its step-0 weights, and nothing raises -- the
    loss and grad norm simply stop moving.

    Returns whether a hook was registered (i.e. whether MXFP4 is in use).
    """
    if not getattr(args, "linear_fp4", False):
        return False

    from atomrl.lumen.quantize import register_mxfp4_weight_optimizer_hooks

    unwrapped = model
    while hasattr(unwrapped, "module"):
        unwrapped = unwrapped.module
    register_mxfp4_weight_optimizer_hooks(unwrapped, optimizer)
    _rank0_print("> MXFP4 weight cache invalidation registered on optimizer step")
    return True


def reset_fp8_state(model: nn.Module) -> None:
    """Reset FP8 scaling state after warmup.

    Unwraps nested ``.module`` attributes (e.g. from FSDP or DDP wrappers)
    before calling :meth:`apply`.
    """

    def _reset(m):
        if hasattr(m, "fp8_initialized"):
            m.fp8_initialized = False
        if hasattr(m, "_quant_manager"):
            m._quant_manager.reset()
        if hasattr(m, "_tl_scaling_manager"):
            m._tl_scaling_manager.reset()

    unwrapped = model
    while hasattr(unwrapped, "module"):
        unwrapped = unwrapped.module
    unwrapped.apply(_reset)
    _rank0_print("> FP8 state reset after warmup")


# ---------------------------------------------------------------------------
# LoRA (via HuggingFace PEFT)
# ---------------------------------------------------------------------------


def _wrap_params_as_fp8_comm(model: nn.Module, fp8_dtype: torch.dtype) -> int:
    """Wrap BF16 parameters with FP8CommTensor for FSDP2 FP8 all-gather."""
    from atomrl.lumen.quantize.comm_tensor import FP8CommTensor

    count = 0
    for module in model.modules():
        for name, param in list(module._parameters.items()):
            if (
                param is not None
                and param.dtype == torch.bfloat16
                and param.requires_grad
            ):
                wrapped = torch.nn.Parameter(
                    FP8CommTensor(param.data, fp8_dtype=fp8_dtype),
                    requires_grad=True,
                )
                module._parameters[name] = wrapped
                count += 1
    return count


def _wrap_frozen_base_as_blockwise2d_fp8(
    model: nn.Module, fp8_dtype: torch.dtype, block_size: int = 128, world_size: int = 1
) -> int:
    """Wrap each LoRA-frozen, blockwise2d base ``weight`` in a Blockwise2DFP8Param.

    The wrapper holds the BF16 master (no quantization here — works for a CPU model
    pre-shard); FSDP2 shards it like a BF16 param and its all-gather extension
    quantizes the local shard to FP8 + 2D scale and ships FP8 on the wire, so the
    forward GEMM consumes FP8 with no per-step full-weight re-quant. Must run BEFORE
    ``fully_shard``.

    Eligibility (patched + frozen + 2D), plus ``N % (block * world_size) == 0`` and
    ``K % block == 0`` so each rank's dim-0 shard is itself block-aligned (per-shard
    quant must not split a 128-row tile). Weights that fail this (e.g. lm_head /
    embedding with N = vocab = 32000, not a multiple of block*world_size) stay BF16.
    """
    from atomrl.lumen.quantize.comm_tensor import Blockwise2DFP8Param

    count = 0
    skipped = 0
    for module in model.modules():
        if not getattr(module, "_quant_enabled", False):
            continue
        w = getattr(module, "weight", None)
        if (
            w is None
            or not isinstance(w, nn.Parameter)
            or w.dim() != 2
            or w.requires_grad
        ):
            continue
        if isinstance(w, Blockwise2DFP8Param):
            continue
        if w.shape[0] % (block_size * world_size) or w.shape[1] % block_size:
            skipped += 1
            continue
        module.weight = nn.Parameter(
            Blockwise2DFP8Param(w.data, fp8_dtype, block_size), requires_grad=False
        )
        module._lumen_frozen = True
        count += 1
    if skipped:
        _rank0_print(
            f"> Blockwise2DFP8Param: skipped {skipped} weights not divisible by "
            f"block*world_size ({block_size}*{world_size}) — kept BF16"
        )
    return count


def validate_fsdp_quant_args(args) -> None:
    """Validate FSDP-only low-precision option combinations."""
    linear_fp4 = getattr(args, "linear_fp4", False)
    linear_fp8 = getattr(args, "linear_fp8", False)
    mxfp4_comm = getattr(args, "fsdp_mxfp4_comm", False)
    fp8_param_storage = getattr(args, "fsdp_fp8_param_storage", False)
    fp8_param_gather = getattr(args, "lumen_fp8_param_gather", False)

    from atomrl.lumen.config import check_linear_quant_exclusive

    check_linear_quant_exclusive(linear_fp8, linear_fp4)
    if mxfp4_comm and not linear_fp4:
        raise ValueError("--fsdp-mxfp4-comm requires --linear-fp4")
    if mxfp4_comm and getattr(args, "fsdp_version", 1) != 2:
        raise ValueError("--fsdp-mxfp4-comm requires --fsdp-version 2")
    if linear_fp4 and fp8_param_storage:
        raise ValueError(
            "--fsdp-fp8-param-storage cannot be combined with --linear-fp4"
        )
    if mxfp4_comm and fp8_param_gather:
        raise ValueError(
            "--fsdp-mxfp4-comm cannot be combined with --lumen-fp8-param-gather"
        )


def _wrap_params_as_mxfp4_comm(
    model: nn.Module, block_size: int, world_size: int = 1
) -> int:
    """Wrap each MXFP4-patched weight in an MXFP4CommTensor for FP4 all-gather.

    The wrapper holds the BF16 master weight. FSDP2 shards it like any BF16
    param. During all-gather, the local shard is quantized to packed MXFP4
    (4x less bytes on the wire), then dequantized back to BF16 after gather.
    Optimizer and gradients see normal BF16 — only communication is compressed.

    Alignment: ``N % (block_size × world_size) == 0`` and ``K % block_size == 0``.
    """
    from atomrl.lumen.quantize.comm_tensor import MXFP4CommTensor

    count = 0
    skipped = []
    for module_name, module in model.named_modules():
        if not getattr(module, "_quant_enabled", False):
            continue
        if getattr(module, "_quant_scaling_type", None) != "mxfp4":
            continue
        w = getattr(module, "weight", None)
        if w is None or not isinstance(w, nn.Parameter) or w.dim() != 2:
            continue
        if isinstance(w, MXFP4CommTensor):
            continue
        if w.shape[0] % (block_size * world_size) or w.shape[1] % block_size:
            weight_name = f"{module_name}.weight" if module_name else "weight"
            skipped.append(f"{weight_name}{tuple(w.shape)}")
            continue
        module.weight = nn.Parameter(
            MXFP4CommTensor(w.data, block_size),
            requires_grad=w.requires_grad,
        )
        count += 1
    if skipped:
        examples = ", ".join(skipped[:8])
        if len(skipped) > 8:
            examples += f", ... (+{len(skipped) - 8} more)"
        _rank0_print(
            f"> MXFP4CommTensor: skipped {len(skipped)} weights not divisible by "
            f"({block_size}*world_size={block_size * world_size}, {block_size}); "
            f"kept BF16 communication: {examples}"
        )
    return count


def apply_fsdp2(
    model: nn.Module,
    args,
    dp_group=None,
) -> nn.Module:
    """Apply FSDP2 (``fully_shard``) to *model*.

    Shards each decoder layer individually (equivalent to FSDP1's
    ``transformer_auto_wrap_policy``), then shards the root model.

    Args:
        model: The model to shard.
        args: CLI arguments (needs ``linear_fp8``, ``sharding_strategy``).
        dp_group: Data-parallel process group (used to derive DeviceMesh size).

    Returns:
        The same model (in-place sharding).
    """
    validate_fsdp_quant_args(args)

    try:
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
    except ImportError as e:
        raise ImportError(
            "FSDP2 (fully_shard) requires PyTorch 2.4+. "
            "Install a compatible PyTorch version or use --fsdp-version 1."
        ) from e

    sharding_strategy = getattr(args, "sharding_strategy", "full_shard")
    if sharding_strategy == "no_shard":
        raise ValueError(
            "--sharding-strategy no_shard is not supported with --fsdp-version 2. "
            "Use --fsdp-version 1 or plain DDP for replicated parameters."
        )

    reshard = sharding_strategy != "shard_grad_op"

    world_size = (
        dist.get_world_size(dp_group) if dp_group is not None else dist.get_world_size()
    )
    mesh = init_device_mesh("cuda", (world_size,))

    _use_mxfp4_comm = getattr(args, "fsdp_mxfp4_comm", False)

    if _use_mxfp4_comm or getattr(args, "fsdp_fp8_param_storage", False):
        # param_dtype MUST stay None here: a non-None param_dtype makes FSDP2 cast
        # every param (incl. the frozen FP8 Blockwise2DFP8Param / MXFP4CommTensor)
        # to that dtype before sharding, which collapses the subclass to a plain
        # BF16 DTensor and bypasses its all-gather extension. With param_dtype=None
        # each param keeps its own dtype — FP8/FP4 wrappers stay as themselves,
        # LoRA adapters stay BF16.
        mp_policy = MixedPrecisionPolicy(param_dtype=None, reduce_dtype=torch.float32)
    elif getattr(args, "linear_fp8", False) or getattr(args, "linear_fp4", False):
        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
        )
    else:
        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
        )

    if getattr(args, "lumen_fp8_param_gather", False):
        from atomrl.lumen.quantize.config import _get_float8_e4m3

        fp8_dtype = _get_float8_e4m3()
        n_wrapped = _wrap_params_as_fp8_comm(model, fp8_dtype)
        _rank0_print(f"> FP8CommTensor wrapping: {n_wrapped} params")

    if getattr(args, "fsdp_fp8_param_storage", False):
        from atomrl.lumen.quantize.config import _get_float8_e4m3

        fp8_dtype = _get_float8_e4m3()
        n_stored = _wrap_frozen_base_as_blockwise2d_fp8(
            model, fp8_dtype, world_size=world_size
        )
        _rank0_print(f"> Blockwise2DFP8Param storage: {n_stored} frozen base weights")

    if _use_mxfp4_comm:
        from atomrl.lumen.quantize.config import MXFP4_BLOCK_SIZE

        n_mxfp4 = _wrap_params_as_mxfp4_comm(
            model, block_size=MXFP4_BLOCK_SIZE, world_size=world_size
        )
        if n_mxfp4 == 0:
            # Silently gathering BF16 is the one outcome this flag must not
            # produce: the run pays the FP4 policy (param_dtype=None) and the
            # per-step re-quant, reports nothing, and looks like the compressed
            # configuration in every log a later comparison would read.
            raise ValueError(
                f"--fsdp-mxfp4-comm wrapped 0 weights at world_size={world_size}, so "
                "parameter all-gather stays BF16. Either no MXFP4-patched 2D Linear "
                "weight exists (check that quantization ran before apply_fsdp2), or "
                f"every candidate failed alignment (N % {MXFP4_BLOCK_SIZE * world_size} "
                f"== 0 and K % {MXFP4_BLOCK_SIZE} == 0) — see the skip list above. "
                "Drop the flag to train with BF16 communication deliberately."
            )
        _rank0_print(
            f"> MXFP4CommTensor wrapping: {n_mxfp4} weights (4x comm reduction)"
        )

    sharded_layers = False
    for module in model.modules():
        if hasattr(module, "layers") and isinstance(module.layers, nn.ModuleList):
            for layer in module.layers:
                fully_shard(
                    layer,
                    mesh=mesh,
                    mp_policy=mp_policy,
                    reshard_after_forward=reshard,
                )
            sharded_layers = True
            break

    fully_shard(
        model,
        mesh=mesh,
        mp_policy=mp_policy,
        reshard_after_forward=reshard,
    )

    n_layers = "unknown"
    if sharded_layers:
        for m in model.modules():
            if hasattr(m, "layers") and isinstance(m.layers, nn.ModuleList):
                n_layers = len(m.layers)
                break

    _rank0_print(
        f"> FSDP2 applied (fully_shard, {n_layers} layers, "
        f"reshard={reshard}, mp_policy={mp_policy})"
    )
    return model


_LORA_TARGET_GROUPS = (
    ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
    ("c_attn", "c_proj", "c_fc"),
)


def _lora_target_modules(model: nn.Module) -> list:
    """Pick the LoRA targets that exist on this model.

    Llama/Qwen attention and MLP projections are preferred. GPT-2 names are
    the fallback used by the unit tests and by any GPT-2-shaped checkpoint.
    """
    names = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
    for group in _LORA_TARGET_GROUPS:
        found = [name for name in group if name in names]
        if found:
            return found
    looked_for = [name for group in _LORA_TARGET_GROUPS for name in group]
    raise ValueError(
        f"No LoRA target modules found in the model. Looked for {looked_for}."
    )


def apply_lora(model: nn.Module, args) -> nn.Module:
    """Apply LoRA adapters via HuggingFace PEFT and freeze the base model."""
    from peft import LoraConfig, TaskType, get_peft_model

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=_lora_target_modules(model),
    )
    model = get_peft_model(model, peft_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    _rank0_print(
        f"> LoRA applied (rank={args.lora_rank}, alpha={args.lora_alpha}) — "
        f"trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)"
    )
    return model
