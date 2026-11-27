# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Lumen FP8 hooks for DSV4 native Megatron training."""

from __future__ import annotations

import os
from argparse import Namespace


def dsv4_linear_fp8_enabled() -> bool:
    return os.environ.get("LUMEN_DSV4_LINEAR_FP8", "0") == "1"


def _fp8_scaling_type() -> str:
    return os.environ.get("LUMEN_DSV4_FP8_SCALING", "blockwise")


def _fp8_block_size() -> int:
    return int(os.environ.get("LUMEN_DSV4_FP8_BLOCK_SIZE", "128"))


def _build_fp8_args() -> Namespace:
    """Minimal args namespace for :func:`enable_fp8_for_dsv4_model`."""
    return Namespace(
        linear_fp8=True,
        linear_fp8_scaling=_fp8_scaling_type(),
        linear_fp8_block_size=_fp8_block_size(),
        linear_fp8_amax_algo=os.environ.get("LUMEN_DSV4_FP8_AMAX_ALGO", "max"),
        linear_fp8_reduce_amax=os.environ.get("LUMEN_DSV4_FP8_REDUCE_AMAX", "0") == "1",
        linear_fp8_amax_history=int(
            os.environ.get("LUMEN_DSV4_FP8_AMAX_HISTORY", "16")
        ),
        linear_fp8_margin=int(os.environ.get("LUMEN_DSV4_FP8_MARGIN", "0")),
        linear_fp8_activation=os.environ.get("LUMEN_DSV4_FP8_ACTIVATION", "1") != "0",
        linear_fp8_wgrad=os.environ.get("LUMEN_DSV4_FP8_WGRAD", "1") != "0",
        lumen_fp8_attn=os.environ.get("LUMEN_DSV4_FP8_ATTN", "none"),
        lumen_gradient_accumulation_fusion=False,
        lumen_delay_wgrad=False,
    )


def enable_fp8_for_dsv4_model(model, args: Namespace | None = None) -> None:
    """Enable Lumen FP8 on all DSV4 Lumen linear modules."""
    from megatron.training.utils import print_rank_0

    from atomrl.lumen.config import LumenConfig
    from atomrl.lumen.models.dsv4.megatron.layers import LumenDuplicatedLinear
    from atomrl.lumen.models.megatron import enable_fp8_for_parallel_linear
    from atomrl.lumen.modules.grouped_linear import LumenGroupedLinear
    from atomrl.lumen.modules.parallel_linear import (
        LumenColumnParallelLinear,
        LumenRowParallelLinear,
    )
    from atomrl.lumen.quantize import ScalingManager

    fp8_args = args if args is not None else _build_fp8_args()
    cfg = LumenConfig.from_args(fp8_args)
    scaling_type = getattr(fp8_args, "linear_fp8_scaling", "blockwise")
    block_size = getattr(fp8_args, "linear_fp8_block_size", 128)

    enable_fp8_for_parallel_linear(
        model,
        scaling_type=scaling_type,
        block_size=block_size,
        quant_config=cfg.quant_config,
        fp8_mha=getattr(fp8_args, "lumen_fp8_attn", "none") == "mha",
        gradient_accumulation_fusion=getattr(
            fp8_args, "lumen_gradient_accumulation_fusion", False
        ),
        delay_wgrad=getattr(fp8_args, "lumen_delay_wgrad", False),
    )

    dup_count = 0
    for module in model.modules():
        if isinstance(module, LumenDuplicatedLinear):
            module.enable_fp8(
                scaling_manager=ScalingManager(cfg.quant_config),
                scaling_type=scaling_type,
                block_size=block_size,
            )
            dup_count += 1

    linear_types = (
        LumenColumnParallelLinear,
        LumenRowParallelLinear,
        LumenGroupedLinear,
        LumenDuplicatedLinear,
    )
    total = sum(1 for m in model.modules() if isinstance(m, linear_types))
    print_rank_0(
        f"> Lumen DSV4 FP8 enabled (scaling={scaling_type}, block={block_size}, "
        f"modules={total}, duplicated={dup_count})"
    )


def register_dsv4_megatron_cli(parser) -> None:
    """Register Lumen FP8 CLI flags on the Megatron parser."""
    from atomrl.lumen.patches.builders import apply_args_patches

    apply_args_patches(parser, names={"common_megatron_args"})
