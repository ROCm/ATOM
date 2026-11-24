# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

###############################################################################
#
# See LICENSE for license information.
###############################################################################

"""Lightweight helpers for the shared Lumen training contract.

This module is intentionally argparse-only so it can be imported without
bringing in the heavier Megatron, Transformers, or quantization stacks.
"""

from .utils import safe_add_argument

__all__ = [
    "add_fsdp_contract_args",
    "add_fsdp_fp8_contract_args",
    "add_fsdp_runtime_contract_args",
    "add_shared_checkpoint_args",
    "add_shared_experiment_args",
]


def add_shared_checkpoint_args(parser_or_group):
    """Register checkpoint-management flags shared across training stacks."""
    safe_add_argument(parser_or_group, "--use-ckpt", action="store_true", default=False)
    safe_add_argument(
        parser_or_group, "--save-ckpt", action="store_true", default=False
    )
    safe_add_argument(
        parser_or_group, "--resume-from-hf", action="store_true", default=False
    )
    safe_add_argument(parser_or_group, "--continual-ckpt-path", type=str, default=None)
    safe_add_argument(parser_or_group, "--ckpt-start-step", type=int, default=0)
    safe_add_argument(
        parser_or_group, "--fp8-params", action="store_true", default=False
    )
    safe_add_argument(parser_or_group, "--initial-ckpt-path", type=str, default=None)
    return parser_or_group


def add_shared_experiment_args(
    parser_or_group,
    *,
    default_target_log_ppl: float = 3.3,
    default_step_time_atol: int = 18000,
):
    """Register experiment-management flags shared across training stacks."""
    safe_add_argument(parser_or_group, "--tag", type=str, default="")
    safe_add_argument(
        parser_or_group, "--target-log-ppl", type=float, default=default_target_log_ppl
    )
    safe_add_argument(
        parser_or_group, "--step-time-atol", type=int, default=default_step_time_atol
    )
    safe_add_argument(parser_or_group, "--eval-every", type=int, default=0)
    safe_add_argument(parser_or_group, "--start-eval-at", type=int, default=0)
    return parser_or_group


def add_fsdp_runtime_contract_args(parser_or_group):
    """Register FSDP runtime args shared across FSDP training entrypoints."""
    safe_add_argument(parser_or_group, "--lr-warmup-steps", type=int, default=0)
    safe_add_argument(parser_or_group, "--eval-interval", type=int, default=0)
    safe_add_argument(
        parser_or_group,
        "--gradient-checkpointing",
        action="store_true",
        default=True,
        help="Enable gradient checkpointing / activation recomputation.",
    )
    safe_add_argument(
        parser_or_group,
        "--no-gradient-checkpointing",
        dest="gradient_checkpointing",
        action="store_false",
        help="Disable gradient checkpointing / activation recomputation.",
    )
    return parser_or_group


def add_fsdp_fp8_contract_args(parser_or_group):
    """Register canonical FSDP FP8 args plus legacy launcher aliases."""
    safe_add_argument(
        parser_or_group,
        "--linear-fp8",
        "--fp8-training",
        dest="linear_fp8",
        action="store_true",
        default=False,
        help="Enable FP8 quantised training for Linear layers. Excludes --linear-fp4.",
    )
    safe_add_argument(
        parser_or_group,
        "--linear-fp4",
        dest="linear_fp4",
        action="store_true",
        default=False,
        help="Enable MXFP4 quantised training for Linear layers (32-element blocks). "
        "Excludes --linear-fp8.",
    )
    safe_add_argument(
        parser_or_group,
        "--linear-fp8-format",
        "--fp8-format",
        dest="linear_fp8_format",
        type=str,
        default="fp8_e4m3",
        choices=["fp8_e4m3", "fp8_e5m2", "hybrid", "mxfp8"],
        help="FP8 data format for linear modules.",
    )
    safe_add_argument(
        parser_or_group,
        "--linear-fp8-scaling",
        "--fp8-scaling",
        dest="linear_fp8_scaling",
        type=str,
        default="delayed",
        choices=["dynamic", "delayed", "blockwise", "blockwise2d", "per_token", "none"],
        help="Scaling recipe for FP8 linear modules.",
    )
    safe_add_argument(
        parser_or_group,
        "--linear-fp8-block-size",
        "--fp8-block-size",
        dest="linear_fp8_block_size",
        type=int,
        default=128,
        help="Block size for blockwise / MXFP8 linear quantisation.",
    )
    safe_add_argument(
        parser_or_group,
        "--linear-fp8-amax-algo",
        "--fp8-amax-algo",
        dest="linear_fp8_amax_algo",
        type=str,
        default="max",
        choices=["max", "most_recent"],
        help="Amax update algorithm for FP8 scaling.",
    )
    safe_add_argument(
        parser_or_group,
        "--linear-fp8-reduce-amax",
        "--fp8-reduce-amax",
        dest="linear_fp8_reduce_amax",
        action="store_true",
        default=False,
        help="All-reduce amax statistics across the data-parallel group.",
    )
    safe_add_argument(
        parser_or_group,
        "--linear-fp8-amax-history",
        "--fp8-amax-history",
        dest="linear_fp8_amax_history",
        type=int,
        default=16,
        help="History length for delayed FP8 scaling.",
    )
    safe_add_argument(
        parser_or_group,
        "--linear-fp8-margin",
        "--fp8-margin",
        dest="linear_fp8_margin",
        type=int,
        default=0,
        help="Margin for FP8 scaling factor computation.",
    )
    safe_add_argument(
        parser_or_group,
        "--linear-fp8-activation",
        "--fp8-activation",
        dest="linear_fp8_activation",
        action="store_true",
        default=True,
        help="Run activation GEMMs in FP8 when linear FP8 is enabled.",
    )
    safe_add_argument(
        parser_or_group,
        "--no-linear-fp8-activation",
        "--no-fp8-activation",
        dest="linear_fp8_activation",
        action="store_false",
        help="Keep activation GEMMs in BF16 even when linear FP8 is enabled.",
    )
    return parser_or_group


def add_fsdp_contract_args(parser_or_group):
    """Register the full shared FSDP contract on a parser or group."""
    add_fsdp_runtime_contract_args(parser_or_group)
    add_fsdp_fp8_contract_args(parser_or_group)
    add_shared_checkpoint_args(parser_or_group)
    add_shared_experiment_args(parser_or_group)
    return parser_or_group
