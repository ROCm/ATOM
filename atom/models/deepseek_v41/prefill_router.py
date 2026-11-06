# SPDX-License-Identifier: MIT
"""Owner-local prefill gate using the original one-K-wave native schedule."""

import torch
from aiter.jit.utils.chip_info import get_gfx
from aiter.tuned_gemm import flydsl_gemm, get_GEMM_A16W16_config

KERNEL = (
    "flydsl_hgemm_abf16_wbf16_bf16_t128x96x64x4_ks1_w2x2x1_bias0_ktail0_gm4_pft_gfx950"
)
CONFIG = {"libtype": "flydsl", "solidx": 307, "splitK": 1, "kernelName": KERNEL}


def compatible_config(config):
    """Admit only the two original full-M backends measured for this gate.

    In particular, a local-M tuned lookup can select two K waves with BF16
    partial rounding. Keep all other original schedules on the full-M path.
    """
    return (
        config.get("libtype") == "flydsl"
        and config.get("kernelName") == KERNEL
        and config.get("splitK") == 1
    ) or (config.get("libtype") == "torch" and config.get("solidx") == 0)


def local_router_supported(moe, local, rows, group):
    weight = moe.gate.weight
    if not (
        getattr(moe, "prefill_local_router", False)
        and 2 <= moe.layer_id < 20
        and not moe.is_hash_layer
        and not moe._use_comm_fused_dispatch
        and group.world_size == 4
        and 4096 <= rows <= 16384
        and local.ndim == 2
        and local.shape == ((rows + 3) // 4, 5120)
        and local.dtype == torch.bfloat16
        and local.is_contiguous()
        and local.is_cuda
        and weight.shape == (384, 5120)
        and weight.dtype == local.dtype
        and weight.device == local.device
        and weight.is_contiguous()
        and not getattr(weight, "is_shuffled", False)
        and moe.gate.bias is None
        and get_gfx() == "gfx950"
    ):
        return False
    config = get_GEMM_A16W16_config(
        M=rows,
        N=384,
        K=5120,
        bias=False,
        dtype=str(local.dtype),
        otype=str(local.dtype),
        scaleAB=False,
        bpreshuffle=False,
    )
    return compatible_config(config)


def owned_router_logits(moe, local, rows, group):
    """Graph-owned output; full-M fallback preserves the original gate call."""
    capacity = local.shape[0]
    start = min(group.rank_in_group * capacity, rows)
    valid = min(capacity, rows - start)
    if local_router_supported(moe, local, rows, group):
        logits = flydsl_gemm(
            local, moe.gate.weight, 307, bias=None, otype=local.dtype, config=CONFIG
        )
        if valid != capacity:
            logits[valid:].zero_()
        return logits
    router_input = local.new_zeros((rows, local.shape[1]))
    router_input[start : start + valid].copy_(local[:valid])
    full_logits = moe.router_logits(router_input)
    logits = full_logits.new_zeros((capacity, full_logits.shape[1]))
    logits[:valid].copy_(full_logits[start : start + valid])
    return logits
