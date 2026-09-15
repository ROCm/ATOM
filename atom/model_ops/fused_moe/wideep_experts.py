# SPDX-License-Identifier: Apache-2.0
"""WideEP transport construction for ATOM's modular fused-MoE path."""

from __future__ import annotations

import os
from functools import lru_cache

import torch
from aiter import QuantType, dtypes

from atom.model_ops.fused_moe.mori_prepare_finalize import (
    MoriDispatchFormat,
    MoriPrepareAndFinalize,
)


def _quantize_wideep_dispatch(
    hidden_states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Use the quantizer validated with AITER's inter-node WideEP path."""
    from aiter.ops.flydsl.kernels.mega_moe.quant import per_1x32_mx_quant

    return per_1x32_mx_quant(hidden_states, quant_mode="fp8")


@lru_cache(maxsize=16)
def _init_wideep_op(
    *,
    rank: int,
    world_size: int,
    model_dim: int,
    experts_per_rank: int,
    topk: int,
    mtpr: int,
):
    """Create the dedicated two-node InterNodeV1LL MORI transport."""
    import mori

    capacity_mtpr = 1 << (mtpr - 1).bit_length()
    config = mori.ops.EpDispatchCombineConfig(
        data_type=dtypes.fp8,
        rank=rank,
        world_size=world_size,
        hidden_dim=model_dim,
        scale_dim=model_dim // 32,
        scale_type_size=1,
        max_num_inp_token_per_rank=capacity_mtpr,
        num_experts_per_rank=experts_per_rank,
        num_experts_per_token=topk,
        max_token_type_size=torch.bfloat16.itemsize,
        kernel_type=mori.ops.EpDispatchCombineKernelType.InterNodeV1LL,
        gpu_per_node=8,
        num_qp_per_pe=2,
        rdma_block_num=int(os.environ.get("MORI_EP_RDMA_BLOCK_NUM", "64")),
        block_num=int(os.environ.get("MORI_EP_BLOCK_NUM", "96")),
        warp_num_per_block=int(os.environ.get("MORI_EP_WARP_PER_BLOCK", "8")),
    )
    return mori.ops.EpDispatchCombineOp(config)


def make_wideep_prepare_finalize(
    *,
    model_dim: int,
    experts: int,
    experts_per_rank: int,
    topk: int,
    mtpr: int,
) -> MoriPrepareAndFinalize:
    """Bind WideEP transport to the shared ATOM prepare/finalize implementation."""
    from aiter.dist.parallel_state import get_ep_group

    # Accessing all2all_manager initializes MORI's symmetric heap before this
    # dedicated operator allocates its communication arena.
    manager = get_ep_group().device_communicator.all2all_manager
    rank = int(manager.rank)
    world_size = int(manager.world_size)
    if world_size != 16:
        raise RuntimeError(f"WideEP requires EP16, got EP{world_size}")
    if experts != experts_per_rank * world_size:
        raise RuntimeError(
            "WideEP requires an evenly sharded expert space: "
            f"experts={experts}, experts_per_rank={experts_per_rank}, "
            f"world_size={world_size}"
        )

    # The dispatched activation is already FP8. AITER's mixed MoE selector
    # must therefore stay on its A8W4 path even for decode-sized M.
    bf16_fp8_bound = os.environ.get("AITER_BF16_FP8_MOE_BOUND")
    if bf16_fp8_bound not in (None, "0"):
        raise RuntimeError(
            "WideEP requires AITER_BF16_FP8_MOE_BOUND=0 because dispatch "
            "produces prequantized FP8 activations"
        )
    os.environ["AITER_BF16_FP8_MOE_BOUND"] = "0"

    op = _init_wideep_op(
        rank=rank,
        world_size=world_size,
        model_dim=model_dim,
        experts_per_rank=experts_per_rank,
        topk=topk,
        mtpr=mtpr,
    )
    return MoriPrepareAndFinalize(
        op,
        max_tokens_per_rank=mtpr,
        num_dispatchers=world_size,
        dispatch_format=MoriDispatchFormat(
            dtype=dtypes.fp8,
            quant_type=QuantType.per_1x32,
            scale_dim=model_dim // 32,
            scale_type_size=1,
        ),
        dispatch_quantizer=_quantize_wideep_dispatch,
        fixed_dispatch_config=(
            int(os.environ.get("MORI_EP_BLOCK_NUM", "96")),
            int(os.environ.get("MORI_EP_WARP_PER_BLOCK", "8")),
        ),
    )
