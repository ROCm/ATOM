# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from functools import cache, lru_cache
from typing import Optional

import torch
from atom.utils.custom_register import direct_register_custom_op
from atom.config import get_current_atom_config
from atom.model_ops.fused_moe.config import _has_module
from aiter.jit.utils.torch_guard import torch_compile_guard

@torch_compile_guard()
def is_rocm_aiter_fusion_shared_expert_enabled()-> bool:
    config = get_current_atom_config()
    dp_size = config.parallel_config.data_parallel_size
    if dp_size > 1 and _has_module("mori"):
        return False
    return True

def is_rocm_aiter_fuse_routed_scaling_factor():
    return True

aiter_topK_meta_data = None


@lru_cache(maxsize=1)
def init_aiter_topK_meta_data(
    n_routed_experts: int,
    n_shared_experts: int,
    top_k: int,
    tp_rank: int,
    tp_size: int,
    shared_experts_score: float = 1.0,
    max_num_tokens: int = 32768,
    is_EP: bool = False,
):
    global aiter_topK_meta_data
    fake_expertid = n_routed_experts + n_shared_experts

    # all layers reuse same buffer
    total_topk_ids = torch.empty(
        (max_num_tokens, top_k + n_shared_experts + is_EP),
        dtype=torch.int32,
        device="cuda",
    )
    ns_topk_ids, s_topk_ids = torch.split(total_topk_ids,
        [top_k, n_shared_experts + is_EP], dim=1
    )
    shared_expert_ids = [n_routed_experts + i for i in range(n_shared_experts + is_EP)]
    if is_EP:
        s_topk_ids_list = [
            [fake_expertid] * (n_shared_experts + is_EP)
        ] * max_num_tokens
        for i in range(tp_rank, max_num_tokens, tp_size):
            s_topk_ids_list[i] = shared_expert_ids
    else:
        s_topk_ids_list = [range(n_routed_experts, fake_expertid)] * max_num_tokens
    s_topk_ids[:] = torch.tensor(s_topk_ids_list, dtype=torch.int32, device="cuda")

    total_topk_weights = torch.empty(
        (max_num_tokens, top_k + n_shared_experts + is_EP),
        dtype=torch.float32,
        device="cuda",
    )
    ns_topk_weights, s_topk_weights = torch.split(total_topk_weights,
        [top_k, n_shared_experts + is_EP], dim=1
    )
    s_topk_weights.fill_(shared_experts_score)
    aiter_topK_meta_data = (total_topk_weights, total_topk_ids)


def rocm_aiter_topk_softmax_impl(
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_fused_shared_experts: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    from aiter import topk_softmax

    token = gating_output.shape[0]
    device = gating_output.device
    if is_rocm_aiter_fusion_shared_expert_enabled() and num_fused_shared_experts > 0:
        assert aiter_topK_meta_data is not None, (
            "AITER topK meta data is not initialized. "
            "Please ensure that init_aiter_topK_meta_data is called before this function."
        )
        total_topk_weights, total_topk_ids = aiter_topK_meta_data
        assert total_topk_weights.shape[0] >= token, (
            f"AITER topK meta data support {total_topk_weights.shape[0]} tokens which "
            f"is determined by max_num_batched_tokens, but got {token} tokens now."
        )
        total_topk_weights = total_topk_weights[:token]
        total_topk_ids = total_topk_ids[:token]
        topk_weights, _ = torch.split(total_topk_weights,
            [topk, total_topk_weights.shape[1] - topk], dim=1
        )
        topk_ids, _ = torch.split(total_topk_ids,
            [topk, total_topk_ids.shape[1] - topk], dim=1
        )
    else:
        topk_ids = torch.empty((token, topk), dtype=torch.int32, device=device)
        topk_weights = torch.empty((token, topk), dtype=torch.float32, device=device)
    token_expert_indicies = torch.empty(
        gating_output.shape[0], topk, dtype=torch.int32, device=gating_output.device
    )
    topk_softmax(
        topk_weights, topk_ids, token_expert_indicies, gating_output, renormalize
    )
    if is_rocm_aiter_fusion_shared_expert_enabled() and num_fused_shared_experts > 0:
        return total_topk_weights, total_topk_ids
    return topk_weights, topk_ids


def rocm_aiter_topk_softmax_fake(
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_fused_shared_experts: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    token = gating_output.shape[0]
    device = gating_output.device
    topk_ids = torch.empty((token, topk), dtype=torch.int32, device=device)
    topk_weights = torch.empty((token, topk), dtype=torch.float32, device=device)
    return topk_weights, topk_ids


def rocm_aiter_biased_grouped_topk_impl(
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    num_expert_group: int,
    topk_group: int,
    need_renorm: bool,
    topk: int,
    routed_scaling_factor: float = 1.0,  # mul to topk_weights
    num_fused_shared_experts: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:

    from aiter import biased_grouped_topk

    token = gating_output.shape[0]
    device = gating_output.device
    if is_rocm_aiter_fusion_shared_expert_enabled() and num_fused_shared_experts > 0:
        assert aiter_topK_meta_data is not None, (
            "AITER topK meta data is not initialized. "
            "Please ensure that init_aiter_topK_meta_data is called before this function."
        )
        total_topk_weights, total_topk_ids = aiter_topK_meta_data
        assert total_topk_weights.shape[0] >= token, (
            f"AITER topK meta data support {total_topk_weights.shape[0]} tokens which "
            f"is determined by max_num_batched_tokens, but got {token} tokens now."
        )
        total_topk_weights = total_topk_weights[:token]
        total_topk_ids = total_topk_ids[:token]
        topk_weights, _ = torch.split(total_topk_weights,
            [topk, total_topk_weights.shape[1] - topk], dim=1
        )
        topk_ids, _ = torch.split(total_topk_ids,
            [topk, total_topk_ids.shape[1] - topk], dim=1
        )
    else:
        topk_ids = torch.empty((token, topk), dtype=torch.int32, device=device)
        topk_weights = torch.empty((token, topk), dtype=torch.float32, device=device)
    biased_grouped_topk(
        gating_output,
        correction_bias,
        topk_weights,
        topk_ids,
        num_expert_group,
        topk_group,
        need_renorm,
        routed_scaling_factor,
    )
    if is_rocm_aiter_fusion_shared_expert_enabled() and num_fused_shared_experts > 0:
        return total_topk_weights, total_topk_ids
    return topk_weights, topk_ids


def rocm_aiter_biased_grouped_topk_fake(
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    num_expert_group: int,
    topk_group: int,
    need_renorm: bool,
    topk: int,
    routed_scaling_factor: float = 1.0,  # mul to topk_weights
    num_fused_shared_experts: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    token = gating_output.shape[0]
    device = gating_output.device
    if is_rocm_aiter_fusion_shared_expert_enabled() and num_fused_shared_experts > 0:
        assert aiter_topK_meta_data is not None, (
            "AITER topK meta data is not initialized. "
            "Please ensure that init_aiter_topK_meta_data is called before this function."
        )
        total_topk_weights, total_topk_ids = aiter_topK_meta_data
        total_topk_ids = torch.empty(
            total_topk_ids.shape,
            dtype=torch.int32,
            device=device,
        )
        total_topk_weights = torch.empty(
            total_topk_weights.shape,
            dtype=torch.float32,
            device=device,
        )
        assert total_topk_weights.shape[0] >= token, (
            f"AITER topK meta data support {total_topk_weights.shape[0]} tokens which "
            f"is determined by max_num_batched_tokens, but got {token} tokens now."
        )
        total_topk_weights = total_topk_weights[:token]
        total_topk_ids = total_topk_ids[:token]
        topk_weights, _ = torch.split(total_topk_weights,
            [topk, total_topk_weights.shape[1] - topk], dim=1
        )
        topk_ids, _ = torch.split(total_topk_ids,
            [topk, total_topk_ids.shape[1] - topk], dim=1
        )
    else:
        topk_ids = torch.empty((token, topk), dtype=torch.int32, device=device)
        topk_weights = torch.empty((token, topk), dtype=torch.float32, device=device)
    if is_rocm_aiter_fusion_shared_expert_enabled() and num_fused_shared_experts > 0:
        return total_topk_weights, total_topk_ids
    return topk_weights, topk_ids


def rocm_aiter_grouped_topk_impl(
    gating_output: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    num_expert_group: int,
    topk_group: int,
    need_renorm: bool,
    topk: int,
    scoring_func: str = "softmax",
    routed_scaling_factor: float = 1.0,  # mul to topk_weights
    num_fused_shared_experts: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:

    from aiter import grouped_topk

    token = gating_output.shape[0]
    device = gating_output.device
    if is_rocm_aiter_fusion_shared_expert_enabled() and num_fused_shared_experts > 0:
        assert aiter_topK_meta_data is not None, (
            "AITER topK meta data is not initialized. "
            "Please ensure that init_aiter_topK_meta_data is called before this function."
        )
        total_topk_weights, total_topk_ids = aiter_topK_meta_data
        assert total_topk_weights.shape[0] >= token, (
            f"AITER topK meta data support {total_topk_weights.shape[0]} tokens which "
            f"is determined by max_num_batched_tokens, but got {token} tokens now."
        )
        total_topk_weights = total_topk_weights[:token]
        total_topk_ids = total_topk_ids[:token]
        topk_weights, _ = torch.split(total_topk_weights,
            [topk, total_topk_weights.shape[1] - topk], dim=1
        )
        topk_ids, _ = torch.split(total_topk_ids,
            [topk, total_topk_ids.shape[1] - topk], dim=1
        )
    else:
        topk_ids = torch.empty((token, topk), dtype=torch.int32, device=device)
        topk_weights = torch.empty((token, topk), dtype=torch.float32, device=device)
    grouped_topk(
        gating_output,
        topk_weights,
        topk_ids,
        num_expert_group,
        topk_group,
        need_renorm,
        scoring_func,
        routed_scaling_factor,
    )
    if is_rocm_aiter_fusion_shared_expert_enabled() and num_fused_shared_experts > 0:
        return total_topk_weights, total_topk_ids
    return topk_weights, topk_ids


def rocm_aiter_grouped_topk_fake(
    gating_output: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    num_expert_group: int,
    topk_group: int,
    need_renorm: bool,
    topk: int,
    scoring_func: str = "softmax",
    routed_scaling_factor: float = 1.0,  # mul to topk_weights
    num_fused_shared_experts: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    token = gating_output.shape[0]
    device = gating_output.device
    if is_rocm_aiter_fusion_shared_expert_enabled() and num_fused_shared_experts > 0:
        assert aiter_topK_meta_data is not None, (
            "AITER topK meta data is not initialized. "
            "Please ensure that init_aiter_topK_meta_data is called before this function."
        )
        total_topk_weights, total_topk_ids = aiter_topK_meta_data
        total_topk_ids = torch.empty(
            total_topk_ids.shape,
            dtype=torch.int32,
            device=device,
        )
        total_topk_weights = torch.empty(
            total_topk_weights.shape,
            dtype=torch.float32,
            device=device,
        )
        assert total_topk_weights.shape[0] >= token, (
            f"AITER topK meta data support {total_topk_weights.shape[0]} tokens which "
            f"is determined by max_num_batched_tokens, but got {token} tokens now."
        )
        total_topk_weights = total_topk_weights[:token]
        total_topk_ids = total_topk_ids[:token]
        topk_weights, _ = torch.split(total_topk_weights,
            [topk, total_topk_weights.shape[1] - topk], dim=1
        )
        topk_ids, _ = torch.split(total_topk_ids,
            [topk, total_topk_ids.shape[1] - topk], dim=1
        )
    else:
        topk_ids = torch.empty((token, topk), dtype=torch.int32, device=device)
        topk_weights = torch.empty((token, topk), dtype=torch.float32, device=device)
    if is_rocm_aiter_fusion_shared_expert_enabled() and num_fused_shared_experts > 0:
        return total_topk_weights, total_topk_ids
    return topk_weights, topk_ids


def rocm_aiter_topk_softmax(
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_fused_shared_experts: int = 0,
) -> tuple[torch.Tensor, ...]:
    return rocm_aiter_topk_softmax_impl(
        gating_output, topk, renormalize, num_fused_shared_experts
    )

direct_register_custom_op(
    op_name="rocm_aiter_biased_grouped_topk_impl",
    op_func=rocm_aiter_biased_grouped_topk_impl,
    mutates_args=[],
    fake_impl=rocm_aiter_biased_grouped_topk_fake,
)

direct_register_custom_op(
    op_name="rocm_aiter_grouped_topk_impl",
    op_func=rocm_aiter_grouped_topk_impl,
    mutates_args=[],
    fake_impl=rocm_aiter_grouped_topk_fake,
)



def rocm_aiter_grouped_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: int = 0,
    topk_group: int = 0,
    scoring_func: str = "softmax",
    e_score_correction_bias: Optional[torch.Tensor] = None,
    num_fused_shared_experts: int = 0,
    routed_scaling_factor: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if e_score_correction_bias is not None:
        return torch.ops.aiter.rocm_aiter_biased_grouped_topk_impl(
            gating_output,
            e_score_correction_bias,
            num_expert_group,
            topk_group,
            renormalize,
            topk,
            routed_scaling_factor,
            num_fused_shared_experts,
        )
    else:
        assert scoring_func == "softmax" or scoring_func == "sigmoid"
        return torch.ops.aiter.rocm_aiter_grouped_topk_impl(
            gating_output,
            num_expert_group,
            topk_group,
            renormalize,
            topk,
            scoring_func,
            routed_scaling_factor,
            num_fused_shared_experts,
        )
