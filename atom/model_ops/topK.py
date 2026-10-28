# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from functools import lru_cache

import torch
from aiter.jit.utils.torch_guard import torch_compile_guard

from atom.config import get_current_atom_config
from atom.model_ops.fused_moe.expert_layout import SharedExpertMode
from atom.model_ops.utils import _has_module
from atom.quant_spec import fuses_mxfp8_shared_into_mxfp4
from atom.utils import envs
from atom.utils.custom_register import direct_register_custom_op
from atom.utils.forward_context import get_forward_context


def is_rocm_aiter_fusion_shared_expert_enabled_for_quant_config(
    quant_config,
    shared_expert_prefix: str | None = None,
    routed_expert_prefix: str | None = None,
) -> bool:
    config = get_current_atom_config()
    if quant_config is None:
        quant_config = config.quant_config

    dp_size = config.parallel_config.data_parallel_size
    # Shared-expert fusion is incompatible with the flattened DP x TP MoE-EP
    # layout (set by the vLLM plugin under DP+EP); disable it there.
    if dp_size > 1 and config.moe_ep_flatten_tp_across_dp:
        return False
    # Only the selected MoRI transport needs this switch; the TP fusion is a
    # different mechanism, and an explicitly selected RCCL/none backend must
    # not be inferred from the mere presence of the ``mori`` Python package.
    # EPLB always fuses, otherwise the env decides.
    requested_all2all = getattr(config, "moe_all2all_backend", "auto")
    mori_selected = (
        requested_all2all in {"auto", "mori"}
        and not envs.ATOM_DISABLE_MORI_EP
        and _has_module("mori")
    )
    if (
        dp_size > 1
        and mori_selected
        and config.enable_dp_attention
        and not (getattr(config, "eplb_enable", False) or envs.ATOM_FUSE_SHARED_EXPERT)
    ):
        return False

    if quant_config is not None and shared_expert_prefix is not None:
        shared_spec = quant_config.get_layer_quant_config(
            shared_expert_prefix,
            check_children=True,
        )
        routed_spec = (
            quant_config.get_layer_quant_config(
                routed_expert_prefix,
                check_children=True,
            )
            if routed_expert_prefix is not None
            else quant_config.global_quant_config
        )
        return (
            shared_spec.quant_dtype == routed_spec.quant_dtype
            and shared_spec.quant_type == routed_spec.quant_type
            and shared_spec.is_dynamic == routed_spec.is_dynamic
        ) or fuses_mxfp8_shared_into_mxfp4(shared_spec, routed_spec)

    # Resolve actual dtypes for shared experts vs routed experts.
    # Find a representative shared expert entry from the exclude list to
    # get its resolved dtype, then compare against the global quant spec
    # (which represents the dtype of non-excluded routed experts).
    # We cannot use a fixed probe prefix because the exclude list format
    # changes after remap_layer_name (native vs plugin mode, packed_modules
    # remap). Instead, pick the first exclude entry that mentions
    # "shared_expert" and resolve its dtype via get_layer_quant_config.
    exclude_layers = quant_config.exclude_layers or []
    for entry in exclude_layers:
        if "shared_experts" in entry or "shared_expert" in entry:
            shared_spec = quant_config.get_layer_quant_config(entry)
            # Compare shared expert dtype against the global (routed) dtype.
            # If they differ, shared experts cannot be fused into the MoE
            # kernel (e.g. shared experts are BF16 while routed are FP4).
            if shared_spec.quant_dtype != quant_config.quant_dtype:
                return False
            break

    return True


@torch_compile_guard()
def is_rocm_aiter_fusion_shared_expert_enabled(
    shared_expert_prefix: str | None = None,
    routed_expert_prefix: str | None = None,
) -> bool:
    config = get_current_atom_config()
    return is_rocm_aiter_fusion_shared_expert_enabled_for_quant_config(
        config.quant_config,
        shared_expert_prefix=shared_expert_prefix,
        routed_expert_prefix=routed_expert_prefix,
    )


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
    _, s_topk_ids = torch.split(
        total_topk_ids, [top_k, n_shared_experts + is_EP], dim=1
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
    _, s_topk_weights = torch.split(
        total_topk_weights, [top_k, n_shared_experts + is_EP], dim=1
    )
    s_topk_weights.fill_(shared_experts_score)
    aiter_topK_meta_data = (total_topk_weights, total_topk_ids)


def topk_meta_rows(num_rows: int) -> tuple[torch.Tensor, torch.Tensor]:
    """This pass's `num_rows` rows of the shared `aiter_topK_meta_data` buffer
    (routed columns first, the prefilled shared ones after). A TBO micro-batch
    takes the rows at its own offset in the whole forward, so two micro-batches
    in flight never write the same rows."""
    assert aiter_topK_meta_data is not None, (
        "AITER topK meta data is not initialized. "
        "Please ensure that init_aiter_topK_meta_data is called before this function."
    )
    context = get_forward_context().context
    start = context.ubatch_token_offset if context is not None else 0
    total_topk_weights, total_topk_ids = aiter_topK_meta_data
    assert total_topk_weights.shape[0] >= start + num_rows, (
        f"AITER topK meta data support {total_topk_weights.shape[0]} tokens which "
        f"is determined by max_num_batched_tokens, but got rows "
        f"[{start}, {start + num_rows}) now."
    )
    rows = slice(start, start + num_rows)
    return total_topk_weights[rows], total_topk_ids[rows]


def reassign_gathered_shared_ids(layer, topk_ids: torch.Tensor) -> None:
    """On routed rows a gather moved between ranks, where the layer's routing
    widened in the prefilled shared columns (LEGACY_AITER): give them this
    rank's prefilled ids, in place. The prefill says by row position which
    rows' shared expert this rank runs (under EP every ep_size-th row, from its
    own rank), so the ids must follow a row's position in the gathered batch
    -- as when routing ran after the gather -- not the one it had on the rank
    that routed it. The weights stay: a scored shared column's are the row's
    own. The shared columns are never written after the prefill, so any
    micro-batch may read rows from 0."""
    if layer.expert_layout.mode is not SharedExpertMode.LEGACY_AITER:
        return
    assert aiter_topK_meta_data is not None, (
        "AITER topK meta data is not initialized. "
        "Please ensure that init_aiter_topK_meta_data is called before this function."
    )
    _, total_topk_ids = aiter_topK_meta_data
    rows = topk_ids.shape[0]
    assert total_topk_ids.shape[0] >= rows, (
        f"AITER topK meta data support {total_topk_ids.shape[0]} tokens, "
        f"but {rows} rows were gathered."
    )
    top_k = layer.top_k
    topk_ids[:, top_k:] = total_topk_ids[:rows, top_k:]


def rocm_aiter_topk_softmax_impl(
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_fused_shared_experts: int = 0,
    num_routing_experts: int = 0,
    fused_shared_experts_scoring_func: str | None = None,
    num_valid_rows: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    from aiter import topk_softmax

    token = gating_output.shape[0]
    device = gating_output.device
    if num_fused_shared_experts > 0:
        total_topk_weights, total_topk_ids = topk_meta_rows(token)
        # The kernel scores the shared columns too when given a scoring func.
        topk_weights = total_topk_weights
        topk_ids = total_topk_ids[:, :topk]
    else:
        topk_ids = torch.empty((token, topk), dtype=torch.int32, device=device)
        topk_weights = torch.empty((token, topk), dtype=torch.float32, device=device)
    token_expert_indicies = torch.empty(
        gating_output.shape[0], topk, dtype=torch.int32, device=gating_output.device
    )
    if fused_shared_experts_scoring_func is None:
        fused_shared_experts_scoring_func = ""
        fused_shared_experts_for_kernel = 0
    else:
        fused_shared_experts_for_kernel = num_fused_shared_experts
    topk_softmax(
        topk_weights,
        topk_ids,
        token_expert_indicies,
        gating_output,
        renormalize,
        fused_shared_experts_for_kernel,
        fused_shared_experts_scoring_func,
        num_valid_rows,
    )
    if num_fused_shared_experts > 0:
        return total_topk_weights, total_topk_ids
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
    num_valid_rows: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:

    from aiter import biased_grouped_topk

    token = gating_output.shape[0]
    device = gating_output.device
    if num_fused_shared_experts > 0:
        total_topk_weights, total_topk_ids = topk_meta_rows(token)
        topk_weights = total_topk_weights[:, :topk]
        topk_ids = total_topk_ids[:, :topk]
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
        num_valid_rows,
    )
    if num_fused_shared_experts > 0:
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
    num_valid_rows: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    token = gating_output.shape[0]
    device = gating_output.device
    if num_fused_shared_experts > 0:
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
        topk_weights, _ = torch.split(
            total_topk_weights, [topk, total_topk_weights.shape[1] - topk], dim=1
        )
        topk_ids, _ = torch.split(
            total_topk_ids, [topk, total_topk_ids.shape[1] - topk], dim=1
        )
    else:
        topk_ids = torch.empty((token, topk), dtype=torch.int32, device=device)
        topk_weights = torch.empty((token, topk), dtype=torch.float32, device=device)
    if num_fused_shared_experts > 0:
        return total_topk_weights, total_topk_ids
    return topk_weights, topk_ids


def rocm_aiter_grouped_topk_impl(
    gating_output: torch.Tensor,
    num_expert_group: int,
    topk_group: int,
    need_renorm: bool,
    topk: int,
    scoring_func: str = "softmax",
    routed_scaling_factor: float = 1.0,  # mul to topk_weights
    num_fused_shared_experts: int = 0,
    num_valid_rows: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:

    from aiter import grouped_topk

    token = gating_output.shape[0]
    device = gating_output.device
    if num_fused_shared_experts > 0:
        total_topk_weights, total_topk_ids = topk_meta_rows(token)
        topk_weights = total_topk_weights[:, :topk]
        topk_ids = total_topk_ids[:, :topk]
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
        scoring_func == "softmax",
        routed_scaling_factor,
        num_valid_rows,
    )
    if num_fused_shared_experts > 0:
        return total_topk_weights, total_topk_ids
    return topk_weights, topk_ids


def rocm_aiter_grouped_topk_fake(
    gating_output: torch.Tensor,
    num_expert_group: int,
    topk_group: int,
    need_renorm: bool,
    topk: int,
    scoring_func: str = "softmax",
    routed_scaling_factor: float = 1.0,  # mul to topk_weights
    num_fused_shared_experts: int = 0,
    num_valid_rows: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    token = gating_output.shape[0]
    device = gating_output.device
    if num_fused_shared_experts > 0:
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
        topk_weights, _ = torch.split(
            total_topk_weights, [topk, total_topk_weights.shape[1] - topk], dim=1
        )
        topk_ids, _ = torch.split(
            total_topk_ids, [topk, total_topk_ids.shape[1] - topk], dim=1
        )
    else:
        topk_ids = torch.empty((token, topk), dtype=torch.int32, device=device)
        topk_weights = torch.empty((token, topk), dtype=torch.float32, device=device)
    if num_fused_shared_experts > 0:
        return total_topk_weights, total_topk_ids
    return topk_weights, topk_ids


def rocm_aiter_topk_softmax(
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_fused_shared_experts: int = 0,
    num_routing_experts: int = 0,
    fused_shared_experts_scoring_func: str | None = None,
    num_valid_rows: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    return rocm_aiter_topk_softmax_impl(
        gating_output,
        topk,
        renormalize,
        num_fused_shared_experts,
        num_routing_experts,
        fused_shared_experts_scoring_func,
        num_valid_rows,
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
    e_score_correction_bias: torch.Tensor | None = None,
    num_fused_shared_experts: int = 0,
    routed_scaling_factor: float = 1.0,
    num_valid_rows: torch.Tensor | None = None,
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
            num_valid_rows,
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
            num_valid_rows,
        )


def mm_topk(
    ids: torch.Tensor | None,
    gating_output: torch.Tensor,
    bias: torch.Tensor,
    bias_alt: torch.Tensor,
    hash_table: torch.Tensor | None,
    vocab_size: int,
    renormalize: bool,
    scaling: float,
    out_ids: torch.Tensor,
    out_weights: torch.Tensor,
    *,
    image_mask: torch.Tensor | None = None,
    num_valid_rows: torch.Tensor | None = None,
) -> None:
    """Fused two-bias routing from #2149, with an optional explicit image mask.

    V4 uses above-vocabulary IDs and may supply a text hash table. An explicit
    mask replaces sentinel detection; without hashing, IDs/vocabulary are unused.
    Output views may have a wider row stride for fused shared-expert columns.
    Rows at or past `num_valid_rows` route to -1.
    """
    from atom.model_ops.triton_mm_topk import mm_topk_triton

    mm_topk_triton(
        ids,
        gating_output,
        bias,
        bias_alt,
        hash_table,
        vocab_size,
        renormalize,
        scaling,
        out_ids,
        out_weights,
        image_mask=image_mask,
        num_valid_rows=num_valid_rows,
    )
