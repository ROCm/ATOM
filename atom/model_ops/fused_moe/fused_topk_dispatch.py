"""ATOM eligibility and context for AITER fused routing."""

import os

import torch


def try_fused_topk(method, layer, logits):
    if os.getenv("ATOM_FUSED_TOPK_DISPATCH", "0") != "1":
        return None
    from aiter.ops.triton.moe.fused_topk_dispatch import fused_topk_dispatch

    from atom.config import get_current_atom_config
    from atom.model_ops.fused_moe.mori_v2_prepare_finalize import MoriV2ModularKernel
    from atom.utils.forward_context import (
        get_forward_context,
        pass_scheduled_tokens_gpu,
    )
    from atom.utils.tbo.ubatching import tbo_active

    backend = getattr(method, "fused_experts", None)
    if not isinstance(backend, MoriV2ModularKernel) or method.use_triton_ep:
        return None
    mega = backend.prepare_finalize.mega
    context = get_forward_context().context
    if (
        mega is None
        or mega._config.dispatch_backend != "mori"
        or context is None
        or tbo_active()
        or layer.moe_parallel_config.pcp_merge
        or layer.top_k != 8
        or not layer.renormalize
        or layer.scoring_func != "softmax"
        or layer.use_grouped_topk
        or layer.custom_routing_function is not None
        or layer.e_score_correction_bias is not None
        or layer.shared_expert_scoring_func is not None
        or layer.expert_layout.num_routed != 256
        or layer.num_fused_shared_experts
        or layer.expert_layout.uses_dispatch_remap
        or layer.expert_layout.shared_is_routed
        or get_current_atom_config().eplb_enable
        or logits.dtype != torch.bfloat16
        or logits.stride(1) != 1
        or logits.shape[0] == 0
        or logits.shape[1] != 256
        or logits.shape[0] != context.running_tokens
    ):
        return None
    n = logits.shape[0]
    weights = torch.empty((n, 8), dtype=torch.float32, device=logits.device)
    ids = torch.empty((n, 8), dtype=torch.int32, device=logits.device)
    pad = torch.empty(n, dtype=torch.bool, device=logits.device)
    valid_rows = pass_scheduled_tokens_gpu(n)
    fused_topk_dispatch(
        logits,
        weights,
        ids,
        None,
        None,
        pad,
        mega._total_recv,
        batch_size=1,
        is_draft=False,
        write_requests=False,
        no_padding=valid_rows is None,
        num_valid_rows=valid_rows,
    )
    backend._fused_topk_result = (ids, pad)
    return weights, ids
