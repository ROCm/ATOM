"""ATOM eligibility for aiter's fused top-k + routing + activation quant.

ATOM_MOE_FUSED_TOPK_ROUTE_QUANT: on the gfx1250 TP grouped MoE path at decode
sizes, the routing top-k is not launched in `select_experts`. Its buffers are
handed to aiter (`defer_topk_softmax`), whose grouped MoE then computes the
top-k, the routing metadata and the MXFP4 activation quant in one launch in
place of three (`topkGatingSoftmax`, `moe_route_psum_fused`,
`moe_fused_quant_preshuffle_routeks`), with bit-identical outputs.

Only a path on which nothing reads the routing between `select_experts` and
the experts qualifies: TP experts through aiter's `fused_moe` (no modular
kernel / all-to-all, no Triton experts, no EP mask), no EPLB, no DP gather, no
PCP merge (DP with replicated experts has none), no micro-batching, plain renormalized softmax top-k over 256 routed
experts. aiter runs the plain top-k itself whenever its own path declines.
"""

import torch

from atom.utils import envs

# aiter's fused kernel serves at most this many rows (MAX_ROWS there): the DP
# decode verify step (85-128 sequences x 4 rows) and the drafter; prefill keeps
# the split path.
_MAX_ROWS = 512


def try_defer_topk_route_quant(method, layer, x, logits):
    """`(topk_weights, topk_ids)` left for aiter's grouped MoE to fill, or None
    to route as usual."""
    if not envs.ATOM_MOE_FUSED_TOPK_ROUTE_QUANT:
        return None
    try:
        from aiter.ops.flydsl.grouped_moe_gfx1250 import (
            defer_topk_softmax,
            fused_topk_route_quant_supported,
        )
    except ImportError:
        return None

    from atom.config import get_current_atom_config
    from atom.utils.forward_context import (
        get_forward_context,
        pass_scheduled_tokens_gpu,
    )
    from atom.utils.tbo.ubatching import tbo_active

    context = get_forward_context().context
    config = get_current_atom_config()
    if (
        context is None
        or context.is_dummy_run
        or method.fused_experts is not None
        or method.use_triton
        or method.use_triton_ep
        or getattr(layer, "_comm_fused_moe", None) is not None
        or getattr(layer, "expert_mask", None) is not None
        or getattr(layer, "balance_router_logits", None) is not None
        or tbo_active()
        or layer.moe_parallel_config.pcp_merge
        # DP with every expert on every rank (ATOM_DP_REPLICATE_EXPERTS) runs
        # its own rows through the MoE: no gather, so it qualifies.
        or (layer.dp_size > 1 and not getattr(layer.moe_parallel_config, "replicate_experts", False))
        or config.eplb_enable
        or not layer.renormalize
        or layer.scoring_func != "softmax"
        or layer.use_grouped_topk
        or layer.custom_routing_function is not None
        or layer.e_score_correction_bias is not None
        or layer.shared_expert_scoring_func is not None
        or layer.num_fused_shared_experts
        or layer.expert_layout.num_routed != 256
        or layer.expert_layout.uses_dispatch_remap
        or layer.expert_layout.shared_is_routed
        or not 0 < logits.shape[0] <= _MAX_ROWS
        or not fused_topk_route_quant_supported(logits, x, layer.top_k, True)
    ):
        return None
    n = logits.shape[0]
    weights = torch.empty((n, layer.top_k), dtype=torch.float32, device=x.device)
    ids = torch.empty((n, layer.top_k), dtype=torch.int32, device=x.device)
    defer_topk_softmax(weights, ids, logits, True, pass_scheduled_tokens_gpu(n))
    return weights, ids
