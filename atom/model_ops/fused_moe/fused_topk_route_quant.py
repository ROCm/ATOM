"""ATOM eligibility for aiter's fused top-k + routing + activation quant.

ATOM_MOE_FUSED_TOPK_ROUTE_QUANT: on the gfx1250 TP grouped MoE path the
routing top-k is not launched in `select_experts`. Its buffers are handed to
aiter (`defer_topk_softmax`), whose grouped MoE then computes the top-k, the
routing metadata and the MXFP4 activation quant fused, with bit-identical
outputs: at decode sizes (up to 512 rows) in one launch in place of three
(`topkGatingSoftmax`, `moe_route_psum_fused`, `moe_token_multidest_quant`), at
prefill sizes (`fused_topk_route_quant_prefill`, up to 128 rows a CU) in two
in place of eight (`topkGatingSoftmax`, the valid-row multiply, two fills,
`moe_route`, `moe_contiguous_psum_remap`, `moe_token_multidest_quant_fusepre`,
`moe_scatter_preshuffle_scale`).

Only a path on which nothing reads the routing between `select_experts` and
the experts qualifies: TP experts through aiter's `fused_moe` (no modular
kernel / all-to-all, no Triton experts, no EP mask), no EPLB, no DP gather, no
PCP merge (DP with replicated experts has none), no micro-batching, plain renormalized softmax top-k over 256 routed
experts. aiter runs the plain top-k itself whenever its own path declines.
"""

import torch

from atom.utils import envs

# aiter's decode kernel serves at most this many rows (MAX_ROWS there): the DP
# decode verify step (85-128 sequences x 4 rows) and the drafter. Larger steps
# (prefill) take aiter's prefill kernel where it has one.
_MAX_ROWS = 512

# ATOM_FOLD_TOPK_ROUTER: the routing the router kernel computed for the next MoE
# call, keyed by its router logits: (logits, geometry, routed, num_valid_rows).
_PREPARED: list = []


def set_prepared_routing(
    logits, geometry, routed, num_valid_rows, required=False
) -> None:
    """Hand the next `try_defer_topk_route_quant` on `logits` the top-k, routing
    and activation quant the folded router kernel already computed. `required`
    (ATOM_FOLD_POSTNORM: the MoE input rows are not the experts' input): raise
    where that call would not consume it."""
    _PREPARED.clear()
    _PREPARED.append((logits, geometry, routed, num_valid_rows))
    _REQUIRED[0] = routed if required else None


_REQUIRED: list = [None]


def _take_prepared(logits):
    if not _PREPARED:
        return None
    p = _PREPARED.pop()
    if p[0].data_ptr() != logits.data_ptr() or p[0].shape[0] != logits.shape[0]:
        return None
    return p


def try_defer_topk_route_quant(method, layer, x, logits):
    """`(topk_weights, topk_ids)` left for aiter's grouped MoE to fill, or None
    to route as usual."""
    required, _REQUIRED[0] = _REQUIRED[0], None
    out = _try_defer_topk_route_quant(method, layer, x, logits)
    if required is not None and (out is None or out[0] is not required[0]):
        raise RuntimeError(
            "ATOM_FOLD_POSTNORM: the folded router's prepared routing was not consumed"
        )
    return out


def _try_defer_topk_route_quant(method, layer, x, logits):
    # ATOM_FOLD_TOPK_ROUTER: a routing prepared for these logits is consumed here
    # either way, so it never outlives this call.
    prepared = _take_prepared(logits)
    if not envs.ATOM_MOE_FUSED_TOPK_ROUTE_QUANT:
        return None
    try:
        from aiter.ops.flydsl.grouped_moe_gfx1250 import (
            defer_topk_softmax,
            fused_topk_route_quant_supported,
        )
    except ImportError:
        return None
    try:
        from aiter.ops.flydsl.grouped_moe_gfx1250 import (
            fused_topk_route_quant_prefill_supported,
        )
    except ImportError:  # an aiter without the prefill kernel
        fused_topk_route_quant_prefill_supported = None

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
        # Qwen3.5's prefill token graphs are recorded on a dummy step
        # (GDN `build_for_token_graph_capture`); refusing dummy steps would
        # bake the split path into every prefill graph. Idle DP peers' dummy
        # steps qualify too: with replicated experts each rank routes its own
        # rows. Under DP without replication the dp_size check below refuses.
        or (context.is_dummy_run and not _dummy_ok(layer))
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
        or (
            layer.dp_size > 1
            and not getattr(layer.moe_parallel_config, "replicate_experts", False)
        )
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
        or not _fusable_rows(
            logits,
            x,
            layer.top_k,
            fused_topk_route_quant_supported,
            fused_topk_route_quant_prefill_supported,
        )
    ):
        return None
    n = logits.shape[0]
    nvr = pass_scheduled_tokens_gpu(n)
    if (
        prepared is not None
        and layer.top_k == 8
        and (prepared[3] is None) == (nvr is None)
        and (nvr is None or prepared[3].data_ptr() == nvr.data_ptr())
    ):
        _, geometry, routed, pnvr = prepared
        weights, ids = routed[0], routed[1]
        defer_topk_softmax(
            weights, ids, logits, True, nvr, prepared=(geometry, routed, pnvr)
        )
        return weights, ids
    weights = torch.empty((n, layer.top_k), dtype=torch.float32, device=x.device)
    ids = torch.empty((n, layer.top_k), dtype=torch.int32, device=x.device)
    defer_topk_softmax(weights, ids, logits, True, nvr)
    return weights, ids


def _dummy_ok(layer) -> bool:
    """Dummy steps (graph recording, idle DP peers) may defer: single-rank
    MoE or DP with every expert on every rank."""
    return layer.dp_size == 1 or getattr(
        layer.moe_parallel_config, "replicate_experts", False
    )


def _fusable_rows(logits, x, top_k, decode_supported, prefill_supported) -> bool:
    """Whether one of aiter's fused kernels takes this many rows."""
    rows = logits.shape[0]
    # aiter may hand part of the decode range to its prefill kernel
    # (AITER_TOPK_ROUTE_QUANT_DECODE_MAX_ROWS), so either kernel qualifies.
    if 0 < rows <= _MAX_ROWS and decode_supported(logits, x, top_k, True):
        return True
    return prefill_supported is not None and prefill_supported(logits, x, top_k, True)
