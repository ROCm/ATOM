# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""aiter MegaMoEInternode (two-node EP16 fused MoE) as ATOM's routed-experts step.

``moe_backend="mega"`` on two-node EP16 replaces dispatch + GEMM1 + activation + GEMM2 +
combine of an MXFP4 MoE layer with aiter's ``MegaMoEInternode`` (A8W4: the op
quantizes the bf16 rows to MXFP8 itself, the activations DSv4.1's reference
runs).

Capacities. EP ranks agree on the largest configured token budget at startup.
The prefill capacity rounds that budget up to a power of two, between 32 and
4096; larger steps are chunked. The scheduler budget also bounds captured
token rows. Instances are shared by every shape-identical MoE layer (weights
are bound per call):

* decode: ``mtpr`` in ``_DECODE_MTPRS`` below the prefill capacity, the
  smallest that fits the group-agreed row count;
* prefill: the budgeted capacity; a larger step is split into chunks.

Every rank must issue the same sequence of collective forwards, so the chunk
count comes from the DP group's agreed maximum (``dp_metadata``), never from
this rank's own row count; a rank with fewer rows runs pad chunks.

Pad rows (ATOM routes them to ``-1``) and the rows past a rank's real count
carry ``-1`` in every slot (weight 0). The op treats an out-of-range expert id
as an invalid slot: no dispatch, no GEMM work, and k2 accumulates it as zero,
so a ragged step costs each rank only its real rows.

Weights: w1/w1 scale in the GEMM1 layout (``shuffle_weight`` (16,16) +
``e8m0_shuffle``), w2/w2 scale in the a16w4 GEMM2 layout
(``shuffle_weight_a16w4`` / ``shuffle_scale_a16w4``), built from the raw
checkpoint layout (gate rows then up rows).
"""

from __future__ import annotations

import logging
from contextlib import ExitStack

import torch

logger = logging.getLogger("atom")

# Decode capacities, smallest first. A unified decode step (incl. speculative
# verify: bs * (1 + num_spec)) runs on the smallest capacity that fits.
_DECODE_MTPRS = (32, 64, 128, 256, 512, 1024)
# Prefill capacity cap: each capacity owns its symmetric buffers and
# communicator, so a larger prefill step runs in chunks of this size instead.
_PREFILL_MTPR = 4096

_INSTANCES: dict = {}
# This process's token budget -> the budget the EP group agreed on.
_AGREED_BUDGETS: dict[int, int] = {}


def close_internode_instances() -> None:
    """Close cached operators before ATOM destroys its distributed groups.

    Each capacity owns a CCO communicator. Close in reverse construction
    order, attempting every capacity even when one teardown raises.
    """
    instances = list(_INSTANCES.values())
    _INSTANCES.clear()
    # A rebuilt runner agrees on its budget again, on every rank alike.
    _AGREED_BUDGETS.clear()
    if instances:
        logger.info("[MEGA-INTERNODE] closing %d cached capacities", len(instances))
    with ExitStack() as cleanup:
        for instance in instances:
            cleanup.callback(instance["op"].close)


def build_internode_weights(layer) -> None:
    """Build internode layouts from raw MXFP4 weights, replacing ATOM's shuffle."""
    from aiter.ops.shuffle import (
        shuffle_scale_a16w4,
        shuffle_weight,
        shuffle_weight_a16w4,
    )
    from aiter.utility.fp4_utils import e8m0_shuffle

    w13 = layer.w13_weight.data  # [E, 2*inter, hidden//2] fp4x2, gate rows then up rows
    s13 = layer.w13_weight_scale.data  # [E, 2*inter, hidden//32] e8m0
    w2 = layer.w2_weight.data  # [E, hidden, inter//2]
    s2 = layer.w2_weight_scale.data  # [E, hidden, inter//32]
    E = int(w13.shape[0])
    if E != layer.local_num_experts:
        raise RuntimeError(
            f"internode MegaMoE: local weights {E} != local_num_experts "
            f"{layer.local_num_experts}"
        )
    # Same calls as aiter's
    # op_tests/multigpu_tests/test_megamoe_internode.py::prepare_weights:
    # 3-D packed weights, 2-D (E*rows, K/32) scales.
    layer._mi_w1 = shuffle_weight(w13, layout=(16, 16)).contiguous()
    layer._mi_w1_scale = e8m0_shuffle(
        s13.reshape(E * s13.shape[1], s13.shape[2])
    ).contiguous()
    layer._mi_w2 = shuffle_weight_a16w4(w2, 16, False).contiguous()
    layer._mi_w2_scale = shuffle_scale_a16w4(
        s2.reshape(E * s2.shape[1], s2.shape[2]), E, False
    ).contiguous()


def _rank_world():
    from aiter.dist.parallel_state import get_ep_group

    group = get_ep_group()
    return int(group.rank_in_group), int(group.world_size)


def _get_instance(
    mtpr,
    *,
    model_dim,
    inter_dim,
    experts,
    topk,
    layer,
    activation,
    swiglu_limit,
    situ_beta=1.0,
    situ_linear_beta=1.0,
):
    # Activation parameters are baked into GEMM1 at compile time.
    rank, world = _rank_world()
    key = (
        torch.cuda.current_device(),
        rank,
        world,
        mtpr,
        model_dim,
        inter_dim,
        experts,
        topk,
        activation,
        swiglu_limit,
        situ_beta,
        situ_linear_beta,
    )
    inst = _INSTANCES.get(key)
    if inst is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                f"internode MegaMoE capacity {mtpr} was not built before graph capture"
            )
        from aiter.ops.flydsl.kernels.mega_moe_internode import MegaMoEInternode

        with torch.inference_mode(False), torch.no_grad():
            op = MegaMoEInternode(
                rank=rank,
                world_size=world,
                model_dim=model_dim,
                inter_dim=inter_dim,
                experts=experts,
                topk=topk,
                quant="a8w4",
                w1=layer._mi_w1,
                w1_scale=layer._mi_w1_scale,
                w2=layer._mi_w2,
                w2_scale=layer._mi_w2_scale,
                max_tok_per_rank=mtpr,
                act=activation,
                swiglu_limit=swiglu_limit,
                situ_beta=situ_beta,
                situ_linear_beta=situ_linear_beta,
            )
            inst = {"op": op}
            _INSTANCES[key] = inst
        if rank == 0:
            logger.info(
                "[MEGA-INTERNODE] built capacity=%d ep=%d experts/rank=%d topk=%d",
                mtpr,
                world,
                experts // world,
                topk,
            )
    return inst


def _run_chunk(inst, layer, x, ids, wts, start, n):
    """Run rows [start, start+n) of this rank (n may be 0) through one op call."""
    op = inst["op"]
    # Record quantization packs these inputs directly into operator-owned storage.
    # Invalid expert IDs are skipped by the operator regardless of their weights.
    # Slicing also handles an empty local chunk without forwarding stale buffer rows.
    end = start + n
    op.bind_weights(layer._mi_w1, layer._mi_w1_scale, layer._mi_w2, layer._mi_w2_scale)
    with torch.inference_mode(False), torch.no_grad():
        return op.forward(x[start:end].contiguous(), wts[start:end], ids[start:end])


def _group_rows() -> int:
    """Return the row count agreed across ranks for this collective step."""
    from atom.utils.forward_context import get_forward_context

    fc = get_forward_context()
    ctx = fc.context
    if ctx is not None and ctx.running_tokens_are_unified and ctx.running_tokens > 0:
        return int(ctx.running_tokens)
    if fc.dp_metadata is not None:
        return int(fc.dp_metadata.max_tokens_across_dp)
    raise RuntimeError(
        "internode MegaMoE needs dp_metadata to agree on the chunk count"
    )


def _group_max_tokens(max_tokens: int) -> int:
    """Agree on the capacity budget before any symmetric operator allocations.

    Runs during weight initialization, before warmup/capture. Even nodes with
    different scheduler budgets must construct the same sequence of operators.
    One collective per process: every MoE layer reuses the agreed budget.
    """
    agreed = _AGREED_BUDGETS.get(max_tokens)
    if agreed is not None:
        return agreed
    import torch.distributed as dist
    from aiter.dist.parallel_state import get_ep_group

    group = get_ep_group()
    budgets = [None] * group.world_size
    with torch.inference_mode(False):
        dist.all_gather_object(budgets, max_tokens, group=group.cpu_group)
    if any(n <= 0 for n in budgets):
        raise ValueError(f"internode MegaMoE token budgets must be positive: {budgets}")
    agreed = _AGREED_BUDGETS[max_tokens] = max(budgets)
    return agreed


class MegaInternodeFusedExperts:
    """Whole-pipeline ``fused_experts`` backend (see MegaFusedExperts)."""

    def __init__(self, layer, *, model_dim: int, inter_dim: int, max_tokens: int):
        self._layer = layer
        self._model_dim = model_dim
        self._inter_dim = inter_dim
        max_tokens = _group_max_tokens(max_tokens)
        # The op takes a power-of-two capacity. Keep the minimum decode
        # capacity; the larger ones are built only when a rank's budget can
        # reach them. EP keeps rows local and the runner caps graph token rows
        # at this same scheduler budget.
        rounded_tokens = 1 << (max_tokens - 1).bit_length()
        self._prefill_mtpr = min(_PREFILL_MTPR, max(_DECODE_MTPRS[0], rounded_tokens))
        self._decode_mtprs = tuple(m for m in _DECODE_MTPRS if m < self._prefill_mtpr)

    def __call__(
        self,
        *,
        hidden_states: torch.Tensor,
        w1=None,
        w2=None,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        global_num_experts: int = -1,
        activation=None,
        apply_router_weight_on_input: bool = False,
        expert_map=None,
        x_scale=None,
        moe_extra_args: dict,
        **_ignored,
    ) -> torch.Tensor:
        from aiter import ActivationType

        assert x_scale is None, "internode MegaMoE takes bf16 rows"
        if apply_router_weight_on_input:
            raise NotImplementedError("internode MegaMoE: apply_router_weight_on_input")
        if activation == ActivationType.Situv2:
            act, limit = "situv2", 0.0
            beta = moe_extra_args["beta"]
            linear_beta = moe_extra_args["linear_beta"]
            beta = 1.0 if beta is None else float(beta)
            linear_beta = 1.0 if linear_beta is None else float(linear_beta)
        elif activation in (None, ActivationType.Silu):
            # SwiGLU; a positive swiglu_limit (DeepSeek-V4: 10) clamps it.
            act, limit = "silu", float(moe_extra_args["swiglu_limit"])
            beta = linear_beta = 1.0
        else:
            raise NotImplementedError(
                f"internode MegaMoE: unsupported activation {activation}"
            )
        del w1, w2, expert_map
        layer = self._layer
        x = hidden_states
        rows = int(x.shape[0])
        topk = int(topk_ids.shape[1])
        ids = topk_ids.to(torch.int32).contiguous()
        wts = topk_weights.to(torch.float32).contiguous()
        build = {
            "model_dim": self._model_dim,
            "inter_dim": self._inter_dim,
            "experts": global_num_experts,
            "topk": topk,
            "layer": layer,
            "activation": act,
            "swiglu_limit": limit,
            "situ_beta": beta,
            "situ_linear_beta": linear_beta,
        }
        # Same construction order on every rank (symmetric allocations).
        inst_p = _get_instance(self._prefill_mtpr, **build)
        inst_ds = [_get_instance(m, **build) for m in self._decode_mtprs]

        group_rows = _group_rows()
        if rows > group_rows:
            raise RuntimeError(
                f"internode MegaMoE: local rows {rows} > group rows {group_rows}"
            )
        out = torch.empty(rows, self._model_dim, dtype=torch.bfloat16, device=x.device)
        for inst_d in inst_ds:
            if group_rows <= inst_d["op"].mtpr:
                res = _run_chunk(inst_d, layer, x, ids, wts, 0, rows)
                out.copy_(res[:rows])
                return out
        cap = self._prefill_mtpr
        for c in range((group_rows + cap - 1) // cap):
            start = c * cap
            n = max(0, min(rows - start, cap))
            res = _run_chunk(inst_p, layer, x, ids, wts, start, n)
            if n > 0:
                out[start : start + n].copy_(res[:n])
        return out
