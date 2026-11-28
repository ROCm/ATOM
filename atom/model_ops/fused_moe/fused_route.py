# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Router logits to dispatch-space routes in one launch.

After the gate GEMM, a V4.1 MoE layer routes in up to three launches:

  1. aiter `topk_gating` (sqrtsoftplus scores, selection-only bias, top-k,
     renormalize, `routed_scaling_factor`, pad rows routed to -1),
  2. `eplb_map_and_record_fused` on the layers EPLB places (logical ->
     physical, expert-load record),
  3. `remap_topk_to_dispatch` (routed ids widened past the per-rank shared
     slots, this rank's shared column appended), only where the shared
     expert is pinned on every rank (`SharedExpertMode.LOCAL_REPLICA`). With
     no shared expert in the routed call (the 48-routed-experts-a-rank
     layout, the shared expert run on its own) the ids stay physical.

`_fused_route_kernel` runs them all, one program a row, and writes the
`[rows, top_k + num_shared]` weights and ids the experts read. Ids, weights and
load counts are bit for bit those of the three launches
(tests/model_ops/test_fused_route_gpu.py).

`FusedRoute.for_layer` builds one where the layer's routing is the one the
kernel implements; `__call__` returns None for a step it does not take (more
than `MAX_ROWS` rows, the owner's routing not plain, other logits), which then
routes the usual way. The routes are placed: `FusedMoE._route` takes them only
where no gather moves the rows before the experts.
"""

from collections.abc import Callable

import torch

try:
    import triton
    import triton.language as tl

    from atom.model_ops.triton_mm_topk import _sort_v4_experts

    _HAS_TRITON = True
except ImportError:  # pragma: no cover - CPU-only builds
    _HAS_TRITON = False

# Decode widths: one program a row. Past 128 rows the per-row expert-load
# atomics (hot experts queue on them) cost more than the separate launches,
# whose record histograms a block first; a wider step keeps those.
MAX_ROWS = 128
# Widest row (top_k + shared) the kernel takes.
_MAX_WIDTH = 8
# The expert count whose aiter tie order (topk_gating_kernel_opt: six experts
# a lane, sort_network_desc<6>) the kernel reproduces, as triton_mm_topk's
# V4_TIES does.
_EXPERTS = 384


if _HAS_TRITON:

    @triton.jit
    def _fused_route_kernel(
        logits_ptr,  # [rows, N] router logits
        bias_ptr,  # [N] fp32 selection-only bias
        num_valid_ptr,  # [1] the pass's real row count (HAS_NUM_VALID only)
        out_w_ptr,  # [rows, TOP_K + NUM_SHARED] fp32
        out_ids_ptr,  # [rows, TOP_K + NUM_SHARED] int32, dispatch space
        rows,
        routed_scale,
        num_routed_physical,
        routed_per_rank,
        shared_base,
        shared_weight,
        dispatch_ptr,  # [num_logical] (MAP only), see eplb._eplb_route_kernel
        l2p_ptr,  # [num_logical * R] (MAP only)
        cnt_ptr,  # [num_logical] (MAP only)
        load_ptr,  # [num_physical] int32 (RECORD only)
        num_logical,
        id_delta,
        num_physical,
        R,
        dp_size,
        dp_rank,
        N: tl.constexpr,
        NP: tl.constexpr,  # next_pow2(N) (512 for the 384 experts)
        TOP_K: tl.constexpr,
        NUM_SHARED: tl.constexpr,
        OUT_P: tl.constexpr,  # next_pow2(TOP_K + NUM_SHARED)
        HAS_NUM_VALID: tl.constexpr,
        MAP: tl.constexpr,
        RECORD: tl.constexpr,
    ):
        row = tl.program_id(0)
        e = tl.arange(0, NP)
        emask = e < N
        cols = tl.arange(0, OUT_P)
        width = TOP_K + NUM_SHARED
        if HAS_NUM_VALID:
            num_valid = tl.load(num_valid_ptr)
        else:
            num_valid = rows
        pad = row >= num_valid

        # 1. topk_gating, as aiter computes it (compute_score<SQRTSOFTPLUS>):
        # +Inf clamped to FLT_MAX, softplus through exp2/log2, rounded sqrt.
        x = tl.load(logits_ptr + row * N + e, mask=emask, other=0.0).to(tl.float32)
        x = tl.where(x > 3.4028234663852886e38, 3.4028234663852886e38, x)
        sp = tl.where(
            x > 20.0,
            x,
            tl.math.log2(1.0 + tl.math.exp2(x * 1.4426950408889634))
            * 0.6931471805599453,
        )
        score = tl.math.sqrt_rn(sp)
        # NaN (x != x) scores weigh nothing and are never picked.
        orig = tl.where(score != score, 0.0, score)  # noqa: PLR0124
        val = score + tl.load(bias_ptr + e, mask=emask, other=0.0).to(tl.float32)
        val = tl.where(val != val, float("-inf"), val)  # noqa: PLR0124
        val = tl.where(emask, val, float("-inf"))
        orig = tl.where(emask, orig, 0.0)
        # Ties break as aiter's topk_gating_kernel_opt: each lane of the
        # 64-lane wave sorts its six experts with sort_network_desc<6>
        # (strict compare-swaps, so equal values keep no id order), then the
        # wave's ballot takes the first set lane. `val` is in that sorted
        # order (lane e % 64, rank e // 64); `sids` are the experts there.
        val, sids = _sort_v4_experts(val)
        tie = (e % 64) * (NP // 64) + e // 64
        # A picked slot leaves the candidates even at -inf: a row of NaN
        # logits pops slot after slot as aiter does, not one slot six times.
        alive = emask
        ssum = 0.0
        ids = tl.zeros((OUT_P,), dtype=tl.int32) - 1
        ws = tl.zeros((OUT_P,), dtype=tl.float32)
        for k in tl.static_range(TOP_K):
            mx = tl.max(tl.where(alive, val, float("-inf")), axis=0)
            best = tl.min(tl.where(alive & (val == mx), tie, 1 << 21), axis=0)
            pick = tie == best
            idx = tl.sum(tl.where(pick, sids, 0), axis=0)
            w = tl.sum(tl.where(e == idx, orig, 0.0), axis=0)
            ssum += w
            ids = tl.where(cols == k, idx, ids)
            ws = tl.where(cols == k, w, ws)
            alive = alive & ~pick
        ws = ws * tl.math.div_rn(routed_scale, tl.maximum(ssum, 1e-20))

        # 2. EPLB, as eplb._eplb_route_kernel. A pad row routes to -1
        # throughout, which the map keeps and the record skips.
        is_routed = cols < TOP_K
        lid = tl.where(pad | ~is_routed, -1, ids).to(tl.int64)
        phys = lid
        if MAP:
            valid = (lid >= 0) & (lid < num_logical)
            is_tail = lid >= num_logical
            safe_lid = tl.where(valid, lid, 0)
            mapped = tl.load(dispatch_ptr + safe_lid, mask=valid, other=0).to(tl.int64)
            need_spread = valid & (mapped < 0)
            cnt = tl.load(cnt_ptr + safe_lid, mask=valid, other=1).to(tl.int64)
            cnt = tl.maximum(cnt, 1)
            token_idx = row.to(tl.int64) * dp_size + dp_rank
            replica_idx = ((token_idx * 2654435769) & 0xFFFFFFFF) % cnt
            remote = tl.load(
                l2p_ptr + safe_lid * R + replica_idx, mask=need_spread, other=0
            ).to(tl.int64)
            mapped = tl.where(need_spread, remote, mapped)
            phys = tl.where(valid, mapped, tl.where(is_tail, lid + id_delta, lid))
        if RECORD:
            # A plain count: nothing in this launch reads it back, so no
            # ordering (the default acq_rel would fence the stores below).
            rec = is_routed & (phys >= 0) & (phys < num_physical)
            tl.atomic_add(
                load_ptr + tl.where(rec, phys, 0),
                tl.full((OUT_P,), 1, tl.int32),
                mask=rec,
                sem="relaxed",
            )

        # 3. remap_topk_to_dispatch: routed physical p -> p + NUM_SHARED *
        # (p // routed_per_rank); this rank's shared columns after them. No
        # shared expert in the call: the physical ids as they are.
        if NUM_SHARED > 0:
            in_range = (phys >= 0) & (phys < num_routed_physical)
            disp = tl.where(
                in_range, phys + NUM_SHARED * (phys // routed_per_rank), phys
            )
            out_id = tl.where(is_routed, disp, shared_base + (cols - TOP_K))
            out_w = tl.where(is_routed, ws, shared_weight)
        else:
            out_id = phys
            out_w = ws
        out_id = tl.where(pad, -1, out_id)
        out_w = tl.where(pad, 0.0, out_w)
        smask = cols < width
        tl.store(out_ids_ptr + row * width + cols, out_id.to(tl.int32), mask=smask)
        tl.store(out_w_ptr + row * width + cols, out_w, mask=smask)


def _is_gfx9() -> bool:
    from aiter.jit.utils.chip_info import get_gfx

    return get_gfx().startswith("gfx9")


def unsupported_reason(layer) -> str | None:
    """Why `layer`'s routing is not the one `_fused_route_kernel` implements,
    or None. Static: the layer's configuration only."""
    from atom.model_ops.moe import Mxfp4MoEMethod

    layout = getattr(layer, "expert_layout", None)
    checks = (
        ("triton", lambda: _HAS_TRITON),
        # Whose `place_experts` the kernel's steps 2-3 are. Its FlyDSL HERD
        # routing never takes a layer with a fused shared expert.
        (
            "mxfp4 method",
            lambda: isinstance(getattr(layer, "quant_method", None), Mxfp4MoEMethod),
        ),
        # Steps 2-3 are `place_experts` where the shared expert is pinned on
        # every rank (LOCAL_REPLICA, dispatch remap) or not in the routed
        # call at all (no remap).
        (
            "local replica or no shared",
            lambda: layout is not None
            and not getattr(layout, "shared_is_routed", False)
            and (layout.uses_dispatch_remap or layer.num_fused_shared_experts == 0),
        ),
        ("no redundant experts", lambda: layout.num_redundant == 0),
        ("sqrtsoftplus", lambda: layer.scoring_func == "sqrtsoftplus"),
        ("renormalize", lambda: bool(layer.renormalize)),
        ("no grouped top-k", lambda: not layer.use_grouped_topk),
        ("routing bias", lambda: layer.e_score_correction_bias is not None),
        (
            "no balance logits",
            lambda: getattr(layer, "balance_router_logits", None) is None,
        ),
        ("no router weight on input", lambda: not layer.apply_router_weight_on_input),
        (
            "width",
            lambda: layer.top_k + layer.num_fused_shared_experts <= _MAX_WIDTH,
        ),
        ("experts", lambda: layout.num_routed == _EXPERTS),
        # aiter's tie order above is its wave64 kernel's.
        ("gfx9", _is_gfx9),
    )
    for name, ok in checks:
        if not ok():
            return name
    return None


class FusedRoute:
    """`layer`'s routes for a step in one launch. `applies`: whether the
    owner's routing is the plain one this step (e.g. no image rows).
    `routed_scaling_factor`: the one the owner's routing scales by (default
    the layer's)."""

    def __init__(
        self,
        layer,
        applies: Callable[[], bool] | None = None,
        routed_scaling_factor: float | None = None,
    ):
        self.layer = layer
        self.applies = applies
        self.routed_scale = float(
            layer.routed_scaling_factor
            if routed_scaling_factor is None
            else routed_scaling_factor
        )

    @classmethod
    def for_layer(
        cls,
        layer,
        applies: Callable[[], bool] | None = None,
        routed_scaling_factor: float | None = None,
    ) -> "FusedRoute | None":
        if unsupported_reason(layer) is not None:
            return None
        return cls(layer, applies, routed_scaling_factor)

    def runs(self, router_logits: torch.Tensor) -> bool:
        layer = self.layer
        bias = layer.e_score_correction_bias
        return (
            router_logits.dim() == 2
            and 1 <= router_logits.shape[0] <= MAX_ROWS
            and router_logits.shape[1] == layer.expert_layout.num_routed
            and router_logits.dtype in (torch.bfloat16, torch.float32)
            and bias.dtype == torch.float32
            and router_logits.is_cuda
            and (self.applies is None or self.applies())
        )

    def __call__(
        self, router_logits: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """`(weights, ids)` in dispatch space, or None where `runs` refuses."""
        if not self.runs(router_logits):
            return None
        from atom.model_ops.eplb import _replica_spread, eplb_route_tables
        from atom.utils.forward_context import pass_scheduled_tokens_gpu

        layer = self.layer
        layout = layer.expert_layout
        logits = router_logits.contiguous()
        rows, n = logits.shape
        top_k, num_shared = layer.top_k, layer.num_fused_shared_experts
        width = top_k + num_shared
        out_w = torch.empty((rows, width), dtype=torch.float32, device=logits.device)
        out_ids = torch.empty((rows, width), dtype=torch.int32, device=logits.device)
        num_valid = pass_scheduled_tokens_gpu(rows)
        tables = eplb_route_tables(layer, logits.device)
        dp_size, dp_rank = _replica_spread(False)
        _fused_route_kernel[(rows,)](
            logits,
            layer.e_score_correction_bias,
            num_valid if num_valid is not None else logits,
            out_w,
            out_ids,
            rows,
            self.routed_scale,
            int(layout.num_routed_physical),
            int(layout.routed_physical_per_rank) if num_shared else 1,
            int(layer.shared_dispatch_base) if num_shared else 0,
            float(layer.shared_expert_weight) if num_shared else 0.0,
            tables.dispatch if tables else None,
            tables.l2p if tables else None,
            tables.cnt if tables else None,
            tables.load if tables else None,
            tables.num_logical if tables else 0,
            tables.id_delta if tables else 0,
            tables.num_physical if tables else 0,
            tables.num_replicas if tables else 1,
            int(dp_size),
            int(dp_rank),
            N=n,
            NP=triton.next_power_of_2(n),
            TOP_K=top_k,
            NUM_SHARED=num_shared,
            OUT_P=triton.next_power_of_2(width),
            HAS_NUM_VALID=num_valid is not None,
            MAP=tables is not None,
            RECORD=tables is not None and tables.load is not None,
            num_warps=1,
        )
        return out_w, out_ids
