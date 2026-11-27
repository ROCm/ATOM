# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Host wrapper of the routed MoE GEMM2 + tail + k5 one-shot all-reduce + residual + RMSNorm + MXFP8 launch
(``kernels/moe_tail_ar_gfx1250.py``): what production runs as the grouped GEMM2 launch followed by
``CustomAllreduce.k5_fused_ar_rms_tail(..., gemma_norm=True, fp8_out=True)``, with the same bits.

Call (every rank of the TP group, same order and shapes: it is a collective):
    g2 = Gemm2Call(...)                       # the operands production passes to its GEMM2 launch
    out, res_out, scale = flydsl_moe_tail_ar(g2, topk_ids, topids_to_rows, gather_w, addend, addend_scale,
                                             residual, norm.weight, eps, inbox, ctr, slot, rank)
``inbox`` / ``ctr`` / ``slot`` / ``rank``: production's k5 context (``ca_comm._k5_ctx`` words: the four ranks'
inbox addresses as this process maps them, the counter address, ``max_rows * 768``, the rank).
"""

from __future__ import annotations

import dataclasses

import torch

from . import moe_tail_ar_layout as LY
from aiter.ops.flydsl.kernels.tensor_shim import _preload_compiled, _run_compiled, current_stream_arg, ptr_arg

# One switch read at call time (the served-step A/B flips it per graph config); the gate is shape-only, so every
# rank decides the same.
ENABLED = True
ROWS = 8
MAXIT = 20_000_000  # k5's own bound
# GEMM2 tile inside the launch: 16 x 512 x 256 on 8 waves (the receiver's 256 threads), 2 ring buffers
TILE_N, N_WARP, TILE_K, NUM_BUFFERS, WAVES_PER_TENSOR_TDM = 512, 8, 256, 2, 2
PROBE = 16
# diagnostic: a list -> every fused launch made through fused_tail gets a per-workgroup timeline buffer appended
TIMELINES = None
# The launch starts with GEMM2, so everything the tail needs must be on the stream before GEMM2. A tail whose
# addend still needs a join (`before`: the shared expert on a side stream) keeps production's two launches: GEMM2
# behind that join measured +1.7 us a layer for production's own kernels (57 layers, TP4). True = diagnostic only.
JOIN_FIRST_OK = False
_SYNCED: set = set()
_STATE = {"c_stores": 0}
# Shared down inside the launch: the glue-epilogue addend's address -> (h E4M3 [8, 768], hs E8M0 [8, 24], down weight
# E4M3 [6144, 768] (16, 16)-preshuffled as uint8, its E8M0 scales uint8). A tail whose addend is registered runs the
# shared down in its own workgroups (h from an earlier launch) and does not read the addend buffer. The caller's
# epilogue `before` of a registered addend must be production's shared down into that addend: every path that reads
# the addend runs `before` first, and the shared-down launch drops it unrun (it computes the same bytes itself).
SHARED_DOWN: dict = {}
_SD: dict = {}  # (device, h, hs, dw, ds addresses) -> (table int64 [5], shared-output buffer bf16 [8, 6144], armed 0xffff)


def _sd_key(h, hs, dw, ds):
    return (h.device.index, h.data_ptr(), hs.data_ptr(), dw.data_ptr(), ds.data_ptr())


def shared_down_table(h, hs, dw, ds):
    """The shared-down table of one layer and its shared-output buffer (armed once, outside any capture)."""
    key = _sd_key(h, hs, dw, ds)
    t = _SD.get(key)
    if t is None:
        assert h.dtype == torch.uint8 and tuple(h.shape) == (ROWS, 768) and h.is_contiguous()
        assert hs.dtype == torch.uint8 and tuple(hs.shape) == (ROWS, 24) and hs.is_contiguous()
        assert dw.numel() == LY.HIDDEN * 768 and ds.numel() == LY.HIDDEN * 24
        sf = torch.empty((ROWS, LY.HIDDEN), dtype=torch.bfloat16, device=h.device)
        sf.view(torch.int16).fill_(-1)
        tab = torch.tensor([h.data_ptr(), hs.data_ptr(), dw.data_ptr(), ds.data_ptr(), sf.data_ptr()],
                           dtype=torch.int64, device=h.device)
        t = _SD[key] = (tab, sf)
    return t


@dataclasses.dataclass
class Gemm2Call:
    """The routed GEMM2's operands, as ``flydsl_grouped_gemm_a8w4_masked`` takes them."""

    out: torch.Tensor  # bf16 [1, contiguous_m, N] grouped output (not written: the launch has its own)
    a: torch.Tensor  # MXFP4 activations (GEMM1's output)
    w: torch.Tensor  # uint8 view of the preshuffled MXFP4 weights
    a_scales: torch.Tensor
    w_scales: torch.Tensor  # int32 view
    m_tile_map: torch.Tensor  # psum
    n_experts: int
    contiguous_m: int
    N: int
    K: int
    a_is_fp4: int = 1
    a_preshuffled: int = 0
    tdm_b_th: int = 0
    lds_soa_load_interleave: int = 0
    rolled_drain: int = 0


def supported(g2: Gemm2Call, tokens: int, has_addend: bool) -> str | None:
    """None when the launch can take this call, else why not (shapes and dtypes only)."""
    if tokens != ROWS:
        return f"{tokens} rows"
    if g2.N != LY.HIDDEN or g2.K % 128:
        return f"N {g2.N} K {g2.K}"
    if g2.contiguous_m != LY.MAX_M_TILES * LY.TILE_M or g2.out.numel() != g2.contiguous_m * g2.N:
        return f"grouped rows {g2.contiguous_m}"
    if g2.out.dtype != torch.bfloat16:
        return "grouped output dtype"
    if not 0 < g2.n_experts <= 4096:
        return f"{g2.n_experts} experts"
    return None


_G: dict = {}


def _scratch_key(device, rows: int):
    return (device.index if device.index is not None else torch.cuda.current_device(), rows)


def grouped_scratch(device, rows: int = LY.MAX_M_TILES * LY.TILE_M) -> torch.Tensor:
    """The launch's own scratch, one a device: the grouped GEMM2 output bf16 [rows, 6144] followed by this rank's
    gathered rows [ROWS, 6144]; armed 0xffff once here (not inside a graph capture: the warm-up call makes it),
    re-armed by every launch for the packs it read."""
    key = _scratch_key(device, rows)
    g = _G.get(key)
    if g is None:
        g = torch.empty((rows + ROWS, LY.HIDDEN), dtype=torch.bfloat16, device=device)
        g.view(torch.int16).fill_(-1)
        _G[key] = g
    return g


def flydsl_moe_tail_ar(
    g2: Gemm2Call,
    topk_ids: torch.Tensor,  # int32 [8, 4]
    topids_to_rows: torch.Tensor,  # int32 [8, 4]
    gather_w: torch.Tensor,  # fp32 [8, 4]
    addend: torch.Tensor | None,  # bf16 [8, N]
    addend_scale: float,
    res: torch.Tensor,  # bf16 [8, N]
    norm_w: torch.Tensor,  # bf16 [N] (Gemma: the kernel adds 1)
    eps: float,
    inbox: list[int],
    ctr: int,
    slot: int,
    rank: int,
    *,
    maxit: int = MAXIT,
    max_rows: int = 256,
    out: torch.Tensor | None = None,
    out_scale: torch.Tensor | None = None,
    res_out: torch.Tensor | None = None,
    timeline: torch.Tensor | None = None,
    stream=None,
    tile_n: int | None = None,
    num_buffers: int | None = None,
    waves_per_tensor_tdm: int | None = None,
    probe: int | None = None,
    grouped: torch.Tensor | None = None,
    fp8_out: bool = True,
    sync: bool = True,
    shared_down=None,
):
    """Returns (out E4M3 [8, N], res_out bf16 [8, N], scale E8M0 [8, N / 32]), or with ``fp8_out`` False (out bf16
    [8, N], res_out): what ``k5_fused_ar_rms_tail(..., gemma_norm=True, fp8_out=...)`` returns after production's
    GEMM2 launch."""
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from aiter.utility.dtypes import fp8, fp8_e8m0

    from .kernels.moe_tail_ar import flagged_c_stores, launch_moe_tail_ar

    tokens, n = res.shape
    why = supported(g2, tokens, addend is not None)
    assert why is None, f"moe_tail_ar: {why}"
    assert len(inbox) == 4 and slot == max_rows * LY.ROW_PACKS and tokens <= max_rows
    for t_, dt, shp in ((topk_ids, torch.int32, (tokens, 4)), (topids_to_rows, torch.int32, (tokens, 4)),
                        (gather_w, torch.float32, (tokens, 4))):  # fmt: skip
        assert t_.dtype == dt and tuple(t_.shape) == shp and t_.is_contiguous() and t_.data_ptr() % 16 == 0
    assert res.dtype == torch.bfloat16 and res.is_contiguous() and res.data_ptr() % 16 == 0
    assert norm_w.dtype == torch.bfloat16 and norm_w.numel() == n and norm_w.data_ptr() % 16 == 0
    if addend is not None:
        assert addend.dtype == torch.bfloat16 and addend.shape == res.shape and addend.is_contiguous()
        assert addend.data_ptr() % 16 == 0
    dev = res.device
    if grouped is None:
        grouped = grouped_scratch(dev, g2.contiguous_m)
    assert grouped.dtype == torch.bfloat16 and tuple(grouped.shape) == (g2.contiguous_m + tokens, n)
    assert grouped.is_contiguous() and grouped.data_ptr() % 16 == 0
    if out is None:
        out = torch.empty((tokens, n), dtype=fp8 if fp8_out else torch.bfloat16, device=dev)
    if out_scale is None and fp8_out:
        out_scale = torch.empty((tokens, n // 32), dtype=fp8_e8m0, device=dev)
    if res_out is None:
        res_out = torch.empty((tokens, n), dtype=torch.bfloat16, device=dev)
    if stream is None:
        stream = current_stream_arg()
    has_add = int(addend is not None)
    has_scale = int(has_add and float(addend_scale) != 1.0)
    tl = 0 if timeline is None else 1
    spec = (
        g2.K, TILE_N if tile_n is None else tile_n, TILE_K, N_WARP, NUM_BUFFERS if num_buffers is None else num_buffers,
        int(g2.a_is_fp4), int(g2.n_experts),
        WAVES_PER_TENSOR_TDM if waves_per_tensor_tdm is None else waves_per_tensor_tdm, int(g2.tdm_b_th),
        int(g2.lds_soa_load_interleave), int(g2.rolled_drain), int(g2.a_preshuffled), tokens, has_add, has_scale,
        max_rows, PROBE if probe is None else probe, int(bool(fp8_out)), tl, int(shared_down is not None),
    )  # fmt: skip
    sdt = None
    if shared_down is not None:
        assert addend is not None and timeline is None
        sdt = shared_down_table(*shared_down)[0]
    ctr_ptr = flyc.from_c_void_p(fx.Int32, int(ctr))
    args = (
        ptr_arg(grouped, fx.BFloat16), ptr_arg(g2.a), ptr_arg(g2.w), ptr_arg(g2.a_scales, fx.Int32),
        ptr_arg(g2.w_scales, fx.Int32), ptr_arg(g2.m_tile_map),
        ptr_arg(topk_ids, fx.Int32), ptr_arg(topids_to_rows, fx.Int32), ptr_arg(gather_w, fx.Float32),
        ptr_arg(addend if addend is not None else res, fx.BFloat16),
        ptr_arg(res, fx.BFloat16), ptr_arg(norm_w, fx.BFloat16), ptr_arg(out), ptr_arg(out_scale if fp8_out else out),
        ptr_arg(res_out, fx.BFloat16), ctr_ptr,
        ptr_arg(timeline, fx.Int32) if timeline is not None else ctr_ptr,
        int(inbox[0]), int(inbox[1]), int(inbox[2]), int(inbox[3]),
        int(g2.contiguous_m), int(slot), int(rank), int(maxit),
        float(addend_scale) if has_scale else 1.0, float(eps),
        stream,
        *spec,
        ptr_arg(sdt if sdt is not None else res, fx.Int64),
    )  # fmt: skip
    key = ("moe_tail_ar", dev.index) + spec
    if key not in _SYNCED:
        # first launch of this build: compile without launching, then barrier the TP group, so the peers' bounded
        # all-reduce polls never wait out a slower rank's JIT compile (no barrier inside a graph capture)
        before = _STATE["c_stores"]
        with flagged_c_stores(_STATE):
            _preload_compiled(launch_moe_tail_ar, *args)
            if sync and torch.distributed.is_initialized() and not torch.cuda.is_current_stream_capturing():
                torch.distributed.barrier()
            _run_compiled(launch_moe_tail_ar, *args, specialization_key=key)
        _STATE.setdefault("traced", {})[key] = _STATE["c_stores"] - before
        _SYNCED.add(key)
    else:
        _run_compiled(launch_moe_tail_ar, *args, specialization_key=key)
    return (out, res_out, out_scale) if fp8_out else (out, res_out)


# ---------------------------------------------------------------------------------------------------------
# production wiring: the grouped MoE keeps its GEMM2 launch back (keep_gemm2), the deferred tail carries it
# (MoeDeferredTail.gemm2) and the k5 all-reduce that takes the tail runs the fused launch (fused_tail)
# ---------------------------------------------------------------------------------------------------------
class KeptGemm2:
    """A routed GEMM2 launch held back for the all-reduce that takes its MoE's deferred tail."""

    __slots__ = ("args", "kw", "ids")

    def __init__(self, args, kw, ids):
        self.args, self.kw, self.ids = args, kw, ids

    def launch(self) -> None:
        """Production's GEMM2 launch, as it was (any consumer other than the fused launch)."""
        from aiter.ops.flydsl.grouped_gemm_mxfp4 import flydsl_grouped_gemm_a8w4_masked

        flydsl_grouped_gemm_a8w4_masked(*self.args, **self.kw)

    def call(self) -> Gemm2Call:
        from aiter.ops.flydsl.grouped_gemm_mxfp4 import _select_tdm_b_th

        out, a, w, a_scales, w_scales, m_tile_map = self.args
        kw = self.kw
        return Gemm2Call(
            out=out, a=a, w=w, a_scales=a_scales, w_scales=w_scales, m_tile_map=m_tile_map,
            n_experts=int(kw["n_experts"]), contiguous_m=int(kw["contiguous_m"]), N=int(kw["N"]), K=int(kw["K"]),
            a_is_fp4=int(kw.get("a_is_fp4", 0)), a_preshuffled=int(kw.get("a_preshuffled", 0)),
            tdm_b_th=_select_tdm_b_th(kw.get("tdm_b_th", 0)),
            lds_soa_load_interleave=int(kw.get("lds_soa_load_interleave", 0)),
            rolled_drain=int(kw.get("rolled_drain", 0)),
        )  # fmt: skip


def keep_gemm2(args, kw, tp_gather, tokens: int) -> KeptGemm2 | None:
    """Called by the grouped MoE in place of its GEMM2 launch (``flydsl_grouped_gemm_a8w4_masked(*args, **kw)``):
    a ``KeptGemm2`` when the launch can move into the all-reduce that takes this call's deferred tail, else None
    (the caller launches GEMM2 as usual). Shapes, dtypes and the defer scope only: every rank decides the same."""
    if not ENABLED or tp_gather is None or tokens != ROWS:
        return None
    from aiter.ops.flydsl.moe_deferred_tail import _defer_scope

    if _defer_scope() is None:
        return None  # nobody will take the tail: the gather-reduce launch reads GEMM2's rows right away
    ids = tp_gather["topk_ids"]
    plain = (
        kw.get("stage1_act", 0) == 0 and kw.get("tile_m") == LY.TILE_M and kw.get("bias") is None
        and kw.get("stage2_scatter") is None and kw.get("shared_fold") is None and not kw.get("out_is_f16")
        and not kw.get("persistent_workers") and int(kw.get("k_waves", 1)) == 1 and int(kw.get("a_is_fp4", 0)) == 1
        and not kw.get("row_major_ascale") and not kw.get("a_row_stride_bytes")
        and not kw.get("a_scale_row_stride_bytes") and not kw.get("m_align") and not kw.get("tdm_as_in_prologue")
        and not kw.get("next_stage_prefetch") and int(kw.get("cluster_m", -1)) in (-1, 1)
        and int(kw.get("cluster_n", -1)) in (-1, 1)
        and ids.dtype == torch.int32 and tuple(ids.shape) == (ROWS, 4) and ids.is_contiguous()
        and ids.data_ptr() % 16 == 0
    )  # fmt: skip
    if not plain:
        return None
    kept = KeptGemm2(args, kw, ids)
    return kept if supported(kept.call(), tokens, True) is None else None


def k5_context(ca):
    """(rank, max_rows, slot, [inbox0..3], ctr) of the custom all-reduce's k5 context (its first eight words)."""
    import ctypes

    raw = ctypes.string_at(int(ca._k5_ctx), 64)
    cw = [int.from_bytes(raw[o:o + 8], "little") for o in range(0, 64, 8)]
    return cw[0], cw[1], cw[2], cw[3:7], cw[7]


def fused_tail(ca, tail, operands, res_inp, w, eps, gemma_norm, lamport_order, fp8_out):
    """``CustomAllreduce.k5_fused_ar_rms_tail`` for a tail that carries its kept GEMM2: the fused launch, or None
    when it cannot take the call (the caller launches the kept GEMM2 and goes on as before). ``operands``:
    ``tail.k5_operands()``."""
    kept = tail.gemm2
    if kept is None or operands is None or not gemma_norm or lamport_order:
        return None
    _, rows_t, w_t, add = operands
    # A registered addend: its `before` is production's shared down into it, which this launch's shared-down
    # workgroups replace (no join). Its output buffer is armed outside a capture (the eager warm-up call).
    sd = SHARED_DOWN.get(add.data_ptr()) if add is not None and TIMELINES is None else None
    if sd is not None and _sd_key(*sd) not in _SD:
        if torch.cuda.is_current_stream_capturing():
            sd = None
        else:
            shared_down_table(*sd)
    if tail.before is not None and not JOIN_FIRST_OK and sd is None:
        return None  # one stream only (see JOIN_FIRST_OK)
    tokens = tail.out.shape[0]
    if tokens != ROWS or tuple(res_inp.shape) != (ROWS, LY.HIDDEN) or res_inp.dtype != torch.bfloat16:
        return None
    dev = res_inp.device
    if _scratch_key(dev, LY.MAX_M_TILES * LY.TILE_M) not in _G and torch.cuda.is_current_stream_capturing():
        return None  # the scratch is armed outside a capture (the eager warm-up call makes it)
    rank, max_rows, slot, inbox, ctr = k5_context(ca)
    if slot != max_rows * LY.ROW_PACKS or tokens > max_rows:
        return None
    from aiter.dist.device_communicators.custom_all_reduce import _K5_AR_MAXIT, _lamport_aligned

    res_inp = _lamport_aligned(res_inp.contiguous())
    w = _lamport_aligned(w.contiguous())
    if sd is not None:
        tail.before = None  # the fallback shared down, computed by this launch instead
    else:
        tail.run_before()
    tail.gemm2 = None
    tlb = None
    if TIMELINES is not None:
        tlb = torch.zeros((LY.grid(TILE_N, ROWS), 8), dtype=torch.int32, device=dev)
        TIMELINES.append(tlb)
    return flydsl_moe_tail_ar(
        kept.call(), kept.ids, rows_t, w_t, add, tail.scale, res_inp, w, eps, inbox, ctr, slot, rank,
        maxit=_K5_AR_MAXIT, max_rows=max_rows, fp8_out=bool(fp8_out), timeline=tlb, shared_down=sd,
    )


__all__ = [
    "SHARED_DOWN", "shared_down_table", "ENABLED", "Gemm2Call", "KeptGemm2", "MAXIT", "ROWS", "flydsl_moe_tail_ar", "fused_tail", "grouped_scratch",
    "keep_gemm2", "supported",
]  # fmt: skip


def register_deferred_tail_launch() -> bool:
    """Hand aiter's deferred MoE tail (``moe_deferred_tail``) this module's GEMM2 hold-back (``keep_gemm2``) and
    fused GEMM2 + tail + all-reduce launch (``fused_tail``). False (with a warning) when aiter has no such hook."""
    try:
        from aiter.ops.flydsl.moe_deferred_tail import register_tail_launch
    except ImportError as e:
        import logging

        logging.getLogger("atom").warning(
            f"MoE tail launch not registered: aiter has no deferred-tail launch hook ({e}); "
            "GEMM2 and the tail all-reduce run as separate launches"
        )
        return False
    register_tail_launch(keep_gemm2, fused_tail)
    return True
