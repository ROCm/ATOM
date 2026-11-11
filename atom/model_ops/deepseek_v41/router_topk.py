# SPDX-License-Identifier: MIT
"""V4.1 routing top-k, in Triton, bit for bit as aiter's ``topk_gating``.

``topk_gating`` (sqrtsoftplus, biased selection, renormalized, scaled) runs a
different register kernel for 384 experts per arch, and its tie order follows
that kernel's lane layout. ``_LAYOUTS`` lists the archs covered here, each
with the layout of the kernel it mirrors:

- gfx950: ``topk_gating_kernel_opt`` (all row counts); one 64-lane wave, six
  experts per lane. Taken up to 2048 rows and only in steps captured into a
  CUDA graph: above that row count, and in eager steps, where the Triton
  launch costs more host time, ``topk_gating`` itself is faster.
- gfx1250: ``topk_gating_kernel_opt_multiwave<.., 384, 4, ..>`` up to 1024
  rows; four 32-lane waves, three experts per lane, and a merge of the four
  wave lists.

Lane l of wave w holds experts c + C i (c = L w + l, C = all lanes of the
row, i below the experts per lane), sorts them with the same compare-exchange
network, runs top-k rounds of a wave argmax (lowest lane on a tie, lane 0
when no lane matches) with per-lane cursors, and, with several waves, merges
their lists with a strict compare, wave 0 first. So ties, NaN rows and the
renorm sum order come out as there. One program per row: the rows are
independent and short.
"""

import torch
import triton
import triton.language as tl
from aiter.jit.utils.chip_info import get_gfx_runtime

EXPERTS = 384

# arch -> (waves, lanes per wave, experts per lane, max rows, graphs only):
# the layout of topk_gating's 384-expert kernel there, and where it is taken
_LAYOUTS = {
    "gfx950": (1, 64, 6, 2048, True),
    "gfx1250": (4, 32, 3, 1024, False),
}


@triton.jit
def _score(x):
    # sqrt(softplus(x)) as aiter's compute_score<SCORE_SQRTSOFTPLUS>
    x = tl.where(x > 3.4028234663852886e38, 3.4028234663852886e38, x)
    sp = tl.where(
        x > 20.0,
        x,
        tl.math.log2(1.0 + tl.math.exp2(x * 1.4426950408889634)) * 0.6931471805599453,
    )
    return tl.sqrt_rn(sp)


@triton.jit
def _slot(base, bias_ptr, c, off, lm, zi, HAS_BIAS: tl.constexpr):
    # experts c + off: selection value, weight, expert id
    s = _score(tl.load(base + off, mask=lm, other=0.0).to(tl.float32))
    # the weight of a NaN score is 0
    o = tl.where(s != s, 0.0, s)  # noqa: PLR0124 - Triton NaN check
    if HAS_BIAS:
        s = s + tl.load(bias_ptr + c + off)[None, :]
    # a NaN selection value sinks to -inf
    s = tl.where(s != s, float("-inf"), s)  # noqa: PLR0124 - Triton NaN check
    return s, o, zi + c[None, :] + off


@triton.jit
def _cas(va, oa, da, vb, ob, db):
    # compare-exchange, descending: swap when va < vb (a NaN compares false)
    sw = va < vb
    return (
        tl.where(sw, vb, va),
        tl.where(sw, ob, oa),
        tl.where(sw, db, da),
        tl.where(sw, va, vb),
        tl.where(sw, oa, ob),
        tl.where(sw, da, db),
    )


@triton.jit
def _pick3(c, a0, a1, a2, other):
    return tl.where(c == 0, a0, tl.where(c == 1, a1, tl.where(c == 2, a2, other)))


@triton.jit
def _pick6(c, a0, a1, a2, a3, a4, a5, other):
    return tl.where(
        c < 3, _pick3(c, a0, a1, a2, other), _pick3(c - 3, a3, a4, a5, other)
    )


@triton.jit
def _v41_route_topk_kernel(
    lg_ptr,
    stride_l,
    bias_ptr,
    nvr_ptr,
    w_ptr,
    id_ptr,
    stride_o,
    M,
    scale,
    TOPK: tl.constexpr,
    BM: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_NVR: tl.constexpr,
    WAVES: tl.constexpr,
    LANES: tl.constexpr,
    EPT: tl.constexpr,
):
    C: tl.constexpr = WAVES * LANES
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    if HAS_NVR:
        nvr = tl.load(nvr_ptr)
    else:
        nvr = M
    c = tl.arange(0, C)
    rmask = rows < M
    base = lg_ptr + rows[:, None] * stride_l + c[None, :]
    lm = rmask[:, None]
    zi = tl.zeros([BM, C], dtype=tl.int32)
    s0, o0, d0 = _slot(base, bias_ptr, c, 0, lm, zi, HAS_BIAS)
    s1, o1, d1 = _slot(base, bias_ptr, c, C, lm, zi, HAS_BIAS)
    s2, o2, d2 = _slot(base, bias_ptr, c, 2 * C, lm, zi, HAS_BIAS)
    if EPT == 3:
        # sort_network_desc<3>: (0,1) (0,2) (1,2)
        s0, o0, d0, s1, o1, d1 = _cas(s0, o0, d0, s1, o1, d1)
        s0, o0, d0, s2, o2, d2 = _cas(s0, o0, d0, s2, o2, d2)
        s1, o1, d1, s2, o2, d2 = _cas(s1, o1, d1, s2, o2, d2)
    else:
        s3, o3, d3 = _slot(base, bias_ptr, c, 3 * C, lm, zi, HAS_BIAS)
        s4, o4, d4 = _slot(base, bias_ptr, c, 4 * C, lm, zi, HAS_BIAS)
        s5, o5, d5 = _slot(base, bias_ptr, c, 5 * C, lm, zi, HAS_BIAS)
        # sort_network_desc<6>: its 12 comparators in order
        s0, o0, d0, s1, o1, d1 = _cas(s0, o0, d0, s1, o1, d1)
        s2, o2, d2, s3, o3, d3 = _cas(s2, o2, d2, s3, o3, d3)
        s4, o4, d4, s5, o5, d5 = _cas(s4, o4, d4, s5, o5, d5)
        s0, o0, d0, s2, o2, d2 = _cas(s0, o0, d0, s2, o2, d2)
        s1, o1, d1, s4, o4, d4 = _cas(s1, o1, d1, s4, o4, d4)
        s3, o3, d3, s5, o5, d5 = _cas(s3, o3, d3, s5, o5, d5)
        s0, o0, d0, s1, o1, d1 = _cas(s0, o0, d0, s1, o1, d1)
        s2, o2, d2, s3, o3, d3 = _cas(s2, o2, d2, s3, o3, d3)
        s4, o4, d4, s5, o5, d5 = _cas(s4, o4, d4, s5, o5, d5)
        s1, o1, d1, s2, o2, d2 = _cas(s1, o1, d1, s2, o2, d2)
        s3, o3, d3, s4, o4, d4 = _cas(s3, o3, d3, s4, o4, d4)
        s2, o2, d2, s3, o3, d3 = _cas(s2, o2, d2, s3, o3, d3)
    cur = zi
    lane = tl.arange(0, LANES)
    kk = tl.arange(0, 8)
    out_w = tl.zeros([BM, 8], dtype=tl.float32)
    out_i = tl.zeros([BM, 8], dtype=tl.int32)
    ssum = tl.zeros([BM], dtype=tl.float32)
    sm_v = tl.zeros([BM, WAVES, 8], dtype=tl.float32)
    sm_w = tl.zeros([BM, WAVES, 8], dtype=tl.float32)
    sm_i = tl.zeros([BM, WAVES, 8], dtype=tl.int32)
    # per wave: top-k rounds of argmax over the lanes' cursor heads
    for k in tl.static_range(TOPK):
        if EPT == 3:
            hv = _pick3(cur, s0, s1, s2, float("-inf"))
            hi = _pick3(cur, d0, d1, d2, 0)
            ho = _pick3(cur, o0, o1, o2, 0.0)
        else:
            hv = _pick6(cur, s0, s1, s2, s3, s4, s5, float("-inf"))
            hi = _pick6(cur, d0, d1, d2, d3, d4, d5, 0)
            ho = _pick6(cur, o0, o1, o2, o3, o4, o5, 0.0)
        hv = tl.reshape(hv, [BM, WAVES, LANES])
        hi = tl.reshape(hi, [BM, WAVES, LANES])
        ho = tl.reshape(ho, [BM, WAVES, LANES])
        cur3 = tl.reshape(cur, [BM, WAVES, LANES])
        m = tl.max(hv, axis=2)
        eq = hv == m[:, :, None]
        wl = tl.min(tl.where(eq, lane[None, None, :], LANES), axis=2)
        wl = tl.where(wl == LANES, 0, wl)
        idx = tl.sum(tl.where(lane[None, None, :] == wl[:, :, None], hi, 0), axis=2)
        won = (cur3 < EPT) & (hi == idx[:, :, None])
        myo = tl.where(won, ho, 0.0)
        wt = tl.sum(
            tl.where(lane[None, None, :] == (idx & (LANES - 1))[:, :, None], myo, 0.0),
            axis=2,
        )
        cur = tl.reshape(cur3 + won.to(tl.int32), [BM, C])
        if WAVES == 1:
            # one wave: its list is the result, summed in selection order
            wt1 = tl.sum(wt, axis=1)
            ssum = ssum + wt1
            out_w = tl.where(kk[None, :] == k, wt1[:, None], out_w)
            out_i = tl.where(kk[None, :] == k, tl.sum(idx, axis=1)[:, None], out_i)
        else:
            sel = kk[None, None, :] == k
            sm_v = tl.where(sel, m[:, :, None], sm_v)
            sm_w = tl.where(sel, wt[:, :, None], sm_w)
            sm_i = tl.where(sel, idx[:, :, None], sm_i)
    if WAVES > 1:
        # merge of the wave lists: strict >, wave 0 first
        wv = tl.arange(0, WAVES)
        ptr = tl.zeros([BM, WAVES], dtype=tl.int32)
        for k in tl.static_range(TOPK):
            at = kk[None, None, :] == ptr[:, :, None]
            hv = tl.sum(tl.where(at, sm_v, 0.0), axis=2)
            hw = tl.sum(tl.where(at, sm_w, 0.0), axis=2)
            hid = tl.sum(tl.where(at, sm_i, 0), axis=2)
            bv = tl.sum(tl.where(wv[None, :] == 0, hv, 0.0), axis=1)
            bw = tl.zeros([BM], dtype=tl.int32)
            for w in tl.static_range(1, WAVES):
                v = tl.sum(tl.where(wv[None, :] == w, hv, 0.0), axis=1)
                up = v > bv
                bv = tl.where(up, v, bv)
                bw = tl.where(up, w, bw)
            pw = wv[None, :] == bw[:, None]
            wt = tl.sum(tl.where(pw, hw, 0.0), axis=1)
            eid = tl.sum(tl.where(pw, hid, 0), axis=1)
            ptr = ptr + pw.to(tl.int32)
            ssum = ssum + wt
            out_w = tl.where(kk[None, :] == k, wt[:, None], out_w)
            out_i = tl.where(kk[None, :] == k, eid[:, None], out_i)
    sc = tl.math.div_rn(
        tl.zeros([BM], dtype=tl.float32) + scale, tl.maximum(ssum, 1e-20)
    )
    out_w = out_w * sc[:, None]
    # rows at or past the pass's real count: weight 0, id -1
    pad = rows >= nvr
    out_w = tl.where(pad[:, None], 0.0, out_w)
    out_i = tl.where(pad[:, None], -1, out_i)
    omask = rmask[:, None] & (kk[None, :] < TOPK)
    tl.store(w_ptr + rows[:, None] * stride_o + kk[None, :], out_w, mask=omask)
    tl.store(id_ptr + rows[:, None] * stride_o + kk[None, :], out_i, mask=omask)


def route_topk_supported(gating_output, bias, topk, renormalize) -> bool:
    layout = _LAYOUTS.get(get_gfx_runtime())
    return (
        layout is not None
        and renormalize
        and 1 <= topk <= 8
        and gating_output.dim() == 2
        and 1 <= gating_output.shape[0] <= layout[3]
        and (not layout[4] or torch.cuda.is_current_stream_capturing())
        and gating_output.shape[1] == EXPERTS
        and gating_output.dtype == torch.bfloat16
        and gating_output.stride(1) == 1
        and (
            bias is None
            or (
                bias.dtype == torch.float32
                and bias.shape == (EXPERTS,)
                and bias.is_contiguous()
            )
        )
    )


def route_topk(gating_output, bias, topk, scale, num_valid_rows=None):
    """(weights fp32 [M, topk], ids int32 [M, topk]); see route_topk_supported."""
    waves, lanes, ept, _, _ = _LAYOUTS[get_gfx_runtime()]
    rows = gating_output.shape[0]
    ids = torch.empty((rows, topk), dtype=torch.int32, device=gating_output.device)
    weights = torch.empty(
        (rows, topk), dtype=torch.float32, device=gating_output.device
    )
    _v41_route_topk_kernel[(rows,)](
        gating_output,
        gating_output.stride(0),
        bias if bias is not None else gating_output,
        num_valid_rows if num_valid_rows is not None else gating_output,
        weights,
        ids,
        weights.stride(0),
        rows,
        scale,
        TOPK=topk,
        BM=1,
        HAS_BIAS=bias is not None,
        HAS_NVR=num_valid_rows is not None,
        WAVES=waves,
        LANES=lanes,
        EPT=ept,
        num_warps=1,
    )
    return weights, ids
