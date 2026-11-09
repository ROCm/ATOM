# SPDX-License-Identifier: MIT
"""V4.1 routing top-k for decode widths on gfx1250, in Triton.

Same outputs, bit for bit, as aiter's ``topk_gating`` (sqrtsoftplus, biased
selection, renormalized, scaled) at the row counts where that op runs its
``topk_gating_kernel_opt_multiwave<.., 384, 4, ..>`` kernel (up to 1024 rows):
the kernel below follows that one step by step. Lane l of wave w holds experts
c + 128 i (c = 32 w + l, i < 3), sorts them with the same compare-exchange
network, runs top-k rounds of a wave argmax (lowest lane on a tie, lane 0 when
no lane matches) with per-lane cursors, and merges the four wave lists with a
strict compare. So ties, NaN rows and the renorm sum order come out as there.
One program per row, one wave: the rows are independent and short.
"""

import torch
import triton
import triton.language as tl
from aiter.jit.utils.chip_info import get_gfx

EXPERTS = 384
MAX_ROWS = 1024  # the multiwave kernel's row range in topk_gating


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
def _pick(c, a0, a1, a2, other):
    return tl.where(c == 0, a0, tl.where(c == 1, a1, tl.where(c == 2, a2, other)))


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
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    if HAS_NVR:
        nvr = tl.load(nvr_ptr)
    else:
        nvr = M
    c = tl.arange(0, 128)
    rmask = rows < M
    base = lg_ptr + rows[:, None] * stride_l + c[None, :]
    lm = rmask[:, None]
    s0 = _score(tl.load(base, mask=lm, other=0.0).to(tl.float32))
    s1 = _score(tl.load(base + 128, mask=lm, other=0.0).to(tl.float32))
    s2 = _score(tl.load(base + 256, mask=lm, other=0.0).to(tl.float32))
    # the weight of a NaN score is 0
    o0 = tl.where(s0 != s0, 0.0, s0)  # noqa: PLR0124 - Triton NaN check
    o1 = tl.where(s1 != s1, 0.0, s1)  # noqa: PLR0124 - Triton NaN check
    o2 = tl.where(s2 != s2, 0.0, s2)  # noqa: PLR0124 - Triton NaN check
    if HAS_BIAS:
        s0 = s0 + tl.load(bias_ptr + c)[None, :]
        s1 = s1 + tl.load(bias_ptr + c + 128)[None, :]
        s2 = s2 + tl.load(bias_ptr + c + 256)[None, :]
    # a NaN selection value sinks to -inf
    s0 = tl.where(s0 != s0, float("-inf"), s0)  # noqa: PLR0124 - Triton NaN check
    s1 = tl.where(s1 != s1, float("-inf"), s1)  # noqa: PLR0124 - Triton NaN check
    s2 = tl.where(s2 != s2, float("-inf"), s2)  # noqa: PLR0124 - Triton NaN check
    zi = tl.zeros([BM, 128], dtype=tl.int32)
    d0 = zi + c[None, :]
    d1 = d0 + 128
    d2 = d0 + 256
    # three-element sorting network: (0,1) (0,2) (1,2)
    s0, o0, d0, s1, o1, d1 = _cas(s0, o0, d0, s1, o1, d1)
    s0, o0, d0, s2, o2, d2 = _cas(s0, o0, d0, s2, o2, d2)
    s1, o1, d1, s2, o2, d2 = _cas(s1, o1, d1, s2, o2, d2)
    cur = zi
    lane = tl.arange(0, 32)
    kk = tl.arange(0, 8)
    sm_v = tl.zeros([BM, 4, 8], dtype=tl.float32)
    sm_w = tl.zeros([BM, 4, 8], dtype=tl.float32)
    sm_i = tl.zeros([BM, 4, 8], dtype=tl.int32)
    # per wave: top-k rounds of argmax over the lanes' cursor heads
    for k in tl.static_range(TOPK):
        hv = tl.reshape(_pick(cur, s0, s1, s2, float("-inf")), [BM, 4, 32])
        hi = tl.reshape(_pick(cur, d0, d1, d2, 0), [BM, 4, 32])
        ho = tl.reshape(_pick(cur, o0, o1, o2, 0.0), [BM, 4, 32])
        cur3 = tl.reshape(cur, [BM, 4, 32])
        m = tl.max(hv, axis=2)
        eq = hv == m[:, :, None]
        wl = tl.min(tl.where(eq, lane[None, None, :], 32), axis=2)
        wl = tl.where(wl == 32, 0, wl)
        idx = tl.sum(tl.where(lane[None, None, :] == wl[:, :, None], hi, 0), axis=2)
        won = (cur3 < 3) & (hi == idx[:, :, None])
        myo = tl.where(won, ho, 0.0)
        wt = tl.sum(
            tl.where(lane[None, None, :] == (idx & 31)[:, :, None], myo, 0.0), axis=2
        )
        cur = tl.reshape(cur3 + won.to(tl.int32), [BM, 128])
        sel = kk[None, None, :] == k
        sm_v = tl.where(sel, m[:, :, None], sm_v)
        sm_w = tl.where(sel, wt[:, :, None], sm_w)
        sm_i = tl.where(sel, idx[:, :, None], sm_i)
    # merge of the four wave lists: strict >, wave 0 first
    wv = tl.arange(0, 4)
    ptr = tl.zeros([BM, 4], dtype=tl.int32)
    ssum = tl.zeros([BM], dtype=tl.float32)
    out_w = tl.zeros([BM, 8], dtype=tl.float32)
    out_i = tl.zeros([BM, 8], dtype=tl.int32)
    for k in tl.static_range(TOPK):
        at = kk[None, None, :] == ptr[:, :, None]
        hv = tl.sum(tl.where(at, sm_v, 0.0), axis=2)
        hw = tl.sum(tl.where(at, sm_w, 0.0), axis=2)
        hid = tl.sum(tl.where(at, sm_i, 0), axis=2)
        bv = tl.sum(tl.where(wv[None, :] == 0, hv, 0.0), axis=1)
        bw = tl.zeros([BM], dtype=tl.int32)
        for w in tl.static_range(1, 4):
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
    return (
        get_gfx() in ("gfx1250", "gfx1250-strict")
        and renormalize
        and 1 <= topk <= 8
        and gating_output.dim() == 2
        and 1 <= gating_output.shape[0] <= MAX_ROWS
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
        num_warps=1,
    )
    return weights, ids
