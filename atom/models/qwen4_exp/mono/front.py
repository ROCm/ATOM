# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The MTP drafter's front end in one launch.

ATOM's drafter (``Qwen4ExpMultiTokenPredictor.forward``) runs it as six: the
tokens' embeddings, ``pre_fc_norm_embedding``, ``fc_embedding``,
``pre_fc_norm_hidden`` over each row's whole HC bundle, ``fc_hidden`` on every
stream and the sum into the streams. A draft step's few rows make each of
them a few microseconds of launch for almost no work. :class:`DraftFront`
runs them as one kernel, ``RB`` rows' four streams per program (the MFMA's 16
rows), ``BN`` output columns each: narrow, so that enough programs stream the
two weights from HBM, with ``BK`` deep steps in flight. Up to ``MAX_ROWS``
rows: past them the ops are faster.

Rounding follows ATOM's ops: the normalized values, both projections and the
sum are rounded to bf16 before use. The norms' sums of squares and the
projections' dot products sum in their own order, so the streams are not
bit-identical to the ops' (a draft's rows only steer which tokens it guesses).
"""

import torch
import triton
import triton.language as tl

BF16 = torch.bfloat16


@triton.jit
def _draft_front_kernel(
    ids_ptr,
    embed_ptr,
    h_ptr,
    g_e_ptr,
    g_h_ptr,
    w_e_ptr,
    w_h_ptr,
    out_ptr,
    n,
    stride_embed,
    stride_h,
    stride_out,
    eps_e,
    eps_h,
    H: tl.constexpr,
    HC: tl.constexpr,
    RB: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    BS: tl.constexpr,
):
    M: tl.constexpr = RB * HC
    m = tl.arange(0, M)
    # A row m: stream m % HC of token row m // HC
    row = tl.program_id(1) * RB + m // HC
    s = m % HC
    rmask = (row < n)[:, None]
    tok = tl.load(ids_ptr + row, mask=row < n, other=0).to(tl.int64)
    e_rows = embed_ptr + tok[:, None] * stride_embed
    h_rows = h_ptr + row.to(tl.int64)[:, None] * stride_h + (s * H)[:, None]

    # each row's sums of squares: its embedding's, its stream's (the bundle's below)
    ss_e = tl.zeros((M,), tl.float32)
    ss_h = tl.zeros((M,), tl.float32)
    for k0 in range(0, H, BS):
        ks = (k0 + tl.arange(0, BS))[None, :]
        e = tl.load(e_rows + ks, mask=rmask, other=0.0).to(tl.float32)
        x = tl.load(h_rows + ks, mask=rmask, other=0.0).to(tl.float32)
        ss_e += tl.sum(e * e, 1)
        ss_h += tl.sum(x * x, 1)
    bundle = tl.sum(tl.reshape(ss_h, (RB, HC)), 1)
    ss_h = tl.reshape(tl.broadcast_to(bundle[:, None], (RB, HC)), (M,))
    rstd_e = tl.math.rsqrt(ss_e / H + eps_e)[:, None]
    rstd_h = tl.math.rsqrt(ss_h / (H * HC) + eps_h)[:, None]

    ns = (tl.program_id(0) * BN + tl.arange(0, BN)).to(tl.int64)[None, :]
    acc_e = tl.zeros((M, BN), tl.float32)
    acc_h = tl.zeros((M, BN), tl.float32)
    for k0 in range(0, H, BK):
        kr = k0 + tl.arange(0, BK)
        ks = kr[None, :]
        ge = (tl.load(g_e_ptr + kr).to(tl.float32) + 1.0)[None, :]
        gh = tl.load(g_h_ptr + (s * H)[:, None] + ks).to(tl.float32) + 1.0
        e = tl.load(e_rows + ks, mask=rmask, other=0.0).to(tl.float32)
        x = tl.load(h_rows + ks, mask=rmask, other=0.0).to(tl.float32)
        en = (e * rstd_e * ge).to(tl.bfloat16)
        hn = (x * rstd_h * gh).to(tl.bfloat16)
        # [BK, BN] of the [N, K] weights
        we = tl.load(w_e_ptr + ns * H + kr[:, None])
        wh = tl.load(w_h_ptr + ns * H + kr[:, None])
        acc_e = tl.dot(en, we, acc_e)
        acc_h = tl.dot(hn, wh, acc_h)
    y = acc_h.to(tl.bfloat16).to(tl.float32) + acc_e.to(tl.bfloat16).to(tl.float32)
    out = out_ptr + row.to(tl.int64)[:, None] * stride_out + (s * H)[:, None] + ns
    tl.store(out, y.to(tl.bfloat16), mask=rmask)


def refusal(predictor) -> str | None:
    """Why :class:`DraftFront` cannot run ``predictor``'s front end; None if it can."""
    p = predictor
    hidden = p.hidden_size
    for name in ("fc_embedding", "fc_hidden"):
        lin = getattr(p, name)
        w = lin.weight
        if (
            lin.bias is not None
            or w.dtype != BF16
            or tuple(w.shape) != (hidden, hidden)
        ):
            return f"{name} is not a bias-free bf16 [{hidden}, {hidden}] weight"
        if not w.is_contiguous():
            return f"{name}'s weight is not contiguous"
    if p.embed_tokens.weight.shape[1] != hidden:
        return "the embeddings are sharded"
    for _, (bn, bk, bs, _, _) in DraftFront.TILES:
        if hidden % max(bk, bs) or hidden % bn:
            return f"hidden size {hidden} is not a multiple of {max(bk, bs, bn)}"
    return None


class DraftFront:
    """``predictor``'s front end: ``(ids [n], target HC streams [n, hc * H])``
    into ``out [n, hc * H]``, the draft layer's input streams."""

    # rows per program; by rows, the output columns per program, the K step of
    # the projections and of the norms, warps and stages. HBM-cold on MI355X
    # (the real MTP weights) 22.0 / 23.3 / 29.0 us at 1 / 4 / 8 rows against
    # the ops' 30.7 / 31.8 / 32.3; at 16 rows no tiling beat the ops' 33.5
    RB = 4
    TILES = ((4, (16, 256, 512, 4, 3)), (8, (32, 256, 512, 4, 2)))
    MAX_ROWS = TILES[-1][0]

    def __init__(self, predictor):
        p = predictor
        self.hidden, self.hc = p.hidden_size, p.hc_count
        self.embed = p.embed_tokens.weight
        self.g_e = p.pre_fc_norm_embedding.weight
        self.g_h = p.pre_fc_norm_hidden.weight
        self.eps_e = p.pre_fc_norm_embedding.variance_epsilon
        self.eps_h = p.pre_fc_norm_hidden.variance_epsilon
        self.w_e, self.w_h = p.fc_embedding.weight, p.fc_hidden.weight

    def __call__(self, ids: torch.Tensor, hidden: torch.Tensor, out: torch.Tensor):
        n = ids.numel()
        assert n <= self.MAX_ROWS
        assert hidden.shape == out.shape == (n, self.hc * self.hidden)
        assert hidden.stride(-1) == out.stride(-1) == 1
        bn, bk, bs, warps, stages = next(t for rows, t in self.TILES if n <= rows)
        grid = (self.hidden // bn, triton.cdiv(n, self.RB))
        _draft_front_kernel[grid](
            ids,
            self.embed,
            hidden,
            self.g_e,
            self.g_h,
            self.w_e,
            self.w_h,
            out,
            n,
            self.embed.stride(0),
            hidden.stride(0),
            out.stride(0),
            self.eps_e,
            self.eps_h,
            H=self.hidden,
            HC=self.hc,
            RB=self.RB,
            BN=bn,
            BK=bk,
            BS=bs,
            num_warps=warps,
            num_stages=stages,
        )
        return out


def draft_front(front, predictor, ids, hidden, out):
    """The drafter's front end into ``out``: ``front`` (a :class:`DraftFront`)
    if there is one and the rows are few enough, else ATOM's ops, op for op."""
    if front is not None and ids.numel() <= front.MAX_ROWS:
        return front(ids, hidden, out)
    p = predictor
    n, hc, h = ids.numel(), p.hc_count, p.hidden_size
    embedding = p.fc_embedding(p.pre_fc_norm_embedding(p.embed_tokens(ids)))
    hidden = p.fc_hidden(p.pre_fc_norm_hidden(hidden).reshape(-1, h))
    torch.add(hidden.view(n, hc, h), embedding[:, None, :], out=out.view(n, hc, h))
    return out
