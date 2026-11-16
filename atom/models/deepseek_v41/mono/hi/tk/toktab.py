# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Ports TileRT 0.1.7 (MIT, Copyright (c) TileRT-team) ``tilert/models/ds_v4_1_flash/toktab.py`` (``build_table``) and
# the device arithmetic of the kernels it runs (``embed_kernel``, ``mhc_v3_kernel<S, 2>``,
# ``q_kv_xf_kg_kernel<S, true, 8, 0>`` with its mHC finalize, ``toktab::kv_extract_kernel``), as the gfx950 ports
# in ``hi/kernels/gfx950`` (``mhc_pre``, ``qkv``) spell it out.
"""The DSpark token tables re-derived on the CPU (dev only: the default, bit-exact build runs the kernels on the GPU,
``tk.toktab_device``): for every token, the first block of a scope (``layers.0``: the target's first
layer; ``mtp.0``: the first draft layer) run on that token alone, at position 0, before its RoPE.

The block's inputs are the token's embedding (4 identical mHC copies, mHC weights (1, 0, 0, 0)), so its outputs are
a function of the token: ``toktab.gather_kernel`` reads them per row instead of running the block. Per token:

- ``q`` (bf16 [8 x 512], this rank's heads): mHC PRE (RMS of the embedding, ``attn_gamma``, 1 x 32 E4M3 + ue8m0),
  q_a (fp8 x fp8 blocks of 32, the MFMA's block sums, the block scales' product, a chain a wave and the waves in
  order), q_norm (IEEE sqrt / division), E4M3 + ue8m0 again, q_b (the same GEMM, 4 waves), bf16, RoPE at
  position 0 (cos 1, sin 0: only signed zeros change);
- ``kv`` (bf16 [512]): the window kv GEMM's bf16 output (8 waves) RMS-normed by ``kv_norm`` as
  ``kv_extract_kernel`` does it (IEEE sqrt and division);
- ``hc`` (f32 [24]): the mHC finalize (``qkv.mhc_finalize_row``): the mix GEMV's partials (160 K slices, fma
  chains), its RMS, sigmoids and the 20-round Sinkhorn of the 4 x 4 comb logits.

Every f32 operation is the device's, in its order (``fma32``: a fused multiply-add; plain ``+`` / ``*`` / ``/`` /
``sqrt``: IEEE float32), except the hardware approximations ``v_rsq_f32`` / ``v_exp_f32`` / ``v_rcp_f32`` (``HW``:
``CpuHw`` models them correctly rounded, ``DeviceHw`` asks the GPU) and the MFMA's block sum (``mfma_blocks``: exact,
rounded once).

Not bit-exact to TileRT's tables with ``CpuHw``: the hardware's ``v_rsq_f32`` is faithful, not correctly rounded (the
same true-value offset from a rounding boundary rounds either way on different inputs), so no CPU formula reproduces
it; ``hc`` passes through it and ``v_exp_f32`` / ``v_rcp_f32`` on every row, ``q`` / ``kv`` only where a PRE output
sits on an E4M3 boundary. Some ``kv`` / ``q`` rows also differ for reasons not yet found (on such ``kv`` rows one
output differs by one bf16 ulp, as if the norm were a few f32 ulps off).

``build_shared`` (a scope's rank-independent rows: q_b's fp8 input, kv, hc) and ``build_q`` (a rank's q from them);
``build_rank_tables`` -> the 6 tensors in TileRT's order (``layers.0``'s q, kv, hc, then ``mtp.0``'s), the shared rows
built once in a group; ``python -m ...tk.toktab --ckpt C --rank R --ref rank<R>.pt``: build and compare byte for
byte.
"""

from __future__ import annotations

import argparse
import time

import torch

V, D, HC, MIX = 129280, 5120, 4, 24
HEADS_LOCAL, HEAD_DIM, Q_LORA, KV_DIM = 8, 512, 1280, 512
NPES = 8
SCOPES = ("layers.0", "mtp.0")
F32, F64, I32, I64 = torch.float32, torch.float64, torch.int32, torch.int64

# the arithmetic's open choices (the defaults: what the device does, as far as the reference tables show)
OPTS = {
    "gemv_tail_fused": False,  # mhc GEMV: j = 20..31 as mul + add (False) or fma
    "mix_fused": False,  # finalize: mixv * scale + base as mul + add (False) or fma
}


def _c(bits: int) -> torch.Tensor:
    """The f32 with these IEEE bits (a 0-d tensor)."""
    return torch.tensor(
        [bits - (1 << 32) if bits >= 1 << 31 else bits], dtype=I32
    ).view(F32)[0]


# ---------------------------------------------------------------------------------------------------------------
# float32 primitives


def fma32(a, b, c) -> torch.Tensor:
    """RN32(a * b + c) of float32 tensors (broadcast), exactly: the f64 product is exact, the sum rounded to odd
    (TwoSum's error picks the odd neighbour), then rounded to f32 once (53 >= 24 + 2 bits).
    """
    p = torch.as_tensor(a, dtype=F32).double() * torch.as_tensor(b, dtype=F32).double()
    cd = torch.as_tensor(c, dtype=F32).double()
    s = p + cd
    bb = s - p
    e = (p - (s - bb)) + (cd - bb)
    bits = s.contiguous().view(I64)
    odd_fix = (e != 0) & ((bits & 1) == 0)
    step = torch.where((e > 0) == (s > 0), 1, -1)
    return torch.where(odd_fix, bits + step, bits).view(F64).float()


def bf16_bits(x: torch.Tensor) -> torch.Tensor:
    """f32 -> bf16 RNE bits (int64, low 16), no NaN fixup (the kernels' ``bf_rne_hi``)."""
    b = x.contiguous().view(I32).to(I64)
    return ((b + ((b >> 16) & 1) + 0x7FFF) >> 16) & 0xFFFF


def as_bf16(bits: torch.Tensor) -> torch.Tensor:
    return (((bits + 0x8000) & 0xFFFF) - 0x8000).to(torch.int16).view(torch.bfloat16)


def bf16_round(x: torch.Tensor) -> torch.Tensor:
    return as_bf16(bf16_bits(x)).float()


def e8m0(byte: torch.Tensor) -> torch.Tensor:
    """ue8m0 -> f32 2^(e - 127); e == 0 -> 2^-127 (the kernels' ``_e8m0_f32``)."""
    b = byte.to(I32)
    return torch.where(b == 0, 0x400000, b << 23).to(I32).view(F32)


class CpuHw:
    """The hardware approximations modelled correctly rounded (f64, rounded once): not what the GPU returns for
    every input (``v_rsq_f32`` / ``v_exp_f32`` / ``v_rcp_f32`` are faithful, not correctly rounded).
    """

    name = "cpu"

    def __call__(self, op: str, x: torch.Tensor) -> torch.Tensor:
        d = x.double()
        if op == "rsq":
            return torch.rsqrt(d).float()
        if op == "rcp":
            return (1.0 / d).float()
        return torch.exp2(d).float()


class DeviceHw:
    """The hardware approximations evaluated by the GPU itself (``kernels/gfx950/hwmath.py``: the instructions the
    decode kernels use), on ``device``; CPU tensors in and out."""

    name = "device"

    def __init__(self, device):
        from ..kernels.gfx950 import hwmath

        self.device = torch.device(device)
        self.launch = {op: hwmath.build(op)[1] for op in hwmath.OPS}

    def __call__(self, op: str, x: torch.Tensor) -> torch.Tensor:
        src = x.contiguous().to(self.device, F32)
        dst = torch.empty_like(src)
        with torch.cuda.device(self.device):
            if src.numel():
                self.launch[op](src.data_ptr(), dst.data_ptr(), src.numel())
            torch.cuda.synchronize(self.device)
        return dst.cpu().view(x.shape)


HW = CpuHw()  # the provider the build uses (``hardware``)


class hardware:  # (a context manager)
    """``with hardware(DeviceHw(dev)):`` the build's hardware approximations from ``provider``."""

    def __init__(self, provider):
        self.provider = provider

    def __enter__(self):
        global HW
        self.saved, HW = HW, self.provider
        return self.provider

    def __exit__(self, *exc):
        global HW
        HW = self.saved


def rsq_hw(x: torch.Tensor) -> torch.Tensor:
    """``v_rsq_f32``."""
    return HW("rsq", x)


def rcp_hw(x: torch.Tensor) -> torch.Tensor:
    """``v_rcp_f32``."""
    return HW("rcp", x)


def exp2_hw(x: torch.Tensor) -> torch.Tensor:
    """``v_exp_f32``."""
    return HW("exp2", x)


def ldexp32(x: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    return (x.double() * torch.exp2(n.double())).float()


def tree64(v: torch.Tensor) -> torch.Tensor:
    """A wave's DPP sum over its last dim (64 lanes): the balanced tree of adjacent pairs."""
    while v.shape[-1] > 1:
        v = v[..., 0::2] + v[..., 1::2]
    return v[..., 0]


def seq_sum(v: torch.Tensor) -> torch.Tensor:
    """((v0 + v1) + v2) + ... over the last dim."""
    acc = v[..., 0]
    for k in range(1, v.shape[-1]):
        acc = acc + v[..., k]
    return acc


def rms_rsq_hw(ms: torch.Tensor) -> torch.Tensor:
    """The kernels' ``v_rsq_f32`` with its denormal range scaling."""
    tiny = ms < _c(0x00800000)
    r = rsq_hw(torch.where(tiny, ms * _c(0x4B800000), ms))
    return torch.where(tiny, r * _c(0x45800000), r)


def exp_ocml(x: torch.Tensor, neg: bool) -> torch.Tensor:
    """``common.exp_ocml``: OCML's expf of (neg ? -x : x)."""
    if neg:
        hi, lo = _c(0xBFB8AA3B), _c(0xB2A5705F)
    else:
        hi, lo = _c(0x3FB8AA3B), _c(0x32A5705F)
    t = x * hi
    err = fma32(x, hi, -t)
    n = torch.round(t)  # llvm.rint: half to even
    err = fma32(x, lo, err)
    f = (t - n) + err
    e = ldexp32(exp2_hw(f), n)
    inf = _c(0x7F800000)
    if neg:
        e = torch.where(x <= _c(0x42CE8ED0), e, 0.0)
        return torch.where(x >= _c(0xC2B17218), e, inf)
    e = torch.where(x >= _c(0xC2CE8ED0), e, 0.0)
    return torch.where(x <= _c(0x42B17218), e, inf)


def div_nr(x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """``qkv._div_nr``: x / s by ``v_rcp_f32`` and two Newton-Raphson refinements."""
    r = rcp_hw(s)
    e = fma32(-s, r, 1.0)
    r = fma32(e, r, r)
    q = x * r
    d = fma32(-s, q, x)
    q = fma32(d, r, q)
    d = fma32(-s, q, x)
    return fma32(d, r, q)


def fp8_e4m3(x: torch.Tensor) -> torch.Tensor:
    """f32 -> E4M3 RNE (the scaled converts with scale 1; the values never exceed 448)."""
    return x.to(torch.float8_e4m3fn)


def block_scale_mhc(amax: torch.Tensor) -> torch.Tensor:
    """mHC PRE's ue8m0 exponent of a block: amax / 448 rounded up to a power of two (``mhc_pre._pre``)."""
    sc = (amax * _c(0x3B124925)).view(I32)
    return ((sc & 0x7F800000) >> 23) + torch.where((sc & 0x7FFFFF) == 0, 0, 1)


def block_scale_qn(amax: torch.Tensor) -> torch.Tensor:
    """q_norm's ue8m0 exponent of a block (``qkv.qkv_qnorm``: + 0x7fffff, the exponent byte)."""
    sb = (amax * _c(0x3B124925)).view(I32).to(I64) + 0x7FFFFF
    return (sb >> 23) & 0xFF


def pow2(e: torch.Tensor) -> torch.Tensor:
    """2^(e - 127) as f32 for 1 <= e <= 254."""
    return (e.to(I32) << 23).view(F32)


# ---------------------------------------------------------------------------------------------------------------
# the device stages


def mfma_blocks(xq: torch.Tensor, w: torch.Tensor, b0: int, b1: int) -> torch.Tensor:
    """The MFMA results of K blocks [b0, b1) (32 each): [T, N, b1 - b0] f32, each block's dot product of fp8 values
    exact (f64: products of 4-bit significands, <= 41 bits in all) and rounded to f32 once.
    """
    t, n = xq.shape[0], w.shape[0]
    xb = xq.view(t, -1, 32)[:, b0:b1].double().permute(1, 0, 2)
    wb = w.view(n, -1, 32)[:, b0:b1].double().permute(1, 2, 0)
    return torch.bmm(xb, wb).float().permute(1, 2, 0)


def gemm(
    xq: torch.Tensor,
    xs: torch.Tensor,
    w: torch.Tensor,
    ws: torch.Tensor,
    waves: int,
) -> torch.Tensor:
    """``qkv_gemv`` / ``qkv_qb``: fp8 x fp8 with ue8m0 blocks of 32 (activation [T, K/32], weight [N/32, K/32]); a
    wave's chain over its K blocks (fma of the block sum and the scales' product), the waves summed in order.
    -> [T, N] f32."""
    nb = xq.shape[1] // 32
    per = nb // waves
    xsf = e8m0(xs)  # [T, nb]
    wsf = e8m0(ws).repeat_interleave(32, 0)  # [N, nb]
    out = None
    for wv in range(waves):
        m = mfma_blocks(xq, w, wv * per, (wv + 1) * per)
        acc = None
        for j in range(per):
            b = wv * per + j
            c = xsf[:, b, None] * wsf[None, :, b]  # exact: powers of two
            # fma(m, c, acc): m * c is exact, so the f64 sum rounded once is the fused result
            p = m[:, :, j].double() * c.double()
            acc = p.float() if acc is None else (p + acc.double()).float()
        out = acc if out is None else out + acc
    return out


def mhc_pre(x: torch.Tensor, gamma: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``mhc_v3_kernel<S, 2>``'s PRE for rows of 4 identical copies of ``x`` [T, 5120] (bf16 values): the RMS-normed,
    gamma-scaled row as 1 x 32 E4M3 (values, [T, 5120] float) and ue8m0 (int64 [T, 160]).
    """
    t = x.shape[0]
    sq = (x * x).view(t, 10, 64, 8)  # exact (bf16 squared)
    ss = seq_sum(sq)  # a lane's 8 columns
    slices = tree64(ss)  # [T, 10]
    tot = tree64(torch.cat([slices, slices.new_zeros(t, 54)], 1))
    ms = fma32(tot, _c(0x394CCCCD), _c(0x1E3CE508))
    r = rms_rsq_hw(ms)
    y = gamma * (r[:, None] * x)
    yb = bf16_round(y).view(t, 160, 32)
    amax = torch.clamp_min(yb.abs().amax(-1), _c(0x38D1B717))
    e = block_scale_mhc(amax)
    inv = ((254 - e).to(I32) << 23).view(F32)
    q = fp8_e4m3(yb * inv[..., None]).float().view(t, D)
    return q, e


def mhc_sumsq(x: torch.Tensor) -> torch.Tensor:
    """PRE's per-slice sums of squares of all 4 copies (ws + 0x1e000): [T, 10]."""
    t = x.shape[0]
    sq = (x * x).view(t, 10, 64, 8)
    sa = sq[..., 0]
    for h in range(HC):
        for c in range(8):
            if h or c:
                sa = sa + sq[..., c]
    return tree64(sa)


def mhc_partials(x: torch.Tensor, fn: torch.Tensor) -> torch.Tensor:
    """The mix GEMV's partials (``mhc_pre._gemv``): [T, 24, 160], each (s0 + s1) + (s2 + s3) of the 4 copies'
    32-column chains (fma for columns 0..19, then mul + add)."""
    t = x.shape[0]
    xk = x.view(t, 1, 1, 160, 32)
    fk = fn.view(MIX, HC, 160, 32)
    acc = fma32(fk[..., 0], xk[..., 0], 0.0)
    for j in range(1, 20):
        acc = fma32(fk[..., j], xk[..., j], acc)
    for j in range(20, 32):
        if OPTS["gemv_tail_fused"]:
            acc = fma32(fk[..., j], xk[..., j], acc)
        else:
            acc = acc + fk[..., j] * xk[..., j]
    return (acc[:, :, 0] + acc[:, :, 1]) + (acc[:, :, 2] + acc[:, :, 3])


def mhc_finalize(
    part: torch.Tensor, sq10: torch.Tensor, scale: torch.Tensor, base: torch.Tensor
) -> torch.Tensor:
    """``qkv.mhc_finalize_row`` (idx 0): hcA [T, 24] f32."""
    lo = seq_sum(part[:, :, :80])
    hi = seq_sum(part[:, :, 80:])
    tot = seq_sum(sq10)
    ms = fma32(tot, _c(0x384CCCCD), _c(0x1E3CE508))
    r = rms_rsq_hw(ms)
    mixv = (lo + hi) * r[:, None]

    def lin(a, s, b):
        return fma32(a, s, b) if OPTS["mix_fused"] else a * s + b

    t = part.shape[0]
    out = torch.empty(t, MIX, dtype=F32)
    y = lin(mixv[:, :8], torch.cat([scale[0].expand(4), scale[1].expand(4)]), base[:8])
    sig = 1.0 / (1.0 + exp_ocml(y, True))
    out[:, :4] = sig[:, :4] + _c(0x358637BD)
    out[:, 4:8] = sig[:, 4:] + sig[:, 4:]
    eps = _c(0x358637BD)
    y = lin(scale[2], mixv[:, 8:], base[8:]).view(t, 4, 4)
    ex = exp_ocml(y - y.amax(-1, keepdim=True), False)

    def quad(v):
        return ((v[..., 0] + v[..., 1]) + (v[..., 2] + v[..., 3]))[..., None]

    def col(v):
        return ((v[:, 0] + v[:, 2]) + (v[:, 1] + v[:, 3]))[:, None]

    p = div_nr(ex, quad(ex)) + eps
    v = div_nr(p, col(p) + eps)
    for _ in range(19):
        q = div_nr(v, quad(v) + eps)
        v = div_nr(q, col(q) + eps)
    out[:, 8:] = v.reshape(t, 16)
    return out


def kv_extract(kv: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    """``toktab::kv_extract_kernel``: [T, 512] bf16 values (the GEMM's) RMS-normed -> bf16 bits (int64)."""
    t = kv.shape[0]
    a = kv[:, :256] * kv[:, :256] + kv[:, 256:] * kv[:, 256:]
    v = a.view(t, 4, 64)
    for k in (32, 16, 8, 4, 2, 1):  # xor butterflies: lane 0's tree
        v = v[..., :k] + v[..., k : 2 * k]
    w = v[..., 0]
    tot = (((0.0 + w[:, 0]) + w[:, 1]) + w[:, 2]) + w[:, 3]
    ms = fma32(tot, _c(0x3B000000), _c(0x1E3CE508))
    tiny = ms < _c(0x0F800000)
    s = torch.sqrt(torch.where(tiny, ms * _c(0x4F800000), ms))
    s = torch.where(tiny, s * _c(0x37800000), s)
    rn = 1.0 / s
    return bf16_bits(gamma * (rn[:, None] * kv))


def q_norm(qa: torch.Tensor, gamma: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``qkv.qkv_qnorm``: q_a [T, 1280] (bf16 values) -> its E4M3 values [T, 1280] and ue8m0 [T, 40]."""
    t = qa.shape[0]
    p = (qa * qa).view(t, 160, 4, 2)
    sq = seq_sum(p[..., 0] + p[..., 1])  # a lane's 4 packs
    lanes = torch.cat([sq, sq.new_zeros(t, 96)], 1).view(t, 4, 64)
    w = tree64(lanes)
    tot = (((0.0 + w[:, 0]) + w[:, 1]) + w[:, 2]) + w[:, 3]
    ms = tot / 1280.0 + _c(0x1E3CE508)
    rstd = 1.0 / torch.sqrt(ms)
    y = bf16_round(gamma * (rstd[:, None] * qa)).view(t, 40, 32)
    amax = torch.clamp_min(y.abs().amax(-1), _c(0x38D1B717))
    e = block_scale_qn(amax)
    inv = ((254 - e).to(I32) << 23).view(F32)
    q = fp8_e4m3(y * inv[..., None]).float().view(t, Q_LORA)
    return q, e


def rope0(x: torch.Tensor) -> torch.Tensor:
    """GPT-J RoPE at position 0 (cos 1, sin 0) of each head's dims 448..511: fma(1, x0, -(0 x1)),
    fma(1, x1, 0 x0) (signed zeros)."""
    t = x.shape[0]
    h = x.view(t, HEADS_LOCAL, HEAD_DIM).clone()
    ev, od = h[:, :, 448::2], h[:, :, 449::2]
    one, zero = torch.ones((), dtype=F32), torch.zeros((), dtype=F32)
    h[:, :, 448::2] = fma32(one, ev, -(zero * od))
    h[:, :, 449::2] = fma32(one, od, zero * ev)
    return h.view(t, -1)


# ---------------------------------------------------------------------------------------------------------------
# a scope's tables


def scope_weights(ckpt, scope: str, rank: int | None = None) -> dict:
    """The checkpoint's tensors of ``scope``'s first block the tables need (``weights.ckpt`` names); ``rank``: also
    that rank's q_b rows (its 8 heads)."""
    from ..weights.ckpt import BLK, WQB_ROWS, get_index

    idx = get_index(ckpt)
    p = f"{scope}."
    a = p + "attn."

    def u8(t):
        return t.contiguous().view(torch.uint8)

    w = {
        "fn": idx.read(p + "hc_attn_fn").float().view(MIX, HC, D),
        "base": idx.read(p + "hc_attn_base").float(),
        "scale": idx.read(p + "hc_attn_scale").float(),
        "attn_gamma": idx.read(p + "attn_norm.weight").float(),
        "qa_w": idx.read(a + "wq_a.weight").float(),
        "qa_s": u8(idx.read(a + "wq_a.scale")),
        "q_norm": idx.read(a + "q_norm.weight").float(),
        "kv_w": idx.read(a + "wkv.weight").float(),
        "kv_s": u8(idx.read(a + "wkv.scale")),
        "kv_norm": idx.read(a + "kv_norm.weight").float(),
    }
    if rank is not None:
        w["qb_w"] = idx.read_rows(
            a + "wq_b.weight", WQB_ROWS * rank, WQB_ROWS * (rank + 1)
        ).float()
        w["qb_s"] = u8(
            idx.read_rows(
                a + "wq_b.scale", WQB_ROWS // BLK * rank, WQB_ROWS // BLK * (rank + 1)
            )
        )
    return w


def shared_rows(w: dict, x: torch.Tensor) -> dict:
    """The rank-independent part of the rows of embeddings ``x`` [T, 5120] (f32 of bf16): ``latent`` (q_b's input,
    E4M3 bytes [T, 1280]), ``latent_e`` (its ue8m0 [T, 40]), ``kv`` (bf16 [T, 512]), ``hc`` (f32 [T, 24]).
    """
    xq, xs = mhc_pre(x, w["attn_gamma"])
    hc = mhc_finalize(mhc_partials(x, w["fn"]), mhc_sumsq(x), w["scale"], w["base"])
    qkv = gemm(
        xq, xs, torch.cat([w["qa_w"], w["kv_w"]]), torch.cat([w["qa_s"], w["kv_s"]]), 8
    )
    qkv = bf16_round(qkv)
    kv = as_bf16(kv_extract(qkv[:, Q_LORA:], w["kv_norm"]))
    lq, le = q_norm(qkv[:, :Q_LORA].contiguous(), w["q_norm"])
    latent = lq.to(torch.float8_e4m3fn).view(torch.uint8)
    return {"latent": latent, "latent_e": le.to(torch.uint8), "kv": kv, "hc": hc}


def q_rows(w: dict, latent: torch.Tensor, latent_e: torch.Tensor) -> torch.Tensor:
    """A rank's q rows (bf16 [T, 4096]) from the latent (``shared_rows``) and its q_b rows (``w["qb_w"]``)."""
    lq = latent.view(torch.float8_e4m3fn).float()
    q = rope0(bf16_round(gemm(lq, latent_e.to(I64), w["qb_w"], w["qb_s"], 4)))
    return as_bf16(bf16_bits(q))


SHARED = {"latent": ((Q_LORA,), torch.uint8), "latent_e": ((Q_LORA // 32,), torch.uint8),
          "kv": ((KV_DIM,), torch.bfloat16), "hc": ((MIX,), F32)}  # fmt: skip


def build_shared(
    w: dict, embed: torch.Tensor, tokens: torch.Tensor, chunk: int = 1024, log=None
) -> dict:
    """``shared_rows`` of ``tokens``, chunk by chunk."""
    n = tokens.numel()
    out = {k: torch.empty((n, *shape), dtype=dt) for k, (shape, dt) in SHARED.items()}
    t0 = time.time()
    for a in range(0, n, chunk):
        b = min(n, a + chunk)
        for k, v in shared_rows(w, embed[tokens[a:b]].float()).items():
            out[k][a:b] = v
        if log and (a // chunk) % 32 == 0:
            log(f"toktab: shared rows {b}/{n} ({time.time() - t0:.0f}s)")
    return out


def build_q(w: dict, shared: dict, chunk: int = 4096) -> torch.Tensor:
    n = shared["latent"].shape[0]
    tq = torch.empty(n, HEADS_LOCAL * HEAD_DIM, dtype=torch.bfloat16)
    for a in range(0, n, chunk):
        tq[a : a + chunk] = q_rows(
            w, shared["latent"][a : a + chunk], shared["latent_e"][a : a + chunk]
        )
    return tq


def build_rank_tables(
    ckpt,
    rank: int,
    tokens: torch.Tensor | None = None,
    group=None,
    src: int = 0,
    chunk: int = 1024,
    log=None,
    hw=None,
) -> list[torch.Tensor]:
    """Rank ``rank``'s 6 tables in TileRT's order: q, kv, hc of ``layers.0``, then of ``mtp.0``. ``group`` (a
    ``torch.distributed`` group, CPU tensors): its rank ``src`` builds the rank-independent rows and broadcasts them,
    every rank its own q. ``hw``: the hardware approximations' provider (default ``CpuHw``).
    """
    import torch.distributed as dist

    from ..weights.ckpt import get_index

    tokens = torch.arange(V) if tokens is None else tokens
    builder = group is None or dist.get_rank(group) == src
    embed = get_index(ckpt).read("embed.weight") if builder else None
    out = []
    for scope in SCOPES:
        if builder:
            with hardware(hw or CpuHw()):
                shared = build_shared(
                    scope_weights(ckpt, scope), embed, tokens, chunk, log
                )
        else:
            shared = {
                k: torch.empty((tokens.numel(), *sh), dtype=dt)
                for k, (sh, dt) in SHARED.items()
            }
        if group is not None:
            for k in SHARED:  # (gloo: byte views)
                dist.broadcast(
                    shared[k].view(torch.uint8),
                    dist.get_global_rank(group, src),
                    group=group,
                )
        out += [
            build_q(scope_weights(ckpt, scope, rank), shared),
            shared["kv"],
            shared["hc"],
        ]
    return out


# ---------------------------------------------------------------------------------------------------------------
# the byte check against a reference (TileRT's g.toktab[r], torch.save'd)


def compare(
    got: list[torch.Tensor], ref: list[torch.Tensor], tokens: torch.Tensor
) -> list[dict]:
    """Per table: rows, bytes, differing bytes, differing rows (``ref`` the full tables, ``got`` rows ``tokens``)."""
    out = []
    for g, r in zip(got, ref):
        r = r[tokens]
        gb = g.contiguous().view(torch.uint8).view(g.shape[0], -1)
        rb = r.contiguous().view(torch.uint8).view(r.shape[0], -1)
        diff = gb != rb
        out.append({"rows": g.shape[0], "bytes": gb.numel(), "diff_bytes": int(diff.sum()),
                    "diff_rows": int(diff.any(1).sum()), "dtype": str(g.dtype).rsplit(".", 1)[-1]})  # fmt: skip
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description="build a rank's DSpark token tables, compare to a reference"
    )
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--ref", help="rank<r>.pt: TileRT's g.toktab[r] (6 tensors)")
    ap.add_argument("--tokens", default="all", help="all | a:b (a token range)")
    ap.add_argument("--scope", choices=("both",) + SCOPES, default="both")
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument(
        "--out", help="save the built tables here (torch.save, TileRT's order)"
    )
    ap.add_argument("--hw", choices=("cpu", "device"), default="cpu",
                    help="the hardware approximations: modelled on the CPU, or evaluated by GPU --device")  # fmt: skip
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--threads", type=int, help="torch CPU threads (default: torch's)")
    a = ap.parse_args(argv)
    if a.threads:
        torch.set_num_threads(a.threads)
    from ..weights.ckpt import get_index

    tokens = torch.arange(V)
    if a.tokens != "all":
        lo, hi = (int(x) for x in a.tokens.split(":"))
        tokens = torch.arange(lo, hi)
    embed = get_index(a.ckpt).read("embed.weight")
    ref = torch.load(a.ref, map_location="cpu", mmap=True) if a.ref else None
    provider = DeviceHw(f"cuda:{a.device}") if a.hw == "device" else CpuHw()
    got = []
    with hardware(provider):
        for k, scope in enumerate(SCOPES):
            if a.scope not in ("both", scope):
                continue
            t0 = time.time()
            shared = build_shared(
                scope_weights(a.ckpt, scope), embed, tokens, a.chunk, log=print
            )
            t1 = time.time()
            tabs = [
                build_q(scope_weights(a.ckpt, scope, a.rank), shared),
                shared["kv"],
                shared["hc"],
            ]
            print(f"{scope}: {tokens.numel()} rows, shared {t1 - t0:.1f}s, q {time.time() - t1:.1f}s (hw {a.hw})",
                  flush=True)  # fmt: skip
            got += tabs
            if ref is not None:
                for name, res in zip(
                    ("q", "kv", "hc"), compare(tabs, ref[3 * k : 3 * k + 3], tokens)
                ):
                    print(f"{scope}.{name}: {res}", flush=True)
    if a.out:
        torch.save(got, a.out)


if __name__ == "__main__":
    main()
