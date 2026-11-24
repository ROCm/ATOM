# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The torch contract of the Gemma add + RMSNorm kernels at hidden 6144 (root README, rule 3).

Per row of H = 6144 columns every kernel computes:

1. the input x, bf16:
   - plain: the caller's input;
   - all-reduce (``allreduce_add_rmsnorm``): chunk c = rows ``[c q, (c + 1) q)``, q = ceil(rows / 4), summed in fp32
     in the rank order c, c + 1, c + 2, c + 3 (mod 4), operands and partial sums with denormals flushed to a zero of
     their sign, one RNE rounding (AITER's ``cross_device_reduce_2stage``, built with -fgpu-flush-denormals-to-zero),
     ``allreduce_rows_reference``; the MoE tail first sums each rank's slots, ``slots4_allreduce_reference``;
2. ``s = fp32(x) + fp32(residual)`` (one fp32 rounding); the residual out is ``bf16(s)``;
3. ``y = (s * rstd) * (w + 1)``, ``rstd = 1 / sqrt(sum(s^2) / 6144 + eps)`` (Gemma's weight offset; the norm reads
   the fp32 s, not the rounded residual out). The kernels sum the squares in fp32 in the Triton kernels' order; the
   reference computes in fp64 but keeps the fp32 range: a sum above the fp32 maximum is +inf, so such a row gets
   rstd 0 and y 0 (NaN where s is inf). ``out = bf16(y)``;
4. MXFP8 out, per 32-column group of y: ``a = max(max |y|,
   2^-126)`` (IEEE maxNum: NaNs are skipped, an all-NaN group gets the floor), the E8M0 scale byte
   ``sb = clamp(ceil(log2(fp32(a) / 448)) + 127, 0, 254)`` (254 for an infinite ``a``; the scale is ``2^(sb - 127)``,
   the division rounded to fp32), the code ``e4m3(y * 2^(127 - sb))``, RNE, saturating at 448 (NaN stays NaN).

Layouts: the bf16 outputs row-major ``[m, 6144]``; codes row-major ``[m, 6144]`` or in the decode GEMM's (16, 16)
tiles (``untile_codes``; rows padded to the tile, padding rows untouched); scales in AITER's shuffled layout
(``unshuffle_scales``; rows padded to 32, padding untouched).

Tolerances (``compare_out`` / ``compare_mxfp8``):

- x of the all-reduce paths and the residual out: bit for bit.
- ``out``: within one bf16 ulp of the fp64 y (``OUT_ULPS``) and equal to ``bf16(y)`` at all but ``OUT_MISMATCH`` of
  the elements: the fp32 sum of the squares moves y by ~1e-7 relative, which flips a rounding now and then.
- scales: within 1 of the reference byte, equal at all but ``SCALE_MISMATCH`` of the groups (the kernels' fp32
  ``v_log_f32`` rounds to the integer just above ``448 * 2^k`` for 1-9 ulp of ``a``, and the sum order moves ``a``).
- codes: the fp64 y requantized at the kernel's own scale bytes; equal at all but ``CODE_MISMATCH`` of the elements,
  else the neighbouring e4m3 magnitude.
- cosine: ``out`` vs y >= ``OUT_COS``; the dequantized MXFP8 output vs y >= ``MXFP8_COS`` (e4m3 keeps 3 mantissa
  bits).

On the tests' inputs ``out`` is at most 0.5006 ulp from y, != bf16(y) at <= 4.1e-5 of the elements, cosine >=
0.9999983; scale bytes differ (by 1) at <= 2.0e-5 of the groups, codes at <= 9.6e-6; the dequantized MXFP8 cosine is
>= 0.99957.
"""

import torch

H = 6144
GROUP = 32  # MXFP8 block along the row
FP8_MAX = 448.0
TINY = 2.0**-126  # fp32 smallest normal: the amax floor and the flush threshold
FP32_MAX = torch.finfo(torch.float32).max
WORLD = 4

OUT_ULPS = 1
OUT_MISMATCH = 1e-3
OUT_COS = 0.99999  # cosine of out vs y over the finite elements
SCALE_MISMATCH = 1e-3
CODE_MISMATCH = 1e-3
MXFP8_COS = 0.999  # cosine of the dequantized MXFP8 output vs y


def _ftz(v: torch.Tensor) -> torch.Tensor:
    """fp32 with denormals flushed to a zero of their sign."""
    return torch.where(v.abs() < TINY, v * 0.0, v)


def allreduce_reference(xs: list[torch.Tensor]) -> torch.Tensor:
    """``allreduce_rows_reference`` of 4 partials whose rows are a multiple of 4 (AITER's 2-stage all-reduce)."""
    assert len(xs) == WORLD and xs[0].shape[0] % WORLD == 0
    return allreduce_rows_reference(xs)


def allreduce_rows_reference(xs: list[torch.Tensor]) -> torch.Tensor:
    """The 4 ranks' ``[rows, H]`` bf16 partials -> their sum, bf16, at any row count (module docstring, step 1)."""
    rows = xs[0].shape[0]
    q = (rows + WORLD - 1) // WORLD
    out = torch.empty_like(xs[0])
    for c in range(WORLD):
        blk = slice(min(c * q, rows), min((c + 1) * q, rows))
        acc = _ftz(xs[c][blk].float())
        for t in range(1, WORLD):
            acc = _ftz(acc + _ftz(xs[(c + t) % WORLD][blk].float()))
        out[blk] = acc.bfloat16()
    return out


def slots4_local_reference(slots: torch.Tensor, shared: torch.Tensor) -> torch.Tensor:
    """A rank's ``[rows, 4, H]`` weighted MoE slots and ``[rows, H]`` shared output -> its bf16 partial
    ``bf16((((s0 + s1) + s2) + s3) + shared)`` (fp32 adds)."""
    acc = slots[:, 0].float()
    for k in range(1, slots.shape[1]):
        acc = acc + slots[:, k].float()
    return (acc + shared.float()).bfloat16()


def slots4_allreduce_reference(
    slots: list[torch.Tensor], shared: list[torch.Tensor]
) -> torch.Tensor:
    """The 4 ranks' slots and shared outputs -> the all-reduced bf16 rows of ``slots4_ar_add_rmsnorm_mxfp8``: the
    ranks' ``slots4_local_reference`` partials through ``allreduce_rows_reference`` (1-8191 rows). Its row-major codes
    use the tiled norm's arithmetic; only their addresses change."""
    return allreduce_rows_reference([slots4_local_reference(s, h) for s, h in zip(slots, shared)])


def add_rmsnorm_reference(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """(y in fp64, the residual out bf16) of ``x + residual`` (module docstring, steps 2 and 3); ``bf16(y)`` is the
    bf16 output."""
    s = x.float() + residual.float()
    s64 = s.double()
    eps32 = torch.tensor(eps, dtype=torch.float32).item()
    ss = (s64 * s64).sum(-1, keepdim=True)
    ss = torch.where(ss > FP32_MAX, torch.full_like(ss, float("inf")), ss)
    y = s64 * torch.rsqrt(ss / H + eps32) * (weight.double() + 1.0)
    return y, s.bfloat16()


def _ceil_log2(d: torch.Tensor) -> torch.Tensor:
    """ceil(log2(d)) of positive fp32 values, exact (frexp)."""
    m, e = torch.frexp(d.double())
    return e.long() - (m == 0.5).long()


def mxfp8_scale_reference(y64: torch.Tensor, bf16_input: bool = False) -> torch.Tensor:
    """``[m, H / 32]`` E8M0 scale bytes (uint8) of the groups of ``y64``."""
    y = y64.bfloat16().double() if bf16_input else y64
    m = y.shape[0]
    mag = torch.nan_to_num(y.abs(), nan=0.0, posinf=float("inf"))  # maxNum skips NaN
    a = mag.view(m, H // GROUP, GROUP).amax(-1).float().clamp_min(TINY)
    sb = (_ceil_log2(a / FP8_MAX) + 127).clamp(0, 254)
    sb = torch.where(torch.isinf(a), torch.full_like(sb, 254), sb)
    return sb.to(torch.uint8)


def mxfp8_codes(
    y64: torch.Tensor, scale_bytes: torch.Tensor, bf16_input: bool = False
) -> torch.Tensor:
    """``[m, H]`` e4m3 codes (uint8) of ``y64`` at the given scale bytes (RNE, saturating at 448)."""
    y = y64.bfloat16().double() if bf16_input else y64
    m = y.shape[0]
    rescale = torch.exp2(127.0 - scale_bytes.double())[:, :, None]
    q = (y.view(m, H // GROUP, GROUP) * rescale).float().clamp(-FP8_MAX, FP8_MAX)
    return q.to(torch.float8_e4m3fn).view(torch.uint8).reshape(m, H)


def dequantize(codes: torch.Tensor, scale_bytes: torch.Tensor) -> torch.Tensor:
    """fp64 ``[m, H]`` of e4m3 codes and E8M0 scale bytes (both uint8)."""
    m = codes.shape[0]
    v = codes.view(torch.float8_e4m3fn).double().view(m, H // GROUP, GROUP)
    return (v * torch.exp2(scale_bytes.double() - 127.0)[:, :, None]).reshape(m, H)


def untile_codes(q: torch.Tensor, m: int) -> torch.Tensor:
    """The row-major ``[m, H]`` codes of a (16, 16)-tiled code buffer: byte ``(r // 16) * 16 H + (c // 64) * 1024 +
    (c % 64 // 16) * 256 + (r % 16) * 16 + c % 16`` holds (row r, column c)."""
    q = q.reshape(-1).view(torch.uint8)
    r = torch.arange(m, device=q.device)[:, None]
    c = torch.arange(H, device=q.device)[None, :]
    off = (
        r // 16 * (16 * H)
        + c // 64 * 1024
        + c % 64 // 16 * 256
        + r % 16 * 16
        + c % 16
    )
    return q[off]


def scale_offsets(m: int, device) -> torch.Tensor:
    """``[m, H / 32]`` byte offsets of (row, group) in AITER's shuffled scale layout (``[ceil32(rows), H / 32]``)."""
    k1 = H // 256
    r = torch.arange(m, device=device)[:, None]
    g = torch.arange(H // GROUP, device=device)[None, :]
    off = ((((r // 32) * k1 + g // 8) * 4 + g % 4) * 16 + r % 16) * 2 + g % 8 // 4
    return off * 2 + r % 32 // 16


def unshuffle_scales(s: torch.Tensor, m: int) -> torch.Tensor:
    """The row-major ``[m, H / 32]`` scale bytes of an AITER-layout buffer."""
    s = s.reshape(-1).view(torch.uint8)
    return s[scale_offsets(m, s.device)]


def _bf16_ulp(v: torch.Tensor) -> torch.Tensor:
    """The bf16 spacing at |v| (2^-133 at zero and among the subnormals)."""
    _, e = torch.frexp(v.double())
    e = torch.where(v == 0, torch.full_like(e, -125), e)
    return torch.exp2((e.double() - 8.0).clamp_min(-133.0))


def compare_out(out: torch.Tensor, y64: torch.Tensor) -> dict:
    """Statistics of a bf16 ``out`` against the fp64 y over the finite elements: ``bad_ulp`` (elements more than
    ``OUT_ULPS`` bf16 ulp away), ``mismatch`` (fraction != bf16(y)), ``nonfinite_ok`` (non-finite exactly where y is,
    NaN for NaN, inf for inf), ``cos``, ``rel_l2``.

    Computed on ``out``'s device: on the CPU 8191 x 6144 rows take seconds, and torch's CPU fp64 -> bf16 rounds twice
    (via fp32), so ``mismatch`` there can differ by ~2^-16."""
    o = out.double()
    y = y64.to(o.device)
    fin = torch.isfinite(y)
    nonfinite_ok = bool(
        torch.equal(torch.isnan(o), torch.isnan(y))
        and torch.equal(o[torch.isinf(y)], y[torch.isinf(y)])
        and bool(torch.isfinite(o[fin]).all())
    )
    of, yf = o[fin], y[fin]
    ulp = torch.maximum(_bf16_ulp(yf), _bf16_ulp(of))
    err = (of - yf).abs()
    yb = yf.bfloat16().double()
    denom = yf.norm().item()
    return {
        "bad_ulp": int((err > OUT_ULPS * ulp).sum()),
        "max_ulps": float((err / ulp).max()) if err.numel() else 0.0,
        "mismatch": float((of != yb).double().mean()) if of.numel() else 0.0,
        "nonfinite_ok": nonfinite_ok,
        "cos": float(torch.nn.functional.cosine_similarity(of, yf, dim=0, eps=1e-300))
        if denom > 0
        else 1.0,
        "rel_l2": float((of - yf).norm() / denom) if denom > 0 else 0.0,
    }


def compare_mxfp8(
    codes: torch.Tensor,
    scale_bytes: torch.Tensor,
    y64: torch.Tensor,
    bf16_input: bool = False,
) -> dict:
    """Statistics of row-major ``codes`` / ``scale_bytes`` (uint8) against the reference of ``y64``, on ``codes``'
    device: ``scale_max_diff``, ``scale_mismatch`` (fraction of groups), ``code_mismatch`` (fraction of codes != the
    reference y requantized at the kernel's scales), ``code_far`` (codes neither equal nor a neighbouring magnitude of
    the same sign), ``cos`` of the dequantized output vs y."""
    scale_bytes = scale_bytes.to(codes.device)
    y64 = y64.to(codes.device)
    ref_sb = mxfp8_scale_reference(y64, bf16_input)
    sd = (scale_bytes.long() - ref_sb.long()).abs()
    ref_codes = mxfp8_codes(y64, scale_bytes, bf16_input)
    nan = lambda c: (c & 0x7F) == 0x7F  # noqa: E731  (e4m3fn NaN, either sign)
    diff = (codes != ref_codes) & ~(nan(codes) & nan(ref_codes))
    mag = lambda c: (c & 0x7F).long()  # noqa: E731
    sign = lambda c: (c & 0x80).long()  # noqa: E731
    near = ((mag(codes) - mag(ref_codes)).abs() <= 1) & (
        (sign(codes) == sign(ref_codes)) | (mag(codes) == 0) | (mag(ref_codes) == 0)
    )
    deq = dequantize(codes, scale_bytes).flatten()
    y = (y64.bfloat16().double() if bf16_input else y64).flatten()
    fin = torch.isfinite(y)
    deq, y = deq[fin], y[fin]
    denom = y.norm().item()
    return {
        "scale_max_diff": int(sd.max()),
        "scale_mismatch": float((sd != 0).double().mean()),
        "code_mismatch": float(diff.double().mean()),
        "code_far": int((diff & ~near).sum()),
        "cos": float(torch.nn.functional.cosine_similarity(deq, y, dim=0, eps=1e-300))
        if denom > 0
        else 1.0,
    }
