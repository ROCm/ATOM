# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The contract of ``qkv_norm_rope_insert`` and ``kv_norm_rope_insert_multilayer``
(root README, rule 3): torch only.

Per 128-dim head ``x`` (bf16), both kernels compute

1. Gemma RMS norm: ``var = sum(x^2) / 128`` in fp32, ``y = RN_bf16(x * rstd * (1 + w))``
   with ``1 + w`` in fp32. ``qkv_norm_rope_insert`` takes ``rstd = 1 / sqrt(var + eps)``
   and the products in fp64 (``eps`` an fp64 argument; correctly rounded to bf16), as
   Inductor compiles vLLM's Triton norm. ``kv_norm_rope_insert_multilayer`` stays in
   fp32 (``eps`` fp32, the hardware ``v_sqrt_f32``, an IEEE divide, ``(x * rstd) * (1 +
   w)``), as the eager Triton norm runs.
2. Partial NeoX RoPE over dims 0..63 (halves of 32) on the bf16 ``y``: ``o1 = y1 cos -
   y2 sin``, ``o2 = y2 cos + y1 sin`` in fp32 (the products of two bf16 are exact),
   rounded to bf16; dims 64..127 pass through. ``cos_sin_cache [rows, 64]`` is cos |
   sin per position; a negative position wraps (``pos + rows``).
3. Insert: e4m3 of ``value / scale`` (IEEE fp32 divide, RNE, saturated at +-448:
   ``v_cvt_scalef32_pk_fp8_f32`` under ``MODE.fp16_ovfl``) into the page slot
   ``[slot // page, head, slot % page]``: 128 K bytes then 128 V bytes. V is not
   normed or rotated. A slot < 0, or a token at or past the slots given, is skipped.

``qkv_norm_rope_insert`` also returns q (normed and rotated, bf16) and writes K = the
normed, rotated k heads.

Tolerances (``check_bf16``, ``check_cache``): the kernels' sum of squares runs in a DPP
tree order (the Triton kernels'), not torch's, and the fp32 path's hardware sqrt is up
to 2 ulp low, so a normed bf16 value may sit one ulp from the reference. RoPE rotates
that ulp of a large input into a smaller output (2..4 -> 1..2: two ulps there) or
cancels it near zero. Every output is within one step (bf16 ulp / e4m3 code) of the
reference or within ``ROW_ATOL`` (2^-6) of its 128-value row's largest reference
magnitude; at least ``EXACT_MIN`` are bit-equal; every cache byte the kernel must not
write is unchanged. Measured on MI355X: the tests' q and cache bytes >= 99.9999%
bit-equal; over 100 inputs of 384 tokens x 64 heads, 168 of 314.6M q values differ, by at most 2 ulps (8.7e-3 of the row maximum); cache
differences are e4m3 subnormals (|v| < 2^-6) two steps apart where RoPE cancelled.
"""

import torch

D = 128
ROT = 64
TOK_B = 2 * D
FP8_MAX = 448.0
EXACT_MIN = 0.999
ROW_ATOL = 2.0**-6  # the RoPE-rotated ulp of a normed input, relative to the row maximum


def _rn_bf16_from_f64(d: torch.Tensor) -> torch.Tensor:
    """fp64 -> bf16 correctly rounded (round to odd fp32, then RNE bf16; torch's own
    cast goes through an RNE fp32 and can round twice)."""
    f = d.float()
    inexact = f.double() != d
    # truncate toward zero, then set the last bit: round to odd
    over = inexact & (f.double().abs() > d.abs())
    bits = f.view(torch.int32)
    bits = torch.where(over, bits - 1, bits)
    bits = torch.where(inexact, bits | 1, bits)
    return bits.view(torch.float32).bfloat16()


def gemma_rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float, fp64_rstd: bool) -> torch.Tensor:
    """``x [..., 128]`` bf16, ``w [128]`` bf16 -> bf16 (step 1 above)."""
    x32 = x.float()
    var = (x32 * x32).sum(-1, keepdim=True) / float(D)
    w1 = 1.0 + w.float()
    if fp64_rstd:
        rstd = 1.0 / torch.sqrt(var.double() + eps)
        return _rn_bf16_from_f64((rstd * x.double()) * w1.double())
    rstd = 1.0 / torch.sqrt(var + torch.tensor(eps, dtype=torch.float32))
    return ((x32 * rstd) * w1).bfloat16()


def rope_neox_partial(y: torch.Tensor, positions: torch.Tensor, cos_sin_cache: torch.Tensor) -> torch.Tensor:
    """``y [n, heads, 128]`` bf16 -> bf16, dims 0..63 rotated (step 2 above)."""
    rows = cos_sin_cache.shape[0]
    pos = torch.where(positions < 0, positions + rows, positions).long()
    cos = cos_sin_cache[pos, : ROT // 2].float()[:, None, :]
    sin = cos_sin_cache[pos, ROT // 2 : ROT].float()[:, None, :]
    y1, y2 = y[..., : ROT // 2].float(), y[..., ROT // 2 : ROT].float()
    out = y.clone()
    out[..., : ROT // 2] = (y1 * cos - y2 * sin).bfloat16()
    out[..., ROT // 2 : ROT] = (y2 * cos + y1 * sin).bfloat16()
    return out


def e4m3_codes(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """e4m3 bytes of ``x / scale`` (step 3 above)."""
    v = (x.float() / scale.float()).clamp(-FP8_MAX, FP8_MAX)
    return v.to(torch.float8_e4m3fn).view(torch.uint8)


def insert_reference(cache, k, v, slots, k_scale, v_scale):
    """Write ``k``, ``v [n, heads, 128]`` into a copy of ``cache [pages, heads, page,
    256]`` at ``slots [n]`` (< 0 skipped). Returns (the copy, the mask of the bytes
    written)."""
    out = cache.clone()
    mask = torch.zeros_like(cache, dtype=torch.bool)
    page = cache.shape[2]
    keep = slots >= 0
    s = slots[keep].long()
    out[s // page, :, s % page, :D] = e4m3_codes(k[keep], k_scale)
    out[s // page, :, s % page, D:] = e4m3_codes(v[keep], v_scale)
    mask[s // page, :, s % page] = True
    return out, mask


def qkv_norm_rope_insert_reference(
    qkv, positions, q_norm_weight, k_norm_weight, cos_sin_cache, eps, num_heads,
    num_kv_heads, kv_cache=None, slot_mapping=None, k_scale=None, v_scale=None,
):  # fmt: skip
    """-> (q_out ``[n, num_heads * 128]`` bf16, (the updated cache copy, the mask of
    the bytes written) or None)."""
    n = qkv.shape[0]
    q, k, v = qkv.view(n, -1, D).split([num_heads, num_kv_heads, num_kv_heads], dim=1)
    q = rope_neox_partial(gemma_rms_norm(q, q_norm_weight, eps, True), positions, cos_sin_cache)
    if kv_cache is None:
        return q.reshape(n, num_heads * D), None
    k = rope_neox_partial(gemma_rms_norm(k, k_norm_weight, eps, True), positions, cos_sin_cache)
    m = min(n, slot_mapping.shape[0])
    cache = insert_reference(kv_cache, k[:m], v[:m], slot_mapping[:m], k_scale, v_scale)
    return q.reshape(n, num_heads * D), cache


def kv_norm_rope_insert_multilayer_reference(
    kv, positions, k_norm_weight, cos_sin_cache, kv_caches, slot_mappings, k_scales,
    v_scales, eps, n_insert=None,
):  # fmt: skip
    """``kv [n, layers * 2 * kv_heads * 128]`` (per layer: K heads then V heads) ->
    per layer (the updated copy of its cache, the mask of the bytes written); rows >=
    ``n_insert`` are not inserted."""
    n = kv.shape[0]
    nl = len(kv_caches)
    nkv = kv_caches[0].shape[1]
    n_insert = n if n_insert is None else n_insert
    x = kv.view(n, nl, 2, nkv, D)
    out = []
    for lay in range(nl):
        k = rope_neox_partial(
            gemma_rms_norm(x[:, lay, 0], k_norm_weight[lay], eps, False), positions, cos_sin_cache
        )
        m = n_insert
        out.append(
            insert_reference(
                kv_caches[lay], k[:m], x[:m, lay, 1], slot_mappings[lay][:m], k_scales[lay], v_scales[lay]
            )
        )
    return out


def _near(g_bits, r_bits, g_val, r_val, sign_bit, mag_mask):
    """Rows of 128: within one step (same sign, adjacent magnitude bits; both
    formats are sign-magnitude) or within ROW_ATOL of the row's largest reference
    magnitude."""
    step = ((g_bits ^ r_bits) & sign_bit == 0) & ((g_bits & mag_mask) - (r_bits & mag_mask)).abs().le(1)
    row_max = r_val.abs().amax(-1, keepdim=True)
    return step | ((g_val - r_val).abs() <= row_max * ROW_ATOL)


def check_bf16(got: torch.Tensor, ref: torch.Tensor, name: str) -> float:
    """bf16 heads: every value near the reference (``_near``), at least EXACT_MIN
    bit-equal; returns the bit-equal fraction."""
    g, r = got.reshape(-1, D), ref.reshape(-1, D)
    gb, rb = g.view(torch.int16).int(), r.view(torch.int16).int()
    near = _near(gb, rb, g.float(), r.float(), 0x8000, 0x7FFF)
    frac = (gb == rb).float().mean().item()
    assert bool(near.all()), (name, int((~near).sum()))
    assert frac >= EXACT_MIN, (name, frac)
    return frac


def check_cache(got, ref, before, written, name: str) -> float:
    """e4m3 cache pages: near the reference (``_near``, per 128-byte K or V row) where
    the kernel writes (``written``), at least EXACT_MIN of those bytes bit-equal, and
    every other byte unchanged; returns the bit-equal fraction of the written bytes."""
    assert torch.equal(got[~written], before[~written]), (name, "a byte the kernel must not write changed")
    rows = written.reshape(-1, D).all(-1)
    g, r = got.reshape(-1, D)[rows], ref.reshape(-1, D)[rows]
    gv = g.view(torch.float8_e4m3fn).float()
    rv = r.view(torch.float8_e4m3fn).float()
    near = _near(g.int(), r.int(), gv, rv, 0x80, 0x7F)
    assert bool(near.all()), (name, int((~near).sum()))
    frac = (g == r).float().mean().item() if g.numel() else 1.0
    assert frac >= EXACT_MIN, (name, frac)
    return frac
