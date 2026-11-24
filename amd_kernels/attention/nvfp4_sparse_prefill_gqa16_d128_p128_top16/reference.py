# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Torch reference of the NVFP4 sparse prefill attention: the op's contract (rule 3).

Inputs
------
- ``q`` ``[T, H, 128]`` (bf16; fp16 on the fp16 kernel's ``q_f16`` path), H = 16 Hkv:
  query head ``h`` reads KV head ``h // 16``.
- ``kv_cache`` ``[pages, 2 Hkv, 128, 72]`` uint8: one page is one 128-token block,
  ``[K side | V side]``. A side is every KV head's packed E2M1 data (8192 bytes per
  head), then every head's E4M3 block scales (1024 bytes per head, one per 16
  consecutive dims of one token). Inside a head (the KV insert's layout; ``pack_kv_pages`` writes it,
  ``unpack_kv_pages`` reads it):
    - K data token-major, ``[128 tokens][64 bytes]``, a byte = dims ``2i`` (low
      nibble), ``2i + 1`` of one token;
    - K scales ``[4 w][4 c][16 n][2 st][2 i]``: token ``32 w + 8 (n // 4) + 4 st +
      n % 4``, group ``2 c + i``;
    - V data ``[16 token octets][16 dim octets][8 dims][4 bytes]``, a byte = one dim
      of tokens ``2j`` (low nibble), ``2j + 1``;
    - V scales group-major, ``[8 groups][128 tokens]``.
- ``topk`` ``[Hkv, T, 16]`` int32 logical blocks per (KV head, token), -1 = none; a
  token's blocks are distinct, never after its own block, and include its own block
  (the indexer's guarantee).
- ``block_table[query_req_id[t], b]`` is block ``b``'s physical page;
  ``query_abs_pos[t]`` the token's position.
- ``k_scale``, ``v_scale``: fp32 global scales (1.0 in the checkpoints).

What is computed
----------------
For token t and head h, over the keys j of t's selected blocks with position <= pos[t]:

    K_j, V_j = e2m1 code x e4m3 block scale      (exact in fp32, bf16 and fp16)
    s_j      = (q[t, h] . K_j) * sm_scale * k_scale
    out[t,h] = (sum_j softmax(s)_j V_j) * v_scale, rounded to bf16

``sparse_prefill_reference`` evaluates this in fp64 over the exactly dequantized K / V
and the given Q.

Where the kernels round (not in the reference)
----------------------------------------------
- bf16 kernel: Q as given (bf16); scores, the online softmax (the scale folded into
  ``exp2(fma(s, c, -m))``, ``c = fp32(sm_scale) * fp32(log2 e) * k_scale``) and the
  accumulation in fp32; P rounded to bf16 (RNE) against the running max before P @ V;
  the four waves' partials combined in fp32; ``v_scale / l`` applied in fp32; output
  rounded to bf16.
- fp16 kernel: the same with Q converted bf16 -> fp16 in the kernel (exact for
  |q| in [2^-14, 65504]; skipped on the ``q_f16`` path), K / V and P in fp16.
- fp8 kernel (experimental): Q, K, V clamped to [-448, 448] and rounded to e4m3,
  P in e4m3; ``sparse_prefill_reference(..., fp8=True)`` rounds Q, K and V the same way.

Tolerances: relative L2 of the output rows against fp64 below ``REL_L2_TOL`` and cosine above ``COS_MIN`` on the
test inputs (Gaussian Q / K / V of std 2, K / V NVFP4-encoded as the insert does). Repeated launches are bitwise
identical.

MXFP8 outputs
-------------
``quant_out`` (the o_proj's input) and ``pack_out`` (the sequence-parallel all-to-all
rows) quantize the kernel's own bf16 output; they must equal ``mxfp8_quantize`` of
that output byte for byte: per 32 dims of one row, amax floored at fp32 ``tiny``, the
e8m0 scale ``clamp(ceil(log2(amax / 448)) + 127, 0, 254)``, codes
``e4m3_rne(x * 2^(127 - scale))``. ``quant_out`` holds row-major codes ``[>= T, H 128]``
and the scales in AITER's preshuffled layout (``aiter_shuffle_scales``, rows padded to
32); ``pack_out`` rows are H x 128 codes then H x 4 scales (``sp_pack_rows``). Rows past
T (and ``out`` itself) stay untouched.
"""

import torch

BLK = 128  # tokens per page = one sparse block
D = 128  # head dim
HG = 16  # query heads per KV head
TOPK = 16  # selected blocks per (KV head, token)
GROUP = 16  # dims per NVFP4 block scale
DATA_B = BLK * D // 2  # packed e2m1 bytes per (page, side, head)
SCALE_B = BLK * D // GROUP  # e4m3 scale bytes per (page, side, head)
FP8_MAX = 448.0

# above the test maxima (MI355X): rel L2 bf16 1.835e-3, fp16 1.816e-3, fp8 6.86e-3; cosine 0.9999983, fp8 0.9999765
REL_L2_TOL = {"bf16": 2.5e-3, "fp16": 2.5e-3, "fp8": 2.0e-2}
COS_MIN = {"bf16": 0.99999, "fp16": 0.99999, "fp8": 0.9999}

_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


# --- NVFP4 encode / decode -----------------------------------------------------------


def e2m1_to_float(codes: torch.Tensor) -> torch.Tensor:
    """fp32 value of 4-bit E2M1 codes (bit 3 the sign)."""
    mag = torch.tensor(_E2M1, dtype=torch.float32, device=codes.device)
    c = codes.long()
    v = mag[c & 7]
    return torch.where((c & 8) != 0, -v, v)


def float_to_e2m1(x: torch.Tensor) -> torch.Tensor:
    """Round-to-nearest-even, saturating E2M1 codes of fp32 ``x`` (the KV insert's rounding, bit for bit)."""
    ax = x.abs()
    twice = 2.0 * ax
    small = (twice > 0.5).long() + (twice >= 1.5).long()
    bits = ax.view(torch.int32).long()
    rounded = bits + 0x1FFFFF + ((bits >> 22) & 1)
    big = ((rounded >> 22) - 252).clamp(max=7)
    mag = torch.where(ax < 1.0, small, big)
    sign = (x.view(torch.int32).long() >> 28) & 8
    return (mag | sign).to(torch.uint8)


def nvfp4_quantize(
    x: torch.Tensor, global_scale: torch.Tensor | float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """``x`` ``[..., 128]`` -> E2M1 codes ``[..., 128]`` and E4M3 scale bytes
    ``[..., 8]``, the insert's encoder: scale ``e4m3(min(amax / 6 / global, 448))``,
    codes ``e2m1(x / (scale * global))`` (fp32, the insert's operation order)."""
    gs = torch.as_tensor(global_scale, dtype=torch.float32, device=x.device)
    g = x.float().reshape(*x.shape[:-1], D // GROUP, GROUP)
    group_max = g.abs().amax(-1)
    sixth = torch.tensor(0.16666667163372040, dtype=torch.float32, device=x.device)
    scale = torch.clamp(group_max * sixth * (1.0 / gs), max=FP8_MAX)
    scale_e4m3 = scale.to(torch.float8_e4m3fn)
    value = scale_e4m3.float()
    quant = torch.where(value != 0.0, 1.0 / (value * gs), torch.zeros_like(value))
    codes = float_to_e2m1(g * quant[..., None]).reshape(x.shape)
    return codes, scale_e4m3.view(torch.uint8)


def _offsets(hkv: int, device):
    """Byte offsets inside a page and nibble shifts for (head, token, dim) and the
    scale offsets for (head, token, group), K and V sides."""
    tok = torch.arange(BLK, device=device)[:, None]
    dim = torch.arange(D, device=device)[None, :]
    grp = torch.arange(D // GROUP, device=device)[None, :]
    k_data = tok * (D // 2) + dim // 2
    k_shift = (dim % 2) * 4 + 0 * tok
    v_data = ((tok // 8 * 16 + dim // 8) * 8 + dim % 8) * 4 + (tok % 8) // 2
    v_shift = (tok % 2) * 4 + 0 * dim
    r = tok % 32
    n = 4 * (r // 8) + r % 4
    k_scale = (
        ((((tok // 32) * 4 + grp // 2) * 16 + n) * 2 + (r % 8) // 4) * 2 + grp % 2
    )
    v_scale = grp * BLK + tok
    head = torch.arange(hkv, device=device)[:, None, None]
    side = hkv * (DATA_B + SCALE_B)
    return {
        "k_data": (head * DATA_B + k_data).reshape(-1),
        "k_shift": k_shift.expand(hkv, -1, -1).reshape(-1),
        "k_scale": (hkv * DATA_B + head * SCALE_B + k_scale).reshape(-1),
        "v_data": (side + head * DATA_B + v_data).reshape(-1),
        "v_shift": v_shift.expand(hkv, -1, -1).reshape(-1),
        "v_scale": (side + hkv * DATA_B + head * SCALE_B + v_scale).reshape(-1),
    }


def pack_kv_pages(
    k_codes: torch.Tensor,
    k_scales: torch.Tensor,
    v_codes: torch.Tensor,
    v_scales: torch.Tensor,
) -> torch.Tensor:
    """Codes ``[pages, Hkv, 128 tokens, 128 dims]`` and scale bytes ``[pages, Hkv, 128
    tokens, 8 groups]`` (uint8) -> the cache ``[pages, 2 Hkv, 128, 72]`` uint8."""
    pages, hkv = k_codes.shape[:2]
    off = _offsets(hkv, k_codes.device)
    page_b = 2 * hkv * (DATA_B + SCALE_B)
    buf = torch.zeros(pages, page_b, dtype=torch.int32, device=k_codes.device)
    for codes, scales, s in ((k_codes, k_scales, "k"), (v_codes, v_scales, "v")):
        nib = codes.reshape(pages, -1).int() << off[f"{s}_shift"].int()
        buf.scatter_add_(1, off[f"{s}_data"].expand(pages, -1), nib)
        buf.scatter_(1, off[f"{s}_scale"].expand(pages, -1), scales.reshape(pages, -1).int())
    return buf.to(torch.uint8).view(pages, 2 * hkv, BLK, 72)


def unpack_kv_pages(
    cache: torch.Tensor, hkv: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Inverse of ``pack_kv_pages``: K codes, K scales, V codes, V scales. ``cache``
    may be a strided view (``stride(0)`` larger than one page)."""
    pages = cache.shape[0]
    flat = cache.reshape(pages, -1).long()
    off = _offsets(hkv, cache.device)
    out = []
    for s in ("k", "v"):
        byte = flat.gather(1, off[f"{s}_data"].expand(pages, -1))
        codes = (byte >> off[f"{s}_shift"]) & 0xF
        scales = flat.gather(1, off[f"{s}_scale"].expand(pages, -1))
        out += [
            codes.to(torch.uint8).view(pages, hkv, BLK, D),
            scales.to(torch.uint8).view(pages, hkv, BLK, D // GROUP),
        ]
    return tuple(out)  # type: ignore[return-value]


def dequant_kv_pages(cache: torch.Tensor, hkv: int) -> tuple[torch.Tensor, torch.Tensor]:
    """K, V ``[pages, Hkv, 128 tokens, 128 dims]`` fp32 (without the global scales),
    exact."""
    k_codes, k_sc, v_codes, v_sc = unpack_kv_pages(cache, hkv)
    out = []
    for codes, sc in ((k_codes, k_sc), (v_codes, v_sc)):
        scale = sc.view(torch.float8_e4m3fn).float().repeat_interleave(GROUP, -1)
        out.append(e2m1_to_float(codes) * scale)
    return out[0], out[1]


# --- attention -----------------------------------------------------------------------


def _e4m3(x: torch.Tensor) -> torch.Tensor:
    return x.float().clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).float()


@torch.inference_mode()
def sparse_prefill_reference(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    topk: torch.Tensor,
    block_table: torch.Tensor,
    query_req_id: torch.Tensor,
    query_abs_pos: torch.Tensor,
    num_kv_heads: int,
    sm_scale: float,
    k_scale: torch.Tensor | float | None = None,
    v_scale: torch.Tensor | float | None = None,
    rows: torch.Tensor | list[int] | None = None,
    fp8: bool = False,
    chunk: int = 32,
) -> torch.Tensor:
    """fp64 attention of token rows ``rows`` (all by default) -> ``[len(rows), H,
    128]`` float64 (see the module docstring)."""
    hkv = num_kv_heads
    dev = q.device
    k, v = dequant_kv_pages(kv_cache, hkv)
    qf = q.float()
    if fp8:
        k, v, qf = _e4m3(k), _e4m3(v), _e4m3(qf)
    k, v = k.double(), v.double()
    ks = 1.0 if k_scale is None else float(k_scale)
    vs = 1.0 if v_scale is None else float(v_scale)
    if rows is None:
        rows = torch.arange(q.shape[0], device=dev)
    rows = torch.as_tensor(rows, device=dev, dtype=torch.long)
    heads = torch.arange(hkv, device=dev)[:, None, None]
    key = torch.arange(BLK, device=dev)
    out = torch.empty(rows.numel(), hkv * HG, D, dtype=torch.float64, device=dev)
    for c0 in range(0, rows.numel(), chunk):
        r = rows[c0 : c0 + chunk]
        blocks = topk[:, r, :].long()  # [Hkv, C, 16]
        valid = blocks >= 0
        req = query_req_id[r].long()[None, :, None]
        pages = block_table.long()[req, blocks.clamp(min=0)]  # [Hkv, C, 16]
        kk = k[pages, heads]  # [Hkv, C, 16, 128 keys, 128 dims]
        vv = v[pages, heads]
        key_pos = blocks[..., None] * BLK + key  # [Hkv, C, 16, 128]
        mask = valid[..., None] & (key_pos <= query_abs_pos[r].long()[None, :, None, None])
        qh = qf[r].double().view(-1, hkv, HG, D).permute(1, 0, 2, 3)  # [Hkv, C, 16, 128]
        s = torch.einsum("kchd,kcbjd->kchbj", qh, kk) * (sm_scale * ks)
        s = s.masked_fill(~mask[:, :, None], float("-inf")).flatten(-2)
        p = torch.softmax(s, -1)
        o = torch.einsum("kchn,kcnd->kchd", p, vv.flatten(2, 3)) * vs
        out[c0 : c0 + r.numel()] = o.permute(1, 0, 2, 3).reshape(r.numel(), hkv * HG, D)
    return out


def rel_l2(out: torch.Tensor, ref: torch.Tensor) -> float:
    o = out.double()
    return float((o - ref).norm() / ref.norm())


def cosine(out: torch.Tensor, ref: torch.Tensor) -> float:
    o, r = out.double().flatten(), ref.flatten()
    return float((o @ r) / (o.norm() * r.norm()))


# --- MXFP8 outputs -------------------------------------------------------------------


def mxfp8_quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """bf16 ``x`` ``[M, K]`` -> e4m3 codes ``[M, K]`` and e8m0 scales ``[M, K / 32]``
    (uint8, row-major), the kernels' epilogue quant (module docstring)."""
    m, kd = x.shape
    xf = x.float().view(m, kd // 32, 32)
    amax = xf.abs().amax(-1).clamp(min=torch.finfo(torch.float32).tiny)
    mant, exp = torch.frexp(amax / FP8_MAX)
    ceil_log2 = torch.where(mant == 0.5, exp - 1, exp)
    sb = (ceil_log2 + 127).clamp(0, 254)
    rs = torch.ldexp(torch.ones_like(amax), (127 - sb).to(torch.int32))
    codes = (xf * rs[..., None]).to(torch.float8_e4m3fn).view(torch.uint8)
    return codes.reshape(m, kd), sb.to(torch.uint8)


def aiter_shuffle_scales(scales: torch.Tensor) -> torch.Tensor:
    """Row-major e8m0 scales ``[M32, G]`` (M32 a multiple of 32, G of 8) -> AITER's
    preshuffled layout ``[M32 / 32, G / 8, 4, 16, 2, 2]`` flattened to ``[M32, G]``:
    (row r, group gi) at ``(r // 32, gi // 8, gi % 4, r % 16, (gi % 8) // 4,
    (r % 32) // 16)``."""
    m32, g = scales.shape
    return (
        scales.reshape(m32 // 32, 2, 16, g // 8, 2, 4)
        .permute(0, 3, 5, 2, 4, 1)
        .reshape(m32, g)
    )


def aiter_unshuffle_scales(shuffled: torch.Tensor) -> torch.Tensor:
    """Inverse of ``aiter_shuffle_scales``."""
    m32, g = shuffled.shape
    return (
        shuffled.reshape(m32 // 32, g // 8, 4, 16, 2, 2)
        .permute(0, 5, 3, 1, 4, 2)
        .reshape(m32, g)
    )


def sp_pack_rows(x: torch.Tensor) -> torch.Tensor:
    """bf16 ``[M, H 128]`` -> the sequence-parallel all-to-all rows ``[M, H 132]``
    uint8: the H x 128 e4m3 codes, then the H x 4 e8m0 scales."""
    codes, scales = mxfp8_quantize(x)
    return torch.cat([codes, scales], 1)


# --- NVIDIA's numerics (the fp8 target) ---
#
# NV's (GB300) MSA Q8KV4 prefill with ``use_prefill_query_quantization`` and its decode (the decode entry's
# ``reference.py`` lists the sources). Q = e4m3_rn_satfinite(float(bf16 q) * fp32(1 / q_scale)) (per-tensor
# ``q_scale``, folded into the softmax scale); K / V = e4m3_rn_satfinite(e2m1 x block scale) (``_e4m3`` of the exact
# values: the same bytes); fp32 scores; an online softmax over the token's blocks in slot order with
# P = E4M3(exp(s - m_running) * 448) into P V, the denominator over the unrounded exp(s - m) * 448,
# O = acc / max(denominator, 1). MSA's own prefill reference rounds per-block partials to bf16 (its kernel's split);
# that is a property of that kernel, not of the rule, and is left out.

P_SCALE = 448.0


def quantize_query_e4m3(q: torch.Tensor, q_scale: float = 1.0) -> torch.Tensor:
    """NV's Q8 of a bf16 ``q`` (``torch.float8_e4m3fn``); ``query.pack_query`` gives
    these codes for ``q_scale`` 1, reordered for the fp8 kernel."""
    assert q.dtype == torch.bfloat16
    inv = torch.tensor(1.0 / q_scale, dtype=torch.float32, device=q.device)
    return (q.float() * inv).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)


@torch.no_grad()
def nv_sparse_prefill_reference(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    topk: torch.Tensor,
    block_table: torch.Tensor,
    query_req_id: torch.Tensor,
    query_abs_pos: torch.Tensor,
    num_kv_heads: int,
    sm_scale: float,
    rows: torch.Tensor | list[int] | None = None,
    q_scale: float = 1.0,
    chunk: int = 32,
) -> torch.Tensor:
    """fp32 ``[len(rows), H, 128]``: NV's Q8KV4 attention of token rows ``rows`` (all
    by default), unit global K / V scales."""
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        hkv, dev = num_kv_heads, q.device
        k, v = (_e4m3(x) for x in dequant_kv_pages(kv_cache, hkv))
        if rows is None:
            rows = torch.arange(q.shape[0], device=dev)
        rows = torch.as_tensor(rows, device=dev, dtype=torch.long)
        qf8 = quantize_query_e4m3(q[rows], q_scale).float()
        pos = query_abs_pos[rows].long()
        req = query_req_id[rows].long()
        key = torch.arange(BLK, device=dev)
        out = torch.zeros(rows.numel(), hkv * HG, D, dtype=torch.float32, device=dev)
        for h in range(hkv):
            heads = slice(h * HG, (h + 1) * HG)
            for c0 in range(0, rows.numel(), chunk):
                rs = slice(c0, min(rows.numel(), c0 + chunk))
                n = rs.stop - rs.start
                blk = topk[h, rows[rs]].long()
                own = torch.div(pos[rs], BLK, rounding_mode="floor")
                blk = torch.where((blk >= 0) & (blk <= own[:, None]), blk, -1)
                page = block_table.long()[req[rs][:, None], blk.clamp(min=0)]
                kk, vv = k[page, h], v[page, h]  # [n, 16 blocks, 128, 128]
                m = torch.full((n, HG), float("-inf"), device=dev)
                den = torch.zeros(n, HG, device=dev)
                acc = torch.zeros(n, HG, D, device=dev)
                for j in range(blk.shape[1]):
                    b = blk[:, j]
                    vis = (b >= 0)[:, None] & (
                        b.clamp(min=0)[:, None] * BLK + key <= pos[rs][:, None]
                    )
                    kj = kk[:, j].masked_fill(~vis[..., None], 0)
                    vj = vv[:, j].masked_fill(~vis[..., None], 0)
                    s = torch.einsum("nhd,ntd->nht", qf8[rs, heads], kj) * (sm_scale * q_scale)
                    s = s.masked_fill(~vis[:, None, :], float("-inf"))
                    new_m = torch.maximum(m, s.amax(-1))
                    safe = torch.where(torch.isfinite(new_m), new_m, 0.0)
                    corr = torch.exp(m - safe)
                    p = torch.exp(s - safe[..., None]) * P_SCALE
                    acc = acc * corr[..., None] + torch.einsum(
                        "nht,ntd->nhd", p.to(torch.float8_e4m3fn).float(), vj
                    )
                    den = den * corr + p.sum(-1)
                    m = new_m
                out[rs, heads] = acc / den.clamp_min(1.0)[..., None]
        return out
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev


# --- The fp8 kernels' P rule: P against its chunk's own max (schedule-independent) ---
#
# NV's slot-order running max makes a P's bits follow the visiting order (the decode entry's ``reference.py``). The
# fp8 kernels of both entries round each P against the max m_c of its chunk, a (token, head)'s 32 keys
# [32 w, 32 w + 32) of a page (the keys wave w owns):
#   P = E4M3(exp2(s c - m_c) x 448),  O = sum_c 2^(m_c - M) P_c V_c / (448 sum_k
#   exp2(s_k c - M)),  M = the token's max, c = sm_scale * log2(e) * k_scale;
# everything else is NV's Q8KV4 (E4M3 Q, K / V, fp8 MFMA, fp32 accumulation, the
# unrounded denominator). The same token gets the same P as a decode row.

LOG2E_F32 = 1.4426950409


def fp8_chunk_attend(q8, k, v, blocks, pos, c):
    """The chunk P rule for ``n`` tokens of one KV head (as the decode entry's):
    ``q8`` ``[n, 16, 128]`` fp32 E4M3 values, ``k`` / ``v`` ``[n, topk, 128, 128]``
    fp32 E4M3 values, ``blocks`` ``[n, topk]`` (-1: none), ``pos`` ``[n]``, ``c`` the
    fp32 exp2 score scale. fp64 (O x 448 sum ``[n, 16, 128]``, the sum ``[n, 16]``)."""
    n, hg, d = q8.shape
    tk = blocks.shape[1]
    tok = torch.arange(BLK, device=q8.device)
    key = blocks.clamp(min=0)[:, :, None] * BLK + tok
    vis = ((blocks >= 0) & (pos >= 0)[:, None])[:, :, None] & (key <= pos[:, None, None])
    kk = k.masked_fill(~vis[..., None], 0)
    vv = v.masked_fill(~vis[..., None], 0)  # bytes past a length may be NaN
    raw = torch.einsum("nhd,njtd->njht", q8, kk)
    raw = raw.masked_fill(~vis[:, :, None, :], float("-inf")).view(n, tk, hg, 4, 32)
    mc = raw.amax(-1) * c  # [n, tk, 16, 4], fp32 as the kernels (max(a) * c)
    safe = torch.where(torch.isfinite(mc), mc, 0.0)
    p = torch.exp2((raw.double() * c - safe[..., None].double()).float())
    p8 = (p * P_SCALE).to(torch.float8_e4m3fn).double()
    pv = torch.einsum("njhcs,njcsd->njhcd", p8, vv.double().view(n, tk, 4, 32, d))
    big = mc.amax((1, 3))
    big = torch.where(torch.isfinite(big), big, 0.0)
    wt = torch.exp2(mc.double() - big[:, None, :, None].double())
    acc = torch.einsum("njhc,njhcd->nhd", wt, pv)
    den = (wt * p.double().sum(-1)).sum((1, 3))
    return acc, den


@torch.no_grad()
def fp8_chunk_prefill_reference(
    q, kv_cache, topk, block_table, query_req_id, query_abs_pos,
    num_kv_heads, sm_scale, rows=None, k_scale=None, v_scale=None, chunk=32,
):
    """fp64 ``[len(rows), H, 128]`` (all rows by default): the fp8 prefill kernels'
    contract, NV's Q8KV4 with P rounded against its chunk's max (above). Between a
    kernel and this: fp32 summation orders and ``v_exp_f32`` ulps."""
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        hkv, dev = num_kv_heads, q.device
        k, v = (_e4m3(x) for x in dequant_kv_pages(kv_cache, hkv))
        if rows is None:
            rows = torch.arange(q.shape[0], device=dev)
        rows = torch.as_tensor(rows, device=dev, dtype=torch.long)
        ks = 1.0 if k_scale is None else float(k_scale)
        vs = 1.0 if v_scale is None else float(v_scale)
        c = float(
            torch.tensor(sm_scale, dtype=torch.float32)
            * torch.tensor(LOG2E_F32, dtype=torch.float32)
            * torch.tensor(ks, dtype=torch.float32)
        )
        q8 = quantize_query_e4m3(q[rows]).float()
        pos = query_abs_pos[rows].long()
        req = query_req_id[rows].long()
        out = torch.zeros(rows.numel(), hkv * HG, D, dtype=torch.float64, device=dev)
        for h in range(hkv):
            heads = slice(h * HG, (h + 1) * HG)
            for c0 in range(0, rows.numel(), chunk):
                rs = slice(c0, min(rows.numel(), c0 + chunk))
                blk = topk[h, rows[rs]].long()
                own = torch.div(pos[rs], BLK, rounding_mode="floor")
                blk = torch.where((blk >= 0) & (blk <= own[:, None]), blk, -1)
                page = block_table.long()[req[rs][:, None], blk.clamp(min=0)]
                acc, den = fp8_chunk_attend(q8[rs, heads], k[page, h], v[page, h], blk, pos[rs], c)
                out[rs, heads] = torch.where(
                    den[..., None] > 0, acc * vs / (den[..., None] * P_SCALE), 0.0
                )
        return out
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev
