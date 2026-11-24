# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The contract of the NVFP4 sparse decode (root README, rule 3): torch only.

The cache
---------
``kv_cache`` is ``[pages, 2 * Hkv, 128, 72]`` uint8, one page per 128-token block: ``[K side | V side]``; a side is
every KV head's packed E2M1 data (8192 bytes per head), then every head's E4M3 block scales (1024 bytes per head, one
per 16 consecutive dims of one token). Inside a head (``layout_offsets``):

- K data: token-major ``[128 tokens][64 bytes]``; a byte holds dims 2i (low nibble)
  and 2i + 1 of one token.
- K scales: ``[4 token quarters w][4 dim chunks c][16 n][2 st][2 i]`` for token
  ``32 w + 8 (n // 4) + 4 st + n % 4`` and group ``2 c + i``.
- V data: ``[16 token octets][16 dim octets][8 dims][4 bytes]``; a byte holds tokens
  2j (low nibble) and 2j + 1 of one dim.
- V scales: group-major, ``[8 groups][128 tokens]``.

An element is ``e2m1 x block scale x global scale``; the global scales ``k_scale`` / ``v_scale`` are fp32 scalars.
``quantize_kv`` is the cache insert's arithmetic (``attention/norm_rope_nvfp4_insert_d128_rot64_p128``): block scale
``e4m3(min(amax * fp32(1/6) * (1 / global), 448))``, codes ``x / (scale * global)`` rounded to the nearest E2M1 value,
ties to even, saturating at 6, the sign kept for zeros.

The attention
-------------
Row ``r`` is query row ``r % ql`` of request ``r // ql`` (``ql`` = ``decode_query_len``) at position
``pos = seq_len - ql + r % ql``. For each KV head ``h`` its 16 query heads ``16 h .. 16 h + 15`` attend over the tokens
of the logical blocks ``topk_idx[h, r, :]`` (-1 = none) that are not past the row's own block ``pos // 128``, every
such token ``<= pos`` (the own block is masked causally), read from page ``block_table[r // ql, block]``. A block
listed twice is attended twice. A row with ``pos < 0`` (a CUDA graph padding request, ``seq_len`` 0) or with no
visible token outputs 0.

The kernel's arithmetic, by ``compute_type``:

- ``torch.bfloat16`` (the 1-KV-head path): K / V elements are ``e4m3_rn_satfinite(e2m1 x block scale)`` widened to
  bf16; bf16 Q; fp32 scores and online softmax (``exp2`` with ``sm_scale x log2(e)``, the K global scale folded in); P
  rounded to bf16 (RNE); fp32 ``P V``; the V global scale applied at the end; bf16 output (RNE).
- ``torch.float16`` (the 4-KV-head path): K / V elements are ``e2m1 x block scale`` exactly (fp16 holds every
  product); Q converted to fp16 (exact for the bf16 values of normalized queries); P rounded to fp16; the rest as
  above.

``sparse_decode_reference`` is fp64 attention over these K / V values (rounded to E4M3 when ``round_kv_to_e4m3``, the
bf16 path) times the global scales, with the bf16 Q and unrounded P. The tests bound the kernel's distance to it:

- ``REL_L2`` (relative L2 over a launch's live rows) and ``COS`` (cosine similarity); measured rel. L2 1.73e-3 -
  2.12e-3 on the bf16 path (the high end on caches of random bytes) and 1.62e-3 - 1.68e-3 on the fp16 path, cosine
  >= 0.9999978, most of it the bf16 output's own rounding.
- Repeated launches are equal bit for bit: the split partials are combined in split order, not in arrival order.

MXFP8 output (``quant_out``)
----------------------------
Instead of the bf16 output the kernel can write ``mxfp8_quantize`` of its own bf16 output (the bytes of vLLM's
``_mxfp8_quantize_kernel``, the o_proj input): per 32 consecutive elements of a row,
``e = clamp(ceil(log2(max(amax, tiny) / 448)) + 127, 0, 254)`` (E8M0) and codes ``e4m3_rn(x * 2^(127 - e))``. Codes go
row-major ``[ceil(rows / 16) * 16, H * 128]`` or in the decode GEMM's (16, 16) tiles (``tile_code_offsets``), scales
in AITER's shuffled layout (``aiter_scale_offsets``); rows past ``rows`` are not written. The tests check the bytes
against ``mxfp8_quantize`` of the bf16 output of the same schedule, and the dequantized output within ``QUANT_REL`` of
``sparse_decode_reference`` per row (measured worst rows 2.65e-2 - 2.95e-2: E4M3's 3 mantissa bits).
"""

import torch

BLOCK = 128  # tokens per page
HEAD_DIM = 128
GROUP = 16  # query heads per KV head
TOPK = 16
SCALE_GROUP = 16  # NVFP4 elements per block scale
PAGE_SIDE_HEAD_BYTES = 9216  # data 8192 + scales 1024, per KV head and side
E2M1_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
TINY_F32 = torch.finfo(torch.float32).tiny

REL_L2 = 5e-3
COS = 0.99999
QUANT_REL = 0.04  # per row, the dequantized MXFP8 output


def layout_offsets(side: int, device):
    """``[128, 128]`` data-byte offsets and nibble shifts of each (token, dim), and ``[128, 8]`` scale-byte offsets of
    each (token, group), inside one head's regions of side 0 (K) or 1 (V)."""
    t = torch.arange(BLOCK, device=device)[:, None]
    d = torch.arange(HEAD_DIM, device=device)[None, :]
    g = torch.arange(HEAD_DIM // SCALE_GROUP, device=device)[None, :]
    if side == 1:
        data = ((t // 8 * 16 + d // 8) * 8 + d % 8) * 4 + (t % 8) // 2
        shift = (t % 2) * 4 + 0 * d
        scale = g * 128 + t
    else:
        data = t * 64 + d // 2
        shift = (d % 2) * 4 + 0 * t
        r = t % 32
        n = 4 * (r // 8) + r % 4
        scale = ((((t // 32) * 4 + g // 2) * 16 + n) * 2 + (r % 8) // 4) * 2 + g % 2
    return data, shift, scale


def quantize_kv(x: torch.Tensor, global_scale) -> tuple[torch.Tensor, torch.Tensor]:
    """``[..., 128]`` values -> E2M1 codes ``[..., 128]`` uint8 (0..15) and E4M3 block scales ``[..., 8]`` uint8, with
    the cache insert's arithmetic."""
    groups = x.float().reshape(*x.shape[:-1], HEAD_DIM // SCALE_GROUP, SCALE_GROUP)
    g = torch.as_tensor(global_scale, dtype=torch.float32, device=x.device)
    sixth = torch.tensor(0.16666667163372040, dtype=torch.float32, device=x.device)
    amax = groups.abs().amax(-1)
    scale = torch.minimum(amax * sixth * (1.0 / g), g.new_tensor(448.0))
    scale_e4m3 = scale.to(torch.float8_e4m3fn)
    value = scale_e4m3.float()
    quant = torch.where(value != 0.0, 1.0 / (value * g), 0.0)
    y = groups * quant[..., None]
    # round-to-nearest-even, saturating E2M1 (the insert's _f32_to_e2m1)
    ax = y.abs()
    twice = 2.0 * ax
    small = (twice > 0.5).int() + (twice >= 1.5).int()
    bits = ax.view(torch.int32)
    rounded = bits + 0x1FFFFF + ((bits >> 22) & 1)
    big = torch.clamp((rounded >> 22) - 252, max=7)
    mag = torch.where(ax < 1.0, small, big)
    codes = mag | ((y.view(torch.int32) >> 28) & 8)
    return codes.reshape(x.shape).to(torch.uint8), scale_e4m3.view(torch.uint8)


def pack_kv(k_codes, k_scales, v_codes, v_scales) -> torch.Tensor:
    """Codes ``[pages, Hkv, 128 tokens, 128 dims]`` and scale bytes ``[pages, Hkv, 128, 8]`` of K and V -> the cache
    ``[pages, 2 * Hkv, 128, 72]`` uint8."""
    pages, hkv = k_codes.shape[:2]
    device = k_codes.device
    sides = []
    for side, (codes, scales) in enumerate(((k_codes, k_scales), (v_codes, v_scales))):
        data_off, shift, scale_off = layout_offsets(side, device)
        nibbles = codes.int() << shift.int()
        data = torch.zeros(pages, hkv, 8192, dtype=torch.int32, device=device)
        index = data_off.reshape(1, 1, -1).expand(pages, hkv, -1)
        data.scatter_add_(-1, index, nibbles.reshape(pages, hkv, -1))
        scale_bytes = torch.empty(pages, hkv, 1024, dtype=torch.uint8, device=device)
        scale_bytes[..., scale_off.reshape(-1)] = scales.reshape(pages, hkv, -1)
        sides += [data.to(torch.uint8).reshape(pages, -1), scale_bytes.reshape(pages, -1)]
    return torch.cat(sides, -1).view(pages, 2 * hkv, BLOCK, 72)


def unpack_kv(kv_cache: torch.Tensor):
    """The inverse of ``pack_kv``: ``(k_codes, k_scales, v_codes, v_scales)``; ``kv_cache`` may be a view whose page
    stride exceeds one page."""
    pages, slots = kv_cache.shape[:2]
    hkv = slots // 2
    flat = kv_cache.reshape(pages, -1)  # a page's own bytes are dense
    out = []
    side_bytes = hkv * PAGE_SIDE_HEAD_BYTES
    for side in range(2):
        region = flat[:, side * side_bytes : (side + 1) * side_bytes]
        data = region[:, : hkv * 8192].reshape(pages, hkv, 8192)
        scales = region[:, hkv * 8192 :].reshape(pages, hkv, 1024)
        data_off, shift, scale_off = layout_offsets(side, kv_cache.device)
        data = data[..., data_off.reshape(-1)].reshape(pages, hkv, BLOCK, HEAD_DIM)
        codes = (data >> shift.to(torch.uint8)) & 0xF
        out += [codes, scales[..., scale_off.reshape(-1)].reshape(pages, hkv, BLOCK, 8)]
    return tuple(out)


def quantize_cache(k: torch.Tensor, v: torch.Tensor, k_scale, v_scale) -> torch.Tensor:
    """K / V ``[pages, 128 tokens, Hkv, 128]`` -> the cache (every slot written)."""
    k_codes, k_scales = quantize_kv(k.transpose(1, 2), k_scale)
    v_codes, v_scales = quantize_kv(v.transpose(1, 2), v_scale)
    return pack_kv(k_codes, k_scales, v_codes, v_scales)


def kv_values(kv_cache: torch.Tensor, round_kv_to_e4m3: bool):
    """(K, V) ``[pages, Hkv, 128 tokens, 128]`` fp32 without the global scales: e2m1 x block scale, or its
    e4m3_rn_satfinite."""
    grid = torch.tensor(E2M1_GRID, device=kv_cache.device)
    k_codes, k_scales, v_codes, v_scales = unpack_kv(kv_cache)
    sides = []
    for codes, scales in ((k_codes, k_scales), (v_codes, v_scales)):
        mag = grid[(codes & 7).long()] * torch.where(codes >= 8, -1.0, 1.0)
        scale = scales.view(torch.float8_e4m3fn).float()
        value = (mag.unflatten(-1, (8, SCALE_GROUP)) * scale[..., None]).flatten(-2)
        if round_kv_to_e4m3:
            value = value.clamp(-448, 448).to(torch.float8_e4m3fn).float()
        sides.append(value)
    return sides[0], sides[1]


@torch.no_grad()
def sparse_decode_reference(
    q: torch.Tensor,  # [rows, Hkv * 16, 128] bf16
    kv_cache: torch.Tensor,  # [pages, 2 * Hkv, 128, 72] uint8
    topk_idx: torch.Tensor,  # [Hkv, >= rows, topk] int32 logical blocks, -1 = none
    block_table: torch.Tensor,  # [reqs, max_blocks] int32
    seq_lens: torch.Tensor,  # [reqs] int32
    sm_scale: float,
    k_scale,
    v_scale,
    decode_query_len: int,
    round_kv_to_e4m3: bool,
    row_chunk: int = 32,
) -> torch.Tensor:
    """fp64 ``[rows, Hkv * 16, 128]``: attention over the decode's K / V values."""
    k_all, v_all = kv_values(kv_cache, round_kv_to_e4m3)
    k_all = k_all.double() * float(k_scale)
    v_all = v_all.double() * float(v_scale)
    rows = q.shape[0]
    hkv = topk_idx.shape[0]
    ql = decode_query_len
    dev = q.device
    req = torch.arange(rows, device=dev) // ql
    pos = seq_lens.long()[req] - ql + torch.arange(rows, device=dev) % ql
    offs = torch.arange(BLOCK, device=dev)
    out = torch.zeros(q.shape, dtype=torch.float64, device=dev)
    for h in range(hkv):
        heads = slice(h * GROUP, (h + 1) * GROUP)
        for r0 in range(0, rows, row_chunk):
            rs = slice(r0, min(rows, r0 + row_chunk))
            p_rs = pos[rs]
            blk = topk_idx[h, rs].long()
            own = torch.div(p_rs, BLOCK, rounding_mode="floor")
            ok = (p_rs[:, None] >= 0) & (blk >= 0) & (blk <= own[:, None])
            blk = torch.where(ok, blk, 0)
            tok = blk[..., None] * BLOCK + offs  # [n, topk, BLOCK]
            ok = (ok[..., None] & (tok <= p_rs[:, None, None])).flatten(1)
            page = block_table[req[rs][:, None], blk].long()[..., None].expand_as(tok)
            k = k_all[page, h, tok % BLOCK].flatten(1, 2)
            v = v_all[page, h, tok % BLOCK].flatten(1, 2)
            # bytes past a length may decode to NaN; zero weight cannot mask NaN
            k = k.masked_fill(~ok[..., None], 0)
            v = v.masked_fill(~ok[..., None], 0)
            s = torch.einsum("nhd,ntd->nht", q[rs, heads].double(), k) * sm_scale
            s = s.masked_fill(~ok[:, None, :], float("-inf"))
            p = torch.softmax(s, -1).nan_to_num(0.0)  # rows that see nothing
            out[rs, heads] = torch.einsum("nht,ntd->nhd", p, v)
    return out


def mxfp8_quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``[rows, K]`` (bf16 values) -> E4M3 code bytes ``[rows, K]`` and E8M0 scale bytes ``[rows, K // 32]``,
    row-major (``ceil(log2)`` taken exactly from the fp32 exponent)."""
    v = x.float().unflatten(-1, (-1, 32))
    amax = v.abs().amax(-1).clamp(min=TINY_F32)
    y = amax / 448.0
    mant, exp = torch.frexp(y)  # y = mant * 2^exp, mant in [0.5, 1)
    ceil_log2 = torch.where(mant == 0.5, exp - 1, exp)
    # below fp32's normal range ceil(log2(y)) + 127 < 1: the scale clamps to 0
    e = torch.where(y < TINY_F32, 0, (ceil_log2 + 127).clamp(0, 254))
    rescale = torch.ldexp(torch.ones_like(amax), 127 - e)
    codes = (v * rescale[..., None]).to(torch.float8_e4m3fn).view(torch.uint8)
    return codes.flatten(-2), e.to(torch.uint8)


def mxfp8_dequantize(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Row-major codes ``[rows, K]`` and scales ``[rows, K // 32]`` -> fp64 values."""
    values = codes.view(torch.float8_e4m3fn).double().unflatten(-1, (-1, 32))
    one = torch.ones_like(scales, dtype=torch.float64)
    return (values * torch.ldexp(one, scales.long() - 127)[..., None]).flatten(-2)


def tile_code_offsets(rows: int, k: int, device) -> torch.Tensor:
    """``[rows, k]`` byte offsets of the codes in the decode GEMM's (16, 16) tiles."""
    m = torch.arange(rows, device=device)[:, None]
    kk = torch.arange(k, device=device)[None, :]
    return (
        (m // 16) * (16 * k)
        + (kk // 64) * 1024
        + ((kk % 64) // 16) * 256
        + (m % 16) * 16
        + kk % 16
    )


def aiter_scale_offsets(rows: int, k: int, device) -> torch.Tensor:
    """``[rows, k // 32]`` byte offsets of the scales in AITER's shuffled layout ``[M//32, K//256, 4, 16, 2, 2]``."""
    m = torch.arange(rows, device=device)[:, None]
    g = torch.arange(k // 32, device=device)[None, :]
    return (
        ((((m // 32) * (k // 256) + g // 8) * 4 + g % 4) * 16 + m % 16) * 2 + (g % 8) // 4
    ) * 2 + (m % 32) // 16


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm())


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double().flatten(), b.double().flatten()
    return float(a @ b / (a.norm() * b.norm()))


# --- NV's Q8KV4 numerics (the fp8 target) ---
#
# MiniMax's MSA Q8KV4 kernels, which NV (GB300) serves M3.1's sparse attention with (decode always; prefill with
# ``use_prefill_query_quantization``):
# - Q: q materialized in bf16, then ``float(bf16 q) * fp32(1 / q_scale)`` converted to E4M3 RN satfinite; ``q_scale``
#   = the layer's per-tensor q scale (the checkpoint's ``q_scale``, else 1.0), folded into the softmax scale.
# - K / V: ``mul.rn.satfinite.e4m3x4.e2m1x4.e4m3x4``: e4m3_rn_satfinite(e2m1 x block scale), unit global scales;
#   ``kv_values(..., True)``.
# - scores in fp32 from the fp8 products, times the softmax scale; an online softmax over the selected blocks in slot
#   order: P = E4M3(exp(s - m_running) * 448) feeds P V (fp32 accumulation), the denominator sums the unrounded
#   exp(s - m) * 448, O = acc / max(denominator, 1), rounded to bf16.
# A kernel may visit keys in another order (waves, splits), so the running max a P is rounded against differs: compare
# with a tolerance (``NV_REL_L2``), not bit for bit.

P_SCALE = 448.0  # NV's FP8 probability scale (the "P448" convention)


def quantize_query_e4m3(q: torch.Tensor, q_scale: float = 1.0) -> torch.Tensor:
    """NV's Q8 of a bf16 ``q``: e4m3_rn_satfinite(float(q) * fp32(1 / q_scale)), as ``torch.float8_e4m3fn``."""
    assert q.dtype == torch.bfloat16
    inv = torch.tensor(1.0 / q_scale, dtype=torch.float32, device=q.device)
    return (q.float() * inv).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)


def nv_attend(qf8, k, v, blocks, pos, sm_scale):
    """NV's online softmax for ``n`` rows of one KV head: ``qf8`` ``[n, 16, 128]`` fp32 (E4M3 values), ``k`` / ``v``
    ``[n, topk, 128, 128]`` fp32 (E4M3 values) of the rows' selected blocks, ``blocks`` ``[n, topk]`` logical blocks
    (-1: none), ``pos`` ``[n]`` the rows' positions (-1: sees nothing). fp32 ``[n, 16, 128]`` before the output
    rounding."""
    n, hg, d = qf8.shape
    tok = torch.arange(BLOCK, device=qf8.device)
    m = torch.full((n, hg), float("-inf"), device=qf8.device)
    den = torch.zeros(n, hg, device=qf8.device)
    acc = torch.zeros(n, hg, d, device=qf8.device)
    for j in range(blocks.shape[1]):
        blk = blocks[:, j]
        key = blk.clamp(min=0)[:, None] * BLOCK + tok
        vis = ((blk >= 0) & (pos >= 0))[:, None] & (key <= pos[:, None])
        kj = k[:, j].masked_fill(~vis[..., None], 0)
        vj = v[:, j].masked_fill(~vis[..., None], 0)  # bytes past a length may be NaN
        s = torch.einsum("nhd,ntd->nht", qf8, kj) * sm_scale
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
    return acc / den.clamp_min(1.0)[..., None]


@torch.no_grad()
def nv_sparse_decode_reference(
    q: torch.Tensor,  # [rows, Hkv * 16, 128] bf16
    kv_cache: torch.Tensor,  # [pages, 2 * Hkv, 128, 72] uint8
    topk_idx: torch.Tensor,  # [Hkv, >= rows, topk] int32 logical blocks, -1 = none
    block_table: torch.Tensor,  # [reqs, max_blocks] int32
    seq_lens: torch.Tensor,  # [reqs] int32
    sm_scale: float,
    decode_query_len: int,
    q_scale: float = 1.0,
    row_chunk: int = 64,
) -> torch.Tensor:
    """fp32 ``[rows, Hkv * 16, 128]``: NV's Q8KV4 attention (above), unit global K / V scales."""
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        k_all, v_all = kv_values(kv_cache, round_kv_to_e4m3=True)
        rows, hkv, ql, dev = q.shape[0], topk_idx.shape[0], decode_query_len, q.device
        qf8 = quantize_query_e4m3(q, q_scale).float()
        req = torch.arange(rows, device=dev) // ql
        pos = seq_lens.long()[req] - ql + torch.arange(rows, device=dev) % ql
        pos = torch.where(pos >= 0, pos, -1)
        out = torch.zeros(q.shape, dtype=torch.float32, device=dev)
        for h in range(hkv):
            heads = slice(h * GROUP, (h + 1) * GROUP)
            for r0 in range(0, rows, row_chunk):
                rs = slice(r0, min(rows, r0 + row_chunk))
                blk = topk_idx[h, rs].long()
                own = torch.div(pos[rs], BLOCK, rounding_mode="floor")
                blk = torch.where((blk >= 0) & (blk <= own[:, None]), blk, -1)
                page = block_table[req[rs][:, None], blk.clamp(min=0)].long()
                out[rs, heads] = nv_attend(
                    qf8[rs, heads], k_all[page, h], v_all[page, h], blk, pos[rs],
                    sm_scale * q_scale,
                )
        return out
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev


LOG2E_F32 = 1.4426950409

# --- The fp8 kernels' P rule: P against its chunk's own max (schedule-independent) ---
#
# NV's rule rounds P = E4M3(exp(s - m) * 448) against the running max m in slot order, so a P's bits follow the order
# and grouping a kernel visits keys in: a row attended alone (4 splits) and inside a big batch (1 split, or a union)
# gets other P bits (min cos per (row, head) 0.99968 on stored layer inputs), which the cross-M gate (cos 0.99999 per
# (row, head)) does not allow. The fp8 kernels keep everything else of NV's rule (E4M3 Q, K / V, P x 448, fp8 MFMA,
# fp32 accumulation, the unrounded denominator) and round each P against the max m_c of its CHUNK, a (row, head)'s 32
# keys [32 w, 32 w + 32) of a block (the keys wave w owns in every kernel):
#   P = E4M3(exp2(s c - m_c) x 448),  O = sum_c 2^(m_c - M) P_c V_c / (448 sum_k exp2(s_k c - M)),
#   M = the row's max, c = sm_scale * log2(e) * k_scale.
# A chunk's P depends on that chunk alone, so every split count, union and M gives the same P; what remains is fp32
# summation order. (A kernel takes a chunk more than 32 below the running max against max - 32: weight < 2^-32, no
# visible difference.)


def fp8_chunk_attend(q8, k, v, blocks, pos, c):
    """The chunk P rule (above) for ``n`` rows of one KV head: ``q8`` ``[n, 16, 128]`` fp32 (E4M3 values), ``k`` /
    ``v`` ``[n, topk, 128, 128]`` fp32 (E4M3 values) of the rows' selected blocks, ``blocks`` ``[n, topk]`` logical
    blocks (-1: none), ``pos`` ``[n]`` the rows' positions (-1: sees nothing), ``c`` the fp32 exp2 score scale. fp64
    ``[n, 16, 128]`` O x 448 sum (divide by the denominator, also returned)."""
    n, hg, d = q8.shape
    tk = blocks.shape[1]
    tok = torch.arange(BLOCK, device=q8.device)
    key = blocks.clamp(min=0)[:, :, None] * BLOCK + tok
    vis = ((blocks >= 0) & (pos >= 0)[:, None])[:, :, None] & (key <= pos[:, None, None])
    kk = k.masked_fill(~vis[..., None], 0)
    vv = v.masked_fill(~vis[..., None], 0)  # bytes past a length may be NaN
    raw = torch.einsum("nhd,njtd->njht", q8, kk)
    raw = raw.masked_fill(~vis[:, :, None, :], float("-inf")).view(n, tk, hg, 4, 32)
    mc = raw.amax(-1) * c  # [n, tk, 16, 4], fp32 as the kernels (max(a) * c)
    safe = torch.where(torch.isfinite(mc), mc, 0.0)
    # exp2 of fma(s, c, -m_c): one rounding, as the kernels' v_fma_f32
    p = torch.exp2((raw.double() * c - safe[..., None].double()).float())
    p8 = (p * P_SCALE).to(torch.float8_e4m3fn).double()
    pv = torch.einsum("njhcs,njcsd->njhcd", p8, vv.double().view(n, tk, 4, 32, d))
    big = mc.amax((1, 3))  # [n, 16]
    big = torch.where(torch.isfinite(big), big, 0.0)
    wt = torch.exp2(mc.double() - big[:, None, :, None].double())  # 0 for dead chunks
    acc = torch.einsum("njhc,njhcd->nhd", wt, pv)
    den = (wt * p.double().sum(-1)).sum((1, 3))
    return acc, den


@torch.no_grad()
def fp8_chunk_decode_reference(
    q: torch.Tensor,  # [rows, Hkv * 16, 128] bf16
    kv_cache: torch.Tensor,  # [pages, 2 * Hkv, 128, 72] uint8
    topk_idx: torch.Tensor,  # [Hkv, >= rows, topk] int32 logical blocks, -1 = none
    block_table: torch.Tensor,  # [reqs, max_blocks] int32
    seq_lens: torch.Tensor,  # [reqs] int32
    sm_scale: float,
    decode_query_len: int,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    rows=None,
    row_chunk: int = 32,
) -> torch.Tensor:
    """fp64 ``[len(rows), Hkv * 16, 128]`` (all rows by default): the fp8 decode kernels' contract, NV's Q8KV4 with P
    rounded against its chunk's max (above). Between a kernel and this: fp32 summation orders and ``v_exp_f32`` ulps
    (a flipped P rounding now and then)."""
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        k_all, v_all = kv_values(kv_cache, round_kv_to_e4m3=True)
        hkv, ql, dev = topk_idx.shape[0], decode_query_len, q.device
        sel = torch.arange(q.shape[0], device=dev) if rows is None else torch.as_tensor(
            rows, device=dev
        ).long()
        q8 = quantize_query_e4m3(q[sel]).float()
        c = float(
            torch.tensor(sm_scale, dtype=torch.float32)
            * torch.tensor(LOG2E_F32, dtype=torch.float32)
            * torch.tensor(k_scale, dtype=torch.float32)
        )
        req = sel // ql
        pos = seq_lens.long()[req] - ql + sel % ql
        pos = torch.where(pos >= 0, pos, -1)
        out = torch.zeros((sel.numel(),) + q.shape[1:], dtype=torch.float64, device=dev)
        for h in range(hkv):
            heads = slice(h * GROUP, (h + 1) * GROUP)
            for r0 in range(0, sel.numel(), row_chunk):
                rs = slice(r0, min(sel.numel(), r0 + row_chunk))
                blk = topk_idx[h, sel[rs]].long()
                own = torch.div(pos[rs], BLOCK, rounding_mode="floor")
                blk = torch.where((blk >= 0) & (blk <= own[:, None]) & (pos[rs, None] >= 0),
                                  blk, -1)
                page = block_table[req[rs][:, None], blk.clamp(min=0)].long()
                acc, den = fp8_chunk_attend(
                    q8[rs, heads], k_all[page, h], v_all[page, h], blk, pos[rs], c
                )
                out[rs, heads] = torch.where(
                    den[..., None] > 0, acc * v_scale / (den[..., None] * P_SCALE), 0.0
                )
        return out
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev
