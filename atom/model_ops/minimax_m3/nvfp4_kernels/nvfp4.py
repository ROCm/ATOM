# Derived from SGLang (https://github.com/sgl-project/sglang), Apache-2.0; modified for ATOM.
"""NVFP4 cache rows (MSA-v1 layout) and their Triton loads.

A row is one head of one token: ``D // 2`` payload bytes (two e2m1 codes per
byte, the even element in the low nibble) and ``D // 16`` E4M3 block scales,
with a unit outer scale -- the layout of the MiniMax-M3 NVFP4 sparse KV pool
(``MhaKvPool(kv_format="nvfp4")``) and of the FP4 index cache (``index_fp4``).

The Gluon reads of the same rows are in ``nvfp4_gluon``, kept apart so the
Triton scorers (``index_topk``) do not need a Triton with Gluon.
"""

import msgspec
import torch
import triton
import triton.language as tl
from aiter.jit.utils.chip_info import get_gfx

# gfx950 converts a packed e2m1 byte in one instruction; elsewhere (gfx1250) the
# codes are decoded with integer ops. Both give the same fp16 values.
_HW_FP4_CVT = tl.constexpr(get_gfx() == "gfx950")
# Whether the FP4 index scorers read rows whole (`nvfp4_load_rows`, one dot over
# D) rather than as even / odd halves (`nvfp4_load_even_odd`, two dots): every
# arch but gfx950, whose one-byte conversion yields the halves.
NVFP4_WHOLE_ROWS = tl.constexpr(get_gfx() != "gfx950")


class Nvfp4Cache(msgspec.Struct, frozen=True):
    """One layer of a packed NVFP4 cache (K, V or index K).

    ``payload`` is uint8 ``[pages, (heads,) page_size, D // 2]`` and ``scales``
    uint8 ``[pages, (heads,) page_size, D // 16]`` (E4M3 bits); the index-K
    cache has no head dimension.
    """

    payload: torch.Tensor
    scales: torch.Tensor

    @property
    def page_size(self) -> int:
        return self.payload.shape[-2]

    @property
    def device(self) -> torch.device:
        return self.payload.device

    @property
    def num_heads(self) -> int:
        return self.payload.shape[1] if self.payload.dim() == 4 else 1

    def payload_strides(self) -> tuple[int, int, int]:
        """(page, head, row) strides in bytes."""
        return _page_head_row_strides(self.payload)

    def scale_strides(self) -> tuple[int, int, int]:
        return _page_head_row_strides(self.scales)


def _page_head_row_strides(buf: torch.Tensor) -> tuple[int, int, int]:
    if buf.dim() == 4:
        return buf.stride(0), buf.stride(1), buf.stride(2)
    return buf.stride(0), 0, buf.stride(1)


@triton.jit
def _e2m1_to_f16(code):
    """One e2m1 code (low 4 bits of an int32) -> fp16 bits, no conversion
    instruction and no subnormal. Magnitude codes 2..7 are 2^(e-1) * (1 + m/2):
    fp16 exponent e + 14, so the bits are (code << 9) + (14 << 10); code 1 is 0.5
    (0x3800) and code 0 is 0. Bit 3 is the sign."""
    mag = code & 7
    bits = tl.where(mag >= 2, (mag << 9) + 0x3800, mag * 0x3800)
    bits = bits | ((code & 8) << 12)
    return bits.to(tl.int16).to(tl.float16, bitcast=True)


@triton.jit
def _fp4x2_to_f16(byte_i32):
    """One packed e2m1 byte -> (low, high) fp16.

    gfx950 does it in one conversion, whose f32 scale only contributes its
    exponent (E8M0), so the E4M3 block scale is applied by the caller and 1.0
    leaves the codes unscaled. Other archs decode the nibbles arithmetically.
    """
    if _HW_FP4_CVT:
        return tl.inline_asm_elementwise(
            "v_cvt_scalef32_pk_f16_fp4 $0, $2, 1.0\n\tv_lshrrev_b32 $1, 16, $0",
            "=&v,=&v,v",
            [byte_i32],
            dtype=(tl.float16, tl.float16),
            is_pure=True,
            pack=1,
        )
    else:
        return _e2m1_to_f16(byte_i32 & 15), _e2m1_to_f16((byte_i32 >> 4) & 15)


@triton.jit
def nvfp4_load_even_odd(
    payload_ptr,
    scale_ptr,
    payload_row_off,
    scale_row_off,
    row_mask,
    D: tl.constexpr,
):
    """Load rows as their even and odd columns, two fp16 ``[N, D//2]`` tiles.

    Byte ``j`` of a row holds elements ``2j`` (low nibble) and ``2j+1``, so the
    split needs no interleave; a dot over D is the sum of the even and odd dots.
    e2m1 times an E4M3 scale has at most 6 significant bits and stays in the fp16
    normal range, so the dequantized values are exact.
    """
    HALF: tl.constexpr = D // 2
    cols = tl.arange(0, HALF)
    packed = tl.load(
        payload_ptr + payload_row_off[:, None] + cols[None, :],
        mask=row_mask[:, None],
        other=0,
    )
    scale_bytes = tl.load(
        scale_ptr + scale_row_off[:, None] + (cols // 8)[None, :],
        mask=row_mask[:, None],
        other=0,
    )
    even, odd = _fp4x2_to_f16(packed.to(tl.int32))
    scale = scale_bytes.to(tl.float8e4nv, bitcast=True).to(tl.float16)
    return even * scale, odd * scale


@triton.jit
def _scaled_f16_word(packed, byte: tl.constexpr, scale):
    """Word ``byte`` of ``_fp4x8_scaled_f16_words``, by integer decode."""
    b = (packed >> (8 * byte)) & 0xFF
    lo = (_e2m1_to_f16(b & 15) * scale).to(tl.int16, bitcast=True).to(tl.int32)
    hi = (_e2m1_to_f16(b >> 4) * scale).to(tl.int16, bitcast=True).to(tl.int32)
    return (lo & 0xFFFF) | (hi << 16)


@triton.jit
def _fp4x8_scaled_f16_words(packed, scale2, unit):
    """One payload dword (eight e2m1 codes, half a 16-element block) and its block
    scale as an fp16 pair -> four int32 words, word j = fp16(code 2j * scale) |
    fp16(code 2j+1 * scale) << 16 (code 2j: the low nibble of byte j). Integer
    decode on every arch: an inline-assembly v_cvt_scale_pk8_f16_fp4 read wrong
    values on MI455X (the recipe's tools/v4/probe_fp4_gfx1250.py), and gfx1250's
    production FP4 reads go through the scaled WMMA (`nvfp4_scaled_dot`) anyway.
    ``unit`` is unused (the convert's E8M0 operand)."""
    s = (scale2 & 0xFFFF).to(tl.int16).to(tl.float16, bitcast=True)
    return (
        _scaled_f16_word(packed, 0, s),
        _scaled_f16_word(packed, 1, s),
        _scaled_f16_word(packed, 2, s),
        _scaled_f16_word(packed, 3, s),
    )


@triton.jit
def nvfp4_load_rows(
    payload_ptr,
    scale_ptr,
    payload_page_off,
    scale_page_off,
    rows,
    D: tl.constexpr,
):
    """Load rows ``rows`` of one page (all in bounds), dequantized: fp16
    ``[len(rows), D]`` in natural element order.

    Rows are contiguous (``D // 2`` payload and ``D // 16`` scale bytes; constant
    strides keep the loads vectorized); ``payload_page_off`` is a multiple of 4.
    The same values as ``nvfp4_load_even_odd`` (exact: e2m1 times an E4M3 scale
    fits fp16), without the even / odd split, so a dot over D is one ``tl.dot``
    against the query in its own order. A payload dword is decoded at once
    (`_fp4x8_scaled_f16_words`), its words the (2j, 2j+1) pairs a dot operand
    packs. On gfx1250 only the tiled decode scorer reads FP4 this way, and the
    FP4 policy never selects it there."""
    HALF: tl.constexpr = D // 2
    G: tl.constexpr = D // 16
    W: tl.constexpr = HALF // 4  # payload dwords a row, two a block
    n: tl.constexpr = rows.shape[0]
    scale = tl.load(
        scale_ptr + scale_page_off + rows[:, None] * G + tl.arange(0, G)[None, :]
    )
    words = tl.load(
        (payload_ptr + payload_page_off).to(tl.pointer_type(tl.int32))
        + rows[:, None] * W
        + tl.arange(0, W)[None, :]
    )
    s = scale.to(tl.float8e4nv, bitcast=True).to(tl.float16)
    sb = s.to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
    s2 = tl.reshape(tl.broadcast_to((sb | (sb << 16))[:, :, None], (n, G, 2)), (n, W))
    unit = tl.full((n, W), 0x7F7F7F7F, tl.int32)
    w0, w1, w2, w3 = _fp4x8_scaled_f16_words(words, s2, unit)
    # Word j of dword c holds elements 8c + 2j, 8c + 2j + 1.
    pairs = tl.reshape(tl.join(tl.join(w0, w2), tl.join(w1, w3)), (n, HALF))
    lo = (pairs & 0xFFFF).to(tl.int16).to(tl.float16, bitcast=True)
    hi = (pairs >> 16).to(tl.int16).to(tl.float16, bitcast=True)
    return tl.reshape(tl.join(lo, hi), (n, D))


# ---------------------------------------------------------------------------
# The scaled-WMMA read (gfx1250): the packed rows go to the matrix core as they
# are -- e2m1 codes with their E4M3 per-16 scales, one
# v_wmma_scale16_f32_16x16x128_f8f6f4 over the whole 128-dim key -- so nothing
# is decoded in registers. The other operand must be an f8f6f4 format: the
# query in E4M3.
# ---------------------------------------------------------------------------
# Whether the FP4 index scorers read through the scaled WMMA (gfx1250).
NVFP4_SCALED_DOT = tl.constexpr(get_gfx() == "gfx1250")
# E8M0 2^-4: the scale of the query's low part (`nvfp4_split_query`).
_Q_LO_E8M0 = tl.constexpr(127 - 4)
# E4M3's largest finite value: the query is clamped to it before rounding.
_E4M3_MAX = tl.constexpr(448.0)


@triton.jit
def nvfp4_load_packed(
    payload_ptr,
    scale_ptr,
    payload_page_off,
    scale_page_off,
    rows,
    D: tl.constexpr,
):
    """Rows ``rows`` of one page as stored: payload uint8 ``[n, D // 2]`` (code
    2j in the low nibble of byte j) and E4M3 scales ``[n, D // 16]``."""
    HALF: tl.constexpr = D // 2
    G: tl.constexpr = D // 16
    payload = tl.load(
        payload_ptr
        + payload_page_off
        + rows[:, None] * HALF
        + tl.arange(0, HALF)[None, :]
    )
    scale = tl.load(
        scale_ptr + scale_page_off + rows[:, None] * G + tl.arange(0, G)[None, :]
    )
    return payload, scale.to(tl.float8e4nv, bitcast=True)


@triton.jit
def nvfp4_split_query(q):
    """The query ``[N, D]`` as two E4M3 terms, q ~= hi + lo * 2^-4: hi = e4m3(q),
    lo = e4m3((q - hi) * 2^4). Together about 8 significant bits (bf16 has 8),
    so a two-term scaled dot scores nearly as the bf16 query does; q is clamped
    to E4M3's range first (an index query is far inside it). Returns (hi, lo,
    lo's E8M0 scale ``[N, D // 16]``)."""
    N: tl.constexpr = q.shape[0]
    D: tl.constexpr = q.shape[1]
    hi = nvfp4_query_e4m3(q)
    q32 = tl.clamp(q.to(tl.float32), -_E4M3_MAX, _E4M3_MAX)
    lo = ((q32 - hi.to(tl.float32)) * 16.0).to(tl.float8e4nv)
    lo_scale = tl.full((N, D // 16), _Q_LO_E8M0, tl.uint8)
    return hi, lo, lo_scale


@triton.jit
def nvfp4_query_e4m3(q):
    """The query rounded to E4M3 (clamped to its range first)."""
    return tl.clamp(q.to(tl.float32), -_E4M3_MAX, _E4M3_MAX).to(tl.float8e4nv)


@triton.jit
def nvfp4_scaled_dot(q_hi, q_lo, q_lo_scale, payload, scale):
    """``[N, n]`` fp32 = (hi + lo * 2^-4) . dequant(rows)^T, the rows exact: two
    scaled dots into one accumulator, hi first. ``q_lo`` None: hi alone (the
    prefill read, whose FP8 twin rounds the query to E4M3 the same way)."""
    k = tl.trans(payload)
    acc = tl.dot_scaled(q_hi, None, "e4m3", k, scale, "e2m1")
    if q_lo is not None:
        acc = tl.dot_scaled(q_lo, q_lo_scale, "e4m3", k, scale, "e2m1", acc=acc)
    return acc


@triton.jit
def nvfp4_block_max(dot, block_start, offs, cutoff, min_cutoff, scale):
    """``[N]`` max over one block's key positions ``block_start + offs`` of
    ``dot`` ``[N, len(offs)]``, causal (position < the row's ``cutoff``), times
    ``scale`` > 0 -- after the max: rounding is monotone, so max(x) * s rounds to
    max(x * s), the same bits as scaling every product, one multiply a row
    instead of one an element.

    ``min_cutoff``: a scalar no greater than any row's ``cutoff``. A block that
    ends at or before it is visible to every row, so its mask is all-true and is
    skipped: on gfx1250 the mask is ~330 of the 32-row scorer's ~700
    instructions a block, and only a request's last block or two need it."""
    BLOCK: tl.constexpr = offs.shape[0]
    if block_start + BLOCK > min_cutoff:
        dot = tl.where(
            (block_start + offs)[None, :] < cutoff[:, None], dot, float("-inf")
        )
    return tl.max(dot, axis=1) * scale
