"""P2 numerics: MXFP4 weights x MXFP8 activations on the gfx950 scaled MFMA.

P2 names that pairing, the format of the decode GEMMs here. Weights are MXFP4
(E2M1). Activations are MXFP8 (E4M3). Both carry one E8M0 scale per 32 values
along K. The GEMMs run ``v_mfma_scale_f32_16x16x128_f8f6f4``, which applies
both scales in hardware, accumulating in fp32.

Activation quantizer::

    amax  = max |x| over the 32 bf16 values
    ratio = fl(amax * fl(1/448))                    # fl(1/448) = 0x3B124925
    E     = expbits(ratio) + (mantissa(ratio) != 0), clamped to [1, 254]
    q     = e4m3fn_rne(x / 2^(E - 127))             # never saturates

The host functions here dequantize stored weights and repack their scales;
the device functions quantize activations and run the MFMA inside a FlyDSL
kernel.
"""

import flydsl.expr as fx
import torch
from flydsl.expr import rocdl
from flydsl.expr.typing import T, as_ir_value

RATIO_BITS = 0x3B124925  # fl(1/448)
SCALE_ONE = 127  # E8M0 identity; also the filler for unused scale bytes


# ----------------------------------------------------------------- host


_FP4_VALUES = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)


def e8m0_to_float(scale):
    """E8M0 exponent bytes -> fp32 powers of two (FlyDSL ``kernels/monokernel/formats.py``)."""
    exponent = scale.view(torch.uint8)
    bits = exponent.to(torch.int32) << 23
    bits = torch.where(exponent == 0, torch.full_like(bits, 0x00400000), bits)
    bits = torch.where(exponent == 0xFF, torch.full_like(bits, 0x7F800001), bits)
    return bits.view(torch.float32)


def dequantize_mxfp4(q, scale):
    """Row-major packed MXFP4 codes + per-1x32 E8M0 scales -> fp32
    (FlyDSL ``kernels/monokernel/formats.py``)."""
    q = q.view(torch.uint8)
    codes = q.repeat_interleave(2, dim=-1)
    codes[..., 0::2] &= 0xF
    codes[..., 1::2] >>= 4
    values = torch.tensor(_FP4_VALUES, dtype=torch.float32, device=q.device)
    return values[codes.long()] * e8m0_to_float(scale).repeat_interleave(32, dim=-1)


def k_iters(K):
    """128-K steps of a stored weight; K is padded to a multiple of 256."""
    return 2 * ((K + 255) // 256)


def scales_k4(small, N, K):
    """Repack ``small`` scales so one dword per lane covers 4 consecutive k_iters.

    Layout ``[N/16][k_iters/4][64 lanes][4]``: the scale of n-block ``nb``,
    k_iter ``ki`` for lane ``l`` is byte ``ki % 4`` of dword
    ``(nb * (k_iters/4) + ki // 4) * 64 + l``, so the MFMA's compile-time
    ``opsel`` picks the k_iter. k_iters are padded to a multiple of 4 with 127.
    """
    nb, ki = (N + 15) // 16, k_iters(K)
    ki4 = (ki + 3) // 4 * 4
    src = small.view(nb, ki, 64)
    out = torch.full((nb, ki4, 64), SCALE_ONE, dtype=torch.uint8, device=small.device)
    out[:, :ki] = src
    return out.view(nb, ki4 // 4, 4, 64).permute(0, 1, 3, 2).contiguous().view(-1)


def scales_chain(small, N, K, parts):
    """Repack ``small`` scales for split-K chains: chain ``p`` of n-block ``nb``
    runs k_iters ``p, p + parts, p + 2 parts, ...``, and lane ``l``'s scales for
    it are contiguous bytes, so a few dwordx4 loads fetch the whole chain.

    Layout ``[N/16][parts][blocks][64 lanes][16 bytes]``: step ``i`` of the chain
    is byte ``i % 16`` of block ``i // 16`` (the MFMA's ``opsel`` picks the byte
    within its dword); steps past the chain are 127.
    """
    nb, ki = (N + 15) // 16, k_iters(K)
    assert ki % parts == 0
    steps = ki // parts
    blocks = (steps + 15) // 16
    src = small.view(nb, steps, parts, 64).permute(0, 2, 1, 3)
    out = torch.full(
        (nb, parts, blocks * 16, 64), SCALE_ONE, dtype=torch.uint8, device=small.device
    )
    out[:, :, :steps] = src
    return (
        out.view(nb, parts, blocks, 16, 64).permute(0, 1, 2, 4, 3).contiguous().view(-1)
    )


# --------------------------------------------------------------- device


def e8m0_fp8(amax):
    """``(E, scale)`` for one 32-value block with absolute maximum ``amax`` (f32)."""
    bits = (fx.Float32(amax) * fx.Int32(RATIO_BITS).bitcast(fx.Float32)).bitcast(
        fx.Int32
    )
    exponent = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).select(
        fx.Int32(1), fx.Int32(0)
    )
    exponent = fx.min(fx.max(exponent, fx.Int32(1)), fx.Int32(254))
    return exponent, (exponent << 23).bitcast(fx.Float32)


def fp8_word(v0, v1, v2, v3, scale):
    """Four f32 (bf16 values) -> one dword of E4M3 bytes, ``v0`` in the low byte."""
    pair = T.vec(2, T.i16)
    word = fx.Vector.filled(2, 0, fx.Int16)
    for half, (lo, hi) in enumerate(((v0, v1), (v2, v3))):
        word = rocdl.cvt_scalef32_pk_fp8_f32(
            pair,
            as_ir_value(word),
            as_ir_value(fx.Float32(lo)),
            as_ir_value(fx.Float32(hi)),
            as_ir_value(scale),
            bool(half),
        )
    return fx.Vector(word).bitcast(fx.Int32)[0]


def mfma_p2(acc, fp8_words, fp4_words, fp8_scale, fp4_scale, fp8_sel=0, fp4_sel=0):
    """``acc += A(E4M3, 16 x 128) @ B(E2M1, 128 x 16)`` with E8M0 scales.

    Lane ``l`` holds row/column ``l % 16``; with ``g = l // 16``:

    - ``fp8_words`` (v8i32): k = 16g..16g+15 in dwords 0-3 and k = 64+16g..64+16g+15
      in dwords 4-7, i.e. two 16-byte halves 64 bytes apart, not one 32-byte run.
    - ``fp4_words`` (v4i32): k = 32g..32g+31, the stored weight block's lane ``l``.
    - scales: K-block ``g`` (k = 32g..32g+31) of that row/column, byte ``*_sel``
      of ``*_scale``.

    Result lane ``l`` element ``i`` is C[4 * g + i][l % 16].
    """
    return fx.Vector(
        rocdl.mfma_scale_f32_16x16x128_f8f6f4(
            T.vec(4, T.f32),
            [
                fp8_words,
                fp4_words,
                acc,
                0,
                4,
                fp8_sel,
                fx.Int32(fp8_scale),
                fp4_sel,
                fx.Int32(fp4_scale),
            ],
        )
    )
