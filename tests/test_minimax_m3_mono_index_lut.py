# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU proof for the index scorer's e4m3 table widen (``fp4_dword_e4m3x8_lut``): mono's
original widen (``nvfp4_dword_e4m3x8``: cvt at scale 1.0, f32 multiply, clamp, e4m3
convert) and the table + v_perm widen both equal e4m3_rn_satfinite(e2m1 * scale) for
every e2m1 code at every nibble position and every positive e4m3 scale byte
(0x00..0x7E; the contract has no 0x7F), emulated bit for bit."""

import struct

import numpy as np
import pytest

torch = pytest.importorskip("torch")

E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
F32 = np.float32


def e4m3_value(byte):
    sign = -1.0 if byte & 0x80 else 1.0
    e, m = (byte >> 3) & 15, byte & 7
    return sign * (m / 8 * 2.0**-6 if e == 0 else (1 + m / 8) * 2.0 ** (e - 7))


def rn_satfinite(x):
    """Independent reference: the nearest finite e4m3 magnitude (ties to the even
    code), saturating at 448, the sign kept."""
    sign = 0x80 if np.signbit(x) else 0
    a = abs(float(x))
    if a >= 448.0:
        return sign | 0x7E
    best = min(range(0x7F), key=lambda b: (abs(e4m3_value(b) - a), b % 2, b))
    return sign | best


def cvt_fp8(x):
    """v_cvt_pk_fp8_f32 (RNE) of an f32 in range: torch's cast."""
    t = torch.tensor([x], dtype=torch.float32)
    return int(t.to(torch.float8_e4m3fn).view(torch.uint8)[0])


def mono_widen(word, scale_byte):
    """``nvfp4_dword_e4m3x8``: eight e4m3 bytes in element order."""
    sc = F32(e4m3_value(scale_byte))
    out = []
    for n in range(8):
        c = (word >> (4 * n)) & 15
        v = F32(E2M1[c & 7]) * sc * F32(-1 if c & 8 else 1)
        out.append(cvt_fp8(float(min(max(v, F32(-448)), F32(448)))))
    return out


def scale_f32(byte):
    """``e4m3_lut_row``'s scale (the f32 of an e4m3 byte)."""
    e, m = (byte >> 3) & 15, byte & 7
    if e == 0:
        return F32(m) * F32(2.0**-9)
    return struct.unpack("f", struct.pack("I", ((e + 120) << 23) | (m << 20)))[0]


def table_words(scale):
    b = [cvt_fp8(float(min(F32(m) * scale, F32(448)))) for m in E2M1]
    return (
        b[0] | b[1] << 8 | b[2] << 16 | b[3] << 24,
        b[4] | b[5] << 8 | b[6] << 16 | b[7] << 24,
    )


def perm_b32(s0, s1, sel):
    """v_perm_b32 with selector bytes < 8: bytes 0..3 are s1's, 4..7 s0's."""
    src = [(w >> (8 * i)) & 255 for w in (s1, s0) for i in range(4)]
    return sum(src[(sel >> (8 * j)) & 255] << (8 * j) for j in range(4))


def lut_widen(word, scale_byte):
    """``fp4_dword_e4m3x8_lut``: eight e4m3 bytes, even elements first (as the
    kernel emits them), put back in element order."""
    t_lo, t_hi = table_words(scale_f32(scale_byte))
    ev = perm_b32(t_hi, t_lo, word & 0x07070707) | ((word << 4) & 0x80808080)
    od = perm_b32(t_hi, t_lo, (word >> 4) & 0x07070707) | (word & 0x80808080)
    eb = [(ev >> (8 * i)) & 255 for i in range(4)]
    ob = [(od >> (8 * i)) & 255 for i in range(4)]
    return [x for pair in zip(eb, ob) for x in pair]


def test_both_widens_equal_e4m3_rn_satfinite_for_every_code_and_scale():
    rng = np.random.default_rng(0)
    n = bad = 0
    for sb in range(0x7F):
        sv = e4m3_value(sb)
        words = [0x76543210, 0xFEDCBA98, 0x89ABCDEF, 0x01234567]
        for p in range(8):  # every code at every nibble position, random elsewhere
            for c in range(16):
                w = int(rng.integers(0, 2**32)) & ~(15 << (4 * p)) | (c << (4 * p))
                words.append(w & 0xFFFFFFFF)
        for w in words:
            lut, mono = lut_widen(w, sb), mono_widen(w, sb)
            for k in range(8):
                c = (w >> (4 * k)) & 15
                x = (-1 if c & 8 else 1) * E2M1[c & 7] * sv  # exact in f64
                want = rn_satfinite(np.float64(x))
                assert want == cvt_fp8(max(min(x, 448.0), -448.0))  # the oracles agree
                bad += (lut[k] != want) + (mono[k] != want)
                n += 1
    assert n == 127 * (4 + 8 * 16) * 8 and bad == 0
