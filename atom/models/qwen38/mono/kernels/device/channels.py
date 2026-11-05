# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Device-side channel access (formats: ``blocks.base``).

Thin, format-aware wrappers over the mailbox so blocks agree on layouts. ``a``
is the channel's ``Addr`` (``k.mb[name]``; MXFP8 channels have ``name.q`` and
``name.s``); indices are element indices within the row-major [rows, width]
tensor unless noted.
"""

import flydsl.expr as fx

from atom.models.qwen38.mono.kernels.device import mailbox as mb
from atom.models.qwen38.mono.kernels.device.ops import bf2_f32

# ---- BF16: two elements per pair


def put_bf16(a, e, vals, tag):
    """Elements ``e .. e + len(vals)`` (2 or 4; ``e`` a multiple of len(vals))."""
    mb.put_bf(a, e, vals, tag)


def poll_bf16(specs, tag):
    """[(Addr, element e (even), n elements in {2, 4})] -> list of f32 lists."""
    got = mb.poll([(a, e // 2, n // 2) for a, e, n in specs], tag)
    out = []
    for words in got:
        vals = []
        for w in words:
            lo, hi = bf2_f32(w)
            vals += [lo, hi]
        out.append(vals)
    return out


# ---- F32: one element per pair


def put_f32(a, e, v, tag):
    mb.put(a, e, v, tag)


def poll_f32(specs, tag):
    """[(Addr, element e)] -> list of f32."""
    return mb.poll_f32(specs, tag)


# ---- I32: one int32 per pair


def put_i32(a, e, v, tag):
    """Element ``e`` (row-major) := ``v`` (Int32 bits; no float round trip)."""
    mb.put_words(a, e, [fx.Int32(v)], tag)


def poll_i32(specs, tag):
    """[(Addr, element e)] -> list of Int32."""
    return [v[0] for v in mb.poll([(a, e, 1) for a, e in specs], tag)]


# ---- MXFP8: four E4M3 per pair in ``.q``, one E8M0 code per pair in ``.s``


def put_mxfp8_word(aq, w, word, tag):
    """FP8 word ``w`` (elements 4w .. 4w + 3, row-major) := ``word`` (Int32)."""
    mb.put_words(aq, w, [word], tag)


def put_mxfp8_scale(as_, g, code, tag):
    """Scale code of 32-group ``g`` (row-major over rows) := ``code`` (Int32, low byte)."""
    mb.put_words(as_, g, [fx.Int32(code)], tag)


def put_words(a, i, words, tag):
    """Raw words (1 or 2) at pair index ``i`` (``mailbox.put_words``)."""
    mb.put_words(a, i, list(words), tag)


def poll_words(specs, tag):
    """[(Addr, pair index, npairs in {1, 2})] -> raw Int32 words (fp8 words, codes)."""
    return mb.poll(specs, tag)
