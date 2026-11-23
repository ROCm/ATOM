# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""A row's temperature draw over the vocab as an exponential race (Gumbel-max):
id v's score b_v log2(e) / T - log2(E_v), b_v its bf16 logit and E_v an
exponential variable from the id's Philox word; the row's argmax score is one
draw of softmax(b / T).

The words: Philox2x32-10 under key ``seed0`` at counter (id / 2 | row <<
``counter_shift``, tag ^ ``seed1``), one word an id (the even id the first).
The tag is new every step and the same on every TP rank, so every rank draws
the same words and picks the same id (the hi head's counter, ``hi/blocks``).
"""

import flydsl.expr as fx
from flydsl.expr import math as fmath
from flydsl.expr import range_constexpr

from atom.mono.device.ops import LOG2E, bf16_round, hw_log2

PHILOX_M, PHILOX_W = 0xD256D193, 0x9E3779B9 - 2**32
# the widest vocab the counter's first word holds beside the row
MAX_VOCAB = 2**18


def counter_shift(vocab: int) -> int:
    """The row's shift in the counter's first word: ids / 2 below it."""
    assert vocab <= MAX_VOCAB, f"race: a vocab of {vocab} ids past {MAX_VOCAB}"
    return max(16, (vocab - 1).bit_length() - 1)


def seed_words(seed: int):
    """A 64-bit seed's (low, high) words as Int32 constants (``seed0``,
    ``seed1``)."""
    words = [(seed >> s) & 0xFFFFFFFF for s in (0, 32)]
    return tuple(fx.Int32(w - (1 << 32) if w >> 31 else w) for w in words)


def philox(c0, c1, key):
    """Philox2x32-10 of counter (c0, c1) under ``key`` -> two uniform words."""
    for _ in range_constexpr(10):
        prod = (fx.Int64(c0) & fx.Int64(0xFFFFFFFF)) * fx.Int64(PHILOX_M)
        hi, lo = fx.Int32(prod >> 32), fx.Int32(prod)
        c0, c1 = hi ^ key ^ c1, lo
        key = key + fx.Int32(PHILOX_W)
    return c0, c1


def _fma(a, b, c):
    return fmath.fma(fx.Float32(a), fx.Float32(b), fx.Float32(c))


def exp_var(word):
    """E = -log2 U (an exponential variable, in log2 units), U = (k + 1/2)
    2^-24 for the word's top 24 bits k: in (0, 1) exactly. The upper half
    works from x = 1 - U (exact, below 1/2): -log2(1 - x) by its series below
    2^-5, else the log of 1 - x."""
    kk = (word >> 8) & 0xFFFFFF
    up = kk >= fx.Int32(1 << 23)
    x = (
        up.select(fx.Int32(0xFFFFFF) - kk, kk).to(fx.Float32) + fx.Float32(0.5)
    ) * fx.Float32(2.0**-24)
    lg = -hw_log2(up.select(fx.Float32(1.0) - x, x))
    ser = _fma(x, _fma(x, _fma(x, _fma(x, fx.Float32(0.2), fx.Float32(0.25)),
                               fx.Float32(1 / 3)), fx.Float32(0.5)), fx.Float32(1.0))  # fmt: skip
    return (up & (x < fx.Float32(2.0**-5))).select(x * ser * fx.Float32(LOG2E), lg)


def race_scores(vals, gid, row, c, tag, seed0, seed1, shift: int):
    """The race scores of a row's 4 consecutive ids ``gid`` .. + 3 (``gid``
    even) of fp32 logits ``vals``, ``c`` = log2(e) / T."""
    words = []
    for h in range_constexpr(2):
        words += list(philox(((gid >> 1) + h) | (row << shift), tag ^ seed1, seed0))
    return [
        _fma(bf16_round(vals[e]), c, -hw_log2(exp_var(words[e])))
        for e in range_constexpr(4)
    ]
