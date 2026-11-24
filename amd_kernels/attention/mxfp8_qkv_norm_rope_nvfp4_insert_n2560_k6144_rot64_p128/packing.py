# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Pack four or eight rows of four FP4 nibbles into four token-major page words.

Each row contains four nibbles in its low 16 bits. Pair rows, exchange odd nibbles to make dimension bytes,
then transpose those bytes. The four output words contain the same nibbles as inserting each row's bits one row
at a time.
"""

import flydsl.expr as fx


def _v_pair(a, b):
    """Two rows -> four bytes: dimension order 0, 2, 1, 3, two token nibbles per byte."""
    value = a | (b << 16)
    exchange = (value ^ (value >> 12)) & 0xF0F0
    return value ^ exchange ^ (exchange << 12)


def _v_perm(hi, lo, select):
    """Select four bytes from (hi:lo); the native byte-permute instruction implements the exact integer mapping."""
    return fx.Int32(fx.rocdl.perm_b32(fx.Int32(hi), fx.Int32(lo), fx.Int32(select)))


def _v_words(nibbles):
    """Return four words, each containing four/eight tokens of one dimension (high bits zero for four tokens)."""
    pairs = [_v_pair(nibbles[i], nibbles[i + 1]) for i in range(0, len(nibbles), 2)]
    x = _v_perm(pairs[1], pairs[0], 0x06020400)
    y = _v_perm(pairs[1], pairs[0], 0x07030501)
    if len(nibbles) == 4:
        return [x & 0xFFFF, (x >> 16) & 0xFFFF, y & 0xFFFF, (y >> 16) & 0xFFFF]
    u = _v_perm(pairs[3], pairs[2], 0x06020400)
    v = _v_perm(pairs[3], pairs[2], 0x07030501)
    return [_v_perm(u, x, 0x05040100), _v_perm(u, x, 0x07060302),
            _v_perm(v, y, 0x05040100), _v_perm(v, y, 0x07060302)]


def _v_expand(nibbles):
    """Four low-16-bit nibbles -> the low nibble of each byte, before the LDS join."""
    value = (nibbles | (nibbles << 8)) & 0x00FF00FF
    return (value | (value << 4)) & 0x0F0F0F0F


def _v_expanded_words(nibbles):
    """Four/eight expanded rows -> page words; each producer has already expanded its own row."""
    pairs = [nibbles[i] | (nibbles[i + 1] << 4) for i in range(0, len(nibbles), 2)]
    x = _v_perm(pairs[1], pairs[0], 0x05010400)
    y = _v_perm(pairs[1], pairs[0], 0x07030602)
    if len(nibbles) == 4:
        return [x & 0xFFFF, (x >> 16) & 0xFFFF, y & 0xFFFF, (y >> 16) & 0xFFFF]
    u = _v_perm(pairs[3], pairs[2], 0x05010400)
    v = _v_perm(pairs[3], pairs[2], 0x07030602)
    return [_v_perm(u, x, 0x05040100), _v_perm(u, x, 0x07060302),
            _v_perm(v, y, 0x05040100), _v_perm(v, y, 0x07060302)]
