# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The draft heads' kernels: a bounded Group64 W4/A8 screen of the vocabulary, refined on the original BF16 rows.

The head weight is packed once (``prepare_weight``: signed 4-bit values ``q + 8`` in 32-bit words, a scale, an error
norm and a quantized norm per 64-column group, the row norm; ``aggregate_weight_certificate``: the per-row whole-vector
error / quantized norms). A step quantizes its input per group to INT8 (``quant_groups``), aggregates whole-vector
input certificates (``aggregate_input_certificate``), screens every row with two dot4 instructions per word
(``screen_global_certificate``: an approximate score and a certified error bound), takes the best screened row's exact
BF16 product as the threshold anchor (``anchor_lower``), recomputes the rows whose interval reaches the threshold on
the original BF16 weight (``refine``) and returns the argmax (aiter ``topk_select``, ties low): the BF16 head's argmax.
``*_rows`` are the same arithmetic with a row axis (one pass over the packed weight for 2 or 4 rows). Norm
certificates and interval reductions are rounded outwards by gamma bounds (arithmetic accounting, not an L2 cutoff).
"""

import torch
import triton
import triton.language as tl

GROUP = 64
BLOCK_N = 32
BLOCK_K = 512


@triton.jit
def quant_groups(X, QEven, QOdd, Scale, Stats, QSum, K: tl.constexpr, G: tl.constexpr):
    group = tl.program_id(0)
    k = group * G + tl.arange(0, G)
    x = tl.load(X + k).to(tl.float32)
    scale = tl.max(tl.abs(x), 0) / 127.0
    inv = tl.where(scale > 0.0, 1.0 / scale, 0.0)
    q = tl.floor(tl.minimum(tl.maximum(x * inv, -127.0), 127.0) + 0.5).to(tl.int32)
    restored = q.to(tl.float32) * scale
    err = x - restored
    # Bounds refer to exact real q*stored_scale. The final additions cover
    # FP32 restoration/subtraction, and gamma covers norm arithmetic.
    u = 5.960464477539063e-8
    gamma = (2 * G + 16) * u / (1.0 - (2 * G + 16) * u)
    inflate = (1.0 + gamma) / (1.0 - gamma)
    xn = tl.sqrt(tl.sum(x * x, 0)) * inflate
    qn = tl.sqrt(tl.sum(restored * restored, 0)) * inflate / (1.0 - u)
    en = tl.sqrt(tl.sum(err * err, 0)) * inflate + (u / (1.0 - u)) * (xn + qn)
    tl.store(Scale + group, scale)
    tl.store(Stats + group * 3, xn)
    tl.store(Stats + group * 3 + 1, en)
    tl.store(Stats + group * 3 + 2, qn)
    tl.store(QSum + group, tl.sum(q, 0))
    # Positive U4 weight nibbles represent signed q_w+8. Input even/odd
    # coordinates are packed into signed INT8 lanes for two dot4 instructions.
    packed = q.to(tl.uint32).reshape((G // 8, 4, 2)) & 255
    shift = tl.arange(0, 4)[None, :, None] * 8
    words = tl.sum(packed << shift, 1)
    lo = tl.sum(tl.where(tl.arange(0, 2)[None, :] == 0, words, 0), 1)
    hi = tl.sum(tl.where(tl.arange(0, 2)[None, :] == 1, words, 0), 1)
    index = group * (G // 8) + tl.arange(0, G // 8)
    tl.store(QEven + index, lo)
    tl.store(QOdd + index, hi)


@triton.jit
def aggregate_input_certificate(
    X,
    XScale,
    XStats,
    Global,
    K: tl.constexpr,
    Groups: tl.constexpr,
    BX: tl.constexpr,
    BG: tl.constexpr,
):
    k = tl.arange(0, BX)
    bits = tl.load(X + k, k < K, 0).to(tl.uint16, bitcast=True)
    all_finite = tl.sum(((bits & 0x7F80) == 0x7F80).to(tl.int32), 0) == 0
    g = tl.arange(0, BG)
    scale = tl.load(XScale + g, g < Groups, 0)
    xn = tl.load(XStats + g * 3, g < Groups, 0)
    en = tl.load(XStats + g * 3 + 1, g < Groups, 0)
    qn = tl.load(XStats + g * 3 + 2, g < Groups, 0)
    # This admission keeps all nonzero norm squares normal and sums finite.
    # Zero groups and arbitrary finite extremes take all-row refinement.
    valid = (
        (scale >= 2.0**-40)
        & (scale <= 2.0**40)
        & (xn >= 2.0**-60)
        & (xn <= 2.0**55)
        & (en >= 2.0**-60)
        & (en <= 2.0**55)
        & (qn >= 2.0**-60)
        & (qn <= 2.0**55)
    )
    admitted = all_finite & (tl.sum(((g < Groups) & ~valid).to(tl.int32), 0) == 0)
    xn = tl.where((g < Groups) & valid, xn, 0.0)
    en = tl.where((g < Groups) & valid, en, 0.0)
    qn = tl.where((g < Groups) & valid, qn, 0.0)
    u = 5.960464477539063e-8
    gamma = (2 * Groups + 16) * u / (1.0 - (2 * Groups + 16) * u)
    inflate = (1.0 + gamma) / (1.0 - gamma)
    a = tl.sqrt(tl.sum(xn * xn, 0)) * inflate
    e = tl.sqrt(tl.sum(en * en, 0)) * inflate
    q = tl.sqrt(tl.sum(qn * qn, 0)) * inflate
    tl.store(Global, tl.where(admitted, a, float("inf")))
    tl.store(Global + 1, tl.where(admitted, e, float("inf")))
    tl.store(Global + 2, tl.where(admitted, q, float("inf")))


@triton.jit
def screen_global_certificate(
    QEven,
    QOdd,
    XScale,
    XGlobal,
    XSum,
    PackedW,
    WScale,
    WErrorNorm,
    WQuantNorm,
    WNorm,
    Y,
    Error,
    Lower,
    BestY,
    BestId,
    N: tl.constexpr,
    K: tl.constexpr,
    G: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    block = tl.program_id(0)
    n = block * BN + tl.arange(0, BN)
    word = tl.arange(0, BK // 8)
    group = tl.arange(0, BK // G)
    y = tl.full((BN,), 0.0, tl.float32)
    for chunk in range(K // BK):
        kw = chunk * (BK // 8) + word
        kg = chunk * (BK // G) + group
        p = tl.load(
            PackedW + n[:, None].to(tl.int64) * (K // 8) + kw[None, :],
            n[:, None] < N,
            0,
        )
        qe = tl.load(QEven + kw)
        qo = tl.load(QOdd + kw)
        even = p & 0x0F0F0F0F
        odd = (p >> 4) & 0x0F0F0F0F
        # Keep dot outputs distinct from every input register. Allowing reuse
        # corrupts this gfx950 fused kernel; the isolated arithmetic still
        # passes, so the constraint must also hold in the fused schedule.
        dots = tl.inline_asm_elementwise(
            "v_dot4_i32_i8 $0, $1, $2, $3;",
            constraints="=&v,v,v,v",
            args=[even, qe[None, :], tl.full((BN, BK // 8), 0, tl.int32)],
            dtype=tl.int32,
            is_pure=True,
            pack=1,
        )
        dots = tl.inline_asm_elementwise(
            "v_dot4_i32_i8 $0, $1, $2, $3;",
            constraints="=&v,v,v,v",
            args=[odd, qo[None, :], dots],
            dtype=tl.int32,
            is_pure=True,
            pack=1,
        )
        dots = tl.sum(dots.reshape((BN, BK // G, G // 8)), 2)
        qsum = tl.load(XSum + kg)
        dots -= 8 * qsum[None, :]
        offset = n[:, None].to(tl.int64) * (K // G) + kg[None, :]
        ws = tl.load(WScale + offset, n[:, None] < N, 0)
        xs = tl.load(XScale + kg)
        y += tl.sum((dots.to(tl.float32) * xs[None, :]) * ws, 1)
    # Norm certificates and interval reductions are rounded outwards by a
    # gamma bound; this is arithmetic accounting, not an L2 acceptance cutoff.
    u = 5.960464477539063e-8
    gamma = (K + 8) * u / (1.0 - (K + 8) * u)
    inflate = (1.0 + gamma) / (1.0 - gamma)
    xn = tl.load(XGlobal)
    en = tl.load(XGlobal + 1)
    qn = tl.load(XGlobal + 2)
    we = tl.load(WErrorNorm + n, n < N, 0)
    wq = tl.load(WQuantNorm + n, n < N, 0)
    wn = tl.load(WNorm + n, n < N, 0)
    quant_error = xn * we + en * wq
    qnorm_product = qn * wq
    dot_error = quant_error * inflate + gamma * (xn * wn + qnorm_product * inflate)
    approx = y.to(tl.bfloat16).to(tl.float32)
    error = (
        dot_error * (1.0 + 1.0 / 256.0) + tl.abs(approx) * (2.0 / 255.0) + 1.0e-6
    ) * inflate
    valid_input = (xn < float("inf")) & (en < float("inf")) & (qn < float("inf"))
    valid_weight = (
        (we >= 0.0)
        & (we <= 2.0**40)
        & (wq >= 0.0)
        & (wq <= 2.0**40)
        & (wn >= 0.0)
        & (wn <= 2.0**40)
    )
    finite = (
        valid_input
        & valid_weight
        & (tl.abs(approx) < float("inf"))
        & (error < float("inf"))
        & (error >= 0.0)
    )
    error = tl.where(finite, error, float("inf"))
    tl.store(Y + n, tl.where(finite, approx, 0.0), n < N)
    tl.store(Error + n, error, n < N)
    lower = tl.max(tl.where((n < N) & finite, approx - error, -float("inf")), 0)
    best = tl.max(tl.where((n < N) & finite, approx, -float("inf")), 0)
    ident = tl.min(tl.where((n < N) & finite & (approx == best), n, 2147483647), 0)
    tl.store(Lower + block, lower)
    tl.store(BestY + block, best)
    tl.store(BestId + block, ident)


@triton.jit
def anchor_lower(
    X,
    W,
    WNorm,
    Lower,
    BestY,
    BestId,
    Threshold,
    Anchor,
    AnchorValue,
    AnchorBound,
    N: tl.constexpr,
    K: tl.constexpr,
    G: tl.constexpr,
    Blocks: tl.constexpr,
    B: tl.constexpr,
    BK: tl.constexpr,
):
    i = tl.arange(0, B)
    lower = tl.max(tl.load(Lower + i, i < Blocks, -float("inf")), 0)
    best = tl.load(BestY + i, i < Blocks, -float("inf"))
    max_y = tl.max(best, 0)
    ids = tl.load(BestId + i, i < Blocks, 2147483647)
    chosen = tl.min(tl.where(best == max_y, ids, 2147483647), 0)
    valid = (chosen < N) & (tl.abs(max_y) < float("inf"))
    # No fixed candidate restriction: if an anchor is unavailable, retain
    # the ordinary max(lower) threshold. Unbounded intervals stay eligible.
    k = tl.arange(0, BK)
    x = tl.load(X + k, k < K, 0).to(tl.float32)
    w = tl.load(W + chosen.to(tl.int64) * K + k, valid & (k < K), 0).to(tl.float32)
    a = tl.sum(x * w, 0)
    u = 5.960464477539063e-8
    gamma = (K + 8) * u / (1.0 - (K + 8) * u)
    inflate = (1.0 + gamma) / (1.0 - gamma)
    xn = tl.sqrt(tl.sum(x * x, 0)) * inflate
    wn = tl.load(WNorm + chosen, valid, 0)
    # One FP32 allowance for this anchor reduction, one for Native, then
    # Native BF16 rounding. This does not require their reduction orders to match.
    d = (2.0 * gamma * xn * wn) * inflate
    scalar_roundoff = 8.0 * u * (tl.abs(a) + d + 1.0)
    certified = a - d - (tl.abs(a) + d) / 256.0 - scalar_roundoff - 1.0e-6
    usable = valid & (tl.abs(certified) < float("inf"))
    tl.store(Threshold, tl.maximum(lower, tl.where(usable, certified, -float("inf"))))
    tl.store(Anchor, tl.where(usable, chosen, -1))
    tl.store(AnchorValue, a)
    tl.store(AnchorBound, tl.where(usable, certified, -float("inf")))


@triton.jit
def refine(
    X,
    W,
    Y,
    Error,
    Threshold,
    Refined,
    Counts,
    N: tl.constexpr,
    K: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    block, row = tl.program_id(0), tl.program_id(1)
    n = block * BN + tl.arange(0, BN)
    y = tl.load(Y + row * N + n, n < N, 0).to(tl.float32)
    e = tl.load(Error + row * N + n, n < N, 0)
    threshold = tl.load(Threshold + row)
    selected = (n < N) & (y + e >= threshold)
    count = tl.sum(selected.to(tl.int32), 0)
    result = tl.full((BN,), -float("inf"), tl.float32)
    if count > 0:
        k = tl.arange(0, BK)
        acc = tl.full((BN,), 0.0, tl.float32)
        for i in range(tl.cdiv(K, BK)):
            kk = i * BK + k
            x = tl.load(X + row * K + kk, kk < K, 0).to(tl.float32)
            w = tl.load(
                W + n[:, None].to(tl.int64) * K + kk[None, :],
                selected[:, None] & (kk[None, :] < K),
                0,
            ).to(tl.float32)
            acc += tl.sum(w * x[None, :], 1)
        result = tl.where(selected, acc.to(tl.bfloat16).to(tl.float32), result)
    tl.store(Refined + row * N + n, result, n < N)
    tl.store(Counts + row * tl.cdiv(N, BN) + block, count)


@triton.jit
def quant_groups_rows(
    X, QEven, QOdd, Scale, Stats, QSum, K: tl.constexpr, G: tl.constexpr
):
    group, row = tl.program_id(0), tl.program_id(1)
    groups: tl.constexpr = K // G
    k = group * G + tl.arange(0, G)
    x = tl.load(X + row * K + k).to(tl.float32)
    scale = tl.max(tl.abs(x), 0) / 127.0
    inv = tl.where(scale > 0.0, 1.0 / scale, 0.0)
    q = tl.floor(tl.minimum(tl.maximum(x * inv, -127.0), 127.0) + 0.5).to(tl.int32)
    restored = q.to(tl.float32) * scale
    err = x - restored
    u = 5.960464477539063e-8
    gamma = (2 * G + 16) * u / (1.0 - (2 * G + 16) * u)
    inflate = (1.0 + gamma) / (1.0 - gamma)
    xn = tl.sqrt(tl.sum(x * x, 0)) * inflate
    qn = tl.sqrt(tl.sum(restored * restored, 0)) * inflate / (1.0 - u)
    en = tl.sqrt(tl.sum(err * err, 0)) * inflate + (u / (1.0 - u)) * (xn + qn)
    g = row * groups + group
    tl.store(Scale + g, scale)
    tl.store(Stats + g * 3, xn)
    tl.store(Stats + g * 3 + 1, en)
    tl.store(Stats + g * 3 + 2, qn)
    tl.store(QSum + g, tl.sum(q, 0))
    packed = q.to(tl.uint32).reshape((G // 8, 4, 2)) & 255
    shift = tl.arange(0, 4)[None, :, None] * 8
    words = tl.sum(packed << shift, 1)
    lo = tl.sum(tl.where(tl.arange(0, 2)[None, :] == 0, words, 0), 1)
    hi = tl.sum(tl.where(tl.arange(0, 2)[None, :] == 1, words, 0), 1)
    index = row * (K // 8) + group * (G // 8) + tl.arange(0, G // 8)
    tl.store(QEven + index, lo)
    tl.store(QOdd + index, hi)


@triton.jit
def aggregate_rows(
    X,
    XScale,
    XStats,
    Global,
    K: tl.constexpr,
    Groups: tl.constexpr,
    BX: tl.constexpr,
    BG: tl.constexpr,
):
    row = tl.program_id(0)
    k = tl.arange(0, BX)
    bits = tl.load(X + row * K + k, k < K, 0).to(tl.uint16, bitcast=True)
    all_finite = tl.sum(((bits & 0x7F80) == 0x7F80).to(tl.int32), 0) == 0
    g = tl.arange(0, BG)
    m = g < Groups
    scale = tl.load(XScale + row * Groups + g, m, 0)
    xn = tl.load(XStats + (row * Groups + g) * 3, m, 0)
    en = tl.load(XStats + (row * Groups + g) * 3 + 1, m, 0)
    qn = tl.load(XStats + (row * Groups + g) * 3 + 2, m, 0)
    valid = (
        (scale >= 2.0**-40)
        & (scale <= 2.0**40)
        & (xn >= 2.0**-60)
        & (xn <= 2.0**55)
        & (en >= 2.0**-60)
        & (en <= 2.0**55)
        & (qn >= 2.0**-60)
        & (qn <= 2.0**55)
    )
    admitted = all_finite & (tl.sum((m & ~valid).to(tl.int32), 0) == 0)
    xn = tl.where(m & valid, xn, 0.0)
    en = tl.where(m & valid, en, 0.0)
    qn = tl.where(m & valid, qn, 0.0)
    u = 5.960464477539063e-8
    gamma = (2 * Groups + 16) * u / (1.0 - (2 * Groups + 16) * u)
    inflate = (1.0 + gamma) / (1.0 - gamma)
    a = tl.sqrt(tl.sum(xn * xn, 0)) * inflate
    e = tl.sqrt(tl.sum(en * en, 0)) * inflate
    q = tl.sqrt(tl.sum(qn * qn, 0)) * inflate
    tl.store(Global + row * 3, tl.where(admitted, a, float("inf")))
    tl.store(Global + row * 3 + 1, tl.where(admitted, e, float("inf")))
    tl.store(Global + row * 3 + 2, tl.where(admitted, q, float("inf")))


@triton.jit
def screen_rows(
    QEven,
    QOdd,
    XScale,
    XGlobal,
    XSum,
    PackedW,
    WScale,
    WErrorNorm,
    WQuantNorm,
    WNorm,
    Y,
    Error,
    Lower,
    BestY,
    BestId,
    N: tl.constexpr,
    K: tl.constexpr,
    G: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    R: tl.constexpr,
):
    block = tl.program_id(0)
    blocks: tl.constexpr = (N + BN - 1) // BN
    groups: tl.constexpr = K // G
    n = block * BN + tl.arange(0, BN)
    r = tl.arange(0, R)
    word = tl.arange(0, BK // 8)
    group = tl.arange(0, BK // G)
    y = tl.full((R, BN), 0.0, tl.float32)
    for chunk in range(K // BK):
        kw = chunk * (BK // 8) + word
        kg = chunk * (BK // G) + group
        # once, every row
        p = tl.load(
            PackedW + n[:, None].to(tl.int64) * (K // 8) + kw[None, :],
            n[:, None] < N,
            0,
        )
        qe = tl.load(QEven + r[:, None] * (K // 8) + kw[None, :])  # [R, BK / 8]
        qo = tl.load(QOdd + r[:, None] * (K // 8) + kw[None, :])
        even = (p & 0x0F0F0F0F)[None, :, :]
        odd = ((p >> 4) & 0x0F0F0F0F)[None, :, :]
        zero = tl.full((R, BN, BK // 8), 0, tl.int32)
        dots = tl.inline_asm_elementwise(
            "v_dot4_i32_i8 $0, $1, $2, $3;",
            constraints="=&v,v,v,v",
            args=[even + zero.to(tl.uint32), qe[:, None, :] + zero.to(tl.uint32), zero],
            dtype=tl.int32,
            is_pure=True,
            pack=1,
        )
        dots = tl.inline_asm_elementwise(
            "v_dot4_i32_i8 $0, $1, $2, $3;",
            constraints="=&v,v,v,v",
            args=[odd + zero.to(tl.uint32), qo[:, None, :] + zero.to(tl.uint32), dots],
            dtype=tl.int32,
            is_pure=True,
            pack=1,
        )
        dots = tl.sum(dots.reshape((R, BN, BK // G, G // 8)), 3)  # [R, BN, BK / G]
        qsum = tl.load(XSum + r[:, None] * groups + kg[None, :])  # [R, BK / G]
        dots -= 8 * qsum[:, None, :]
        # [BN, BK / G]
        ws = tl.load(
            WScale + n[:, None].to(tl.int64) * groups + kg[None, :], n[:, None] < N, 0
        )
        xs = tl.load(XScale + r[:, None] * groups + kg[None, :])  # [R, BK / G]
        y += tl.sum((dots.to(tl.float32) * xs[:, None, :]) * ws[None, :, :], 2)
    u = 5.960464477539063e-8
    gamma = (K + 8) * u / (1.0 - (K + 8) * u)
    inflate = (1.0 + gamma) / (1.0 - gamma)
    xn = tl.load(XGlobal + r * 3)[:, None]
    en = tl.load(XGlobal + r * 3 + 1)[:, None]
    qn = tl.load(XGlobal + r * 3 + 2)[:, None]
    we = tl.load(WErrorNorm + n, n < N, 0)[None, :]
    wq = tl.load(WQuantNorm + n, n < N, 0)[None, :]
    wn = tl.load(WNorm + n, n < N, 0)[None, :]
    quant_error = xn * we + en * wq
    qnorm_product = qn * wq
    dot_error = quant_error * inflate + gamma * (xn * wn + qnorm_product * inflate)
    approx = y.to(tl.bfloat16).to(tl.float32)
    error = (
        dot_error * (1.0 + 1.0 / 256.0) + tl.abs(approx) * (2.0 / 255.0) + 1.0e-6
    ) * inflate
    valid_input = (xn < float("inf")) & (en < float("inf")) & (qn < float("inf"))
    valid_weight = (
        (we >= 0.0)
        & (we <= 2.0**40)
        & (wq >= 0.0)
        & (wq <= 2.0**40)
        & (wn >= 0.0)
        & (wn <= 2.0**40)
    )
    finite = (
        valid_input
        & valid_weight
        & (tl.abs(approx) < float("inf"))
        & (error < float("inf"))
        & (error >= 0.0)
    )
    error = tl.where(finite, error, float("inf"))
    live = (n < N)[None, :] & (r[:, None] >= 0)
    out = r[:, None] * N + n[None, :]
    tl.store(Y + out, tl.where(finite, approx, 0.0), live)
    tl.store(Error + out, error, live)
    ok = live & finite
    lower = tl.max(tl.where(ok, approx - error, -float("inf")), 1)
    best = tl.max(tl.where(ok, approx, -float("inf")), 1)
    ident = tl.min(tl.where(ok & (approx == best[:, None]), n[None, :], 2147483647), 1)
    tl.store(Lower + r * blocks + block, lower)
    tl.store(BestY + r * blocks + block, best)
    tl.store(BestId + r * blocks + block, ident)


@triton.jit
def anchor_rows(
    X,
    W,
    WNorm,
    Lower,
    BestY,
    BestId,
    Threshold,
    Anchor,
    AnchorValue,
    AnchorBound,
    N: tl.constexpr,
    K: tl.constexpr,
    Blocks: tl.constexpr,
    B: tl.constexpr,
    BK: tl.constexpr,
):
    row = tl.program_id(0)
    i = tl.arange(0, B)
    lower = tl.max(tl.load(Lower + row * Blocks + i, i < Blocks, -float("inf")), 0)
    best = tl.load(BestY + row * Blocks + i, i < Blocks, -float("inf"))
    max_y = tl.max(best, 0)
    ids = tl.load(BestId + row * Blocks + i, i < Blocks, 2147483647)
    chosen = tl.min(tl.where(best == max_y, ids, 2147483647), 0)
    valid = (chosen < N) & (tl.abs(max_y) < float("inf"))
    k = tl.arange(0, BK)
    x = tl.load(X + row * K + k, k < K, 0).to(tl.float32)
    w = tl.load(W + chosen.to(tl.int64) * K + k, valid & (k < K), 0).to(tl.float32)
    a = tl.sum(x * w, 0)
    u = 5.960464477539063e-8
    gamma = (K + 8) * u / (1.0 - (K + 8) * u)
    inflate = (1.0 + gamma) / (1.0 - gamma)
    xn = tl.sqrt(tl.sum(x * x, 0)) * inflate
    wn = tl.load(WNorm + chosen, valid, 0)
    d = (2.0 * gamma * xn * wn) * inflate
    scalar_roundoff = 8.0 * u * (tl.abs(a) + d + 1.0)
    certified = a - d - (tl.abs(a) + d) / 256.0 - scalar_roundoff - 1.0e-6
    usable = valid & (tl.abs(certified) < float("inf"))
    tl.store(
        Threshold + row, tl.maximum(lower, tl.where(usable, certified, -float("inf")))
    )
    tl.store(Anchor + row, tl.where(usable, chosen, -1))
    tl.store(AnchorValue + row, a)
    tl.store(AnchorBound + row, tl.where(usable, certified, -float("inf")))


def prepare_weight(weight):
    n, k = weight.shape
    assert weight.dtype == torch.bfloat16 and weight.is_contiguous()
    assert n % BLOCK_N == 0 and k % BLOCK_K == 0
    groups = k // GROUP
    packed = torch.empty((n, k // 8), device=weight.device, dtype=torch.uint32)
    scale = torch.empty((n, groups), device=weight.device, dtype=torch.float32)
    error = torch.empty_like(scale)
    qnorm = torch.empty_like(scale)
    norm = torch.empty((n,), device=weight.device, dtype=torch.float32)

    def upper(x):
        value = x.float()
        return torch.nextafter(value, torch.full_like(value, float("inf")))

    for begin in range(0, n, 1024):
        stop = min(n, begin + 1024)
        w = weight[begin:stop].float().reshape(-1, groups, GROUP)
        assert bool(torch.isfinite(w).all()), "finite checkpoint head weight required"
        s = w.abs().amax(-1) / 7.0
        q = (w / torch.where(s > 0, s, 1.0)[..., None]).round().clamp(-7, 7)
        restored = q.double() * s.double()[..., None]
        scale[begin:stop] = s
        error[begin:stop] = upper((w.double() - restored).norm(dim=-1))
        qnorm[begin:stop] = upper(restored.norm(dim=-1))
        norm[begin:stop] = upper(w.double().flatten(1).norm(dim=-1))
        unsigned = (q.to(torch.int64) + 8).reshape(stop - begin, k // 8, 8)
        shifts = torch.arange(8, device=weight.device, dtype=torch.int64) * 4
        packed[begin:stop] = (unsigned << shifts).sum(-1).to(torch.uint32)
    return packed, scale, error, qnorm, norm


def aggregate_weight_certificate(group_certificate):
    # Input FP32 certificates are immutable upper bounds. FP64 squares are
    # exact, and 2^-40 dominates gamma_128 for the sum/sqrt/product. The final
    # nextafter covers conversion to FP32 and its rounded midpoint.
    value = group_certificate.double().square().sum(dim=1).sqrt() * (1.0 + 2.0**-40)
    value = torch.where(
        torch.isfinite(value) & (value >= 0),
        value,
        torch.full_like(value, float("inf")),
    ).float()
    return torch.nextafter(value, torch.full_like(value, float("inf")))
