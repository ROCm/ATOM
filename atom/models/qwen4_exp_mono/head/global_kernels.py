"""Scalar Group64 screening; only the error certificate becomes global.

The unchanged dot4 path keeps both early-clobber outputs and every group scale.
An independent once-per-input reduction publishes conservative whole-vector
norms. Invalid/unsafe certificate arithmetic retains every vocabulary row.
"""

import triton
import triton.language as tl


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
