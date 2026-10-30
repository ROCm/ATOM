"""Group64 W4/A8 intervals with an original-BF16 current-score anchor.

This is a standalone candidate, not a service route. Only candidate rows are
refined, using the accepted original-weight refinement and Native topk.
"""

import torch
import triton
import triton.language as tl
from aiter import topk_select

from atom.models.qwen4_exp_mono.head.screen import refine

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
def screen_groups(
    QEven,
    QOdd,
    XScale,
    XStats,
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
    quant_error = tl.full((BN,), 0.0, tl.float32)
    qnorm_product = tl.full((BN,), 0.0, tl.float32)
    xnorm_squared = tl.full((), 0.0, tl.float32)
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
        we = tl.load(WErrorNorm + offset, n[:, None] < N, 0)
        wq = tl.load(WQuantNorm + offset, n[:, None] < N, 0)
        xs = tl.load(XScale + kg)
        xn = tl.load(XStats + kg * 3)
        en = tl.load(XStats + kg * 3 + 1)
        qn = tl.load(XStats + kg * 3 + 2)
        y += tl.sum((dots.to(tl.float32) * xs[None, :]) * ws, 1)
        quant_error += tl.sum(xn[None, :] * we + en[None, :] * wq, 1)
        qnorm_product += tl.sum(qn[None, :] * wq, 1)
        xnorm_squared += tl.sum(xn * xn, 0)
    # Norm certificates and interval reductions are rounded outwards by a
    # gamma bound; this is arithmetic accounting, not an L2 acceptance cutoff.
    u = 5.960464477539063e-8
    gamma = (K + 8) * u / (1.0 - (K + 8) * u)
    inflate = (1.0 + gamma) / (1.0 - gamma)
    xn = tl.sqrt(xnorm_squared) * inflate
    wn = tl.load(WNorm + n, n < N, 0)
    dot_error = quant_error * inflate + gamma * (xn * wn + qnorm_product * inflate)
    approx = y.to(tl.bfloat16).to(tl.float32)
    error = (
        dot_error * (1.0 + 1.0 / 256.0) + tl.abs(approx) * (2.0 / 255.0) + 1.0e-6
    ) * inflate
    finite = (tl.abs(approx) < float("inf")) & (error < float("inf")) & (error >= 0.0)
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


class HeadScreenW4:
    def __init__(self, weight):
        self.weight = weight
        self.n, self.k = weight.shape
        assert self.n == 248320 and self.k == 2560
        self.packed, self.ws, self.we, self.wqnorm, self.wnorm = prepare_weight(weight)
        alloc = lambda *shape, dtype=torch.float32: torch.empty(
            shape, device=weight.device, dtype=dtype
        )
        groups, blocks = self.k // GROUP, triton.cdiv(self.n, BLOCK_N)
        self.qeven = alloc(self.k // 8, dtype=torch.uint32)
        self.qodd = torch.empty_like(self.qeven)
        self.xscale, self.xstats = alloc(groups), alloc(groups, 3)
        self.xsum = alloc(groups, dtype=torch.int32)
        self.approx = alloc(1, self.n, dtype=torch.bfloat16)
        self.error, self.refined = alloc(1, self.n), alloc(1, self.n)
        self.lower, self.best_y = alloc(blocks), alloc(blocks)
        self.best_id = alloc(blocks, dtype=torch.int32)
        self.threshold = alloc(1)
        self.anchor = alloc(1, dtype=torch.int32)
        self.anchor_value, self.anchor_bound = alloc(1), alloc(1)
        self.counts = alloc(1, triton.cdiv(self.n, 16), dtype=torch.int32)
        self.row_lens = torch.full(
            (1,), self.n, device=weight.device, dtype=torch.int32
        )
        self.ids = alloc(1, dtype=torch.int32)
        self.last_screen = self.last_anchor = self.last_refine = None

    def run(self, x, out=None):
        assert (
            x.shape == (1, self.k) and x.is_contiguous() and x.dtype == torch.bfloat16
        )
        quant_groups[(self.k // GROUP,)](
            x,
            self.qeven,
            self.qodd,
            self.xscale,
            self.xstats,
            self.xsum,
            self.k,
            GROUP,
            num_warps=4,
        )
        blocks = triton.cdiv(self.n, BLOCK_N)
        self.last_screen = screen_groups[(blocks,)](
            self.qeven,
            self.qodd,
            self.xscale,
            self.xstats,
            self.xsum,
            self.packed,
            self.ws,
            self.we,
            self.wqnorm,
            self.wnorm,
            self.approx,
            self.error,
            self.lower,
            self.best_y,
            self.best_id,
            self.n,
            self.k,
            GROUP,
            BLOCK_N,
            BLOCK_K,
            num_warps=4,
        )
        self.last_anchor = anchor_lower[(1,)](
            x,
            self.weight,
            self.wnorm,
            self.lower,
            self.best_y,
            self.best_id,
            self.threshold,
            self.anchor,
            self.anchor_value,
            self.anchor_bound,
            self.n,
            self.k,
            GROUP,
            blocks,
            triton.next_power_of_2(blocks),
            triton.next_power_of_2(self.k),
            num_warps=8,
        )
        self.last_refine = refine[(triton.cdiv(self.n, 16), 1)](
            x,
            self.weight,
            self.approx,
            self.error,
            self.threshold,
            self.refined,
            self.counts,
            self.n,
            self.k,
            16,
            512,
            num_warps=4,
        )
        answer = self.ids if out is None else out
        topk_select(
            self.refined, 1, tie="low", end=self.row_lens, output_idx=answer.view(-1, 1)
        )
        return answer
