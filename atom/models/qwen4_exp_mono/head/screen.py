"""Bounded vocabulary screening, followed by original BF16 row products.

This module only produces draft argmax ids. It is not a logits substitute.
The FP8 view selects possible winners; selected rows use the original weight.
FP32 accumulation order can differ from tgemm, so BF16 output L2 still needs
measurement and whole-model acceptance is independent of the bound check.
"""

import torch
import triton
import triton.language as tl
from aiter import topk_select
from aiter.ops.gemm_op_a8w8 import gemm_a8w8_bpreshuffle


@triton.jit
def quant_input(X, Q, Scale, Stats, K: tl.constexpr, BK: tl.constexpr):
    row = tl.program_id(0)
    k = tl.arange(0, BK)
    x = tl.load(X + row * K + k, k < K, 0).to(tl.float32)
    scale = tl.max(tl.abs(x), 0) / 448.0
    inverse = tl.where(scale > 0, 1.0 / scale, 0.0)
    q = tl.minimum(tl.maximum(x * inverse, -448.0), 448.0).to(Q.dtype.element_ty)
    xhat = q.to(tl.float32) * scale
    error = x - xhat
    tl.store(Q + row * K + k, q, k < K)
    tl.store(Scale + row, scale)
    tl.store(Stats + row * 3, tl.sqrt(tl.sum(x * x, 0)))
    tl.store(Stats + row * 3 + 1, tl.sqrt(tl.sum(error * error, 0)))
    tl.store(Stats + row * 3 + 2, tl.sqrt(tl.sum(xhat * xhat, 0)))


@triton.jit
def intervals(
    Y, WStats, XStats, Error, Lower, N: tl.constexpr, K: tl.constexpr, BN: tl.constexpr
):
    block, row = tl.program_id(0), tl.program_id(1)
    n = block * BN + tl.arange(0, BN)
    y = tl.load(Y + row * N + n, n < N, 0).to(tl.float32)
    xn = tl.load(XStats + row * 3)
    en = tl.load(XStats + row * 3 + 1)
    qn = tl.load(XStats + row * 3 + 2)
    we = tl.load(WStats + n * 3, n < N, 0)
    wq = tl.load(WStats + n * 3 + 1, n < N, 0)
    wn = tl.load(WStats + n * 3 + 2, n < N, 0)
    # Cauchy-Schwarz quantization bound, FP32 accumulation/scaling budget,
    # and BOTH the approximate and reference BF16 endpoint roundings.
    gamma = (K + 8) * 5.960464477539063e-8 / (1.0 - (K + 8) * 5.960464477539063e-8)
    dot_error = (xn * we + en * wq) * 1.00001 + gamma * (xn * wn + qn * wq)
    e = dot_error * (1.0 + 1.0 / 256.0) + tl.abs(y) * (2.0 / 255.0) + 1.0e-6
    tl.store(Error + row * N + n, e, n < N)
    lower = tl.max(tl.where(n < N, y - e, -float("inf")), 0)
    tl.store(Lower + row * tl.cdiv(N, BN) + block, lower)


@triton.jit
def max_lower(Lower, Threshold, Blocks: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    i = tl.arange(0, B)
    v = tl.load(Lower + row * Blocks + i, i < Blocks, -float("inf"))
    tl.store(Threshold + row, tl.max(v, 0))


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


def prepare_weight(weight):
    assert weight.dtype == torch.bfloat16 and weight.is_contiguous()
    n, k = weight.shape
    assert n % 16 == 0 and k % 32 == 0
    q = torch.empty_like(weight, dtype=torch.float8_e4m3fn)
    scales = torch.empty((n, 1), device=weight.device, dtype=torch.float32)
    stats = torch.empty((n, 3), device=weight.device, dtype=torch.float32)
    # Use double for the offline norm certificates. Inflate after conversion
    # to FP32; normal runtime only reads these immutable per-vocabulary rows.
    for begin in range(0, n, 4096):
        stop = min(n, begin + 4096)
        w = weight[begin:stop].float()
        scale = w.abs().amax(1, keepdim=True) / 448.0
        qq = (w / torch.where(scale > 0, scale, 1.0)).clamp(-448, 448).to(q.dtype)
        restored = qq.float() * scale
        q[begin:stop] = qq
        scales[begin:stop] = scale
        stats[begin:stop] = (
            torch.stack(
                [
                    (w.double() - restored.double()).norm(dim=1),
                    restored.double().norm(dim=1),
                    w.double().norm(dim=1),
                ],
                1,
            ).float()
            * 1.00001
        )
    shuffled = (
        q.view(n // 16, 16, k // 32, 2, 16)
        .permute(0, 2, 3, 1, 4)
        .contiguous()
        .view(n, k)
    )
    return shuffled, scales, stats


class HeadScreen:
    def __init__(self, weight, rows=1):
        self.weight = weight
        self.wq, self.ws, self.wstats = prepare_weight(weight)
        self.rows, (self.n, self.k) = rows, weight.shape
        f = lambda *shape: torch.empty(shape, device=weight.device, dtype=torch.float32)
        self.q = torch.empty(
            (rows, self.k), device=weight.device, dtype=torch.float8_e4m3fn
        )
        self.scale, self.xstats = f(rows, 1), f(rows, 3)
        self.error, self.refined = f(rows, self.n), f(rows, self.n)
        self.lower, self.threshold = f(rows, triton.cdiv(self.n, 1024)), f(rows)
        self.counts = torch.empty(
            (rows, triton.cdiv(self.n, 16)), device=weight.device, dtype=torch.int32
        )
        self.last_refine = None
        self.row_lens = torch.full(
            (rows,), self.n, device=weight.device, dtype=torch.int32
        )
        self.ids = torch.empty((rows,), device=weight.device, dtype=torch.int32)

    def run(self, x, out=None):
        assert x.shape == (self.rows, self.k) and x.is_contiguous()
        quant_input[(self.rows,)](
            x, self.q, self.scale, self.xstats, self.k, triton.next_power_of_2(self.k)
        )
        self.approx = gemm_a8w8_bpreshuffle(
            self.q, self.wq, self.scale, self.ws, dtype=torch.bfloat16
        )
        intervals[(triton.cdiv(self.n, 1024), self.rows)](
            self.approx,
            self.wstats,
            self.xstats,
            self.error,
            self.lower,
            self.n,
            self.k,
            1024,
        )
        max_lower[(self.rows,)](
            self.lower,
            self.threshold,
            triton.cdiv(self.n, 1024),
            triton.next_power_of_2(triton.cdiv(self.n, 1024)),
        )
        self.last_refine = refine[(triton.cdiv(self.n, 16), self.rows)](
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
