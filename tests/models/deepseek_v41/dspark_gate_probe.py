# SPDX-License-Identifier: MIT
"""Diagnostic Engram fusion with the pinned eager single-token reduction tree."""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _sum_lanes(a0, a1, a2, a3):
    value = ((a0 + a1) + a2) + a3
    lane = tl.arange(0, 64)
    value = tl.gather(value, lane, 0) + tl.gather(value, lane + 64, 0)
    for offset in tl.static_range(6):
        value = value + tl.gather(value, (lane + (1 << offset)) % 64, 0)
    return tl.sum(tl.where(lane == 0, value, 0), 0)


@triton.jit
def _gate(H, KV, W, O, Stats, DIM: tl.constexpr, HC: tl.constexpr):
    row, branch = tl.program_id(0), tl.program_id(1)
    lanes = tl.arange(0, 128)
    h0 = tl.full((128,), 0, tl.float32)
    h1, h2, h3 = h0, h0, h0
    k0, k1, k2, k3 = h0, h0, h0, h0
    d0, d1, d2, d3 = h0, h0, h0, h0
    for tile in range(DIM // 512):
        x = lanes * 4 + tile * 512
        r0 = tl.load(H + (row * HC + branch) * DIM + x).to(tl.float32)
        r1 = tl.load(H + (row * HC + branch) * DIM + x + 1).to(tl.float32)
        r2 = tl.load(H + (row * HC + branch) * DIM + x + 2).to(tl.float32)
        r3 = tl.load(H + (row * HC + branch) * DIM + x + 3).to(tl.float32)
        v0 = tl.load(KV + (row * (HC + 1) + branch) * DIM + x).to(tl.float32)
        v1 = tl.load(KV + (row * (HC + 1) + branch) * DIM + x + 1).to(tl.float32)
        v2 = tl.load(KV + (row * (HC + 1) + branch) * DIM + x + 2).to(tl.float32)
        v3 = tl.load(KV + (row * (HC + 1) + branch) * DIM + x + 3).to(tl.float32)
        w0 = tl.load(W + branch * DIM + x)
        w1 = tl.load(W + branch * DIM + x + 1)
        w2 = tl.load(W + branch * DIM + x + 2)
        w3 = tl.load(W + branch * DIM + x + 3)
        h0, h1, h2, h3 = h0 + r0 * r0, h1 + r1 * r1, h2 + r2 * r2, h3 + r3 * r3
        k0, k1, k2, k3 = k0 + v0 * v0, k1 + v1 * v1, k2 + v2 * v2, k3 + v3 * v3
        d0, d1, d2, d3 = (
            d0 + (r0 * w0) * v0,
            d1 + (r1 * w1) * v1,
            d2 + (r2 * w2) * v2,
            d3 + (r3 * w3) * v3,
        )
    hmean = _sum_lanes(h0, h1, h2, h3) * (1.0 / DIM)
    kmean = _sum_lanes(k0, k1, k2, k3) * (1.0 / DIM)
    dot_sum = _sum_lanes(d0, d1, d2, d3)
    rstd = tl.div_rn(1.0, libdevice.sqrt(hmean + 1e-20)) * tl.div_rn(
        1.0, libdevice.sqrt(kmean + 1e-20)
    )
    dot = (dot_sum * rstd) * (DIM**-0.5)
    signed = libdevice.copysign(libdevice.sqrt(tl.maximum(tl.abs(dot), 1e-6)), dot)
    gate = tl.div_rn(1.0, 1.0 + libdevice.exp(-signed))
    base = (row * HC + branch) * 8
    tl.store(Stats + base, hmean)
    tl.store(Stats + base + 1, kmean)
    tl.store(Stats + base + 2, dot_sum)
    tl.store(Stats + base + 3, gate)
    tl.store(Stats + base + 4, rstd)
    tl.store(Stats + base + 5, dot)
    tl.store(Stats + base + 6, signed)
    tl.store(Stats + base + 7, libdevice.exp(-signed))
    x = tl.arange(0, triton.next_power_of_2(DIM))
    hidden = tl.load(H + (row * HC + branch) * DIM + x, x < DIM, 0).to(tl.float32)
    value = tl.load(KV + (row * (HC + 1) + HC) * DIM + x, x < DIM, 0).to(tl.float32)
    tl.store(O + (row * HC + branch) * DIM + x, hidden + gate * value, x < DIM)


def fused_gate(hidden, projected, weight):
    assert hidden.shape[-2:] == (4, 5120)
    rows = hidden.numel() // (4 * 5120)
    hidden, projected = hidden.contiguous(), projected.contiguous()
    output = torch.empty_like(hidden)
    stats = torch.empty(rows, 4, 8, device=hidden.device, dtype=torch.float32)
    _gate[(rows, 4)](
        hidden,
        projected,
        weight,
        output,
        stats,
        5120,
        4,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return output, stats


def fixed_moment_gate(hidden, projected, weight, token_mask=None):
    """Isolate reductions from scalar math using the existing Torch operations.

    The fused probe's nonlinear output is deliberately discarded: it does not
    reproduce Torch's scalar rounding. This is a numerical probe, not an optimized
    implementation suitable for production or a claimed fusion speedup.
    """
    _, stats = fused_gate(hidden, projected, weight)
    stats = stats.view(*hidden.shape[:-1], 8)
    rstd = torch.rsqrt(stats[..., 0] + 1e-20) * torch.rsqrt(stats[..., 1] + 1e-20)
    dot = stats[..., 2] * rstd * hidden.shape[-1] ** -0.5
    gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
    if token_mask is not None:
        gate = gate.masked_fill(~token_mask.unsqueeze(-1), 0)
    value = projected[..., hidden.shape[-2] * hidden.shape[-1] :]
    return (hidden.float() + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(
        hidden.dtype
    )
