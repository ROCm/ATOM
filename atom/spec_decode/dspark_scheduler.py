# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""DSpark confidence-scheduled verification: Hardware-Aware Prefix Scheduler.

Implements paper Algorithm 1 (DeepSeek-AI, 2026, DSpark §3.2.2): given per-
position confidence scores for a batch of draft blocks and an engine throughput
profile ``SPS(B)``, choose a per-request verification length ``ell_r`` that
maximizes system-wide token throughput ``Theta = tau * SPS(B)``.

Pure, GPU-free, CPU-testable. The proposer (``atom/spec_decode/eagle.py``) calls
``schedule_prefix_lengths`` after computing (and STS-calibrating) confidence, and
truncates each draft block to ``ell_r`` before verification.

Phase-2 scope:
  * ``calibrate_confidence`` — Sequential Temperature Scaling (STS) apply-side.
  * ``schedule_prefix_lengths`` — greedy admission with early-stop (Algorithm 1).
The early-stop ``break`` is a CORRECTNESS guarantee (non-anticipating property,
paper Appendix A), not a perf optimization — see ``_GREEDY_EARLY_STOP``.
"""

from __future__ import annotations

from typing import Sequence

import torch

# Early-stop in the greedy admission loop. Required for losslessness under the
# smooth-SPS assumption (paper §3.5). Disabling it does an unconstrained global
# search that crosses the SPS sawtooth (paper §5.2) but then needs the async
# two-step causal barrier to stay lossless — that variant lives at the call site.
_GREEDY_EARLY_STOP = True


def survival_probabilities(confidence: torch.Tensor) -> torch.Tensor:
    """Cumulative-product survival probabilities ``a_{r,j} = prod_{i<=j} c_{r,i}``.

    Args:
        confidence: [R, gamma] per-position conditional acceptance probs in (0,1).
    Returns:
        [R, gamma] cumulative survival probabilities, monotonically non-increasing
        along the block axis.
    """
    if confidence.ndim != 2:
        raise ValueError(f"confidence must be [R, gamma], got {tuple(confidence.shape)}")
    return confidence.float().cumprod(dim=1)


def calibrate_confidence(
    confidence: torch.Tensor,
    sts_temperatures: torch.Tensor | None,
) -> torch.Tensor:
    """Apply Sequential Temperature Scaling (STS) to raw confidence (paper §3.2.1).

    STS is an order-preserving per-position temperature on the confidence LOGIT:
    ``c_k <- sigmoid(logit(c_k) / T_k)``. Temperatures are fit offline (held-out
    grid search minimizing cumulative-product ECE); this is the cheap apply side.

    Args:
        confidence: [R, gamma] raw sigmoid confidence in (0, 1).
        sts_temperatures: [gamma] positive per-position temperatures, or None
            (T=1, i.e. no calibration — functionally valid, just less accurate).
    Returns:
        [R, gamma] calibrated confidence in (0, 1).
    """
    c = confidence.float().clamp(1e-6, 1 - 1e-6)
    if sts_temperatures is None:
        return c
    t = sts_temperatures.float().to(c.device)
    if t.numel() != c.shape[1]:
        raise ValueError(
            f"sts_temperatures length {t.numel()} != block size {c.shape[1]}"
        )
    if torch.any(t <= 0):
        raise ValueError("STS temperatures must be strictly positive (order-preserving).")
    logit = torch.log(c) - torch.log1p(-c)  # logit(c)
    return torch.sigmoid(logit / t)


def expected_throughput(
    survival: torch.Tensor,
    ell: Sequence[int],
    sps_table: torch.Tensor,
) -> float:
    """System-wide expected throughput ``Theta = tau * SPS(B)`` for given lengths.

    tau = sum_r (1 + sum_{j<ell_r} a_{r,j})   (bonus token + expected accepts)
    B   = sum_r (1 + ell_r)                    (bonus + verified draft tokens)

    Args:
        survival: [R, gamma] cumulative survival probs.
        ell: length-R verification lengths (0..gamma).
        sps_table: 1-D throughput profile indexed by batch token count B.
    """
    R = survival.shape[0]
    B = R + int(sum(ell))
    tau = float(R)
    for r in range(R):
        if ell[r] > 0:
            tau += float(survival[r, : ell[r]].sum())
    return tau * _sps_lookup(sps_table, B)


def _sps_lookup(sps_table: torch.Tensor, B: int) -> float:
    """Clamp-indexed lookup into the (possibly short) profiled cost table."""
    idx = max(0, min(B, sps_table.shape[0] - 1))
    return float(sps_table[idx])


def build_sps_table(
    token_points: Sequence[int],
    sps_points: Sequence[float],
    max_b: int,
) -> torch.Tensor:
    """Build a dense ``sps_table[B]`` (B in 0..max_b) from measured sample points.

    The engine throughput ``SPS(B)`` (steps/sec at a forward of B tokens) is only
    profiled at a handful of batch sizes (the captured CUDA-graph sizes). This
    densifies those samples into a per-B lookup the scheduler can index in O(1).

    Interpolation: piecewise-linear between measured points, flat-held outside the
    measured range. Linear keeps the curve smooth (the scheduler's early-stop
    losslessness assumes a smoothly decaying SPS; paper §3.5). Real hardware is a
    sawtooth (paper §5.2) only at finer granularity than the captured sizes, so
    linear between captured points is a faithful first-order model.

    Args:
        token_points: measured forward sizes (num_tokens), any order.
        sps_points: throughput (steps/sec) at each token_point. Same length.
        max_b: table covers indices 0..max_b inclusive.
    Returns:
        [max_b + 1] fp32 tensor; sps_table[B] = SPS at a B-token forward.
    """
    if len(token_points) != len(sps_points):
        raise ValueError("token_points and sps_points must have equal length")
    if len(token_points) == 0:
        raise ValueError("need at least one measured point")
    if max_b < 0:
        raise ValueError("max_b must be non-negative")

    pts = sorted(zip(token_points, sps_points))  # by token count ascending
    xs = [float(x) for x, _ in pts]
    ys = [float(y) for _, y in pts]

    table = torch.empty(max_b + 1, dtype=torch.float32)
    j = 0
    for b in range(max_b + 1):
        if b <= xs[0]:
            table[b] = ys[0]
        elif b >= xs[-1]:
            table[b] = ys[-1]
        else:
            # Advance to the segment [xs[j], xs[j+1]] containing b.
            while j + 1 < len(xs) and b > xs[j + 1]:
                j += 1
            x0, x1, y0, y1 = xs[j], xs[j + 1], ys[j], ys[j + 1]
            w = (b - x0) / (x1 - x0) if x1 > x0 else 0.0
            table[b] = y0 + w * (y1 - y0)
    return table


def schedule_prefix_lengths(
    confidence: torch.Tensor,
    sps_table: torch.Tensor,
    *,
    sts_temperatures: torch.Tensor | None = None,
    early_stop: bool = _GREEDY_EARLY_STOP,
) -> list[int]:
    """Hardware-Aware Prefix Scheduler (paper Algorithm 1).

    Greedily admits draft tokens across ALL requests in descending order of
    survival probability, extending throughput ``Theta = tau * SPS(B)`` one token
    at a time, until throughput stops increasing.

    Args:
        confidence: [R, gamma] per-position confidence (raw; STS applied here if
            ``sts_temperatures`` given).
        sps_table: 1-D engine throughput profile (steps/sec) indexed by batch
            token count B. Calibrated once at engine init.
        sts_temperatures: optional [gamma] STS temperatures.
        early_stop: stop at the first throughput drop (lossless under smooth SPS;
            paper §3.5). Set False for an unconstrained global search across the
            SPS sawtooth (paper §5.2) — only lossless with the async barrier.
    Returns:
        length-R list of verification lengths ``ell_r`` in 0..gamma.
    """
    ell = schedule_prefix_lengths_tensor(
        confidence,
        sps_table,
        sts_temperatures=sts_temperatures,
        early_stop=early_stop,
    )
    return ell.tolist()  # syncs; for tests / non-hot-path callers only


def schedule_prefix_lengths_tensor(
    confidence: torch.Tensor,
    sps_table: torch.Tensor,
    *,
    sts_temperatures: torch.Tensor | None = None,
    early_stop: bool = _GREEDY_EARLY_STOP,
) -> torch.Tensor:
    """Vectorized Hardware-Aware Prefix Scheduler — returns ``ell`` as a device
    tensor with NO host sync (no ``.item()``/``.tolist()``), for the decode hot
    path. ``schedule_prefix_lengths`` wraps this and adds ``.tolist()``.

    Vectorization insight: survival ``a_{r,j}`` is monotone non-increasing in j
    (cumprod), so admitting the GLOBAL top-m tokens by survival automatically
    respects intra-block prefix order. The greedy admission therefore collapses
    to a single descending sort + cumulative sum:

        admit top-m (m=0..N) -> B(m) = R + m ; tau(m) = R + cumsum(sorted a)[m]
        Theta(m) = tau(m) * SPS(B(m))

    Early-stop returns the first local max of Theta (paper §3.5, lossless under
    smooth SPS); otherwise the global argmax (needs the async barrier to stay
    lossless, paper §5.2). Zero-survival tokens sort to the tail and can only
    decrease Theta, so they are never admitted (no explicit filtering needed).

    Returns: int64 tensor [R] of verification lengths in 0..gamma.
    """
    if confidence.ndim != 2:
        raise ValueError(f"confidence must be [R, gamma], got {tuple(confidence.shape)}")
    R, gamma = confidence.shape
    device = confidence.device
    if R == 0:
        return torch.zeros(0, dtype=torch.long, device=device)

    calibrated = calibrate_confidence(confidence, sts_temperatures)
    a = survival_probabilities(calibrated)  # [R, gamma]
    N = R * gamma

    flat = a.reshape(-1)  # [N]
    req = (
        torch.arange(R, device=device)
        .view(R, 1)
        .expand(R, gamma)
        .reshape(-1)
    )  # [N] request id per flat position

    sorted_vals, sort_idx = torch.sort(flat, descending=True)  # [N]
    sorted_req = req[sort_idx]  # [N]

    sps = sps_table.to(device=device, dtype=torch.float32)
    last = sps.numel() - 1

    # Theta(m) for m = 0..N admitted tokens. index m == number admitted.
    m = torch.arange(N + 1, device=device)  # [N+1]
    tau = R + torch.cat([flat.new_zeros(1), sorted_vals.cumsum(0)])  # [N+1]
    B = (R + m).clamp(max=last)  # [N+1] batch token count, clamped into table
    theta = tau * sps[B]  # [N+1]

    if early_stop:
        # First m with Theta(m+1) <= Theta(m): m* = that m (the local peak). The
        # reference breaks at the first non-increase, keeping the prior best.
        nonincrease = theta[1:] <= theta[:-1]  # [N] ; index k -> step m=k -> k+1
        has_drop = nonincrease.any()
        first_drop = nonincrease.int().argmax()  # 0 if none (guarded below)
        m_star = torch.where(has_drop, first_drop, torch.tensor(N, device=device))
    else:
        m_star = theta.argmax()  # first global max

    # ell[r] = count of admitted (top-m_star) candidates belonging to request r.
    # Sync-free: mask by position < m_star (m_star stays a 0-dim tensor).
    admitted = torch.arange(N, device=device) < m_star  # [N] bool
    ell = torch.zeros(R, dtype=torch.long, device=device)
    ell.scatter_add_(0, sorted_req, admitted.long())
    return ell
