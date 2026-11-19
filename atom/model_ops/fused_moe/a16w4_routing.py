# SPDX-License-Identifier: MIT
"""Normalize routes for the explicit clamped A16W4 backend, on the input stream."""

import torch
import triton
import triton.language as tl


@triton.jit
def _prepare_routes(
    IDS,
    W,
    OUT_IDS,
    OUT_W,
    VALID,
    SI0,
    SI1,
    SW0,
    SW1,
    EXPERTS: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
):
    row = tl.program_id(0)
    s = tl.arange(0, BK)
    e = tl.load(IDS + row * SI0 + s * SI1, s < K, other=-1)
    w = tl.load(W + row * SW0 + s * SW1, s < K, other=0.0).to(tl.float32)
    valid = (s < K) & (e >= 0) & (e < EXPERTS)
    if VALID is not None:
        valid = valid & (row < tl.load(VALID))
    # Some sort backends retain a single slot per (token, expert). Compute
    # that expert once with the sum of its route weights, not just the last
    # weight. Leave unique routes (including their weight bits) untouched.
    same = (e[:, None] == e[None, :]) & valid[:, None] & valid[None, :]
    prior = tl.sum((same & (s[None, :] < s[:, None])).to(tl.int32), axis=1)
    n = tl.sum(same.to(tl.int32), axis=1)
    total = tl.sum(tl.where(same, w[None, :], 0.0), axis=1)
    keep = valid & (prior == 0)
    tl.store(OUT_IDS + row * K + s, tl.where(keep, e, -1), s < K)
    tl.store(
        OUT_W + row * K + s, tl.where(keep, tl.where(n == 1, w, total), 0.0), s < K
    )


def prepare_a16w4_routes(ids, weights, num_experts, scheduled_tokens_gpu=None):
    if ids.ndim != 2 or weights.shape != ids.shape or weights.device != ids.device:
        raise ValueError(
            "A16W4 needs matching [tokens, top_k] IDs and weights on one device"
        )
    rows, k = ids.shape
    if not 0 < k <= 16:
        raise ValueError("A16W4 route preparation requires 1 <= top_k <= 16")
    if scheduled_tokens_gpu is not None and (
        scheduled_tokens_gpu.dtype != torch.int32
        or scheduled_tokens_gpu.numel() != 1
        or scheduled_tokens_gpu.device != ids.device
    ):
        raise ValueError("scheduled_tokens_gpu must be a device-local int32 count")
    oi = torch.empty((rows, k), device=ids.device, dtype=torch.int32)
    ow = torch.empty((rows, k), device=ids.device, dtype=torch.float32)
    if rows:
        _prepare_routes[(rows,)](
            ids,
            weights,
            oi,
            ow,
            scheduled_tokens_gpu,
            *ids.stride(),
            *weights.stride(),
            num_experts,
            k,
            triton.next_power_of_2(k),
            num_warps=1,
        )
    return oi, ow
