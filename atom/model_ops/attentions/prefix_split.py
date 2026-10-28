# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Split-KV prefix attention on the varlen FMHA, without a kernel change.

A prefix-cached prefill attends a few new tokens over a long context. The
FMHA tiles the query rows (128 packed GQA rows a workgroup), so such a step
launches a handful of workgroups that each walk the whole context: ~12 us per
1k keys on gfx1250, flat in the query count. The context is cut into
``n_full`` slices and every (request, slice) pair becomes a varlen sequence of
its own in ONE launch -- the same query rows repeated, no causal mask, so the
grid grows ``n_full``-fold -- while a second, causal launch covers each
request's tail (its new tokens' own keys). The partial outputs are merged
through their log-sum-exps.

Both launches read the gathered K/V as laid out (every request's keys
contiguous, requests in order), so a varlen sequence with zero query rows is
placed over every range a launch must skip: the kernel assigns no tiles to it
and the K layout stays contiguous.

The plan is built on the host from the step's lengths (`build_plan`) and
uploaded by the attention metadata builder; `merge` is one Triton launch.
"""
import os

import numpy as np
import torch
import triton
import triton.language as tl

# Slice length the prefix is cut into, and the most slices a request gets.
# 16 full slices on a 88k context take a layer's prefix attention from 1.07 ms
# to 0.24 ms; 32 to 0.21 ms, where the fixed costs dominate.
SLICE_KEYS = int(os.getenv("ATOM_PREFIX_SPLIT_SLICE_KEYS", 3072))
MAX_FULL_SLICES = 31
# Below this prefix the single launch is already cheap. (Both knobs are for
# tests that want the split exercised on short prompts.)
MIN_PREFIX_KEYS = int(os.getenv("ATOM_PREFIX_SPLIT_MIN_KEYS", 4096))
MAX_PARTIAL_ROWS = 16384


def build_plan(seqlens_q: np.ndarray, context_lens: np.ndarray):
    """The split layout for prefill rows with ``seqlens_q`` new tokens over
    ``context_lens`` keys each (both include the new tokens), or None when
    the step does not qualify. int32 arrays plus the two launches' maxima."""
    seqlens_q = np.asarray(seqlens_q, dtype=np.int64)
    context_lens = np.asarray(context_lens, dtype=np.int64)
    prefix = context_lens - seqlens_q
    if len(seqlens_q) == 0 or prefix.min() < MIN_PREFIX_KEYS or seqlens_q.min() <= 0:
        return None
    n_full = int(min(MAX_FULL_SLICES, max(1, -(-int(prefix.max()) // SLICE_KEYS))))
    # The repeated query rows and the slice partials scale with n_full * tokens;
    # past ~16k rows the merge's reads outweigh the parallelism.
    n_full = max(1, min(n_full, MAX_PARTIAL_ROWS // int(seqlens_q.sum())))
    n = len(seqlens_q)
    cu_q = np.concatenate(([0], np.cumsum(seqlens_q)))
    cu_k = np.concatenate(([0], np.cumsum(context_lens)))
    # launch A: per request n_full full slices (T rows each) then a spacer over its tail
    cu_q_a = np.zeros(n * (n_full + 1) + 1, dtype=np.int64)
    cu_k_a = np.zeros_like(cu_q_a)
    # launch B: per request a spacer over its prefix then its causal tail
    cu_q_b = np.zeros(2 * n + 1, dtype=np.int64)
    cu_k_b = np.zeros_like(cu_q_b)
    q_rep = np.empty(n_full * int(seqlens_q.sum()), dtype=np.int32)
    merge_rows = np.empty((n_full, int(seqlens_q.sum())), dtype=np.int32)
    ra = 0
    row = 0
    for i in range(n):
        T, P, kb = int(seqlens_q[i]), int(prefix[i]), int(cu_k[i])
        bounds = (np.arange(n_full + 1) * P) // n_full
        for s in range(n_full):
            cu_q_a[ra + 1] = cu_q_a[ra] + T
            cu_k_a[ra + 1] = kb + bounds[s + 1]
            q_rep[row : row + T] = np.arange(cu_q[i], cu_q[i] + T)
            merge_rows[s, cu_q[i] : cu_q[i] + T] = np.arange(row, row + T)
            row += T
            ra += 1
        cu_q_a[ra + 1] = cu_q_a[ra]  # spacer: no query rows over the tail
        cu_k_a[ra + 1] = kb + int(context_lens[i])
        ra += 1
        cu_q_b[2 * i + 1] = cu_q_b[2 * i]  # spacer over the prefix
        cu_k_b[2 * i + 1] = kb + P
        cu_q_b[2 * i + 2] = cu_q_b[2 * i] + T
        cu_k_b[2 * i + 2] = kb + int(context_lens[i])
    return dict(
        n_full=n_full,
        max_q=int(seqlens_q.max()),
        max_k_a=int(np.diff(cu_k_a).max()),
        cu_q_a=cu_q_a.astype(np.int32),
        cu_k_a=cu_k_a.astype(np.int32),
        cu_q_b=cu_q_b.astype(np.int32),
        cu_k_b=cu_k_b.astype(np.int32),
        q_rep=q_rep,
        merge_rows=merge_rows.reshape(-1),
    )


@triton.jit
def _merge_kernel(
    o_a, lse_a, o_b, lse_b, rows, out,
    T, N_FULL,
    stride_oa_row, stride_oa_head, stride_ob_row, stride_ob_head,
    stride_la_head, stride_lb_head, stride_out_row, stride_out_head,
    H: tl.constexpr, D: tl.constexpr,
):
    t = tl.program_id(0)
    h = tl.program_id(1)
    d = tl.arange(0, D)
    # tail partial first
    m = tl.load(lse_b + h * stride_lb_head + t)
    acc = tl.load(o_b + t * stride_ob_row + h * stride_ob_head + d).to(tl.float32)
    w_sum = 1.0
    for s in range(N_FULL):
        r = tl.load(rows + s * T + t)
        l = tl.load(lse_a + h * stride_la_head + r)
        o = tl.load(o_a + r * stride_oa_row + h * stride_oa_head + d).to(tl.float32)
        m_new = tl.maximum(m, l)
        a = tl.exp(m - m_new)
        b = tl.exp(l - m_new)
        acc = acc * a + o * b
        w_sum = w_sum * a + b
        m = m_new
    tl.store(out + t * stride_out_row + h * stride_out_head + d, (acc / w_sum).to(out.dtype.element_ty))


def merge(o_a, lse_a, o_b, lse_b, merge_rows, n_full, out=None):
    """``out[t, h] = sum_s w_s o_s`` over the n_full slice partials of row t
    (``o_a`` rows ``merge_rows[s, t]``) and the tail partial ``o_b[t]``, the
    weights the softmax of their log-sum-exps. LSE tensors are [H, rows]."""
    T, H, D = o_b.shape
    if out is None:
        out = torch.empty_like(o_b)
    _merge_kernel[(T, H)](
        o_a, lse_a, o_b, lse_b, merge_rows, out,
        T, n_full,
        o_a.stride(0), o_a.stride(1), o_b.stride(0), o_b.stride(1),
        lse_a.stride(0), lse_b.stride(0), out.stride(0), out.stride(1),
        H=H, D=D,
    )
    return out
