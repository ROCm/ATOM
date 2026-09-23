# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""A query token's prefill attention must not depend on its batch-mates.

`sparse_attn_v4_paged_prefill` has no unit test at all, and it is the last
consumer on the chunked-prefill path whose batch-composition behaviour is
unmeasured. Swapping OPUS for Triton (`ATOM_FORCE_ATTN_TRITON=1`) changes the
IMPLEMENTATION but not the contract, so it cannot settle this either.

Why it is under suspicion: end-to-end, GSM8K prompts SHORT enough never to be
chunked still lose ~3.9pp once the checkpoint ladder starts cutting the long
ones (509 docs, 0.9568 -> 0.9175). Such a prompt is prefilled in a single
chunk and its own index buffers are provably composition-invariant, so the
only way it can be damaged is through a batch-mate. A kernel that derived any
loop bound from a batch-wide maximum instead of the per-token indptr delta
would do exactly that: a token with a short KV span would read past its slice
into cells belonging to the next token — and in a mixed batch the next token
often belongs to another request.

Each arm gives the victim byte-identical Q, KV and index content; only the
mates change. Per-token attention is independent, so the outputs must match
bitwise.
"""

import numpy as np
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip(
        "exercises the V4 prefill attention kernel; needs a real GPU",
        allow_module_level=True,
    )

from atom.model_ops.v4_kernels.paged_prefill import sparse_attn_v4_paged_prefill

DEV = "cuda"
H = 64  # DeepSeek-V4-Flash: 64 query heads, MQA
# The bf16 OPUS kernel is compiled for D=512 only ("Only D=512 is compiled for
# pa_sparse_prefill_opus_fwd") -- the RoPE dims ride a separate pool on the fp8
# 2buff path, not this one.
D = 512
PAGES = 512
SCALE = 1.0 / (D**0.5)

# The victim's own spans are SHORT; the mate's are LONG. If any loop bound
# came from the batch's widest token instead of this token's own indptr
# delta, the victim would over-read exactly here.
VICTIM_PREFIX = [3, 4, 5, 6]
VICTIM_EXTEND = [1, 2, 3, 4]
MATE_PREFIX = [200, 180, 220, 160]
MATE_EXTEND = [1, 1, 1, 1]


def _indptr(counts):
    v = np.zeros(len(counts) + 1, np.int32)
    v[1:] = np.cumsum(counts)
    return torch.tensor(v, dtype=torch.int32, device=DEV)


def _pool(seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    return torch.randn(PAGES, D, generator=g, dtype=torch.float32, device=DEV).to(
        torch.bfloat16
    )


def _tokens(prefix_counts, extend_counts, seed):
    """Per-token Q rows, prefix slot lists and extend row lists for one group."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    n = len(prefix_counts)
    q = torch.randn(n, H, D, generator=g, dtype=torch.float32, device=DEV).to(
        torch.bfloat16
    )
    kv = torch.randn(n, D, generator=g, dtype=torch.float32, device=DEV).to(
        torch.bfloat16
    )
    # Deterministic, seed-independent slot lists so the victim's prefix cells
    # are the same pool rows in every arm.
    prefix = [
        [(seed * 7 + t * 13 + k) % PAGES for k in range(c)]
        for t, c in enumerate(prefix_counts)
    ]
    # Extend rows are LOCAL to the group; the caller offsets them.
    extend = [[max(t - k, 0) for k in range(c)] for t, c in enumerate(extend_counts)]
    return q, kv, prefix, extend


def _run(groups):
    """groups = list of (q, kv, prefix_lists, extend_lists), concatenated."""
    q = torch.cat([g[0] for g in groups])
    kv = torch.cat([g[1] for g in groups])
    prefix, extend, off = [], [], 0
    for gq, _gkv, gp, ge in groups:
        prefix.extend(gp)
        extend.extend([[r + off for r in rows] for rows in ge])
        off += gq.shape[0]

    p_idx = torch.tensor(
        [c for row in prefix for c in row] or [0], dtype=torch.int32, device=DEV
    )
    e_idx = torch.tensor(
        [c for row in extend for c in row] or [0], dtype=torch.int32, device=DEV
    )
    out = sparse_attn_v4_paged_prefill(
        q,
        _pool(),
        p_idx,
        _indptr([len(r) for r in prefix]),
        kv,
        e_idx,
        _indptr([len(r) for r in extend]),
        torch.zeros(H, dtype=torch.float32, device=DEV),
        SCALE,
    )
    torch.cuda.synchronize()
    return out


def _victim():
    return _tokens(VICTIM_PREFIX, VICTIM_EXTEND, seed=1)


def _mate():
    return _tokens(MATE_PREFIX, MATE_EXTEND, seed=2)


def test_victim_alone_produces_signal():
    """Arms the comparisons below: a zero output would match anything."""
    out = _run([_victim()])
    assert out.float().abs().mean() > 1e-3, "victim attention produced ~nothing"


@pytest.mark.parametrize("where", ["after", "before", "both sides"])
def test_victim_output_is_independent_of_batch_mates(where):
    """The victim's rows must be bitwise equal however the batch is packed."""
    victim = _victim()
    alone = _run([victim])

    if where == "after":
        groups, lo = [_mate(), victim], len(MATE_PREFIX)
    elif where == "before":
        groups, lo = [victim, _mate()], 0
    else:
        groups, lo = [_mate(), victim, _mate()], len(MATE_PREFIX)
    got = _run(groups)[lo : lo + len(VICTIM_PREFIX)]

    diff = (alone.float() - got.float()).abs().max()
    assert torch.equal(alone, got), (
        f"victim rows changed with mates {where}; max|diff|={diff} "
        f"(victim spans prefix={VICTIM_PREFIX} vs mate prefix={MATE_PREFIX})"
    )


# Production magnitudes: a CSA slice runs up to `index_topk` (512) topk cells
# plus a 131-cell SWA window, and a real chunk carries hundreds of tokens, so
# the batch crosses whatever tiling the kernel does over the token axis. The
# spread between victim and mate is the point: the ladder's mixed batch is
# exactly a short-span token sitting next to a long-span one.
REAL_VICTIM_PREFIX = [131, 96, 64, 33, 8, 1] * 6
REAL_VICTIM_EXTEND = [128, 96, 64, 33, 8, 1] * 6
REAL_MATE_PREFIX = [643, 600, 512, 512] * 9
REAL_MATE_EXTEND = [128, 128, 128, 128] * 9


@pytest.mark.parametrize("where", ["after", "before"])
def test_victim_output_is_independent_at_production_span_sizes(where):
    victim = _tokens(REAL_VICTIM_PREFIX, REAL_VICTIM_EXTEND, seed=11)
    mate = _tokens(REAL_MATE_PREFIX, REAL_MATE_EXTEND, seed=12)
    alone = _run([victim])
    assert alone.float().abs().mean() > 1e-3, "victim produced ~nothing"

    if where == "after":
        groups, lo = [mate, victim], len(REAL_MATE_PREFIX)
    else:
        groups, lo = [victim, mate], 0
    got = _run(groups)[lo : lo + len(REAL_VICTIM_PREFIX)]

    bad = (alone != got).any(dim=-1).any(dim=-1).nonzero().flatten().tolist()
    diff = (alone.float() - got.float()).abs().max()
    assert torch.equal(alone, got), (
        f"victim rows changed with a long-span mate {where}: "
        f"tokens {bad[:12]} of {len(REAL_VICTIM_PREFIX)} differ, max|diff|={diff}"
    )
