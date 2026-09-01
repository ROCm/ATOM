# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""A sequence's prefill indices must not depend on its batch-mates.

`test_prefill_indices_paged.py` pins this kernel to its reference. That cannot
settle the contract: both derive a token's slice from the same batch-wide
cumsums, so a rule that is wrong about WHERE a token's cells start would move
both together and they would still agree.

The property that matters here is composition invariance, and this kernel is
where the checkpoint ladder's two batch shapes actually differ:

    prefix_swa_count[t] = max(chunk_start[bid] - swa_low[t], 0)

is non-zero ONLY for a resumed chunk. A batch of fresh prompts has it zero
everywhere, a batch of resumed chunks has it non-zero everywhere, and only a
MIXED batch carries both — while every count feeds one cumsum shared by the
whole batch. End-to-end, accuracy tracks the fraction of prefill batches that
are mixed, so this is the shape under suspicion.

Both directions are covered: a fresh sequence behind a resumed one (the fresh
one owns no SWA prefix, so its extend/HCA sections carry the signal), and a
resumed one behind a fresh one (which exercises the SWA prefix itself).
"""

import numpy as np
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip(
        "exercises a Triton index-builder on real tensors; needs a GPU",
        allow_module_level=True,
    )

from atom.model_ops.attentions.v4_pool_geometry import (
    CSA_RATIO,
    DENSE_RATIO,
    UnifiedPoolGeometry,
)
from atom.model_ops.attentions.v4_pool_geometry import HCA_RATIO as HCA_POOL_RATIO
from atom.model_ops.v4_kernels.paged_prefill_indices import (
    write_v4_paged_prefill_indices,
)

DEV = "cuda"
WIN = 8
CACHE_SIZE = 11
HCA_RATIO = 8
RATIOS = [DENSE_RATIO, CSA_RATIO, HCA_POOL_RATIO, CSA_RATIO, HCA_POOL_RATIO]
GEOMETRY = UnifiedPoolGeometry(
    RATIOS, num_blocks=40, num_slots=4, ring_slots=CACHE_SIZE, block_size=256
)
BT_WIDTH = 8

# Every sequence is (chunk_start, n_tokens, state_slot, block_table_row). The
# block table row and state slot travel WITH the sequence, so a victim reads
# the same pool rows no matter which batch position it lands in.
FRESH = (0, 5, 3, [5, 9, 13, 17, 21, 25, 29, 33])
RESUMED = (16, 4, 0, [2, 6, 10, 14, 18, 22, 26, 30])
RESUMED_LONG = (24, 6, 1, [3, 7, 11, 15, 19, 23, 27, 31])
FRESH_MATE = (0, 7, 2, [4, 8, 12, 16, 20, 24, 28, 32])


def _csa_head(pos):
    """Per-token CSA reservation. A pure function of the token's own position,
    so it cannot smuggle batch composition into the expected layout."""
    return int(pos) % 4


def _run(seqs):
    """Build and run one batch; return the buffers plus each seq's token span."""
    starts, lens, slots, bts = zip(*seqs)
    bid = np.repeat(np.arange(len(seqs), dtype=np.int32), lens)
    pos = np.concatenate([np.arange(s, s + n) for s, n in zip(starts, lens)])
    cs_pt = np.asarray(starts, dtype=np.int64)[bid]

    extend_count = np.minimum(pos - cs_pt + 1, WIN)
    prefix_swa_count = np.maximum(cs_pt - np.maximum(pos - WIN + 1, 0), 0)
    n_hca = (pos + 1) // HCA_RATIO
    csa_head = np.asarray([_csa_head(p) for p in pos])

    total = len(pos)
    cu = np.zeros(len(seqs), dtype=np.int32)
    cu[1:] = np.cumsum(lens)[:-1]

    def _ptr(counts):
        v = np.zeros(total + 1, np.int64)
        v[1:] = np.cumsum(counts)
        return torch.tensor(v, dtype=torch.int32, device=DEV)

    ptrs = {
        "extend_indptr": _ptr(extend_count),
        "prefix_swa_indptr": _ptr(prefix_swa_count),
        "prefix_csa_indptr": _ptr(prefix_swa_count + csa_head),
        "prefix_hca_indptr": _ptr(prefix_swa_count + n_hca),
    }
    bufs = {
        name.replace("_indptr", "_indices"): torch.full(
            (max(int(p[-1]), 1),), -9, dtype=torch.int32, device=DEV
        )
        for name, p in ptrs.items()
    }
    write_v4_paged_prefill_indices(
        positions=torch.tensor(pos, dtype=torch.int32, device=DEV),
        bid_per_token=torch.tensor(bid, dtype=torch.int32, device=DEV),
        chunk_start_per_seq=torch.tensor(starts, dtype=torch.int32, device=DEV),
        cu_seqlens_q_per_seq=torch.tensor(cu, dtype=torch.int32, device=DEV),
        state_slot_per_seq=torch.tensor(slots, dtype=torch.int32, device=DEV),
        block_tables=torch.tensor(bts, dtype=torch.int32, device=DEV),
        T=total,
        win=WIN,
        geometry=GEOMETRY,
        hca_ratio=HCA_RATIO,
        **ptrs,
        **bufs,
    )
    torch.cuda.synchronize()
    spans = []
    off = 0
    for n in lens:
        spans.append((off, off + n))
        off += n
    return ptrs, bufs, spans, cu


def _slices(ptrs, bufs, span, section):
    """The victim's per-token cell lists, one list per token."""
    lo, hi = span
    p = ptrs[section.replace("_indices", "_indptr")].cpu().numpy()
    b = bufs[section].cpu().numpy()
    return [b[p[t] : p[t + 1]].tolist() for t in range(lo, hi)]


@pytest.mark.parametrize(
    "victim,mates,label",
    [
        (FRESH, [RESUMED], "fresh behind a resumed chunk"),
        (FRESH, [RESUMED_LONG], "fresh behind a longer resumed chunk"),
        (FRESH, [RESUMED, FRESH_MATE], "fresh behind resumed + fresh"),
        (RESUMED, [FRESH], "resumed behind a fresh prompt"),
        (RESUMED, [FRESH_MATE, FRESH], "resumed behind two fresh prompts"),
    ],
)
def test_a_sequences_indices_do_not_move_with_its_batch_mates(victim, mates, label):
    """Same sequence, same slot, same blocks — only the batch-mates change."""
    p_alone, b_alone, s_alone, _ = _run([victim])
    p_with, b_with, s_with, cu_with = _run([*mates, victim])
    v_at = len(mates)

    armed = False
    for section in (
        "extend_indices",
        "prefix_swa_indices",
        "prefix_csa_indices",
        "prefix_hca_indices",
    ):
        alone = _slices(p_alone, b_alone, s_alone[0], section)
        with_mates = _slices(p_with, b_with, s_with[v_at], section)
        if section == "extend_indices":
            # These are offsets into the batch's CONCATENATED token stream, so
            # the victim's own cells legitimately shift by its cu_seqlens_q.
            # Everything else indexes the pool and must be absolute-equal.
            shift = int(cu_with[v_at])
            with_mates = [[c - shift for c in cells] for cells in with_mates]
        if any(cells for cells in alone):
            armed = True
        assert alone == with_mates, (
            f"{label}: {section} moved for the victim\n"
            f"  alone      = {alone}\n"
            f"  with mates = {with_mates}"
        )
    assert armed, f"{label}: every section was empty; the case proves nothing"


def test_a_resumed_chunk_really_reserves_swa_prefix_cells():
    """Arms the suite: the mixed shape must actually produce both count kinds."""
    ptrs, bufs, spans, _ = _run([FRESH, RESUMED])
    fresh = _slices(ptrs, bufs, spans[0], "prefix_swa_indices")
    resumed = _slices(ptrs, bufs, spans[1], "prefix_swa_indices")
    assert all(not cells for cells in fresh), "a fresh chunk must own no SWA prefix"
    assert any(cells for cells in resumed), (
        "a resumed chunk must own SWA prefix cells, or the mixed shape this "
        "whole file is about never occurs in the fixture"
    )
