# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The V4 paged prefill-indices kernel under a bounded-replay floor.

A replaying seq starts from a fresh state slot, so its ring holds nothing below
its replay start. `swa_floor_per_seq` raises `swa_low` to that start; the
kernel and its reference must agree, and agree with the counts the caller sizes
the buffers by.
"""

import numpy as np
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip(
        "compares a Triton kernel against its reference; needs a real GPU",
        allow_module_level=True,
    )

from atom.model_ops.attentions.pool_layout.v4_pool_geometry import (
    CSA_RATIO,
    DENSE_RATIO,
    UnifiedPoolGeometry,
)
from atom.model_ops.attentions.pool_layout.v4_pool_geometry import (
    HCA_RATIO as HCA_POOL_RATIO,
)
from atom.model_ops.v4_kernels.paged_prefill_indices import (
    write_v4_paged_prefill_indices,
    write_v4_paged_prefill_indices_reference,
)

DEV = "cuda"
WIN = 8
HCA_RATIO = 8
GEOMETRY = UnifiedPoolGeometry(
    [DENSE_RATIO, CSA_RATIO, HCA_POOL_RATIO, CSA_RATIO, HCA_POOL_RATIO],
    num_blocks=40,
    num_slots=4,
    ring_slots=11,
    block_size=256,
)


def _indptr(counts, n):
    v = np.zeros(n + 1, np.int64)
    v[1:] = np.cumsum(counts)
    return torch.tensor(v, dtype=torch.int32, device=DEV)


def _run(fn, floor):
    """One seq replaying from `floor`, in a second chunk: chunk_start=16,
    positions 16..19. Without a floor pos 16 reads ring positions 9..15."""
    positions = torch.tensor([16, 17, 18, 19], dtype=torch.int32, device=DEV)
    pos = positions.cpu().numpy().astype(np.int64)
    swa_low = np.maximum(np.maximum(pos - WIN + 1, 0), floor)
    extend_count = np.minimum(pos - 16 + 1, WIN)
    prefix_swa_count = np.maximum(16 - swa_low, 0)
    n_hca = (pos + 1) // HCA_RATIO
    ptrs = {
        "extend_indptr": _indptr(extend_count, 4),
        "prefix_swa_indptr": _indptr(prefix_swa_count, 4),
        "prefix_csa_indptr": _indptr(prefix_swa_count, 4),
        "prefix_hca_indptr": _indptr(prefix_swa_count + n_hca, 4),
    }
    bufs = {
        name.replace("_indptr", "_indices"): torch.full(
            (max(int(p[-1]), 1),), -9, dtype=torch.int32, device=DEV
        )
        for name, p in ptrs.items()
    }
    fn(
        positions=positions,
        bid_per_token=torch.zeros(4, dtype=torch.int32, device=DEV),
        chunk_start_per_seq=torch.tensor([16], dtype=torch.int32, device=DEV),
        cu_seqlens_q_per_seq=torch.tensor([0], dtype=torch.int32, device=DEV),
        state_slot_per_seq=torch.tensor([2], dtype=torch.int32, device=DEV),
        block_tables=torch.arange(1, 9, dtype=torch.int32, device=DEV)[None, :],
        T=4,
        win=WIN,
        geometry=GEOMETRY,
        hca_ratio=HCA_RATIO,
        swa_floor_per_seq=torch.tensor([floor], dtype=torch.int32, device=DEV),
        **ptrs,
        **bufs,
    )
    return bufs, prefix_swa_count


@pytest.mark.parametrize("floor", [0, 12, 16])
def test_kernel_matches_reference_under_a_floor(floor):
    ref, counts = _run(write_v4_paged_prefill_indices_reference, floor)
    ker, _ = _run(write_v4_paged_prefill_indices, floor)
    torch.cuda.synchronize()
    for section in ref:
        assert torch.equal(ker[section], ref[section]), section
    if floor == 0:
        assert list(counts) == [7, 6, 5, 4]
    if floor == 12:
        assert list(counts) == [4, 4, 4, 4]
        # The window opens at the floor, not at pos - WIN + 1 = 9.
        expected = GEOMETRY.window_params(DENSE_RATIO).index(2, 12)
        assert int(ref["prefix_swa_indices"][0]) == expected
    if floor == 16:
        assert counts.sum() == 0


def test_no_floor_reads_from_zero():
    """`swa_floor_per_seq=None` is the pre-replay kernel: same rows as a 0 floor."""
    with_zero, _ = _run(write_v4_paged_prefill_indices, 0)
    torch.cuda.synchronize()
    positions = torch.tensor([16, 17, 18, 19], dtype=torch.int32, device=DEV)
    pos = positions.cpu().numpy().astype(np.int64)
    swa_low = np.maximum(pos - WIN + 1, 0)
    prefix_swa_count = np.maximum(16 - swa_low, 0)
    n_hca = (pos + 1) // HCA_RATIO
    ptrs = {
        "extend_indptr": _indptr(np.minimum(pos - 16 + 1, WIN), 4),
        "prefix_swa_indptr": _indptr(prefix_swa_count, 4),
        "prefix_csa_indptr": _indptr(prefix_swa_count, 4),
        "prefix_hca_indptr": _indptr(prefix_swa_count + n_hca, 4),
    }
    bufs = {
        name.replace("_indptr", "_indices"): torch.full(
            (max(int(p[-1]), 1),), -9, dtype=torch.int32, device=DEV
        )
        for name, p in ptrs.items()
    }
    write_v4_paged_prefill_indices(
        positions=positions,
        bid_per_token=torch.zeros(4, dtype=torch.int32, device=DEV),
        chunk_start_per_seq=torch.tensor([16], dtype=torch.int32, device=DEV),
        cu_seqlens_q_per_seq=torch.tensor([0], dtype=torch.int32, device=DEV),
        state_slot_per_seq=torch.tensor([2], dtype=torch.int32, device=DEV),
        block_tables=torch.arange(1, 9, dtype=torch.int32, device=DEV)[None, :],
        T=4,
        win=WIN,
        geometry=GEOMETRY,
        hca_ratio=HCA_RATIO,
        **ptrs,
        **bufs,
    )
    torch.cuda.synchronize()
    for section, rows in bufs.items():
        assert torch.equal(rows, with_zero[section]), section
