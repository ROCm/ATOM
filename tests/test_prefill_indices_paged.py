# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Gate for the V4 paged prefill-indices kernel: kernel vs reference, no model.

Exercises `prefix_swa_count > 0`, the cross-request prefix-hit boundary path.
The compress sections stay block-table addressed; only the SWA section is a
per-request ring now, which is why the module keeps its `paged` name.
"""

import numpy as np
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip(
        "compares a Triton kernel against its reference; needs a real GPU",
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
    write_v4_paged_prefill_indices_reference,
)

DEV = "cuda"
WIN = 8
CACHE_SIZE = 11  # SWA ring slots per request
HCA_RATIO = 8  # causal-cap ratio, independent of the pool's class ratios
RATIOS = [DENSE_RATIO, CSA_RATIO, HCA_POOL_RATIO, CSA_RATIO, HCA_POOL_RATIO]
# block_size 256 puts 2 HCA rows in a block, which is what HCA_ROWS_PER_BLOCK
# below exercises; the pool's own ratios stay the real ones.
GEOMETRY = UnifiedPoolGeometry(
    RATIOS, num_blocks=40, num_slots=4, ring_slots=CACHE_SIZE, block_size=256
)


def _indptr(counts, n):
    v = np.zeros(n + 1, np.int64)
    v[1:] = np.cumsum(counts)
    return torch.tensor(v, dtype=torch.int32, device=DEV)


def build(geometry):
    """seq0: fresh chunk (chunk_start=0), pos 0..4 -> prefix_swa_count == 0.
    seq1: prefix-cache hit, chunk_start=16, recompute pos 16..19 -> count > 0."""
    torch.manual_seed(0)
    bid_per_token = torch.tensor([0] * 5 + [1] * 4, dtype=torch.int32, device=DEV)
    positions = torch.tensor(
        [0, 1, 2, 3, 4, 16, 17, 18, 19], dtype=torch.int32, device=DEV
    )
    chunk_start = torch.tensor([0, 16], dtype=torch.int32, device=DEV)
    state_slot = torch.tensor([3, 0], dtype=torch.int32, device=DEV)
    block_tables = torch.randint(1, 40, (2, 8), dtype=torch.int32, device=DEV)
    total = positions.shape[0]

    # Per-token counts, mirroring `_build_paged_prefill_meta`.
    pos = positions.cpu().numpy().astype(np.int64)
    bid = bid_per_token.cpu().numpy()
    cs_pt = chunk_start.cpu().numpy()[bid]
    extend_count = np.minimum(pos - cs_pt + 1, WIN)
    prefix_swa_count = np.maximum(cs_pt - np.maximum(pos - WIN + 1, 0), 0)
    n_hca = (pos + 1) // HCA_RATIO
    csa_head = np.array([2, 1, 0, 3, 2, 1, 0, 2, 1])  # arbitrary; not written here

    ptrs = {
        "extend_indptr": _indptr(extend_count, total),
        "prefix_swa_indptr": _indptr(prefix_swa_count, total),
        "prefix_csa_indptr": _indptr(prefix_swa_count + csa_head, total),
        "prefix_hca_indptr": _indptr(prefix_swa_count + n_hca, total),
    }

    def run(fn):
        bufs = {
            name.replace("_indptr", "_indices"): torch.full(
                (int(p[-1]),), -9, dtype=torch.int32, device=DEV
            )
            for name, p in ptrs.items()
        }
        fn(
            positions=positions,
            bid_per_token=bid_per_token,
            chunk_start_per_seq=chunk_start,
            cu_seqlens_q_per_seq=torch.tensor([0, 5], dtype=torch.int32, device=DEV),
            state_slot_per_seq=state_slot,
            block_tables=block_tables,
            T=total,
            win=WIN,
            geometry=geometry,
            hca_ratio=HCA_RATIO,
            **ptrs,
            **bufs,
        )
        return bufs

    ref = run(write_v4_paged_prefill_indices_reference)
    ker = run(write_v4_paged_prefill_indices)
    torch.cuda.synchronize()
    # The boundary path must actually be exercised, or every assertion below is
    # about the fresh-chunk case only. And two buffers both left at the sentinel
    # compare equal, so check the reference wrote something at all.
    assert prefix_swa_count.max() > 0
    assert not (ref["extend_indices"] == -9).any(), "reference wrote no indices"
    return {"ref": ref, "ker": ker, "ptrs": ptrs, "slot": int(state_slot[1])}


@pytest.fixture(scope="module")
def two_seq():
    return build(GEOMETRY)


# A pool with no dense layer at all — V4-Pro's shape once a DSpark draft moves
# the only ratio-0 layer's window into a state field. The class then leaves the
# geometry, and the builders used to ask for it unconditionally.
NO_DENSE_GEOMETRY = UnifiedPoolGeometry(
    [CSA_RATIO, HCA_POOL_RATIO, CSA_RATIO, HCA_POOL_RATIO],
    num_blocks=40,
    num_slots=4,
    ring_slots=CACHE_SIZE,
    block_size=256,
)


@pytest.fixture(scope="module")
def no_dense():
    return build(NO_DENSE_GEOMETRY)


@pytest.mark.parametrize(
    "section", ["extend_indices", "prefix_csa_indices", "prefix_hca_indices"]
)
def test_the_served_classes_are_unaffected_by_a_missing_one(no_dense, section):
    ref, ker = no_dense["ref"][section], no_dense["ker"][section]
    assert (ref != -9).any(), f"{section} was not written"
    assert torch.equal(ker, ref), f"{section} mismatch\nref={ref}\nker={ker}"


def test_a_missing_class_gets_no_rows_rather_than_borrowed_ones(no_dense):
    """An absent class is launched with a served class's parameters, so what
    this guards against is not a crash but a plausible row: the SWA prefix
    buffer filled with CSA addresses, which no reader downstream would flag."""
    for side in ("ref", "ker"):
        assert (no_dense[side]["prefix_swa_indices"] == -9).all(), side


@pytest.mark.parametrize(
    "section",
    [
        "extend_indices",
        "prefix_swa_indices",
        "prefix_csa_indices",
        "prefix_hca_indices",
    ],
)
def test_kernel_matches_reference(two_seq, section):
    ref, ker = two_seq["ref"][section], two_seq["ker"][section]
    assert torch.equal(ker, ref), f"{section} mismatch\nref={ref}\nker={ker}"


def test_prefix_window_start_maps_to_its_ring_row(two_seq):
    """seq1's first token is pos=16, so its window opens at swa_low=9. That entry
    must be the row the geometry gives for pos 9, not a block-table lookup."""
    expected = GEOMETRY.window_params(DENSE_RATIO).index(two_seq["slot"], 9)
    start = int(two_seq["ptrs"]["prefix_swa_indptr"][5])  # token 5 == seq1's first
    assert int(two_seq["ref"]["prefix_swa_indices"][start]) == expected


# --- coverage for more than one HCA row per block -------------------------
# Regression for the HCA paged-gather bug. With V4 block_size=256, ratio=128 the
# compressor packs hca_rows_per_block = block_size // ratio = 2 HCA entries per physical
# block (cache view [num_blocks, hca_rows_per_block, D]), so entry e -> block
# block_tables[bid, e // rows] at row e % rows -> phys*envelope_rows + row. The
# pre-fix gather assumed one row per block and silently read the wrong blocks.
HCA_ROWS_PER_BLOCK = 256 // 128
_BT_K2 = [5, 9, 13, 17, 21, 25, 29, 33]


def _run_k2(fn, out_hca):
    """One seq, one token at pos=31 -> n_hca = min(32 // 8, 4) = 4 entries
    spanning two physical blocks. chunk_start=0, so there is no SWA prefix and
    the HCA section is the only thing under test."""
    one = lambda v: torch.tensor([v], dtype=torch.int32, device=DEV)
    zero_ptr = torch.zeros(2, dtype=torch.int32, device=DEV)
    fn(
        positions=one(31),
        bid_per_token=one(0),
        chunk_start_per_seq=one(0),
        cu_seqlens_q_per_seq=one(0),
        state_slot_per_seq=one(0),
        block_tables=torch.tensor([_BT_K2], dtype=torch.int32, device=DEV),
        extend_indptr=torch.tensor([0, WIN], dtype=torch.int32, device=DEV),
        prefix_swa_indptr=zero_ptr,
        prefix_csa_indptr=zero_ptr,
        prefix_hca_indptr=torch.tensor([0, 4], dtype=torch.int32, device=DEV),
        extend_indices=torch.full((WIN,), -9, dtype=torch.int32, device=DEV),
        prefix_swa_indices=torch.full((1,), -9, dtype=torch.int32, device=DEV),
        prefix_csa_indices=torch.full((1,), -9, dtype=torch.int32, device=DEV),
        prefix_hca_indices=out_hca,
        T=1,
        win=WIN,
        geometry=GEOMETRY,
        hca_ratio=8,
        hca_rows_per_block=HCA_ROWS_PER_BLOCK,
    )
    return out_hca


@pytest.fixture(scope="module")
def hca_k2():
    empty = lambda: torch.full((4,), -9, dtype=torch.int32, device=DEV)
    ref = _run_k2(write_v4_paged_prefill_indices_reference, empty())
    ker = _run_k2(write_v4_paged_prefill_indices, empty())
    torch.cuda.synchronize()
    return ref, ker


def test_hca_k2_kernel_matches_reference(hca_k2):
    ref, ker = hca_k2
    assert torch.equal(
        ker, ref
    ), f"rows_per_block={HCA_ROWS_PER_BLOCK} HCA kernel != ref\nref={ref}\nker={ker}"


def test_hca_k2_offsets_are_block_packed(hca_k2):
    """Independent oracle: where the compressor actually writes entry e."""
    _, ker = hca_k2
    oracle = torch.tensor(
        [
            _BT_K2[e // HCA_ROWS_PER_BLOCK] * GEOMETRY.envelope_rows
            + e % HCA_ROWS_PER_BLOCK
            for e in range(4)
        ],
        dtype=torch.int32,
        device=DEV,
    )
    assert torch.equal(ker, oracle), (
        f"rows_per_block={HCA_ROWS_PER_BLOCK} HCA compress offset wrong "
        f"(the HCA paged-gather bug)\n"
        f"got={ker.tolist()}\nexp={oracle.tolist()}"
    )
