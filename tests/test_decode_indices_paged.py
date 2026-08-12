# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Gate for the V4 paged decode-indices kernel: kernel vs reference, no model.

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
    HCA_RATIO,
    UnifiedPoolGeometry,
)
from atom.model_ops.v4_kernels import hca_compress_paged_offsets
from atom.model_ops.v4_kernels.paged_decode_indices import (
    write_v4_paged_decode_indices,
    write_v4_paged_decode_indices_reference,
)

DEV = "cuda"
WIN = 8
CACHE_SIZE = 11  # ring slots per request; prime-ish to expose the modulo
BS = 3
RATIOS = [DENSE_RATIO, CSA_RATIO, HCA_RATIO, CSA_RATIO, HCA_RATIO, DENSE_RATIO]
GEOMETRY = UnifiedPoolGeometry(
    RATIOS, num_blocks=4, num_slots=6, ring_slots=CACHE_SIZE, block_size=256
)
# One decode token per seq plus a CG-pad token, whose `-1` batch id is the only
# thing keeping every consumer off it. Positions vary so n = min(pos+1, win) and
# windows span multiple blocks (exercises per-window-position block lookup).
POSITIONS = [5, 20, 13, 7]
BATCH_ID = [0, 1, 2, -1]
T = len(BATCH_ID)
# Non-identity slots: a bug that indexes by batch id would still pass on arange.
SLOTS = [3, 0, 4]
CSA_HEAD = [3, 0, 5, 0]
HCA_HEAD = [1, 2, 0, 0]


def build(geometry):
    """Run kernel and reference over one shared decode batch."""
    torch.manual_seed(0)
    positions = torch.tensor(POSITIONS, dtype=torch.int32, device=DEV)
    batch_id_per_token = torch.tensor(BATCH_ID, dtype=torch.int32, device=DEV)
    slots = torch.tensor(SLOTS, dtype=torch.int32, device=DEV)
    n_per = torch.minimum(positions + 1, torch.full_like(positions, WIN)).tolist()

    def indptr(heads):
        # A pad token gets a zero-length slice, exactly as the CPU builders
        # give it.
        v = [0]
        for t in range(T):
            live = BATCH_ID[t] >= 0
            v.append(v[-1] + (heads[t] + n_per[t] if live else 0))
        return torch.tensor(v, dtype=torch.int32, device=DEV)

    ptrs = {
        "swa_indptr": indptr([0] * T),
        "csa_indptr": indptr(CSA_HEAD),
        "hca_indptr": indptr(HCA_HEAD),
    }

    def run(fn):
        # -7 marks "kernel must not touch this": the compress heads are filled
        # elsewhere, so only the SWA tail of each slice should change.
        bufs = {
            name.replace("_indptr", "_indices"): torch.full(
                (int(p[-1]),), -7, dtype=torch.int32, device=DEV
            )
            for name, p in ptrs.items()
        }
        dest = {
            r: torch.full((T,), -7, dtype=torch.int32, device=DEV)
            for r in (DENSE_RATIO, CSA_RATIO, HCA_RATIO)
        }
        fn(
            state_slot_per_seq=slots,
            batch_id_per_token=batch_id_per_token,
            positions=positions,
            dest_rows=dest,
            T=T,
            win=WIN,
            geometry=geometry,
            **ptrs,
            **bufs,
        )
        return {**bufs, "dest": dest}

    ref = run(write_v4_paged_decode_indices_reference)
    ker = run(write_v4_paged_decode_indices)
    torch.cuda.synchronize()
    return {"ref": ref, "ker": ker, "ptrs": ptrs}


@pytest.fixture(scope="module")
def indices():
    out = build(GEOMETRY)
    # Two buffers both left at the sentinel compare equal, so check the SWA
    # section — the one this kernel fills completely — actually got written.
    assert not (out["ref"]["swa_indices"] == -7).any(), "reference wrote no SWA indices"
    return out


@pytest.mark.parametrize("section", ["swa_indices", "csa_indices", "hca_indices"])
def test_kernel_matches_reference(indices, section):
    ref, ker = indices["ref"][section], indices["ker"][section]
    assert torch.equal(ker, ref), f"{section} mismatch\nref={ref}\nker={ker}"


def test_window_start_maps_to_its_ring_row(indices):
    """seq1 pos=20, n=win=8 -> window [13..20]; its first entry must be the row
    the geometry gives for pos 13, not a block-table lookup."""
    expected = GEOMETRY.window_params(DENSE_RATIO).index(SLOTS[1], 13)
    start = int(indices["ptrs"]["swa_indptr"][1])  # seq1 slice (swa head == 0)
    assert int(indices["ref"]["swa_indices"][start]) == expected


@pytest.mark.parametrize("ratio", [DENSE_RATIO, CSA_RATIO, HCA_RATIO])
def test_destination_row_is_the_last_of_this_token_own_window(indices, ratio):
    """The fused SWA write takes the row from here rather than deriving it, so
    it has to be the same row the token's own window position resolves to —
    otherwise the write and the read disagree by exactly one layout change."""
    ker = indices["ker"]["dest"][ratio]
    assert torch.equal(ker, indices["ref"]["dest"][ratio])
    params = GEOMETRY.window_params(ratio)
    for t in range(T):
        b = int(BATCH_ID[t])
        if b < 0:
            # Left at the sentinel the fixture pre-filled: this buffer is
            # defined only where the batch id is, and every consumer gates on
            # the same batch id rather than on the row.
            assert int(ker[t]) == -7, f"token {t} is CG-pad; nothing may write it"
            continue
        assert int(ker[t]) == params.index(SLOTS[b], int(POSITIONS[t])), f"token {t}"


def test_the_three_buffers_disagree_by_class(indices):
    """The buffers used to carry one shared value per token. They must not now:
    each serves a different compress class, whose window rows are interleaved by
    that class's own layer stride."""
    start = int(indices["ptrs"]["swa_indptr"][1])
    csa_start = int(indices["ptrs"]["csa_indptr"][1]) + CSA_HEAD[1]
    hca_start = int(indices["ptrs"]["hca_indptr"][1]) + HCA_HEAD[1]
    swa_row = int(indices["ker"]["swa_indices"][start])
    csa_row = int(indices["ker"]["csa_indices"][csa_start])
    hca_row = int(indices["ker"]["hca_indices"][hca_start])
    assert len({swa_row, csa_row, hca_row}) == 3, (swa_row, csa_row, hca_row)
    for ratio, row in (
        (DENSE_RATIO, swa_row),
        (CSA_RATIO, csa_row),
        (HCA_RATIO, hca_row),
    ):
        assert row == GEOMETRY.window_params(ratio).index(SLOTS[1], 13)


# A pool with no dense layer at all. V4-Pro's trunk is entirely CSA and HCA and
# its one ratio-0 layer is the draft slot, so a draft that carries its window in
# a state field takes the dense class out of the geometry with it. The builders
# used to ask for that class unconditionally and died on a bare KeyError before
# any of this ran.
NO_DENSE_GEOMETRY = UnifiedPoolGeometry(
    [CSA_RATIO, HCA_RATIO, CSA_RATIO, HCA_RATIO],
    num_blocks=4,
    num_slots=6,
    ring_slots=CACHE_SIZE,
    block_size=256,
)


@pytest.fixture(scope="module")
def no_dense():
    return build(NO_DENSE_GEOMETRY)


@pytest.mark.parametrize("section", ["csa_indices", "hca_indices"])
def test_the_served_classes_are_unaffected_by_a_missing_one(no_dense, section):
    ref, ker = no_dense["ref"][section], no_dense["ker"][section]
    # Only the SWA tail of each slice belongs to this kernel; the compress head
    # keeps the sentinel. So the check is that some of it moved, not all.
    assert (ref != -7).any(), f"{section} was not written"
    assert torch.equal(ker, ref), f"{section} mismatch\nref={ref}\nker={ker}"


def test_a_missing_class_gets_no_rows_rather_than_borrowed_ones(no_dense):
    """The parameters an absent class is launched with belong to another class,
    so the failure this guards against is not a crash but a plausible row: the
    SWA buffer filled with CSA addresses, which no reader would flag."""
    for side in ("ref", "ker"):
        assert (no_dense[side]["swa_indices"] == -7).all(), side
        assert (no_dense[side]["dest"][DENSE_RATIO] == -7).all(), side


# --- HCA compress paged offsets with more than one row per block ----------
# Regression for the HCA paged-gather bug. With V4 block_size=256 and ratio=128
# each physical block packs hca_rows_per_block=2 HCA entries, so entry e -> block
# block_tables[bid, e // rows] at row e % rows -> phys*envelope_rows + row.
# The pre-fix math assumed one row per block and read the wrong blocks.
_BT = np.array([[5, 9, 13, 17], [2, 6, 10, 14]], dtype=np.int32)  # [bs, blocks]
_ENTRY = np.array([0, 1, 2, 3, 0, 1, 2], dtype=np.int64)  # seq0: 4, seq1: 3
_BID = np.array([0, 0, 0, 0, 1, 1, 1], dtype=np.int64)
_ENVELOPE_ROWS = 10_000


def test_hca_compress_offsets_are_block_packed():
    hca_rows_per_block = 2
    got = hca_compress_paged_offsets(
        _ENTRY, _BID, _BT, _ENVELOPE_ROWS, hca_rows_per_block
    )
    expected = np.array(
        [
            int(_BT[b][e // hca_rows_per_block]) * _ENVELOPE_ROWS
            + e % hca_rows_per_block
            for e, b in zip(_ENTRY.tolist(), _BID.tolist())
        ],
        dtype=np.int32,
    )
    assert np.array_equal(got, expected), (
        f"rows_per_block={hca_rows_per_block} decode HCA compress offset wrong "
        f"(the HCA paged-gather bug)\n"
        f"got={got.tolist()}\nexp={expected.tolist()}"
    )


def test_one_row_per_block_reduces_to_the_block_stride():
    got = hca_compress_paged_offsets(_ENTRY, _BID, _BT, _ENVELOPE_ROWS, 1)
    expected = np.array(
        [
            int(_BT[b][e]) * _ENVELOPE_ROWS
            for e, b in zip(_ENTRY.tolist(), _BID.tolist())
        ],
        dtype=np.int32,
    )
    assert np.array_equal(
        got, expected
    ), "with one row per block an entry is just its block's first row"
