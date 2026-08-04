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

from atom.model_ops.v4_kernels import hca_compress_paged_offsets
from atom.model_ops.v4_kernels.paged_decode_indices import (
    write_v4_paged_decode_indices,
    write_v4_paged_decode_indices_reference,
)

DEV = "cuda"
WIN = 8
CACHE_SIZE = 11  # ring slots per request; prime-ish to expose the modulo
BS = 3
# One decode token per seq (T == BS); positions vary so n = min(pos+1, win) and
# windows span multiple blocks (exercises per-window-position block lookup).
POSITIONS = [5, 20, 13]
# Non-identity slots: a bug that indexes by batch id would still pass on arange.
SLOTS = [3, 0, 4]
CSA_HEAD = [3, 0, 5]
HCA_HEAD = [1, 2, 0]


@pytest.fixture(scope="module")
def indices():
    """Run kernel and reference over one shared decode batch."""
    torch.manual_seed(0)
    positions = torch.tensor(POSITIONS, dtype=torch.int32, device=DEV)
    batch_id_per_token = torch.arange(BS, dtype=torch.int32, device=DEV)
    slots = torch.tensor(SLOTS, dtype=torch.int32, device=DEV)
    n_per = torch.minimum(positions + 1, torch.full_like(positions, WIN)).tolist()

    def indptr(heads):
        v = [0]
        for t in range(BS):
            v.append(v[-1] + heads[t] + n_per[t])
        return torch.tensor(v, dtype=torch.int32, device=DEV)

    ptrs = {
        "swa_indptr": indptr([0] * BS),
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
        fn(
            state_slot_per_seq=slots,
            batch_id_per_token=batch_id_per_token,
            positions=positions,
            T=BS,
            win=WIN,
            cache_size=CACHE_SIZE,
            **ptrs,
            **bufs,
        )
        return bufs

    ref = run(write_v4_paged_decode_indices_reference)
    ker = run(write_v4_paged_decode_indices)
    torch.cuda.synchronize()
    # Two buffers both left at the sentinel compare equal, so check the SWA
    # section — the one this kernel fills completely — actually got written.
    assert not (ref["swa_indices"] == -7).any(), "reference wrote no SWA indices"
    return {"ref": ref, "ker": ker, "ptrs": ptrs}


@pytest.mark.parametrize("section", ["swa_indices", "csa_indices", "hca_indices"])
def test_kernel_matches_reference(indices, section):
    ref, ker = indices["ref"][section], indices["ker"][section]
    assert torch.equal(ker, ref), f"{section} mismatch\nref={ref}\nker={ker}"


def test_window_start_maps_to_its_ring_row(indices):
    """seq1 pos=20, n=win=8 -> window [13..20]; its first entry must be the ring
    row for pos 13, not a block-table lookup."""
    expected = SLOTS[1] * CACHE_SIZE + 13 % CACHE_SIZE
    start = int(indices["ptrs"]["swa_indptr"][1])  # seq1 slice (swa head == 0)
    assert int(indices["ref"]["swa_indices"][start]) == expected


# --- HCA compress paged offsets at k2_hca > 1 -------------------------------
# Regression for the HCA paged-gather bug. With V4 block_size=256 and ratio=128
# each physical block packs k2_hca=2 HCA entries, so entry e -> block
# block_tables[bid, e//k2], slot e%k2 -> swa_pages + phys*k2 + s. The pre-fix
# math used swa_pages + block_tables[bid, e] (i.e. assumed k2 == 1) and read the
# wrong blocks.
_BT = np.array([[5, 9, 13, 17], [2, 6, 10, 14]], dtype=np.int32)  # [bs, blocks]
_ENTRY = np.array([0, 1, 2, 3, 0, 1, 2], dtype=np.int64)  # seq0: 4, seq1: 3
_BID = np.array([0, 0, 0, 0, 1, 1, 1], dtype=np.int64)
_SWA_PAGES = 10_000


def test_hca_compress_offsets_are_block_packed():
    k2 = 2
    got = hca_compress_paged_offsets(_ENTRY, _BID, _BT, _SWA_PAGES, k2)
    expected = np.array(
        [
            _SWA_PAGES + int(_BT[b][e // k2]) * k2 + e % k2
            for e, b in zip(_ENTRY.tolist(), _BID.tolist())
        ],
        dtype=np.int32,
    )
    assert np.array_equal(got, expected), (
        f"k2={k2} decode HCA compress offset wrong (the HCA paged-gather bug)\n"
        f"got={got.tolist()}\nexp={expected.tolist()}"
    )


def test_hca_compress_offsets_reduce_to_legacy_at_k2_one():
    got = hca_compress_paged_offsets(_ENTRY, _BID, _BT, _SWA_PAGES, 1)
    expected = np.array(
        [_SWA_PAGES + int(_BT[b][e]) for e, b in zip(_ENTRY.tolist(), _BID.tolist())],
        dtype=np.int32,
    )
    assert np.array_equal(got, expected), "k2==1 must equal legacy swa_pages + bt"
