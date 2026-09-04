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

from atom.model_ops.attentions.pool_layout.v4_pool_geometry import (
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


# --- HCA compress section built in the kernel from the block tables -------
# The section used to be a numpy scatter in the caller, shipped whole every
# step. The kernel fills it now, so what needs holding is that it lands on the
# same rows the numpy oracle names, that head and tail tile each slice with no
# cell left over, and that a caller which does not opt in still gets nothing.
BT_COLS = 16
ROWS_PER_BLOCK = 2


def _hca_case(bs, n_hca_per_seq, positions, geometry=GEOMETRY):
    """One decode token per seq plus a `-1` pad token, with HCA heads sized
    from `n_hca_per_seq` — the same `n + <HCA count>` slice the builder sizes,
    parametrized here so a case can pick the count directly."""
    t_total = bs + 1
    batch_id = list(range(bs)) + [-1]
    rng = np.random.default_rng(bs * 17 + sum(n_hca_per_seq))
    bt_np = rng.integers(0, 4, size=(bs, BT_COLS)).astype(np.int32)
    n_per = [min(p + 1, WIN) for p in positions]

    ptr = [0]
    for t in range(t_total):
        live = batch_id[t] >= 0
        ptr.append(ptr[-1] + (n_per[t] + n_hca_per_seq[t] if live else 0))

    dev = {"dtype": torch.int32, "device": DEV}
    hca_indices = torch.full((ptr[-1],), -7, **dev)
    swa_indptr = torch.tensor(
        np.cumsum([0] + [n_per[t] if batch_id[t] >= 0 else 0 for t in range(t_total)]),
        **dev,
    )
    write_v4_paged_decode_indices(
        state_slot_per_seq=torch.tensor(SLOTS[:1] * bs, **dev),
        batch_id_per_token=torch.tensor(batch_id, **dev),
        positions=torch.tensor(positions, **dev),
        swa_indptr=swa_indptr,
        csa_indptr=None,
        hca_indptr=torch.tensor(ptr, **dev),
        swa_indices=torch.full((int(swa_indptr[-1]),), -7, **dev),
        csa_indices=None,
        hca_indices=hca_indices,
        dest_rows={
            r: torch.full((t_total,), -7, **dev)
            for r in (DENSE_RATIO, CSA_RATIO, HCA_RATIO)
        },
        T=t_total,
        win=WIN,
        geometry=geometry,
        hca_block_tables=torch.from_numpy(bt_np).to(DEV),
        hca_rows_per_block=ROWS_PER_BLOCK,
    )
    torch.cuda.synchronize()
    return hca_indices.cpu().numpy(), ptr, bt_np, n_per


def _oracle_heads(ptr, bt_np, n_hca_per_seq, geometry):
    """The rows the replaced numpy scatter would have written, per slice."""
    out = {}
    for t, n_hca in enumerate(n_hca_per_seq):
        if n_hca == 0:
            continue
        e = np.arange(n_hca, dtype=np.int64)
        out[t] = hca_compress_paged_offsets(
            e,
            np.full(n_hca, t, dtype=np.int64),
            bt_np,
            geometry.envelope_rows,
            ROWS_PER_BLOCK,
        )
    return out


# Counts that cross the two-rows-per-block boundary in both directions, plus a
# seq with nothing committed and a batch bigger than one wave of tokens.
@pytest.mark.parametrize(
    "n_hca_per_seq,positions",
    [
        ([1, 2, 3], [5, 20, 13]),
        ([0, 1, 8], [0, 3, 30]),
        ([31, 2, 0], [40, 1, 9]),
        ([7] * 12, list(range(1, 13))),
    ],
    ids=["small", "one-empty", "wide-and-empty", "twelve-seqs"],
)
def test_hca_head_matches_the_numpy_scatter_it_replaces(n_hca_per_seq, positions):
    bs = len(n_hca_per_seq)
    got, ptr, bt_np, _ = _hca_case(bs, n_hca_per_seq + [0], positions + [0])
    for t, want in _oracle_heads(ptr, bt_np, n_hca_per_seq, GEOMETRY).items():
        head = got[ptr[t] : ptr[t] + len(want)]
        assert np.array_equal(head, want), f"slice {t}: got {head}, want {want}"


@pytest.mark.parametrize(
    "n_hca_per_seq,positions",
    [([1, 2, 3], [5, 20, 13]), ([0, 1, 8], [0, 3, 30]), ([7] * 12, list(range(1, 13)))],
    ids=["small", "one-empty", "twelve-seqs"],
)
def test_head_and_tail_tile_every_slice(n_hca_per_seq, positions):
    """No cell of a live slice keeps the poison, and no pad slice exists.

    This is what lets the caller drop the `-1` pre-fill: if the two sections
    ever stopped meeting, the gap would be a stale index into the pool.
    """
    bs = len(n_hca_per_seq)
    got, ptr, _, _ = _hca_case(bs, n_hca_per_seq + [0], positions + [0])
    assert not (got == -7).any(), f"uncovered cells at {np.flatnonzero(got == -7)}"
    assert ptr[bs + 1] == ptr[bs], "the -1 pad token was given a slice"


def test_without_block_tables_the_head_is_left_alone():
    """The other five callers fill this section themselves, from a different
    row formula, so opting out has to mean the kernel writes none of it."""
    n_hca_per_seq, positions = [3, 4, 5], [5, 20, 13]
    bs = len(n_hca_per_seq)
    t_total = bs + 1
    n_per = [min(p + 1, WIN) for p in positions] + [0]
    batch_id = list(range(bs)) + [-1]
    ptr = [0]
    for t in range(t_total):
        live = batch_id[t] >= 0
        ptr.append(ptr[-1] + ((n_per[t] + (n_hca_per_seq + [0])[t]) if live else 0))

    dev = {"dtype": torch.int32, "device": DEV}
    hca_indices = torch.full((ptr[-1],), -7, **dev)
    swa_indptr = torch.tensor(np.cumsum([0] + n_per), **dev)
    write_v4_paged_decode_indices(
        state_slot_per_seq=torch.tensor(SLOTS[:1] * bs, **dev),
        batch_id_per_token=torch.tensor(batch_id, **dev),
        positions=torch.tensor(positions + [0], **dev),
        swa_indptr=swa_indptr,
        csa_indptr=None,
        hca_indptr=torch.tensor(ptr, **dev),
        swa_indices=torch.full((int(swa_indptr[-1]),), -7, **dev),
        csa_indices=None,
        hca_indices=hca_indices,
        dest_rows={
            r: torch.full((t_total,), -7, **dev)
            for r in (DENSE_RATIO, CSA_RATIO, HCA_RATIO)
        },
        T=t_total,
        win=WIN,
        geometry=GEOMETRY,
    )
    torch.cuda.synchronize()
    got = hca_indices.cpu().numpy()
    for t in range(bs):
        head = got[ptr[t] : ptr[t] + n_hca_per_seq[t]]
        assert (head == -7).all(), f"slice {t} head was written: {head}"
