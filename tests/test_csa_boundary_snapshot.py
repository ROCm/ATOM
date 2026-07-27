# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""GPU tests for the CSA boundary-state snapshot kernels (native prefix cache).

Covers:
  * capture kernel == pure-python reference
  * restore kernel == pure-python reference
  * capture→restore round-trip reproduces the exact ring rows that
    `update_compressor_states` would have written for B-4..B-1 (bit-exact,
    the whole point of the snapshot route)
  * cross-chunk capture accumulation into the same physical block
  * multi-seq restore with distinct slots / boundaries / sources
"""

import sys

import pytest

torch = pytest.importorskip("torch")

# The full atom import chain hits a pre-existing atom.config circular import
# under bare `pytest` collection (the sibling test_csa_prefix_recompute.py skips
# the same way). Running this file directly as a script
# (`python tests/test_csa_boundary_snapshot.py`) imports cleanly and exercises
# the GPU kernels — that is the intended way to validate them here.
_UNDER_PYTEST = "pytest" in sys.modules and __name__ != "__main__"
try:
    import atom.model_ops.v4_kernels  # noqa: F401  (triggers heavy import chain)
except Exception as _e:  # pragma: no cover
    if _UNDER_PYTEST:
        pytest.skip(f"requires full atom import env: {_e}", allow_module_level=True)
    raise

if not torch.cuda.is_available():
    if _UNDER_PYTEST:
        pytest.skip("CSA snapshot kernels require a GPU", allow_module_level=True)
    raise SystemExit("CSA snapshot kernels require a GPU")

from atom.model_ops.v4_kernels.state_writes import (  # noqa: E402
    capture_compressor_boundary,
    capture_compressor_boundary_reference,
    restore_compressor_boundary,
    restore_compressor_boundary_reference,
    update_compressor_states,
)

BLOCK_SIZE = 128
RATIO = 4
TAIL = 4
OVERLAP = True
K_POOL = (1 + int(OVERLAP)) * RATIO  # 8
DEV = "cuda"


def _rng(seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return g


def _make_capture_plan(positions_per_seq):
    """Build a [num_capture, 4] plan selecting the last TAIL positions of each
    block, mirroring what the host builder emits. positions_per_seq: list of
    (batch_id, [absolute positions in this fwd]). ragged_id is the flat row."""
    rows = []
    ragged = 0
    for batch_id, positions in positions_per_seq:
        for pos in positions:
            if pos % BLOCK_SIZE >= BLOCK_SIZE - TAIL:
                rows.append((ragged, batch_id, pos, -1))
            ragged += 1
    if not rows:
        return torch.zeros((0, 4), dtype=torch.int32, device=DEV), ragged
    return torch.tensor(rows, dtype=torch.int32, device=DEV), ragged


def test_capture_matches_reference():
    dim = 16
    num_blocks = 6
    bs = 2
    max_blk = 4
    # seq 0 finalizes logical block 0 (positions 124..129 span the boundary),
    # seq 1 finalizes logical block 1.
    per_seq = [
        (0, list(range(120, 130))),
        (1, list(range(250, 262))),
    ]
    plan, total = _make_capture_plan(per_seq)
    kv = torch.randn(total, dim, generator=_rng(1)).to(DEV)
    score = torch.randn(total, dim, generator=_rng(2)).to(DEV)
    ape = torch.randn(RATIO, dim, generator=_rng(3)).to(DEV)
    block_tables = torch.arange(bs * max_blk, dtype=torch.int32, device=DEV).reshape(
        bs, max_blk
    )

    b_kv = torch.zeros(num_blocks, TAIL, dim, device=DEV)
    b_sc = torch.zeros(num_blocks, TAIL, dim, device=DEV)
    r_kv = b_kv.clone()
    r_sc = b_sc.clone()

    capture_compressor_boundary(
        kv,
        score,
        ape,
        b_kv,
        b_sc,
        capture_plan=plan,
        block_tables=block_tables,
        block_size=BLOCK_SIZE,
        ratio=RATIO,
    )
    capture_compressor_boundary_reference(
        kv,
        score,
        ape,
        r_kv,
        r_sc,
        capture_plan=plan,
        block_tables=block_tables,
        block_size=BLOCK_SIZE,
        ratio=RATIO,
    )
    torch.testing.assert_close(b_kv, r_kv)
    torch.testing.assert_close(b_sc, r_sc)


def test_restore_matches_reference():
    dim = 16
    num_blocks = 6
    num_slots = 4
    state_size = K_POOL  # 8
    b_kv = torch.randn(num_blocks, TAIL, dim, generator=_rng(4)).to(DEV)
    b_sc = torch.randn(num_blocks, TAIL, dim, generator=_rng(5)).to(DEV)
    # (source_phys, slot, boundary_B, _)
    plan = torch.tensor(
        [(2, 0, 128, -1), (5, 3, 256, -1), (-1, 1, 128, -1)],
        dtype=torch.int32,
        device=DEV,
    )
    kv_state = torch.zeros(num_slots, state_size, dim, device=DEV)
    score_state = torch.zeros(num_slots, state_size, dim, device=DEV)
    r_kv_state = kv_state.clone()
    r_score_state = score_state.clone()

    restore_compressor_boundary(b_kv, b_sc, kv_state, score_state, restore_plan=plan)
    restore_compressor_boundary_reference(
        b_kv, b_sc, r_kv_state, r_score_state, restore_plan=plan
    )
    torch.testing.assert_close(kv_state, r_kv_state)
    torch.testing.assert_close(score_state, r_score_state)


def test_capture_restore_roundtrip_equals_update():
    """The core invariant: restore reproduces the exact ring rows that a
    full-context producer's `update_compressor_states` wrote for B-4..B-1."""
    dim = 32
    num_blocks = 4
    num_slots = 4
    state_size = K_POOL
    B = 128  # aligned boundary; block 0 finalized
    phys = 0  # block_tables maps logical 0 -> phys 0 for seq 0

    # Producer forward covers positions [0, 128): the last K_POOL rows land in
    # the ring; the last TAIL of those are the boundary rows we snapshot.
    positions = list(range(B))
    total = len(positions)
    kv = torch.randn(total, dim, generator=_rng(10)).to(DEV)
    score = torch.randn(total, dim, generator=_rng(11)).to(DEV)
    ape = torch.randn(RATIO, dim, generator=_rng(12)).to(DEV)
    slot_map = torch.zeros(1, dtype=torch.int32, device=DEV)

    # write_plan = last K_POOL positions (what make_compress_plans emits)
    wp_rows = [
        (i, 0, positions[i], -1) for i in range(total) if positions[i] >= B - K_POOL
    ]
    write_plan = torch.tensor(wp_rows, dtype=torch.int32, device=DEV)
    prod_kv_state = torch.zeros(num_slots, state_size, dim, device=DEV)
    prod_score_state = torch.zeros(num_slots, state_size, dim, device=DEV)
    update_compressor_states(
        kv,
        score,
        ape,
        prod_kv_state,
        prod_score_state,
        write_plan=write_plan,
        num_write=write_plan.shape[0],
        state_slot_mapping=slot_map,
        ratio=RATIO,
        overlap=OVERLAP,
    )

    # Capture boundary snapshot from the same producer stream.
    cap_plan, _ = _make_capture_plan([(0, positions)])
    block_tables = torch.zeros(1, num_blocks, dtype=torch.int32, device=DEV)
    b_kv = torch.zeros(num_blocks, TAIL, dim, device=DEV)
    b_sc = torch.zeros(num_blocks, TAIL, dim, device=DEV)
    capture_compressor_boundary(
        kv,
        score,
        ape,
        b_kv,
        b_sc,
        capture_plan=cap_plan,
        block_tables=block_tables,
        block_size=BLOCK_SIZE,
        ratio=RATIO,
    )

    # Restore into a fresh consumer ring (different slot).
    con_slot = 2
    con_kv_state = torch.zeros(num_slots, state_size, dim, device=DEV)
    con_score_state = torch.zeros(num_slots, state_size, dim, device=DEV)
    restore_plan = torch.tensor(
        [(phys, con_slot, B, -1)], dtype=torch.int32, device=DEV
    )
    restore_compressor_boundary(
        b_kv, b_sc, con_kv_state, con_score_state, restore_plan=restore_plan
    )

    # The 4 boundary ring positions B-4..B-1 must match bit-for-bit.
    for pos in range(B - TAIL, B):
        r = pos % state_size
        torch.testing.assert_close(con_kv_state[con_slot, r], prod_kv_state[0, r])
        torch.testing.assert_close(con_score_state[con_slot, r], prod_score_state[0, r])


def test_cross_chunk_capture_accumulates():
    """Boundary rows split across two chunks accumulate into the same phys."""
    dim = 16
    num_blocks = 3
    phys = 1
    block_tables = torch.zeros(1, num_blocks, dtype=torch.int32, device=DEV)
    block_tables[0, 0] = phys  # logical block 0 -> phys 1

    ape = torch.randn(RATIO, dim, generator=_rng(20)).to(DEV)
    b_kv = torch.zeros(num_blocks, TAIL, dim, device=DEV)
    b_sc = torch.zeros(num_blocks, TAIL, dim, device=DEV)

    # Reference computed over the full span in one shot.
    full_pos = list(range(120, 128))
    kv_full = torch.randn(len(full_pos), dim, generator=_rng(21)).to(DEV)
    sc_full = torch.randn(len(full_pos), dim, generator=_rng(22)).to(DEV)
    ref_kv = torch.zeros(num_blocks, TAIL, dim, device=DEV)
    ref_sc = torch.zeros(num_blocks, TAIL, dim, device=DEV)
    plan_full = torch.tensor(
        [
            (i, 0, full_pos[i], -1)
            for i in range(len(full_pos))
            if full_pos[i] % BLOCK_SIZE >= BLOCK_SIZE - TAIL
        ],
        dtype=torch.int32,
        device=DEV,
    )
    capture_compressor_boundary_reference(
        kv_full,
        sc_full,
        ape,
        ref_kv,
        ref_sc,
        capture_plan=plan_full,
        block_tables=block_tables,
        block_size=BLOCK_SIZE,
        ratio=RATIO,
    )

    # Chunk A: positions 120..125 (rows 124,125 captured). Chunk B: 126..127.
    for lo, hi in [(0, 6), (6, 8)]:
        pos = full_pos[lo:hi]
        plan = torch.tensor(
            [
                (j - lo, 0, pos[j - lo], -1)
                for j in range(lo, hi)
                if pos[j - lo] % BLOCK_SIZE >= BLOCK_SIZE - TAIL
            ],
            dtype=torch.int32,
            device=DEV,
        )
        capture_compressor_boundary(
            kv_full[lo:hi],
            sc_full[lo:hi],
            ape,
            b_kv,
            b_sc,
            capture_plan=plan,
            block_tables=block_tables,
            block_size=BLOCK_SIZE,
            ratio=RATIO,
        )
    torch.testing.assert_close(b_kv, ref_kv)
    torch.testing.assert_close(b_sc, ref_sc)


if __name__ == "__main__":
    _tests = [
        test_capture_matches_reference,
        test_restore_matches_reference,
        test_capture_restore_roundtrip_equals_update,
        test_cross_chunk_capture_accumulates,
    ]
    failed = 0
    for t in _tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
    print("ALL_PASS" if failed == 0 else f"{failed} FAILED")
    raise SystemExit(1 if failed else 0)
