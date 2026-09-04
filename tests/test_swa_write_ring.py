# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Gate for the SWA ring `swa_write`: kernel vs reference, no model.

Validates the ring addressing — the last-N tokens of each seq land where
`v4_pool_geometry` says they should — and that the Triton kernel matches the
pure-PyTorch reference.

Every case runs once per compress class. A class's layer stride is what forces
the window to be chopped and interleaved, so a formula that happened to work
for the wide stride can still be wrong for the narrow one; the classes are the
axis along which this kernel actually varies.

What this file deliberately does NOT test: cross-request reuse. Under the paged
predecessor two seqs sharing a physical block wrote SWA to the same rows, and
that was the property #1417 needed. A ring is private by construction, so reuse
is no longer a property of the write at all: it comes from the checkpoint copy
that carries the ring into the resuming request's slot. Testing it here would
assert something the write cannot provide.
"""

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
from atom.model_ops.v4_kernels.state_writes import (
    swa_scatter_rows,
    swa_scatter_rows_reference,
    swa_write,
    swa_write_reference,
)

DEV = "cuda"
BS = 3
RING_SLOTS = 11  # real V4 = window + max_spec_steps; prime-ish to expose modulo
HEAD_DIM = 16
NUM_SLOTS = 5
RATIOS = [DENSE_RATIO, CSA_RATIO, HCA_RATIO, CSA_RATIO, HCA_RATIO, DENSE_RATIO]
BLOCK_SIZE = 256
# Per-seq token counts this fwd, and global positions. `START_POS` is chosen so
# seq 0 stays inside the first lap, seq 1 straddles a wrap, and seq 2 is several
# laps in — the case a paged table never exercised.
TOK_COUNTS = [RING_SLOTS + 3, 5, RING_SLOTS * 2]
START_POS = [0, RING_SLOTS - 2, 4 * RING_SLOTS + 3]
# Non-identity, non-contiguous slot assignment: a bug that ignores the slot and
# uses batch_idx would still pass with slots == arange.
SLOTS = [3, 0, 4]


@pytest.fixture(scope="module")
def geometry():
    return UnifiedPoolGeometry(
        RATIOS,
        num_blocks=2,
        num_slots=NUM_SLOTS,
        ring_slots=RING_SLOTS,
        block_size=BLOCK_SIZE,
    )


@pytest.fixture(scope="module")
def batch():
    torch.manual_seed(0)
    cu = torch.zeros(BS + 1, dtype=torch.int32, device=DEV)
    cu[1:] = torch.cumsum(torch.tensor(TOK_COUNTS, dtype=torch.int32), 0)
    total = int(cu[-1])
    positions = torch.cat(
        [
            torch.arange(START_POS[b], START_POS[b] + TOK_COUNTS[b], dtype=torch.int32)
            for b in range(BS)
        ]
    ).to(DEV)
    return {
        "cu": cu,
        "positions": positions,
        "kv": torch.randn(total, HEAD_DIM, dtype=torch.bfloat16, device=DEV),
        "slots": torch.tensor(SLOTS, dtype=torch.int32, device=DEV),
    }


@pytest.fixture(scope="module")
def written(geometry, batch):
    """Run both implementations once per class over the shared batch."""
    out = {}
    for ratio in (DENSE_RATIO, CSA_RATIO, HCA_RATIO):
        params = geometry.window_params(ratio)
        plane = (geometry.plane_rows, HEAD_DIM)
        ref = torch.zeros(*plane, dtype=torch.bfloat16, device=DEV)
        got = torch.zeros(*plane, dtype=torch.bfloat16, device=DEV)
        # Capped at the ring: beyond it a seq's own tokens collide (see swa_write).
        args = (batch["kv"], batch["positions"], batch["cu"], batch["slots"])
        swa_write_reference(*args, ref, params, RING_SLOTS)
        swa_write(*args, got, params, RING_SLOTS)
        out[ratio] = {"got": got, "ref": ref, "params": params}
    torch.cuda.synchronize()
    return out


RATIO_IDS = [DENSE_RATIO, CSA_RATIO, HCA_RATIO]


@pytest.mark.parametrize("ratio", RATIO_IDS)
def test_kernel_matches_reference(written, ratio):
    got, ref = written[ratio]["got"], written[ratio]["ref"]
    assert torch.equal(
        got, ref
    ), f"kernel != reference; max|diff|={(got.float() - ref.float()).abs().max()}"


@pytest.mark.parametrize("ratio", RATIO_IDS)
def test_last_token_lands_where_the_geometry_says(written, batch, ratio):
    """Spot-check a known mapping: seq 2's last token, several laps in."""
    b = 2
    last_pos = START_POS[b] + TOK_COUNTS[b] - 1
    row = written[ratio]["params"].index(SLOTS[b], last_pos)
    assert torch.equal(
        written[ratio]["got"][row], batch["kv"][int(batch["cu"][b + 1]) - 1]
    )


@pytest.mark.parametrize("ratio", RATIO_IDS)
def test_nothing_outside_the_written_windows_is_touched(written, ratio):
    """The strong form of the old unowned-slot check: enumerate exactly the rows
    the three writing seqs may reach, and require every other row of the plane —
    including the compress region — to be untouched. Under a flat overlapping
    view a stray row is a KV block, not just another request's window."""
    params = written[ratio]["params"]
    allowed = {params.index(slot, pos) for slot in SLOTS for pos in range(RING_SLOTS)}
    live = (written[ratio]["got"].abs().sum(-1) > 0).nonzero().flatten().tolist()
    assert set(live) <= allowed, sorted(set(live) - allowed)


@pytest.mark.parametrize("ratio", RATIO_IDS)
def test_a_wrapping_seq_leaves_exactly_one_ring_live(written, ratio):
    """More than `ring_slots` tokens must overwrite the seq's OWN older rows."""
    params = written[ratio]["params"]
    rows = [params.index(SLOTS[2], pos) for pos in range(RING_SLOTS)]
    block = written[ratio]["got"][rows]
    live = int((block.abs().sum(-1) > 0).sum())
    assert live == RING_SLOTS, f"seq2 wrote {TOK_COUNTS[2]} tokens, {live} rows live"


def test_over_wide_write_is_rejected(written, batch):
    """Must fail loudly, not race. The one contract the paged predecessor did
    not need: block addressing was injective on position, a ring is not."""
    with pytest.raises(AssertionError, match="exceeds the ring"):
        swa_write(
            batch["kv"],
            batch["positions"],
            batch["cu"],
            batch["slots"],
            written[CSA_RATIO]["got"],
            written[CSA_RATIO]["params"],
            RING_SLOTS + 1,
        )


@pytest.mark.parametrize("ratio", RATIO_IDS)
def test_scatter_rows_lands_where_the_window_formula_says(geometry, batch, ratio):
    """`swa_scatter_rows` is the decode counterpart: the caller hands in the row
    instead of the kernel deriving it, so what has to be pinned is that a row
    built from `WindowParams` still reaches the same place — and that a `-1`
    row or a `-1` batch id writes nothing."""
    params = geometry.window_params(ratio)
    kv = batch["kv"]
    total = kv.shape[0]
    # One row per token from the geometry, with two tokens deliberately
    # disabled: one by its destination, one by the CG-pad sentinel.
    positions = batch["positions"]
    bid = torch.zeros(total, dtype=torch.int32, device=DEV)
    for b in range(BS):
        bid[int(batch["cu"][b]) : int(batch["cu"][b + 1])] = b
    dest = torch.tensor(
        [
            params.index(SLOTS[int(bid[t])], int(positions[t]) % RING_SLOTS)
            for t in range(total)
        ],
        dtype=torch.int32,
        device=DEV,
    )
    # Two tokens of one seq more than a ring apart share a row, and which of
    # them wins is a race in both implementations. Decode never produces that
    # (a seq contributes at most `1 + mtp_k` consecutive positions), so drop all
    # but the last writer rather than pinning an order nobody guarantees.
    last = {int(dest[t]): t for t in range(total)}
    for t in range(total):
        if last[int(dest[t])] != t:
            dest[t] = -1
    dest[1] = -1
    bid[2] = -1

    plane = (geometry.plane_rows, HEAD_DIM)
    got = torch.zeros(*plane, dtype=torch.bfloat16, device=DEV)
    ref = torch.zeros(*plane, dtype=torch.bfloat16, device=DEV)
    swa_scatter_rows(kv, dest, bid, got)
    swa_scatter_rows_reference(kv, dest, bid, ref)
    torch.cuda.synchronize()
    assert torch.equal(got, ref)
    live = {t for t in range(total) if t != 2 and int(dest[t]) >= 0}
    assert live, "the batch disabled every token; the check proves nothing"
    for t in live:
        assert torch.equal(got[int(dest[t])], kv[t]), f"token {t}"
    touched = set((got.abs().sum(-1) > 0).nonzero().flatten().tolist())
    assert touched <= {int(dest[t]) for t in live}


def _empty_plane(geometry):
    return torch.zeros(geometry.plane_rows, HEAD_DIM, dtype=torch.bfloat16, device=DEV)


def test_a_slot_past_the_plane_writes_nothing(geometry, batch):
    """A bad slot must drop its write, not land elsewhere in the pool.

    Why that matters is in `_swa_write_kernel`. Not a reference comparison:
    `swa_write_reference` indexes with torch, which raises on an out-of-range
    row instead of skipping, so the two disagree here by construction.
    """
    params = geometry.window_params(DENSE_RATIO)
    plane = _empty_plane(geometry)
    # One past the last slot the plane numbers, so `slot * slot_rows` lands
    # exactly one slot beyond the end — positive, which a `slot < 0` guard
    # would wave through.
    bad = torch.full(
        (BS,), geometry.plane_rows // params.slot_rows, dtype=torch.int32, device=DEV
    )
    args = (batch["kv"], batch["positions"], batch["cu"], bad)
    swa_write(*args, plane, params, RING_SLOTS)
    torch.cuda.synchronize()
    assert not plane.any(), "an out-of-plane slot still wrote into the pool"


def test_scatter_rows_ignores_a_row_past_the_plane(geometry, batch):
    """Same bound on the decode counterpart, which only checked `row < 0`."""
    total = int(batch["cu"][-1])
    plane = _empty_plane(geometry)
    dest = torch.full((total,), geometry.plane_rows, dtype=torch.int32, device=DEV)
    bid = torch.zeros(total, dtype=torch.int32, device=DEV)
    swa_scatter_rows(batch["kv"], dest, bid, plane)
    torch.cuda.synchronize()
    assert not plane.any(), "an out-of-plane dest row still wrote into the pool"
