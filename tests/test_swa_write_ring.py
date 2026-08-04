# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Gate for the SWA ring `swa_write`: kernel vs reference, no model.

Validates the ring addressing — the last-N tokens of each seq land at
`slot * cache_size + pos % cache_size` — and that the Triton kernel matches the
pure-PyTorch reference.

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

from atom.model_ops.v4_kernels.state_writes import swa_write, swa_write_reference

DEV = "cuda"
BS = 3
CACHE_SIZE = 11  # real V4 = window + max_spec_steps; prime-ish to expose modulo
HEAD_DIM = 16
NUM_SLOTS = 5
NUM_ROWS = NUM_SLOTS * CACHE_SIZE
# Per-seq token counts this fwd, and global positions. `START_POS` is chosen so
# seq 0 stays inside the first lap, seq 1 straddles a wrap, and seq 2 is several
# laps in — the case a paged table never exercised.
TOK_COUNTS = [CACHE_SIZE + 3, 5, CACHE_SIZE * 2]
START_POS = [0, CACHE_SIZE - 2, 4 * CACHE_SIZE + 3]
# Non-identity, non-contiguous slot assignment: a bug that ignores the slot and
# uses batch_idx would still pass with slots == arange.
SLOTS = [3, 0, 4]


@pytest.fixture(scope="module")
def written():
    """Run both implementations once over a shared batch.

    Module-scoped: every assertion below reads the same write, so re-running the
    kernel per test would only add launch time and a chance for the two to drift
    apart between cases.
    """
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
    kv = torch.randn(total, HEAD_DIM, dtype=torch.bfloat16, device=DEV)
    slots = torch.tensor(SLOTS, dtype=torch.int32, device=DEV)

    ref = torch.zeros(NUM_ROWS, HEAD_DIM, dtype=torch.bfloat16, device=DEV)
    out = torch.zeros(NUM_ROWS, HEAD_DIM, dtype=torch.bfloat16, device=DEV)
    # Capped at the ring: beyond it a seq's own tokens collide (see swa_write).
    swa_write_reference(kv, positions, cu, slots, ref, CACHE_SIZE, CACHE_SIZE)
    swa_write(kv, positions, cu, slots, out, CACHE_SIZE, CACHE_SIZE)
    torch.cuda.synchronize()
    return {
        "out": out,
        "ref": ref,
        "kv": kv,
        "cu": cu,
        "positions": positions,
        "slots": slots,
    }


def test_kernel_matches_reference(written):
    out, ref = written["out"], written["ref"]
    assert torch.equal(
        out, ref
    ), f"kernel != reference; max|diff|={(out.float() - ref.float()).abs().max()}"


def test_last_token_lands_on_its_ring_row(written):
    """Spot-check a known mapping: seq 2's last token, several laps in."""
    b = 2
    last_pos = START_POS[b] + TOK_COUNTS[b] - 1
    row = SLOTS[b] * CACHE_SIZE + last_pos % CACHE_SIZE
    assert torch.equal(
        written["out"][row], written["kv"][int(written["cu"][b + 1]) - 1]
    )


def test_unowned_slots_are_untouched(written):
    """No seq may write a slot it does not own — the isolation the ring is for."""
    for s in set(range(NUM_SLOTS)) - set(SLOTS):
        block = written["out"][s * CACHE_SIZE : (s + 1) * CACHE_SIZE]
        assert not block.any(), f"unowned slot {s} was written"


def test_a_wrapping_seq_leaves_exactly_one_ring_live(written):
    """More than `cache_size` tokens must overwrite the seq's OWN older rows."""
    s = SLOTS[2]
    block = written["out"][s * CACHE_SIZE : (s + 1) * CACHE_SIZE]
    live = int((block.abs().sum(-1) > 0).sum())
    assert live == CACHE_SIZE, f"seq2 wrote {TOK_COUNTS[2]} tokens, {live} rows live"


def test_over_wide_write_is_rejected(written):
    """Must fail loudly, not race. The one contract the paged predecessor did
    not need: block addressing was injective on position, a ring is not."""
    with pytest.raises(AssertionError, match="exceeds the ring"):
        swa_write(
            written["kv"],
            written["positions"],
            written["cu"],
            written["slots"],
            written["out"],
            CACHE_SIZE,
            CACHE_SIZE + 1,
        )
