# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""`cp_gather_indexer_k_quant_cache` packs a batch's committed K by order.

The FP8 indexer prefill path gathers every sequence's committed index rows
into ONE `[total_committed, head_dim]` buffer, laid out back to back in batch
order, and then tells the scorer where each sequence starts via
`cu_committed`. Two things about that are batch-composition-dependent by
construction:

* a sequence's rows land at an offset set by whoever precedes it, and its
  source blocks come from `block_table[batch_idx]` — so a mis-derived
  `batch_idx` reads another request's keys;
* the kernel picks `BLOCK_Y_SIZE` (1/2/4/8/16/32) from `num_tokens` alone,
  and `num_tokens` is the batch's TOTAL committed count. Adding a batch-mate
  can therefore change the block geometry under a sequence that did not
  itself change — and the kernel derives `batch_idx` through an
  *uninitialized* `__shared__` array with no barrier between the write and
  the read.

Neither is reachable from a single-sequence test, which is all this kernel
had. Every predecessor length below is chosen to land `num_tokens` in a
different `BLOCK_Y_SIZE` bucket.
"""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip(
        "exercises an aiter GPU gather kernel; needs a real GPU",
        allow_module_level=True,
    )

from aiter import dtypes
from aiter.ops.cache import cp_gather_indexer_k_quant_cache

DEV = "cuda"
HEAD_DIM = 128
ROW_BYTES = HEAD_DIM + 4  # 128 fp8 cols + one fp32 scale, as ATOM sizes it
ROWS_PER_BLOCK = 64  # csa_rows_per_block = block_size // 4
NUM_BLOCKS = 64
BT_WIDTH = 16

VICTIM_LEN = 40
VICTIM_BLOCKS = [32, 33]
PRED_BLOCKS = list(range(16))


def _cache(seed=0):
    """A cache whose every byte is distinct noise, so any misroute shows."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    raw = torch.randint(
        1,
        255,
        (NUM_BLOCKS, ROWS_PER_BLOCK, ROW_BYTES),
        generator=g,
        dtype=torch.uint8,
        device=DEV,
    )
    return raw.view(dtypes.fp8)


def _block_table(rows):
    bt = torch.zeros((len(rows), BT_WIDTH), dtype=torch.int32, device=DEV)
    for i, blocks in enumerate(rows):
        bt[i, : len(blocks)] = torch.tensor(blocks, dtype=torch.int32, device=DEV)
    return bt


def _gather(cache, block_table, seq_lens, preshuffle=True):
    """Run the kernel; return (k bytes, scale floats) for the whole batch."""
    cu = torch.zeros(len(seq_lens) + 1, dtype=torch.int32, device=DEV)
    cu[1:] = torch.tensor(seq_lens, dtype=torch.int32, device=DEV).cumsum(0)
    total = int(cu[-1])
    k = torch.empty((total, HEAD_DIM), device=DEV, dtype=dtypes.fp8)
    scale = torch.empty((total, 1), device=DEV, dtype=torch.float32)
    cp_gather_indexer_k_quant_cache(
        cache, k, scale.view(dtypes.fp8), block_table, cu, preshuffle
    )
    torch.cuda.synchronize()
    return k.view(torch.uint8).clone(), scale.clone()


# Predecessor lengths chosen so `num_tokens = pred + 40` crosses every
# BLOCK_Y_SIZE bucket the host wrapper dispatches on (<32, <64, <128, <256,
# <512, else). The victim's own rows are identical in all of them.
@pytest.mark.parametrize(
    "pred_len,bucket",
    [
        (0, "victim alone (BLOCK_Y=2)"),
        (1, "BLOCK_Y=2, offset 1 — unaligned to everything"),
        (30, "BLOCK_Y=4"),
        (100, "BLOCK_Y=8"),
        (220, "BLOCK_Y=16"),
        (480, "BLOCK_Y=32"),
    ],
)
def test_victim_rows_are_independent_of_the_predecessor(pred_len, bucket):
    """The victim's gathered rows must not move when a batch-mate joins."""
    cache = _cache()
    alone_k, alone_s = _gather(cache, _block_table([VICTIM_BLOCKS]), [VICTIM_LEN])

    if pred_len == 0:
        pytest.skip("this row IS the baseline")

    bt = _block_table([PRED_BLOCKS, VICTIM_BLOCKS])
    k, s = _gather(cache, bt, [pred_len, VICTIM_LEN])
    got_k = k[pred_len : pred_len + VICTIM_LEN]
    got_s = s[pred_len : pred_len + VICTIM_LEN]

    assert (alone_k != 0).any(), "baseline gathered nothing; test is vacuous"
    assert torch.equal(alone_k, got_k), (
        f"{bucket}: the victim's keys changed when a {pred_len}-token "
        f"predecessor joined the batch "
        f"({int((alone_k != got_k).sum())}/{alone_k.numel()} bytes differ)"
    )
    assert torch.equal(alone_s, got_s), f"{bucket}: the victim's scales changed"


def test_each_sequence_reads_only_its_own_blocks():
    """Routing: seq `j`'s rows must come from `block_table[j]`, nothing else.

    Uses the non-preshuffled layout so the expected source byte is a plain
    walk over the block rather than a re-derivation of the kernel's own tile
    mapping (which would make the assertion circular).

    A block is NOT 64 interleaved 132-byte rows: it is `64 * 128` key bytes
    followed by `64 * 4` scale bytes. Both the writer
    (`indexer_k_quant_and_cache`) and this gather address it that way.
    """
    cache = _cache(seed=3)
    flat = cache.view(torch.uint8).reshape(NUM_BLOCKS, ROWS_PER_BLOCK * ROW_BYTES)
    keys = flat[:, : ROWS_PER_BLOCK * HEAD_DIM].reshape(
        NUM_BLOCKS, ROWS_PER_BLOCK, HEAD_DIM
    )
    scales = (
        flat[:, ROWS_PER_BLOCK * HEAD_DIM :]
        .contiguous()
        .view(torch.float32)
        .reshape(NUM_BLOCKS, ROWS_PER_BLOCK)
    )

    lens = [70, VICTIM_LEN]
    bt = _block_table([PRED_BLOCKS, VICTIM_BLOCKS])
    k, s = _gather(cache, bt, lens, preshuffle=False)

    rows = [PRED_BLOCKS, VICTIM_BLOCKS]
    base = 0
    for seq, n in enumerate(lens):
        for i in range(n):
            blk = rows[seq][i // ROWS_PER_BLOCK]
            row = i % ROWS_PER_BLOCK
            assert torch.equal(k[base + i], keys[blk, row]), (
                f"seq {seq} row {i} read the wrong keys: expected block "
                f"{blk} row {row}"
            )
            assert torch.equal(s[base + i, 0], scales[blk, row]), (
                f"seq {seq} row {i} read the wrong scale: expected block "
                f"{blk} row {row}"
            )
        base += n


def test_a_zero_length_sequence_does_not_displace_its_neighbours():
    """A fresh prompt contributes no committed K — an empty span in the pack.

    That is the batch shape the checkpoint ladder creates: a resumed chunk
    with committed rows beside a first-chunk prompt with none. The empty
    sequence must claim no rows and shift nobody.
    """
    cache = _cache(seed=5)
    bt3 = _block_table([PRED_BLOCKS, VICTIM_BLOCKS, VICTIM_BLOCKS])

    k_ref, _ = _gather(cache, _block_table([PRED_BLOCKS, VICTIM_BLOCKS]), [70, 40])
    # Same two contributors, with an empty sequence wedged between them.
    k_gap, _ = _gather(cache, bt3, [70, 0, 40])

    assert k_gap.shape == k_ref.shape
    assert torch.equal(
        k_ref, k_gap
    ), "an empty sequence between two contributors changed the packed rows"
