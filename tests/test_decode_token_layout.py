# SPDX-License-Identifier: MIT
"""A decode step's per-token index arrays, against the copies they replaced.

`AiterAttentionMetadataBuilder` and `AiterMLAMetadataBuilder` held
byte-identical transcriptions of both, one of them a per-token Python loop. The
references below are those transcriptions, so what is checked is that the
shipped code reproduces the two it replaced.

The shape is a rectangle: every sequence forwards the same `max_seqlen_q` rows,
so sequence `i` covers `[context_lens[i] - max_seqlen_q, context_lens[i])`.
`context_lens` may already have been reduced by the caller when drafts were
rejected, which is why the fixtures vary it independently of the block tables
rather than deriving one from the other.

`token_layout/slots.py` is shared with prefill and gets its ragged cases in
`test_prefill_token_layout.py`; what it is asked here is the rectangle.

No GPU, no AITER: both modules are numpy-only and loaded by path.
"""

from __future__ import annotations

import array
import importlib.util
import pathlib

import numpy as np
import pytest

_PATH = (
    pathlib.Path(__file__).resolve().parent.parent
    / "atom/model_ops/attentions/token_layout/decode.py"
)
_spec = importlib.util.spec_from_file_location("_decode_under_test", _PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
decode_positions = _mod.decode_positions

_SLOTS = _PATH.with_name("slots.py")
_sspec = importlib.util.spec_from_file_location("_slots_under_test", _SLOTS)
_smod = importlib.util.module_from_spec(_sspec)
_sspec.loader.exec_module(_smod)
slot_mapping = _smod.slot_mapping

B = 16  # block size


# ── the transcriptions this replaced, verbatim ─────────────────────────────


def ref_positions(context_lens, max_seqlen_q):
    scheduled_bs = len(context_lens)
    return np.tile(np.arange(max_seqlen_q, dtype=np.int32), scheduled_bs) + np.repeat(
        context_lens - max_seqlen_q, max_seqlen_q
    )


def ref_slot_mapping(block_tables, context_lens, max_seqlen_q, block_size):
    """The per-token loop both builders held before `slots.py` served them."""
    return [
        block_table[pos // block_size] * block_size + (pos % block_size)
        for block_table, seq_len in zip(block_tables, context_lens)
        for pos in range(seq_len - max_seqlen_q, seq_len)
    ]


def tables_for(context_lens, block_size=B, first_block=1000):
    """One `array("i")` per sequence, ids far apart so a row mix-up is loud."""
    return [
        array.array(
            "i",
            range(
                first_block + i * 500,
                first_block + i * 500 + -(-int(s) // block_size),
            ),
        )
        for i, s in enumerate(context_lens)
    ]


def packed(tables, cols=None, fill=-999):
    """What `prepare_block_tables` leaves behind, with poison padding: a wrong
    row base or block index then lands on a negative slot rather than a
    plausible one."""
    cols = cols or max(len(t) for t in tables) + 3
    dst = np.full((len(tables), cols), fill, dtype=np.int32)
    for i, t in enumerate(tables):
        dst[i, : len(t)] = t
    return dst


# (context_lens, max_seqlen_q)
CASES = [
    ([64], 1),
    ([64, 128, 65], 1),
    ([64], 4),  # a speculative step: 1 committed + 3 drafts
    ([100, 48, 17], 4),
    ([B, B + 1, B - 1], 2),  # spans that do and do not cross a block edge
    ([B * 4], 8),
    ([1024, 4096, 33], 6),
    ([7], 7),  # the whole sequence is this step
]


@pytest.mark.parametrize(
    "ctx,max_q", CASES, ids=lambda v: str(v) if isinstance(v, int) else f"bs{len(v)}"
)
def test_positions_match_the_builders_transcription(ctx, max_q):
    ctx_np = np.asarray(ctx, dtype=np.int32)
    assert np.array_equal(decode_positions(ctx_np, max_q), ref_positions(ctx_np, max_q))


@pytest.mark.parametrize(
    "ctx,max_q", CASES, ids=lambda v: str(v) if isinstance(v, int) else f"bs{len(v)}"
)
def test_slot_mapping_matches_the_builders_transcription(ctx, max_q):
    """The shared gather, on a decode rectangle, against the loop it replaced."""
    ctx_np = np.asarray(ctx, dtype=np.int32)
    tables = tables_for(ctx_np)
    positions = decode_positions(ctx_np, max_q)
    got = slot_mapping(positions, np.full(len(ctx), max_q, np.int32), packed(tables), B)
    assert got.tolist() == ref_slot_mapping(tables, ctx_np, max_q, B)
    assert got.shape[0] == len(ctx) * max_q


def test_a_token_lands_on_the_slot_its_position_names():
    """The two arrays have to agree with each other, not just each with its own
    reference: the slot of token `t` is the slot of `positions[t]`."""
    ctx = np.asarray([100, 48], dtype=np.int32)
    tables = tables_for(ctx)
    positions = decode_positions(ctx, 4)
    slots = slot_mapping(positions, np.full(2, 4, np.int32), packed(tables), B).tolist()
    for t, (pos, slot) in enumerate(zip(positions.tolist(), slots)):
        row = tables[t // 4]
        assert slot == row[pos // B] * B + pos % B


def test_each_sequence_reads_its_own_row():
    """The corpus would pass with every row identical, so pin the one property
    that makes the per-sequence table load-bearing."""
    ctx = np.asarray([B * 2, B * 2], dtype=np.int32)
    tables = [array.array("i", [5, 6]), array.array("i", [900, 901])]
    positions = decode_positions(ctx, B)
    slots = slot_mapping(positions, np.full(2, B, np.int32), packed(tables), B)
    assert slots[0] == 6 * B and slots[B] == 901 * B


def test_positions_end_on_the_last_token_of_each_sequence():
    """A decode step's last row is `context_lens[i] - 1`; off by one here puts
    every draft one slot early and the answer stays plausible."""
    ctx = np.asarray([100, 48, 7], dtype=np.int32)
    got = decode_positions(ctx, 3).reshape(3, 3)
    assert got[:, -1].tolist() == [99, 47, 6]
    assert got[:, 0].tolist() == [97, 45, 4]


def test_positions_written_into_a_caller_buffer():
    ctx = np.asarray([64, 32], dtype=np.int32)
    dst = np.full(2 * 4 + 5, -7, dtype=np.int32)
    got = decode_positions(ctx, 4, out=dst[:-5])
    assert np.shares_memory(got, dst)
    assert np.array_equal(dst[:-5], ref_positions(ctx, 4))
    assert (dst[-5:] == -7).all()


def test_a_trimmed_context_moves_the_whole_span():
    """When drafts are rejected the caller shortens `context_lens` and trims the
    block-table row; the span must follow the context, not the row length."""
    full = np.asarray([100], dtype=np.int32)
    tables = tables_for(full)
    trimmed = np.asarray([97], dtype=np.int32)
    assert decode_positions(trimmed, 2).tolist() == [95, 96]
    # The row is NOT trimmed here on purpose: the packed table the gather reads
    # keeps every entry, and a token's block index stays under the trimmed
    # length, so both must give the loop's answer over the untrimmed row.
    got = slot_mapping(
        decode_positions(trimmed, 2), np.full(1, 2, np.int32), packed(tables), B
    )
    assert got.tolist() == ref_slot_mapping(tables, trimmed, 2, B)
