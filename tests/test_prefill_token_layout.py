# SPDX-License-Identifier: MIT
"""CPU unit tests for a prefill step's per-token index arrays.

`token_layout/prefill.py` replaced a per-token Python loop, and this checks it
against that loop, written out below as the specification. The two formulations
are independent down to the input representation: the reference walks each
sequence's own ragged BLOCK list and extends a range per block, while the code
under test reads the left-aligned 2-D buffer `prepare_block_tables` packs and
derives every token's slot from its position. Same answers or the vectorization
is wrong.

`token_layout/slots.py` is exercised here too, on prefill-shaped inputs -- the
ragged axes and cached prefixes below are what make its edge cases, and it is
one function serving both sides rather than a prefill one. Its decode-shaped
cases are in `test_decode_token_layout.py`.

No GPU, no AITER: both modules are numpy-only and loaded by path, so this never
triggers `atom.model_ops.__init__`.

Correctness only. The speedup table that motivated the vectorization is a
reporting tool, not a test, and lives in
`/app/logs_claude/tool/prefill_token_layout_perf.py`; what stays here is the one
regression guard, and it alternates its arms -- see its own comment for why
that is not optional.
"""

import array
import importlib.util
import pathlib

import numpy as np
import pytest

_DIR = (
    pathlib.Path(__file__).resolve().parent.parent
    / "atom/model_ops/attentions/token_layout"
)


def _load(name):
    spec = importlib.util.spec_from_file_location(
        f"_{name}_under_test", _DIR / f"{name}.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


prefill_positions = _load("prefill").prefill_positions
slot_mapping = _load("slots").slot_mapping

BLOCK_SIZE = 16
B = BLOCK_SIZE


# ── reference: the per-token loop the vectorization replaced ────────────────


def ref_positions(context_lens, cached_lens):
    positions = []
    for seqlen, cached in zip(context_lens, cached_lens):
        positions.extend(range(cached, seqlen))
    return np.asarray(positions, dtype=np.int64)


def ref_slot_mapping(context_lens, cached_lens, block_tables, block_size):
    slots = []
    for seqlen, cached, table in zip(context_lens, cached_lens, block_tables):
        first_blk = cached // block_size
        last_blk = (seqlen - 1) // block_size
        for blk in range(first_blk, last_blk + 1):
            start = table[blk] * block_size
            lo = cached % block_size if blk == first_blk else 0
            hi = ((seqlen - 1) % block_size) + 1 if blk == last_blk else block_size
            slots.extend(range(start + lo, start + hi))
    return np.asarray(slots, dtype=np.int64)


# ── inputs, as the step already holds them ─────────────────────────────────


def step_inputs(context_lens, cached_lens):
    """`(context_lens, seqlens_q, cached_lens, cu_seqlens_q, token_offsets)`.

    int32 like `ScheduledBatch.context_lens` / `num_scheduled_tokens` and the
    `cu_seqlens_q` mirror, so the dtypes under test are the shipped ones.
    """
    ctx = np.asarray(context_lens, dtype=np.int32)
    cached = np.asarray(cached_lens, dtype=np.int32)
    seqlens_q = ctx - cached
    cu_q = np.zeros(len(ctx) + 1, dtype=np.int32)
    np.cumsum(seqlens_q, out=cu_q[1:])
    return ctx, seqlens_q, cached, cu_q, np.arange(int(cu_q[-1]), dtype=np.int64)


def tables_for(context_lens, extra_rows=0, first_block=1000, block_size=BLOCK_SIZE):
    """One `array("i")` per sequence, wide enough for its whole context.

    Distinct, far-apart block ids per sequence: a slot built from the wrong
    sequence's table then lands nowhere near the right answer.
    """
    out = [
        array.array(
            "i",
            range(
                first_block + i * 500,
                first_block + i * 500 + (s + block_size - 1) // block_size,
            ),
        )
        for i, s in enumerate(context_lens)
    ]
    # A real batch's block_tables covers every sequence, not just the prefill
    # ones, so callers slice it -- keep rows past the end in the corpus.
    out += [array.array("i", [90000 + k] * 8) for k in range(extra_rows)]
    return out


def packed(tables, rows, cols=None, fill=-999):
    """`tables` left-aligned into a fixed-width 2-D int32 buffer.

    What `pack_rows` leaves behind, except that the padding is poison rather
    than zero: every column past a row's length must be unreachable, and a
    zero there would let an off-by-one read as block 0 and look plausible.
    """
    cols = cols or max(len(t) for t in tables[:rows]) + 3
    dst = np.full((rows, cols), fill, dtype=np.int32)
    for i, t in enumerate(tables[:rows]):
        dst[i, : len(t)] = t
    return dst


CASES = [
    # fresh prompts, block-aligned start
    ([B], [0], 0),
    ([1], [0], 0),
    ([B * 4], [0], 0),
    ([37, 64, 5], [0, 0, 0], 0),
    # chunked prefill: resumes mid-block, on a block edge, one block in
    ([100], [7], 0),
    ([100], [B], 0),
    ([100], [B + 1], 0),
    # a single-token chunk, aligned and not
    ([B + 1], [B], 0),
    ([B], [B - 1], 0),
    ([1000], [999], 0),
    # ends mid-block, exactly on an edge, one past one
    ([B * 3], [B], 0),
    ([B * 3 - 1], [B], 0),
    ([B * 3 + 1], [B], 0),
    # every shape above mixed into one batch
    ([16, 100, 17, 48], [0, 7, 16, 47], 0),
    # block_tables carrying rows past the prefill batch
    ([48, 33], [16, 0], 3),
    ([100], [37], 1),
]

_rng = np.random.default_rng(20260903)
for _ in range(12):
    _n = int(_rng.integers(1, 9))
    _ctx = _rng.integers(1, 2048, size=_n)
    _cached = (_ctx * _rng.random(_n) * 0.9).astype(np.int64)
    CASES.append((_ctx.tolist(), _cached.tolist(), 0))


def ids(cases):
    return [f"bs{len(c[0])}_T{sum(c[0]) - sum(c[1])}" for c in cases]


# ── tests ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("ctx,cached,extra", CASES, ids=ids(CASES))
def test_positions_match_the_per_token_loop(ctx, cached, extra):
    del extra
    _, seqlens_q, cached_lens, cu_q, offsets = step_inputs(ctx, cached)
    got = prefill_positions(offsets, cached_lens, cu_q, seqlens_q)
    assert np.array_equal(got, ref_positions(ctx, cached))


@pytest.mark.parametrize("ctx,cached,extra", CASES, ids=ids(CASES))
def test_slot_mapping_matches_the_per_block_loop(ctx, cached, extra):
    _, seqlens_q, cached_lens, cu_q, offsets = step_inputs(ctx, cached)
    positions = prefill_positions(offsets, cached_lens, cu_q, seqlens_q)
    tables = tables_for(ctx, extra_rows=extra)
    got = slot_mapping(positions, seqlens_q, packed(tables, len(ctx)), B)
    assert got.shape[0] == offsets.shape[0]
    assert np.array_equal(got, ref_slot_mapping(ctx, cached, tables, B))


def test_positions_written_into_a_caller_buffer():
    """The shipped caller passes the pinned mirror as `out`, so no temporary
    stands between this arithmetic and what the GPU reads."""
    ctx, cached = [100, 48], [64, 0]
    _, seqlens_q, cached_lens, cu_q, offsets = step_inputs(ctx, cached)
    dst = np.full(offsets.shape[0] + 4, -7, dtype=np.int64)
    got = prefill_positions(offsets, cached_lens, cu_q, seqlens_q, out=dst[:-4])
    assert np.shares_memory(got, dst)
    assert np.array_equal(dst[:-4], ref_positions(ctx, cached))
    assert (dst[-4:] == -7).all()  # nothing past the requested width


def test_slot_mapping_written_into_a_caller_buffer():
    """Same for the slot mirror, which is the buffer the KV write addresses."""
    ctx, cached = [100, 48], [64, 0]
    _, seqlens_q, cached_lens, cu_q, offsets = step_inputs(ctx, cached)
    positions = prefill_positions(offsets, cached_lens, cu_q, seqlens_q)
    tables = tables_for(ctx)
    dst = np.full(offsets.shape[0] + 4, -7, dtype=np.int64)
    got = slot_mapping(positions, seqlens_q, packed(tables, len(ctx)), B, out=dst[:-4])
    assert np.shares_memory(got, dst)
    assert np.array_equal(dst[:-4], ref_slot_mapping(ctx, cached, tables, B))
    assert (dst[-4:] == -7).all()


def test_slot_mapping_reads_each_sequences_own_row():
    """The corpus would pass if every row were identical, so pin the one
    property that makes the per-row base offset load-bearing."""
    ctx, cached = [B * 2, B * 2], [0, 0]
    _, seqlens_q, cached_lens, cu_q, offsets = step_inputs(ctx, cached)
    positions = prefill_positions(offsets, cached_lens, cu_q, seqlens_q)
    tables = [array.array("i", [5, 6]), array.array("i", [900, 901])]
    got = slot_mapping(positions, seqlens_q, packed(tables, 2), B)
    assert got[0] == 5 * B and got[B] == 6 * B
    assert got[2 * B] == 900 * B and got[3 * B] == 901 * B


@pytest.mark.parametrize("cols", [4, 5, 9, 64])
def test_row_stride_does_not_change_the_answer(cols):
    """The buffer is as wide as the longest context the engine allows, not as
    wide as this batch needs, so the stride must not reach the arithmetic."""
    ctx, cached = [48, 33, 64], [16, 0, 32]
    _, seqlens_q, cached_lens, cu_q, offsets = step_inputs(ctx, cached)
    positions = prefill_positions(offsets, cached_lens, cu_q, seqlens_q)
    tables = tables_for(ctx)
    got = slot_mapping(positions, seqlens_q, packed(tables, 3, cols=cols), B)
    assert np.array_equal(got, ref_slot_mapping(ctx, cached, tables, B))


def test_padding_columns_are_never_read():
    """`packed` poisons the padding; a wrong row base or an off-by-one block
    index would pick a poison entry up and land at a negative slot."""
    ctx, cached = [40, 40], [0, 24]
    _, seqlens_q, cached_lens, cu_q, offsets = step_inputs(ctx, cached)
    positions = prefill_positions(offsets, cached_lens, cu_q, seqlens_q)
    tables = tables_for(ctx)
    got = slot_mapping(
        positions, seqlens_q, packed(tables, 2, cols=32, fill=-(1 << 20)), B
    )
    assert (got >= 0).all()
    assert np.array_equal(got, ref_slot_mapping(ctx, cached, tables, B))


@pytest.mark.parametrize("block_size", [1, 2, 16, 256, 3, 24, 100])
def test_every_block_size_agrees_with_the_loop(block_size):
    """A power of two splits the position with a shift and a mask, anything
    else with `divmod`. Both paths, and the sizes either side of the seam.

    Contexts run past several blocks at the widest size tested: at 256 a
    100-token sequence never leaves block 0, and a wrong shift would agree.
    """
    ctx, cached = [1600, 999, 301, 1], [512, 0, 300, 0]
    _, seqlens_q, cached_lens, cu_q, offsets = step_inputs(ctx, cached)
    positions = prefill_positions(offsets, cached_lens, cu_q, seqlens_q)
    tables = tables_for(ctx, block_size=block_size)
    got = slot_mapping(positions, seqlens_q, packed(tables, len(ctx)), block_size)
    assert np.array_equal(got, ref_slot_mapping(ctx, cached, tables, block_size))


def test_a_slot_past_int32_does_not_wrap():
    """`block_id * block_size` leaves int32 on a large enough pool. The gather
    is int32 and the destination int64, so the multiply is where it has to
    widen -- do it after and the value is already wrong."""
    block_size = 4096
    high = np.int32(2_000_000)  # * 4096 = 8.2e9, past 2**31
    ctx, cached = [2], [0]
    _, seqlens_q, cached_lens, cu_q, offsets = step_inputs(ctx, cached)
    positions = prefill_positions(offsets, cached_lens, cu_q, seqlens_q)
    table = np.array([[high]], dtype=np.int32)
    got = slot_mapping(positions, seqlens_q, table, block_size)
    assert got[0] == int(high) * block_size
    assert got[1] == int(high) * block_size + 1


def test_a_wider_table_is_refused():
    """`.cast` guards the layout but not the width: an int64 table reads as
    twice as many int32s and answers, so the dtype needs its own guard."""
    # Two blocks per sequence, so a table read at the wrong width would reach
    # a different entry rather than land on the same first column.
    _, seqlens_q, cached_lens, cu_q, offsets = step_inputs([B * 2, B * 2], [0, 0])
    positions = prefill_positions(offsets, cached_lens, cu_q, seqlens_q)
    table = np.array([[3, 4], [7, 8]], dtype=np.int64)
    with pytest.raises(TypeError):
        slot_mapping(positions, seqlens_q, table, B)
    # ...and the same ids at the shipped width do answer.
    got = slot_mapping(positions, seqlens_q, table.astype(np.int32), B)
    assert np.array_equal(got[::B], [3 * B, 4 * B, 7 * B, 8 * B])


def test_a_non_contiguous_table_is_refused():
    """A copy here would be read-only garbage that still produced numbers, so
    the flattening must raise rather than fall back."""
    wide = np.zeros((2, 8), dtype=np.int32)
    _, seqlens_q, cached_lens, cu_q, offsets = step_inputs([B, B], [0, 0])
    positions = prefill_positions(offsets, cached_lens, cu_q, seqlens_q)
    with pytest.raises(TypeError):
        slot_mapping(positions, seqlens_q, wide[:, ::2], B)


def test_positions_resume_at_the_cached_prefix():
    """A chunked prefill's second chunk must not restart at zero."""
    _, seqlens_q, cached_lens, cu_q, offsets = step_inputs([100], [64])
    got = prefill_positions(offsets, cached_lens, cu_q, seqlens_q)
    assert got[0] == 64 and got[-1] == 99


def test_caller_scratch_is_used_and_not_read_past():
    """The shipped caller hands in a resident scratch wider than the batch.

    Two properties: the answer does not change, and nothing past the token axis
    is touched -- the buffer is sized for the largest step, not this one.
    """
    ctx, cached = [100, 48], [64, 0]
    _, seqlens_q, cached_lens, cu_q, offsets = step_inputs(ctx, cached)
    positions = prefill_positions(offsets, cached_lens, cu_q, seqlens_q)
    tables = tables_for(ctx)
    n = offsets.shape[0]
    scratch = np.full(n + 32, -7, dtype=np.int64)
    out = np.empty(n, dtype=np.int64)

    got = slot_mapping(
        positions, seqlens_q, packed(tables, 2), B, out=out, scratch=scratch
    )

    assert np.array_equal(got, ref_slot_mapping(ctx, cached, tables, B))
    assert (scratch[n:] == -7).all(), "wrote past the token axis"


def test_scratch_does_not_corrupt_the_positions_it_reads():
    """`positions` is the caller's pinned mirror and is read again after this
    call, so the split must not write back through it."""
    ctx, cached = [64], [0]
    _, seqlens_q, cached_lens, cu_q, offsets = step_inputs(ctx, cached)
    positions = prefill_positions(offsets, cached_lens, cu_q, seqlens_q)
    before = positions.copy()
    slot_mapping(
        positions,
        seqlens_q,
        packed(tables_for(ctx), 1),
        B,
        out=np.empty(positions.size, np.int64),
        scratch=np.empty(positions.size, np.int64),
    )
    assert np.array_equal(positions, before)


# ── the one perf property worth a gate ─────────────────────────────────────


def test_vectorized_beats_the_loop():
    """The vectorization's whole reason to exist, at a shape a real step runs.

    Arms alternate round by round. Timing one to completion and then the other
    lets each run with numpy's free list holding exactly its own allocation
    pattern, which a step -- one call between a great deal of unrelated
    allocation -- never gets: on this function the two harnesses disagree by
    2.6x. The floor is far below the measured margin so the gate answers "did
    the vectorization survive", not "how fast is this runner today".
    """
    import timeit

    bs, qlen = 16, 1024
    ctx, cached_list = [qlen] * bs, [0] * bs
    _, seqlens_q, cached_lens, cu_q, offsets = step_inputs(ctx, cached_list)
    tables = tables_for(ctx)
    table_2d = packed(tables, bs)
    out = np.empty(offsets.size, np.int64)
    scratch = np.empty(offsets.size, np.int64)

    def new():
        pos = prefill_positions(offsets, cached_lens, cu_q, seqlens_q)
        return slot_mapping(pos, seqlens_q, table_2d, B, out=out, scratch=scratch)

    def legacy():
        # Both halves, or the arms are not doing the same work: `new` derives
        # positions before it can place a slot.
        ref_positions(ctx, cached_list)
        return ref_slot_mapping(ctx, cached_list, tables, B)

    assert np.array_equal(new(), legacy()), "timing two different answers"
    fast, slow = [], []
    for _ in range(5):
        fast.append(min(timeit.repeat(new, repeat=3, number=2)))
        slow.append(min(timeit.repeat(legacy, repeat=3, number=2)))
    ratio = min(slow) / min(fast)
    assert ratio >= 2.0, f"vectorized only {ratio:.2f}x the per-token loop"


@pytest.mark.parametrize("ctx,cached,extra", CASES, ids=ids(CASES))
def test_the_whole_corpus_through_the_shipped_buffers(ctx, cached, extra):
    """Everything above runs the allocate-my-own path; production runs neither
    -- it hands in both buffers, and that is the path that has to be right."""
    _, seqlens_q, cached_lens, cu_q, offsets = step_inputs(ctx, cached)
    positions = prefill_positions(
        offsets, cached_lens, cu_q, seqlens_q, out=np.empty(offsets.size, np.int64)
    )
    tables = tables_for(ctx, extra_rows=extra)
    got = slot_mapping(
        positions,
        seqlens_q,
        packed(tables, len(ctx)),
        B,
        out=np.empty(offsets.size, np.int64),
        scratch=np.empty(offsets.size + 8, np.int64),
    )
    assert np.array_equal(got, ref_slot_mapping(ctx, cached, tables, B))
