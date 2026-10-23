# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The per-step words the mono kernels read, written in one launch before the
step's layers (as a torch chain, a dozen small kernels a step).

- The step counter, once a step: the layers' mailbox tags
  (``mailbox.launch_tag``) are unique per (step, layer) only if it advances.
- Each row's cache length ``L`` (its position) as the int32 the QSA tables'
  ``pos`` entry points at.
- The state addresses of each row's slots: in every DeltaNet layer's table
  (``[rows, DELTA_STATE]``: conv in, conv out, recurrent in, out) and in the
  PLE table's ``conv_in`` / ``conv_out``.
- Each row's block-table row and caches, in every QSA layer's table
  (``pages``, ``key`` .. ``index_cmp``).

A padded row (a graph wider than the batch: its slot < 0) reads and writes
the runner's null state instead: null slots, and a null page of each QSA cache
through a null block table at position ``PAD_POS``. Its block-table row and
position are stale, and the key it writes would land in another request's page.

Every other table entry (weights, RoPE, norms, the rows' own buffers) is fixed
when the runner prepares the width and is not rewritten. Every input is a
persistent buffer, so a captured graph replays this launch as it is.

**MTP verify steps** (seq builds, ``write_verify_meta``): the ``rows = k + 1``
rows are one request's tokens at positions ``L .. L + k``, its spec slots
``s[0 .. k]``, ``a`` the tokens its last step accepted. ATOM keeps per request

- the recurrent state of token ``t`` in slot ``s[t]``: row 0 reads
  ``s[a - 1]``, row ``r`` writes ``s[r]``;
- one conv window ``[3 + k, QKV]`` in slot ``s[0]``, read at rows
  ``a - 1 .. a + 1`` and rewritten whole as ``[old[a], old[a + 1], x0 .. xk]``.
  Row ``r``'s snapshot (the 3 rows after it) is that window's rows
  ``r .. r + 2``, so row 0 reads at row ``a - 1`` and row ``r`` writes at row
  ``r`` of the same window;
- one PLE conv window ``[QKV, 9 + k]`` in slot ``s[0]``, read at columns
  ``a - 1 .. a + 7`` and rewritten as the 8 columns after ``a - 1`` then
  ``x0 .. xk``. The kernels' 9 columns pack two channels a word, which an odd
  ``a - 1`` breaks, so the step gathers the window it reads into ``[QKV, 9]``
  (fused here) and the rows write their 9-column snapshots ``[rows, QKV, 9]``,
  which ``commit_ple`` scatters back after the layers: column ``j`` of the new
  window is snapshot ``r``'s column ``j - r`` for any ``r`` that has it.

The QSA table's one read row is the request's; the PLE table's ``conv_in`` /
``conv_out`` entries name the gathered window and the snapshots, fixed when
the width is prepared.

**MTP draft steps** (``write_draft_meta``, the draft runner's one QSA layer):
the draft's own counter, and its rows' cache rows and block-table row from
ATOM's QSA metadata for the draft forward (``logical_positions``, which trail
the draft's RoPE positions by one)."""

import triton
import triton.language as tl

from .kernels.stages import ple as ple_stage
from .kernels.stages import qsa as qsa_stage
from .kernels.stages.common import DELTA_STATE

PLE_WORDS = len(ple_stage.STATE)
PLE_CONV_IN = ple_stage.STATE.index("conv_in")
QSA_WORDS = len(qsa_stage.STATE)
QSA_PAGES = qsa_stage.STATE.index("pages")
QSA_KEY = qsa_stage.STATE.index("key")
QSA_CACHES = ("key", "value", "index_key", "index_cmp")
DELTA_WORDS = len(DELTA_STATE)
assert ple_stage.STATE[PLE_CONV_IN + 1] == "conv_out"
assert qsa_stage.STATE[QSA_KEY : QSA_KEY + len(QSA_CACHES)] == QSA_CACHES
# a padded row's cache length: inside the null page, short of a group end
PAD_POS = 1
assert DELTA_STATE == ("conv_in", "conv_out", "rec_in", "rec_out")
# the PLE conv window the kernels read and write per channel
PLE_COLS = 9
# elements of the PLE window a verify step's program gathers or commits
PLE_BLOCK = 1024


@triton.jit
def _slot_addr(slot, base, stride, null):
    return tl.where(slot >= 0, base + slot * stride, null)


@triton.jit
def _step_meta_kernel(
    counter,
    positions,
    pos,
    slots_in,
    slots_out,
    gdn_conv,
    gdn_rec,
    delta_tab,
    conv_slot,
    rec_slot,
    null_conv,
    null_rec,
    ple_tab,
    ple_base,
    ple_slot,
    null_ple,
    qsa_tab,
    qsa_caches,
    pages_base,
    table_bytes,
    null_pages,
    null_caches,
    rows,
    n_delta,
    n_qsa,
    PLE_WORDS: tl.constexpr,
    PLE_CONV_IN: tl.constexpr,
    QSA_WORDS: tl.constexpr,
    QSA_PAGES: tl.constexpr,
    QSA_KEY: tl.constexpr,
    CACHES: tl.constexpr,
    PAD_POS: tl.constexpr,
    DELTA_WORDS: tl.constexpr,
    ROWS: tl.constexpr,
    DELTA: tl.constexpr,
    QSA: tl.constexpr,
):
    tl.store(counter, tl.load(counter) + 1)
    r = tl.arange(0, ROWS)
    live = r < rows
    s_in = tl.load(slots_in + r, mask=live, other=-1).to(tl.int64)
    s_out = tl.load(slots_out + r, mask=live, other=-1).to(tl.int64)
    pad = s_out < 0
    p = tl.load(positions + r, mask=live, other=0)
    tl.store(pos + r, tl.where(pad, PAD_POS, p).to(tl.int32), mask=live)

    ple = ple_tab + r * PLE_WORDS + PLE_CONV_IN
    tl.store(ple, _slot_addr(s_in, ple_base, ple_slot, null_ple), mask=live)
    tl.store(ple + 1, _slot_addr(s_out, ple_base, ple_slot, null_ple), mask=live)

    layer = tl.arange(0, DELTA)[:, None]
    m = (layer < n_delta) & live[None, :]
    conv = tl.load(gdn_conv + layer, mask=layer < n_delta, other=0)
    rec = tl.load(gdn_rec + layer, mask=layer < n_delta, other=0)
    si, so = s_in[None, :], s_out[None, :]
    entry = delta_tab + (layer * rows + r[None, :]) * DELTA_WORDS
    tl.store(entry, _slot_addr(si, conv, conv_slot, null_conv), mask=m)
    tl.store(entry + 1, _slot_addr(so, conv, conv_slot, null_conv), mask=m)
    tl.store(entry + 2, _slot_addr(si, rec, rec_slot, null_rec), mask=m)
    tl.store(entry + 3, _slot_addr(so, rec, rec_slot, null_rec), mask=m)

    q = tl.arange(0, QSA)[:, None]
    m = (q < n_qsa) & live[None, :]
    pages = tl.where(pad, null_pages, pages_base + r.to(tl.int64) * table_bytes)
    entry = qsa_tab + (q * rows + r[None, :]) * QSA_WORDS
    tl.store(entry + QSA_PAGES, tl.broadcast_to(pages[None, :], (QSA, ROWS)), mask=m)
    for c in tl.static_range(CACHES):
        cache = tl.load(qsa_caches + q * CACHES + c, mask=q < n_qsa, other=0)
        null = tl.load(null_caches + c)
        tl.store(entry + QSA_KEY + c, tl.where(pad[None, :], null, cache), mask=m)


def write_step_meta(
    counter, positions, slots_in, slots_out, block_tables, tables, nulls
) -> None:
    """Advance ``counter`` and write the step's words of ``tables`` (the
    runner's ``Tables``: ``pos`` [rows] int32, ``delta`` [DeltaNet layers,
    rows, 4], ``ple`` [rows, ple.STATE], ``qsa`` [QSA layers, rows,
    qsa.STATE] int64; ``gdn_conv`` / ``gdn_rec`` int64 [DeltaNet layers], each
    layer's slot 0 address, ``*_slot`` the slot strides in bytes, ``ple_base``
    the PLE pool's, ``qsa_caches`` int64 [QSA layers, QSA_CACHES] each layer's
    cache addresses) for rows at ``positions`` (int64) with state slots
    ``slots_in`` / ``slots_out`` and block table ``block_tables`` (int32
    ``[requests, pages]``, row r the row's). ``nulls`` holds the null state's
    addresses: slots ``conv``, ``rec``, ``ple``, the block table ``pages`` and
    ``caches``, an int64 tensor of a null page's address per ``QSA_CACHES``."""
    t = tables
    rows = t.pos.shape[0]
    n_delta, n_qsa = t.delta.shape[0], t.qsa.shape[0]
    assert positions.shape[0] >= rows and slots_in.shape[0] >= rows
    assert block_tables.shape[0] >= rows and block_tables.stride(1) == 1
    _step_meta_kernel[(1,)](
        counter,
        positions,
        t.pos,
        slots_in,
        slots_out,
        t.gdn_conv,
        t.gdn_rec,
        t.delta,
        t.conv_slot,
        t.rec_slot,
        nulls["conv"],
        nulls["rec"],
        t.ple,
        t.ple_base,
        t.ple_slot,
        nulls["ple"],
        t.qsa,
        t.qsa_caches,
        block_tables.data_ptr(),
        block_tables.stride(0) * block_tables.element_size(),
        nulls["pages"],
        nulls["caches"],
        rows,
        n_delta,
        n_qsa,
        PLE_WORDS=PLE_WORDS,
        PLE_CONV_IN=PLE_CONV_IN,
        QSA_WORDS=QSA_WORDS,
        QSA_PAGES=QSA_PAGES,
        QSA_KEY=QSA_KEY,
        CACHES=len(QSA_CACHES),
        PAD_POS=PAD_POS,
        DELTA_WORDS=DELTA_WORDS,
        ROWS=max(2, triton.next_power_of_2(rows)),
        DELTA=max(2, triton.next_power_of_2(n_delta)),
        QSA=max(2, triton.next_power_of_2(n_qsa)),
    )


@triton.jit
def _verify_meta_kernel(
    counter,
    positions,
    pos,
    slots,
    accepted,
    gdn_conv,
    gdn_rec,
    delta_tab,
    conv_slot,
    rec_slot,
    conv_row,
    qsa_tab,
    pages,
    ple_state,
    ple_window,
    ple_slot,
    channels,
    rows,
    n_delta,
    n_qsa,
    QSA_WORDS: tl.constexpr,
    QSA_PAGES: tl.constexpr,
    DELTA_WORDS: tl.constexpr,
    COLS: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
    ROWS: tl.constexpr,
    DELTA: tl.constexpr,
    QSA: tl.constexpr,
):
    s0 = tl.load(slots).to(tl.int64)
    a = tl.load(accepted).to(tl.int64)

    # every program: its block of the PLE window, columns a - 1 .. a + 7
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < channels * COLS
    src = ple_state + s0 * ple_slot + (i // COLS) * WIDTH + a - 1 + i % COLS
    tl.store(ple_window + i, tl.load(src, mask=m), mask=m)

    if tl.program_id(0) == 0:
        tl.store(counter, tl.load(counter) + 1)
        r = tl.arange(0, ROWS)
        live = r < rows
        p = tl.load(positions + r, mask=live, other=0)
        tl.store(pos + r, p.to(tl.int32), mask=live)

        s = tl.load(slots + r, mask=live, other=0).to(tl.int64)
        s_in = tl.load(slots + a - 1).to(tl.int64)
        layer = tl.arange(0, DELTA)[:, None]
        lm = (layer < n_delta) & live[None, :]
        conv = tl.load(gdn_conv + layer, mask=layer < n_delta, other=0)
        rec = tl.load(gdn_rec + layer, mask=layer < n_delta, other=0)
        window = conv + s0 * conv_slot
        rr = r[None, :].to(tl.int64)
        entry = delta_tab + (layer * rows + r[None, :]) * DELTA_WORDS
        tl.store(entry, window + (a - 1) * conv_row + 0 * rr, mask=lm)
        tl.store(entry + 1, window + rr * conv_row, mask=lm)
        tl.store(entry + 2, rec + s_in * rec_slot + 0 * rr, mask=lm)
        tl.store(entry + 3, rec + s[None, :] * rec_slot, mask=lm)

        # the QSA tables' one read row: the request's block-table row
        q = tl.arange(0, QSA)
        tl.store(qsa_tab + q * rows * QSA_WORDS + QSA_PAGES, pages, mask=q < n_qsa)


@triton.jit
def _ple_commit_kernel(
    ple_state,
    snaps,
    slots,
    ple_slot,
    channels,
    COLS: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    s0 = tl.load(slots).to(tl.int64)
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < channels * WIDTH
    c, j = i // WIDTH, i % WIDTH
    # column j of the new window: snapshot r's column j - r, r = max(j - 8, 0)
    r = tl.maximum(j - (COLS - 1), 0)
    v = tl.load(snaps + (r * channels + c) * COLS + j - r, mask=m)
    tl.store(ple_state + s0 * ple_slot + i, v, mask=m)


def _ple_geometry(ple_state, window):
    """(channels, window width) of the PLE pool ``[slots, channels, width]``,
    checked against the gathered window ``[channels, PLE_COLS]``."""
    _, channels, width = ple_state.shape
    assert window.shape == (channels, PLE_COLS) and ple_state.is_contiguous()
    return channels, width


def write_verify_meta(
    counter, positions, slots, accepted, block_tables, tables, ple_state, ple_window
) -> None:
    """``write_step_meta`` for a verify step (seq build, the module's MTP
    section): advance ``counter``; write ``tables``' ``pos`` from
    ``positions`` (int64, the rows' tokens), each DeltaNet table row's state
    addresses from the request's spec slots ``slots`` (int32 ``[k + 1]``) and
    ``accepted`` (int32 ``[1]``, ``a``), and the QSA tables' block-table row
    (``block_tables`` row 0); and gather the PLE window ``ple_window`` ``[QKV,
    9]`` from ``ple_state`` ``[slots, QKV, 9 + k]``."""
    t = tables
    rows = t.pos.shape[0]
    n_delta, n_qsa = t.delta.shape[0], t.qsa.shape[0]
    channels, width = _ple_geometry(ple_state, ple_window)
    assert positions.shape[0] >= rows and slots.shape[0] >= rows
    assert width == PLE_COLS + rows - 1 and block_tables.stride(1) == 1
    grid = (triton.cdiv(channels * PLE_COLS, PLE_BLOCK),)
    _verify_meta_kernel[grid](
        counter,
        positions,
        t.pos,
        slots,
        accepted,
        t.gdn_conv,
        t.gdn_rec,
        t.delta,
        t.conv_slot,
        t.rec_slot,
        t.conv_slot // (rows + 2),
        t.qsa,
        block_tables.data_ptr(),
        ple_state,
        ple_window,
        channels * width,
        channels,
        rows,
        n_delta,
        n_qsa,
        QSA_WORDS=QSA_WORDS,
        QSA_PAGES=QSA_PAGES,
        DELTA_WORDS=DELTA_WORDS,
        COLS=PLE_COLS,
        WIDTH=width,
        BLOCK=PLE_BLOCK,
        ROWS=max(2, triton.next_power_of_2(rows)),
        DELTA=max(2, triton.next_power_of_2(n_delta)),
        QSA=max(2, triton.next_power_of_2(n_qsa)),
    )


def commit_ple(ple_state, snaps, slots) -> None:
    """After a verify step's layers: rewrite the request's PLE window
    (``ple_state`` ``[slots, QKV, 9 + k]``, slot ``slots[0]``) from the rows'
    snapshots ``snaps`` ``[k + 1, QKV, 9]``, as ATOM's own update leaves it."""
    channels, width = _ple_geometry(ple_state, snaps[0])
    assert snaps.shape[0] == width - PLE_COLS + 1 and snaps.is_contiguous()
    _ple_commit_kernel[(triton.cdiv(channels * width, PLE_BLOCK),)](
        ple_state,
        snaps,
        slots,
        channels * width,
        channels,
        COLS=PLE_COLS,
        WIDTH=width,
        BLOCK=PLE_BLOCK,
    )


@triton.jit
def _draft_meta_kernel(
    counter,
    logical,
    pos,
    qsa_tab,
    pages,
    rows,
    QSA_WORDS: tl.constexpr,
    QSA_PAGES: tl.constexpr,
    ROWS: tl.constexpr,
):
    tl.store(counter, tl.load(counter) + 1)
    r = tl.arange(0, ROWS)
    live = r < rows
    p = tl.load(logical + r, mask=live, other=0)
    tl.store(pos + r, p.to(tl.int32), mask=live)
    entry = qsa_tab + r * QSA_WORDS + QSA_PAGES
    tl.store(entry, pages + 0 * r.to(tl.int64), mask=live)


def write_draft_meta(counter, logical, block_tables, pos, qsa_tab) -> None:
    """A draft step's words (one request, the module's MTP draft section):
    advance ``counter``, write ``pos`` [rows] int32 from the rows' cache rows
    ``logical`` (int64) and point every row of the draft layer's QSA table
    ``qsa_tab`` [rows, qsa.STATE] at the request's block-table row
    (``block_tables`` row 0)."""
    rows = pos.shape[0]
    assert logical.shape[0] >= rows and block_tables.stride(1) == 1
    _draft_meta_kernel[(1,)](
        counter,
        logical,
        pos,
        qsa_tab,
        block_tables.data_ptr(),
        rows,
        QSA_WORDS=QSA_WORDS,
        QSA_PAGES=QSA_PAGES,
        ROWS=max(2, triton.next_power_of_2(rows)),
    )
