# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""``step_meta``'s verify and draft words for a step of several requests
(``groups``), one launch each.

A verify step of G groups of ``S = k + 1`` rows: group ``g`` is rows
``g·S .. g·S + k``, request ``g``'s tokens, its spec slots row ``g`` of ATOM's
``spec_state_indices_tensor`` ``[requests, S]``, ``a`` its
``num_accepted_tokens[g]`` and its pages block-table row ``g``. Each group's
words are ``write_verify_meta``'s for that request, at its rows of the
width's tables; each group's PLE window is its own ``[QKV, 9]``.

A padded group (slot ``s[0]`` < 0: a graph wider than the batch) reads and
writes the runner's null state, as a padded decode row does
(``write_step_meta``): its window and recurrent state the null slots
(``a`` = 1), its pages the null block table, its caches the null pages, its
positions ``PAD_POS + r``. Its PLE window reads as zeros, the null slot's
contents, and ``commit_ple_groups`` skips it.

Every table row of a group gets its pages and caches, though the seq builds
read only the group's first (``build_layer``).

A draft step's rows (``write_group_draft_meta``) take the block-table row of
their request: row ``r``'s is ``r // group``, ``group`` = S at step 0 (the
verify's rows) and 1 at a later step (one row per request).
"""

import triton
import triton.language as tl

from .step_meta import (
    DELTA_WORDS,
    PAD_POS,
    PLE_BLOCK,
    PLE_COLS,
    QSA_CACHES,
    QSA_KEY,
    QSA_PAGES,
    QSA_WORDS,
    _ple_geometry,
)


@triton.jit
def _group_verify_meta_kernel(
    counter,
    positions,
    pos,
    slots,
    slot_stride,
    accepted,
    gdn_conv,
    gdn_rec,
    delta_tab,
    conv_slot,
    rec_slot,
    conv_row,
    null_conv,
    null_rec,
    qsa_tab,
    qsa_caches,
    pages_base,
    table_bytes,
    null_pages,
    null_caches,
    ple_state,
    ple_window,
    ple_slot,
    channels,
    rows,
    n_delta,
    n_qsa,
    S: tl.constexpr,
    QSA_WORDS: tl.constexpr,
    QSA_PAGES: tl.constexpr,
    QSA_KEY: tl.constexpr,
    CACHES: tl.constexpr,
    PAD_POS: tl.constexpr,
    DELTA_WORDS: tl.constexpr,
    COLS: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
    ROWS: tl.constexpr,
    DELTA: tl.constexpr,
    QSA: tl.constexpr,
):
    g = tl.program_id(1)
    s0 = tl.load(slots + g * slot_stride).to(tl.int64)
    pad, keep = s0 < 0, s0 >= 0
    a = tl.load(accepted + g, mask=keep, other=1).to(tl.int64)
    a = tl.where(pad, 1, a)

    # every program of the group: its block of the group's PLE window,
    # columns a - 1 .. a + 7 (a padded group's: the null slot's zeros)
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < channels * COLS
    src = ple_state + s0 * ple_slot + (i // COLS) * WIDTH + a - 1 + i % COLS
    v = tl.load(src, mask=m & keep, other=0)
    tl.store(ple_window + g * channels * COLS + i, v, mask=m)

    if tl.program_id(0) == 0:
        if g == 0:
            tl.store(counter, tl.load(counter) + 1)
        r = tl.arange(0, ROWS)
        live = r < S
        row = g * S + r
        p = tl.load(positions + row, mask=live, other=0)
        tl.store(pos + row, tl.where(pad, PAD_POS + r, p).to(tl.int32), mask=live)

        group_slots = slots + g * slot_stride
        s = tl.load(group_slots + r, mask=live & keep, other=0).to(tl.int64)
        s_in = tl.load(group_slots + a - 1, mask=keep, other=0).to(tl.int64)
        layer = tl.arange(0, DELTA)[:, None]
        lm = (layer < n_delta) & live[None, :]
        conv = tl.load(gdn_conv + layer, mask=layer < n_delta, other=0)
        rec = tl.load(gdn_rec + layer, mask=layer < n_delta, other=0)
        window = tl.where(pad, null_conv, conv + s0 * conv_slot)
        rr = r[None, :].to(tl.int64)
        entry = delta_tab + (layer * rows + row[None, :]) * DELTA_WORDS
        tl.store(entry, window + (a - 1) * conv_row + 0 * rr, mask=lm)
        tl.store(entry + 1, window + rr * conv_row, mask=lm)
        rec_in = tl.where(pad, null_rec, rec + s_in * rec_slot)
        tl.store(entry + 2, rec_in + 0 * rr, mask=lm)
        rec_out = tl.where(pad, null_rec, rec + s[None, :] * rec_slot)
        tl.store(entry + 3, rec_out, mask=lm)

        # the group's QSA rows: its block-table row and caches
        q = tl.arange(0, QSA)[:, None]
        qm = (q < n_qsa) & live[None, :]
        pages = tl.where(pad, null_pages, pages_base + g.to(tl.int64) * table_bytes)
        entry = qsa_tab + (q * rows + row[None, :]) * QSA_WORDS
        tl.store(
            entry + QSA_PAGES, tl.broadcast_to(pages + 0 * rr, (QSA, ROWS)), mask=qm
        )
        for c in tl.static_range(CACHES):
            cache = tl.load(qsa_caches + q * CACHES + c, mask=q < n_qsa, other=0)
            null = tl.load(null_caches + c)
            tl.store(entry + QSA_KEY + c, tl.where(pad, null, cache) + 0 * rr, mask=qm)


def write_group_verify_meta(
    counter,
    positions,
    slots,
    accepted,
    block_tables,
    tables,
    ple_state,
    ple_window,
    nulls,
) -> None:
    """``write_verify_meta`` for a step of G groups (the module docstring):
    ``slots`` ATOM's spec slots (int32 ``[requests, S]``, rows past the batch
    -1), ``accepted`` its ``num_accepted_tokens`` (int32, row ``g`` group
    ``g``'s), ``block_tables`` int32 ``[requests, pages]``, ``tables`` the
    runner's ``Tables`` of ``G·S`` rows, ``ple_window`` the groups' windows
    ``[G, QKV, 9]`` and ``nulls`` the runner's null state
    (``write_step_meta``)."""
    t = tables
    rows = t.pos.shape[0]
    groups = ple_window.shape[0]
    s = rows // groups
    n_delta, n_qsa = t.delta.shape[0], t.qsa.shape[0]
    channels, width = _ple_geometry(ple_state, ple_window[0])
    assert rows == groups * s and slots.shape[1] == s and slots.stride(1) == 1
    assert positions.shape[0] >= rows and ple_window.is_contiguous()
    assert width == PLE_COLS + s - 1 and block_tables.stride(1) == 1
    grid = (triton.cdiv(channels * PLE_COLS, PLE_BLOCK), groups)
    _group_verify_meta_kernel[grid](
        counter,
        positions,
        t.pos,
        slots,
        slots.stride(0),
        accepted,
        t.gdn_conv,
        t.gdn_rec,
        t.delta,
        t.conv_slot,
        t.rec_slot,
        t.conv_slot // (s + 2),
        nulls["conv"],
        nulls["rec"],
        t.qsa,
        t.qsa_caches,
        block_tables.data_ptr(),
        block_tables.stride(0) * block_tables.element_size(),
        nulls["pages"],
        nulls["caches"],
        ple_state,
        ple_window,
        channels * width,
        channels,
        rows,
        n_delta,
        n_qsa,
        S=s,
        QSA_WORDS=QSA_WORDS,
        QSA_PAGES=QSA_PAGES,
        QSA_KEY=QSA_KEY,
        CACHES=len(QSA_CACHES),
        PAD_POS=PAD_POS,
        DELTA_WORDS=DELTA_WORDS,
        COLS=PLE_COLS,
        WIDTH=width,
        BLOCK=PLE_BLOCK,
        ROWS=max(2, triton.next_power_of_2(s)),
        DELTA=max(2, triton.next_power_of_2(n_delta)),
        QSA=max(2, triton.next_power_of_2(n_qsa)),
    )


@triton.jit
def _group_ple_commit_kernel(
    ple_state,
    snaps,
    slots,
    slot_stride,
    ple_slot,
    channels,
    S: tl.constexpr,
    COLS: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    g = tl.program_id(1)
    s0 = tl.load(slots + g * slot_stride).to(tl.int64)
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = (i < channels * WIDTH) & (s0 >= 0)
    c, j = i // WIDTH, i % WIDTH
    # column j of the new window: snapshot r's column j - r, r = max(j - 8, 0)
    r = g * S + tl.maximum(j - (COLS - 1), 0)
    v = tl.load(snaps + (r * channels + c) * COLS + j - (r - g * S), mask=m)
    tl.store(ple_state + s0 * ple_slot + i, v, mask=m)


def commit_ple_groups(ple_state, snaps, slots) -> None:
    """``commit_ple`` for each live group of a G-group verify step: rewrite
    request ``g``'s PLE window (slot ``slots[g, 0]``) from its rows'
    snapshots ``snaps[g·S .. g·S + k]``; a padded group's (slot < 0) are
    dropped."""
    channels, width = _ple_geometry(ple_state, snaps[0])
    s = width - PLE_COLS + 1
    groups = snaps.shape[0] // s
    assert snaps.shape[0] == groups * s and snaps.is_contiguous()
    assert slots.shape[1] == s and slots.stride(1) == 1
    grid = (triton.cdiv(channels * width, PLE_BLOCK), groups)
    _group_ple_commit_kernel[grid](
        ple_state,
        snaps,
        slots,
        slots.stride(0),
        channels * width,
        channels,
        S=s,
        COLS=PLE_COLS,
        WIDTH=width,
        BLOCK=PLE_BLOCK,
    )


@triton.jit
def _group_draft_meta_kernel(
    counter,
    logical,
    pos,
    qsa_tab,
    pages_base,
    table_bytes,
    rows,
    GROUP: tl.constexpr,
    QSA_WORDS: tl.constexpr,
    QSA_PAGES: tl.constexpr,
    ROWS: tl.constexpr,
):
    tl.store(counter, tl.load(counter) + 1)
    r = tl.arange(0, ROWS)
    live = r < rows
    p = tl.load(logical + r, mask=live, other=0)
    tl.store(pos + r, p.to(tl.int32), mask=live)
    pages = pages_base + (r // GROUP).to(tl.int64) * table_bytes
    tl.store(qsa_tab + r * QSA_WORDS + QSA_PAGES, pages, mask=live)


def write_group_draft_meta(counter, logical, block_tables, pos, qsa_tab, group):
    """``write_draft_meta`` for a draft step of several requests: advance
    ``counter``, write ``pos`` [rows] int32 from the rows' cache rows
    ``logical`` (int64) and point row ``r`` of the draft layer's QSA table
    ``qsa_tab`` [rows, qsa.STATE] at block-table row ``r // group``
    (``block_tables`` int32 ``[requests, pages]``)."""
    rows = pos.shape[0]
    assert logical.shape[0] >= rows and block_tables.stride(1) == 1
    assert block_tables.shape[0] >= rows // group
    _group_draft_meta_kernel[(1,)](
        counter,
        logical,
        pos,
        qsa_tab,
        block_tables.data_ptr(),
        block_tables.stride(0) * block_tables.element_size(),
        rows,
        GROUP=group,
        QSA_WORDS=QSA_WORDS,
        QSA_PAGES=QSA_PAGES,
        ROWS=max(2, triton.next_power_of_2(rows)),
    )
