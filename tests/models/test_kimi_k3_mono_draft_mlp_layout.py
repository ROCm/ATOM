# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only geometry of the gfx1250 DSpark draft projection kernels
(``draft_mlp_layout``), and the audit of every peer address a launch of
``draft_proj_ar_gfx1250`` stores to or loads from: on this platform a store
or load outside a peer's mapped bank does not fault, it resets the tray."""

import itertools

import numpy as np
import pytest

from atom.models.kimi_k3.mono.config import HIDDEN, MAX_ROWS
from atom.models.kimi_k3.mono.kernels.stages import draft_mlp_layout as lay
from atom.mono.plan.architecture import GFX1250

WIDTHS = (1024, 1792, 2048, 3584)


def test_the_ctas_cover_the_output_columns_once():
    assert lay.CTAS * lay.CTA_COLUMNS == HIDDEN
    assert lay.TASKS * 16 == lay.CTA_COLUMNS and lay.CTA_PAIRS * 2 == lay.CTA_COLUMNS
    # a CTA's rows lie in one 128-row block of the weight's scales
    for cta in range(lay.CTAS):
        first = cta * lay.CTA_COLUMNS
        assert (
            first // 128
            == (first + lay.CTA_COLUMNS - 1) // 128
            == lay.weight_block(cta)
        )
        assert lay.task_weight_byte(cta, 1, 1792) == (first + 16) * 1792


@pytest.mark.parametrize("k", WIDTHS)
@pytest.mark.parametrize("rows", (1, 7, 16, 17, 28, 33, 48, 49, 56, 64))
def test_a_ctas_lds_holds_its_weights_tokens_codes_stage_and_scratch(rows, k):
    limit = GFX1250.max_workgroup_lds_bytes
    for phases in lay.TOKEN_PHASES:
        offsets = [
            0,
            lay.token_offset(k),
            lay.code_offset(rows, k),
            lay.stage_offset(rows, k),
            lay.scratch_offset(rows, k),
            lay.lds_words(rows, k, phases),
        ]
        sizes = [
            lay.weight_words(k),
            lay.token_words(rows, k),
            lay.code_words(rows, k),
            lay.stage_words(rows),
            lay.scratch_words(rows, k, phases),
        ]
        assert [b - a for a, b in itertools.pairwise(offsets)] == sizes
        # a phase's activation as its eight TDM copies land it
        assert lay.scratch_words(rows, k, phases) * 4 == rows * (k // phases) * 2
        assert lay.token_rows_a_wave(rows) * lay.WAVES >= rows
    assert lay.weight_words(k) * 4 == lay.TASKS * 16 * k
    assert lay.token_words(rows, k) * 4 == lay.tiles(rows) * 16 * k
    phases = lay.token_phases(rows, k)
    fits = [4 * lay.lds_words(rows, k, n) <= limit for n in lay.TOKEN_PHASES]
    assert (phases is None) == (not any(fits))
    if phases is not None:
        # the fewest phases that fit
        assert phases == lay.TOKEN_PHASES[fits.index(True)]
    assert (lay.refusal(rows, k, 8) is None) == (phases is not None)


def test_the_draft_shards_at_tp8_fit_every_width():
    for k in (1024, 1792):
        for rows in range(1, MAX_ROWS + 1):
            assert lay.refusal(rows, k, 8) is None
    # o_proj's activation is copied whole at every width, down_proj's up to
    # 48 rows and in K halves past them
    assert {lay.token_phases(r, 1024) for r in range(1, MAX_ROWS + 1)} == {1}
    assert [lay.token_phases(r, 1792) for r in (1, 28, 48, 49, 56, 64)] == [
        1, 1, 1, 2, 2, 2
    ]  # fmt: skip


def test_shapes_the_kernel_does_not_take_are_refused():
    assert lay.refusal(0, 1792, 8) and lay.refusal(MAX_ROWS + 1, 1792, 8)
    assert lay.refusal(7, 1792 + 128, 8)  # an odd count of K groups
    assert lay.refusal(7, 1000, 8) and lay.refusal(7, 1792, 9)
    assert lay.refusal(64, 3584, 4)  # past the LDS
    # a TP4 shard of down_proj fits up to two token tiles, in K quarters
    assert lay.refusal(32, 3584, 4) is None and lay.token_phases(32, 3584) == 4
    assert lay.refusal(33, 3584, 4)
    assert {lay.refusal(r, 2048, 4) for r in range(1, MAX_ROWS + 1)} == {None}


@pytest.mark.parametrize("k", WIDTHS)
@pytest.mark.parametrize("rows", (1, 7, 21, 56, 64))
def test_a_quant_group_is_eight_consecutive_lanes_of_one_round(rows, k):
    total = lay.quant_items(rows, k)
    assert total == rows * (k // 128) * 8 and total % lay.QUANT_LANES == 0
    assert lay.quant_rounds(rows, k) * lay.THREADS >= total
    assert lay.quant_items(rows, k, 2) * 2 == total
    if (k // 128) % 4 == 0:
        assert lay.quant_items(rows, k, 4) * 4 == total
    assert lay.THREADS % lay.QUANT_LANES == 0 and lay.LANES % lay.QUANT_LANES == 0
    assert lay.QUANT_LANES * lay.QUANT_LANE_ELEMENTS == lay.K_GROUP


def _launch_lines(tp, rows, parity):
    """Every (source rank, CTA, item) line word a launch pushes, in a rank's
    bank region: ``[source][cta][live item]``."""
    items = np.arange(lay.line_rounds(rows) * lay.THREADS, dtype=np.int64)
    live = items[lay.line_live(rows, items)]
    cta = np.arange(lay.CTAS, dtype=np.int64)[:, None]
    bank = parity * lay.bank_words(tp, rows)
    return np.stack(
        [lay.line_word(source, rows, bank, cta, live[None, :]) for source in range(tp)]
    )


@pytest.mark.parametrize("tp", (1, 2, 4, 8))
def test_every_pushed_and_polled_line_lies_in_the_destination_bank(tp):
    """The audit. A launch at ``rows`` rows on rank ``r`` stores 16 B at
    ``peers[p] + 4 line_word(r, rows, bank, cta, item)`` for every rank p,
    and loads 16 B at ``peers[r] + 4 line_word(p, ..)`` for every rank p --
    the same word function, so a polled line is one its source pushes in the
    same launch. For every width, both epoch parities and every source rank:
    each line is 16-byte aligned inside ``peer_bytes(tp, rows)`` of the
    destination's base, no two (source, CTA, item) share a line, and the
    launch's lines are exactly the epoch's bank."""
    for rows in range(1, MAX_ROWS + 1):
        mapped = lay.peer_bytes(tp, rows)
        lines_a_bank = lay.triples(rows) * tp * (HIDDEN // 2)
        assert mapped == 2 * lines_a_bank * lay.LINE_BYTES
        for parity in (0, 1):
            words = _launch_lines(tp, rows, parity)
            first = 4 * words
            assert first.min() >= 0 and (first % lay.LINE_BYTES == 0).all()
            assert (first + lay.LINE_BYTES).max() <= mapped
            # the i32 word index and its i64 byte offset do not wrap
            assert words.max() < 2**31
            local = (words - parity * lay.bank_words(tp, rows)) // lay.LINE_WORDS
            assert np.array_equal(np.sort(local, axis=None), np.arange(lines_a_bank))
            # a source's lines stay in its own slice of the bank
            for source in range(tp):
                slice_lines = lines_a_bank // tp
                assert local[source].min() == source * slice_lines
                assert local[source].max() == (source + 1) * slice_lines - 1
        # a region mapped for more rows holds every narrower launch
        assert mapped <= lay.peer_bytes(tp, MAX_ROWS)


@pytest.mark.parametrize("rows", (1, 7, 48, 49, 56, 64))
def test_dead_items_are_neither_pushed_nor_polled(rows):
    items = np.arange(lay.line_rounds(rows) * lay.THREADS)
    live = lay.line_live(rows, items)
    assert live.sum() == lay.triples(rows) * lay.CTA_PAIRS
    assert live[: live.sum()].all() and not live[live.sum() :].any()
    # the stage holds every live line's three tokens
    assert lay.stage_words(rows) == live.sum() * lay.LINE_TOKENS


def test_the_bank_layout_is_the_o_proj_all_reduces_one_shot():
    pytest.importorskip("flydsl", reason="the kernels build with FlyDSL")
    pytest.importorskip("aiter", reason="the kernel under test needs aiter")
    from atom.models.kimi_k3.mono.kernels.stages import k2_oproj_ar_gfx1250 as oar

    assert (lay.LINE_TOKENS, lay.LINE_WORDS) == (oar.LINE_TOKENS, oar.LINE_WORDS)
    assert (lay.CTAS, lay.CTA_PAIRS) == (oar.CTAS, oar.CTA_PAIRS)
    for tp in (1, 2, 4, 8):
        for rows in (1, 7, 48, 49, 64):
            assert lay.bank_words(tp, rows) == oar.oneshot_bank_words(tp, rows)
            assert lay.peer_bytes(tp, rows) <= oar.oproj_peer_bytes(tp, rows, True)
            count = oar.triples(rows)
            for rank, cta, item in (
                (0, 0, 0),
                (tp - 1, 223, count * 16 - 1),
                (0, 100, 17),
            ):
                # k2_oproj_ar_gfx1250._push_oneshot's word
                want = oar.LINE_WORDS * (
                    (rank * count + item // 16) * (HIDDEN // 2) + cta * 16 + item % 16
                )
                assert lay.line_word(rank, rows, 0, cta, item) == want


@pytest.mark.parametrize("rows", (1, 7, 16, 17, 33, 48, 64))
def test_gate_up_clusters_cover_the_shard_once(rows):
    inter = 1792
    clusters = lay.gate_up_clusters(inter)
    assert clusters * lay.GATE_UP_COLUMNS == inter
    assert lay.gate_up_ctas(inter) == 4 * clusters <= 256
    # every 16-row task of the [2 inter, HIDDEN] shard is one slot of one
    # cluster: gate rows, then up rows
    rows_seen = sorted(
        lay.gate_up_slot_row(c, slot, inter)
        for c in range(clusters)
        for slot in range(lay.GATE_UP_SLOTS)
    )
    assert rows_seen == list(range(0, 2 * inter, 16))
    for c in (0, 5, clusters - 1):
        for owner in range(lay.GATE_UP_OWNERS):
            column = lay.gate_up_column(c, owner)
            # an owner's gate slot and up slot are the same columns
            assert lay.gate_up_slot_row(c, owner, inter) == column
            assert lay.gate_up_slot_row(c, 2 + owner, inter) == inter + column
    assert sorted(
        lay.gate_up_column(c, o) for c in range(clusters) for o in range(2)
    ) == list(range(0, inter, 16))
    for split in lay.GATE_UP_SPLITS:
        assert lay.gate_up_refusal(rows, inter, split) is None
    assert 4 * lay.gate_up_lds_words(rows) <= GFX1250.max_workgroup_lds_bytes
    assert lay.gate_up_chains(rows) == 4 * lay.tiles(rows) <= 2 * lay.WAVES


@pytest.mark.parametrize("split_k", lay.GATE_UP_SPLITS)
def test_a_clusters_ctas_close_every_split_once_in_k_order(split_k):
    closed = {}
    for rank in range(lay.GATE_UP_CLUSTER):
        for piece in range(2):
            split = lay.gate_up_piece_split(split_k, rank, piece)
            if split is not None:
                # the K groups the split's chain ran over, from zero
                first = piece if split_k == 8 else 0
                groups = range(
                    rank * lay.QUARTER_GROUPS + first * lay.PIECE_GROUPS,
                    rank * lay.QUARTER_GROUPS + (piece + 1) * lay.PIECE_GROUPS,
                )
                assert split not in closed
                closed[split] = list(groups)
    assert sorted(closed) == list(range(split_k))
    per = lay.GATE_UP_GROUPS // split_k
    for split, groups in closed.items():
        # production's split z: K groups [z K / split_k, (z + 1) K / split_k)
        assert groups == list(range(split * per, (split + 1) * per))


@pytest.mark.parametrize("split_k", lay.GATE_UP_SPLITS)
@pytest.mark.parametrize("rows", (7, 21, 48, 64))
def test_the_split_exchange_gives_every_sum_its_own_words(rows, split_k):
    inter = 1792
    index = np.indices(
        (
            lay.gate_up_clusters(inter),
            lay.GATE_UP_SLOTS,
            split_k,
            lay.tiles(rows),
            lay.LANES,
        )
    )
    at = lay.gate_up_partial_index(*index, split_k, rows).reshape(-1)
    assert np.array_equal(np.sort(at), np.arange(at.size) * 8)
    assert (at.max() + 8) * 4 <= lay.gate_up_partial_bytes(inter)


def test_gate_up_mirrors_the_dispatchs_split_k_and_refuses_the_rest():
    for split in (1, 2, 16):
        assert lay.gate_up_refusal(7, 1792, split)
    assert lay.gate_up_refusal(7, 1800, 4) and lay.gate_up_refusal(0, 1792, 4)
    assert lay.gate_up_refusal(65, 1792, 8)
    assert lay.gate_up_refusal(7, 3584, 4)  # more CTAs than CUs
