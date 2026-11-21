# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Kimi-K3's gfx1250 step tail on the CPU (no GPU, no FlyDSL): its layout --
every address a launch pushes to or polls on a peer, the LDS regions -- and
the mirror of production's LM head GEMM dispatch."""

import itertools

import pytest

from atom.models.kimi_k3.step_tail import head_dispatch
from atom.models.kimi_k3.step_tail import layout as L

WIDTHS = [
    (rows, qlen)
    for qlen in range(2, L.MAX_QLEN + 1)
    for rows in range(qlen, L.MAX_ROWS + 1, qlen)
    if not L.step_refusal(rows, qlen)
]


def test_the_serving_widths_are_built():
    # DSpark 7 up to 8 requests, DSpark 4 up to 12
    for rows in range(8, 65, 8):
        assert (rows, 8) in WIDTHS
    for rows in range(5, 61, 5):
        assert (rows, 5) in WIDTHS


@pytest.mark.parametrize(
    "rows, qlen, why",
    [
        (64, 1, "rows a request"),
        (63, 8, "whole requests"),
        (72, 8, "past"),
        (36, 2, "past"),
        (0, 8, "whole requests"),
        (18, 9, "rows a request"),
    ],
)
def test_other_widths_are_refused(rows, qlen, why):
    assert why in L.step_refusal(rows, qlen)


@pytest.mark.parametrize("rows, qlen", WIDTHS)
def test_every_pushed_and_polled_line_is_inside_the_destination_region(rows, qlen):
    """The remote address audit, TP8, every width: a line pushed or polled
    lies whole inside the destination rank's mapped region, on a line
    boundary, and no rank pushes to itself."""
    pushes, polls = L.peer_addresses(rows, qlen)
    assert pushes and polls
    for _, source, dest, at in pushes:
        assert source != dest
        assert 0 <= at and at + L.LINE <= L.PEER_BYTES and at % L.LINE == 0
    for _, rank, at in polls:
        assert 0 <= rank < L.TP
        assert 0 <= at and at + L.LINE <= L.PEER_BYTES and at % L.LINE == 0


@pytest.mark.parametrize("rows, qlen", [(64, 8), (60, 5), (8, 8), (32, 2)])
def test_a_rank_polls_the_line_its_source_pushed_and_no_other(rows, qlen):
    pushes, polls = L.peer_addresses(rows, qlen)
    for dest in range(L.TP):
        pushed = {(kind, at) for kind, source, d, at in pushes if d == dest}
        polled = {(kind, at) for kind, rank, at in polls if rank == dest}
        # what it polls and nobody pushes are its own lines, which it keeps
        own = polled - pushed
        step = L.Step(rows, qlen)
        per_bank = rows + step.drafts * step.requests
        assert len(own) == 2 * per_bank
        assert pushed <= polled


def test_no_two_lines_share_bytes_and_the_two_exchanges_are_apart():
    seen = {}
    for bank, source, row in itertools.product((0, 1), range(L.TP), range(L.MAX_ROWS)):
        at = L.sample_line(bank, source, row)
        assert seen.setdefault(at, ("sample", bank, source, row)) == (
            "sample",
            bank,
            source,
            row,
        )
        assert at + L.LINE <= L.MARKOV_OFF
    for bank, position, source, request in itertools.product(
        (0, 1), range(L.MAX_DRAFTS), range(L.TP), range(L.MAX_REQ)
    ):
        at = L.markov_line(bank, position, source, request)
        assert at >= L.MARKOV_OFF and at + L.LINE <= L.PEER_BYTES
        key = ("markov", bank, position, source, request)
        assert seen.setdefault(at, key) == key


def test_a_bank_outlives_the_next_launch():
    # epoch e's lines sit in bank e & 1: launch e + 1 writes the other bank
    for row in range(L.MAX_ROWS):
        assert L.sample_line(0, 3, row) != L.sample_line(1, 3, row)


@pytest.mark.parametrize("rows", sorted({rows for rows, _ in WIDTHS}))
def test_the_head_fits_lds(rows):
    lds = L.head_lds(rows)
    assert lds.size <= L.LDS_LIMIT
    assert (
        L.NBUF * L.phase_buffer(rows)
        <= lds.red
        < lds.wbest
        < lds.peer
        < lds.ids
        < lds.size
    )
    assert L.phase_buffer(rows) == L.w_buffer() + L.x_buffer(rows)
    assert lds.wbest - lds.red >= max(L.RED_BYTES, L.row_groups(rows) * L.THREADS * 8)
    assert lds.peer - lds.wbest >= L.WAVES * L.MAX_QLEN * 8
    assert lds.ids - lds.peer >= L.MAX_QLEN * L.TP * 8
    # a staged row holds a phase's columns and the TDM's pad
    assert L.x_stride() == (L.PHASE_STEPS * L.K_STEP + L.LDS_PAD) * 2
    # the waves' x copies cover every row once
    per = L.x_rows_per_wave(rows)
    assert (per - 1) * L.WAVES < rows <= per * L.WAVES
    assert L.NBUF == L.DEPTH + 1 and L.THREADS == L.WAVES * L.LANES


def test_the_head_grid_covers_the_shard_once():
    tiles = [
        cta * L.TILE_WAVES + wave
        for cta in range(L.CTAS)
        for wave in range(L.TILE_WAVES)
    ]
    assert tiles == list(range(L.TILES))
    assert L.TILES * L.TILE == L.VR and L.STEPS * L.K_STEP == L.HIDDEN


def test_split_steps_follow_the_production_kernels():
    # one pass over K
    assert L.split_steps("flydsl", 1, 128) == ()
    assert L.split_steps("opus", 1, 128) == ()
    # AITER's split-K workspace kernel: 56 tiles of 128 in 3 slices of 19, 19, 18
    assert L.split_steps("opus", 3, 128) == (76, 152)
    # its launcher's clamp: 7 tiles of 1024 cannot make 8 slices
    assert L.split_steps("opus", 8, 1024) == tuple(range(32, 224, 32))
    assert L.split_steps("flydsl", 2, 128) == (112,)
    assert L.split_steps("flydsl", 4, 64) == (56, 112, 168)
    with pytest.raises(ValueError):
        L.split_steps("flydsl", 3, 128)
    with pytest.raises(ValueError):
        L.split_steps("asm", 2, 128)
    for splits in (L.split_steps("opus", 3, 128), L.split_steps("flydsl", 4, 64)):
        assert all(0 < s < L.STEPS for s in splits) and list(splits) == sorted(splits)


@pytest.mark.parametrize(
    "libtype, name, split_k, want",
    [
        (
            "flydsl",
            "flydsl_hgemm_abf16_wbf16_bf16_t64x128x128x5_ks1_w2x2x1_bias0_ktail0_gm0_pht_gfx1250",
            1,
            ("flydsl", 1, 128),
        ),
        (
            "opus",
            "opus_gemm_gfx1250_splitk_clusterlaunch_tdm_ws_128x16x256x128_1x2_16x16x32_0x0x0_c1x2_p3w1",
            3,
            ("opus", 3, 128),
        ),
        (
            "opus",
            "opus_gemm_gfx1250_splitk_clusterlaunch_tdm_ws_128x32x128x128_2x1_16x16x32_0x0x0_c1x4_p3w2",
            3,
            ("opus", 3, 128),
        ),
    ],
)
def test_the_tuned_rows_production_runs_are_mirrored(libtype, name, split_k, want):
    assert head_dispatch.parse_head_kernel(libtype, name, split_k) == want


@pytest.mark.parametrize(
    "libtype, name",
    [
        ("triton", ""),
        ("torch", None),
        ("asm", "_ZN5aiter24bf16gemm_bf16_tn_256x256E"),
        # a gfx950 row, another OPUS pipeline
        (
            "flydsl",
            "flydsl_hgemm_abf16_wbf16_bf16_t64x96x128x3_ks1_w2x2x1_bias0_ktail0_gm0_stg1_bnt_pbs_gfx950",
        ),
        (
            "opus",
            "opus_gemm_gfx1250_4wave_wl_co_128x64x192x128_4x1_16x16x32_0x0x0_w1x4_c2x2_p3_v1024w1",
        ),
    ],
)
def test_a_kernel_without_a_mirror_refuses_the_width(libtype, name):
    with pytest.raises(ValueError):
        head_dispatch.parse_head_kernel(libtype, name, 1)


# --- the draft launch (draft head + Markov rounds) ----------------------------


def test_the_draft_launch_serving_shapes_are_built():
    from atom.models.kimi_k3.step_tail import markov_layout as M

    for requests in range(1, 9):  # DSpark 7 up to 8 requests
        assert not M.draft_refusal(requests, 7, 256)
    for requests in (8, 12, 16):  # DSpark 4
        assert not M.draft_refusal(requests, 4, 256)
    assert not M.draft_refusal(8, 7, 512)
    assert "drafts a request" in M.draft_refusal(8, 8, 256)
    assert "past" in M.draft_refusal(10, 7, 256)
    assert "past" in M.draft_refusal(17, 2, 256)
    assert "Markov rank" in M.draft_refusal(8, 7, 128)


def test_the_markov_rounds_lds_sits_below_the_draft_logits():
    from atom.models.kimi_k3.step_tail import markov_layout as M

    for rank in M.RANKS:
        ml = M.markov_lds(rank)
        offsets = [ml.w2, ml.w1, ml.x, ml.best, ml.lane, ml.wave, ml.peer, ml.size]
        assert offsets == sorted(offsets) and len(set(offsets)) == len(offsets)
        assert ml.w1 - ml.w2 == L.TILE_WAVES * L.TILE * M.pitch(rank)
        assert ml.x - ml.w1 == L.MAX_REQ * M.pitch(rank)
        assert ml.lane - ml.best == L.THREADS * 8
        assert ml.size - ml.peer == L.MAX_REQ * L.TP * 8
        for rows in range(1, L.MAX_ROWS + 1):
            lds = L.head_lds(rows)
            # the rounds reuse the head's phase buffers; the draft logits are above them
            assert (
                ml.size <= lds.red and lds.red + L.WAVES * L.MAX_ROWS * 32 <= lds.wbest
            )


def test_the_draft_launch_lines_are_inside_every_region():
    # the rank's own line of a round is stored to every rank's region, its own too
    for bank, position, source, request in itertools.product(
        (0, 1), range(L.MAX_DRAFTS), range(L.TP), range(L.MAX_REQ)
    ):
        at = L.markov_line(bank, position, source, request)
        assert L.MARKOV_OFF <= at and at + L.LINE <= L.PEER_BYTES and at % L.LINE == 0
