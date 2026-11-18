# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only geometry of the gfx1250 fused K2 tail (all-reduce, routed norm,
finalize in one launch)."""

import pytest

from atom.models.kimi_k3.mono.config import HIDDEN, LATENT
from atom.models.kimi_k3.mono.kernels.stages import k2_allreduce_gfx1250 as ar
from atom.models.kimi_k3.mono.kernels.stages import k2_tail_gfx1250 as tail
from atom.models.kimi_k3.mono.kernels.stages.gemv_layout import (
    TDM_CLUSTER,
    TDM_ROW_STRIDE,
    tdm_task_words,
)
from atom.mono.plan.architecture import GFX1250

WIDTHS = range(1, tail.tail_max_rows() + 1)


def test_tail_takes_the_widths_of_the_seven_element_gathering_all_reduce():
    rows = tail.tail_max_rows()
    assert ar.allreduce7_lines(rows) <= ar.MAX_CTAS * ar.THREADS
    assert ar.allreduce7_lines(rows + 1) > ar.MAX_CTAS * ar.THREADS
    assert ar.build_gather_allreduce(8, rows).name.startswith(
        "k3_k2_gfx1250_gather_allreduce7"
    )


@pytest.mark.parametrize("rows", WIDTHS)
def test_grid_holds_every_all_reduce_cta_and_the_finalize_in_whole_clusters(rows):
    grid = tail.tail_grid(rows)
    assert grid % TDM_CLUSTER == 0 and tail.PAIRS <= grid <= ar.MAX_CTAS
    ctas = tail.tail_ar_ctas(rows)
    assert ctas <= grid
    # a line a thread covers both partials; a routed row is two CTAs' lines
    assert ctas * ar.THREADS * ar.LINE7_ELEMENTS == rows * (LATENT + HIDDEN)
    assert ar.LINE7_ROUTED == 2 * ar.THREADS
    # every epoch counter is in some thread's reach
    assert grid * ar.THREADS >= ar.MAX_CTAS


@pytest.mark.parametrize("rows", WIDTHS)
def test_lds_regions_fit_and_never_overlap_while_live(rows):
    layout = tail.tail_layout(rows)
    assert layout.words * 4 <= GFX1250.max_workgroup_lds_bytes
    assert layout.depth == tail.TAIL_DEPTH and 2 <= layout.tokens <= tail.SLABS
    weights = layout.gemv + layout.tokens * rows * TDM_ROW_STRIDE // 2
    # the all-reduce's scratch only aliases token slabs, filled after it
    assert tail.AR_WORDS <= weights
    stage = 2 * tdm_task_words()
    stages = layout.depth + 1
    assert layout.words == weights + stages * stage
    # the epilogue's sums sit on weight stages the last slab does not read
    tiles = -(-rows // 16)
    red_end = layout.red + 2 * tiles * tail.ACCUMULATOR_WORDS
    assert weights <= layout.red and red_end <= layout.words
    last = weights + (tail.SLABS - 1) % stages * stage
    assert red_end <= last or layout.red >= last + stage


def test_norm_reduces_in_aiter_rmsnorm2d_thread_order():
    # 256 threads of two 8-element runs at LATENT: the last wave holds none
    assert tail.NORM_THREADS == 224
    assert 2 * tail.NORM_RUN * tail.NORM_THREADS == LATENT
    assert tail.NORM_THREADS % GFX1250.wave_size == 0


@pytest.mark.parametrize("identity", (False, True))
@pytest.mark.parametrize("rows", (1, 20, 40, 42))
def test_builder_launches_the_grid_on_the_tail_abi(rows, identity):
    kernel = tail.build_k2_tail(8, rows, 1e-5, identity_routes=identity)
    assert kernel.abi is tail.TAIL_ABI
    assert kernel.grid == (tail.tail_grid(rows),) and kernel.block == (ar.THREADS,)
    assert kernel.ar_ctas == tail.tail_ar_ctas(rows) and kernel.eps == 1e-5


def test_builder_refuses_widths_past_a_line_a_thread():
    with pytest.raises(ValueError):
        tail.build_k2_tail(8, tail.tail_max_rows() + 1, 1e-5)


def test_flags_hold_a_word_a_cta_and_the_profile_stamps():
    assert tail.tail_flag_words() == ar.MAX_CTAS
    assert tail.tail_flag_words(profile=True) == ar.MAX_CTAS * (1 + tail.STAMPS)


def test_split_waves_push_and_reduce_every_line_of_the_cta_once():
    assert tail.PUSH_WAVES * 2 == tail.WAVES
    pushed = sorted(
        t + k * tail.PUSH_THREADS
        for t in range(tail.PUSH_THREADS)
        for k in range(tail.SPLIT_LINES)
    )
    reduced = sorted(
        line
        for tid in range(tail.PUSH_THREADS, ar.THREADS)
        for line in tail._split_lines({"tid": tid})
    )
    assert pushed == reduced == list(range(ar.THREADS))


@pytest.mark.parametrize("rows", WIDTHS)
def test_split_lds_holds_the_lines_and_gains_inside_the_token_slabs(rows):
    layout = tail.tail_layout(rows, split=True)
    assert layout.words * 4 <= GFX1250.max_workgroup_lds_bytes
    weights = layout.gemv + layout.tokens * rows * TDM_ROW_STRIDE // 2
    assert tail.SPLIT_AR_WORDS <= weights
    assert tail.SPLIT_AR_WORDS == (
        tail.AR_WORDS + ar.THREADS * ar.LINE_WORDS + ar.THREADS * ar.LINE7_ELEMENTS
    )
    # the split's scratch moves nothing at the widths it fits beside
    plain = tail.tail_layout(rows)
    if tail.SPLIT_AR_WORDS <= plain.gemv + plain.tokens * rows * TDM_ROW_STRIDE // 2:
        assert layout == plain


def test_split_and_bounded_builds_are_named_apart(monkeypatch):
    plain = tail.build_k2_tail(4, 20, 1e-5, identity_routes=True, split=False)
    split = tail.build_k2_tail(4, 20, 1e-5, identity_routes=True, split=True)
    bounded = tail.build_k2_tail(4, 20, 1e-5, split=True, poll_bound=1 << 22)
    assert "_sw" not in plain.name and split.name == plain.name + "_sw"
    assert bounded.name.endswith(f"_sw_b{1 << 22}")
    monkeypatch.setenv("ATOM_KIMI_MONO_TAIL_SPLIT", "1")
    assert tail.build_k2_tail(4, 20, 1e-5, identity_routes=True).name == split.name
    # a bounded build's give-up mark sits past every width's flags
    assert tail.tail_ar_ctas(tail.tail_max_rows()) <= tail.GAVE_UP_WORD


def test_split_switch_defaults_off(monkeypatch):
    from atom.utils import envs

    monkeypatch.delenv("ATOM_KIMI_MONO_TAIL_SPLIT", raising=False)
    assert envs.ATOM_KIMI_MONO_TAIL_SPLIT is False
    monkeypatch.setenv("ATOM_KIMI_MONO_TAIL_SPLIT", "1")
    assert envs.ATOM_KIMI_MONO_TAIL_SPLIT is True
