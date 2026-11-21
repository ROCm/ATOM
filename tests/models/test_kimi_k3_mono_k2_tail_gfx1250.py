# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only geometry of the gfx1250 fused K2 tail (all-reduce, routed norm,
finalize in one launch)."""

import pytest

from atom.models.kimi_k3.mono.config import HIDDEN, LATENT

pytest.importorskip("flydsl", reason="the K3 gfx1250 mono kernels are FlyDSL")

from atom.models.kimi_k3.mono.kernels.stages import k2_allreduce_gfx1250 as ar
from atom.models.kimi_k3.mono.kernels.stages import k2_tail_gfx1250 as tail
from atom.models.kimi_k3.mono.kernels.stages.gemv_layout import (
    TDM_CLUSTER,
    TDM_ROW_STRIDE,
    tdm_task_words,
)
from atom.mono.plan.architecture import GFX1250

WIDTHS = range(1, tail.tail_max_rows() + 1)


def test_tail_takes_a_share_a_cta_at_the_widths_of_the_seven_element_all_reduce():
    rows = tail.tail_line_rows()
    assert rows == 42 and tail.tail_max_rows() == 64
    assert tail.tail_rounds(rows) == 1 and tail.tail_rounds(rows + 1) == 2
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
    # every share a CTA's in some round, the routed ones (and their
    # partners) in round 0
    assert ctas <= tail.tail_rounds(rows) * grid and 2 * rows <= grid
    assert (ctas <= grid) == (tail.tail_rounds(rows) == 1)
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
@pytest.mark.parametrize("rows", (1, 20, 40, 42, 43, 50, 55, 60, 64))
def test_builder_launches_the_grid_on_the_tail_abi(rows, identity):
    kernel = tail.build_k2_tail(8, rows, 1e-5, identity_routes=identity)
    assert kernel.abi is tail.TAIL_ABI
    assert kernel.grid == (tail.tail_grid(rows),) and kernel.block == (ar.THREADS,)
    assert kernel.ar_ctas == tail.tail_ar_ctas(rows) and kernel.eps == 1e-5


def test_builder_refuses_widths_past_the_cap():
    with pytest.raises(ValueError):
        tail.build_k2_tail(8, tail.tail_max_rows() + 1, 1e-5)


def test_a_profile_tail_takes_a_share_a_cta():
    with pytest.raises(ValueError, match="share a CTA"):
        tail.build_k2_tail(8, 64, 1e-5, profile=True)


def test_flags_hold_a_word_a_cta_and_the_profile_stamps():
    assert tail.tail_flag_words() == ar.MAX_CTAS
    assert tail.tail_flag_words(profile=True) == ar.MAX_CTAS * (1 + tail.STAMPS)
    # a bounded build's give-up mark sits past every CTA's flag and stamps
    assert tail.GAVE_UP_WORD == tail.tail_flag_words(profile=True)
    # then thread 0's polls a CTA
    assert tail.tail_flag_words(bounded=True) == tail.GAVE_UP_WORD + 1 + ar.MAX_CTAS
    assert tail.tail_flag_words(True, bounded=True) == tail.tail_flag_words(
        bounded=True
    )
    with pytest.raises(ValueError, match="poll bound"):
        tail.build_k2_tail(4, 8, 1e-5, identity_routes=True, test_skip_push=True)
    dry = tail.build_k2_tail(
        4, 8, 1e-5, identity_routes=True, poll_bound=9, test_skip_push=True
    )
    assert dry.name.endswith("_np_b9")


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


@pytest.mark.parametrize("rows", range(1, tail.tail_line_rows() + 1))
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
    monkeypatch.setenv("ATOM_KIMI_MONO_TAIL_EARLY_GATHER", "0")
    plain = tail.build_k2_tail(4, 20, 1e-5, identity_routes=True, split=False)
    split = tail.build_k2_tail(4, 20, 1e-5, identity_routes=True, split=True)
    bounded = tail.build_k2_tail(4, 20, 1e-5, split=True, poll_bound=1 << 22)
    assert "_sw" not in plain.name and split.name == plain.name + "_sw"
    assert bounded.name.endswith(f"_sw_b{1 << 22}")
    monkeypatch.setenv("ATOM_KIMI_MONO_TAIL_SPLIT", "1")
    assert tail.build_k2_tail(4, 20, 1e-5, identity_routes=True).name == split.name


def test_split_waves_refuse_rounds_and_the_switch_skips_them(monkeypatch):
    wide = tail.tail_line_rows() + 1
    assert tail.tail_rounds(wide) > 1
    with pytest.raises(ValueError, match="split waves take a share a CTA"):
        tail.build_k2_tail(4, wide, 1e-5, identity_routes=True, split=True)
    monkeypatch.setenv("ATOM_KIMI_MONO_TAIL_SPLIT", "1")
    plain = tail.build_k2_tail(4, wide, 1e-5, identity_routes=True, split=False)
    assert tail.build_k2_tail(4, wide, 1e-5, identity_routes=True).name == plain.name


def test_tail_defaults_on(monkeypatch):
    from atom.utils import envs

    monkeypatch.delenv("ATOM_KIMI_MONO_TAIL", raising=False)
    assert envs.ATOM_KIMI_MONO_TAIL is True
    monkeypatch.setenv("ATOM_KIMI_MONO_TAIL", "0")
    assert envs.ATOM_KIMI_MONO_TAIL is False


def test_split_switch_defaults_off(monkeypatch):
    from atom.utils import envs

    monkeypatch.delenv("ATOM_KIMI_MONO_TAIL_SPLIT", raising=False)
    assert envs.ATOM_KIMI_MONO_TAIL_SPLIT is False
    monkeypatch.setenv("ATOM_KIMI_MONO_TAIL_SPLIT", "1")
    assert envs.ATOM_KIMI_MONO_TAIL_SPLIT is True


def test_early_gather_builds_apart_and_only_a_share_a_cta(monkeypatch):
    monkeypatch.delenv("ATOM_KIMI_MONO_TAIL_EARLY_GATHER", raising=False)
    plain = tail.build_k2_tail(4, 20, 1e-5, identity_routes=True)
    early = tail.build_k2_tail(4, 20, 1e-5, identity_routes=True, early_gather=True)
    assert "_eg" not in plain.name and early.name == plain.name + "_eg"
    assert early.grid == plain.grid and early.abi is plain.abi
    monkeypatch.setenv("ATOM_KIMI_MONO_TAIL_EARLY_GATHER", "1")
    assert tail.build_k2_tail(4, 20, 1e-5, identity_routes=True).name == early.name
    # split waves gather on their own path; past one round the shares' lines
    # are gathered together as before
    split = tail.build_k2_tail(4, 20, 1e-5, identity_routes=True, split=True)
    assert "_eg" not in split.name
    wide = tail.tail_line_rows() + 1
    assert "_eg" not in tail.build_k2_tail(4, wide, 1e-5, identity_routes=True).name


def test_early_gather_switch_defaults_off(monkeypatch):
    from atom.utils import envs

    monkeypatch.delenv("ATOM_KIMI_MONO_TAIL_EARLY_GATHER", raising=False)
    assert envs.ATOM_KIMI_MONO_TAIL_EARLY_GATHER is False
    monkeypatch.setenv("ATOM_KIMI_MONO_TAIL_EARLY_GATHER", "1")
    assert envs.ATOM_KIMI_MONO_TAIL_EARLY_GATHER is True


def test_split_weight_issue_builds_apart_and_only_a_share_a_cta():
    plain = tail.build_k2_tail(4, 8, 1e-5, identity_routes=True)
    deep = tail.build_k2_tail(4, 8, 1e-5, identity_routes=True, depth=6, first_depth=3)
    assert "_d6_" in deep.name and deep.name.endswith("_f3")
    assert deep.name != plain.name
    assert deep.grid == plain.grid and deep.abi is plain.abi
    with pytest.raises(ValueError, match="share a CTA"):
        tail.build_k2_tail(4, 20, 1e-5, split=True, first_depth=3)
    with pytest.raises(ValueError, match="share a CTA"):
        tail.build_k2_tail(4, tail.tail_line_rows() + 1, 1e-5, first_depth=3)


def test_poll_together_and_resident_switches_default_off(monkeypatch):
    from atom.utils import envs

    for name in ("POLL_TOGETHER", "RESIDENT"):
        monkeypatch.delenv(f"ATOM_KIMI_MONO_TAIL_{name}", raising=False)
        assert getattr(envs, f"ATOM_KIMI_MONO_TAIL_{name}") is False
    plain = tail.build_k2_tail(4, 8, 1e-5, identity_routes=True)
    assert "_pt" not in plain.name and "_rs" not in plain.name
    assert f"_d{tail.TAIL_DEPTH}_" in plain.name
    monkeypatch.setenv("ATOM_KIMI_MONO_TAIL_POLL_TOGETHER", "1")
    polled = tail.build_k2_tail(4, 8, 1e-5, identity_routes=True)
    assert polled.name == plain.name + "_pt"
    assert polled.grid == plain.grid and polled.abi is plain.abi
    # one rank keeps the volatile loads
    assert "_pt" not in tail.build_k2_tail(1, 8, 1e-5, identity_routes=True).name
    # past one round the shares poll together too
    wide = tail.tail_line_rows() + 1
    assert tail.build_k2_tail(4, wide, 1e-5, identity_routes=True).name.endswith("_pt")


def test_resident_switch_takes_the_widths_whose_slabs_all_fit(monkeypatch):
    monkeypatch.delenv("ATOM_KIMI_MONO_TAIL_POLL_TOGETHER", raising=False)
    monkeypatch.setenv("ATOM_KIMI_MONO_TAIL_RESIDENT", "1")
    most = tail.tail_resident_rows()
    assert most == 13
    for rows in (1, 5, 8, most):
        layout = tail.tail_layout(rows, tail.SLABS - 1)
        assert layout.depth == tail.SLABS - 1 and layout.tokens == tail.SLABS
        assert layout.words * 4 <= GFX1250.max_workgroup_lds_bytes
        kernel = tail.build_k2_tail(4, rows, 1e-5, identity_routes=True)
        assert kernel.name.endswith(f"_id_f{tail.TAIL_DEPTH}_rs")
        assert f"_d{tail.SLABS - 1}_t{tail.SLABS}_" in kernel.name
    for rows in (most + 1, 16, 32, 42, 43, 64):
        assert "_rs" not in tail.build_k2_tail(4, rows, 1e-5, identity_routes=True).name
    # an explicit layout, split waves or profile-less variants keep theirs
    assert "_rs" not in tail.build_k2_tail(4, 8, 1e-5, depth=3).name
    assert "_rs" not in tail.build_k2_tail(4, 8, 1e-5, split=True).name
    with pytest.raises(ValueError, match="resident"):
        tail.build_k2_tail(4, most + 1, 1e-5, depth=tail.SLABS - 1, resident=True)
    with pytest.raises(ValueError, match="resident"):
        tail.build_k2_tail(4, 8, 1e-5, resident=True)


def test_test_only_builds_are_named_apart():
    plain = tail.build_k2_tail(4, 8, 1e-5, identity_routes=True)
    held = tail.build_k2_tail(4, 8, 1e-5, identity_routes=True, push_delay_us=3)
    assert held.name == plain.name + "_pd3"
    spy = tail.build_k2_tail(4, 8, 1e-5, identity_routes=True, profile=True, spy=True)
    assert spy.name.endswith("_profile_spy")
    with pytest.raises(ValueError, match="profile"):
        tail.build_k2_tail(4, 8, 1e-5, identity_routes=True, spy=True)


def test_every_profile_point_has_a_stamp_word():
    assert tail.P_START == 0 and tail.P_SPARE1 == tail.STAMPS - 1 <= 31


@pytest.mark.parametrize("rows", WIDTHS)
def test_helpers_push_every_line_once_inside_the_bank(rows):
    lines = ar.allreduce7_lines(rows)
    most = tail.tail_helpers_max(rows)
    if tail.tail_rounds(rows) > 1:
        assert most == 0
        return
    for helpers in (0, 1, 2, 4, 8):
        if helpers > most:
            with pytest.raises(ValueError):
                tail.build_k2_tail(4, rows, 1e-5, identity_routes=True, helpers=helpers)
            continue
        pushed = tail.tail_push_lines(rows, helpers)
        # every line by exactly one thread, a thread at most one line
        assert sorted(pushed.values()) == list(range(lines))
        for (bid, _), line in pushed.items():
            routed = line < rows * ar.LINE7_ROUTED
            assert (bid >= tail.tail_ar_ctas(rows)) == bool(helpers and routed)
        for tp in (4, 8):
            bank_words = tp * lines * ar.LINE_WORDS
            last = (
                bank_words
                + ((tp - 1) * lines + max(pushed.values()) + 1) * ar.LINE_WORDS
            )
            assert last * 4 <= 2 * bank_words * 4 <= ar.allreduce_peer_bytes(tp, rows)


def test_helper_builds_are_named_apart_and_take_identity_routes():
    plain = tail.build_k2_tail(4, 8, 1e-5, identity_routes=True)
    helped = tail.build_k2_tail(4, 8, 1e-5, identity_routes=True, helpers=4)
    assert helped.name == plain.name + "_h4" and helped.grid == plain.grid
    with pytest.raises(ValueError, match="identity"):
        tail.build_k2_tail(4, 8, 1e-5, helpers=4)
    assert tail.tail_helpers_max(8) == 8 and tail.tail_helpers_max(16) == 4
    assert tail.tail_helpers_max(24) == 1 and tail.tail_helpers_max(32) == 0


def test_a_late_odd_tail_is_a_bounded_tool_build(monkeypatch):
    for name in ("POLL_TOGETHER", "RESIDENT"):
        monkeypatch.delenv(f"ATOM_KIMI_MONO_TAIL_{name}", raising=False)
    plain = tail.build_k2_tail(8, 8, 1e-5, identity_routes=True, poll_bound=9)
    late = tail.build_k2_tail(
        8, 8, 1e-5, identity_routes=True, poll_bound=9, poll_together=True,
        test_late_odd=300,
    )  # fmt: skip
    assert "_lo300" in late.name and "_pt" in late.name
    assert late.name.replace("_pt", "").replace("_lo300", "") == plain.name
    assert late.grid == plain.grid and late.abi is plain.abi
    with pytest.raises(ValueError, match="late-odd"):
        tail.build_k2_tail(8, 8, 1e-5, identity_routes=True, test_late_odd=300)
    with pytest.raises(ValueError, match="late-odd"):
        tail.build_k2_tail(
            8, 8, 1e-5, identity_routes=True, poll_bound=9, helpers=8,
            test_late_odd=300,
        )  # fmt: skip
