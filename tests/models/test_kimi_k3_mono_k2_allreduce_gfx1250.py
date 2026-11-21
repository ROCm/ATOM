# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only geometry of the gfx1250 in-kernel K2 all-reduce."""

import pytest

from atom.models.kimi_k3.mono.config import HIDDEN, LATENT

pytest.importorskip("flydsl", reason="the K3 gfx1250 mono kernels are FlyDSL")

from atom.models.kimi_k3.mono.kernels.stages import k2_allreduce_gfx1250 as ar

WIDE = (50, 55, 60, 64)


def _rows(*rows):
    """``rows``, those past the token cap skipped until it is raised."""
    from atom.models.kimi_k3.mono.kernels.token_tiling import GFX1250_MAX_TOKENS

    return [
        pytest.param(
            r,
            marks=pytest.mark.skipif(r > GFX1250_MAX_TOKENS, reason="token cap"),
        )
        for r in rows
    ]


@pytest.mark.parametrize("rows", (1, 5, 16, 40, 48, *WIDE))
def test_lines_cover_the_routed_and_shared_partials(rows):
    lines = ar.allreduce_lines(rows)
    assert lines * ar.LINE_ELEMENTS == rows * (LATENT + HIDDEN)
    ctas = ar.allreduce_ctas(rows)
    assert 1 <= ctas <= ar.MAX_CTAS
    # one line a thread, so no thread loads after its own remote stores
    assert ctas * ar.allreduce_threads(rows) >= lines
    # every epoch counter is in some thread's reach
    assert ctas * ar.allreduce_threads(rows) >= ar.MAX_CTAS


@pytest.mark.parametrize("rows", range(1, 81))
def test_six_element_gathers_never_load_past_a_routed_row(rows):
    # _gathered_words' load window: from min(column, ROUTED_WORDS - 3), the
    # words shifted back; past the last route row's end may be unmapped
    words = ar.PAYLOAD_WORDS
    for line in range(-(-rows * LATENT // ar.LINE_ELEMENTS)):
        column = line * words % ar.ROUTED_WORDS
        start = min(column, ar.ROUTED_WORDS - words)
        shift = column - start
        assert 0 <= start and start + words <= ar.ROUTED_WORDS
        for w in range(min(words, ar.ROUTED_WORDS - column)):
            index = min(w + shift, words - 1)
            assert index == w + shift and start + index == column + w


def test_peer_banks_hold_two_epochs_of_every_source():
    assert ar.allreduce_peer_bytes(8, 40) == 2 * 8 * ar.allreduce_lines(40) * 16
    assert ar.allreduce_peer_bytes(8, 48) > ar.allreduce_peer_bytes(8, 40)


def test_o_proj_all_reduce_gives_each_rank_whole_ctas_of_columns():
    pytest.importorskip("aiter", reason="the kernel under test needs aiter")
    from atom.models.kimi_k3.mono.kernels.stages import k2_oproj_ar_gfx1250 as oar

    assert oar.CTAS * oar.CTA_COLUMNS == HIDDEN
    assert oar.CTAS <= ar.MAX_CTAS
    for tp in (2, 4, 8):
        assert HIDDEN % (tp * oar.CTA_COLUMNS) == 0
        assert oar.own_pairs(tp) * 2 * tp == HIDDEN
    # partials from every rank for its own columns, then every column's sum, a
    # line a column pair of three tokens (40 rows: 14 triples, the last padded)
    assert oar.triples(40) == 14
    assert oar.bank_words(8, 40) == 4 * 14 * (8 * oar.own_pairs(8) + HIDDEN // 2)
    assert oar.oproj_peer_bytes(8, 48, oneshot=False) == 2 * oar.bank_words(8, 48) * 4
    # one hop: every rank's partials for every column
    assert oar.oneshot_bank_words(8, 48) == 4 * 16 * 8 * (HIDDEN // 2)
    assert oar.oproj_peer_bytes(8, 48, oneshot=True) == (
        2 * oar.oneshot_bank_words(8, 48) * 4
    )


@pytest.mark.parametrize("rows", (1, 5, 16, 40, 48, *WIDE))
def test_gathering_all_reduce_takes_a_line_a_thread(rows):
    kernel = ar.build_gather_allreduce(8, rows)
    assert kernel.abi is ar.GATHER_ALLREDUCE_ABI
    lines = ar.allreduce7_lines(rows)
    if lines <= ar.MAX_CTAS * ar.THREADS:
        # seven-element lines; a CTA's routed lines all of one token
        assert lines * ar.LINE7_ELEMENTS == rows * (LATENT + HIDDEN)
        assert kernel.block == (ar.THREADS,)
        assert kernel.grid == (-(-lines // ar.THREADS),)
        assert ar.LINE7_ROUTED % ar.THREADS == 0
    else:
        assert kernel.grid == ar.build_allreduce(8, rows).grid
        # a CTA's lines (one a thread) cover fewer words than a routed row
        assert kernel.block[0] * ar.PAYLOAD_WORDS <= ar.ROUTED_WORDS
    # every epoch counter is in some thread's reach
    assert kernel.grid[0] * kernel.block[0] >= ar.MAX_CTAS


def test_one_shot_o_proj_banks_hold_every_ranks_partials_of_every_column():
    pytest.importorskip("aiter", reason="the kernel under test needs aiter")
    from atom.models.kimi_k3.mono.kernels.stages import k2_oproj_ar_gfx1250 as oar

    for tp in (2, 4, 8):
        words = oar.oneshot_bank_words(tp, 40)
        assert words == 4 * 14 * tp * (HIDDEN // 2)
        both = oar.oproj_peer_bytes(tp, 40, True)
        assert both == 2 * 4 * max(words, oar.bank_words(tp, 40))
        assert oar.oproj_peer_bytes(tp, 40, False) == 2 * 4 * oar.bank_words(tp, 40)


@pytest.mark.parametrize("rows", (10, 40))
def test_gathering_all_reduce_profile_and_barrier_builds(rows):
    profile = ar.build_gather_allreduce(4, rows, profile=True)
    assert profile.abi is ar.GATHER_ALLREDUCE_PROFILE_ABI
    assert profile.abi.names[-1] == "stamps"
    assert profile.name.endswith("_profile")
    plain = ar.build_gather_allreduce(4, rows, prepush_barrier=True)
    assert plain.abi is ar.GATHER_ALLREDUCE_ABI
    assert isinstance(plain.launcher, ar.TailedLauncher)
    bare = ar.build_gather_allreduce(4, rows, prepush_barrier=False)
    assert bare.name == plain.name + "_nb"
    assert bare.grid == plain.grid


def test_o_proj_all_reduce_takes_a_line_a_thread_a_round():
    pytest.importorskip("aiter", reason="the kernel under test needs aiter")
    from atom.models.kimi_k3.mono.kernels.stages import k2_oproj_ar_gfx1250 as oar

    assert all(oar.line_rounds(rows) == 1 for rows in range(1, 49))
    assert all(oar.line_rounds(rows) == 2 for rows in range(49, 65))
    assert oar.triples(64) * oar.CTA_PAIRS <= 2 * oar.THREADS


@pytest.mark.parametrize("rows", _rows(40, 48, *WIDE))
@pytest.mark.parametrize("oneshot", (False, True))
def test_o_proj_all_reduce_builds_to_64_rows(rows, oneshot):
    from atom.models.kimi_k3.mono.kernels.execution_plan import Gfx1250KimiBuildPlan

    pytest.importorskip("aiter", reason="the kernel under test needs aiter")
    from atom.models.kimi_k3.mono.kernels.stages import k2_oproj_ar_gfx1250 as oar

    plan = Gfx1250KimiBuildPlan(
        threads=256, blocks=256, lds_bytes=320 * 1024, tokens=rows, nblocks=1
    )
    kernel = oar.build_oproj_allreduce(plan, 8, 1536, oneshot=oneshot)
    assert kernel.grid == (oar.CTAS,) and kernel.block == (oar.THREADS,)
    assert f"_s{rows}" in kernel.name


def _oproj_plan(rows=8):
    from atom.models.kimi_k3.mono.kernels.execution_plan import Gfx1250KimiBuildPlan

    return Gfx1250KimiBuildPlan(
        threads=256, blocks=256, lds_bytes=320 * 1024, tokens=rows, nblocks=1
    )


def test_poll_together_switches_default_off_and_take_more_than_one_rank(monkeypatch):
    pytest.importorskip("aiter", reason="the kernel under test needs aiter")
    from atom.models.kimi_k3.mono.kernels.stages import k2_allreduce_gfx1250 as ar
    from atom.models.kimi_k3.mono.kernels.stages import k2_oproj_ar_gfx1250 as oar
    from atom.utils import envs

    for name in ("OPROJ_AR", "GATHER_AR"):
        monkeypatch.delenv(f"ATOM_KIMI_MONO_{name}_POLL_TOGETHER", raising=False)
        assert getattr(envs, f"ATOM_KIMI_MONO_{name}_POLL_TOGETHER") is False
    plan = _oproj_plan()
    oprojs = {
        hop: oar.build_oproj_allreduce(plan, 8, 1536, oneshot=hop).name
        for hop in (False, True)
    }
    gathers = {rows: ar.build_gather_allreduce(8, rows).name for rows in (8, 48)}
    assert not any("_pt" in name for name in (*oprojs.values(), *gathers.values()))
    monkeypatch.setenv("ATOM_KIMI_MONO_OPROJ_AR_POLL_TOGETHER", "1")
    # each switch its own kernel's
    assert ar.build_gather_allreduce(8, 8).name == gathers[8]
    monkeypatch.setenv("ATOM_KIMI_MONO_GATHER_AR_POLL_TOGETHER", "1")
    for hop, name in oprojs.items():
        assert (
            oar.build_oproj_allreduce(plan, 8, 1536, oneshot=hop).name == name + "_pt"
        )
        assert "_pt" not in oar.build_oproj_allreduce(plan, 1, 1536, oneshot=hop).name
        off = oar.build_oproj_allreduce(plan, 8, 1536, oneshot=hop, poll_together=False)
        assert off.name == name and off.abi is oar.OPROJ_AR_ABI
    for rows, name in gathers.items():
        assert ar.build_gather_allreduce(8, rows).name == name + "_pt"
        assert "_pt" not in ar.build_gather_allreduce(1, rows).name
        assert ar.build_gather_allreduce(8, rows, poll_together=False).name == name


def test_poll_helpers_default_to_volatile_loads():
    pytest.importorskip("aiter", reason="the kernel under test needs aiter")
    from atom.models.kimi_k3.mono.kernels.stages import k2_allreduce_gfx1250 as ar

    assert ar.POLL_VOLATILE == ar.SCOPE_SYS | ar.VOLATILE
    assert ar.POLL_TOGETHER == ar.SCOPE_SYS
    # no context: the helpers are called as ever
    assert ar.poll_context(False, None, None, 0, 0) is None
    assert ar.poll_context(True, None, None, 0, 0)["modifier"] == ar.POLL_TOGETHER
    assert ar.poll_context(False, 7, None, 0, 0)["modifier"] == ar.POLL_VOLATILE
    # a tool build's marks sit past the epoch counters
    assert ar.POLLS_WORD >= ar.MAX_CTAS
    assert ar.GAVE_UP_WORD >= ar.POLLS_WORD + ar.MAX_CTAS


def test_tool_builds_are_named_apart_and_dry_pushes_take_a_bound(monkeypatch):
    pytest.importorskip("aiter", reason="the kernel under test needs aiter")
    from atom.models.kimi_k3.mono.kernels.stages import k2_allreduce_gfx1250 as ar
    from atom.models.kimi_k3.mono.kernels.stages import k2_oproj_ar_gfx1250 as oar

    for name in ("OPROJ_AR", "GATHER_AR"):
        monkeypatch.delenv(f"ATOM_KIMI_MONO_{name}_POLL_TOGETHER", raising=False)
    plan = _oproj_plan()
    plain = oar.build_oproj_allreduce(plan, 8, 1536, oneshot=True).name
    build = lambda **kw: oar.build_oproj_allreduce(plan, 8, 1536, oneshot=True, **kw)
    assert build(poll_bound=9).name == plain + "_b9"
    assert build(poll_bound=9, test_skip_push=True).name == plain + "_b9_np"
    assert build(poll_together=True, test_swap_late=2).name == plain + "_pt_sw2"
    with pytest.raises(ValueError, match="bound"):
        build(test_skip_push=True)
    with pytest.raises(ValueError, match="one-shot"):
        oar.build_oproj_allreduce(plan, 8, 1536, oneshot=False, test_swap_late=0)
    for rows in (8, 48):
        name = ar.build_gather_allreduce(8, rows).name
        assert ar.build_gather_allreduce(8, rows, poll_bound=9).name == name + "_b9"
        late = ar.build_gather_allreduce(8, rows, poll_together=True, test_swap_late=3)
        assert late.name == name + "_pt_sw3"
        with pytest.raises(ValueError, match="bound"):
            ar.build_gather_allreduce(8, rows, test_skip_push=True)
