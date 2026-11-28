# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""K2's hand-offs between its stages (``layer``'s notes above
``LATQ_FIRST_ROWS``): the latent gathered before the route is waited for up to
``LATQ_FIRST_ROWS`` rows, every pass's pairs in one poll, the router's K parts
added by the route (RLOGP), lnorm's sums of squares from the owner (LSQ). Each
keeps every sum and its order; only the round trips between them change."""

from dataclasses import fields

import pytest
import torch

pytest.importorskip("flydsl")
pytest.importorskip("aiter")

from flydsl.compiler.protocol import dsl_size_of

from atom.models.kimi_k3.mono import runner
from atom.models.kimi_k3.mono.config import MAX_ROWS, groups
from atom.models.kimi_k3.mono.kernels import layer as k2
from atom.models.kimi_k3.mono.kernels.stages import moe
from atom.mono.device import gemv
from atom.mono.plan.execution import THREADS
from atom.mono.plan.trace import Space
from atom.mono.runtime.compile import _code_object, kernel_resources, trace

GONE = {"latq_first", "batch_polls", "route_parts", "ln_sq"}


def test_the_hand_offs_are_not_build_fields():
    assert not {f.name for f in fields(k2.K2Build)} & GONE


def test_a_default_build_keeps_its_symbol():
    key = k2.K2Build(tokens=25, nblocks=3)
    assert k2.symbol(key) == "_ZN4atom27k3_mono_k2_s25_nb3_r1_g0_t0E"


def test_the_extra_peer_regions_sit_past_the_others():
    for s in range(1, MAX_ROWS + 1):
        lay = k2.peer_layout(s)
        assert set(moe.EXTRA_PEER) <= set(lay)
        end = max(o + n for r, (o, n) in lay.items() if r not in moe.EXTRA_PEER)
        assert all(lay[r][0] >= end for r in moe.EXTRA_PEER)
    off = runner.ix_peer_offset(MAX_ROWS)
    assert off >= max(k2.peer_bytes(s) for s in range(1, MAX_ROWS + 1))


@pytest.mark.parametrize("s", [1, 5, 20, 64])
def test_route_parts_reads_every_part_the_router_writes(s):
    """RLOGP: the pairs the router's parts write (``early_out``) are the ones
    the route reads (``route_logits``), each once, inside the region."""
    per, rows, parts, tp = moe.E // moe.TP, moe.ROWS, moe.R_PARTS, moe.TP
    wrote = [
        ((rank * parts + q) * s + t) * per + i * rows + r
        for rank in range(tp)
        for q in range(parts)
        for t in range(s)
        for i in range(moe.R_GROUPS)
        for r in range(rows)
    ]
    read = [
        ((e // per * parts + q) * s + t) * per + e % per
        for e in range(moe.E)
        for q in range(parts)
        for t in range(s)
    ]
    assert sorted(wrote) == sorted(read) == list(range(len(read)))
    assert len(read) == moe.peer_pairs(s, ("rlogp",))["rlogp"]


@pytest.mark.parametrize("s", range(1, MAX_ROWS + 1))
def test_the_latent_first_bitmaps_fit_the_stage_union(s):
    """``latq_first``: the bitmaps past the ug member's words (so the latent
    it gathered first survives the route table's build), and the member no
    larger than the early one: the stage union does not grow."""
    ug = dsl_size_of(moe.ug_lds(s))
    assert ug == 4 * moe.ug_words(s)
    mask = dsl_size_of(moe.mask_lds(s))
    assert dsl_size_of(moe.mask_lds(s, True)) == ug + mask
    sg = min(s, moe.TG)
    xcols = max(gemv.slab_width(moe.early_shape(k)[0], sg) for k in (0, 1, 2))
    assert ug + mask <= sg * (xcols + gemv.LDS_PAD) * 2


@pytest.mark.parametrize("s", range(1, MAX_ROWS + 1))
def test_batched_lnorm_waits_on_its_own_half(s):
    """``stage_lnorm_batched`` polls a pass past ``s * 64`` items at the last
    token: that token is in the same half (the last), so a half never waits
    for the other's LMSQ / LSQ."""
    n, halves = s * 64, moe.ln_halves(s)
    npass = moe.passes(n)
    for h in range(halves):
        ps = moe.ln_passes(npass, halves, h)
        tokens = {
            e // 64 for p in ps for e in range(p * THREADS, (p + 1) * THREADS) if e < n
        }
        for p in ps:
            if (p + 1) * THREADS > n:
                assert s - 1 in tokens


@pytest.mark.parametrize("s", range(1, MAX_ROWS + 1))
def test_batched_reduce_covers_reduce_rows(s):
    """``reduce_batched`` over the step: the (token, column) items of
    ``reduce_rows`` group by group, each once."""
    tg = moe.TG
    by_group = sorted(
        (g * tg + e // 64, e % 64)
        for g in range(groups(s))
        for e in range(min(tg, s - g * tg) * 64)
    )
    whole = sorted((e // 64, e % 64) for e in range(s * 64))
    assert by_group == whole
    assert moe.passes(s * 64) <= 8  # one poll batch, 16 VGPRs of pairs


def _resources(launcher):
    (artifact,) = launcher._mem_cache.values()
    return kernel_resources(_code_object(artifact._ir_text))


@pytest.mark.parametrize("rows", [5, 15, 20, 60])
def test_compile_takes_the_hand_offs(rows, tmp_path, monkeypatch):
    monkeypatch.setenv("ARCH", "gfx950")
    monkeypatch.setenv("FLYDSL_RUNTIME_CACHE_DIR", str(tmp_path))
    # on a GPU runner an earlier test may have initialized CUDA; the build
    # itself must not
    cuda_was_initialized = torch.cuda.is_initialized()
    launcher = k2.build(k2.K2Build(tokens=rows, nblocks=4))
    rec = trace(launcher, k2.ABI)
    assert not rec.unnamed
    access = {(a.stage, a.region, a.space, a.kind) for a in rec.accesses}
    r = _resources(launcher)
    assert r["scratch"] == r["vgpr_spill"] == 0, r
    assert r["lds"] <= 163840, r
    assert ("k2.early", "rlogp", Space.PEER, "put") in access
    assert ("k2.route", "rlogp", Space.PEER, "poll") in access
    assert ("k2.lnorm", "lsq", Space.PEER, "put") in access
    assert ("k2.lnorm", "lsq", Space.PEER, "poll") in access
    assert not {a[1] for a in access} & {"rpart", "rlog", "lmsq"}
    assert cuda_was_initialized or not torch.cuda.is_initialized()
    print({"rows": rows, "vgpr": r["vgpr"], "lds": r["lds"]})
