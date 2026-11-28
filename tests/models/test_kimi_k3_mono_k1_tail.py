# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""K1's KDA tail hand-offs and its GEMV weight staging. A chunk task convs the
whole head's q / k itself (no QK hand-off between a head's chunks), a task's
f_b rows and ReplaySSM records are issued with its other inputs, a head split
over parts writes its conv state after the norm's poll, and S20-shaped steps
keep the GEMV's weights two column groups ahead (``kda.gemv_staging``). Each
keeps every sum and its order; only the round trips between them move."""

from dataclasses import fields
from types import SimpleNamespace

import pytest

pytest.importorskip("flydsl")
pytest.importorskip("aiter")

from atom.models.kimi_k3.mono import runner
from atom.models.kimi_k3.mono.kernels.attention import kda as k1
from atom.mono.runtime.compile import _code_object, kernel_resources, trace


def test_the_tail_hand_offs_are_not_build_fields():
    names = {f.name for f in fields(k1.KdaPreBuild)}
    assert not names & {"own_qk", "early_in", "cs_late"}


def test_no_qk_hand_off_region():
    key = k1.KdaPreBuild(tokens=10, qlen=5, nblocks=3, delta=True, state_len=7)
    assert "qk" not in k1.scratch_layout(key)


@pytest.mark.parametrize("rows", [17, 18, 19, 20])
def test_w_ahead_at_two_groups_with_a_short_second(rows):
    assert k1.groups(rows) == 2 and rows - k1.TG <= k1.W_AHEAD_TAIL
    got = k1.gemv_staging(rows)
    assert got == {"gemv_unit": 1, "w_ahead": 2, "x_ahead": 3}
    assert k1.gemv_staging(rows, w_ahead=False) == {}


@pytest.mark.parametrize("rows", [5, 10, 16, 25, 30, 35, 60])
def test_w_ahead_nowhere_else(rows):
    assert k1.gemv_staging(rows) == k1.gemv_staging(rows, w_ahead=False)
    assert "w_ahead" not in k1.gemv_staging(rows)


def test_only_a_build_ahead_names_w_ahead():
    base = {"qlen": 5, "nblocks": 3, "delta": True, "state_len": 7}
    plain = k1.KdaPreBuild(tokens=10, **base)
    assert "_wa" not in k1.symbol(plain)
    ahead = k1.KdaPreBuild(tokens=20, **base, **k1.gemv_staging(20))
    assert "_wa2" in k1.symbol(ahead)


def test_a_spilled_build_ahead_falls_back_to_the_staging_it_replaced():
    base = {"qlen": 5, "nblocks": 3, "delta": True, "state_len": 7}
    ahead = k1.KdaPreBuild(tokens=20, **base, **k1.gemv_staging(20))
    lean = runner._lean_k1(ahead)
    assert lean == k1.KdaPreBuild(tokens=20, **base)
    assert runner._lean_k1(lean) == k1.KdaPreBuild(tokens=20, **base, gemv_unit=1)


def _runner():
    obj = object.__new__(runner.K3MonoDecodeRunner)
    obj.state_len = 7
    obj.state16 = False
    obj.replay_cap = 0
    seam = SimpleNamespace(eps=1e-5, out_eps=1e-5)
    kda = SimpleNamespace(
        blocks=3, write_idx=-1, seam=seam, onorm_eps=1e-5, lower_bound=-5.0
    )
    obj.plans = [SimpleNamespace(layer_id=7, mono=True, kda=kda)]
    return obj


@pytest.mark.parametrize("rows", [5, 20, 35])
def test_keys_take_the_staging(rows):
    key = _runner()._k1_keys(rows, 5)[7]
    staging = k1.gemv_staging(rows)
    assert {f: getattr(key, f) for f in staging} == staging


def _resources(launcher):
    (artifact,) = launcher._mem_cache.values()
    return kernel_resources(_code_object(artifact._ir_text))


@pytest.mark.parametrize("state16", [False, True])
@pytest.mark.parametrize("rows", [5, 10, 20, 30])
def test_compile_has_no_qk_hand_off(rows, state16, tmp_path, monkeypatch):
    """Chunk tasks (5, 10 rows) and head tasks over four and two parts (20,
    30; 20 with the weights two units ahead), ReplaySSM records issued with
    the inputs (replay_cap 16): no QK region, no spill."""
    assert not k1.head_tasks(10, 5) and k1.head_tasks(20, 5)
    assert k1.head_parts(20, 5) == 4 and k1.head_parts(30, 5) == 2
    monkeypatch.setenv("ARCH", "gfx950")
    monkeypatch.setenv("FLYDSL_RUNTIME_CACHE_DIR", str(tmp_path))
    key = k1.KdaPreBuild(
        tokens=rows,
        qlen=5,
        nblocks=3,
        delta=True,
        state_len=7,
        state16=state16,
        replay_cap=16,
        replay_window=5,
        **k1.gemv_staging(rows),
    )
    launcher = k1.build(key)
    rec = trace(launcher, k1.ABI)
    assert not rec.unnamed
    assert "qk" not in {a.region for a in rec.accesses}
    r = _resources(launcher)
    assert r["scratch"] == r["vgpr_spill"] == 0, r
    print({"rows": rows, "state16": state16, "vgpr": r["vgpr"], "lds": r["lds"]})
