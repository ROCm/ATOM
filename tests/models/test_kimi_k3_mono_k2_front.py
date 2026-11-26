# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""K2's front staging: past one token group the router's and latent's x staged
``layer.EARLY_AHEAD`` units ahead, two latent passes a poll, and where
``layer.yield_pays`` the shared gate/up's x staged after theirs (ESTG). Each
keeps every sum and its order; only when loads and polls are issued
changes."""

from dataclasses import fields

import pytest
import torch

pytest.importorskip("flydsl")
pytest.importorskip("aiter")

from atom.models.kimi_k3.mono.config import MAX_ROWS, groups
from atom.models.kimi_k3.mono.kernels import layer as k2
from atom.models.kimi_k3.mono.kernels.stages import moe
from atom.mono.device import gemv
from atom.mono.device.sync import POLL_MAX
from atom.mono.plan.execution import THREADS
from atom.mono.plan.trace import Space
from atom.mono.runtime.compile import _code_object, kernel_resources, trace


def test_the_front_is_not_build_fields():
    names = {f.name for f in fields(k2.K2Build)}
    assert not names & {"early_ahead", "latq_batch", "early_yield"}


@pytest.mark.parametrize("rows", [1, 4, 5, 9, 10, 15, 16, 17, 20, 35, 60, 64])
def test_the_yield_pays_within_one_group_of_two_slabs(rows):
    assert k2.yield_pays(rows) == (10 <= rows <= 16)
    assert k2.yield_pays(rows) == (
        groups(rows) == 1 and len(gemv.slabs(moe.HIDDEN, rows)) > 1
    )


@pytest.mark.parametrize("s", range(1, MAX_ROWS + 1))
def test_the_yield_flags_sit_past_every_region(s):
    """ESTG is laid out last: every other scratch region keeps its offset, and
    it holds a flag per router and latent task."""
    lay = moe.scratch_layout(moe.MoeBuild(tokens=s))
    names = list(lay)
    assert names[-1] == "estg"
    end = max(o + n for name, (o, n) in lay.items() if name != "estg")
    assert lay["estg"][0] >= end
    assert lay["estg"][1] == (moe.R_TASKS + moe.L_TASKS) * 8
    assert moe.R_TASKS + moe.L_TASKS <= THREADS
    assert "estg" in moe.REGIONS


@pytest.mark.parametrize("s", range(1, MAX_ROWS + 1))
def test_batched_latent_gather_covers_every_block_once(s):
    """``load_latq_batched``: each (token, 32-block) by the thread and pass the
    unbatched gather gives it, a poll's pairs within ``POLL_MAX``."""
    nblk, npass = s * (moe.LAT // 32), moe.latq_passes(s)
    assert 5 * moe.LATQ_BATCH <= POLL_MAX
    seen = []
    for j in range(-(-npass // moe.LATQ_BATCH)):
        n = min(moe.LATQ_BATCH, npass - j * moe.LATQ_BATCH)
        for p in range(n):
            i = j * moe.LATQ_BATCH + p
            seen += [
                tid + THREADS * i for tid in range(THREADS) if tid + THREADS * i < nblk
            ]
    assert seen == list(range(nblk))


def _resources(launcher):
    (artifact,) = launcher._mem_cache.values()
    return kernel_resources(_code_object(artifact._ir_text))


@pytest.mark.parametrize("rows", [5, 10, 20, 60])
def test_compile_takes_the_front(rows, tmp_path, monkeypatch):
    """ESTG only where ``layer.yield_pays``."""
    monkeypatch.setenv("ARCH", "gfx950")
    monkeypatch.setenv("FLYDSL_RUNTIME_CACHE_DIR", str(tmp_path))
    cuda_was_initialized = torch.cuda.is_initialized()
    launcher = k2.build(k2.K2Build(tokens=rows, nblocks=4))
    rec = trace(launcher, k2.ABI)
    assert not rec.unnamed
    access = {(a.stage, a.region, a.space, a.kind) for a in rec.accesses}
    r = _resources(launcher)
    assert r["scratch"] == r["vgpr_spill"] == 0, r
    assert r["lds"] <= 163840, r
    estg = {
        ("k2.early", "estg", Space.SCRATCH, "put"),
        ("k2.early", "estg", Space.SCRATCH, "poll"),
    }
    if k2.yield_pays(rows):
        assert estg <= access
    else:
        assert not {a for a in access if a[1] == "estg"}
    assert cuda_was_initialized or not torch.cuda.is_initialized()
    print({"rows": rows, "vgpr": r["vgpr"], "lds": r["lds"]})
