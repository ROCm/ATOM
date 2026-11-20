# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""K2's MLA post build (``ATOM_MONO_MLA_POST``), traced compile-only: no GPU,
but FlyDSL and ``ARCH`` (``compile.trace``).

The post build must hand off exactly what the default build does, plus its
``mpdone`` pairs: one stage on every CTA that publishes a task's pair before it
polls all of them, as an all-reduce pushes before it waits.
"""

import os

import pytest

pytest.importorskip("flydsl", reason="K2 is a FlyDSL kernel")
if not os.environ.get("ARCH"):
    pytest.skip("compile-only tracing needs ARCH", allow_module_level=True)

from atom.models.kimi_k3.mono.kernels import layer as k2
from atom.mono.plan.trace import Space
from atom.mono.runtime.compile import trace


@pytest.fixture(autouse=True)
def _no_disk_cache(monkeypatch):
    # a build served from FlyDSL's disk cache, or one this process already
    # built (``layer._BUILDS``), is not traced
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    monkeypatch.setattr(k2, "_BUILDS", {})


def _accesses(rec, keep):
    return [(a.region, a.space, a.kind) for a in rec.accesses if keep(a.region)]


def test_post_build_adds_only_its_exchange():
    base_key = k2.K2Build(tokens=1, nblocks=2)
    post_key = k2.K2Build(tokens=1, nblocks=2, mla_post=True)
    base = trace(k2.build(base_key), k2.abi_for(base_key))
    post = trace(k2.build(post_key), k2.abi_for(post_key))
    assert not base.unnamed and not post.unnamed
    others = _accesses(post, lambda r: r != "mpdone")
    assert others == _accesses(base, lambda r: True)
    mp = [a for a in post.accesses if a.region == "mpdone"]
    assert mp and all(a.stage == "k2.mlapost" for a in mp)
    assert all(a.space is Space.SCRATCH for a in mp)
    # each o_proj call site: its task's put, then the poll of every pair
    assert [a.kind for a in mp] == ["put", "poll"] * (len(mp) // 2)
    assert "k2.mlapost" not in base.stages
    # o_proj still pushes its partials (``ar``) from ``k2.o``
    puts = {a.stage for a in post.accesses if a.region == "ar" and a.kind == "put"}
    assert puts == {"k2.o"}
