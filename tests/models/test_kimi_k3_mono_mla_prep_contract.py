# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""K2's MLA prep build (``ATOM_MONO_MLA_PREP``, P0), traced compile-only: no
GPU, but FlyDSL and ``ARCH`` (``compile.trace``).

The prep build hands off exactly what the default build does before its tail;
the tail (``k2.mlaprep``) reads layer M-1's output from FING, which ``k2.up``
filled, exchanges its own seam's pairs and publishes PXRDY for the next stage.
"""

import os

import pytest

pytest.importorskip("flydsl", reason="K2 is a FlyDSL kernel")
if not os.environ.get("ARCH"):
    pytest.skip("compile-only tracing needs ARCH", allow_module_level=True)

from atom.models.kimi_k3.mono.kernels import layer as k2
from atom.models.kimi_k3.mono.kernels.stages import mla_prep as pr
from atom.mono.plan.trace import Space
from atom.mono.runtime.compile import trace


@pytest.fixture(autouse=True)
def _no_disk_cache(monkeypatch):
    # a build served from FlyDSL's disk cache, or one this process already
    # built (``layer._BUILDS``), is not traced
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    monkeypatch.setattr(k2, "_BUILDS", {})


def test_prep_build_adds_only_its_tail():
    base_key = k2.K2Build(tokens=1, nblocks=2)
    key = k2.K2Build(tokens=1, nblocks=2, mla_prep=1, prep_nblocks=2)
    base = trace(k2.build(base_key), k2.abi_for(base_key))
    rec = trace(k2.build(key), k2.abi_for(key))
    assert not rec.unnamed
    assert rec.stages[-1] == "k2.mlaprep" and "k2.mlaprep" not in base.stages
    before = [(a.region, a.kind) for a in rec.accesses if a.stage != "k2.mlaprep"]
    assert before == [(a.region, a.kind) for a in base.accesses]
    tail = [a for a in rec.accesses if a.stage == "k2.mlaprep"]
    assert {a.region for a in tail} == set(pr.REGIONS) | {"fing"}
    for a in tail:
        if a.region == "fing":
            assert a.kind == "poll" and a.space is Space.PEER
        else:
            assert a.space is Space.SCRATCH
    # FING is put by the up stage, before the tail polls it
    assert {
        a.stage for a in base.accesses if a.region == "fing" and a.kind == "put"
    } == {"k2.up"}
    # PXRDY is the next stage's (P1's): put here, polled by nobody yet
    assert [a.kind for a in tail if a.region == "pxrdy"] == ["put"]
    # every other region of the tail is put before it is polled
    for r in ("ppart", "pwgt", "pmsq", "pamx"):
        kinds = [a.kind for a in tail if a.region == r]
        assert kinds[0] == "put" and "poll" in kinds, (r, kinds)


def test_prep_symbol_and_abi():
    key = k2.K2Build(tokens=1, nblocks=2, mla_prep=1, prep_nblocks=2)
    assert k2.abi_for(key).names[-len(pr.ARGS) :] == pr.ARGS
    assert k2.abi_for(k2.K2Build(tokens=1, nblocks=2)) is k2.ABI
    with pytest.raises(AssertionError):
        k2.build(k2.K2Build(tokens=17, nblocks=2, mla_prep=1, prep_nblocks=2))
    with pytest.raises(AssertionError):
        k2.build(k2.K2Build(tokens=1, nblocks=2, mla_prep=1, prep_nblocks=3))
