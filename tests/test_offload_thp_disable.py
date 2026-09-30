# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""``_disable_thp_for_pinned_alloc``: THP opt-out before LMCache's pinned pool.

``ctypes.CDLL`` is mocked so no real ``prctl`` reaches the test process.
"""

from __future__ import annotations

import ctypes
from types import SimpleNamespace

import pytest

from atom.kv_transfer.offload import _offload_common as oc


class _FakeLibc:
    def __init__(self, rc=0, exc=None):
        self.rc = rc
        self.exc = exc
        self.calls = []

    def prctl(self, *args):
        self.calls.append(args)
        if self.exc is not None:
            raise self.exc
        return self.rc


@pytest.fixture
def fake_libc(monkeypatch):
    monkeypatch.delenv("ATOM_PD_HOST_LANDING_BLOCKS", raising=False)
    holder = {"libc": _FakeLibc(), "names": []}

    def _cdll(name, *args, **kwargs):
        holder["names"].append(name)
        return holder["libc"]

    monkeypatch.setattr(ctypes, "CDLL", _cdll)
    return holder


def test_prctl_thp_disable_is_attempted(fake_libc):
    oc._disable_thp_for_pinned_alloc(SimpleNamespace(numa_mode=None))
    assert fake_libc["names"] == ["libc.so.6"]
    # PR_SET_THP_DISABLE == 41, arg2 == 1 turns it on.
    assert fake_libc["libc"].calls == [(41, 1, 0, 0, 0)]


def test_missing_numa_mode_attr_still_disables(fake_libc):
    oc._disable_thp_for_pinned_alloc(SimpleNamespace())
    assert fake_libc["libc"].calls == [(41, 1, 0, 0, 0)]


@pytest.mark.parametrize(
    "libc",
    [_FakeLibc(rc=-1), _FakeLibc(exc=AttributeError("no prctl"))],
    ids=["nonzero-rc", "raises"],
)
def test_prctl_failure_is_swallowed(fake_libc, libc):
    fake_libc["libc"] = libc
    oc._disable_thp_for_pinned_alloc(SimpleNamespace(numa_mode=None))
    assert libc.calls == [(41, 1, 0, 0, 0)]


def test_cdll_load_failure_is_swallowed(monkeypatch):
    monkeypatch.delenv("ATOM_PD_HOST_LANDING_BLOCKS", raising=False)

    def _cdll(*args, **kwargs):
        raise OSError("libc.so.6: cannot open shared object file")

    monkeypatch.setattr(ctypes, "CDLL", _cdll)
    oc._disable_thp_for_pinned_alloc(SimpleNamespace(numa_mode=None))


def test_skipped_when_lmcache_numa_mode_set(fake_libc):
    # LMCache's NUMA path mmaps + mbinds + hipHostRegisters itself: no
    # hipHostMalloc, no MADV_HUGEPAGE, nothing to opt out of.
    oc._disable_thp_for_pinned_alloc(SimpleNamespace(numa_mode="auto"))
    assert fake_libc["names"] == []
    assert fake_libc["libc"].calls == []


def test_skipped_when_host_landing_pool_configured(fake_libc, monkeypatch):
    # The process-wide flag would also void the host landing pool's
    # MADV_HUGEPAGE, which it needs to stay under the NIC's 4 KiB-page
    # RDMA registration cap.
    monkeypatch.setenv("ATOM_PD_HOST_LANDING_BLOCKS", "1024")
    oc._disable_thp_for_pinned_alloc(SimpleNamespace(numa_mode=None))
    assert fake_libc["libc"].calls == []


@pytest.mark.parametrize("value", ["0", "", "garbage"])
def test_host_landing_off_values_still_disable(fake_libc, monkeypatch, value):
    monkeypatch.setenv("ATOM_PD_HOST_LANDING_BLOCKS", value)
    oc._disable_thp_for_pinned_alloc(SimpleNamespace(numa_mode=None))
    assert fake_libc["libc"].calls == [(41, 1, 0, 0, 0)]
