"""Page-locked runs are `mlock`ed while registered: `_PageRuns.acquire`
locks exactly the range it registers, in one call, `release` unlocks it, and
a refused lock only warns."""

import ctypes
import errno
import logging
import mmap
import resource
import sys

import numpy as np
import pytest

from atom.model_ops.engram import host_memory

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="mlock / VmLck are Linux-only"
)

PAGE = host_memory._PAGE


class Cudart:
    """Accepts every registration, or refuses the `refuse_call`-th."""

    def __init__(self, refuse_call=None):
        self.refuse_call = refuse_call
        self.calls = 0
        self.registered = {}

    def cudaHostRegister(self, ptr, size, flags):
        self.calls += 1
        if self.calls == self.refuse_call:
            return 1
        self.registered[ptr] = ptr + size
        return 0

    def cudaHostUnregister(self, ptr):
        del self.registered[ptr]
        return 0


class Libc:
    """Records `mlock2` / `munlock` spans; `mlock2` fails with `lock_errno`."""

    def __init__(self, lock_errno=0):
        self.lock_errno = lock_errno
        self.locked, self.unlocked = [], []

    def mlock2(self, addr, n, flags):
        assert flags == host_memory._MLOCK_ONFAULT
        self.locked.append((addr, addr + n))
        if self.lock_errno:
            ctypes.set_errno(self.lock_errno)
            return -1
        return 0

    def munlock(self, addr, n):
        self.unlocked.append((addr, addr + n))
        return 0


@pytest.fixture
def fake_libc(monkeypatch):
    libc = Libc()
    monkeypatch.setattr(host_memory, "_libc", lambda: libc)
    monkeypatch.setattr(host_memory, "_UNLOCKED_WARNED", [])
    # registration through the scripted cudart, not ROCr's lock
    monkeypatch.setattr(host_memory, "_device_lock", lambda: None)
    return libc


def test_acquire_locks_and_registers_the_range_in_one_call(fake_libc):
    runs, rt = host_memory._PageRuns(), Cudart()
    first = runs.acquire(rt, 0, 8 * PAGE)
    second = runs.acquire(rt, 16 * PAGE, 18 * PAGE)
    assert fake_libc.locked == [(0, 8 * PAGE), (16 * PAGE, 18 * PAGE)]
    assert rt.registered == {0: 8 * PAGE, 16 * PAGE: 18 * PAGE}
    for run in (first, second):
        runs.release(rt, run)
    assert sorted(fake_libc.unlocked) == [(0, 8 * PAGE), (16 * PAGE, 18 * PAGE)]
    assert rt.registered == {} and len(runs) == 0


def test_refused_registration_leaves_nothing_locked(fake_libc):
    runs, rt = host_memory._PageRuns(), Cudart(refuse_call=1)
    assert runs.acquire(rt, 2 * PAGE, 10 * PAGE) is None
    assert fake_libc.unlocked == fake_libc.locked == [(2 * PAGE, 10 * PAGE)]
    assert rt.registered == {} and len(runs) == 0


@pytest.mark.parametrize("err, hint", [(errno.EPERM, True), (errno.EAGAIN, False)])
def test_refused_lock_registers_and_warns_once(
    fake_libc, monkeypatch, caplog, err, hint
):
    fake_libc.lock_errno = err
    monkeypatch.setattr(
        host_memory.resource, "getrlimit", lambda what: (64 << 10, 64 << 10)
    )
    runs, rt = host_memory._PageRuns(), Cudart()
    with caplog.at_level(logging.WARNING):
        held = [runs.acquire(rt, 0, 4 * PAGE)]
        held.append(runs.acquire(rt, 4 * PAGE, 8 * PAGE))
    # Still registered where the lock was refused.
    assert set(rt.registered) == {0, 4 * PAGE}
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "GiB" in warnings[0] and host_memory.os.strerror(err) in warnings[0]
    assert ("RLIMIT_MEMLOCK" in warnings[0]) == hint
    for run in held:
        runs.release(rt, run)


def _vm_lck_kb() -> int:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmLck:"):
                return int(line.split()[1])
    raise AssertionError("no VmLck in /proc/self/status")


def _filled_pages(pages: int) -> np.ndarray:
    """A shared mapping of `pages` pages, every one written: resident, as the
    shared copy's are once its ranks filled them."""
    table = np.frombuffer(
        mmap.mmap(-1, pages * mmap.PAGESIZE, flags=mmap.MAP_SHARED), dtype=np.uint8
    )
    table[:] = 1
    return table


def test_a_real_mapping_is_locked_then_unlocked(monkeypatch):
    """Real `mlock` on resident shared pages."""
    monkeypatch.setattr(host_memory, "_UNLOCKED_WARNED", [])
    monkeypatch.setattr(host_memory, "_device_lock", lambda: None)
    table = _filled_pages(4)
    lo = table.ctypes.data
    hi = lo + table.nbytes
    soft, _ = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    if soft != resource.RLIM_INFINITY and soft < 2 * table.nbytes:
        pytest.skip("RLIMIT_MEMLOCK too small for 4 pages")
    runs, rt = host_memory._PageRuns(), Cudart()
    before = _vm_lck_kb()
    held = runs.acquire(rt, lo, hi)
    try:
        assert _vm_lck_kb() - before == table.nbytes // 1024
    finally:
        runs.release(rt, held)
    assert _vm_lck_kb() == before


def test_a_real_refusal_only_warns(monkeypatch, caplog):
    monkeypatch.setattr(host_memory, "_UNLOCKED_WARNED", [])
    monkeypatch.setattr(host_memory, "_device_lock", lambda: None)
    table = _filled_pages(1)
    lo = table.ctypes.data
    hi = lo + table.nbytes
    soft, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    try:
        resource.setrlimit(resource.RLIMIT_MEMLOCK, (0, hard))
    except (ValueError, OSError):
        pytest.skip("cannot lower RLIMIT_MEMLOCK")
    runs, rt = host_memory._PageRuns(), Cudart()
    try:
        libc = host_memory._libc()
        if libc.mlock2(lo, hi - lo, 0) == 0:
            libc.munlock(lo, hi - lo)
            pytest.skip("mlock allowed despite RLIMIT_MEMLOCK=0 (CAP_IPC_LOCK)")
        before = _vm_lck_kb()
        with caplog.at_level(logging.WARNING):
            held = runs.acquire(rt, lo, hi)
        runs.release(rt, held)
    finally:
        resource.setrlimit(resource.RLIMIT_MEMLOCK, (soft, hard))
    assert held is not None and _vm_lck_kb() == before
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1 and "mlock" in warnings[0].getMessage()
