"""In-place page runs are `mlock`ed while registered: `_PageRuns.acquire`
locks exactly the runs it creates, `release` unlocks a run with its last
reference, and a refused lock only warns."""

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
    """Records `mlock2` / `munlock` / populate spans; `mlock2` fails with
    `lock_errno`."""

    def __init__(self, lock_errno=0):
        self.lock_errno = lock_errno
        self.locked, self.unlocked, self.populated = [], [], []

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

    def madvise(self, addr, n, advice):
        self.populated.append((addr, addr + n))
        return 0


def _merged(spans):
    out = []
    for lo, hi in sorted(spans):
        if out and lo <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], hi))
        else:
            out.append((lo, hi))
    return out


@pytest.fixture
def fake_libc(monkeypatch):
    libc = Libc()
    monkeypatch.setattr(host_memory, "_libc", lambda: libc)
    monkeypatch.setattr(host_memory, "_UNLOCKED_WARNED", [])
    # Several chunks on several threads, as for a real shard.
    monkeypatch.setattr(host_memory, "_CHUNK_BYTES", 2 * PAGE)
    monkeypatch.setattr(host_memory, "_BYTES_PER_THREAD", 2 * PAGE)
    monkeypatch.setattr(host_memory, "_THREADS_FROM", 1)
    return libc


def test_acquire_locks_exactly_the_new_runs(fake_libc):
    runs, rt = host_memory._PageRuns(), Cudart()
    ro = host_memory._HOST_REGISTER_READ_ONLY
    first = runs.acquire(rt, 0, 8 * PAGE, ro)
    assert fake_libc.locked == [(0, 8 * PAGE)]
    # Faulted in, in several chunks (on several threads).
    assert len(fake_libc.populated) > 1
    assert _merged(fake_libc.populated) == [(0, 8 * PAGE)]
    fake_libc.locked.clear()
    fake_libc.populated.clear()
    # Overlaps the first run: only the uncovered tail is locked.
    second = runs.acquire(rt, 4 * PAGE, 12 * PAGE, ro)
    assert fake_libc.locked == [(8 * PAGE, 12 * PAGE)]
    assert _merged(fake_libc.populated) == [(8 * PAGE, 12 * PAGE)]
    # A writable range is locked, and left for registration to fault in.
    fake_libc.populated.clear()
    third = runs.acquire(rt, 16 * PAGE, 18 * PAGE)
    assert fake_libc.locked[-1] == (16 * PAGE, 18 * PAGE)
    assert fake_libc.populated == []
    for held in (first, second, third):
        runs.release(rt, held)
    assert sorted(fake_libc.unlocked) == [
        (0, 8 * PAGE),
        (8 * PAGE, 12 * PAGE),
        (16 * PAGE, 18 * PAGE),
    ]


def test_release_unlocks_a_run_with_its_last_reference(fake_libc):
    runs, rt = host_memory._PageRuns(), Cudart()
    held = [runs.acquire(rt, 0, 4 * PAGE) for _ in range(2)]
    runs.release(rt, held[0])
    assert fake_libc.unlocked == [] and rt.registered
    runs.release(rt, held[1])
    assert fake_libc.unlocked == [(0, 4 * PAGE)] and rt.registered == {}


def test_refused_registration_leaves_nothing_locked(fake_libc):
    """A refused gap is unlocked, and so is every run the same acquire created
    before it; runs other ranges still hold stay locked."""
    runs, rt = host_memory._PageRuns(), Cudart()
    other = runs.acquire(rt, 4 * PAGE, 6 * PAGE)
    rt.refuse_call = rt.calls + 2  # [2, 4) registers, [6, 10) is refused
    assert runs.acquire(rt, 2 * PAGE, 10 * PAGE) is None
    assert sorted(fake_libc.unlocked) == [(2 * PAGE, 4 * PAGE), (6 * PAGE, 10 * PAGE)]
    assert rt.registered == {4 * PAGE: 6 * PAGE}
    runs.release(rt, other)
    assert rt.registered == {} and len(runs) == 0


@pytest.mark.parametrize("err, hint", [(errno.EPERM, True), (errno.EAGAIN, False)])
def test_refused_lock_populates_and_warns_once(
    fake_libc, monkeypatch, caplog, err, hint
):
    fake_libc.lock_errno = err
    monkeypatch.setattr(
        host_memory.resource, "getrlimit", lambda what: (64 << 10, 64 << 10)
    )
    runs, rt = host_memory._PageRuns(), Cudart()
    ro = host_memory._HOST_REGISTER_READ_ONLY
    with caplog.at_level(logging.WARNING):
        held = [runs.acquire(rt, 0, 4 * PAGE, ro)]
        held.append(runs.acquire(rt, 4 * PAGE, 8 * PAGE, ro))
    # Still registered, and faulted in where the lock was refused.
    assert set(rt.registered) == {0, 4 * PAGE}
    assert _merged(fake_libc.populated) == [(0, 8 * PAGE)]
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "GiB" in warnings[0] and host_memory.os.strerror(err) in warnings[0]
    assert ("RLIMIT_MEMLOCK" in warnings[0]) == hint
    for refs in held:
        runs.release(rt, refs)


def _vm_lck_kb() -> int:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmLck:"):
                return int(line.split()[1])
    raise AssertionError("no VmLck in /proc/self/status")


def _mapped_file(tmp_path, pages: int) -> np.memmap:
    path = tmp_path / "table.bin"
    path.write_bytes(b"\x01" * (pages * mmap.PAGESIZE))
    return np.memmap(path, dtype=np.uint8, mode="r")


def test_a_real_mapping_is_locked_then_unlocked(tmp_path, monkeypatch):
    """Real `mlock` on a read-only shared file mapping, as an in-place
    registration of the checkpoint sees it."""
    monkeypatch.setattr(host_memory, "_UNLOCKED_WARNED", [])
    table = _mapped_file(tmp_path, 4)
    lo = table.ctypes.data
    hi = lo + table.nbytes
    soft, _ = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    if soft != resource.RLIM_INFINITY and soft < 2 * table.nbytes:
        pytest.skip("RLIMIT_MEMLOCK too small for 4 pages")
    runs, rt = host_memory._PageRuns(), Cudart()
    before = _vm_lck_kb()
    held = runs.acquire(rt, lo, hi, host_memory._HOST_REGISTER_READ_ONLY)
    try:
        assert _vm_lck_kb() - before == table.nbytes // 1024
    finally:
        runs.release(rt, held)
    assert _vm_lck_kb() == before


def test_a_real_refusal_only_warns(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(host_memory, "_UNLOCKED_WARNED", [])
    table = _mapped_file(tmp_path, 1)
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
            held = runs.acquire(rt, lo, hi, host_memory._HOST_REGISTER_READ_ONLY)
        runs.release(rt, held)
    finally:
        resource.setrlimit(resource.RLIMIT_MEMLOCK, (soft, hard))
    assert held is not None and _vm_lck_kb() == before
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1 and "mlock" in warnings[0].getMessage()
