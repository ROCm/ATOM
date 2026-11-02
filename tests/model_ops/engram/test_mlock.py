"""In-place UVA registrations are `mlock`ed: `_mlock` locks a shared read-only
file mapping (what an in-place registration pins) and only warns when the
kernel refuses; `mlock_in_place` locks only what is registered in place."""

import ctypes
import logging
import mmap
import resource
import sys

import numpy as np
import pytest
import torch

from atom.model_ops.engram import host_memory

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="mlock / VmLck are Linux-only"
)


def _vm_lck_kb() -> int:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmLck:"):
                return int(line.split()[1])
    raise AssertionError("no VmLck in /proc/self/status")


def _table(tmp_path, pages: int) -> np.memmap:
    path = tmp_path / "table.bin"
    path.write_bytes(b"\x01" * (pages * mmap.PAGESIZE))
    return np.memmap(path, dtype=np.uint8, mode="r")


def _mlock_allowed(lo: int, nbytes: int) -> bool:
    """Whether this process may lock `nbytes` at `lo` (probe, then unlock)."""
    libc = ctypes.CDLL(None, use_errno=True)
    libc.mlock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    libc.munlock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    if libc.mlock(lo, nbytes) != 0:
        return False
    libc.munlock(lo, nbytes)
    return True


def test_mlock_locks_the_mapping(tmp_path):
    table = _table(tmp_path, 4)
    lo = table.ctypes.data
    if not _mlock_allowed(lo, table.nbytes):
        pytest.skip("RLIMIT_MEMLOCK too small for 4 pages")
    before = _vm_lck_kb()
    host_memory._mlock([(lo, lo + table.nbytes)])
    assert _vm_lck_kb() - before >= table.nbytes // 1024


def test_mlock_refusal_only_warns(tmp_path, caplog):
    table = _table(tmp_path, 1)
    lo = table.ctypes.data
    soft, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    try:
        resource.setrlimit(resource.RLIMIT_MEMLOCK, (0, hard))
    except (ValueError, OSError):
        pytest.skip("cannot lower RLIMIT_MEMLOCK")
    try:
        if _mlock_allowed(lo, table.nbytes):
            pytest.skip("mlock allowed despite RLIMIT_MEMLOCK=0 (CAP_IPC_LOCK)")
        before = _vm_lck_kb()
        with caplog.at_level(logging.WARNING):
            host_memory._mlock([(lo, lo + table.nbytes)] * 3)
    finally:
        resource.setrlimit(resource.RLIMIT_MEMLOCK, (soft, hard))
    assert _vm_lck_kb() == before
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1 and "mlock" in warnings[0].getMessage()


def test_mlock_in_place_skips_what_is_not_registered(monkeypatch):
    """Only ranges `_PAGE_RUNS` holds (in place) are locked; a copy's memory
    is locked by `hipHostMalloc` already."""
    page = host_memory._PAGE
    runs = host_memory._PageRuns()
    monkeypatch.setattr(host_memory, "_PAGE_RUNS", runs)
    locked = []
    monkeypatch.setattr(host_memory, "_mlock", lambda ranges: locked.extend(ranges))
    registered = torch.ones(3 * page, dtype=torch.uint8)
    copied = torch.ones(3 * page, dtype=torch.uint8)
    base = registered.data_ptr()
    lo = base - base % page
    hi = -(-(base + registered.numel()) // page) * page
    runs._runs[lo] = [hi, 1]
    host_memory.mlock_in_place([registered, copied])
    assert locked == [(lo, hi)]
