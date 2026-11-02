"""ATOM_ENGRAM_MLOCK: `_mlock` locks a shared read-only file mapping (what an
in-place registration pins) and only warns when the kernel refuses."""

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


def test_mlock_locks_the_mapping(tmp_path):
    soft, _ = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    pages = 4
    if soft != resource.RLIM_INFINITY and soft < 2 * pages * mmap.PAGESIZE:
        pytest.skip("RLIMIT_MEMLOCK too small")
    table = _table(tmp_path, pages)
    lo = table.ctypes.data
    before = _vm_lck_kb()
    host_memory._mlock(lo, lo + table.nbytes)
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
        with caplog.at_level(logging.WARNING):
            host_memory._mlock(lo, lo + table.nbytes)
    finally:
        resource.setrlimit(resource.RLIMIT_MEMLOCK, (soft, hard))
    if not caplog.records:
        pytest.skip("mlock allowed despite RLIMIT_MEMLOCK=0 (CAP_IPC_LOCK)")
    assert "mlock" in caplog.text
