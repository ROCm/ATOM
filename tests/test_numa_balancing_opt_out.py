# SPDX-License-Identifier: MIT

"""`opt_out_of_numa_balancing`: MPOL_LOCAL over MPOL_DEFAULT, nothing else.

Each case runs in a child process: the task memory policy it sets would
otherwise leak into the test runner and every test after this one.
"""

from __future__ import annotations

import ctypes
import multiprocessing
import platform

import pytest

from atom.utils import numa_utils

_NRS = numa_utils._SYS_MEMPOLICY.get(platform.machine())
pytestmark = pytest.mark.skipif(_NRS is None, reason="no mempolicy syscalls here")


def _mode() -> int:
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    mode = ctypes.c_int(-1)
    assert (
        libc.syscall(
            ctypes.c_long(_NRS[1]),
            ctypes.byref(mode),
            None,
            ctypes.c_ulong(0),
            None,
            ctypes.c_ulong(0),
        )
        == 0
    )
    return mode.value


def _set(mode: int) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    assert (
        libc.syscall(
            ctypes.c_long(_NRS[0]), ctypes.c_int(mode), None, ctypes.c_ulong(0)
        )
        == 0
    )


def _child(q, start_mode, env_on, balancing):
    import os

    os.environ["ATOM_NUMA_BALANCING_OPT_OUT"] = "1" if env_on else "0"
    if start_mode is not None:
        _set(start_mode)
    real_open = open

    def fake_open(path, *a, **k):  # the sysctl, without needing it set on the host
        if path == "/proc/sys/kernel/numa_balancing":
            import io

            return io.StringIO(balancing)
        return real_open(path, *a, **k)

    numa_utils.open = fake_open
    try:
        did = numa_utils.opt_out_of_numa_balancing("test")
    finally:
        del numa_utils.open
    q.put((did, _mode()))


def _run(start_mode=None, env_on=True, balancing="1\n"):
    ctx = multiprocessing.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_child, args=(q, start_mode, env_on, balancing))
    p.start()
    out = q.get(timeout=60)
    p.join(timeout=60)
    return out


def test_default_policy_becomes_local():
    # the case it exists for: the default policy (MPOL_F_MOF, scanned) replaced
    assert _run() == (True, numa_utils._MPOL_LOCAL)


def test_operator_policy_is_kept():
    # an explicit policy (numactl --localalloc / --preferred / --membind) must
    # survive: overwriting it would undo the operator's choice
    assert _run(start_mode=numa_utils._MPOL_LOCAL) == (False, numa_utils._MPOL_LOCAL)


def test_switch_off_and_balancing_off_change_nothing():
    assert _run(env_on=False) == (False, numa_utils._MPOL_DEFAULT)
    assert _run(balancing="0\n") == (False, numa_utils._MPOL_DEFAULT)
