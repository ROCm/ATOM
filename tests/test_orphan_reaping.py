# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
"""CPU unit test for atom.utils.enable_orphan_reaping (bug5 IPC lifecycle).

No GPU required. Builds a 3-level process tree:

    main (this test)  ->  parent P  ->  child G

G arms ``enable_orphan_reaping(SIGKILL)`` and then blocks forever. We kill P
and assert that G is reaped by the OS (not left orphaned holding resources).

Liveness is detected with a pipe whose write end only G keeps open: while G is
alive the pipe stays open; the instant G dies (for any reason, including the
kernel-delivered PR_SET_PDEATHSIG) its fds close and main sees EOF. This is
independent of who reaps the zombie, so it works whether G reparents to a
subreaper or to init.

A negative-control test forks a child that does NOT arm reaping and asserts it
is instead left orphaned/alive after its parent dies — proving the test
actually exercises the mechanism the fix installs.
"""

import ast
import logging
import multiprocessing
import os
import select
import signal
import sys
import textwrap
import time

import pytest

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="PR_SET_PDEATHSIG is Linux-only",
)

_UTILS_SRC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "atom",
    "utils",
    "__init__.py",
)


def _load_enable_orphan_reaping():
    """Return the REAL enable_orphan_reaping from atom/utils/__init__.py.

    Prefer a normal import (works when the full atom package is importable, e.g.
    inside a container with an up-to-date aiter). Fall back to extracting the
    exact function body from the source file with AST, so this test still ties
    to the shipped source on a CPU-only host where importing the whole `atom`
    package pulls torch/aiter. Either way we exercise the real function text.
    """
    try:
        from atom.utils import enable_orphan_reaping

        return enable_orphan_reaping
    except ImportError:
        # A missing heavy dep (torch/aiter on a CPU-only host) is the expected
        # reason the package import fails here; fall back to reading the function
        # straight from source. Non-import errors (SyntaxError/runtime) still
        # propagate, and the fallback exec()s the *same* source, so a genuine
        # defect in enable_orphan_reaping is not masked either way.
        with open(_UTILS_SRC) as f:
            src = f.read()
        tree = ast.parse(src)
        for node in tree.body:
            if (
                isinstance(node, ast.FunctionDef)
                and node.name == "enable_orphan_reaping"
            ):
                fn_src = ast.get_source_segment(src, node)
                ns = {
                    "signal": signal,
                    "os": os,
                    "sys": sys,
                    "multiprocessing": multiprocessing,
                    "logger": logging.getLogger("atom-test"),
                }
                exec(textwrap.dedent(fn_src), ns)
                return ns["enable_orphan_reaping"]
        raise RuntimeError("enable_orphan_reaping not found in source")


_enable_orphan_reaping = _load_enable_orphan_reaping()


def _spawn_tree(arm: bool):
    """Fork main -> P -> G. G optionally arms orphan reaping, then blocks.

    Returns (parent_pid, read_fd). ``read_fd`` gets a line with G's pid as the
    ready signal, and later reaches EOF exactly when G dies.
    """
    r, w = os.pipe()
    pid_p = os.fork()
    if pid_p == 0:
        # ---- process P (the "parent" we will kill) ----
        os.close(r)
        pid_g = os.fork()
        if pid_g == 0:
            # ---- process G (the child that should die with its parent) ----
            try:
                if arm:
                    _enable_orphan_reaping(signal.SIGKILL)
                os.write(w, f"{os.getpid()}\n".encode())
                while True:
                    time.sleep(3600)
            finally:
                os._exit(0)
        else:
            # P: drop the write end so only G holds it, then block forever.
            os.close(w)
            while True:
                time.sleep(3600)
    else:
        # ---- main ----
        os.close(w)  # only G now holds the write end
        return pid_p, r


def _read_ready_pid(read_fd, timeout=15.0):
    deadline = time.monotonic() + timeout
    buf = b""
    while b"\n" not in buf:
        remaining = deadline - time.monotonic()
        assert remaining > 0, "timed out waiting for child ready signal"
        rl, _, _ = select.select([read_fd], [], [], remaining)
        assert rl, "timed out waiting for child ready signal"
        buf += os.read(read_fd, 64)
    return int(buf.split(b"\n")[0])


def _wait_eof(read_fd, timeout):
    """Return True if the pipe reaches EOF (child died) within timeout."""
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        rl, _, _ = select.select([read_fd], [], [], remaining)
        if not rl:
            return False
        if os.read(read_fd, 64) == b"":  # EOF -> G's fds closed -> G dead
            return True


def test_child_dies_when_parent_dies():
    """G arms orphan reaping; killing P must cause the kernel to kill G."""
    pid_p, read_fd = _spawn_tree(arm=True)
    gpid = None
    try:
        gpid = _read_ready_pid(read_fd)
        os.kill(pid_p, signal.SIGKILL)
        reaped = _wait_eof(read_fd, timeout=10.0)
        assert (
            reaped
        ), "child was orphaned: PR_SET_PDEATHSIG did not fire when parent died"
    finally:
        os.close(read_fd)
        _reap(pid_p)
        if gpid is not None:
            try:
                os.kill(gpid, signal.SIGKILL)  # ensure no leak if assert failed
            except ProcessLookupError:
                pass
            _reap(gpid)


def test_child_orphaned_without_arming():
    """Negative control: without reaping armed, G survives P's death (orphan)."""
    pid_p, read_fd = _spawn_tree(arm=False)
    gpid = None
    try:
        gpid = _read_ready_pid(read_fd)
        os.kill(pid_p, signal.SIGKILL)
        # Expect NO EOF: the child stays alive (orphaned) after the parent dies.
        reaped = _wait_eof(read_fd, timeout=3.0)
        assert not reaped, "control child unexpectedly died without PR_SET_PDEATHSIG"
    finally:
        os.close(read_fd)
        _reap(pid_p)
        if gpid is not None:
            try:
                os.kill(gpid, signal.SIGKILL)  # clean up the deliberate orphan
            except ProcessLookupError:  # already gone (e.g. external teardown)
                pass
            _reap(gpid)


def _reap(pid):
    if pid is None:
        return
    try:
        os.waitpid(pid, 0)
    except (ChildProcessError, OSError):
        pass


if __name__ == "__main__":
    test_child_dies_when_parent_dies()
    print("PASS: child reaped when parent dies (orphan reaping armed)")
    test_child_orphaned_without_arming()
    print("PASS: child orphaned when reaping NOT armed (control)")
    print("\nALL PASS")
