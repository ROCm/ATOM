# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""A runner process that raises must end at once, not at interpreter teardown.

On DeepSeek-V4-Pro CI, 7 of 8 ModelRunners printed an OOM traceback and then
never exited: the normal teardown after an uncaught exception blocked, the
manager's sentinel never fired, and the group sat for the 30-minute gloo
timeout. `run_exiting_on_crash` ends the process with `os._exit` instead.

The stand-in for the blocked teardown is a non-daemon thread that never
finishes: `threading._shutdown` joins it, so a plain uncaught exception leaves
the process alive -- the control case below shows that it does.
"""

import sys
import threading
import time

import pytest

from atom.utils import get_mp_context, run_exiting_on_crash

_EXIT_WITHIN_S = 20.0


def _block_teardown_then_raise():
    threading.Thread(target=threading.Event().wait, daemon=False).start()
    raise RuntimeError("simulated runner OOM")


def _return_cleanly():
    return None


def _exit_with(code):
    sys.exit(code)


def _start(target, *args):
    proc = get_mp_context().Process(target=target, args=args, name="FakeRunner0/1")
    proc.start()
    return proc


def _join(proc, timeout):
    proc.join(timeout)
    alive = proc.is_alive()
    if alive:
        proc.kill()
        proc.join(5)
    return alive


def test_a_raising_runner_exits_at_once_and_reports(capfd):
    proc = _start(run_exiting_on_crash, _block_teardown_then_raise)
    start = time.monotonic()
    assert not _join(proc, _EXIT_WITHIN_S), "the crashed process outlived its crash"
    assert time.monotonic() - start < _EXIT_WITHIN_S
    assert proc.exitcode == 1
    err = capfd.readouterr().err
    assert "simulated runner OOM" in err
    assert "FakeRunner0/1 crashed" in err


def test_without_it_the_same_crash_leaves_the_process_alive():
    """The control: proves the stand-in reproduces the hang this guards against."""
    proc = _start(_block_teardown_then_raise)
    assert _join(proc, 5.0), "control did not hang; the test above proves nothing"


def test_a_clean_return_exits_zero():
    proc = _start(run_exiting_on_crash, _return_cleanly)
    assert not _join(proc, _EXIT_WITHIN_S)
    assert proc.exitcode == 0


@pytest.mark.parametrize("code,expected", [(0, 0), (None, 0), (3, 3)])
def test_sys_exit_keeps_its_code(code, expected):
    proc = _start(run_exiting_on_crash, _exit_with, code)
    assert not _join(proc, _EXIT_WITHIN_S)
    assert proc.exitcode == expected
