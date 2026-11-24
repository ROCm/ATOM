# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""engine.thp_disable: `atom.utils.disable_thp` sets PR_SET_THP_DISABLE for the
process and the processes it starts; off, nothing changes. Each case runs in
its own interpreter so the test runner's own flag is never touched."""

import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="prctl is Linux-only"
)

_PROBE = r"""
import ctypes, subprocess, sys
from atom.utils import disable_thp
libc = ctypes.CDLL(None)
libc.prctl.argtypes = [ctypes.c_int] + [ctypes.c_ulong] * 4
get = lambda: libc.prctl(42, 0, 0, 0, 0)
enabled = sys.argv[1] == "on"
before = get()
first = disable_thp("test", enabled=enabled)
second = disable_thp(enabled=enabled)
child = subprocess.run(
    [sys.executable, "-c",
     "import ctypes; l = ctypes.CDLL(None); l.prctl.argtypes = [ctypes.c_int] + [ctypes.c_ulong] * 4; print(l.prctl(42, 0, 0, 0, 0))"],
    capture_output=True, text=True, check=True,
).stdout.strip()
print(before, first, second, get(), child)
"""


def _run(enabled):
    out = (
        subprocess.run(
            [sys.executable, "-c", _PROBE, "on" if enabled else "off"],
            capture_output=True,
            text=True,
            check=True,
        )
        .stdout.strip()
        .splitlines()[-1]
    )
    return out.split()


def test_off_changes_nothing():
    before, first, second, after, child = _run(False)
    assert (first, second) == ("False", "False")
    assert after == before == child


def test_the_default_is_off():
    from atom.plugin_recipe import EngineOptions
    from atom.utils import disable_thp

    assert not EngineOptions().thp_disable
    assert disable_thp("test") is False


def test_on_disables_for_process_and_children():
    _before, first, second, after, child = _run(True)
    assert (first, second) == ("True", "True")
    assert after == "1"
    assert child == "1"


def test_a_recipe_sets_it_in_the_server_process_before_any_worker(monkeypatch):
    """`plugin_recipe.from_cli` applies engine.thp_disable where it applies the
    recipe's env: in the process that starts the workers."""
    from types import SimpleNamespace

    import atom.utils
    from atom import plugin_recipe

    calls = []
    monkeypatch.setattr(
        atom.utils,
        "disable_thp",
        lambda label="", *, enabled=False: calls.append((label, enabled)),
    )
    on = SimpleNamespace(engine=plugin_recipe.EngineOptions(thp_disable=True))
    off = SimpleNamespace(engine=plugin_recipe.EngineOptions())
    plugin_recipe.apply_host_options(None)
    plugin_recipe.apply_host_options(off)
    plugin_recipe.apply_host_options(on)
    assert calls == [("server", False), ("server", True)]
