# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""ATOM_MONO_POLL_SLEEP_UNITS: the gfx950 Mono mailbox retry spacing (16 unless
set; 1 is main's polling, ROCm/ATOM#3369). Plain Python: no GPU, no FlyDSL."""

import json
import logging
import os
import subprocess
import sys

import pytest

from atom.mono import polling


@pytest.mark.parametrize(
    "raw, units", [(None, 16), ("", 16), (" ", 16), ("1", 1), ("8", 8), ("127", 127)]
)
def test_units(raw, units):
    assert polling._units(raw) == units


@pytest.mark.parametrize("raw", ["0", "128", "-1", "65536", "abc", "1.5", "16x"])
def test_a_bad_value_is_an_error(raw):
    # s_sleep keeps bits [6:0] of its immediate: 128 would sleep as 0
    with pytest.raises(ValueError, match=polling.ENV):
        polling._units(raw)


def _probe(value):
    """polling's constants and the digests that hash polling.py, in a fresh
    process with the variable set to ``value`` (None: unset), as a spawned
    worker sees them."""
    env = {k: v for k, v in os.environ.items() if k != polling.ENV}
    if value is not None:
        env[polling.ENV] = value
    code = (
        "import json;from atom.mono import polling as p;"
        "from atom.mono.plan.build_key import source_digest as d;"
        "print(json.dumps([p.POLL_SLEEP_UNITS,p.SPIN_PAUSE_ASM,p.MAIN_POLLING,"
        "d('mono/polling.py'),d('mono'),d('mono/device')]))"
    )
    run = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    return run.returncode, run.stdout, run.stderr


def _constants(value):
    code, out, err = _probe(value)
    assert code == 0, err
    return json.loads(out.strip().splitlines()[-1])


def test_the_default_is_16_and_1_is_main():
    unset, sixteen, one, eight = (_constants(v) for v in (None, "16", "1", "8"))
    assert unset == sixteen
    assert unset[:3] == [16, "s_sleep 16", False]
    # main: s_sleep 1 in sync / step_begin, s_nop 0 in spin_pause and helpers.poll
    assert one[:3] == [1, "s_nop 0", True]
    assert eight[:3] == [8, "s_sleep 8", False]
    # every digest that hashes polling.py tells the values apart ...
    for i in (3, 4):
        assert len({unset[i], one[i], eight[i]}) == 3
    # ... and one that does not is left alone
    assert unset[5] == one[5] == eight[5]


def test_a_bad_value_fails_the_import():
    code, _, err = _probe("128")
    assert code != 0 and f"{polling.ENV}='128'" in err


@pytest.mark.parametrize(
    "units, warns", [(1, True), (15, True), (16, False), (32, False)]
)
def test_config_warns_below_the_default(units, warns, monkeypatch, caplog):
    from atom.config import _check_mono_poll_sleep_units

    monkeypatch.setattr(polling, "POLL_SLEEP_UNITS", units)
    with caplog.at_level(logging.WARNING, logger="atom"):
        _check_mono_poll_sleep_units()
    hits = [r for r in caplog.records if polling.ENV in r.getMessage()]
    assert len(hits) == (1 if warns else 0)
