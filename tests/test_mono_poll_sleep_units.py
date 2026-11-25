# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""ATOM_MONO_POLL_SLEEP_UNITS: the gfx950 Mono mailbox retry spacing (4 unless
set, 16 beside a KV transfer or offload connector that copies on streams
beside the forward's; 1 is main's polling, ROCm/ATOM#3369). Plain Python: no
GPU, no FlyDSL."""

import json
import logging
import os
import subprocess
import sys

import pytest

from atom.config import _kv_copies_beside_mono, _resolve_mono_poll_sleep_units
from atom.mono import polling_env


@pytest.mark.parametrize(
    "raw, units", [(None, 4), ("", 4), (" ", 4), ("1", 1), ("16", 16), ("127", 127)]
)
def test_units(raw, units):
    assert polling_env.units(raw) == units


@pytest.mark.parametrize("raw", ["0", "128", "-1", "65536", "abc", "1.5", "16x"])
def test_a_bad_value_is_an_error(raw):
    # s_sleep keeps bits [6:0] of its immediate: 128 would sleep as 0
    with pytest.raises(ValueError, match=polling_env.ENV):
        polling_env.units(raw)


def _probe(value):
    """polling's constants and the digests that hash polling.py, in a fresh
    process with the variable set to ``value`` (None: unset), as a spawned
    worker sees them."""
    env = {k: v for k, v in os.environ.items() if k != polling_env.ENV}
    if value is not None:
        env[polling_env.ENV] = value
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


def test_the_default_is_4_and_1_is_main():
    unset, four, one, sixteen = (_constants(v) for v in (None, "4", "1", "16"))
    assert unset == four
    assert unset[:3] == [4, "s_sleep 4", False]
    # main: s_sleep 1 in sync / step_begin, s_nop 0 in spin_pause and helpers.poll
    assert one[:3] == [1, "s_nop 0", True]
    assert sixteen[:3] == [16, "s_sleep 16", False]
    # every digest that hashes polling.py tells the values apart ...
    for i in (3, 4):
        assert len({unset[i], one[i], sixteen[i]}) == 3
    # ... and one that does not is left alone
    assert unset[5] == one[5] == sixteen[5]


def test_a_bad_value_fails_the_import():
    code, _, err = _probe("128")
    assert code != 0 and f"{polling_env.ENV}='128'" in err


def test_the_process_that_builds_config_reads_copy_safe():
    """Config sets the spacing before its own process imports ``polling``:
    a process that builds Config and then the kernels (a plugin worker) gets
    ``COPY_SAFE`` beside KV copies, not the default its import froze."""
    env = {k: v for k, v in os.environ.items() if k != polling_env.ENV}
    code = (
        "import sys;import atom.config as c;"
        "early='atom.mono.polling' in sys.modules;"
        "c._resolve_mono_poll_sleep_units(True);"
        "from atom.mono import polling as p;"
        "print(early, p.POLL_SLEEP_UNITS)"
    )
    run = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 0, run.stderr
    assert run.stdout.split()[-2:] == ["False", str(polling_env.COPY_SAFE)]


@pytest.mark.parametrize("kv_copies, units", [(False, None), (True, "16")])
def test_config_defaults_beside_kv_copies(kv_copies, units, monkeypatch):

    # set first: teardown then restores the original state, removing what the
    # resolver writes (a bare delenv of an unset name records nothing)
    monkeypatch.setenv(polling_env.ENV, "4")
    monkeypatch.delenv(polling_env.ENV)
    _resolve_mono_poll_sleep_units(kv_copies)
    assert os.environ.get(polling_env.ENV) == units


@pytest.mark.parametrize(
    "kv_copies, units, warns",
    [(True, 1, True), (True, 15, True), (True, 16, False), (True, 32, False),
     (False, 1, False)],
)  # fmt: skip
def test_config_warns_below_copy_safe(kv_copies, units, warns, monkeypatch, caplog):

    monkeypatch.setenv(polling_env.ENV, str(units))
    with caplog.at_level(logging.WARNING, logger="atom"):
        _resolve_mono_poll_sleep_units(kv_copies)
    hits = [r for r in caplog.records if polling_env.ENV in r.getMessage()]
    assert len(hits) == (1 if warns else 0)
    assert os.environ[polling_env.ENV] == str(units)


@pytest.mark.parametrize(
    "kvc, model_type, beside",
    [({}, "deepseek_v41", False),
     (None, "deepseek_v41", False),
     ({"kv_connector": "mooncake", "kv_role": "kv_consumer"}, "deepseek_v41", False),
     ({"kv_connector": " Mooncake "}, "deepseek_v41_dspark", False),
     ({"kv_connector": "mooncake"}, "minimax_m3", True),
     ({"kv_connector": "mooncake"}, "", True),
     ({"kv_connector": "multi"}, "deepseek_v41", True),
     ({"kv_connector": "lmcache"}, "deepseek_v41", True),
     ("mooncake", "deepseek_v41", True)],
)  # fmt: skip
def test_kv_copies_beside_mono(kvc, model_type, beside):
    """Only a connector whose copies run on the forward's stream (Mooncake
    with V4.1's slot regions) leaves the Mono polls at the default spacing."""
    assert _kv_copies_beside_mono(kvc, model_type) is beside
