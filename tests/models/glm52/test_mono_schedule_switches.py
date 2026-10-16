# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The GLM-5.2 mono kernel's schedule switches are discoverable.

Six knobs on the vendored FlyDSL kernel's schedule were read straight from
`os.environ` inside `kernels/kernel.py` and `kernels/layout.py`, so they were
live in production while appearing in neither `atom/utils/envs.py` nor
`docs/environment_variables.md`. Nothing was miscomputed by that -- every one
of them is bit-identical either way -- and this is engineering debt, not a
correctness bug: an operator could set a switch nobody could find, and nobody
reading `envs.py` could learn the switch existed.

The fix is routing, so the tests are about the route: each switch is declared
once, documented once, and still has the default and the override behaviour it
had when it read `os.environ` itself.

`test_no_mono_kernel_module_reads_os_environ_directly` is a structural
assertion on purpose. "These modules hold no undeclared switch" IS a
structural property, not a stand-in for a behavioural one, and it is the only
thing that catches a seventh switch being added the old way later -- which is
exactly how these six arrived.
"""

import ast
import pathlib

import pytest

import atom.models.glm52.mono.kernels as mono_kernels
from atom.utils import envs

SWITCHES = {
    # name: (default with the environment clear, a value to override with)
    "MONO_DN_HOIST": (True, "0"),
    "MONO_R2_OFF": (False, "1"),
    "MONO_SPLIT_QHEADS": (True, "0"),
    "MONO_UG_SCALE_PAIR": (True, "0"),
    "MONO_DN_BATCH": (None, "5"),
    "MONO_PTPC_CHUNK": (None, "3"),
}

# Located from the packages themselves rather than by counting `..` from this
# file: a wrong path would make `rglob` match nothing and the completeness
# check below pass by finding no files at all.
KERNELS = pathlib.Path(mono_kernels.__file__).parent
DOCS = pathlib.Path(envs.__file__).parents[2] / "docs/environment_variables.md"


@pytest.fixture
def clean_env(monkeypatch):
    for name in SWITCHES:
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize("name", sorted(SWITCHES))
def test_every_switch_is_declared_in_envs(name):
    # The whole point of the change: `envs.py` is where ATOM says what is
    # settable, so a switch missing from it is a switch nobody can find.
    assert name in envs.environment_variables


@pytest.mark.parametrize("name", sorted(SWITCHES))
def test_every_switch_is_documented(name):
    assert f"**{name}**" in DOCS.read_text()


@pytest.mark.parametrize("name", sorted(SWITCHES))
def test_the_default_survived_the_move(clean_env, name):
    # Each of these is byte-for-byte what the `os.environ` read it replaced
    # returned with the variable unset. The two int switches read back None
    # rather than a number because "unset" means "the measured per-width
    # default", which no single value reproduces.
    assert getattr(envs, name) is SWITCHES[name][0]


@pytest.mark.parametrize("name", sorted(SWITCHES))
def test_a_switch_still_reads_the_environment_live(clean_env, monkeypatch, name):
    # These are read at BUILD time, not import time, so `envs.__getattr__` must
    # re-evaluate on every access. A cached value would silently ignore a
    # switch set after the first read and make an A/B report the same number
    # twice.
    default, override = SWITCHES[name]
    monkeypatch.setenv(name, override)
    changed = getattr(envs, name)
    assert changed != default
    assert changed == (int(override) if default is None else override == "1")


@pytest.mark.parametrize(
    "name", sorted(n for n in SWITCHES if SWITCHES[n][0] is not None)
)
@pytest.mark.parametrize(
    ("written", "means"),
    [
        ("true", True),
        ("on", True),
        ("yes", True),
        ("1", True),
        ("false", False),
        ("off", False),
        ("no", False),
        ("0", False),
        (" 0 ", False),
        ("", False),
    ],
)
def test_a_flag_accepts_what_a_bool_is_written_as(
    clean_env, monkeypatch, name, written, means
):
    # The kernel's own test was `== "1"`, which read every other value as off
    # -- so `MONO_R2_OFF=true` turned round 2 ON, the opposite of what the
    # operator wrote, silently, in a switch whose only job is to make an A/B
    # mean something. Survivable while these were undiscoverable; documenting
    # them as `bool` is a promise to accept what a bool is written as.
    monkeypatch.setenv(name, written)
    assert getattr(envs, name) is means


def test_the_two_width_defaults_are_unchanged(clean_env):
    from atom.models.glm52.mono.kernels.layout import (
        down_prefetch_batch,
        ptpc_stage_chunk,
    )

    # the measured winners, which must not move when the switch is unset
    assert down_prefetch_batch(4) == 9
    assert down_prefetch_batch(6) == 4
    assert down_prefetch_batch(6, native_fp4_mfma=True) == 4
    assert down_prefetch_batch(12, native_fp4_mfma=True) == 3
    assert ptpc_stage_chunk(6) == 2
    assert ptpc_stage_chunk(12) == 4


@pytest.mark.parametrize(
    ("name", "helper", "args", "value"),
    [
        ("MONO_DN_BATCH", "down_prefetch_batch", (6,), 5),
        ("MONO_PTPC_CHUNK", "ptpc_stage_chunk", (12,), 3),
    ],
)
def test_an_override_still_reaches_the_helper(
    clean_env, monkeypatch, name, helper, args, value
):
    # End to end through the vendored helper, not just through `envs`: the
    # route is only fixed if the kernel side actually consumes it.
    import atom.models.glm52.mono.kernels.layout as mono_layout

    monkeypatch.setenv(name, str(value))
    assert getattr(mono_layout, helper)(*args) == value


@pytest.mark.parametrize(
    ("name", "helper", "args"),
    [
        ("MONO_DN_BATCH", "down_prefetch_batch", (6,)),
        ("MONO_PTPC_CHUNK", "ptpc_stage_chunk", (12,)),
    ],
)
def test_an_empty_override_falls_back_to_the_default(
    clean_env, monkeypatch, name, helper, args
):
    # `VAR=` is how a shell script clears a switch inline. It read as unset
    # before the move (the old code tested the raw string for truthiness) and
    # must still, or clearing a switch would raise `int('')` instead.
    import atom.models.glm52.mono.kernels.layout as mono_layout

    monkeypatch.setenv(name, "")
    assert getattr(mono_layout, helper)(*args) == getattr(mono_layout, helper)(*args)
    assert getattr(envs, name) is None


def test_no_mono_kernel_module_reads_os_environ_directly():
    # The completeness check. Parsed rather than grepped so a mention in a
    # comment or docstring cannot satisfy it, and a real read cannot hide in
    # one. Every switch these modules honour must be declared in `envs.py`,
    # which is what makes the list in `docs/environment_variables.md`
    # exhaustive rather than merely current.
    offenders = []
    scanned = sorted(KERNELS.rglob("*.py"))
    # Without this the test passes when the path is wrong and nothing is read.
    assert {p.name for p in scanned} >= {"kernel.py", "layout.py", "op.py"}
    for path in scanned:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Attribute):
                continue
            base = node.value
            if isinstance(base, ast.Name) and base.id == "os":
                if node.attr in ("environ", "getenv"):
                    offenders.append(f"{path.name}:{node.lineno} os.{node.attr}")
            elif (
                isinstance(base, ast.Attribute)
                and base.attr == "environ"
                and isinstance(base.value, ast.Name)
                and base.value.id == "os"
            ):
                offenders.append(f"{path.name}:{node.lineno} os.environ.{node.attr}")
    assert not offenders, (
        "undeclared environment reads in the vendored GLM-5.2 mono kernels; "
        f"declare them in atom/utils/envs.py and document them: {offenders}"
    )
