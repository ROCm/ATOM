# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The step-gap hooks at Kimi-K3's decode installer
(``step_tail.dispatch.install_mono_decode`` -> ``step_gap.install``): off by
default, each switch its own hook, and the mono installer's result untouched
either way."""

import pytest

pytest.importorskip("aiter", reason="the installer imports the mono dispatch")
pytest.importorskip("triton", reason="the block metadata mirror's kernel is Triton")

import atom.models.kimi_k3.step_gap.install as gap
import atom.models.kimi_k3.step_tail.dispatch as tail
from atom.models.kimi_k3.step_gap import block_meta, embed_local

SWITCHES = ("ATOM_KIMI_MONO_STEP_LEAN", "ATOM_KIMI_MONO_EMBED_LOCAL")


@pytest.fixture
def installer(monkeypatch):
    """The mono installer and the two hooks, recorded; the tail off."""
    calls = []
    monkeypatch.delenv("ATOM_KIMI_MONO_STEP_TAIL", raising=False)
    for name in SWITCHES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(
        tail.mono_dispatch,
        "install_mono_decode",
        lambda model, cfg, drafter, metadata_builder=None: (
            calls.append(("mono", model, cfg, drafter, metadata_builder)) or "installed"
        ),
    )
    monkeypatch.setattr(
        block_meta, "install", lambda drafter: calls.append(("lean", drafter))
    )
    monkeypatch.setattr(
        embed_local, "install", lambda model: calls.append(("embed", model))
    )
    return calls


def test_off_nothing_of_the_step_gap_is_reached(installer, monkeypatch):
    monkeypatch.setattr(gap, "install", lambda *a: pytest.fail("step gap asked"))
    assert (
        tail.install_mono_decode("m", "cfg", "d", metadata_builder="b") == "installed"
    )
    assert installer == [("mono", "m", "cfg", "d", "b")]


def test_step_lean_binds_the_drafter_before_mono_installs(installer, monkeypatch):
    monkeypatch.setenv("ATOM_KIMI_MONO_STEP_LEAN", "1")
    assert tail.install_mono_decode("m", "cfg", "d") == "installed"
    assert installer == [("lean", "d"), ("mono", "m", "cfg", "d", None)]


def test_step_lean_without_a_drafter_binds_nothing(installer, monkeypatch):
    monkeypatch.setenv("ATOM_KIMI_MONO_STEP_LEAN", "1")
    assert tail.install_mono_decode("m", "cfg", None) == "installed"
    assert installer == [("mono", "m", "cfg", None, None)]


def test_embed_local_binds_the_models_embedding(installer, monkeypatch):
    monkeypatch.setenv("ATOM_KIMI_MONO_EMBED_LOCAL", "1")
    assert tail.install_mono_decode("m", "cfg", "d") == "installed"
    assert installer == [("embed", "m"), ("mono", "m", "cfg", "d", None)]


def test_both_switches_run_both_hooks(installer, monkeypatch):
    for name in SWITCHES:
        monkeypatch.setenv(name, "1")
    assert tail.install_mono_decode("m", "cfg", "d") == "installed"
    assert installer[:2] == [("lean", "d"), ("embed", "m")]
