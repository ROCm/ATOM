# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""The IPC receive call must not pass FP8 args a non-vLLM server cannot accept.

ATOMRayServer.update_weights_from_ipc is (use_shm, version) -- no prequantized_fp8.
Passing a third POSITIONAL argument to it raised "TypeError: too many positional
arguments" from inside Ray's own arg validation, which broke release examples 4, 5
and 9 (every ATOM rollout) while every vLLM example passed. Nothing mocked at the
vLLM server could catch that, so the contract is pinned here instead: with FP8 sync
off -- the default -- the call must be exactly two positional args.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest


def _atom_signature():
    from atomrl.engine.inference import atom_ray_server

    return inspect.signature(atom_ray_server.ATOMRayServer.update_weights_from_ipc)


def test_atom_server_still_takes_only_use_shm_and_version():
    """If ATOM ever gains the parameter, this test should be revisited, not deleted."""
    params = [p for p in _atom_signature().parameters if p != "self"]
    assert params == ["use_shm", "version"], params


def test_the_trainer_call_site_does_not_pass_fp8_positionally():
    """Guard the shape of the call, not the runtime -- spinning up Ray here is not
    worth it, and the defect was purely one of arity at the call site."""
    src = (
        Path(__file__).resolve().parents[2] / "atomrl" / "trainer" / "rl_trainer.py"
    ).read_text(encoding="utf-8")
    assert (
        "update_weights_from_ipc.remote(use_shm, version, fp8_sync)" not in src
    ), "fp8_sync is being passed positionally again; ATOMRayServer cannot accept it"
    assert "update_weights_from_ipc.remote(use_shm, version, **recv_kw)" in src


@pytest.mark.parametrize(
    "fp8_sync,expected", [(False, {}), (True, {"prequantized_fp8": True})]
)
def test_the_kwarg_is_present_only_when_fp8_sync_is_on(fp8_sync, expected):
    """Mirrors the call site's own expression, so a change to it fails here."""
    recv_kw = {"prequantized_fp8": True} if fp8_sync else {}
    assert recv_kw == expected
