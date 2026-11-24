# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""The engine kwargs ATOMRayServer supplies rather than inherit from ATOM.

There is one left, `_pin_cudagraph_mode`, and it applies exactly when
torch.compile is on -- never for an eager rollout, where no graphs are captured
and the value would do nothing.

`_pin_sleep_keeps_memory_resident` used to sit beside it, forcing a no-eager
rollout to keep its weights and KV pool allocated across sleep. It was removed
once ATOM's own default (release) was measured to work: see
`test_no_sleep_pin_is_supplied`.
"""

from __future__ import annotations

import inspect
from pathlib import Path

from atomrl.engine.inference.atom_ray_server import ATOMRayServer


def _server() -> ATOMRayServer:
    return ATOMRayServer(
        model_name="/models/Qwen3-8B-Base", engine_kwargs={}, replica_rank=0
    )


# `_pin_cudagraph_mode` resolves a string mode through `atom.config`, which is
# not installed where these tests run. An already-resolved mode skips that, and
# the sleep key is what is under test either way.
_RESOLVED_MODE = object()


def test_no_sleep_pin_is_supplied() -> None:
    # Sleep now releases, because that is ATOM's own default and it was measured
    # to work: example 9 on ATOM main ran twice with 24/24 releases and
    # recaptures and never once derived a negative KV pool. Supplying nothing is
    # what keeps AtomRL from deciding this on ATOM's behalf a second time, so
    # the absence is the assertion -- both of the pin, and of the key it set.
    assert not hasattr(ATOMRayServer, "_pin_sleep_keeps_memory_resident")

    for kwargs in (
        {
            "enforce_eager": False,
            "compilation_config": {"cudagraph_mode": _RESOLVED_MODE},
        },
        {"compilation_config": {"level": 3, "cudagraph_mode": _RESOLVED_MODE}},
        {"enforce_eager": True},
        {},
    ):
        _server()._pin_cudagraph_mode(kwargs)
        assert "sleep_keeps_memory_resident" not in kwargs


def test_an_explicit_setting_is_left_alone() -> None:
    # The knob has not gone away, it is just nobody's default but ATOM's, so a
    # config that asks for resident sleep still gets it.
    for value in (True, False):
        kwargs = {
            "enforce_eager": False,
            "sleep_keeps_memory_resident": value,
            "compilation_config": {"cudagraph_mode": _RESOLVED_MODE},
        }
        _server()._pin_cudagraph_mode(kwargs)
        assert kwargs["sleep_keeps_memory_resident"] is value


def test_the_shared_gate_reads_both_ways_of_asking_for_torch_compile() -> None:
    # run_dapo.sh passes enforce_eager=false *and* compilation_config.level=3 for
    # ATOM FP8; either alone still means graphs will be captured.
    for kwargs, expected in (
        ({"enforce_eager": False}, True),
        ({"compilation_config": {"level": 3}}, True),
        ({"enforce_eager": False, "compilation_config": {"level": 3}}, True),
        ({"compilation_config": {"level": 0}}, False),
        ({"compilation_config": {"level": None}}, False),
        ({"enforce_eager": True}, False),
        ({}, False),
    ):
        assert ATOMRayServer._is_no_eager(kwargs) is expected, kwargs


def _run_dapo_atom_branch() -> str:
    # The `if MODE = atomfp8|atombf16` block, up to the `fi` that closes it.
    text = (
        Path(__file__).resolve().parents[2] / "examples/DAPO/run_dapo.sh"
    ).read_text(encoding="utf-8")
    start = text.index('if [ "$MODE" = "atomfp8" ]')
    return text[start : text.index("\nfi\n", start)]


def test_the_atom_branch_routes_decode_off_the_assembly_kernel() -> None:
    # ATOM's assembly paged-decode kernel returns finite but wrong output when a
    # sequence's context occupies exactly 16 pages with a partial last one. The
    # damage is a handful of arbitrary logprobs per run, which mismatch/abs_diff
    # cannot see and only the quadratic mismatch/chi2_token reports, so nothing
    # fails loudly if this goes missing.
    #
    # It belongs to the ATOM path rather than to particular example numbers:
    # examples/DAPO/examples.yaml lists it for examples 4, 5 and 9, but anyone
    # invoking run_dapo.sh directly gets the same broken kernel, so the default
    # is set here too.
    branch = _run_dapo_atom_branch()
    assert 'export ATOM_FORCE_ATTN_TRITON="${ATOM_FORCE_ATTN_TRITON:-1}"' in branch

    # Overridable, so a measurement of the assembly path is still reachable.
    assert "ATOM_FORCE_ATTN_TRITON=1\n" not in branch


def test_the_replica_manager_forwards_the_decode_override_to_its_actors() -> None:
    # Exporting it only matters if it crosses into the Ray actors: ATOM reads it
    # inside the rollout worker, and an actor's environment comes from the
    # runtime_env the replica manager builds, not from the trainer it forked
    # from. The allow-list is what carries it across.
    source = Path(inspect.getfile(ATOMRayServer)).read_text()
    # The multi-line form; the single-line `for key in (...)` earlier in the
    # file iterates sampling params and is a different list.
    start = source.index("for key in (\n")
    forwarded = source[start : source.index("):", start)]
    assert '"ATOM_FORCE_ATTN_TRITON"' in forwarded
