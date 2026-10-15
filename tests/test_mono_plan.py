# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The mono framework's trace-time hand-off contract: each breach is reported."""

import dataclasses

import pytest

from atom.mono.plan import build_key
from atom.mono.plan.build_key import key_tuple, symbol_params
from atom.mono.plan.check import ContractError, RegionDecl, check
from atom.mono.plan.trace import Space, enter_stage, note, note_unnamed, recording

S, P = Space.SCRATCH, Space.PEER


def _trace(steps):
    """steps: stage names (enter) and (region, space, kind) accesses, in order."""
    with recording() as rec:
        for s in steps:
            if isinstance(s, str):
                enter_stage(s)
            else:
                note(*s)
    return rec


DECLS = [
    RegionDecl("mid", S, "ug"),
    RegionDecl("partials", P, "down", exchange=True),
]
GOOD = [
    "ug", ("mid", S, "put"),
    "down", ("mid", S, "poll"), ("partials", P, "put"), ("partials", P, "poll"),
]  # fmt: skip


def test_a_well_formed_kernel_passes():
    check(_trace(GOOD), DECLS)


def test_nothing_is_recorded_outside_a_recording():
    enter_stage("ug")
    note("mid", S, "put")  # no recorder: a no-op, not an error


def _breach(steps, decls=DECLS):
    with pytest.raises(ContractError) as err:
        check(_trace(steps), decls)
    return str(err.value)


def test_a_consumer_before_its_producer_is_a_deadlock():
    # e12: the down tasks waiting on up / gate tasks their own CTAs run later
    steps = ["down", ("mid", S, "poll"), ("partials", P, "put"), ("partials", P, "poll"),
             "ug", ("mid", S, "put")]  # fmt: skip
    assert "runs before its writer ug" in _breach(steps)


def test_a_second_writer():
    assert "put by down, its writer is ug" in _breach(GOOD + [("mid", S, "put")])


def test_the_wrong_scope():
    steps = GOOD[:2] + ["down", ("mid", P, "poll")] + GOOD[4:]
    assert "as peer, declared scratch" in _breach(steps)


def test_a_self_poll_of_a_non_exchange():
    assert "polled by its own writer ug" in _breach(
        GOOD[:2] + [("mid", S, "poll")] + GOOD[2:]
    )


def test_an_exchange_polled_before_pushed():
    steps = GOOD[:4] + [("partials", P, "poll"), ("partials", P, "put")]
    assert "polled in down before any put" in _breach(steps)


def test_a_writer_that_never_runs():
    assert "its writer ug never runs" in _breach(GOOD[2:])


def test_an_undeclared_region():
    assert "undeclared region x" in _breach(GOOD + [("x", S, "put")])


def test_a_stale_declaration():
    assert "region unused is never accessed" in _breach(
        GOOD, DECLS + [RegionDecl("unused", S, "ug")]
    )


def test_an_unnamed_address():
    with recording() as rec:
        for s in GOOD:
            enter_stage(s) if isinstance(s, str) else note(*s)
        note_unnamed("poll")
    with pytest.raises(ContractError, match="poll on an unnamed address in down"):
        check(rec, DECLS)


@dataclasses.dataclass(frozen=True)
class _Key:
    tokens: int = dataclasses.field(metadata={"sym": "s"})
    eps: float
    heads: object = None


def test_key_tuple_names_every_field_then_the_sources():
    assert key_tuple(_Key(4, 1e-6, 2), "abc") == (
        ("tokens", 4),
        ("eps", 1e-6),
        ("heads", 2),
        ("sources", "abc"),
    )


def test_an_object_field_cannot_be_keyed():
    with pytest.raises(TypeError, match="_Key.heads"):
        key_tuple(_Key(4, 1e-6, object()), "abc")


def test_source_digest_follows_only_its_own_paths(tmp_path, monkeypatch):
    for model in ("a", "b"):
        (tmp_path / model / "kernels").mkdir(parents=True)
        (tmp_path / model / "kernels" / "k.py").write_text("x = 1\n")
    monkeypatch.setattr(build_key, "_ATOM", tmp_path)
    digest = build_key.source_digest.__wrapped__
    a0, b0 = digest("a"), digest("b")
    (tmp_path / "b" / "kernels" / "k.py").write_text("x = 2\n")
    assert digest("a") == a0  # another model's edit rebuilds none of a's kernels
    assert digest("b") != b0
    (tmp_path / "a" / "kernels" / "helper.py").write_text("y = 1\n")
    assert digest("a") != a0  # a new helper file is a source too


def test_source_digest_refuses_a_missing_path(tmp_path, monkeypatch):
    monkeypatch.setattr(build_key, "_ATOM", tmp_path)
    with pytest.raises(FileNotFoundError):
        build_key.source_digest.__wrapped__("nowhere")


def test_symbol_params_are_the_sym_fields():
    assert symbol_params(_Key(4, 1e-6, 2)) == {"s": 4}
