# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The step-gap host code that needs neither aiter, FlyDSL nor Triton, so it
runs wherever the tests do: the three switches, the embedding table's canonical
form, digest and agreement, and how the context entry takes either shape of
the draft's interface."""

from types import SimpleNamespace as NS

import pytest
import torch

from atom.models.kimi_k3.step_gap import draft_context as dc
from atom.models.kimi_k3.step_gap import embed_table as et
from atom.utils import envs

# ---- the switches -------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "parts"),
    [
        (None, set()),
        ("0", set()),
        ("", set()),
        ("1", {"kv", "store"}),
        ("kv", {"kv"}),
        ("kv,store", {"kv", "store"}),
        ("store", {"store"}),
    ],
)
def test_the_context_switch_is_a_set_of_parts_off_by_default(monkeypatch, value, parts):
    if value is None:
        monkeypatch.delenv("ATOM_KIMI_MONO_DRAFT_CONTEXT", raising=False)
    else:
        monkeypatch.setenv("ATOM_KIMI_MONO_DRAFT_CONTEXT", value)
    assert envs.ATOM_KIMI_MONO_DRAFT_CONTEXT == frozenset(parts)
    assert bool(envs.ATOM_KIMI_MONO_DRAFT_CONTEXT) == bool(parts)


@pytest.mark.parametrize(
    "name", ["ATOM_KIMI_MONO_EMBED_LOCAL", "ATOM_KIMI_MONO_STEP_LEAN"]
)
@pytest.mark.parametrize(
    ("value", "on"), [(None, False), ("0", False), ("", False), ("1", True)]
)
def test_the_other_switches_are_off_by_default(monkeypatch, name, value, on):
    if value is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, value)
    assert getattr(envs, name) is on


# ---- the embedding table --------------------------------------------------------


def _bf16(values):
    raw = torch.tensor(
        [v - 65536 if v >= 32768 else v for v in values], dtype=torch.int16
    )
    return raw.view(torch.bfloat16)


def _bits(t):
    return [int(v) & 0xFFFF for v in t.view(torch.int16).flatten().tolist()]


@pytest.mark.parametrize(
    ("x", "canonical"),
    [
        (0x0000, 0x0000),
        (0x8000, 0x0000),  # -0.0
        (0x0001, 0x0000),  # the smallest denormal
        (0x007F, 0x0000),  # the largest
        (0x8040, 0x0000),  # a negative denormal
        (0x0080, 0x0080),  # the smallest normal
        (0x8080, 0x8080),
        (0x3F80, 0x3F80),
        (0xBF80, 0xBF80),
        (0x7F7F, 0x7F7F),  # the largest normal
        (0x7F80, 0x7F80),  # +inf
        (0xFF80, 0xFF80),
        (0x7FC0, 0x7FC0),  # a quiet NaN
        (0x7FA0, 0x7FE0),  # a signalling NaN, quieted, its payload kept
        (0xFF81, 0xFFC1),
    ],
)
def test_the_canonical_form(x, canonical):
    assert _bits(et.canonicalize_(_bf16([x]).clone())) == [canonical]


def test_canonicalize_is_in_place_idempotent_and_keeps_normals():
    every = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    table = every.clone().view(256, 256)
    out = et.canonicalize_(table)
    assert out is table
    once = table.clone()
    assert torch.equal(
        et.canonicalize_(table).view(torch.int16), once.view(torch.int16)
    )
    raw, src = once.view(torch.int16).flatten(), every.view(torch.int16)
    exponent = (src >> 7) & 0xFF
    normal = (exponent != 0) & (exponent != 0xFF)
    assert torch.equal(raw[normal], src[normal])
    assert int((raw[exponent == 0] != 0).sum()) == 0
    # 2 zeros and 254 denormals to +0.0 (all but +0.0 itself), 126 NaNs quieted
    assert int((raw != src).sum()) == 255 + 126
    with pytest.raises(TypeError):
        et.canonicalize_(torch.zeros(4, 4))


def test_a_table_of_more_rows_than_a_pass_is_canonical_throughout():
    rows = et.PASS_ROWS + 3
    table = torch.zeros(rows, 2, dtype=torch.int16)
    table[:, 0] = -32768  # -0.0 in every row
    table[:, 1] = 0x3F80
    out = et.canonicalize_(table.view(torch.bfloat16)).view(torch.int16)
    assert int(out[:, 0].abs().sum()) == 0 and bool((out[:, 1] == 0x3F80).all())


def test_the_checksum_tells_tables_apart():
    torch.manual_seed(3)
    a = torch.randn(24, 16).to(torch.bfloat16)
    b = a.clone()
    assert et.checksum(a) == et.checksum(b)
    b[3, 5] = -b[3, 5]
    assert et.checksum(a) != et.checksum(b)
    swapped = a.clone()
    swapped[[0, 1]] = swapped[[1, 0]]
    assert et.checksum(a) != et.checksum(swapped)
    assert 0 <= et.checksum(a) < 1 << 62


def test_group_min_is_this_ranks_values_without_a_group():
    assert et.group_min((1, 7, -7), None) == (1, 7, -7)
    assert et.group_min((True,), None) == (1,)


# ---- the context entry, for either shape of the draft's interface ---------------


def _taking_runner(written):
    return NS(
        kernel=lambda rows: "kernel",
        write=lambda kernel, hidden, positions, slots: written.append(slots) or True,
        refuse=lambda why: None,
    )


def test_the_core_takes_the_slots_given_or_the_forward_contexts(monkeypatch):
    written = []
    hidden = torch.zeros(8, dc.HIDDEN, dtype=torch.bfloat16)
    model = NS(
        mono_context=_taking_runner(written),
        project_context=lambda source: hidden,
        context_layers=[],
    )
    monkeypatch.setattr(dc, "step_rows", lambda positions: 8)
    monkeypatch.setattr(dc, "context_slots", lambda rows: f"context slots [:{rows}]")
    assert dc.write_context_kv(model, "taps", "positions", "given slots") is True
    assert dc.write_context_kv(model, "joined", "positions") is True
    assert written == ["given slots", "context slots [:8]"]
    # a caller that has no slots, or passes more than the slots: production's
    assert dc.write_context_kv(model, "taps", "positions", None) is False
    assert dc.write_context_kv(model, "taps", "positions", "slots", 8) is False
    assert len(written) == 2


def test_a_refused_write_is_finished_with_the_callers_own_arguments(monkeypatch):
    calls = []
    layer = NS(write_context_kv=lambda *a: calls.append(a))
    runner = NS(
        kernel=lambda rows: "kernel", write=lambda *a: False, refuse=lambda why: None
    )
    hidden = torch.zeros(8, dc.HIDDEN, dtype=torch.bfloat16)
    monkeypatch.setattr(dc, "step_rows", lambda positions: 8)
    monkeypatch.setattr(dc, "context_slots", lambda rows: "context slots")
    for rest in (("slots",), ()):
        model = NS(
            mono_context=runner,
            project_context=lambda source: hidden,
            context_layers=[layer],
        )
        assert dc.write_context_kv(model, "source", "positions", *rest) is True
    assert calls == [(hidden, "positions", "slots"), (hidden, "positions")]


def test_a_step_the_entry_does_not_take_launches_nothing(monkeypatch):
    model = NS(
        mono_context=None, project_context=lambda source: pytest.fail("projected")
    )
    monkeypatch.setattr(dc, "step_rows", lambda positions: 0)
    monkeypatch.setattr(dc, "bind", lambda model: pytest.fail("bound"))
    assert dc.write_context_kv(model, "source", "positions", "slots") is False
