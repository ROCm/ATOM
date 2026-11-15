# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The V4.1 mono hi hook's step-phase bumps (``PeerGroup.end_step``), deferred to one launch at the target forward's
end (``HiLayers.end_step``: the post-verify's program 0, or ``bump_phases``), on CPU stand-ins: no GPU, no peers. The
bump kernel itself runs in the Triton interpreter in ``tests/models/deepseek_v41/test_mono_hi_bridge.py``
(``check_post_verify_pad``)."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("triton")


def _group(tp=2):
    from atom.models.deepseek_v41.mono.hi.runtime.peer import PeerGroup

    g = object.__new__(PeerGroup)
    g.tp, g.since_step, g._phase, g.last_parity, g.ids = tp, 0, 0, None, set()
    g.addresses = torch.tensor([100, 200, 0], dtype=torch.int64)
    return g


def test_end_step_adds_now_or_hands_the_word_back():
    """Odd launches since the last call: the host phase flips, and the word gets + 1 now, or (``defer``) is handed
    back un-bumped (the peer table's entry tp, a view the caller bumps); even: nothing, None.
    """
    g = _group()
    g.since_step = 41  # (a step's launches on the group: 40 layers and the head)
    assert (
        g.end_step() is None and g.addresses.tolist() == [100, 200, 1] and g._phase == 1
    )
    g.since_step = 1
    word = g.end_step(defer=True)
    assert g.addresses.tolist() == [100, 200, 1] and g._phase == 0 and g.since_step == 0
    assert word.data_ptr() == g.addresses[2:].data_ptr() and word.numel() == 1
    g.since_step = 2
    assert g.end_step(defer=True) is None and g._phase == 0


def test_draft_end_step_collects_due_words():
    from atom.models.deepseek_v41.mono.hi_draft import _end_step

    g, due = _group(), []
    g.since_step = 1
    _end_step(g, due)
    assert len(due) == 1 and g.addresses.tolist() == [100, 200, 0]
    g.since_step = 1
    _end_step(g)  # (outside a forward: its own add)
    assert len(due) == 1 and g.addresses.tolist() == [100, 200, 1]
    _end_step(None, due)
    assert len(due) == 1


@pytest.mark.parametrize("verify", [True, False])
def test_the_forward_bumps_its_due_phases_once_at_its_end(monkeypatch, verify):
    """``HiLayers.end_step``: the words collected since ``begin_step`` go to the post-verify's launch (a served
    verify), else to one ``bump_phases``; the collection closes (None: each group's own add outside a forward).
    """
    from atom.models.deepseek_v41.mono import hi_runner
    from atom.models.deepseek_v41.mono.round import post_verify

    calls = []
    hook = object.__new__(hi_runner.HiLayers)
    words = [torch.zeros(1, dtype=torch.int64) for _ in range(3)]
    hook.phase_due = list(words)
    hook.split = {6: (1, 6)}
    hook._end_step = lambda *a: calls.append("end")
    hook._ctx_after_target = lambda *a: calls.append("ctx")
    hook.pv = SimpleNamespace(
        launch=lambda h, c, s, r, phases=(): calls.append(("pv", list(phases)))
    )
    monkeypatch.setattr(
        post_verify, "bump_phases", lambda w: calls.append(("bump", list(w)))
    )
    step = SimpleNamespace(max_q_len=6 if verify else 3)
    hook.end_step(None, step, 6)
    assert calls[:2] == ["end", "ctx"] and len(calls) == 3
    kind, got = calls[2]
    assert (
        kind == ("pv" if verify else "bump")
        and all(a is b for a, b in zip(got, words))
        and len(got) == 3
    )
    assert hook.phase_due is None
    # nothing due: no extra launch without a post-verify
    calls.clear()
    hook.phase_due = []
    hook.end_step(None, SimpleNamespace(max_q_len=3), 6)
    assert calls == ["end", "ctx"]


# ---------------------------------------------------------------- the fast entry (hi ``plan.program.FAST_ENTRY_ENV``)


def _step(words=None):
    from atom.models.deepseek_v41.mono.hi.plan.program import STEP_WORDS

    return torch.zeros(STEP_WORDS if words is None else words, dtype=torch.int32)


def test_fast_entry_phase_word_next_to_the_step_counter():
    """A group whose programs read the step phase at a slot of their step tensor (``bind_phase``, at their launch):
    ``end_step`` adds to / hands back that int64 word (the step tensor's ``phase_word(slot)``), not the peer table's
    entry; the counter and the live words are untouched."""
    from atom.models.deepseek_v41.mono.hi.plan.program import phase_word

    g, step = _group(), _step()
    step[:4] = torch.tensor([7, 6, 6, 1], dtype=torch.int32)
    g.bind_phase(step, 2, "p")
    g.bind_phase(step, 2, "p")  # (every launch: the same word)
    g.since_step = 41
    assert g.end_step() is None and g._phase == 1
    w = phase_word(2)
    assert step[w : w + 2].view(torch.int64).item() == 1 and step[:4].tolist() == [
        7,
        6,
        6,
        1,
    ]
    assert g.addresses.tolist() == [100, 200, 0]
    g.since_step = 1
    word = g.end_step(defer=True)
    assert (
        word.dtype == torch.int64
        and word.numel() == 1
        and word.data_ptr() == step[w:].data_ptr()
    )
    assert int(word.item()) == 1  # (the caller bumps it)
    with pytest.raises(ValueError):
        g.bind_phase(step, 1, "p")  # (one word a group)


def test_fast_entry_slots_are_a_groups_own():
    """Two live groups on one step tensor take distinct slots; a retired group's slot can be taken over (cleared);
    a step tensor without the phase words is refused (no out-of-bounds phase read)."""
    a, b, c, step = _group(), _group(), _group(), _step()
    a._retired = b._retired = c._retired = False
    a.bind_phase(step, 0, "a")
    with pytest.raises(ValueError):
        b.bind_phase(step, 0, "b")
    b.bind_phase(step, 1, "b")
    a._retired = True
    step[4:6].view(torch.int64).fill_(5)
    c.bind_phase(step, 0, "c")
    assert step[4:6].view(torch.int64).item() == 0
    with pytest.raises(ValueError):
        _group().bind_phase(_step(4), 0, "short")


def test_fast_entry_plan(monkeypatch):
    """With the env on, a shared TP program takes the fast entry and the composition's slot (keyed); off: neither, its
    key of before."""
    from atom.models.deepseek_v41.mono.hi.plan import program as P
    from atom.models.deepseek_v41.mono.hi.plan.trace import Space

    def make():
        r = P.Region("x", 64, ("s",), Space.PEER)
        return P.Program("t", P.Scenario(1, 1, 2), [P.Stage("s", 4, lambda k, t: None, writes=(r,))],
                         peer_shared=True)  # fmt: skip

    monkeypatch.delenv(P.FAST_ENTRY_ENV, raising=False)
    off = P.plan(make())
    assert not off.fast_entry and off.phase_slot == -1 and P.OWN_ARG not in off.args
    assert not any(str(x).startswith("phase@") for x in off.key())
    monkeypatch.setenv(P.FAST_ENTRY_ENV, "1")
    with P.phase_slot(2):
        on = P.plan(make())
    assert (
        on.fast_entry
        and on.phase_slot == 2
        and P.OWN_ARG in on.args
        and "phase@2" in on.key()
    )
    assert P.plan(make()).phase_slot == 0  # (the default slot outside the context)
    assert P.step_words(1) == P.STEP_WORDS and P.step_words(4) == P.STEP_WORDS
    with pytest.raises(P.PlanError):
        P.phase_word(P.PHASE_SLOTS)
