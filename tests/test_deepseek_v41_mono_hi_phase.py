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
    back un-bumped (the peer table's entry tp, a view the caller bumps); even: nothing, None."""
    g = _group()
    g.since_step = 41  # (a step's launches on the group: 40 layers and the head)
    assert g.end_step() is None and g.addresses.tolist() == [100, 200, 1] and g._phase == 1
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
    verify), else to one ``bump_phases``; the collection closes (None: each group's own add outside a forward)."""
    from atom.models.deepseek_v41.mono import hi_runner
    from atom.models.deepseek_v41.mono.round import post_verify

    calls = []
    hook = object.__new__(hi_runner.HiLayers)
    words = [torch.zeros(1, dtype=torch.int64) for _ in range(3)]
    hook.phase_due = list(words)
    hook.split = {6: (1, 6)}
    hook._end_step = lambda *a: calls.append("end")
    hook._ctx_after_target = lambda *a: calls.append("ctx")
    hook.pv = SimpleNamespace(launch=lambda h, c, s, r, phases=(): calls.append(("pv", list(phases))))
    monkeypatch.setattr(post_verify, "bump_phases", lambda w: calls.append(("bump", list(w))))
    step = SimpleNamespace(max_q_len=6 if verify else 3)
    hook.end_step(None, step, 6)
    assert calls[:2] == ["end", "ctx"] and len(calls) == 3
    kind, got = calls[2]
    assert kind == ("pv" if verify else "bump") and all(a is b for a, b in zip(got, words)) and len(got) == 3
    assert hook.phase_due is None
    # nothing due: no extra launch without a post-verify
    calls.clear()
    hook.phase_due = []
    hook.end_step(None, SimpleNamespace(max_q_len=3), 6)
    assert calls == ["end", "ctx"]
