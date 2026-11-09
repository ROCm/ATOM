# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""A step the Qwen mono programs serve writes its rows' RoPE positions into ATOM's QSA position cache.

The programs never read the cache, but ATOM's QSA pools a group at the position it reads back for the group's first
token, on a later step mono does not serve.
"""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("triton")
pytest.importorskip("aiter")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")

UNWRITTEN = -7


def _context(slots):
    md = SimpleNamespace(qsa_metadata=SimpleNamespace(slot_mapping=slots))
    return SimpleNamespace(attn_metadata=md)


def _cache(pages=3, page=64):
    return torch.full((pages, page, 1, 3), UNWRITTEN, dtype=torch.int64, device="cuda")


def _assert_written(cache, slots, positions):
    axes = (positions.expand(3, -1) if positions.ndim == 1 else positions).cpu()
    want = torch.full(
        (cache.shape[0] * cache.shape[1], 3), UNWRITTEN, dtype=torch.int64
    )
    for row, slot in enumerate(slots.tolist()):
        if slot >= 0:
            want[slot] = axes[:, row]
    assert torch.equal(cache.view(-1, 3).cpu(), want)


@pytest.mark.parametrize("mrope", [False, True])
def test_served_verify_step_writes_target_position_caches(monkeypatch, mrope):
    pytest.importorskip("flydsl")
    from atom.models.qwen38.mono import dispatch
    from atom.models.qwen38.mono import runner as mono_runner

    # 2 requests x SPEC rows, one padded row
    slots = torch.tensor([70, 71, 72, -1, 130, 131, 132, 133], device="cuda")
    positions = torch.arange(1349, 1357, device="cuda")
    if mrope:
        positions = torch.stack((positions, positions + 100, positions + 200))
    monkeypatch.setattr(mono_runner, "get_forward_context", lambda: _context(slots))
    target = mono_runner.TargetRunner.__new__(mono_runner.TargetRunner)
    layers = [SimpleNamespace(rope_position_cache=_cache()) for _ in range(2)]
    target._qsa_layers = {3: layers[0], 7: layers[1]}
    hidden = torch.zeros(8, 1, 1)
    target.step_meta = lambda *args: object()
    target.forward = lambda input_ids, meta: hidden
    model = dispatch.MonoTargetModel(torch.nn.Identity(), target)

    assert model(torch.zeros(8, dtype=torch.long, device="cuda"), positions) is hidden
    for q in layers:
        _assert_written(q.rope_position_cache, slots, positions)


@pytest.mark.parametrize("first_step", [False, True])
def test_served_draft_step_writes_draft_position_cache(monkeypatch, first_step):
    from atom.models.qwen38.mono import draft_fallback

    if first_step:
        # SPEC rows a request (2 requests), the MLP half on each request's selected row
        slots = torch.tensor([60, 61, 62, 63, 180, 181, 182, -1], device="cuda")
        positions = torch.arange(1300, 1308, device="cuda")
        selected_rows = torch.tensor([2, 6], device="cuda")
    else:
        # a later step: one row a request, the third request's row padded
        slots = torch.tensor([66, 185, -1], device="cuda")
        positions = torch.tensor([1301, 1666, 0], device="cuda")
        selected_rows = None
    monkeypatch.setattr(draft_fallback, "get_forward_context", lambda: _context(slots))
    got = (object(), object())
    draft = SimpleNamespace(
        qsa=SimpleNamespace(rope_position_cache=_cache()),
        serve=lambda model, hidden, selected_rows: got,
    )
    monkeypatch.setattr(draft_fallback, "native", lambda *args: pytest.fail("served"))

    hidden = torch.zeros(slots.numel(), 8, device="cuda")
    assert (
        draft_fallback.forward(None, draft, None, positions, hidden, selected_rows)
        is got
    )
    _assert_written(draft.qsa.rope_position_cache, slots, positions)
