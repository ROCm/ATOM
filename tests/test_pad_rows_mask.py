# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Which rows `step_pad_rows` selects as DP padding: a target pass from its
`cu_seqlens_q`, a draft pass from the request count the target recorded.

A real row selected as padding comes back from the MoE as zeros: a silent
accuracy loss, not an error. CPU only.
"""

from types import SimpleNamespace

import pytest
import torch

import atom.utils.forward_context as fc
from atom.utils.tbo import ubatching


def _cu(lengths, running_bs):
    """`cu_seqlens_q` as the builder publishes it: padded requests have length 0."""
    cu = torch.zeros(running_bs + 1, dtype=torch.int32)
    cu[1 : len(lengths) + 1] = torch.tensor(lengths, dtype=torch.int32).cumsum(0)
    cu[len(lengths) + 1 :] = cu[len(lengths)]
    return cu


@pytest.fixture
def steps(monkeypatch):
    monkeypatch.setattr(fc, "_row_index_device", None)
    monkeypatch.setattr(fc, "_pad_rows_device", None)
    monkeypatch.setattr(fc, "_real_requests_device", None)
    monkeypatch.setattr(ubatching, "tbo_active", lambda: False)
    fc.enable_pad_rows_device(256, torch.device("cpu"))

    def capturing(flag):
        monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: flag)

    def target(lengths, running_bs, running, *, capture=False, cu=True):
        """A target forward: a fresh forward context, as `set_forward_context`."""
        context = SimpleNamespace(
            is_draft=False,
            scheduled_tokens=sum(lengths),
            running_tokens=running,
            running_bs=running_bs,
        )
        fwd = SimpleNamespace(
            context=context,
            attn_metadata=SimpleNamespace(
                cu_seqlens_q=_cu(lengths, running_bs) if cu else None
            ),
            pad_rows_done=set(),
        )
        monkeypatch.setattr(fc, "get_forward_context", lambda: fwd)
        capturing(capture)
        return fwd

    def draft(fwd, scheduled, running, *, capture=False):
        """A draft pass on the target's context, as `_publish_draft_shape`."""
        fwd.context.is_draft = True
        fwd.context.scheduled_tokens = scheduled
        fwd.context.running_tokens = running
        capturing(capture)

    return SimpleNamespace(target=target, draft=draft, capturing=capturing)


def _tail(mask, real):
    return not mask[:real].any() and mask[real:].all()


@pytest.mark.parametrize("capture", [False, True])
def test_target_rows_past_the_real_tokens_are_padding(steps, capture):
    steps.target([8] * 5, running_bs=7, running=56, capture=capture)
    assert _tail(fc.step_pad_rows(56), 40)


def test_ragged_target_uses_the_real_token_count(steps):
    steps.target([3, 8, 1], running_bs=4, running=16)
    assert _tail(fc.step_pad_rows(16), 12)


@pytest.mark.parametrize("q", [4, 1])  # DSpark [bs, T] block, Eagle mid-step
def test_draft_rows_past_the_recorded_requests_are_padding(steps, q):
    fwd = steps.target([8] * 5, running_bs=7, running=56)
    fc.step_pad_rows(56)
    steps.draft(fwd, scheduled=5 * q, running=7 * q)
    assert _tail(fc.step_pad_rows(7 * q), 5 * q)


def test_unpadded_target_still_records_requests_for_its_drafts(steps):
    # A ragged verify can fill its recorded width while the draft is widened.
    fwd = steps.target([3, 5], running_bs=4, running=8)
    assert fc.step_pad_rows(8) is None
    steps.draft(fwd, scheduled=4, running=8)
    assert _tail(fc.step_pad_rows(8), 4)


@pytest.mark.parametrize("rows", [7, 48])
def test_rows_that_are_not_the_pass_width_are_left_alone(steps, rows):
    """Some other row set (a PCP shard, a sub-slice): it has no padded tail."""
    fwd = steps.target([1] * 5, running_bs=32, running=32)
    assert fc.step_pad_rows(rows) is None
    steps.draft(fwd, scheduled=5, running=32)
    assert _tail(fc.step_pad_rows(32), 5)  # the requests were still recorded


@pytest.mark.parametrize("cause", ["tbo", "no_cu"])
def test_a_target_that_cannot_tell_records_no_padding(steps, monkeypatch, cause):
    fwd = steps.target([8] * 5, running_bs=7, running=56, cu=cause != "no_cu")
    if cause == "tbo":
        monkeypatch.setattr(ubatching, "tbo_active", lambda: True)
    assert fc.step_pad_rows(56) is None
    monkeypatch.setattr(ubatching, "tbo_active", lambda: False)
    steps.draft(fwd, scheduled=20, running=28, capture=True)
    assert not fc.step_pad_rows(28).any()


def test_a_pass_derives_its_mask_once(steps):
    fwd = steps.target([8] * 5, running_bs=7, running=56)
    mask = fc.step_pad_rows(56)
    fwd.attn_metadata.cu_seqlens_q.fill_(56)  # a later layer recomputes nothing
    again = fc.step_pad_rows(56)
    assert again.data_ptr() == mask.data_ptr() and _tail(again, 40)


def test_capture_records_even_after_an_eager_warmup(steps):
    fwd = steps.target([8] * 5, running_bs=7, running=56)
    assert fc.step_pad_rows(56) is not None
    steps.capturing(True)
    fc.step_pad_rows(56)
    assert ("requests", True) in fwd.pad_rows_done
    assert ("mask", False, 56, True) in fwd.pad_rows_done


def test_nothing_is_selected_until_enabled(monkeypatch):
    monkeypatch.setattr(fc, "_pad_rows_device", None)
    monkeypatch.setattr(
        fc,
        "get_forward_context",
        lambda: SimpleNamespace(context=SimpleNamespace(), pad_rows_done=set()),
    )
    assert fc.step_pad_rows(8) is None


def test_enabling_never_shrinks_or_replaces_the_buffers(steps):
    buf = fc._pad_rows_device
    fc.enable_pad_rows_device(64, torch.device("cpu"))
    assert fc._pad_rows_device is buf and buf.shape == (256, 1)
