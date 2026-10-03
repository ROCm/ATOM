# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""The device DP pad-row mask in atom/utils/forward_context.py.

A padded decode step runs `running_tokens` rows and only the leading
`scheduled_tokens` carry a request. A real row selected as padding comes back
from the MoE as zeros, which is a silent accuracy loss, not an error. These pin
which rows `step_pad_rows` selects, eager and while capturing. CPU only.
"""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

import atom.utils.forward_context as fc


@pytest.fixture
def step(monkeypatch):
    monkeypatch.setattr(fc, "_pad_rows_device", None)
    monkeypatch.setattr(fc, "_row_index_device", None)
    fc.enable_pad_rows_device(256, torch.device("cpu"))

    def _step(*, scheduled, running, capturing=False):
        context = SimpleNamespace(scheduled_tokens=scheduled, running_tokens=running)
        monkeypatch.setattr(
            fc, "get_forward_context", lambda: SimpleNamespace(context=context)
        )
        # What `set_forward_context` does for the step, before the capture flips.
        monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
        fc.publish_scheduled_tokens(scheduled)
        monkeypatch.setattr(
            torch.cuda, "is_current_stream_capturing", lambda: capturing
        )

    return _step


@pytest.mark.parametrize("capturing", [False, True])
def test_only_the_tail_past_scheduled_is_selected(step, capturing):
    step(scheduled=12 * 7, running=16 * 7, capturing=capturing)
    pad = fc.step_pad_rows(16 * 7)
    assert pad.shape == (16 * 7, 1)
    assert not pad[: 12 * 7].any() and pad[12 * 7 :].all()


def test_the_capture_reads_the_device_mask_not_the_host_count(step, monkeypatch):
    """A recording must follow what is published at replay, not what the
    capture context said -- which is always "every row is real"."""
    step(scheduled=32, running=32, capturing=True)
    pad = fc.step_pad_rows(32)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    fc.publish_scheduled_tokens(20)
    assert not pad[:20].any() and pad[20:].all()


def test_an_unpadded_eager_step_selects_nothing(step):
    step(scheduled=40, running=40)
    assert fc.step_pad_rows(40) is None


@pytest.mark.parametrize("rows", [7, 48])
def test_rows_that_are_not_the_step_width_are_left_alone(step, rows):
    """Some other row set (a TBO ubatch, a PCP shard): it has no padded tail."""
    step(scheduled=5, running=32)
    assert fc.step_pad_rows(rows) is None


def test_rows_past_the_published_mask_are_left_alone(step):
    step(scheduled=5, running=512)
    assert fc.step_pad_rows(512) is None


def test_publish_is_a_no_op_until_enabled_and_skipped_while_capturing(monkeypatch):
    monkeypatch.setattr(fc, "_pad_rows_device", None)
    monkeypatch.setattr(fc, "_row_index_device", None)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    fc.publish_scheduled_tokens(7)  # nothing allocated, nothing written
    assert fc.get_pad_rows_device() is None
    fc.enable_pad_rows_device(128, torch.device("cpu"))
    buf = fc.get_pad_rows_device()
    assert not buf.any()  # every row real until the first publish
    fc.enable_pad_rows_device(64, torch.device("cpu"))
    assert fc.get_pad_rows_device() is buf  # never shrunk or replaced
    fc.publish_scheduled_tokens(84)
    assert not buf[:84].any() and buf[84:].all()
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    fc.publish_scheduled_tokens(3)
    assert not buf[:84].any() and buf[84:].all()
