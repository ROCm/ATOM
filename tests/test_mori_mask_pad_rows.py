# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""DP pad rows routed to expert -1 before the mori dispatch.

A padded decode step runs `running_tokens` rows and only the leading
`scheduled_tokens` carry a request. mori IntraNode dispatch sends nothing for a
negative expert id, so masking the tail makes padding free -- but only the
tail: a real row masked by mistake comes back from combine as zeros, which is a
silent accuracy loss, not an error. These pin which rows are touched.
"""

from types import SimpleNamespace

import pytest

pytest.importorskip("aiter", reason="needs the AITER GPU kernel library")

import torch

import atom.model_ops.fused_moe.mori_prepare_finalize as mpf
import atom.utils.forward_context as fc

TOPK = 6


def _pf(monkeypatch, *, scheduled, running, capturing=False):
    context = SimpleNamespace(scheduled_tokens=scheduled, running_tokens=running)
    monkeypatch.setattr(
        mpf, "get_forward_context", lambda: SimpleNamespace(context=context)
    )
    monkeypatch.setattr(fc, "_pad_rows_device", None)
    monkeypatch.setattr(fc, "_row_index_device", None)
    fc.enable_pad_rows_device(256, torch.device("cpu"))
    # What `set_forward_context` does for the step, before the capture flips.
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    fc.publish_scheduled_tokens(scheduled)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing)
    pf = mpf.MoriPrepareAndFinalize.__new__(mpf.MoriPrepareAndFinalize)
    # One op: no TBO ubatch slot, so supports_async() is False.
    pf._mori_ops = (object(),)
    pf._mask_pad_rows = True
    return pf


def _ids(rows):
    return torch.arange(rows * TOPK, dtype=torch.int32).reshape(rows, TOPK) % 384


@pytest.mark.parametrize("capturing", [False, True])
def test_only_the_tail_past_scheduled_is_masked(monkeypatch, capturing):
    pf = _pf(monkeypatch, scheduled=12 * 7, running=16 * 7, capturing=capturing)
    ids = _ids(16 * 7)
    out = pf.mask_pad_topk_ids(ids)
    assert out.dtype == torch.int32
    assert torch.equal(out[: 12 * 7], ids[: 12 * 7])
    assert (out[12 * 7 :] == -1).all()
    # The router's own tensor is not written.
    assert torch.equal(ids, _ids(16 * 7))


def test_the_capture_reads_the_device_count_not_the_host_one(monkeypatch):
    """A recording must follow what is published at replay, not what the
    capture context said -- which is always "every row is real"."""
    pf = _pf(monkeypatch, scheduled=32, running=32, capturing=True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    fc.publish_scheduled_tokens(20)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    out = pf.mask_pad_topk_ids(_ids(32))
    assert (out[:20] >= 0).all() and (out[20:] == -1).all()


def test_an_unpadded_step_is_returned_as_is(monkeypatch):
    pf = _pf(monkeypatch, scheduled=40, running=40)
    ids = _ids(40)
    assert pf.mask_pad_topk_ids(ids) is ids


@pytest.mark.parametrize("rows", [7, 48])
def test_rows_that_are_not_the_step_width_are_left_alone(monkeypatch, rows):
    """Some other row set (a PCP shard, a sub-slice): it has no padded tail."""
    pf = _pf(monkeypatch, scheduled=5, running=32)
    ids = _ids(rows)
    assert pf.mask_pad_topk_ids(ids) is ids


def test_tbo_ubatches_are_left_alone(monkeypatch):
    pf = _pf(monkeypatch, scheduled=5, running=32)
    monkeypatch.setattr(pf, "supports_async", lambda: True)
    ids = _ids(32)
    assert pf.mask_pad_topk_ids(ids) is ids


def test_disabled_is_identity(monkeypatch):
    pf = _pf(monkeypatch, scheduled=5, running=32)
    pf._mask_pad_rows = False
    ids = _ids(32)
    assert pf.mask_pad_topk_ids(ids) is ids


def test_rows_past_the_published_mask_are_left_alone(monkeypatch):
    pf = _pf(monkeypatch, scheduled=5, running=512)
    ids = _ids(512)
    assert pf.mask_pad_topk_ids(ids) is ids


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
