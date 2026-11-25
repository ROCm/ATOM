# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""A deferred DSpark decode's device verdict and early output.

* The Aiter builder takes the previous verdict on the device only for its own
  decode metadata on an eligible verify step; the shift it runs gives the
  metadata the host builds from the verdict (GPU).
* Early output takes the previous step's ids, logprobs and drafts from the same
  FIFO entries, in the same order, as the late path."""

from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch
from aiter_stub import atom_without_aiter


@pytest.fixture(scope="module")
def m():
    with atom_without_aiter() as load:
        yield SimpleNamespace(
            runner=load("atom.model_engine.model_runner"),
            aiter_attention=load("atom.model_ops.attentions.aiter_attention"),
        )


@pytest.mark.parametrize(
    "q,eligible,mrope,own_stage,want",
    [
        (8, True, False, False, True),
        (1, True, False, False, False),  # no verify window
        (8, False, False, False, False),  # the runner's side says no
        (8, True, True, False, False),  # M-RoPE positions
        (8, True, False, True, False),  # a builder with its own decode metadata
    ],
)
def test_the_builder_consumes_the_device_verdict_only_for_its_own_metadata(
    m, q, eligible, mrope, own_stage, want
):
    base = m.aiter_attention.AiterAttentionMetadataBuilder
    cls = type("Own", (base,), {"stage_decode": lambda *a: None}) if own_stage else base
    builder = object.__new__(cls)
    builder.model_runner = SimpleNamespace(
        device_verdict_eligible=eligible, use_mrope=mrope
    )
    assert builder.consumes_device_verdict(None, q) is want


class _FakeGpuTensor:
    def __init__(self, values):
        self.values = values

    def to(self, device, non_blocking=False):
        return self.values.clone()


@pytest.fixture
def processor(monkeypatch, m):
    original_empty = torch.empty
    monkeypatch.setattr(
        torch, "empty", lambda *a, pin_memory=False, **k: original_empty(*a, **k)
    )
    monkeypatch.setattr(torch.cuda, "Event", mock.Mock)
    monkeypatch.setattr(torch.cuda, "current_stream", mock.Mock)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: mock.MagicMock())
    p = object.__new__(m.runner.tokenIDProcessor)
    p.async_copy_stream = mock.Mock()
    p._output_request_capacity = 4
    p.num_spec_tokens = 2
    p.is_deferred_out = True
    p.clean()
    return p


def _step(p, batch, ids, drafts, early):
    """One deferred step's report: (ids by request, drafts), early or late."""
    if early:
        got = p.take_deferred_output()
    report = p.prepare_sampled_ids(batch, _FakeGpuTensor(ids), mock.Mock())
    got_drafts = p.prepare_draft_ids(batch, drafts)
    if early:
        assert (report[0], got_drafts) == (got[0], got[2])
        assert p._early_taken is None
    return report[0], got_drafts.tolist()


@pytest.mark.parametrize("early", [False, True])
def test_early_output_reports_what_the_late_path_does(processor, early):
    b1 = SimpleNamespace(req_ids=[5, 6], return_logprobs=None)
    b2 = SimpleNamespace(req_ids=[5, 6], return_logprobs=None)
    b3 = SimpleNamespace(req_ids=[6], return_logprobs=None)
    d1 = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
    d2 = torch.tensor([[7, 8], [9, 10]], dtype=torch.int32)
    d3 = torch.tensor([[11, 12]], dtype=torch.int32)
    assert _step(processor, b1, torch.tensor([30, 40]), d1, False) == ({}, [])
    # step 2 reports step 1's ids and drafts; step 3 step 2's
    ids, drafts = _step(processor, b2, torch.tensor([31, 41]), d2, early)
    assert ids == {5: (30,), 6: (40,)} and drafts == [[1, 2], [3, 4]]
    ids, drafts = _step(processor, b3, torch.tensor([42]), d3, early)
    assert ids == {5: (31,), 6: (41,)} and drafts == [[7, 8], [9, 10]]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_the_shift_gives_the_metadata_the_host_builds_from_the_verdict():
    from atom.model_ops.attentions.token_layout.batch_ids import build_cu_seqlens
    from atom.model_ops.attentions.token_layout.decode import decode_positions
    from atom.model_ops.attentions.token_layout.slots import slot_mapping
    from atom.model_ops.spec_decode_shift import shift_decode_metadata

    q, page, kv_block, running_bs, W = 8, 16, 16, 5, 128
    reserved = np.array([300, 17, 1030, 40], dtype=np.int32)  # 4 scheduled rows
    src = np.array([2, -1, 0, 1], dtype=np.int32)  # previous-batch rows
    prev_rejected = np.array([3, 7, 5], dtype=np.int32)
    r = np.where(src >= 0, prev_rejected[np.maximum(src, 0)], 0).astype(np.int32)
    bs = len(reserved)
    lengths = np.full(bs, q, dtype=np.int32)
    cu = np.zeros(running_bs + 1, dtype=np.int32)
    build_cu_seqlens(lengths, running_bs, out=cu)
    tables = np.random.default_rng(0).integers(0, 9000, (running_bs, W)).astype(np.int32)

    def host(ctx):
        pos = decode_positions(ctx, lengths, cu)
        slots = slot_mapping(pos, lengths, tables, page)
        indptr = np.zeros(running_bs + 1, dtype=np.int32)
        build_cu_seqlens(-(-ctx // kv_block), running_bs, out=indptr)
        return ctx, pos, slots, indptr

    want = host(reserved - r)
    ctx, pos, slots, indptr = (torch.from_numpy(np.array(x)).cuda() for x in host(reserved))
    shift_decode_metadata(
        decode_src=torch.from_numpy(src).cuda(),
        rejected=torch.from_numpy(prev_rejected).cuda(),
        cu_seqlens_q=torch.from_numpy(cu).cuda(),
        block_tables=torch.from_numpy(tables).cuda(),
        context_lens=ctx,
        positions=pos,
        slot_mapping=slots,
        kv_indptr=indptr,
        bs=bs,
        running_bs=running_bs,
        max_q=q,
        page_size=page,
        kv_block_size=kv_block,
    )
    for name, a, b in zip(("ctx", "positions", "slots", "kv_indptr"), want, (ctx, pos, slots, indptr)):
        assert np.array_equal(np.asarray(a), b.cpu().numpy()), name
