# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""The presence-penalty kernels (`triton_presence_penalty.py`) against the
plain rule (`presence_penalty_reference.expected_logits`), step by step."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("triton", reason="the kernels are Triton")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="ROCm GPU required"
)

from presence_penalty_reference import bit_rows, expected_logits

from atom.model_ops.presence_penalty import PresencePenalties
from atom.model_ops.triton_presence_penalty import (
    mark_emitted,
    penalize_rows,
)

VOCAB = 248320  # Qwen3.5's; ids 31, 32 and VOCAB - 1 sit on word edges


def _batch(req_ids, penalties, final=None, resumed=None):
    return SimpleNamespace(
        total_seqs_num=len(req_ids),
        req_ids=list(req_ids),
        presence_penalties=np.asarray(penalties, dtype=np.float32),
        presence_resumed=resumed or {},
        is_final_chunk=final,
    )


def _stage(table, batch, lengths=None):
    host = np.zeros(PresencePenalties.staging_size(8, 5), dtype=np.int32)
    buffer = SimpleNamespace(
        np=host, gpu=torch.zeros(host.size, dtype=torch.int32, device="cuda")
    )
    rows, count = table.stage(batch, lengths, buffer)
    buffer.gpu[:count].copy_(torch.from_numpy(host[:count]))
    return rows


def _step(table, rows, logits, input_ids=None, max_drafts=0):
    penalize_rows(table.seen, rows, logits, input_ids, max_drafts)


def _logits(rows, dtype):
    return torch.randn(rows, VOCAB, device="cuda").to(dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_kernels_follow_the_plain_rule(dtype):
    torch.manual_seed(0)
    table = PresencePenalties(VOCAB, 2, "cuda")
    ids = [10, 11, 12]
    pens = [1.5, 1.0, -0.75]
    generated = {10: [], 11: [], 12: []}

    # Prefill: request 11 runs a middle chunk; it claims no row, and its
    # sampled id is not output.
    batch = _batch(ids, pens, final=[True, False, True])
    rows = _stage(table, batch)
    logits = _logits(3, dtype)
    raw = logits.clone()
    _step(table, rows, logits)
    torch.testing.assert_close(logits, raw, rtol=0, atol=0)
    mark_emitted(
        table.seen, rows, torch.tensor([31, 7, VOCAB - 1], device="cuda"), VOCAB
    )
    generated[10] += [31]
    generated[12] += [VOCAB - 1]

    # MTP-4 verify: request 10 drafts [63, 63, 32, 100] (a repeat, then 32 on
    # a word edge); 11 has one row; 12 drafts [0, VOCAB - 1] (already output).
    drafts = {10: [63, 63, 32, 100], 11: [], 12: [0, VOCAB - 1]}
    anchors = {10: 31, 11: 7, 12: VOCAB - 1}
    lengths = np.array([len(drafts[r]) + 1 for r in ids])
    batch = _batch(ids, pens)
    rows = _stage(table, batch, lengths)
    input_ids = torch.tensor(
        [t for r in ids for t in [anchors[r], *drafts[r]]], device="cuda"
    )
    row_request = [i for i, r in enumerate(ids) for _ in range(len(drafts[r]) + 1)]
    row_drafts = [drafts[r][:j] for r in ids for j in range(len(drafts[r]) + 1)]
    logits = _logits(int(lengths.sum()), dtype)
    logits[0, 31] = -float("inf")  # a masked id stays masked
    expected = expected_logits(
        logits, [generated[r] for r in ids], row_request, row_drafts, pens
    )
    _step(table, rows, logits, input_ids, max_drafts=4)
    torch.testing.assert_close(logits, expected, rtol=0, atol=0)

    # 10 accepts 63, 63 and rejects 32 (bonus 40); 11 emits 9; 12 accepts 0.
    emitted = torch.tensor(
        [[63, 63, 40, -1, -1], [9, -1, -1, -1, -1], [0, 5, -1, -1, -1]],
        device="cuda",
    )
    mark_emitted(table.seen, rows, emitted, VOCAB)
    generated[10] += [63, 40]
    generated[11] += [9]
    generated[12] += [0, 5]
    bits = bit_rows(table.seen)
    for i, r in enumerate(ids):
        slot = int(rows.emit_slots[i])
        assert bits[slot].nonzero()[0].tolist() == sorted(set(generated[r]))

    # Decode: the rejected drafts 32 and 100 are not request 10's output.
    rows = _stage(table, _batch(ids, pens))
    logits = _logits(3, dtype)
    expected = expected_logits(
        logits, [generated[r] for r in ids], [0, 1, 2], [[], [], []], pens
    )
    _step(table, rows, logits)
    torch.testing.assert_close(logits, expected, rtol=0, atol=0)

    # 10 finishes and 12 is preempted (their rows go back); a new request
    # takes 10's row and starts with nothing; 12 resumes on another row with
    # its output marked again.
    released = {int(rows.emit_slots[0]), int(rows.emit_slots[2])}
    table.release([10, 12])
    resumed = {12: np.asarray(generated[12], dtype=np.int32)}
    rows = _stage(table, _batch([13, 11, 12], [2.0, 1.0, -0.75], None, resumed))
    assert {int(rows.emit_slots[0]), int(rows.emit_slots[2])} == released
    logits = _logits(3, dtype)
    expected = expected_logits(
        logits,
        [[], generated[11], generated[12]],
        [0, 1, 2],
        [[], [], []],
        [2.0, 1.0, -0.75],
    )
    _step(table, rows, logits)
    torch.testing.assert_close(logits, expected, rtol=0, atol=0)
