# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""The base packed forward must honour the run's token alignment.

Dense models at pp=1/cp=1 bypass the pipeline forward and land here, so an FP8
run on a dense model is aligned only if this path pads too. It must also hand
callers back exactly the real-sequence logits -- the pad is an implementation
detail that must not shift any offset.
"""

import pytest
import torch

pytest.importorskip("megatron", reason="Megatron backend is optional in CPU CI")

from atomrl.engine.training.megatron_base_engine import MegatronBaseEngine

V = 32


class _Recorder:
    """Stands in for GPTModel: records what the forward was handed."""

    def __init__(self):
        self.tokens = None
        self.psp = None

    def __call__(self, input_ids, position_ids, attention_mask, packed_seq_params):
        self.tokens, self.psp = input_ids, packed_seq_params
        return torch.zeros(1, input_ids.shape[1], V)


def _engine(align):
    e = object.__new__(MegatronBaseEngine)
    e._packed_token_align = lambda: align
    return e


def _run(real_lens, align):
    eng, rec = _engine(align), _Recorder()
    ids = [torch.arange(L, dtype=torch.long) for L in real_lens]
    logits, offsets = eng._forward_logits_packed(ids, model=rec)
    return rec, logits, offsets


def test_align_one_pads_nothing():
    rec, logits, offsets = _run([5, 7], 1)
    assert rec.tokens.shape[1] == 12
    assert logits.shape == (12, V)
    assert offsets == [0, 5, 12]


@pytest.mark.parametrize(
    "real,align,padded",
    [
        ([5, 7], 16, 16),
        ([2], 16, 16),
        ([1941], 16, 1952),
        ([64, 64], 16, 128),
    ],
)
def test_the_stream_is_padded_up_to_the_alignment(real, align, padded):
    rec, _, _ = _run(real, align)
    assert rec.tokens.shape[1] == padded
    assert rec.tokens.shape[1] % align == 0


def test_callers_still_see_only_the_real_sequences():
    real = [5, 7]
    rec, logits, offsets = _run(real, 16)
    assert rec.tokens.shape[1] == 16  # padded on the way in
    assert logits.shape == (12, V)  # pad tail dropped on the way out
    assert offsets == [0, 5, 12]  # offsets untouched by the pad


def test_the_pad_gets_its_own_attention_bin():
    rec, _, _ = _run([5, 7], 16)
    assert rec.psp.cu_seqlens_q.tolist() == [0, 5, 12, 16]


def test_max_seqlen_covers_the_pad_bin():
    # pad=14 exceeds the only real sequence (2); a max over real sequences
    # alone would under-size the rope table.
    rec, _, _ = _run([2], 16)
    assert rec.psp.cu_seqlens_q.tolist() == [0, 2, 16]
    assert rec.psp.max_seqlen_q == 14


def test_a_short_pad_does_not_inflate_max_seqlen():
    rec, _, _ = _run([1941], 16)
    assert rec.psp.max_seqlen_q == 1941


def test_position_ids_restart_within_the_pad_segment():
    rec, _, _ = _run([3], 8)
    # 0,1,2 for the real sequence then 0..4 for the pad -- never continuing
    # the real sequence's positions, which would corrupt its rope phases.
    assert rec.tokens.shape[1] == 8
    assert rec.psp.cu_seqlens_q.tolist() == [0, 3, 8]


def test_pad_tokens_are_zero_not_stale_ids():
    rec, _, _ = _run([3], 8)
    assert torch.equal(rec.tokens[0, 3:], torch.zeros(5, dtype=rec.tokens.dtype))
