# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Speculative decode (DSpark) inside a mixed prefill+decode batch.

A mixed batch is `[prefill rows | decode rows]`. Under spec, a prefill row
verifies as a 0-draft span sampling one logit row, and a decode row as its
usual anchor + drafts; everything downstream of the LM head indexes the batch
by request. See docs/mixed_dspark_design.md for the design these pin.

Layout used throughout: two prefill rows of 5 and 3 tokens, then two decode
rows of anchor + 3 drafts (mtp_k = 3):

    token rows  [0..4 | 5..7 | 8..11 | 12..15]        16 tokens
    logit rows  [0    | 1    | 2..5  | 6..9  ]        10 rows (LM head gather)
"""

import importlib.util
import types
from types import SimpleNamespace

import numpy as np
import pytest
import torch

N_P, N_D, K = 2, 2, 3
PREFILL_LENS = [5, 3]
N_P_TOKENS = sum(PREFILL_LENS)
NUM_SCHEDULED = np.array(PREFILL_LENS + [K + 1] * N_D, dtype=np.int32)


def _mixed_batch(**overrides):
    fields = {
        "is_mixed": True,
        "is_dummy_run": False,
        "total_seqs_num_prefill": N_P,
        "total_seqs_num_decode": N_D,
        "total_seqs_num": N_P + N_D,
        "total_tokens_num_prefill": N_P_TOKENS,
        "num_scheduled_tokens": NUM_SCHEDULED.copy(),
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


# ── verify layout in logit-row space ────────────────────────────────────────


def test_verify_spans_put_prefill_rows_at_one_sample_each():
    from atom.model_engine.model_runner import ModelRunner

    runner = SimpleNamespace(drafter=object())
    lens, cu_end, shift = ModelRunner._verify_spans(runner, _mixed_batch())
    assert lens.tolist() == [1, 1, 4, 4]
    assert cu_end.tolist() == [1, 2, 6, 10]  # ends at the 10 logit rows
    assert shift == N_P_TOKENS - N_P


def test_verify_spans_skip_pure_prefill_and_dummies():
    from atom.model_engine.model_runner import ModelRunner

    runner = SimpleNamespace(drafter=object())
    assert ModelRunner._verify_spans(runner, _mixed_batch(is_mixed=False)) is None
    assert ModelRunner._verify_spans(runner, _mixed_batch(is_dummy_run=True)) is None
    assert ModelRunner._verify_spans(SimpleNamespace(), _mixed_batch()) is None


def _index_runner(cap=64):
    from atom.model_engine.model_runner import ModelRunner

    runner = SimpleNamespace(
        arange_np=np.arange(cap, dtype=np.int32),
        forward_vars={
            name: SimpleNamespace(np=np.full(cap, -1, dtype=np.int32))
            for name in (
                "target_logits_indices",
                "cu_num_draft_tokens",
                "bonus_logits_indices",
            )
        },
    )
    runner._get_cumsum_and_arange = types.MethodType(
        ModelRunner._get_cumsum_and_arange, runner
    )
    return runner


def test_spec_indices_and_draft_gather_over_the_mixed_layout():
    """The real index builder, fed the mixed spans, then the draft-id gather."""
    from atom.model_engine.model_runner import ModelRunner
    from atom.spec_decode.drafter import Drafter

    lens, cu_end, shift = ModelRunner._verify_spans(
        SimpleNamespace(drafter=object()), _mixed_batch()
    )
    runner = _index_runner()
    drafter = SimpleNamespace(mtp_k=K, runner=runner)
    names = ["target_logits_indices", "cu_num_draft_tokens", "bonus_logits_indices"]
    group = SimpleNamespace(
        counts=[None] * 3, indices={n: i for i, n in enumerate(names)}
    )
    num_draft, total = Drafter.prepare_spec_decode_indices(drafter, lens, cu_end, group)
    var = runner.forward_vars
    assert num_draft.tolist() == [0, 0, 3, 3]
    assert total == N_D * K
    # Each decode row's first three logit rows are scored; prefill rows none.
    target = var["target_logits_indices"].np[:total]
    assert target.tolist() == [2, 3, 4, 6, 7, 8]
    # One bonus per request: a prefill row's single row, a decode row's last.
    assert var["bonus_logits_indices"].np[: N_P + N_D].tolist() == [0, 1, 5, 9]
    assert var["cu_num_draft_tokens"].np[: N_P + N_D].tolist() == [0, 0, 3, 6]

    # `calc_spec_decode_metadata` gathers `input_ids[1:][target]` from what it
    # is handed; the runner hands it `input_ids[shift:]`. The drafts are the
    # tokens after each decode row's anchor: 9..11 and 13..15.
    input_ids = torch.arange(N_P_TOKENS + N_D * (K + 1))
    drafts = input_ids[shift:][1:][torch.from_numpy(target).long()]
    assert drafts.tolist() == [9, 10, 11, 13, 14, 15]


# ── verify outcome: the verdict, re-expressed in token rows ─────────────────


@pytest.fixture
def mixed_forward_context():
    from atom.utils import forward_context as fc_mod

    # The mixed carrier's whole-batch spans (`_mixed_carrier_spans`).
    global_cu = torch.tensor([0, 5, 8, 12, 16], dtype=torch.int32)
    ctx = SimpleNamespace(
        attn_metadata=SimpleNamespace(cu_seqlens_q=global_cu),
        context=SimpleNamespace(is_mixed=True, num_prefill_seqs=N_P),
    )
    prev = getattr(fc_mod._forward_context_local, "ctx", None)
    fc_mod._forward_context_local.ctx = ctx
    try:
        yield ctx
    finally:
        fc_mod._forward_context_local.ctx = prev


def _logit_verdict(accepted):
    """What the rejection kernel writes, in LOGIT rows: rejected is
    `K - accepted`, anchor is `bonus_row - num_draft + accepted` (see
    `_finish_request`). Bonus rows and draft counts are the mixed layout's."""
    bonus_rows = [0, 1, 5, 9]
    num_draft = [0, 0, K, K]
    reject = torch.tensor([K - a for a in accepted], dtype=torch.int32)
    anchors = torch.tensor(
        [b - d + a for b, d, a in zip(bonus_rows, num_draft, accepted)],
        dtype=torch.int32,
    )
    return reject, anchors


def test_mixed_verdict_moves_to_token_rows(mixed_forward_context):
    """Prefill rows reject nothing and anchor on their chunk's last token;
    decode rows shift by `n_p_tokens - n_p`. Left at `mtp_k`, a prefill row's
    reject count rolls the request's ctx back K the moment it decodes."""
    from atom.model_engine.model_runner import ModelRunner

    reject, anchors = _logit_verdict([0, 0, 1, 3])
    ModelRunner._settle_mixed_verdict(None, _mixed_batch(), reject, anchors)
    assert reject.tolist() == [0, 0, 2, 0]
    # prefill: last token of each chunk; decode: segment start + accepted.
    assert anchors.tolist() == [4, 7, 9, 15]


def test_pure_decode_verdict_is_left_alone(mixed_forward_context):
    from atom.model_engine.model_runner import ModelRunner

    reject, anchors = _logit_verdict([0, 0, 1, 3])
    before = (reject.clone(), anchors.clone())
    ModelRunner._settle_mixed_verdict(
        None, _mixed_batch(is_mixed=False), reject, anchors
    )
    assert torch.equal(reject, before[0]) and torch.equal(anchors, before[1])


def test_unverified_step_anchors_every_row_on_its_last(mixed_forward_context):
    """With no verdict the drafter anchors on each span's last row, which over
    the carrier's whole-batch spans is right for prefill and decode rows alike."""
    from atom.spec_decode.drafter import Drafter

    anchors = Drafter.prepare_inputs(None, N_P + N_D)
    assert anchors.tolist() == [4, 7, 11, 15]


# ── DSpark length shrink stays off on mixed steps ───────────────────────────


def test_mixed_step_is_never_shrunk_even_with_unchanged_composition():
    """Two mixed steps can carry identical req_ids (a chunked prefill beside
    the same decodes), so "the composition changed" does not keep ragged off.
    The prefill guard at the top of `_dspark_apply_q_bucket` does."""
    from atom.model_engine.model_runner import ModelRunner

    runner = SimpleNamespace(
        drafter=SimpleNamespace(uses_confidence_schedule=True, mtp_k=K)
    )
    batch = _mixed_batch(req_ids=[1, 2, 3, 4])
    assert ModelRunner._dspark_apply_q_bucket(runner, batch) is None
    assert batch.num_scheduled_tokens.tolist() == NUM_SCHEDULED.tolist()


# ── V4 builder pieces (deepseek_v4_attn imports aiter at module scope) ──────

requires_aiter = pytest.mark.skipif(
    importlib.util.find_spec("aiter") is None,
    reason="deepseek_v4_attn imports aiter at module scope; CI has none",
)


@requires_aiter
def test_carrier_spans_cover_the_whole_batch():
    from atom.model_ops.attentions.deepseek_v4_attn import _mixed_carrier_spans

    prefill_meta = SimpleNamespace(
        cu_seqlens_q=torch.tensor([0, 5, 8], dtype=torch.int32),
        state_slot_out=torch.tensor([11, 12], dtype=torch.int32),
    )
    decode_meta = SimpleNamespace(
        cu_seqlens_q=torch.tensor([0, 4, 8], dtype=torch.int32),
        state_slot_out=torch.tensor([21, 22], dtype=torch.int32),
    )
    running_bs = 6  # a DP ladder rung wider than this rank's 4 requests
    cu, slots = _mixed_carrier_spans(
        prefill_meta, decode_meta, N_P, N_D, N_P_TOKENS, running_bs
    )
    assert cu.tolist() == [0, 5, 8, 12, 16]
    assert cu[: N_P + 1].tolist() == prefill_meta.cu_seqlens_q.tolist()
    assert slots.tolist() == [11, 12, 21, 22, 0, 0]


@requires_aiter
def test_decode_view_slices_spec_fields_and_says_where_its_rows_start():
    from atom.model_ops.attentions.deepseek_v4_attn import _MixedDecodeView

    n = N_P + N_D
    batch = SimpleNamespace(
        total_seqs_num=n,
        total_seqs_num_decode=N_D,
        total_tokens_num_decode=N_D * (K + 1),
        total_tokens_num_prefill=N_P_TOKENS,
        is_dummy_run=False,
        num_spec_step=K,
        context_lens=np.arange(n),
        block_tables=[[i] for i in range(n)],
        state_slots_committed=list(range(n)),
        num_scheduled_tokens=NUM_SCHEDULED.copy(),
        num_cached_tokens=np.zeros(n, dtype=np.int32),
        last_block_num_tokens=np.ones(n, dtype=np.int32),
        scheduled_spec_decode_tokens=np.arange(n * K).reshape(n, K),
        num_rejected=np.array([9, 9, 1, 2], dtype=np.int32),
        num_bonus=np.array([9, 9, 2, 1], dtype=np.int32),
        scheduled_tokens=np.arange(N_P_TOKENS + N_D * (K + 1)),
    )
    view = _MixedDecodeView(batch, N_P)
    assert view.row_offset == N_P
    assert view.num_rejected.tolist() == [1, 2]
    assert view.num_bonus.tolist() == [2, 1]
    assert view.scheduled_spec_decode_tokens.tolist() == [[6, 7, 8], [9, 10, 11]]
    # Token-axis: the length guard cannot catch this one, so it is sliced.
    assert view.scheduled_tokens.tolist() == list(range(8, 16))


# ── token inputs: the mixed branch of prepare_input_ids, end to end ─────────


@pytest.mark.skipif(not torch.cuda.is_available(), reason="stages through the GPU")
def test_mixed_deferred_inputs_under_spec(monkeypatch):
    """Every kind of row a mixed spec step can hold, in one batch:

    - 40: a prefill middle chunk that was ALSO in the previous batch -- must not
      be taken for a carried-over decode (why the map runs over decode rows).
    - 10: a decode row carried over: anchor + drafts from last step's device
      buffers, and its rejected/bonus remapped from last step's status.
    - 30: last step's PREFILL row, decoding for the first time: its anchor and
      drafts come from the propose on that prefill row.
    - 50: a decode row not in the previous batch: scheduler-staged drafts.
    """
    from tests.test_h2d_runner_publication import runner_with_buffers

    runner = runner_with_buffers(monkeypatch, "direct", speculative=True)
    processor = runner.tokenID_processor
    processor.use_spec = True
    processor.num_spec_tokens = K
    runner._gate_staging_reuse()

    processor.prev_batch = SimpleNamespace(req_ids=[10, 20, 30, 40], is_dummy_run=False)
    processor.prev_token_ids = torch.tensor(
        [100, 200, 300, 400], dtype=torch.int32, device="cuda"
    )
    processor.draft_token_ids = torch.tensor(
        [[r + 1, r + 2, r + 3] for r in (100, 200, 300, 400)],
        dtype=torch.int32,
        device="cuda",
    )
    processor.pre_num_decode_token_per_seq = K + 1
    processor.prev_rejected_num = np.array([1, 0, 2, 0], dtype=np.int32)
    processor.prev_bonus_num = np.array([2, 3, 1, 3], dtype=np.int32)

    spec = np.zeros((4, K), dtype=np.int32)
    spec[3] = [501, 502, 503]
    batch = SimpleNamespace(
        is_mixed=True,
        is_dummy_run=False,
        req_ids=[40, 10, 30, 50],
        scheduled_tokens=np.array(
            [7, 8, 9] + [-1] * 4 + [-1] * 4 + [500, -1, -1, -1], dtype=np.int32
        ),
        total_tokens_num=15,
        total_tokens_num_prefill=3,
        total_tokens_num_decode=12,
        total_seqs_num_prefill=1,
        total_seqs_num_decode=3,
        total_seqs_num=4,
        num_scheduled_tokens=np.array([3, 4, 4, 4], dtype=np.int32),
        scheduled_spec_decode_tokens=spec,
        num_rejected=np.array([5, 0, 0, 0], dtype=np.int32),
        num_bonus=np.array([5, 0, 0, 0], dtype=np.int32),
        # False: keep the status set above rather than drain the (empty) queue.
        produces_output=lambda: False,
    )
    runner.attn_metadata_builder.publish_cu_seqlens_q(
        batch, SimpleNamespace(running_bs=4)
    )
    # Decode-local spans, from 0, anchor + 3 drafts each.
    assert runner.forward_vars["cu_seqlens_q"].np[:4].tolist() == [0, 4, 8, 12]

    ids = processor.prepare_input_ids(batch, K + 1).clone()
    runner._mark_staging_h2d_enqueued()
    runner.h2d_owner.completion.synchronize()
    assert ids.cpu().tolist() == (
        [7, 8, 9]  # prefill middle chunk, as scheduled
        + [100, 101, 102, 103]  # 10: carried-over decode
        + [300, 301, 302, 303]  # 30: first decode after its prefill row
        + [500, 501, 502, 503]  # 50: scheduler-staged
    )
    # Prefill row: nothing rejected. Carried-over rows: last step's status.
    assert processor.num_rejected.tolist() == [0, 1, 2, 0]
    assert processor.num_bonus.tolist() == [0, 2, 1, 0]
