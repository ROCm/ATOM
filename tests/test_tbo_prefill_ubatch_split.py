# SPDX-License-Identifier: MIT
"""The TBO splitter cuts every prefill at its token midpoint: through a request
that straddles it, and through the padding of a prefill with no request (an
idle DP rank's)."""

import types

import numpy as np
import pytest
import torch

try:
    from atom.model_ops.attentions.backends import CommonAttentionBuilder
    from atom.model_ops.attentions.token_layout.prefill import PrefillBatch
    from atom.utils.forward_context import Context, ForwardContext
    from atom.utils.tbo.ubatch_splitting import cut_prefill_ubatches
    from atom.utils.tbo.ubatch_wrapper import UBatchWrapper
    from atom.utils.tbo.ubatching import prefill_tbo_vote
except ImportError as _e:  # pragma: no cover - environment, not the code
    from import_guard import skip_if_dependency_missing

    skip_if_dependency_missing(_e, "the attention builders import AITER")


def _split(num_reqs, lens, running_tokens):
    return cut_prefill_ubatches(
        num_reqs, np.asarray(lens, dtype=np.int32), running_tokens
    )


@pytest.mark.parametrize("running_tokens", [2, 3, 16])
def test_a_prefill_with_no_request_cuts_its_padding_at_the_midpoint(running_tokens):
    slices = _split(0, [], running_tokens)
    half = running_tokens // 2
    assert [(s.token_slice.start, s.token_slice.stop) for s in slices] == [
        (0, half),
        (half, running_tokens),
    ]
    assert all(s.request_slice == slice(0, 0) for s in slices)


def test_one_request_splits_at_its_token_midpoint():
    slices = _split(1, [100], 100)
    assert [s.token_slice for s in slices] == [slice(0, 50), slice(50, 100)]
    assert [s.request_slice for s in slices] == [slice(0, 1), slice(0, 1)]


def test_a_prefill_of_one_token_does_not_split():
    assert _split(1, [1], 1) is None
    assert prefill_tbo_vote(1)[1] is False


@pytest.mark.parametrize("tokens", [2, 3, 7, 16])
def test_a_rank_s_vote_gathers_the_counts_its_split_runs(tokens):
    """Every rank gathers its vote's counts for each micro-batch's
    collectives; a split running other counts would hang them."""
    _, can_split, ub0, ub1 = prefill_tbo_vote(tokens)
    assert can_split
    slices = _split(1, [tokens], tokens)
    assert [s.running_tokens for s in slices] == [ub0, ub1]


def test_a_request_the_cut_halves_resumes_in_the_second_micro_batch():
    """Each micro-batch is a prefill batch of its own: the request the
    midpoint cuts forwards its first tokens in the first, its context ending
    at the cut, and its rest in the second, cached up to the cut -- what a
    chunked prefill of it would be."""
    step = PrefillBatch(
        context_lens=np.array([10, 4], dtype=np.int32),
        seqlens_q=np.array([6, 4], dtype=np.int32),
        block_tables=[[0], [1]],
    )
    cu_seqlens_q = np.array([0, 6, 10])
    first, second = (
        step.cut(ub.request_slice, ub.scheduled_token_slice, cu_seqlens_q)
        for ub in _split(2, [6, 4], 10)
    )
    assert first.seqlens_q.tolist() == [5]
    assert first.cached_lens.tolist() == [4]
    assert first.context_lens.tolist() == [9]
    assert second.seqlens_q.tolist() == [1, 4]
    assert second.cached_lens.tolist() == [9, 0]
    assert second.context_lens.tolist() == [10, 4]
    assert second.block_tables == [[0], [1]]


def _idle_batch():
    return types.SimpleNamespace(
        total_seqs_num=0,
        total_seqs_num_prefill=0,
        num_scheduled_tokens=np.zeros(0, dtype=np.int32),
        is_dummy_run=False,
    )


def test_a_prefill_with_no_request_lays_out_its_padding_as_one_request():
    """Every kernel takes a request: an idle rank's prefill runs its padding as
    one, in the query layout and the metadata alike."""
    builder = types.SimpleNamespace(
        prefill_batch=lambda batch, tokens: CommonAttentionBuilder.prefill_batch(
            builder, batch, tokens
        )
    )
    lengths = CommonAttentionBuilder.query_lengths(builder, _idle_batch(), True, 6)
    assert lengths.tolist() == [6]
    padding = CommonAttentionBuilder.prefill_batch(builder, _idle_batch(), 6)
    assert padding.cached_lens.tolist() == [0]
    assert padding.block_tables == []


def test_an_idle_rank_s_micro_batches_are_padding_requests_of_their_own():
    """Neither of an idle rank's micro-batches resumes the other's padding:
    each is one request of its own tokens, nothing cached."""
    slices = _split(0, [], 7)
    ubatches = CommonAttentionBuilder._prefill_ubatches(
        types.SimpleNamespace(), _idle_batch(), slices, None
    )
    assert [b.seqlens_q.tolist() for _, b in ubatches] == [[3], [4]]
    assert [b.cached_lens.tolist() for _, b in ubatches] == [[0], [0]]


def test_an_idle_rank_s_micro_batches_run_their_padding_with_the_group():
    """Joining its peers' ragged prefill, an idle rank's micro-batches each run
    their padding token, gathered by every rank's count like its peers' own:
    the MoE takes the tokens the micro-batch runs, not its zero scheduled."""
    slices = _split(0, [], 2)
    wrapper = UBatchWrapper(torch.nn.Identity(), types.SimpleNamespace())
    positions = torch.zeros(2, dtype=torch.int64)
    parent = ForwardContext(
        attn_metadata=types.SimpleNamespace(
            ubatch_token_order=None,
            ubatches=[types.SimpleNamespace(scheduled_tokens_gpu=None)] * 2,
        ),
        context=Context(
            positions=positions,
            is_prefill=True,
            scheduled_bs=0,
            scheduled_tokens=0,
            running_bs=1,
            running_tokens=2,
            running_tokens_are_unified=False,
        ),
        ubatch_slices=slices,
    )
    for i, ub in enumerate(slices):
        child = wrapper._make_ubatch_context(parent, ub, i, positions, None).context
        assert (child.scheduled_tokens, child.running_tokens) == (0, 1)
        assert child.running_bs == 1
        assert not child.running_tokens_are_unified
