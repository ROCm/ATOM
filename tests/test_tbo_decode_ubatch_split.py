# SPDX-License-Identifier: MIT
"""A TBO decode step's micro-batches are cut by request, as its graphs are
recorded (`cut_decode_ubatches`): each takes half the bucket's requests and
half its running tokens, the same on every step of the bucket, so a recording
replays any of them, and its requests' tokens -- packed however this rank
scheduled them -- are gathered into that half and scattered back
(`decode_ubatch_token_order`)."""

import types

import numpy as np
import pytest
import torch

from atom.config import Config
from atom.utils import CpuGpuBuffer
from atom.utils.forward_context import Context, ForwardContext, get_forward_context
from atom.utils.tbo import ubatch_wrapper as wrapper_module
from atom.utils.tbo.ubatch_splitting import (
    cut_decode_ubatches,
    decode_ubatch_token_order,
)
from atom.utils.tbo.ubatch_wrapper import UBatchWrapper


def test_a_decode_is_cut_by_request_into_equal_buckets():
    """Three requests of a 4-bucket: two in the first half, one and a padding
    request in the second, each half its own `running_bs`."""
    slices = cut_decode_ubatches(4, 8, np.array([2, 2, 2]))
    assert [s.request_slice for s in slices] == [slice(0, 2), slice(2, 3)]
    assert [s.running_bs for s in slices] == [2, 2]
    assert [s.token_slice for s in slices] == [slice(0, 4), slice(4, 8)]
    assert [s.scheduled_token_slice for s in slices] == [slice(0, 4), slice(4, 6)]
    assert [s.scheduled_tokens for s in slices] == [4, 2]


def test_shrunk_requests_fill_their_half_and_leave_padding():
    """Requests shorter than the bucket's `max_seqlen_q` (a DSpark q-bucket
    shrink) still split by request; each half pads past its own tokens."""
    slices = cut_decode_ubatches(4, 12, np.array([2, 1, 3]))
    assert [s.request_slice for s in slices] == [slice(0, 2), slice(2, 3)]
    assert [s.scheduled_token_slice for s in slices] == [slice(0, 3), slice(3, 6)]
    assert [(s.running_bs, s.running_tokens) for s in slices] == [(2, 6), (2, 6)]


def test_requests_that_overflow_a_half_are_refused():
    with pytest.raises(ValueError, match="do not fit"):
        cut_decode_ubatches(4, 4, np.array([2, 2, 2]))


def test_the_token_order_is_a_permutation_own_tokens_first():
    slices = cut_decode_ubatches(4, 12, np.array([2, 1, 3]))
    order = np.empty(12, dtype=np.int64)
    assert decode_ubatch_token_order(slices, 6, order) == 12
    assert order.tolist() == [0, 1, 2, 6, 7, 8, 3, 4, 5, 9, 10, 11]


def test_the_merge_puts_each_token_back_where_it_was_packed():
    slices = cut_decode_ubatches(4, 12, np.array([2, 1, 3]))
    order = np.empty(12, dtype=np.int64)
    decode_ubatch_token_order(slices, 6, order)
    ub0 = torch.full((6, 2), 1.0)
    ub1 = torch.full((6, 2), 2.0)
    out = UBatchWrapper._merge_ubatch_outputs([ub0, ub1], torch.from_numpy(order))
    assert out[:, 0].tolist() == [1.0] * 3 + [2.0] * 3 + [1.0] * 3 + [2.0] * 3


def test_decode_tbo_makes_every_bucket_past_two_even():
    """The ladder a decode TBO deployment records holds only even buckets past
    2, so every bucket that splits halves into as many requests."""
    config = types.SimpleNamespace(
        compilation_config=types.SimpleNamespace(
            cudagraph_capture_sizes=[1, 2, 3, 4, 5, 7, 16]
        ),
        enable_tbo_decode=True,
    )
    assert Config.capture_sizes.fget(config) == [1, 2, 4, 6, 8, 16]
    config.enable_tbo_decode = False
    assert Config.capture_sizes.fget(config) == [1, 2, 3, 4, 5, 7, 16]


def _forward_vars(running_tokens, device):
    return {
        "ubatch_token_order": CpuGpuBuffer(
            running_tokens,
            dtype=torch.int64,
            device=device,
            pin_memory=device != "cpu",
        ),
        "ubatch_scheduled_tokens": CpuGpuBuffer(
            2, dtype=torch.int32, device=device, pin_memory=device != "cpu"
        ),
    }


def _split_decode_metadata(var, slices, scheduled_tokens):
    """A split decode's step metadata as the TBO wrapper receives it: the
    token order and each micro-batch's scheduled tokens `prepare_query_metadata`
    staged, handed over on the metadata (`_attach_decode_ubatch_order`)."""
    decode_ubatch_token_order(slices, scheduled_tokens, var["ubatch_token_order"].np)
    var["ubatch_scheduled_tokens"].np[:] = [s.scheduled_tokens for s in slices]
    for name in ("ubatch_token_order", "ubatch_scheduled_tokens"):
        var[name].gpu.copy_(var[name].cpu)
    scheduled = var["ubatch_scheduled_tokens"].gpu
    return types.SimpleNamespace(
        ubatch_token_order=var["ubatch_token_order"].gpu[: slices[-1].token_slice.stop],
        ubatches=[
            types.SimpleNamespace(scheduled_tokens_gpu=scheduled[i : i + 1])
            for i in range(len(slices))
        ],
    )


class _TokenWise(torch.nn.Module):
    """A model computing each of its pass's scheduled tokens on its own, its
    other tokens zero, from the device count a recording replays."""

    def forward(self, input_ids, positions):
        scheduled = get_forward_context().context.scheduled_tokens_gpu
        tokens = torch.arange(input_ids.shape[0], device=input_ids.device)
        out = (input_ids * 1000 + positions).to(torch.float32)
        return torch.where(tokens < scheduled, out, 0.0)[:, None]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_a_split_step_replays_its_recording_like_the_whole_step(monkeypatch):
    """One recording of the bucket serves steps whose requests are packed
    differently: each returns what the unsplit step does, token for token."""
    running_bs, running_tokens = 4, 20
    var = _forward_vars(running_tokens, "cuda")
    builder = types.SimpleNamespace(
        ubatch_forward=lambda metadata: torch.cuda.stream(torch.cuda.current_stream()),
    )
    wrapper = UBatchWrapper(_TokenWise(), builder)
    input_ids = torch.zeros(running_tokens, dtype=torch.int64, device="cuda")
    positions = torch.zeros_like(input_ids)
    current = {}
    monkeypatch.setattr(wrapper_module, "get_forward_context", lambda: current["ctx"])

    def step(lengths):
        slices = cut_decode_ubatches(running_bs, running_tokens, lengths)
        scheduled_tokens = int(lengths.sum())
        current["ctx"] = ForwardContext(
            attn_metadata=_split_decode_metadata(var, slices, scheduled_tokens),
            context=Context(
                positions=None,
                scheduled_bs=len(lengths),
                scheduled_tokens=scheduled_tokens,
                running_bs=running_bs,
                running_tokens=running_tokens,
            ),
            ubatch_slices=slices,
        )
        input_ids.copy_(torch.randint(1, 100, (running_tokens,)))
        positions.copy_(torch.randint(0, 999, (running_tokens,)))
        return scheduled_tokens

    step(np.array([5, 5, 5, 5]))
    wrapper(input_ids, positions)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        recorded = wrapper(input_ids, positions)
    for lengths in (
        np.array([3, 3, 3]),
        np.array([1, 5, 2, 5]),
        np.array([5, 5, 5, 5]),
    ):
        scheduled = step(lengths)
        eager = wrapper(input_ids, positions)
        graph.replay()
        whole = (input_ids * 1000 + positions).to(torch.float32)[:, None]
        whole[scheduled:] = 0.0
        torch.testing.assert_close(eager, whole, rtol=0, atol=0)
        torch.testing.assert_close(recorded, whole, rtol=0, atol=0)


def test_micro_batch_contexts_read_their_tokens_and_counts_in_order():
    """A micro-batch's positions are its own tokens' and its scheduled count is
    its own requests', both off the step's token order; scratch is laid out
    end to end, so two in flight never share it."""
    slices = cut_decode_ubatches(4, 12, np.array([2, 1, 3]))
    metadata = _split_decode_metadata(_forward_vars(12, "cpu"), slices, 6)
    order = metadata.ubatch_token_order
    wrapper = UBatchWrapper(torch.nn.Identity(), types.SimpleNamespace())
    positions = torch.arange(12, dtype=torch.int64)
    parent = ForwardContext(
        attn_metadata=metadata,
        context=Context(
            positions=positions,
            scheduled_bs=3,
            scheduled_tokens=6,
            running_bs=4,
            running_tokens=12,
        ),
        ubatch_slices=slices,
    )
    children = [
        wrapper._make_ubatch_context(parent, s, i, positions[order], order).context
        for i, s in enumerate(slices)
    ]
    assert [c.scheduled_tokens for c in children] == [3, 3]
    assert [int(c.scheduled_tokens_gpu) for c in children] == [3, 3]
    assert [c.running_bs for c in children] == [2, 2]
    assert [c.ubatch_token_offset for c in children] == [0, 6]
    assert children[1].positions[:3].tolist() == [3, 4, 5]
    assert children[1].ubatch_token_order[:3].tolist() == [3, 4, 5]
