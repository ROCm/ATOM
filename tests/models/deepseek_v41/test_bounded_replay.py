# SPDX-License-Identifier: MIT
"""Decoder SWA bounded replay (`models/deepseek_v41/bounded_replay.py`)."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from atom.model_ops.attentions.deepseek_v41.metadata import BatchStep, RequestSpan
from atom.models.deepseek_v41.bounded_replay import (
    decoder_replay_unsupported,
    late_layer_start,
    late_layer_tail_layout,
)


def _step(lengths, positions):
    spans, offset = [], 0
    for i, (n, p) in enumerate(zip(lengths, positions)):
        spans.append(RequestSpan(100 + i, p, offset, n, 7 + i))
        offset += n
    empty = torch.empty(0)
    return BatchStep(tuple(spans), empty, empty, empty, empty, empty, scheduled=offset)


def test_tail_layout_keeps_each_requests_last_rows():
    lengths, positions, tail = [5, 300, 133, 134], [0, 1000, 40, 7], 133
    spans, indices, replay_start = late_layer_tail_layout(
        _step(lengths, positions), tail
    )
    kept = [min(n, tail) for n in lengths]
    assert [s.length for s in spans] == kept
    assert [s.offset for s in spans] == list(np.cumsum([0] + kept[:-1]))
    assert [s.slot for s in spans] == [7, 8, 9, 10]
    assert [s.request_id for s in spans] == [100, 101, 102, 103]
    # every tail ends where its request ends, in position and in row
    assert [s.end for s in spans] == [p + n for p, n in zip(positions, lengths)]
    offsets = np.cumsum([0] + lengths[:-1])
    expected = np.concatenate(
        [np.arange(o + n - k, o + n) for o, n, k in zip(offsets, lengths, kept)]
    )
    np.testing.assert_array_equal(indices, expected)
    assert indices.dtype == np.int64 and np.all(np.diff(indices) > 0)
    # a request kept whole reads its own history; a trimmed one none below
    # its tail
    assert replay_start.tolist() == [0, 1000 + 300 - 133, 0, 7 + 134 - 133]


def _config(**overrides):
    config = dict(
        num_hidden_layers=40,
        kv_source_layer_ids=[2, 8, 14, 20],
        engram_layer_ids=[1, 14],
    )
    config.update(overrides)
    return SimpleNamespace(**config)


def test_v41_flash_layout_replays_from_layer_21():
    config = _config()
    assert decoder_replay_unsupported(config) is None
    assert late_layer_start(config) == 21


@pytest.mark.parametrize(
    "overrides, reason",
    [
        ({"kv_source_layer_ids": []}, "no kv_source_layer_ids"),
        ({"kv_source_layer_ids": [2, 39]}, "no layer follows"),
        ({"engram_layer_ids": [1, 30]}, "Engram"),
    ],
)
def test_refusals_name_their_reason(overrides, reason):
    assert reason in decoder_replay_unsupported(_config(**overrides))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernels")
def test_both_index_builders_floor_the_window_at_the_replay_start():
    """`_indptr_scan` counts each row's window from the same first position
    `_indices` writes it from; a drift between the two leaves holes or
    overruns in the prefix plane."""
    from atom.model_ops.attentions.deepseek_v41.indices import _indices, _indptr_scan

    window, tokens = 128, 40
    # two prefill requests whose tails start at positions 500 and 90; the
    # second keeps its whole history (replay start 0)
    lengths, starts, floors = [24, 16], [500, 90], [500, 0]
    batches = torch.tensor(
        sum(([b] * n for b, n in enumerate(lengths)), []),
        dtype=torch.int32,
        device="cuda",
    )
    positions = torch.tensor(
        sum((list(range(s, s + n)) for s, n in zip(starts, lengths)), []),
        dtype=torch.int32,
        device="cuda",
    )
    cu = torch.tensor([0, 24, 40], dtype=torch.int32, device="cuda")
    replay_start = torch.tensor(floors, dtype=torch.int32, device="cuda")
    pptr = torch.empty(tokens + 1, dtype=torch.int32, device="cuda")
    eptr = torch.empty(tokens + 1, dtype=torch.int32, device="cuda")
    _indptr_scan[(1,)](
        batches, positions, cu, pptr, eptr, tokens, replay_start,
        DECODE=False, WINDOW=window, RATIO=1, TOPK=0, EXTEND=True, BLOCK=64,
    )  # fmt: skip
    counts = (pptr[1:] - pptr[:-1]).tolist()
    expected = []
    for b, n in enumerate(lengths):
        first = starts[b]  # a prefill row's history ends at its chunk start
        for j in range(n):
            lo = max(starts[b] + j - window + 1, 0, floors[b])
            expected.append(max(first - lo, 0))
    assert counts == expected
    # the first replayed request reads no window row at all: its whole
    # history in these layers is below the replay start
    assert counts[:24] == [0] * 24


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernels")
def test_a_replay_start_keeps_the_suffix_of_each_rows_window():
    """`_indices` writes what `_indptr_scan` reserved: with a replay start each
    row's window segment is the tail of the same row's segment without one."""
    from atom.model_ops.attentions.deepseek_v41.cache import PagedAttentionCache
    from atom.model_ops.attentions.deepseek_v41.indices import fill_step_indptrs
    from atom.model_ops.attentions.pool_layout.v41_pool_geometry import (
        V41PoolGeometry,
    )
    from atom.models.deepseek_v41.config import AttentionMode, LayerAttentionSpec
    from tests.attentions.deepseek_v41.helpers import PagedRequest, begin_step

    geo = V41PoolGeometry(
        2, ((0, 1),), 32, 8, 512, 32, layer_ratios=(0, 1), index_topk=4
    )
    cache = PagedAttentionCache(geo, 32, 4, "cuda")
    requests = (
        PagedRequest(1, 40, 0, 6, 0, (0, 1)),
        PagedRequest(2, 20, 6, 5, 1, (2,)),
    )
    spec = LayerAttentionSpec(1, 0, AttentionMode.WINDOW)

    def segments(step):
        step.group_indices.clear()
        step.indptrs = fill_step_indptrs(step, geo, cache.indptr_buffers)
        prefix, pptr, _, _ = cache.attention_indices(spec, step)
        bounds = pptr.tolist()
        return [prefix[a:b].tolist() for a, b in zip(bounds, bounds[1:])]

    step = begin_step(cache, requests)
    full = segments(step)
    step.swa_replay_start = torch.tensor([37, 0], dtype=torch.int32, device="cuda")
    floored = segments(step)
    for p, (whole, kept) in enumerate(zip(full, floored)):
        position = step.positions[p].item()
        request = 0 if p < 6 else 1
        floor = (37, 0)[request]
        history_end = (40, 20)[request]
        first = max(position - geo.window_size + 1, 0, floor)
        assert len(kept) == max(history_end - first, 0)
        assert kept == whole[len(whole) - len(kept) :]
