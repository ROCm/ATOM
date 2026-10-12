# SPDX-License-Identifier: MIT
"""Decoder SWA bounded replay (`models/deepseek_v41/bounded_replay.py`)."""

from itertools import pairwise
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from atom.model_ops.attentions.deepseek_v41.metadata import (
    BatchStep,
    RequestSpan,
)
from atom.models.deepseek_v41.bounded_replay import (
    build_late_layer_tail,
    decoder_replay_unsupported,
    late_layer_start,
    late_layer_tail_layout,
    replay_rows,
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
    config = {
        "num_hidden_layers": 40,
        "kv_source_layer_ids": [2, 8, 14, 20],
        "engram_layer_ids": [1, 14],
    }
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernels on GPU")
def test_both_index_builders_floor_the_window_at_the_replay_start():
    """`_indptr_scan` counts each row's window from the same first position
    `_indices` writes it from; a drift between the two leaves holes or
    overruns in the prefix plane."""
    from atom.model_ops.attentions.deepseek_v41.indices import _indptr_scan

    window, tokens = 128, 40
    # two prefill requests whose tails start at positions 500 and 90; the
    # second keeps its whole history (replay start 0)
    lengths, starts, floors = [24, 16], [500, 90], [500, 0]
    batches = torch.tensor(
        [b for b, n in enumerate(lengths) for _ in range(n)],
        dtype=torch.int32,
        device="cuda",
    )
    positions = torch.tensor(
        [p for s, n in zip(starts, lengths) for p in range(s, s + n)],
        dtype=torch.int32,
        device="cuda",
    )
    cu = torch.tensor([0, 24, 40], dtype=torch.int32, device="cuda")
    replay_start = torch.tensor(floors, dtype=torch.int32, device="cuda")
    pptr = torch.empty(tokens + 1, dtype=torch.int32, device="cuda")
    eptr = torch.empty(tokens + 1, dtype=torch.int32, device="cuda")
    _indptr_scan[(1,)](
        batches, positions, cu, pptr, eptr, tokens, replay_start,
        DECODE=False, WINDOW=window, RATIO=1, TOPK=0, EXTEND=True, SPLIT=False,
        BLOCK=64,
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernels on GPU")
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
        return [prefix[a:b].tolist() for a, b in pairwise(bounds)]

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


def _spec(layer_id, mode, topk_owner=None, candidate_source=None):
    return SimpleNamespace(
        layer_id=layer_id,
        mode=mode,
        topk_owner=topk_owner,
        candidate_source=candidate_source,
    )


def test_tail_step_carries_early_selections_and_candidates_by_row():
    """The tail step's per-row tensors are the forward's at the tail rows;
    selections and candidates come over only from early owners."""
    from atom.models.deepseek_v41.config import AttentionMode

    lengths, positions, tail_len = [5, 300, 140], [0, 1000, 7], 133
    spans, offset = [], 0
    for i, (n, p) in enumerate(zip(lengths, positions)):
        spans.append(RequestSpan(100 + i, p, offset, n, 7 + i))
        offset += n
    width = offset
    forward_positions = torch.cat(
        [torch.arange(p, p + n) for p, n in zip(positions, lengths)]
    )
    rows = torch.arange(width)
    step = BatchStep(
        tuple(spans),
        forward_positions,
        torch.tensor([0, 5, 305, 445], dtype=torch.int32),
        torch.tensor([7, 8, 9], dtype=torch.int32),
        torch.zeros(width, dtype=torch.int32),
        torch.arange(12, dtype=torch.int32).view(3, 4),
        scheduled=width,
        visible={1: rows.int() * 10},
        plans={1: "plan"},
    )
    step.selected = {20: rows.view(1, width, 1).expand(1, width, 3).int()}
    step.candidates = {20: rows.view(width, 1).int() + 1000}
    late = [
        _spec(21, AttentionMode.REUSE, topk_owner=20, candidate_source=20),
        _spec(24, AttentionMode.REINDEX, topk_owner=24, candidate_source=20),
        _spec(25, AttentionMode.REUSE, topk_owner=24, candidate_source=20),
    ]
    tail = build_late_layer_tail(step, tail_len, late)
    kept = tail.token_indices
    _, expected, replay_start = late_layer_tail_layout(step, tail_len)
    torch.testing.assert_close(kept, torch.from_numpy(expected))
    t = tail.step
    assert t.scheduled == t.width == kept.numel() == 5 + 133 + 133
    assert t.max_q_len == 133
    assert t.cu_seqlens_q.tolist() == [0, 5, 138, 271]
    assert t.batch_ids.tolist() == [0] * 5 + [1] * 133 + [2] * 133
    assert t.swa_replay_start.tolist() == replay_start.tolist()
    torch.testing.assert_close(t.positions, forward_positions[kept])
    torch.testing.assert_close(t.visible[1], step.visible[1][kept])
    assert t.plans is step.plans
    # the early owner's selection and candidates come over by row; the late
    # REINDEX owner (24) selects again on the tail, so nothing of it is carried
    assert set(t.selected) == {20} and set(t.candidates) == {20}
    torch.testing.assert_close(t.selected[20], step.selected[20][:, kept])
    torch.testing.assert_close(t.candidates[20], step.candidates[20][kept])
    # indptrs are the context manager's to fill, on the GPU
    assert not t.indptrs


def _forward(lengths, **overrides):
    step = SimpleNamespace(
        requests=tuple(
            RequestSpan(i, 0, sum(lengths[:i]), n, i) for i, n in enumerate(lengths)
        ),
        decode=False,
        width=sum(lengths),
        scheduled=sum(lengths),
    )
    metadata = SimpleNamespace(
        step=step,
        image_mask=None,
        cache=SimpleNamespace(geometry=SimpleNamespace(ring_slots=133)),
    )
    context = SimpleNamespace(is_prefill=True, is_dummy_run=False, is_draft=False)
    forward = SimpleNamespace(
        context=context, attn_metadata=metadata, ubatch_slices=None
    )
    for key, value in overrides.items():
        target, name = key.split(".")
        setattr(
            {"ctx": context, "md": metadata, "fwd": forward, "step": step}[target],
            name,
            value,
        )
    return forward


@pytest.mark.parametrize(
    "lengths, overrides, expected",
    [
        ([500, 20], {}, 133),
        ([133, 20], {}, None),  # nothing longer than the ring
        ([500], {"ctx.is_prefill": False}, None),
        ([500], {"ctx.is_dummy_run": True}, None),
        ([500], {"ctx.is_draft": True}, None),
        ([500], {"fwd.ubatch_slices": [object()]}, None),
        ([500], {"step.decode": True}, None),
        # image rows replay: the router reads the tail's slice of the mask
        ([500], {"md.image_mask": torch.ones(1, 500, dtype=torch.bool)}, 133),
        ([500], {"step.width": 512}, None),  # padded (DP attention) step
    ],
)
def test_only_an_unpadded_prefill_replays(lengths, overrides, expected):
    forward = _forward(lengths, **overrides)
    assert replay_rows(forward, forward.attn_metadata.step.width) == expected


def test_the_late_layers_see_the_tails_image_mask():
    """The late layers' MoE router reads the forward's image mask; during the
    tail it must be the tail's rows, and the forward's afterwards."""
    pytest.importorskip("triton")
    from atom.models.deepseek_v41.bounded_replay import LateLayerTail, late_layer_tail

    step = _step([300, 20], [0, 0])
    rows = torch.tensor([167 + i for i in range(133)] + list(range(300, 320)))
    mask = (torch.arange(320) % 3 == 0)[None]
    tail_step = SimpleNamespace(indptrs={}, tiles={})
    metadata = SimpleNamespace(
        step=step,
        image_mask=mask,
        cache=SimpleNamespace(geometry=None, indptr_buffers=None),
    )
    from atom.model_ops.attentions.deepseek_v41 import indices

    seen = {}
    original = indices.fill_step_indptrs
    indices.fill_step_indptrs = lambda *a: {}
    try:
        with late_layer_tail(metadata, LateLayerTail(rows, tail_step)):
            seen["mask"] = metadata.image_mask.clone()
            seen["step"] = metadata.step
    finally:
        indices.fill_step_indptrs = original
    torch.testing.assert_close(seen["mask"], mask[:, rows])
    assert seen["step"] is tail_step
    assert metadata.image_mask is mask and metadata.step is step


def test_late_rows_land_back_at_their_forward_rows(monkeypatch):
    """A replay's late-layer outputs and DSpark aux captures come back at the
    forward rows of each request's tail, whatever the mix of lengths."""
    pytest.importorskip("aiter")
    from atom.models.deepseek_v41 import runtime
    from atom.models.deepseek_v41.bounded_replay import LateLayerTail

    lengths, tail_len, hidden = [3, 9, 5, 12], 4, 6
    step = _step(lengths, [0, 50, 7, 100])
    _, indices, _ = late_layer_tail_layout(step, tail_len)
    rows = torch.from_numpy(indices)
    total = sum(lengths)
    model = runtime.DeepseekV41RuntimeModel.__new__(runtime.DeepseekV41RuntimeModel)
    torch.nn.Module.__init__(model)
    aux = [torch.full((64, hidden), -1.0), torch.full((64, hidden), -2.0)]

    def late(*state):
        # the stage sees only the tail; tag each row with its forward row
        kept = state[0].shape[0]
        assert kept == rows.numel()
        aux[1][:kept] = rows[:, None].float() + 0.5
        return rows[:, None].float().expand(kept, hidden).clone()

    model.late = late
    # capture 0 is an early layer's (full rows already), capture 1 a late one's
    model.late_aux_layers, model.aux_buffers = (1,), aux
    metadata = SimpleNamespace(step=step)
    monkeypatch.setattr(
        runtime, "get_forward_context", lambda: SimpleNamespace(attn_metadata=metadata)
    )
    monkeypatch.setattr(runtime, "late_layer_tail", _swap_step_only)
    # residual [N, hc, H], pre [N, hc], pending [N, H], post [N, hc],
    # combination [N, hc, hc]: rows first, as the early graph returns them
    state = (
        torch.zeros(total, 4, 2),
        torch.zeros(total, 4),
        torch.zeros(total, 2),
        torch.zeros(total, 4),
        torch.zeros(total, 4, 4),
    )
    from atom.model_ops.deepseek_v41.mhc import SinglePassHCState

    tail_state = SinglePassHCState(*state).take_rows(rows)
    out = model._late_on_tail(tail_state, LateLayerTail(rows, step), total)
    assert out.shape == (total, hidden)
    torch.testing.assert_close(out[rows], rows[:, None].float().expand(-1, hidden))
    # rows outside the tail are zero, not undefined
    outside = torch.ones(total, dtype=torch.bool)
    outside[rows] = False
    assert torch.all(out[outside] == 0)
    torch.testing.assert_close(
        aux[1][rows], rows[:, None].float().expand(-1, hidden) + 0.5
    )
    # an early layer's capture is left alone
    assert torch.all(aux[0] == -1.0)
    assert metadata.step is step


def _swap_step_only(metadata, tail):
    """`late_layer_tail` without the GPU indptr fill."""
    import contextlib

    @contextlib.contextmanager
    def swap():
        full = metadata.step
        metadata.step = tail.step
        try:
            yield
        finally:
            metadata.step = full

    return swap()


def test_hidden_state_export_turns_replay_off(monkeypatch):
    """With replay forbidden (TorchSpec export reads every row), a replayable
    prefill runs the late layers on every row."""
    pytest.importorskip("aiter")
    from atom.models.deepseek_v41 import runtime

    model = runtime.DeepseekV41RuntimeModel.__new__(runtime.DeepseekV41RuntimeModel)
    torch.nn.Module.__init__(model)
    model.replay, model.replay_enabled, model.late_specs = True, True, ()
    rows_seen = []
    model.early = lambda input_ids, embeds: (
        torch.zeros(input_ids.numel(), 4, 2),
        torch.zeros(input_ids.numel(), 4),
        torch.zeros(input_ids.numel(), 2),
        torch.zeros(input_ids.numel(), 4),
        torch.zeros(input_ids.numel(), 4, 4),
    )
    model.late = lambda *state: rows_seen.append(state[0].shape[0]) or state[0]
    forward = _forward([500])
    monkeypatch.setattr(runtime, "get_forward_context", lambda: forward)
    monkeypatch.setattr(
        runtime,
        "build_late_layer_tail",
        lambda *a: SimpleNamespace(token_indices=torch.arange(367, 500)),
    )
    monkeypatch.setattr(model, "_late_on_tail", lambda *a: rows_seen.append("tail"))
    model(torch.zeros(500, dtype=torch.int64), None)
    assert rows_seen == ["tail"]
    assert model.set_decoder_replay(False) is False
    model(torch.zeros(500, dtype=torch.int64), None)
    assert rows_seen == ["tail", 500]
