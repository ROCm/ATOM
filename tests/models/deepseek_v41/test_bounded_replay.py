# SPDX-License-Identifier: MIT
"""Decoder SWA bounded replay (`models/deepseek_v41/bounded_replay.py`)."""

from itertools import pairwise
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from atom.model_ops.attentions.deepseek_v41.metadata import RequestSpan
from atom.models.deepseek_v41.bounded_replay import (
    build_replay_plan,
    carry_selections,
    decoder_replay_unsupported,
    late_layer_start,
    replayable,
    rows_of,
)

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton on GPU")


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


def _cache(slots=4, pages=64):
    from atom.model_ops.attentions.deepseek_v41.cache import PagedAttentionCache
    from atom.model_ops.attentions.pool_layout.v41_pool_geometry import (
        V41PoolGeometry,
    )

    # a window of 8 rows, PAGEs of 32
    geo = V41PoolGeometry(
        2, ((0, 1),), 32, 8, 512, 32, layer_ratios=(0, 1), index_topk=4
    )
    return PagedAttentionCache(geo, pages, slots, "cuda")


def _segments(cache, step):
    """Each row's `(prefix, extend)` index lists for a window layer."""
    from atom.model_ops.attentions.deepseek_v41.indices import fill_step_indptrs
    from atom.models.deepseek_v41.config import AttentionMode, LayerAttentionSpec

    step.group_indices.clear()
    step.indptrs = fill_step_indptrs(step, cache.geometry, cache.indptr_buffers)
    prefix, pptr, extend, eptr = cache.attention_indices(
        LayerAttentionSpec(1, 0, AttentionMode.WINDOW), step
    )
    return [
        (prefix[a:b].tolist(), extend[c:d].tolist())
        for (a, b), (c, d) in zip(pairwise(pptr.tolist()), pairwise(eptr.tolist()))
    ]


@cuda
def test_a_replay_start_keeps_the_suffix_of_each_rows_window():
    """`_indices` writes what `_indptr_scan` reserved: with a replay start each
    row's window segment is the tail of the same row's segment without one."""
    from tests.attentions.deepseek_v41.helpers import PagedRequest, begin_step

    cache = _cache()
    requests = (
        PagedRequest(1, 40, 0, 6, 0, (0, 1)),
        PagedRequest(2, 20, 6, 5, 1, (2,)),
    )
    step = begin_step(cache, requests)
    full = _segments(cache, step)
    step.swa_replay_start = torch.tensor([37, 0], dtype=torch.int32, device="cuda")
    floored = _segments(cache, step)
    for p, ((whole, _), (kept, _)) in enumerate(zip(full, floored)):
        position = step.positions[p].item()
        floor, history_end = ((37, 40), (0, 20))[0 if p < 6 else 1]
        first = max(position - 7, 0, floor)
        assert len(kept) == max(history_end - first, 0)
        assert kept == whole[len(whole) - len(kept) :]


@cuda
def test_rows_past_where_the_ring_holds_read_it_there():
    """A forward recomputing rows below the point its ring already holds
    (`swa_cached_until`): its rows from that point read the ring below it, as
    a forward resuming there would; the rows below read each other only."""
    from tests.attentions.deepseek_v41.helpers import PagedRequest, begin_step

    cache = _cache()
    # one request recomputing 32..40 below its resume point 40, one ordinary
    requests = (
        PagedRequest(1, 32, 0, 16, 0, (0, 1)),
        PagedRequest(2, 20, 16, 5, 1, (2,)),
    )
    step = begin_step(
        cache,
        requests,
        floors=np.asarray([32, 0], np.int32),
        cached_until=np.asarray([40, 0], np.int32),
    )
    resumed = _segments(cache, step)
    # the same rows from 40 on, as a forward resuming there would build them
    alone = begin_step(cache, (PagedRequest(1, 40, 0, 8, 0, (0, 1)),))
    expected = _segments(cache, alone)
    for j in range(16):
        prefix, extend = resumed[j]
        position = 32 + j
        if position < 40:
            # nothing below the forward's start, the rest its own rows
            assert prefix == [] and extend == list(range(j + 1))
        else:
            want_prefix, want_extend = expected[position - 40]
            assert prefix == want_prefix
            assert extend == [r + 8 for r in want_extend]
    # the ordinary request reads as it would alone
    other = _segments(cache, begin_step(cache, (PagedRequest(2, 20, 0, 5, 1, (2,)),)))
    assert resumed[16:] == [(p, [r + 16 for r in e]) for p, e in other]


def _query_rows(step, rows):
    from dataclasses import replace

    from atom.models.deepseek_v41.bounded_replay import QueryRows

    ids = torch.full_like(step.batch_ids, -1)
    ids[rows] = step.batch_ids[rows]
    rows = torch.tensor(rows, device="cuda")
    return QueryRows(
        rows,
        step.positions[rows],
        replace(step, batch_ids=ids),
        torch.cat((rows, torch.tensor([step.width], device="cuda"))),
    )


@cuda
def test_queries_read_the_index_lists_every_row_would():
    """Some rows of a step as its queries: their index lists are the ones the
    step builds for those rows, adjacent once the other rows are padding."""
    from atom.models.deepseek_v41.config import AttentionMode, LayerAttentionSpec
    from tests.attentions.deepseek_v41.helpers import PagedRequest, begin_step

    cache = _cache()
    requests = (
        PagedRequest(1, 40, 0, 12, 0, (0, 1)),
        PagedRequest(2, 20, 12, 5, 1, (2,)),
    )
    step = begin_step(cache, requests)
    full = _segments(cache, step)
    queries = _query_rows(step, [9, 10, 11, 16])
    from atom.model_ops.attentions.deepseek_v41.indices import fill_step_indptrs

    queries.index.indptrs = fill_step_indptrs(
        queries.index, cache.geometry, cache.indptr_buffers
    )
    prefix, pptr, extend, eptr = queries.indices(
        cache, LayerAttentionSpec(1, 0, AttentionMode.WINDOW)
    )
    got = [
        (prefix[a:b].tolist(), extend[c:d].tolist())
        for (a, b), (c, d) in zip(pairwise(pptr.tolist()), pairwise(eptr.tolist()))
    ]
    assert got == [full[r] for r in (9, 10, 11, 16)]


def _forward_step(cache, lengths, positions, finals):
    from tests.attentions.deepseek_v41.helpers import PagedRequest, begin_step

    requests, offset = [], 0
    for i, (n, p) in enumerate(zip(lengths, positions)):
        pages = tuple(range(8 * i, 8 * i + -(-(p + n) // 32)))
        requests.append(PagedRequest(100 + i, p, offset, n, i, pages))
        offset += n
    return begin_step(cache, requests, finals=finals)


@cuda
def test_a_replay_plan_runs_each_finishing_request_on_its_last_rows():
    cache = _cache(pages=128)
    # a finishing prompt of 300 rows, an early chunk, one shorter than a window
    step = _forward_step(cache, [300, 200, 5], [0, 0, 64], [True, False, True])
    step.visible = {1: torch.arange(step.width, device="cuda", dtype=torch.int32)}
    plan = build_replay_plan(step, 128)
    tail = plan.tail
    assert [(s.request_id, s.position, s.length) for s in tail.requests] == [
        (100, 172, 128),
        (102, 64, 5),
    ]
    assert plan.tail_rows.tolist() == list(range(172, 300)) + list(range(500, 505))
    assert tail.visible[1].tolist() == plan.tail_rows.tolist()
    # every decoder layer's window starts at the tail
    assert tail.swa_replay_start.tolist() == [172, 64]
    assert tail.slots.tolist() == [0, 2]
    # layer 39's queries: each tail's last row
    assert plan.q39.rows.tolist() == [127, 132]
    assert plan.q39.positions.tolist() == [299, 68]
    assert plan.last_rows.tolist() == [299, 504]
    assert plan.q39.index.batch_ids.tolist().count(-1) == 131
    assert not tail.plans


@cuda
def test_a_forward_finishing_no_prompt_runs_nothing_past_layer_19():
    cache = _cache()
    step = _forward_step(cache, [300], [0], [False])
    assert build_replay_plan(step, 128).tail is None


@cuda
def test_carried_selections_come_from_layers_that_do_not_run_on_the_rows():
    from atom.models.deepseek_v41.config import AttentionMode

    cache = _cache(pages=128)
    step = _forward_step(cache, [300], [0], [True])
    width = step.width
    rows = torch.arange(width, device="cuda", dtype=torch.int32)
    step.selected = {
        14: rows.view(1, width, 1).expand(1, width, 3).contiguous(),
        20: rows.view(1, width, 1).clone(),
    }
    step.candidates = {14: rows.view(width, 1) + 1000}
    plan = build_replay_plan(step, 128)
    spec = SimpleNamespace
    specs = [
        spec(layer_id=20, topk_owner=20, candidate_source=14, mode=AttentionMode.FULL),
        spec(layer_id=21, topk_owner=20, candidate_source=14, mode=AttentionMode.REUSE),
        spec(layer_id=22, topk_owner=14, candidate_source=14, mode=AttentionMode.REUSE),
    ]
    carry_selections(step, plan, specs)
    kept = plan.tail_rows
    # layer 20 selects again on the tail; layer 14 does not run on it
    assert set(plan.tail.selected) == {14} and set(plan.tail.candidates) == {14}
    torch.testing.assert_close(plan.tail.selected[14], step.selected[14][:, kept])
    torch.testing.assert_close(plan.tail.candidates[14], step.candidates[14][kept])


def _forward(lengths, **overrides):
    step = SimpleNamespace(
        requests=tuple(
            RequestSpan(i, 0, sum(lengths[:i]), n, i) for i, n in enumerate(lengths)
        ),
        decode=False,
        width=sum(lengths),
        scheduled=sum(lengths),
    )
    metadata = SimpleNamespace(step=step, image_mask=None)
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
    "overrides, expected",
    [
        ({}, True),
        ({"ctx.is_prefill": False}, False),
        ({"ctx.is_dummy_run": True}, False),
        ({"ctx.is_draft": True}, False),
        ({"fwd.ubatch_slices": [object()]}, False),
        ({"step.decode": True}, False),
        # image rows replay: the router reads its rows' slice of the mask
        ({"md.image_mask": torch.ones(1, 500, dtype=torch.bool)}, True),
        ({"step.width": 512}, False),  # padded (DP attention) step
    ],
)
def test_only_an_unpadded_prefill_replays(overrides, expected):
    forward = _forward([500], **overrides)
    assert replayable(forward, 500) == expected


def test_a_row_step_sees_its_rows_image_mask(monkeypatch):
    """The decoder layers' MoE router reads the forward's image mask; while a row
    step runs it must be that step's rows, and the outer one's afterwards."""
    pytest.importorskip("triton")
    from atom.model_ops.attentions.deepseek_v41 import indices

    outer = SimpleNamespace(indptrs={1: 0}, tiles={1: 0})
    inner = SimpleNamespace(indptrs={}, tiles={})
    rows = torch.tensor([167 + i for i in range(133)] + list(range(300, 320)))
    mask = (torch.arange(320) % 3 == 0)[None]
    metadata = SimpleNamespace(
        step=outer,
        queries=None,
        image_mask=mask,
        cache=SimpleNamespace(geometry=None, indptr_buffers=None),
    )
    monkeypatch.setattr(indices, "fill_step_indptrs", lambda *a: {"filled": True})
    seen = {}
    with rows_of(metadata, inner, rows=rows):
        seen["mask"] = metadata.image_mask.clone()
        seen["step"] = metadata.step
    torch.testing.assert_close(seen["mask"], mask[:, rows])
    assert seen["step"] is inner and inner.indptrs == {"filled": True}
    assert metadata.image_mask is mask and metadata.step is outer
    # the inner step's indptrs overwrote the outer one's buffers
    assert outer.indptrs == {} and outer.tiles == {}


def test_aux_captures_land_back_at_their_forward_rows():
    """A decoder layer's DSpark capture wrote the tail's rows at the buffer's
    head; they move to the forward rows, an early layer's capture untouched."""
    pytest.importorskip("aiter")
    from atom.models.deepseek_v41 import runtime

    model = runtime.DeepseekV41RuntimeModel.__new__(runtime.DeepseekV41RuntimeModel)
    torch.nn.Module.__init__(model)
    rows = torch.tensor([3, 4, 9, 10, 11])
    aux = [torch.full((16, 2), -1.0), torch.full((16, 2), -2.0)]
    aux[1][:5] = rows[:, None].float()
    model.late_aux_layers, model.aux_buffers = (1,), aux
    model._move_aux_rows(rows)
    torch.testing.assert_close(aux[1][rows], rows[:, None].float().expand(-1, 2))
    assert torch.all(aux[0] == -1.0)


def test_hidden_state_export_refuses_a_replaying_model():
    """TorchSpec export reads every prefill row; replay is fixed at launch, so
    the export refuses a model that replays rather than turning it off under
    a prompt whose earlier chunks already skipped the decoder."""
    pytest.importorskip("aiter")
    from atom.rollout.model_runner_ext import RLHFModelRunner

    runner = SimpleNamespace(label="r", _find_model_attr=lambda name: lambda: True)
    with pytest.raises(ValueError, match="--no-decoder-swa-bounded-replay"):
        RLHFModelRunner.configure_hidden_states(runner, [1], {})


@pytest.mark.parametrize("image", [None, 7])
def test_a_replayed_history_marks_image_positions_dead(image):
    """The Engram history a replayed forward starts at reads the tokens
    before it as a full prefill does: an image position (it carries the
    image id) is DEAD."""
    pytest.importorskip("aiter")
    from atom.model_engine.sequence import Replay
    from atom.model_ops.attentions.deepseek_v41.backend import (
        DeepseekV41MetadataBuilder,
    )
    from tests.model_ops.engram.test_hash_bounds import build, tiny_config

    builder = DeepseekV41MetadataBuilder.__new__(DeepseekV41MetadataBuilder)
    mapping = build(tiny_config())
    builder.engram = SimpleNamespace(mapping=mapping)
    builder.geometry = SimpleNamespace(history_size=mapping.config.max_ngram_size - 1)
    builder.config = SimpleNamespace(image_token_id=image)
    resumes, floors, cached, histories = builder._replay_layout(
        {1: Replay(384, True, (5, 7))},
        np.array([0, 1]),
        [RequestSpan(1, 0, 0, 10, 0), RequestSpan(2, 256, 10, 200, 1)],
    )
    assert (resumes.tolist(), floors.tolist(), cached.tolist()) == (
        [0, 384],
        [0, 256],
        [0, 0],
    )
    assert histories[0].tolist() == [-1, -1]
    assert sorted(histories[1].tolist()) == ([5, 7] if image is None else [-1, 5])
