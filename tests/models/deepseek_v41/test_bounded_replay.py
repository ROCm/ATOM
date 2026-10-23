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
    context = SimpleNamespace(
        is_prefill=True,
        is_dummy_run=False,
        is_draft=False,
        running_tokens_are_unified=False,
    )
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
        # DP attention replays as its rank reported to the group ...
        (
            {
                "fwd.dp_metadata": object(),
                "fwd.dp_stages": SimpleNamespace(applies=True),
            },
            True,
        ),
        # ... and not when the step's sync carried no stage table
        ({"fwd.dp_metadata": object()}, False),
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
        {1: Replay(384, True, (5, 7), 256)},
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


# ── DP attention: the step's DP sync carries each stage's token counts ───


def _batch(lengths, finals, *, prefill=True, dummy=False):
    total = sum(lengths)
    return SimpleNamespace(
        num_scheduled_tokens=np.asarray(lengths, np.int32),
        is_final_chunk=finals,
        total_tokens_num=total,
        total_tokens_num_prefill=total if prefill else 0,
        is_dummy_run=dummy,
    )


def _report(lengths, finals, **kw):
    from atom.models.deepseek_v41.bounded_replay import decoder_stage_report

    return decoder_stage_report(_batch(lengths, finals, **kw), 128)


def _stage_forward(sizes, rank, rows, report):
    """A forward context of a DP step whose sync carried stage ``rows``
    (None: no rank's report applies), built the way `set_forward_context`
    builds it."""
    from atom.utils.forward_context import (
        DPMetadata,
        ForwardContext,
        _bounded_replay_dp_stages,
    )

    parallel = SimpleNamespace(data_parallel_size=len(sizes), data_parallel_rank=rank)
    sync = SimpleNamespace(dp_stage_rows=rows)
    context = SimpleNamespace(
        scheduled_tokens=sizes[rank],
        running_tokens_across_dp=tuple(sizes),
        running_tokens_are_unified=False,
    )
    return ForwardContext(
        context=context,
        dp_metadata=DPMetadata.make(
            parallel, sizes[rank], torch.tensor(sizes, dtype=torch.int32)
        ),
        dp_stages=_bounded_replay_dp_stages(parallel, report, sync),
    )


@pytest.mark.parametrize(
    "kw, expected",
    [
        # a decoding (or dummy) rank runs all its rows through both stages
        ({"lengths": [40], "finals": None, "prefill": False}, (0, 1, 40, 40)),
        ({"lengths": [6], "finals": None, "dummy": True}, (0, 1, 6, 6)),
        # two finishing prompts: tails capped at the window, one row each
        ({"lengths": [500, 64], "finals": [True, True]}, (1, 1, 128 + 64, 2)),
        # a zero-row scheduler entry is no request
        ({"lengths": [500, 0, 64], "finals": [True, True, True]}, (1, 1, 192, 2)),
        # a non-final chunk: one placeholder row, no decoder work of its own
        ({"lengths": [16384], "finals": [False]}, (1, 0, 1, 1)),
    ],
)
def test_a_rank_reports_its_stage_tokens_from_the_batch(kw, expected):
    assert _report(**kw).wire() == expected


def _cpu_step(lengths, positions, finals):
    """A `BatchStep` on the CPU, the fields `build_replay_plan` reads."""
    from atom.model_ops.attentions.deepseek_v41.metadata import BatchStep

    spans, offset = [], 0
    for i, (n, p) in enumerate(zip(lengths, positions)):
        spans.append(RequestSpan(100 + i, p, offset, n, i))
        offset += n
    width = sum(lengths)
    step = BatchStep(
        tuple(spans),
        torch.arange(width, dtype=torch.int64),
        torch.tensor(np.cumsum([0, *lengths]), dtype=torch.int32),
        torch.arange(len(lengths), dtype=torch.int32),
        torch.repeat_interleave(
            torch.arange(len(lengths), dtype=torch.int32), torch.tensor(lengths)
        ),
        torch.zeros(len(lengths), 4, dtype=torch.int32),
        scheduled=width,
    )
    step.request_finals = np.asarray(finals)
    return step


@pytest.mark.parametrize(
    "lengths, finals",
    [
        ([300, 200, 5], [True, False, True]),
        ([300], [False]),
        ([64, 64], [True, True]),
    ],
)
def test_the_plan_runs_what_the_rank_reported(lengths, finals, monkeypatch):
    """The counts a rank reports from its batch are the counts its replay plan
    runs, or its peers' collectives would wait on rows it never sends."""
    from atom.models.deepseek_v41 import bounded_replay
    from atom.models.deepseek_v41.bounded_replay import planned_stage_tokens

    # The Triton row gather, in torch: this runs where Triton is absent.
    monkeypatch.setattr(
        bounded_replay,
        "_gather_rows",
        lambda rows, *tensors, dim=0: tuple(
            t.index_select(dim, rows.long()) for t in tensors
        ),
    )

    step = _cpu_step(lengths, [0] * len(lengths), finals)
    plan = build_replay_plan(step, 128)
    reported = _report(lengths, finals).tokens
    assert planned_stage_tokens(plan, sum(lengths)) == reported
    assert planned_stage_tokens(None, 40) == _report([40], None, prefill=False).tokens


def test_replay_follows_the_ranks_own_report():
    """Under DP the forward acts on what its rank reported, not on a second
    reading of the step: one decision for the report and the forward."""
    from atom.models.deepseek_v41.bounded_replay import replayable

    rows = ((1, 0), (1, 1), (128, 40), (1, 40))
    replaying = _stage_forward([500, 40], 0, rows, _report([500], [True]))
    decoding = _stage_forward([500, 40], 1, rows, _report([40], None, prefill=False))
    assert replayable(replaying, 500)
    assert not replayable(decoding, 40)
    plain = _stage_forward([500, 40], 0, None, _report([500], [True]))
    assert plain.dp_stages is None and not replayable(plain, 500)


def test_the_group_table_comes_from_the_steps_dp_sync():
    rows = ((1, 0), (1, 1), (128, 40), (1, 40))
    stages = _stage_forward([500, 40], 0, rows, _report([500], [True])).dp_stages
    assert stages.applies and stages.active
    assert stages.stages["decoder"].tokens_across_dp == (128, 40)
    assert stages.stages["last_layer"].tokens_across_dp == (1, 40)
    assert stages.stages["decoder"].local_tokens == 128


def test_a_group_with_no_decoder_work_skips_the_decoder():
    """Every rank is on a non-final chunk: nobody has a tail, so no rank runs
    placeholder rows through layers 20..39."""
    rows = ((1, 1), (0, 0), (1, 1), (1, 1))
    stages = _stage_forward(
        [16384, 16384], 0, rows, _report([16384], [False])
    ).dp_stages
    assert stages is not None and not stages.active


def test_a_placeholder_count_equal_to_the_ranks_own_still_replays():
    """A 1-row non-final prefill beside a decoding peer: the placeholder count
    equals the rank's own, yet the rank replays, since the decision is an
    explicit flag, not a changed count."""
    rows = ((1, 0), (0, 1), (1, 40), (1, 40))
    stages = _stage_forward([1, 40], 0, rows, _report([1], [False])).dp_stages
    assert stages.applies and stages.active


def test_a_placeholder_step_cuts_the_image_mask_to_its_rows():
    from dataclasses import dataclass

    from atom.models.deepseek_v41.bounded_replay import placeholder_rows_of

    @dataclass
    class Step:
        requests: tuple

    step = Step(requests=(object(),))
    mask = torch.tensor([[True, False, True, True]])
    metadata = SimpleNamespace(step=step, queries=object(), image_mask=mask)
    outer_queries = metadata.queries
    rows = torch.tensor([0, 1])
    with placeholder_rows_of(metadata, rows):
        assert metadata.step.requests == ()
        assert metadata.queries is None
        assert metadata.image_mask.tolist() == [[True, False]]
    assert metadata.step is step and metadata.queries is outer_queries
    assert metadata.image_mask is mask


def test_a_stage_sizes_the_moe_gather_and_restores_it():
    """`use_dp_stage` switches to the metadata prepared with the forward
    context, and back; outside a replaying step it changes nothing."""
    rows = ((1, 0), (1, 1), (128, 40), (1, 40))
    forward = _stage_forward([500, 40], 1, rows, _report([40], None, prefill=False))
    outer = forward.dp_metadata
    with forward.use_dp_stage("decoder"):
        assert forward.dp_metadata.get_sizes_across_dp() == [128, 40]
        assert forward.context.scheduled_tokens == 40
        assert forward.context.running_tokens_across_dp == (128, 40)
        with forward.use_dp_stage("last_layer"):
            assert forward.dp_metadata.get_sizes_across_dp() == [1, 40]
        assert forward.dp_metadata.get_sizes_across_dp() == [128, 40]
    assert forward.dp_metadata is outer
    assert forward.context.running_tokens_across_dp == (500, 40)
    plain = _stage_forward([500, 40], 1, None, None)
    with plain.use_dp_stage("decoder"):
        assert plain.dp_metadata.get_sizes_across_dp() == [500, 40]


def test_decoder_stages_run_each_layer_under_its_counts():
    """The runner every decoder path goes through (tail, placeholder and full
    rows): layers 20..38 under the decoder counts and layer 39 under the
    last-layer counts, each once -- a skipped layer or a wrong table leaves a
    peer waiting in a mismatched collective."""
    from atom.models.deepseek_v41.bounded_replay import run_decoder_stages

    rows = ((1, 0), (0, 1), (1, 128), (1, 1))
    forward = _stage_forward([16384, 500], 0, rows, _report([16384], [False]))
    log = []

    def layer(i):
        def run(state=None):
            log.append((i, tuple(forward.dp_metadata.get_sizes_across_dp())))
            return state

        return run

    def middle(state):
        for i in range(21, 39):
            state = layer(i)(state)
        return state

    run_decoder_stages(forward, layer(20), middle, layer(39))
    assert [i for i, _ in log] == list(range(20, 40))
    assert all(sizes == (1, 128) for i, sizes in log if i < 39)
    assert log[-1] == (39, (1, 1))
    assert forward.dp_metadata.get_sizes_across_dp() == [16384, 500]


@cuda
@pytest.mark.parametrize("fuse_quant", [False, True])
@pytest.mark.parametrize("has_tail", [False, True])
def test_runtime_replay_gathers_attention_inputs_under_dp(
    monkeypatch, fuse_quant, has_tail
):
    """Both a real tail and a DP placeholder keep the quantization paired with
    its BF16 rows and mHC seam while joining the agreed decoder stages."""
    from atom.model_ops.deepseek_v41.mhc import SinglePassHCState
    from atom.models.deepseek_v41 import runtime
    from atom.utils import forward_context

    cache = _cache()
    step = _forward_step(cache, [6, 4], [0, 0], [False, has_tail])
    plan = build_replay_plan(step, 128)
    sizes = ((1, 0), (int(has_tail), 1), (4 if has_tail else 1, 40), (1, 40))
    forward = _stage_forward([10, 40], 0, sizes, _report([6, 4], [False, has_tail]))
    mask = torch.arange(10, device="cuda")[None] % 2 == 0
    metadata = forward.attn_metadata = SimpleNamespace(
        cache=cache, step=step, queries=None, image_mask=mask, engram_embeddings=None
    )
    monkeypatch.setattr(forward_context, "_forward_context", forward)
    hidden = torch.arange(10 * 512, device="cuda").reshape(1, 10, 512).bfloat16()
    quant = (torch.arange(hidden.numel(), device="cuda") % 239).to(torch.uint8)
    quant = quant.reshape(hidden.shape).view(torch.float8_e4m3fn)
    scale = (torch.arange(160, device="cuda") % 64 + 96).to(torch.uint8)
    scale = scale.reshape(1, 10, 16).view(torch.float8_e8m0fnu)
    normed = (hidden, quant, scale) if fuse_quant else hidden
    state = SinglePassHCState.from_embeddings(hidden, 4)
    combination = torch.arange(10 * 4 * 4, device="cuda").reshape(1, 10, 4, 4).float()
    seam = (state.residual, state.pre_mix, state.pre_mix + 1, combination)
    taken = plan.tail_rows if has_tail else torch.zeros(1, device="cuda").long()
    expected_hidden = hidden.index_select(1, taken.long())
    calls = []

    class Attention(torch.nn.Module):
        def _compress_batch(self, actual, *args):
            assert actual is hidden  # compression still sees every BF16 row
            calls.append("compress")

        def forward(self, actual, hidden_scale, *args, hidden_fp8=None):
            assert metadata.step.requests
            calls.append("attention")
            return actual

    layer = runtime.RuntimeBlock.__new__(runtime.RuntimeBlock)
    torch.nn.Module.__init__(layer)
    layer.layer_name, layer.attn = "replay_test", Attention()
    layer.prepare_attention = lambda *args: (normed, *seam)
    forward.no_compile_layers = {layer.layer_name: (layer, None)}

    def attention_forward(actual, *args):
        sources = normed if fuse_quant else (normed,)
        outputs = actual if fuse_quant else (actual,)
        for source, output in zip(sources, outputs, strict=True):
            assert output.dtype == source.dtype
            torch.testing.assert_close(
                output.view(torch.uint8),
                source.view(torch.uint8).index_select(1, taken.long()),
                rtol=0,
                atol=0,
            )
        return runtime.RuntimeBlock.attention_forward(layer, actual, *args)

    def ffn_forward(output, *actual_seam):
        for source, actual in zip(seam, actual_seam, strict=True):
            torch.testing.assert_close(actual, source.index_select(1, taken.long()))
        torch.testing.assert_close(
            output, expected_hidden if has_tail else torch.zeros_like(expected_hidden)
        )
        assert forward.dp_metadata.get_sizes_across_dp() == [taken.numel(), 40]
        assert torch.equal(metadata.image_mask, mask.index_select(1, taken.long()))
        calls.append("ffn")
        return SinglePassHCState.from_embeddings(output, 4)

    layer.attention_forward, layer.ffn_forward = attention_forward, ffn_forward
    model = runtime.DeepseekV41RuntimeModel.__new__(runtime.DeepseekV41RuntimeModel)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(hidden_size=512)
    model.layers = torch.nn.ModuleList([layer])
    model.decoder_start, model.last_layer = 0, 1
    model.decoder_specs, model.late_aux_layers = (), ()
    model._rope = lambda _: None

    def late_layers(state):
        assert forward.dp_metadata.get_sizes_across_dp() == [taken.numel(), 40]
        calls.append("late")
        return state

    def last_layer(index, state, queries=None):
        assert forward.dp_metadata.get_sizes_across_dp() == [1, 40]
        assert metadata.queries is queries
        calls.append("last")
        output = state.collapse()
        if queries is not None:
            output = output.index_select(1, queries.rows.long())
        return SinglePassHCState.from_embeddings(output, 4)

    model._late_layers, model._layer = late_layers, last_layer
    actual = model._replayed(state, step.width, plan)
    expected = torch.zeros_like(hidden.squeeze(0))
    if has_tail:
        expected[9] = hidden[0, 9]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert calls == [
        "compress",
        *(["attention"] if has_tail else []),
        "ffn",
        "late",
        "last",
    ]
    assert forward.dp_metadata.get_sizes_across_dp() == [10, 40]
    assert metadata.step is step and metadata.queries is None
    assert metadata.image_mask is mask
