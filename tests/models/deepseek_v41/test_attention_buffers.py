# SPDX-License-Identifier: MIT
"""Reuse metadata storage and bind only outputs read across AF regions."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("aiter")


@pytest.fixture
def backend():
    from atom.model_ops.attentions.deepseek_v41.cache import PagedAttentionCache
    from atom.models.deepseek_v41.config import AttentionMode as Mode
    from atom.models.deepseek_v41.config import LayerAttentionSpec as Spec
    from atom.utils.cuda_graph import StableOutputs

    cache = PagedAttentionCache.__new__(PagedAttentionCache)
    cache.geometry = SimpleNamespace(
        index_fp4=True,
        index_topk=4,
        index_block_rows=8,
        index_blocks_per_page=lambda ratio: 1,
        window_size=4,
        packed=True,
        batch_topk=lambda ratio: 4,
    )
    cache.max_tokens, cache.num_slots, cache.candidate_count = 8, 8, 2
    cache.result_buffers = StableOutputs()
    cache.workspace = None
    cache.topology = (
        Spec(0, 1, Mode.FULL, 0, 0, 0),
        Spec(1, 1, Mode.REUSE, 0, 0, 0),
        Spec(2, 1, Mode.REINDEX, 0, 2, 0, 2, 2),
        Spec(3, 1, Mode.REUSE, 0, 2, 0, 2, 2),
        Spec(4, 1, Mode.REINDEX, 0, 4, 0),
    )
    return cache


def _step(width, *, decode=False, namespace="main"):
    from atom.model_ops.attentions.deepseek_v41.metadata import BatchStep, RequestSpan

    requests = (
        tuple(RequestSpan(i, 32, i, 1, i) for i in range(width))
        if decode
        else (RequestSpan(0, 32, 0, width, 0),)
    )
    return BatchStep(
        requests,
        torch.arange(width),
        (
            torch.arange(width + 1, dtype=torch.int32)
            if decode
            else torch.tensor([0, width], dtype=torch.int32)
        ),
        torch.arange(len(requests), dtype=torch.int32),
        (
            torch.arange(width, dtype=torch.int32)
            if decode
            else torch.zeros(width, dtype=torch.int32)
        ),
        torch.ones(len(requests), 2, dtype=torch.int32),
        scheduled=width,
        max_q_len=1 if decode else width,
        is_prefill=not decode,
        visible={1: torch.ones(width, dtype=torch.int32)},
        buffer_namespace=namespace,
        indptrs={1: (torch.arange(width + 1), torch.empty(0, dtype=torch.int32), 4)},
    )


@pytest.mark.parametrize("decode", [False, True])
def test_only_shared_results_get_persistent_storage(backend, decode):
    step = _step(3, decode=decode)
    names = (
        "positions",
        "cu_seqlens_q",
        "slots",
        "batch_ids",
        "block_tables",
        "indptrs",
    )
    inputs = {name: getattr(step, name) for name in names}
    backend.update_step(step, fixed_results=True)
    assert all(getattr(step, name) is value for name, value in inputs.items())
    assert set(step.selected) == {0, 2}  # Layer 4's selection stays local.
    assert set(step.candidates) == {0}
    assert step.candidate_producers == {0: 2}
    assert not step.candidate_blocks and not step.group_indices


def test_single_reindex_keeps_translation_local_and_tail_keeps_carried_results(backend):
    step = _step(3, namespace="tail")
    step.layer_specs = backend.topology[2:4]
    # Candidates produced outside this view have already been gathered by the tail.
    carried = torch.full((3, 2), 17, dtype=torch.int32)
    step.candidates[0] = carried
    backend.update_step(step, fixed_results=True)
    assert step.candidates[0] is carried
    assert set(step.selected) == {2}
    assert not step.candidate_blocks and not step.candidate_producers
    assert not any(
        k[0][1].startswith("candidate_") for k in backend.result_buffers._slots
    )


@pytest.mark.parametrize(
    "shared_selection,shared_candidates,tail,decode",
    [
        pytest.param(False, False, False, True, id="local-decode"),
        pytest.param(True, True, False, False, id="shared-prefill"),
        pytest.param(True, False, True, True, id="tail-selection"),
        pytest.param(False, True, True, False, id="tail-candidates"),
    ],
)
def test_warmup_results_do_not_become_fixed_scorer_destinations(
    backend, shared_selection, shared_candidates, tail, decode
):
    from atom.model_ops.deepseek_v41.paged_scoring import _score_outputs

    producer, reuse, *_, reindex = backend.topology
    topology = (producer,)
    if shared_selection:
        topology += (reuse,)
    if shared_candidates:
        topology += (reindex,)
    step = _step(3, decode=decode)
    if tail:
        step.layer_specs = topology
    else:
        backend.topology = topology
    backend.update_step(step, fixed_results=True)
    fixed = backend.selection_outputs(producer, step)
    for iteration in range(3):
        step.begin_forward()
        outputs = backend.selection_outputs(producer, step)
        for actual, saved, shared in zip(
            outputs, fixed, (shared_selection, shared_candidates)
        ):
            if shared:
                assert actual.data_ptr() == saved.data_ptr()
            else:
                assert actual is None
        selected, candidates = _score_outputs(3, 4, 2, "cpu", outputs)
        selected.fill_(iteration)
        candidates.fill_(iteration + 10)
        backend.store_selection(producer, step, selected, candidates)
        assert step.selected[producer.layer_id].data_ptr() == selected.data_ptr()
        assert step.candidates[producer.layer_id] is candidates


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs")
def test_full_graph_owns_local_selections_after_capture_step_is_released(backend):
    import gc
    import weakref

    from atom.model_ops.deepseek_v41.paged_scoring import _score_outputs

    producer = backend.topology[0]
    backend.topology = (producer,)
    step = _step(3, decode=True)
    backend.update_step(step, fixed_results=True)
    inputs = torch.ones(3, 4, device="cuda", dtype=torch.int32)
    result = torch.empty_like(inputs)

    def forward(step):
        selected, candidates = _score_outputs(
            3, 4, 2, "cuda", backend.selection_outputs(producer, step)
        )
        selected.copy_(inputs)
        candidates.copy_(inputs[:, :2] + 10)
        backend.store_selection(producer, step, selected, candidates)
        result.copy_(selected + candidates[:, :1])

    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    try:
        with torch.cuda.stream(stream):
            forward(step)
            warm_selected, warm_candidates = step.selected[0], step.candidates[0]
            step.begin_forward()
            # Fail before capture on the old implementation, without recording
            # a graph that writes into a potentially released warmup allocation.
            assert backend.selection_outputs(producer, step) == (None, None)
            with torch.cuda.graph(graph, stream=stream):
                forward(step)
            assert step.selected[0].data_ptr() != warm_selected.data_ptr()
            assert step.candidates[0].data_ptr() != warm_candidates.data_ptr()
        torch.cuda.current_stream().wait_stream(stream)
        step_ref = weakref.ref(step)
        del step
        gc.collect()
        assert step_ref() is None
        warm_selected.fill_(-111)
        warm_candidates.fill_(-222)
        # Ordinary allocations cannot take over the graph's local temporaries.
        churn = [torch.full_like(inputs, -333) for _ in range(8)]
        for value in (5, 11):
            inputs.fill_(value)
            graph.replay()
            torch.testing.assert_close(result, torch.full_like(result, 2 * value + 10))
            assert warm_selected.eq(-111).all() and warm_candidates.eq(-222).all()
            assert all(t.eq(-333).all() for t in churn)
    finally:
        graph.reset()


def test_existing_tile_workspace_is_used_without_another_slot(backend, monkeypatch):
    from atom.model_ops.attentions.deepseek_v41 import cache as cache_module

    table = torch.empty(8, 2, dtype=torch.int32)
    backend.geometry.index_fp4 = False
    backend.geometry.block_size = 64
    backend.geometry.index_blocks_per_page = lambda ratio: 1
    backend.workspace = SimpleNamespace(
        unit_table=lambda ratio, tokens, width: table[:tokens]
    )

    def unit_table(tables, batch_ids, units, *, workspace, ratio):
        assert workspace is backend.workspace
        return workspace.unit_table(ratio, batch_ids.numel(), tables.shape[1])

    monkeypatch.setattr(cache_module, "unit_table", unit_table)
    step = _step(3)
    backend.update_step(step, fixed_results=True)
    assert step.tiles[1].data_ptr() == table.data_ptr()
    assert not any(k[0][1] == "tiles" for k in backend.result_buffers._slots)


def test_shared_accessors_work_after_producer_replay_without_python(
    backend, monkeypatch
):
    from atom.model_ops.attentions.deepseek_v41 import cache as cache_module

    def translate(ids, *args, out, rows_per_block):
        out[0].copy_(ids + 100)
        out[1].fill_(ids[0, 0])

    def indices(spec, step, first, layers, *, out):
        out[0].fill_(step.selected[first][0, 0, 0])
        return out

    monkeypatch.setattr(cache_module, "bind_candidates", translate)
    monkeypatch.setattr(backend, "_index_group", indices)
    first = _step(3, decode=True)
    backend.update_step(first, fixed_results=True)
    first.candidates[0].fill_(1)
    first.selected[2].fill_(2)
    captured_blocks = backend.candidate_blocks(first, 0, 1, layer=2)
    backend.attention_indices(backend.topology[2], first)
    captured_indices = first.group_indices[2][0]

    for value in (7, 19):
        current = _step(3, decode=True)
        backend.update_step(current, fixed_results=True)
        assert not current.candidate_blocks and not current.group_indices
        current.candidates[0].fill_(value)
        current.selected[2].fill_(value + 1)
        # A recorded producer updates its captured output addresses, without
        # publishing dictionaries on the new BatchStep.
        captured_blocks.table.copy_(current.candidates[0] + 100)
        captured_blocks.context.fill_(value)
        captured_indices.fill_(value + 1)
        blocks = backend.candidate_blocks(current, 0, 1, layer=4)
        selected, *_ = backend.attention_indices(backend.topology[3], current)
        assert blocks.table.data_ptr() == captured_blocks.table.data_ptr()
        assert blocks.table.eq(value + 100).all()
        assert selected.eq(value + 1).all()
        # Conversely, eager producers must refresh the captured consumers.
        current.candidates[0].fill_(value + 2)
        current.selected[2].fill_(value + 3)
        backend.candidate_blocks(current, 0, 1, layer=2)
        backend.attention_indices(backend.topology[2], current)
        assert captured_blocks.table.eq(value + 102).all()
        assert captured_indices.eq(value + 3).all()


def test_step_updates_keep_microbatch_and_query_indptrs_separate(backend, monkeypatch):
    from atom.model_ops.attentions.deepseek_v41 import cache as cache_module

    events = []
    backend.indptr_buffers = object()
    microbatch_buffers = object()
    backend.publish_replay_layout = lambda step, bs: events.append(("layout", step, bs))

    def fill(step, geometry, buffers):
        events.append(("indices", step, buffers))
        return {"owner": buffers}

    monkeypatch.setattr(cache_module, "fill_step_indptrs", fill)
    # Only GPU transport is substituted: exercise the production view routing.
    main = SimpleNamespace(positions=SimpleNamespace(is_cuda=True))
    backend.update_step(main, running_bs=3, indptr_buffers=microbatch_buffers)
    assert events == [("layout", main, 3), ("indices", main, microbatch_buffers)]
    assert main.indptrs == {"owner": microbatch_buffers}

    query = SimpleNamespace()
    backend.update_step(main, queries=SimpleNamespace(index=query))
    assert events[-1] == ("indices", query, backend.indptr_buffers)
    assert len(events) == 3  # Tail floors were already published by the gather.
    assert query.indptrs == {"owner": backend.indptr_buffers}
    assert main.indptrs == {"owner": microbatch_buffers}


@pytest.mark.parametrize("shared", [False, True])
def test_indexer_publishes_kernel_outputs_without_a_copy(backend, monkeypatch, shared):
    from atom.models.deepseek_v41 import attention

    indexer = attention.Indexer.__new__(attention.Indexer)
    torch.nn.Module.__init__(indexer)
    indexer.spec = backend.topology[0]
    indexer.head_dim, indexer.heads = 2, 1
    indexer.topk, indexer.topk_blocks, indexer.block_size = 4, 2, 8
    indexer._plane = lambda *args: (None, None, None)
    step = _step(3)
    if shared:
        backend.update_step(step, fixed_results=True)
    backend.geometry.index_fp4 = False
    selected = step.selected.get(0)
    candidates = step.candidates.get(0)
    calls = []

    def score(*args, out, **kwargs):
        if shared:
            assert out[0].data_ptr() == selected.data_ptr()
            assert out[1] is candidates
        else:
            assert out is None
            out = (
                torch.empty(3, 4, dtype=torch.int32),
                torch.empty(3, 2, dtype=torch.int32),
            )
        out[0].fill_(len(calls) + 1)
        out[1].fill_(len(calls) + 10)
        calls.append(out)
        return out

    def no_copy(*args, **kwargs):
        pytest.fail("Publishing scorer outputs must not issue a copy")

    monkeypatch.setattr(attention, "score_topk_paged", score)
    monkeypatch.setattr(torch.Tensor, "copy_", no_copy)
    for _ in range(2):
        indexer.score(None, None, backend, step)
        assert step.selected[0].data_ptr() == calls[-1][0].data_ptr()
        assert step.candidates[0] is calls[-1][1]
    if shared:
        assert selected.eq(2).all() and candidates.eq(11).all()


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("fixed_selection", [False, True])
def test_scorer_kernels_write_supplied_outputs(packed, fixed_selection, monkeypatch):
    from atom.model_ops.deepseek_v41 import paged_scoring as scoring

    rows, topk, count = 5, 4, 2
    selected = (
        torch.full((rows, topk), -99, dtype=torch.int32) if fixed_selection else None
    )
    chosen = torch.full((rows, count), -99, dtype=torch.int32)
    logits = torch.empty(rows, 8)
    visible = torch.arange(rows, dtype=torch.int32) + 8
    bands = ((0, 2), (2, rows))
    workspace = SimpleNamespace(
        packed=packed,
        logits_rows=lambda *args: 2,
        logits=lambda n, width: logits[:n, :width],
        packed_logits=lambda: logits.flatten(),
        packed_maxima=lambda: logits.flatten(),
    )
    layout = (
        SimpleNamespace(
            visible=visible,
            offsets=torch.zeros(rows, dtype=torch.int32),
            block_offsets=torch.zeros(rows, dtype=torch.int32),
            bands=bands,
        )
        if packed
        else None
    )
    ragged = (
        SimpleNamespace(
            block_tables=torch.empty(1, 4),
            pages_per_block=1,
            band=lambda *args: None,
        )
        if packed
        else None
    )
    units = SimpleNamespace(values=torch.empty(4, 2), fp4=False)
    query, weights = torch.empty(rows, 1, 2), torch.empty(rows, 1)
    tiles = torch.empty(rows, 4)
    monkeypatch.setattr(scoring, "_band_logits", lambda *args, **kwargs: None)

    def topk_kernel(scores, next_n, seen, out, *args, **kwargs):
        out.copy_(seen[:, None].expand_as(out))

    def candidates_kernel(scores, seen, block_size, out):
        out.copy_((seen + 100)[:, None].expand_as(out))

    monkeypatch.setattr(scoring, "top_k_per_row_decode", topk_kernel)
    monkeypatch.setattr(scoring, "pick_candidate_blocks", candidates_kernel)
    monkeypatch.setattr(
        scoring,
        "pick_candidate_blocks_packed",
        lambda scores, offsets, maxima, blocks, seen, size, out, width: candidates_kernel(
            scores, seen, size, out
        ),
    )
    real_empty = torch.empty
    allocations = []

    def empty(*args, **kwargs):
        allocations.append(args)
        return real_empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", empty)
    for delta in (1, 3):
        visible.add_(delta)
        actual = scoring.score_topk_quantized(
            query,
            weights,
            units,
            tiles,
            visible,
            topk=topk,
            candidate_count=count,
            workspace=workspace,
            packed=layout,
            ragged=ragged,
            out=(selected, chosen),
        )
        if fixed_selection:
            assert actual[0] is selected
        assert actual[1] is chosen
        torch.testing.assert_close(actual[0], visible[:, None].expand(rows, topk))
        torch.testing.assert_close(chosen, (visible + 100)[:, None].expand(rows, count))
    assert len(allocations) == (0 if fixed_selection else 2)


@pytest.mark.parametrize("bad", ["shape", "dtype", "stride", "unexpected_candidates"])
def test_invalid_scorer_outputs_fail_before_scoring(bad):
    from atom.model_ops.deepseek_v41.paged_scoring import _score_outputs

    selected = torch.empty(3, 4, dtype=torch.int32)
    chosen = None
    if bad == "shape":
        selected = selected[:2]
    elif bad == "dtype":
        selected = selected.float()
    elif bad == "stride":
        selected = torch.empty(3, 8, dtype=torch.int32)[:, ::2]
    else:
        chosen = torch.empty(3, 2, dtype=torch.int32)
    with pytest.raises(ValueError, match="output"):
        _score_outputs(3, 4, 0, "cpu", (selected, chosen))
