# SPDX-License-Identifier: MIT
"""Prefill microbatches preserve histories, cache plans and cross-layer state."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("aiter")

from atom.model_ops.attentions.deepseek_v41.backend import DeepseekV41MetadataBuilder
from atom.model_ops.attentions.deepseek_v41.cache import PagedAttentionCache
from atom.model_ops.attentions.pool_layout.v41_pool_geometry import V41PoolGeometry
from atom.utils.forward_context import Context, ForwardContext, _forward_context_local
from atom.utils.tbo.ubatch_splitting import UBatchSlice, _split_prefill_token_midpoint
from atom.utils.tbo.ubatch_wrapper import UBatchWrapper
from tests.attentions.deepseek_v41.helpers import metadata_buffers


def attach_ubatch_buffers(builder, max_tokens):
    """What `DeepseekV41MetadataBuilder.__init__` sets up under prefill TBO,
    for a builder assembled without it: the `ub{i}_` step buffers in the
    runner's forward_vars and each microbatch's indptrs."""
    builder.max_num_batched_tokens = max_tokens
    builder.model_runner.forward_vars.update(builder._ubatch_step_buffers())
    builder._ubatch_indptrs = [
        {
            ratio: tuple(
                torch.empty(max_tokens + 1, dtype=torch.int32, device=builder.device)
                for _ in range(2)
            )
            for ratio in builder.geometry.layer_ratios
        }
        for _ in range(builder._NUM_TBO_UBATCHES)
    ]


def make_parent(device, lengths=(10, 4), starts=(3, 8)):
    geo = V41PoolGeometry(
        4,
        ((0, 2), (2, 1)),
        32,
        4,
        512,
        32,
        layer_ratios=(0, 1, 2),
        index_topk=4,
    )
    cache = PagedAttentionCache(geo, 8, 4, device, max_tokens=32)
    builder = DeepseekV41MetadataBuilder.__new__(DeepseekV41MetadataBuilder)
    builder.device, builder.geometry, builder.block_size = device, geo, 32
    builder.cache = cache
    builder.model_runner = SimpleNamespace(
        forward_vars=metadata_buffers(4, 32, 4, device, geo)
    )
    attach_ubatch_buffers(builder, 32)
    batch = SimpleNamespace(
        is_dummy_run=False,
        state_slots_committed=[3, 1][: len(lengths)],
        req_ids=tuple(range(len(lengths))),
        num_scheduled_tokens=lengths,
        context_lens=tuple(a + b for a, b in zip(starts, lengths)),
        block_tables=((0, 1, 2, 3), (4, 5, 6, 7))[: len(lengths)],
        total_tokens_num=sum(lengths),
        total_seqs_num=len(lengths),
    )
    # The runner publishes the query prefix before the public builder entry.
    cu = builder.model_runner.forward_vars["cu_seqlens_q"]
    cu.np[0] = 0
    for i, length in enumerate(lengths):
        cu.np[i + 1] = cu.np[i] + length
    cu.copy_to_gpu(len(lengths) + 1)
    parent, _ = builder.prepare_prefill(batch, len(lengths))
    parent.engram_embeddings = {
        layer: torch.arange(sum(lengths) * 4, device=device).view(1, -1, 4) + layer
        for layer in (1, 3)
    }
    return builder, parent


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("cut", [3, 7, 10, 11])
def test_prefill_slices_keep_absolute_positions_and_independent_plans(device, cut):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("GPU required")
    builder, parent = make_parent(device)
    steps = []
    before = {
        name: value.gpu.clone()
        for name, value in builder.model_runner.forward_vars.items()
    }
    for i, (start, end, rs) in enumerate(
        (
            (0, cut, slice(0, 1 if cut <= 10 else 2)),
            (cut, 14, slice(0 if cut < 10 else 1, 2)),
        )
    ):
        part = UBatchSlice(rs, slice(start, end))
        child = builder.build_ubatch_prefill_metadata(
            parent, part, rs.stop - rs.start, i
        )
        steps.append(child.step)
        assert child.cache is parent.cache
        assert child.step.width == end - start
        torch.testing.assert_close(
            child.step.positions, parent.step.positions[start:end]
        )
        assert child.step.slots.tolist() == parent.step.slots[rs].tolist()
        torch.testing.assert_close(
            child.step.block_tables, parent.step.block_tables[rs]
        )
        assert child.step.cu_seqlens_q.tolist() == (
            [s.offset for s in child.step.requests] + [end - start]
        )
        for layer, rows in parent.engram_embeddings.items():
            torch.testing.assert_close(
                child.engram_embeddings[layer], rows[:, start:end]
            )
            assert (
                child.engram_embeddings[layer].untyped_storage().data_ptr()
                == rows.untyped_storage().data_ptr()
            )
        for ratio, plan in child.step.plans.items():
            expected = [
                [span.offset + j, bid, span.position + j]
                for bid, span in enumerate(child.step.requests)
                for j in range(span.length)
                if (span.position + j + 1) % ratio == 0
            ]
            assert plan.compress_plan_gpu[: plan.num_compress, :3].tolist() == expected
            torch.testing.assert_close(
                child.step.visible[ratio], ((child.step.positions + 1) // ratio).int()
            )
    # The parent's step buffers are untouched; the microbatches write only
    # their own `ub{i}_` copies.
    for name, value in builder.model_runner.forward_vars.items():
        if not name.startswith(("ub0_", "ub1_")):
            torch.testing.assert_close(value.gpu, before[name], rtol=0, atol=0)
    for ratio in steps[0].plans:
        assert (
            steps[0].plans[ratio].compress_plan_gpu.data_ptr()
            != steps[1].plans[ratio].compress_plan_gpu.data_ptr()
        )
    for ratio in steps[0].indptrs:
        assert (
            steps[0].indptrs[ratio][0].data_ptr()
            != steps[1].indptrs[ratio][0].data_ptr()
        )
        assert (
            steps[0].indptrs[ratio][0].data_ptr()
            != parent.step.indptrs[ratio][0].data_ptr()
        )
    for field in ("selected", "candidates", "tiles"):
        getattr(steps[0], field)[2] = torch.tensor([17])
        steps[1].begin_forward()
        assert getattr(steps[0], field)[2].item() == 17
    first = steps[1].requests[0]
    plan = steps[1].plans[2]
    if plan.num_compress and first.length >= 2:
        assert plan.compress_plan_gpu[0, 3].item() == first.position % 2
    assert parent.cache.pending is None


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_prefill_children_publish_independent_packed_score_plans(device):
    """Main's packed scorer must see each child's absolute rows and bands."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("GPU required")
    from dataclasses import replace

    from atom.model_ops.attentions.deepseek_v41.score_planner import (
        ScorePlanner,
        score_layout,
    )
    from atom.model_ops.deepseek_v41.score_workspace import ScoreWorkspace

    builder, parent = make_parent(device)
    workspace = ScoreWorkspace(
        replace(builder.geometry, index_fp4=True, index_dim=128, index_block_rows=8),
        32,
        4,
        device,
        pages=8,
    )
    builder.add_step_planner(ScorePlanner(workspace, (1, 2), 32))
    attach_ubatch_buffers(builder, 32)
    parent.cache.workspace = workspace
    parts = (
        UBatchSlice(slice(0, 1), slice(0, 7)),
        UBatchSlice(slice(0, 2), slice(7, 14)),
    )
    first = builder.build_ubatch_prefill_metadata(parent, parts[0], 1, 0)
    snapshots = {name: rows.clone() for name, rows in first.step.planned.items()}
    second = builder.build_ubatch_prefill_metadata(parent, parts[1], 2, 1)
    for name, expected in snapshots.items():
        torch.testing.assert_close(first.step.planned[name], expected)
        assert (
            first.step.planned[name].data_ptr() != second.step.planned[name].data_ptr()
        )
    for child, part in zip((first, second), parts):
        for ratio in (1, 2):
            layout = score_layout(child.step, ratio)
            expected = ((parent.step.positions[part.token_slice] + 1) // ratio).int()
            torch.testing.assert_close(layout.visible, expected)
            spans = (expected + 63) // 64 * 64
            offsets = torch.cumsum(spans, dim=0) - spans
            torch.testing.assert_close(layout.offsets, offsets.int())
            torch.testing.assert_close(
                layout.block_offsets, (offsets // workspace.block_rows).int()
            )
            assert layout.bands == ((0, 7),)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_kv_release_keeps_persistent_ubatch_buffers_for_rebind(device):
    """KV views can die while fixed-address execution metadata is reused."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("GPU required")
    import gc
    import weakref

    builder, parent = make_parent(device)
    buffers = builder._ubatch_buffers(0)
    indptrs = builder._ubatch_indptrs[0]
    pointers = {name: buf.gpu.data_ptr() for name, buf in buffers.items()}
    cache_ref = weakref.ref(builder.cache)
    builder.copies = object()
    del parent
    builder.release_kv_pools()
    gc.collect()
    assert cache_ref() is None
    assert builder.cache is builder.copies is None
    assert builder._ubatch_indptrs[0] is indptrs
    assert {
        name: buf.gpu.data_ptr() for name, buf in builder._ubatch_buffers(0).items()
    } == pointers

    builder.cache = PagedAttentionCache(builder.geometry, 8, 4, device, max_tokens=32)
    batch = SimpleNamespace(
        is_dummy_run=False,
        state_slots_committed=[3, 1],
        req_ids=(0, 1),
        num_scheduled_tokens=(10, 4),
        context_lens=(13, 12),
        block_tables=((0, 1, 2, 3), (4, 5, 6, 7)),
        total_tokens_num=14,
        total_seqs_num=2,
    )
    parent, _ = builder.prepare_prefill(batch, 2)
    part = UBatchSlice(slice(0, 2), slice(7, 14))
    child = builder.build_ubatch_prefill_metadata(parent, part, 2, 0)
    torch.testing.assert_close(child.step.positions, parent.step.positions[7:14])
    assert child.cache is builder.cache
    assert {
        name: buf.gpu.data_ptr() for name, buf in builder._ubatch_buffers(0).items()
    } == pointers


def test_exceptional_exit_orders_comm_stream_before_compute(monkeypatch):
    """A ubatch unwinding on the comm stream still fences its own buffers.

    The parent records each child's reuse event on the compute stream, so a
    comm stream left unordered lets that event report complete while comm
    kernels are still reading the child's buffers -- and the next forward
    overwrites them.
    """
    import threading

    from atom.utils.tbo import ubatching

    calls = []

    class FakeStream:
        def __init__(self, name):
            self.name = name

        def wait_event(self, event):
            calls.append(("wait", self.name, event.name))

    class FakeEvent:
        def __init__(self, name):
            self.name = name

        def record(self, stream):
            calls.append(("record", self.name, stream.name))

    compute, comm = FakeStream("compute"), FakeStream("comm")
    monkeypatch.setattr(torch.cuda, "set_stream", lambda stream: None)
    monkeypatch.setitem(ubatching._THREAD_ID_TO_CONTEXT, threading.get_ident(), 0)
    monkeypatch.setattr(ubatching, "_CURRENT_CONTEXTS", [None])

    ctx = ubatching.TBOContext(
        ubatch_id=0,
        compute_stream=compute,
        comm_stream=comm,
        forward_context=None,
        ready_barrier=None,
        cpu_wait_event=threading.Event(),
        cpu_signal_event=threading.Event(),
        gpu_comm_done_event=FakeEvent("comm_done"),
        gpu_compute_done_event=FakeEvent("compute_done"),
    )
    # Where a forward that raised mid-communication leaves us.
    ctx.current_stream = comm

    assert ctx.__exit__(RuntimeError, RuntimeError("failed child"), None) is False

    assert ("record", "comm_done", "comm") in calls
    assert ("wait", "compute", "comm_done") in calls
    assert ctx.current_stream is compute


def test_failing_recv_hook_still_signals_partner_and_fences(monkeypatch):
    """A receive hook that raises must not skip the exit's cleanup.

    The partner wakes on `cpu_signal_event`/`done` and the parent fences this
    ubatch's storage against the comm stream. Both hang off the same exit, so
    an async receive failure that skipped them would wedge the partner and
    leave comm kernels racing the next forward's writes.
    """
    import threading

    from atom.utils.tbo import ubatching

    calls = []

    class FakeStream:
        def __init__(self, name):
            self.name = name

        def wait_event(self, event):
            calls.append(("wait", self.name, event.name))

    class FakeEvent:
        def __init__(self, name):
            self.name = name

        def record(self, stream):
            calls.append(("record", self.name, stream.name))

    compute, comm = FakeStream("compute"), FakeStream("comm")
    monkeypatch.setattr(torch.cuda, "set_stream", lambda stream: None)
    monkeypatch.setitem(ubatching._THREAD_ID_TO_CONTEXT, threading.get_ident(), 0)
    monkeypatch.setattr(ubatching, "_CURRENT_CONTEXTS", [None])

    signal = threading.Event()
    ctx = ubatching.TBOContext(
        ubatch_id=0,
        compute_stream=compute,
        comm_stream=comm,
        forward_context=None,
        ready_barrier=None,
        cpu_wait_event=threading.Event(),
        cpu_signal_event=signal,
        gpu_comm_done_event=FakeEvent("comm_done"),
        gpu_compute_done_event=FakeEvent("compute_done"),
    )
    ctx.current_stream = comm
    ctx.recv_hook = lambda: (_ for _ in ()).throw(RuntimeError("recv failed"))

    # The hook's failure surfaces rather than being swallowed...
    with pytest.raises(RuntimeError, match="recv failed"):
        ctx.__exit__(None, None, None)

    # ...and every piece of cleanup still ran.
    assert ("record", "comm_done", "comm") in calls
    assert ("wait", "compute", "comm_done") in calls
    assert ctx.current_stream is compute
    assert ctx.done is True
    assert signal.is_set()


@pytest.mark.parametrize("failure_point", ["record", "set_stream", "wait_event"])
@pytest.mark.parametrize("unwinding", [False, True])
def test_stream_cleanup_failure_always_releases_partner(
    monkeypatch, failure_point, unwinding
):
    import threading

    from atom.utils.tbo import ubatching

    def fail_at(point):
        if point == failure_point:
            raise RuntimeError("async HIP failure")

    class Stream:
        def wait_event(self, event):
            fail_at("wait_event")

    class Event:
        def record(self, stream):
            fail_at("record")

    compute, comm = Stream(), Stream()
    signal, wait = threading.Event(), threading.Event()
    wait.set()
    ctx = ubatching.TBOContext(
        ubatch_id=0,
        compute_stream=compute,
        comm_stream=comm,
        forward_context=None,
        ready_barrier=None,
        cpu_wait_event=wait,
        cpu_signal_event=signal,
        gpu_comm_done_event=Event(),
        gpu_compute_done_event=Event(),
    )
    ctx.current_stream = comm
    monkeypatch.setattr(torch.cuda, "set_stream", lambda stream: fail_at("set_stream"))
    monkeypatch.setitem(ubatching._THREAD_ID_TO_CONTEXT, threading.get_ident(), 0)
    monkeypatch.setattr(ubatching, "_CURRENT_CONTEXTS", [ctx])
    if unwinding:
        assert ctx.__exit__(ValueError, ValueError("original failure"), None) is False
    else:
        with pytest.raises(RuntimeError, match="async HIP failure"):
            ctx.__exit__(None, None, None)
    assert ctx.done
    assert signal.is_set()
    assert not wait.is_set()
    assert ubatching._CURRENT_CONTEXTS == [None]
    assert threading.get_ident() not in ubatching._THREAD_ID_TO_CONTEXT


def test_clean_exit_on_compute_adds_no_redundant_ordering(monkeypatch):
    """Already on the compute stream, the exit records nothing of its own."""
    import threading

    from atom.utils.tbo import ubatching

    calls = []

    class FakeStream:
        def __init__(self, name):
            self.name = name

        def wait_event(self, event):
            calls.append(("wait", self.name, event.name))

    class FakeEvent:
        def __init__(self, name):
            self.name = name

        def record(self, stream):
            calls.append(("record", self.name, stream.name))

    compute, comm = FakeStream("compute"), FakeStream("comm")
    monkeypatch.setattr(torch.cuda, "set_stream", lambda stream: None)
    monkeypatch.setitem(ubatching._THREAD_ID_TO_CONTEXT, threading.get_ident(), 0)
    monkeypatch.setattr(ubatching, "_CURRENT_CONTEXTS", [None])

    ctx = ubatching.TBOContext(
        ubatch_id=0,
        compute_stream=compute,
        comm_stream=comm,
        forward_context=None,
        ready_barrier=None,
        cpu_wait_event=threading.Event(),
        cpu_signal_event=threading.Event(),
        gpu_comm_done_event=FakeEvent("comm_done"),
        gpu_compute_done_event=FakeEvent("compute_done"),
    )

    ctx.__exit__(None, None, None)

    assert calls == []
    assert ctx.current_stream is compute


def test_parent_engram_join_runs_when_a_microbatch_fails():
    from atom.model_ops.engram.device.staging import EngramStagedRows

    events = []

    class Rows(EngramStagedRows):
        def __init__(self):
            dict.__init__(self)

        def stage(self, *, tbo=False):
            assert tbo
            events.append("stage")

        def join(self):
            events.append("join")

    parent = SimpleNamespace(engram_embeddings=Rows())
    builder = DeepseekV41MetadataBuilder.__new__(DeepseekV41MetadataBuilder)
    with (
        pytest.raises(RuntimeError, match="failed child"),
        builder.ubatch_forward(parent),
    ):
        raise RuntimeError("failed child")
    assert events == ["stage", "join"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_narrowed_tile_rows_stay_compact_and_within_their_microbatch():
    """A child table packs rows at its own width, not the allocation's.

    `tile_slice` hands out a region whose rows are as wide as the workspace
    was built for, but a request whose live `block_tables` is narrower fills
    only part of each. The kernel is handed `out.stride(0)`, so it writes at
    the same compact pitch the view reads at -- the unused tail of a row is
    simply never addressed, and the rows stay inside the slice they were cut
    from.
    """
    from atom.model_ops.deepseek_v41.score_workspace import ScoreWorkspace
    from atom.model_ops.deepseek_v41.unit_table import unit_table
    from tests.models.deepseek_v41.reference_unit_table import unit_table_reference

    units, alloc_columns, max_tokens = 2, 8, 8
    geometry = SimpleNamespace(
        index_fp4=False,
        index_block_rows=8,
        owners=((0, 1),),
        index_blocks_per_page=lambda ratio: units,
        rows_per_page=lambda ratio: 16,
    )
    workspace = ScoreWorkspace(geometry, max_tokens, alloc_columns, "cuda")

    torch.manual_seed(1409)
    live_columns, tokens = 3, 4
    assert live_columns < alloc_columns  # armed: the row has an unused tail
    batch_ids = torch.arange(tokens, dtype=torch.int32, device="cuda")
    first_pages = torch.randint(
        0, 999, (tokens, live_columns), dtype=torch.int32, device="cuda"
    )
    second_pages = first_pages + 500

    halves = [slice(0, tokens), slice(tokens, 2 * tokens)]
    tables = [
        unit_table(
            pages,
            batch_ids,
            units,
            workspace=workspace.tile_slice(half),
            ratio=1,
        )
        for pages, half in zip((first_pages, second_pages), halves)
    ]

    # Each child names its own requests' pages, so neither wrote over the
    # other's rows, and each matches the oracle element for element.
    for table, pages in zip(tables, (first_pages, second_pages)):
        assert table.stride(0) == live_columns * units
        assert torch.equal(table, unit_table_reference(pages, batch_ids, units))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_prefill_microbatches_keep_workspace_tiles_across_layer_interleaving():
    from atom.model_ops.deepseek_v41.score_workspace import ScoreWorkspace

    builder, parent = make_parent("cuda")
    workspace = ScoreWorkspace(builder.geometry, 32, 4, "cuda")
    parent.cache.workspace = workspace
    parts = _split_prefill_token_midpoint(2, [10, 4], 2, None)
    children = [
        builder.build_ubatch_prefill_metadata(
            parent, part, part.request_slice.stop - part.request_slice.start, i
        )
        for i, part in enumerate(parts)
    ]
    for ratio in (1, 2):
        first = parent.cache.unit_tiles(children[0].step, ratio)
        expected = first.clone()
        second = parent.cache.unit_tiles(children[1].step, ratio)
        # The first worker revisits this ratio after its partner's attention.
        # Its memoized table must still name its own requests' pages.
        again = parent.cache.unit_tiles(children[0].step, ratio)
        assert again is first
        torch.testing.assert_close(again, expected, rtol=0, atol=0)
        assert first.data_ptr() != second.data_ptr()
        for table in (first, second):
            assert (
                table.untyped_storage().data_ptr()
                == workspace._tiles[ratio].untyped_storage().data_ptr()
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_real_tbo_workers_share_one_uva_prefetch_and_keep_cross_layer_state(
    monkeypatch,
):
    from atom.model_ops.engram.device.hashing import engram_row_indices_reference
    from atom.utils.forward_context import get_forward_context
    from atom.utils.tbo.ubatching import (
        tbo_switch_to_compute_sync,
        tbo_yield_and_switch_from_compute_to_comm,
    )
    from tests.model_ops.engram.test_hash_bounds import build, tiny_config
    from tests.model_ops.engram.test_overlap import (
        fill_step,
        make_batch,
        make_staging,
        make_step,
    )

    builder, parent = make_parent("cuda", lengths=(9, 5))
    mapping = build(tiny_config())
    staging, _backing = make_staging(mapping)
    engram_step = make_step(staging.uva.hash_tables, 32, verify=False)
    calls = []
    start = staging.start
    join = staging.join
    monkeypatch.setattr(
        staging, "start", lambda width: (calls.append("stage"), start(width))[-1]
    )
    monkeypatch.setattr(staging, "join", lambda: (calls.append("join"), join())[-1])

    class Model(torch.nn.Module):
        def forward(self, ids, positions, inputs_embeds=None):
            ctx = get_forward_context()
            step = ctx.attn_metadata.step
            step.begin_forward()
            step.selected[0] = ids.clone()
            # Alternate workers twice while the earlier layer's state lives.
            for _ in range(2):
                tbo_yield_and_switch_from_compute_to_comm()
                copy = ids.clone()
                tbo_switch_to_compute_sync()
            rows = ctx.attn_metadata.engram_embeddings.get(mapping.config.layer_ids[-1])
            return rows[0].float() + (copy + step.selected[0]).float()[:, None]

    wrapper = UBatchWrapper(Model(), builder)
    ids = torch.arange(14, device="cuda", dtype=torch.int32)
    slices = _split_prefill_token_midpoint(2, [9, 5], 2, None)
    for seed in (7, 31):
        _, tokens, histories, masks = make_batch(mapping, [9, 5], seed)
        fill_step(engram_step, tokens, histories, masks, starts=(3, 8))
        # Staging snapshots before advancing the real cursor. Both workers
        # must still read the original history through their parent lookup.
        rows = staging.prepare(engram_step, 14)
        layer = mapping.config.layer_ids[-1]
        indices = engram_row_indices_reference(mapping, layer, tokens, histories, masks)
        expected = (
            staging.host.prefetcher._tables[layer]
            ._tensor[torch.as_tensor(indices)]
            .reshape(14, -1)
            .to(device="cuda", dtype=torch.float32)
        )
        expected = expected + ids.float()[:, None] * 2
        calls.clear()
        parent.engram_embeddings = rows
        ctx = ForwardContext(
            attn_metadata=parent,
            no_compile_layers={},
            kv_cache_data={},
            context=Context(
                positions=parent.step.positions,
                is_prefill=True,
                scheduled_bs=2,
                running_bs=2,
                scheduled_tokens=14,
                running_tokens=14,
            ),
            ubatch_slices=slices,
        )
        monkeypatch.setattr(_forward_context_local, "ctx", ctx, raising=False)
        result = wrapper(ids, parent.step.positions)
        torch.cuda.synchronize()
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
        assert calls == ["stage", "join"]
        assert _forward_context_local.ctx is ctx
    for table in staging.host.prefetcher._tables.values():
        table.disable_uva()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
@pytest.mark.parametrize("cut", [1, 7, 10, 11, 13])
def test_split_attention_reads_preceding_microbatch_window(cut, monkeypatch):
    from atom.model_ops.v4_kernels import sparse_attn_v4_paged_2src

    torch.manual_seed(37)
    builder, parent = make_parent("cuda")
    cache = parent.cache
    query = torch.randn(14, 8, 512, device="cuda", dtype=torch.bfloat16) * 0.1
    kv = torch.randn(1, 14, 512, device="cuda", dtype=torch.bfloat16)
    sink = torch.zeros(8, device="cuda")
    from atom.models.deepseek_v41.config import AttentionMode, LayerAttentionSpec

    spec = LayerAttentionSpec(layer_id=0, ratio=0, mode=AttentionMode.WINDOW)
    cache.state.view("window").normal_()
    before = cache.backing.clone()

    def attend(step, start, end):
        q = query[start:end].clone()
        values = kv[:, start:end]
        prefix, pptr, extend, eptr = cache.attention_indices(spec, step)
        assert step.is_prefill and not step.decode
        output = sparse_attn_v4_paged_2src(
            q,
            cache.pool,
            prefix,
            pptr,
            values[0],
            extend,
            eptr,
            sink,
            512**-0.5,
            out=q,
        )
        cache.write_window(0, values, step)
        return output

    expected = attend(parent.step, 0, 14)
    final_window = cache.state.view("window").clone()
    cache.backing.copy_(before)
    parts = [
        UBatchSlice(slice(0, 1 if cut <= 10 else 2), slice(0, cut)),
        UBatchSlice(slice(0 if cut < 10 else 1, 2), slice(cut, 14)),
    ]
    from atom.utils.forward_context import get_forward_context
    from atom.utils.tbo.ubatching import (
        tbo_current_ubatch_id,
        tbo_switch_to_compute_sync,
        tbo_yield_and_switch_from_compute_to_comm,
    )

    order = []

    class Model(torch.nn.Module):
        def forward(self, ids, positions, inputs_embeds=None):
            index = tbo_current_ubatch_id()
            step = get_forward_context().attn_metadata.step
            part = parts[index]
            assert step.is_prefill and not step.decode
            order.append((index, "attention"))
            output = attend(step, part.token_slice.start, part.token_slice.stop)
            order.append((index, "write_window"))
            # Match the production boundary: attention writes its window before
            # FFN yields to the other worker. GPU launches remain asynchronous.
            tbo_yield_and_switch_from_compute_to_comm()
            torch.cuda._sleep(2_000_000)
            tbo_switch_to_compute_sync()
            return output

    ctx = ForwardContext(
        attn_metadata=parent,
        no_compile_layers={},
        kv_cache_data={},
        context=Context(
            positions=parent.step.positions,
            is_prefill=True,
            scheduled_bs=2,
            running_bs=2,
            scheduled_tokens=14,
            running_tokens=14,
        ),
        ubatch_slices=parts,
    )
    monkeypatch.setattr(_forward_context_local, "ctx", ctx, raising=False)
    actual = UBatchWrapper(Model(), builder)(
        torch.arange(14, device="cuda", dtype=torch.int32), parent.step.positions
    )
    assert order == [
        (0, "attention"),
        (0, "write_window"),
        (1, "attention"),
        (1, "write_window"),
    ]
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(cache.state.view("window"), final_window, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
@pytest.mark.parametrize("unified", [False, True])
@pytest.mark.parametrize("layers", [1, 8])
@pytest.mark.parametrize("stress_lifetime", [False, True])
@pytest.mark.parametrize("level", [0, 3])
@pytest.mark.parametrize("dispatch", ["fallback", "complete"])
def test_dp_moe_original_schedule_preserves_outputs(
    monkeypatch, tmp_path, unified, layers, stress_lifetime, level, dispatch
):
    """Exercise the real MoE scheduler with delayed, local collective stand-ins.

    Keep the original MoE yield-before-communication schedule.
    Exercise shared FusedMoE allocation ownership with delayed compute
    and enough partner allocations to reuse an unprotected comm output.
    The synthetic complete=True return checks downstream consumers and flag
    preservation only. It does not exercise comm-fused backend internals;
    that backend's factory excludes TBO.
    """
    from atom.config import CompilationConfig, CUDAGraphMode
    from atom.model_ops import moe
    from atom.models.deepseek_v41.model import Block
    from atom.models.deepseek_v41.moe import MoE
    from atom.models.deepseek_v41.runtime import RuntimeBlock
    from atom.utils.decorators import support_torch_compile
    from atom.utils.forward_context import get_forward_context
    from atom.utils.tbo.ubatching import tbo_active, tbo_current_ubatch_id

    builder, parent = make_parent("cuda")
    calls = []
    width = 5120 if stress_lifetime else 1
    config = SimpleNamespace(
        enable_dp_attention=True,
        compilation_config=CompilationConfig(
            level=level,
            cudagraph_mode=CUDAGraphMode.FULL,
            cache_dir=str(tmp_path),
            splitting_ops=[],
            compile_sizes=[],
        ),
    )
    monkeypatch.setattr(moe, "get_current_atom_config", lambda: config)
    monkeypatch.setattr(moe, "get_dp_group", lambda: None)

    def gather(hidden, router, eager, ctx, group):
        assert eager == (not unified)
        calls.append((tbo_current_ubatch_id(), "gather"))
        torch.cuda._sleep(2_000_000)
        return hidden.clone(), router.clone(), len(hidden), [len(hidden)]

    def scatter(hidden, *args):
        calls.append((tbo_current_ubatch_id(), "scatter"))
        if stress_lifetime:
            scratch = [torch.empty_like(hidden) for _ in range(32)]
            for value in scratch:
                value.fill_(12345)
        torch.cuda._sleep(2_000_000)
        return hidden.clone()

    def experts(**kwargs):
        calls.append((tbo_current_ubatch_id(), "experts"))
        return kwargs["x"] * 2 + kwargs["router_logits"]

    monkeypatch.setattr(moe, "dp_gather_hidden_and_router", gather)
    monkeypatch.setattr(moe, "reduce_scatterv", scatter)
    monkeypatch.setattr(moe, "reduce_scatter_with_unpadding", scatter)
    layer = SimpleNamespace(
        dp_size=2,
        moe_parallel_config=SimpleNamespace(
            use_all2all_kernels=False, dp_logical_ratio=1
        ),
        quant_method=SimpleNamespace(apply=experts),
        reduce_results=False,
        top_k=1,
        renormalize=False,
        use_grouped_topk=False,
        global_num_experts=1,
        expert_map=None,
        topk_group=None,
        num_expert_group=None,
        custom_routing_function=None,
        scoring_func="softmax",
        e_score_correction_bias=None,
        shared_expert_scoring_func=None,
        activation=None,
        apply_router_weight_on_input=False,
        prefix="test",
    )

    def run_experts(hidden, router):
        # Compile on the main thread before exercising the TBO workers, as
        # ModelRunner warmup does. The custom op keeps scheduling at runtime.
        if not tbo_active():
            return hidden * 2 + router
        context = get_forward_context().context
        context.running_tokens_are_unified = unified
        iteration = getattr(context, "test_moe_iteration", 0)
        if iteration == 0:
            calls.append((tbo_current_ubatch_id(), "prepare"))
        context.test_moe_iteration = iteration + 1
        result = moe.FusedMoE.forward_impl_graph(layer, hidden, router)
        if stress_lifetime and iteration + 1 == layers:
            torch.cuda._sleep(100_000_000)
        return result

    layer.forward_impl = run_experts
    config.compilation_config.static_forward_context["test"] = layer

    class RoutedExperts(torch.nn.Module):
        def forward(self, hidden, router):
            return torch.ops.aiter.moe_forward(hidden, router, "test")

        def forward_maybe_comm_fused(self, hidden, router, shared, **kwargs):
            if dispatch == "complete":
                # Synthetic completed output; bypass Module.__call__ and hooks.
                return torch.ops.aiter.moe_forward(hidden, router, "test"), True
            return self(hidden, router), False

    class Gate(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # This synthetic gate exercises the ordinary GEMM fallback.
            self.register_buffer("weight", torch.empty(0, 0, device="cuda"))

        def forward(self, hidden):
            return hidden + 3

    def block_init(self):
        torch.nn.Module.__init__(self)
        self.ffn = MoE.__new__(MoE)
        torch.nn.Module.__init__(self.ffn)
        self.ffn.experts = RoutedExperts()
        self.ffn.gate = Gate()

    monkeypatch.setattr(Block, "__init__", block_init)

    @support_torch_compile(dynamic_arg_dims={"input_ids": 0, "positions": 0})
    class Model(torch.nn.Module):
        def __init__(self, atom_config):
            super().__init__()
            # Keep the real V4.1 routed-expert dispatch boundary.
            self.block = RuntimeBlock()

        def forward(self, input_ids, positions, inputs_embeds=None):
            result = input_ids.float()[:, None].expand(-1, width).contiguous()
            for _ in range(layers):
                result, complete = self.block.ffn.routed_expert_forward(result)
                assert complete == (dispatch == "complete")
            return result + torch.ones_like(result)

    ids = torch.arange(14, device="cuda", dtype=torch.int32)
    ctx = ForwardContext(
        attn_metadata=parent,
        no_compile_layers={},
        kv_cache_data={},
        context=Context(
            positions=parent.step.positions,
            is_prefill=True,
            scheduled_bs=2,
            running_bs=2,
            scheduled_tokens=14,
            running_tokens=14,
        ),
        ubatch_slices=_split_prefill_token_midpoint(2, [10, 4], 2, None),
    )
    monkeypatch.setattr(_forward_context_local, "ctx", ctx, raising=False)
    model = Model(config)
    with torch.inference_mode():
        model(ids, parent.step.positions)
    if level == 3:
        assert len(model.compiled_codes) == 1
    wrapper = UBatchWrapper(model, builder)
    for _ in range(3):
        calls.clear()
        actual = wrapper(ids, parent.step.positions)
        scale = 3**layers
        torch.testing.assert_close(
            actual,
            (ids.float()[:, None] * scale + 3 * (scale - 1) // 2 + 1).expand(-1, width),
        )
        expected = [(0, "prepare"), (1, "prepare")]
        for _ in range(layers):
            expected.extend(
                [
                    (0, "gather"),
                    (0, "experts"),
                    (1, "gather"),
                    (1, "experts"),
                    (0, "scatter"),
                    (1, "scatter"),
                ]
            )
        assert calls == expected


@pytest.mark.parametrize("unified", [False, True])
def test_prefill_microbatch_preserves_dp_shape_mode(monkeypatch, unified):
    import atom.config as config_module
    from atom.utils.forward_context import DPMetadata

    builder, parent = make_parent("cpu")
    ctx = ForwardContext(
        attn_metadata=parent,
        no_compile_layers={},
        kv_cache_data={},
        context=Context(
            positions=parent.step.positions,
            is_prefill=True,
            scheduled_bs=2,
            running_bs=2,
            scheduled_tokens=14,
            running_tokens=14,
            running_tokens_are_unified=unified,
        ),
        ubatch_slices=_split_prefill_token_midpoint(2, [10, 4], 2, None),
        dp_metadata=object(),
        ub_max_tokens_across_dp=[11, 12],
        ub_tokens_across_dp=((7, 3, 5, 11), (7, 4, 6, 12)),
    )
    monkeypatch.setattr(
        config_module,
        "get_current_atom_config",
        lambda: SimpleNamespace(
            parallel_config=SimpleNamespace(data_parallel_size=4, data_parallel_rank=0)
        ),
    )

    def unexpected_collective(*args):
        pytest.fail("precomputed token counts must not trigger another collective")

    monkeypatch.setattr(DPMetadata, "num_tokens_across_dp", unexpected_collective)
    wrapper = UBatchWrapper(torch.nn.Identity(), builder, dp_gather_scatter=True)
    counts = wrapper._compute_ub_running_tokens(ctx, 2, 2, 4, torch.device("cpu"))
    assert counts == ([11, 12] if unified else [7, 7])
    metadata = wrapper._make_ubatch_dp_metadata(ctx, 2)
    for i, part in enumerate(ctx.ubatch_slices):
        child = wrapper._make_ubatch_context(
            ctx,
            part,
            part.request_slice.stop - part.request_slice.start,
            i,
            ub_running_tokens=counts[i],
            dp_metadata=metadata[i],
            running_tokens_across_dp=ctx.ub_tokens_across_dp[i],
        )
        assert child.context.running_tokens_are_unified is unified
        assert child.context.running_tokens == counts[i]
        assert child.dp_metadata.get_sizes_across_dp() == list(
            ctx.ub_tokens_across_dp[i]
        )


@pytest.mark.parametrize("dp_attention", [False, True])
def test_forward_does_not_repeat_request_admission(monkeypatch, dp_attention):
    from atom.models.deepseek_v41.runtime import v41_begin_forward

    metadata = SimpleNamespace(
        image_mask=torch.ones(1, 1, dtype=torch.bool),
        step=SimpleNamespace(begin_forward=lambda: None, requests=()),
    )
    context = SimpleNamespace(
        attn_metadata=metadata,
        dp_metadata=object() if dp_attention else None,
    )
    monkeypatch.setattr(_forward_context_local, "ctx", context, raising=False)
    hidden = torch.empty(1, 1, 1)
    # Admission rejects unsupported media before any distributed forward.
    # Repeating that check here could strand peers in their collectives.
    v41_begin_forward(hidden)


def test_tbo_rejects_compacted_scheduler_rows_before_slicing():
    builder, parent = make_parent("cpu", lengths=(0, 4))
    assert parent.scheduler_rows == (1,)
    with pytest.raises(ValueError, match="uncompacted scheduler"):
        builder.build_ubatch_prefill_metadata(
            parent, UBatchSlice(slice(0, 1), slice(0, 4)), 1
        )


@pytest.mark.parametrize("backend,expected", [("none", 0), ("v4", 0), ("v41", -1)])
def test_tbo_communication_priority_is_backend_owned(monkeypatch, backend, expected):
    from atom.model_ops.attentions.deepseek_v4_attn import (
        DeepseekV4AttentionMetadataBuilder,
    )

    cls = {
        "v4": DeepseekV4AttentionMetadataBuilder,
        "v41": DeepseekV41MetadataBuilder,
    }.get(backend)
    builder = None if cls is None else cls.__new__(cls)
    priorities = []
    stream = object()

    def create_stream(*, priority):
        priorities.append(priority)
        return stream

    monkeypatch.setattr(torch.cuda, "Stream", create_stream)
    wrapper = UBatchWrapper(torch.nn.Identity(), builder)
    wrapper._ensure_comm_stream()
    wrapper._ensure_comm_stream()
    assert priorities == [expected]
    assert wrapper.comm_stream is stream


def test_tbo_preserves_v4_forward_signature(monkeypatch):
    from atom.models import deepseek_v4

    monkeypatch.setattr(deepseek_v4, "_pcp_active", lambda: False)
    monkeypatch.setattr(deepseek_v4, "_moe_pcp_merge_active", lambda: False)
    context = SimpleNamespace(context=SimpleNamespace(input_ids=None))
    monkeypatch.setattr(deepseek_v4, "get_forward_context", lambda: context)

    class Model(torch.nn.Module):
        forward = deepseek_v4.DeepseekV4ForCausalLM.forward
        _need_ids_gather = False

        def model(self, input_ids, positions):
            return input_ids + positions

    ids = torch.arange(4)
    positions = ids + 10
    monkeypatch.setattr(
        _forward_context_local,
        "ctx",
        SimpleNamespace(ubatch_slices=None),
        raising=False,
    )
    torch.testing.assert_close(UBatchWrapper(Model())(ids, positions), ids + positions)
    torch.testing.assert_close(context.context.input_ids, ids)
