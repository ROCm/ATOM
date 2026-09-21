# SPDX-License-Identifier: MIT
"""Paging, sparse causality and exact state recovery across request lifetimes."""

from dataclasses import replace

import numpy as np
import pytest
import torch

pytest.importorskip("aiter", reason="the paged cache and V4 kernels reach AITER")

from atom.model_engine.page_unit_checkpoint import (
    CheckpointRestoreOp,
    CheckpointStoreOp,
    PagedStateCheckpointSpec,
)
from atom.model_ops.attentions.deepseek_v41.cache import (
    PagedAttentionCache,
)
from atom.model_ops.attentions.deepseek_v41.checkpoints import StateCopies
from atom.model_ops.attentions.deepseek_v41.metadata import RequestSpan
from tests.attentions.deepseek_v41.helpers import geometry


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_checkpoint_fork_rollback_relocation_and_slot_reuse(
    small_config, device, packed
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("ROCm GPU required")
    # A window wide enough that one image outgrows a PAGE unit: an FP8 index
    # plane puts 32 tokens in a PAGE, and the fixture's own 4-token window
    # leaves a state entry that fits in a single one.
    geo = replace(geometry(small_config), packed=packed, window_size=32)
    spec = PagedStateCheckpointSpec(
        geo.paged_bytes, geo.state_bytes, geo.layout_id, geo.state_bytes
    )
    assert spec.units_per_checkpoint > 1
    # Enough PAGEs for the interleaved unit ids below, which is what makes the
    # image land in units that are not consecutive.
    pages = max(40, 2 * spec.units_per_checkpoint)
    cache = PagedAttentionCache(geo, pages, 4, device)
    # Exactly the pages the scheduler can name, as V4's pool is: the sentinel
    # rows of a CUDAGraph-sized compression plan are skipped rather than
    # landed somewhere, so the pool buys nothing for them.
    assert cache.num_pages == pages
    assert cache.backing.numel() == geo.paged_extents(pages)[1] + 4 * geo.state_bytes
    copies = StateCopies(cache, spec, 4)
    copies.warmup()
    # Fill ALL bytes, including padding and FP32 compressor tails. The cursor
    # remains interpretable while the rest proves byte-exact recovery.
    cache.state_bytes[2].copy_(
        torch.randint(
            0, 256, cache.state_bytes[2].shape, dtype=torch.uint8, device=device
        )
    )
    cache.cursor[2] = torch.tensor([3, 19, -1, 27], device=device)
    original = copies.entry(2).clone()
    units = tuple(range(1, 2 * spec.units_per_checkpoint, 2))
    store = CheckpointStoreOp(2, units, spec.image_bytes, spec.layout_id)
    restore = CheckpointRestoreOp(1, units, spec.image_bytes, spec.layout_id)
    copies.execute([store], [restore])
    torch.testing.assert_close(copies.entry(1), original, rtol=0, atol=0)
    cache.state_bytes[1].fill_(93)  # rejected / cancelled tentative suffix
    copies.execute([], [restore])
    torch.testing.assert_close(copies.entry(1), original, rtol=0, atol=0)
    cache.state_bytes[0].fill_(71)
    old_zero = copies.entry(0).clone()
    copies.relocate([(1, 0), (0, 1)])
    torch.testing.assert_close(copies.entry(0), original, rtol=0, atol=0)
    torch.testing.assert_close(copies.entry(1), old_zero, rtol=0, atol=0)
    span = RequestSpan(27, 3, 0, 1, 0, (32, 33))
    step = cache.begin_step([span])
    np.testing.assert_array_equal(cache.prepare_state(step), [[19, -1, 27]])
    # Recycled slot begins at zero and drops every old state field.
    fresh = cache.begin_step([replace(span, request_id=28, position=0)])
    np.testing.assert_array_equal(cache.prepare_state(fresh), [[-1, -1, -1]])
    assert cache.state.view("window")[:, 0].count_nonzero() == 0
    assert cache.state.view("compress_kv")[:, 0].count_nonzero() == 0
    assert cache.cursor[0, 0] == 0
    with pytest.raises(ValueError, match="recoverable boundary"):
        cache.prepare_state(step)
    for bad in (
        replace(restore, layout_id="wrong"),
        replace(restore, total_bytes=3),
        replace(restore, unit_ids=(0,) * len(units)),
    ):
        with pytest.raises(ValueError):
            copies.execute([], [bad])
    with pytest.raises(IndexError):
        copies.entry(-1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("topk", [4, 64])
def test_a_rows_prefix_slice_is_exactly_as_long_as_what_gets_written(ratio, topk):
    """The indptr reserves what the writer writes, per row, to the slot.

    `_indptr_scan` derives a row's length from its position; `_indices` writes
    a window segment at one end and one id per non-negative selection at the
    other. A row where the two disagree leaves the difference between them
    untouched, and `sparse_attn_v4_paged_decode` is called with
    `has_invalid=False` -- it dereferences every slot the indptr claims, so
    that gap is an out-of-range read of whatever the allocation held.

    Checked against the writer's own rule rather than against the scan's,
    which is the only way the two can be caught disagreeing.
    """
    from types import SimpleNamespace

    from atom.model_ops.attentions.pool_layout.v41_pool_geometry import V41PoolGeometry

    geo = V41PoolGeometry(
        2, ((0, ratio),), 32, 8, 512, 32, layer_ratios=(ratio,), index_topk=topk
    )
    cache = PagedAttentionCache(geo, 32, 4, "cuda")
    # Decode rows on both sides of the window boundary, and a request whose
    # visible count is under `topk` beside one well over it: the short row is
    # where the writer's per-row count and the scan's can disagree, and the
    # long one is what keeps `topk` from being the only bound in play.
    spans = (
        RequestSpan(1, 3, 0, 1, 0, (0, 1, 2)),
        RequestSpan(2, 200, 1, 1, 1, tuple(range(3, 16))),
    )
    step = cache.begin_step(spans, running_bs=2, running_tokens=2, max_q_len=1)
    assert step.decode
    visible = (step.positions + 1) // ratio
    # The selection the scorers emit: ascending ids, `-1` past `min(visible,
    # topk)`, which is the count the scan assumes.
    counts = visible.clamp(max=topk)
    columns = torch.arange(topk, device="cuda")
    selection = torch.where(
        columns < counts[:, None], columns.expand(step.width, topk), -1
    ).int()
    step.selected[0] = selection.unsqueeze(0)
    spec = SimpleNamespace(layer_id=1, ratio=ratio, kv_owner=0, topk_owner=0)
    _, pptr, _, _ = cache.attention_indices(spec, step)
    window = (step.positions + 1).clamp(max=geo.window_size)
    written = window + (selection >= 0).sum(-1)
    assert torch.equal(pptr.diff().long(), written.long()), (
        f"reserved={pptr.diff().tolist()} written={written.tolist()} "
        f"window={window.tolist()} visible={visible.tolist()}"
    )


def test_a_graph_sized_plans_sentinel_rows_land_on_the_page_nobody_owns():
    """The rows a fixed grid adds beyond the batch address nothing live.

    A plan cut for a CUDAGraph is `running_bs * per-seq bound` rows whatever
    the batch, and the tail is `-1` in both fields. The index and packed-main
    scatters are torch advanced indexing, where `-1` is the LAST page and the
    last row of it -- a live request's, at every shape this runs. The
    destination is the one PAGE the scheduler cannot name instead.
    """
    from atom.model_ops.attentions.pool_layout.v41_pool_geometry import V41PoolGeometry
    from atom.model_ops.v4_kernels import make_compress_plans
    from atom.utils import CpuGpuBuffer

    # Two owners, because one leaves a PAGE exactly `rows * dim` wide and the
    # scatter's plane contiguous -- the shape that hides a destination folding
    # the two axes into one.
    geo = V41PoolGeometry(2, ((0, 2), (1, 2)), 32, 4, 512, 32, speculative_tokens=1)
    cache = PagedAttentionCache(geo, 6, 2, "cpu")
    running_bs, max_q_len = 2, 2
    spans = (RequestSpan(1, 4, 0, 2, 0, (3, 5)),)
    plans = make_compress_plans(
        np.asarray([2], dtype=np.int32),
        np.asarray([6], dtype=np.int32),
        geo.compress_ratios,
        plan_buffers={
            2: {
                name: CpuGpuBuffer(8, 4, dtype=torch.int32, device="cpu")
                for name in ("compress", "write")
            }
        },
        running_bs=running_bs,
        max_q_len=max_q_len,
        extra_write=1,
    )
    step = cache.begin_step(
        spans, running_bs=running_bs, running_tokens=running_bs * max_q_len, plans=plans
    )
    plan = step.plans[2]
    # The capacity, which is what the kernel's grid and every row derived from
    # it are; `num_compress` is the count this batch happened to produce.
    assert plan.compress_plan_gpu.shape[0] > plan.num_compress > 0
    live = plan.compress_plan_gpu[:, 1] >= 0
    # Positive control: with no sentinel row there is nothing to place, and the
    # assertions below would hold for a scatter that ignores the question.
    assert live.any() and not live.all()
    # Derived here rather than handed to the scatter: the scatter works the
    # same rows out of the plan itself, so these assertions check that
    # derivation instead of echoing a value the test supplied.
    rows = plan.compress_plan_gpu[:, 2] // 2
    per_page = geo.rows_per_page(2)
    pages = cache.pages.view("main_0")[0]
    pages.fill_(0)
    # A value no row shares, so a row that landed says which one it was.
    value = torch.arange(1, rows.numel() + 1, dtype=pages.dtype, device=pages.device)
    filled = value[None, :, None].expand(1, -1, pages.shape[-1])
    cache._scatter_rows(pages, step, filled, 2)
    table = step.block_tables[plan.compress_plan_gpu[:, 1].long()]
    for i in live.nonzero().flatten().tolist():
        page = table[i, rows[i] // per_page]
        assert pages[page, rows[i] % per_page, 0] == value[i]
    # Exactly the live rows were written: a sentinel reaching `-1` would have
    # landed on the last page's last row, which is a live request's.
    assert int((pages != 0).any(-1).sum()) == int(live.sum())


def test_key_rope_positions_span_the_grid_and_carry_its_sentinels():
    """Where every plan row's index key gets rotated, published by the plan.

    A boundary's key is rotated at its compression group's FIRST token and not
    at its own: `fused_compress` rotated that group's main latent there, and a
    key rotated anywhere else selects other rows.

    Two things a shorter or hand-filled tensor would get past: the grid is the
    capacity rather than `num_compress`, so a sentinel row needs a position
    too -- and at ratio 2 a sentinel's is `-2`, which is what a fill that
    reasoned "sentinels are -1" would most likely miss.
    """
    from atom.model_ops.attentions.pool_layout.v41_pool_geometry import V41PoolGeometry
    from atom.model_ops.v4_kernels import make_compress_plans
    from atom.utils import CpuGpuBuffer

    geo = V41PoolGeometry(2, ((0, 2), (1, 2)), 32, 4, 512, 32, speculative_tokens=1)

    def build(with_key_rope):
        buffers = {
            name: CpuGpuBuffer(8, 4, dtype=torch.int32, device="cpu")
            for name in ("compress", "write")
        }
        if with_key_rope:
            buffers["key_rope"] = CpuGpuBuffer(8, dtype=torch.int64, device="cpu")
        return make_compress_plans(
            np.asarray([2], dtype=np.int32),
            np.asarray([6], dtype=np.int32),
            geo.compress_ratios,
            plan_buffers={2: buffers},
            running_bs=2,
            max_q_len=2,
            extra_write=1,
        )[2]

    plan = build(True)
    positions = plan.compress_plan_gpu[:, 2]
    # Armed: the grid really does run past the rows this batch filled, so the
    # sentinel half of the comparison below is not vacuous.
    assert plan.compress_plan_gpu.shape[0] > plan.num_compress > 0
    assert (positions < 0).any()
    assert plan.key_rope_positions_gpu.dtype == torch.int64
    assert torch.equal(plan.key_rope_positions_gpu, positions.long() // 2 * 2)
    # Named rather than left to the comparison: this is the value the ratio-1
    # intuition gets wrong, and naming it makes the failure say so.
    assert plan.key_rope_positions_gpu[plan.num_compress] == -2
    # A caller that declares no buffer pays nothing and is handed nothing.
    assert build(False).key_rope_positions_gpu is None


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_a_deferred_prepare_state_names_the_stale_slot_one_step_later(
    small_config, device
):
    """`histories=False` ships the rows without waiting; the verdict follows.

    The caller that asks for no history is the one whose Engram rows and
    advanced cursor are both worked out on the device, so nothing on the host
    reads this step's committed history -- and nothing needs this step's
    verdict either. What must not happen is the verdict going missing: a slot
    holding the wrong position has to be named, late but named, and by the
    request that wanted it.
    """
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("ROCm GPU required")
    geo = replace(geometry(small_config), window_size=32)
    cache = PagedAttentionCache(geo, 8, 4, device)
    cache.cursor[0] = torch.tensor([3, 19, -1, 27], device=device)

    good = RequestSpan(27, 3, 0, 1, 0, (0, 1))
    step = cache.begin_step([good])
    assert cache.prepare_state(step, histories=False) is None

    # Same slot, but the scheduler believes it is four tokens further on than
    # the cursor says. Deferred, so this call is the one that ships the rows.
    stale = RequestSpan(28, 7, 0, 1, 0, (0, 1))
    later = cache.begin_step([stale])
    assert cache.prepare_state(later, histories=False) is None
    with pytest.raises(ValueError, match="Request 28 needs state at 7"):
        cache.prepare_state(cache.begin_step([good]), histories=False)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_a_deferred_probe_skips_the_slot_its_own_step_resets(small_config, device):
    """A fresh request reuses whatever the last tenant left; that is not stale."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("ROCm GPU required")
    geo = replace(geometry(small_config), window_size=32)
    cache = PagedAttentionCache(geo, 8, 4, device)
    cache.cursor[0] = torch.tensor([91, 19, -1, 27], device=device)
    fresh = RequestSpan(31, 0, 0, 1, 0, (0, 1))
    cache.prepare_state(cache.begin_step([fresh]), histories=False)
    assert cache.cursor[0, 0] == 0
    # The probe carries slot 0's pre-reset row; position 0 is what excludes it.
    cache.prepare_state(cache.begin_step([replace(fresh, request_id=32)]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
@pytest.mark.parametrize("decode", [False, True])
@pytest.mark.parametrize("width", [0, 96, 1025])
@pytest.mark.parametrize(
    "ratios", [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)]
)
def test_shared_indptr_launch_matches_row_counts_and_replay(decode, width, ratios):
    from types import SimpleNamespace

    from atom.model_ops.attentions.deepseek_v41.indices import fill_step_indptrs
    from atom.model_ops.attentions.pool_layout.v41_pool_geometry import V41PoolGeometry

    geo = V41PoolGeometry(
        len(ratios),
        tuple((i, ratio) for i, ratio in enumerate(ratios) if ratio),
        32,
        128,
        512,
        32,
        layer_ratios=ratios,
        index_topk=64,
    )
    live = max(0, width - 7)
    lengths = [live // 3, live // 3, live - 2 * (live // 3)]
    cu = torch.tensor(
        [0, lengths[0], sum(lengths[:2]), live], dtype=torch.int32, device="cuda"
    )
    batches = torch.cat(
        [
            torch.full((n,), b, device="cuda", dtype=torch.int32)
            for b, n in enumerate(lengths)
        ]
        + [torch.full((width - live,), -1, device="cuda", dtype=torch.int32)]
    )
    positions = torch.zeros(width, device="cuda", dtype=torch.int32)
    buffers = {
        r: (
            torch.full((width + 1,), -7, device="cuda", dtype=torch.int32),
            torch.full((width + 1,), -9, device="cuda", dtype=torch.int32),
        )
        for r in geo.layer_ratios
    }
    step = SimpleNamespace(
        width=width,
        decode=decode,
        positions=positions,
        batch_ids=batches,
        cu_seqlens_q=cu,
    )
    assert tuple(fill_step_indptrs(step, geo, buffers)) == ratios
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        built = fill_step_indptrs(step, geo, buffers)
    for starts in [(0, 127, 300), (511, 3, 120)]:
        pos = []
        windows, reaches = [], []
        for start, n in zip(starts, lengths):
            for j in range(n):
                absolute = start + j
                pos.append(absolute)
                windows.append(
                    max(0, (absolute + 1 if decode else start) - max(absolute - 127, 0))
                )
                reaches.append(min(j + 1, 128))
        positions.copy_(
            torch.tensor(pos + [0] * (width - live), device="cuda", dtype=torch.int32)
        )
        graph.replay()
        for ratio, (prefix, extend, topk) in built.items():
            counts = [
                w + (min((p + 1) // ratio, topk) if ratio else 0)
                for w, p in zip(windows, pos)
            ] + [0] * (width - live)
            expected = torch.tensor(
                [0] + list(np.cumsum(counts)), device="cuda", dtype=torch.int32
            )
            torch.testing.assert_close(prefix, expected, rtol=0, atol=0)
            if decode:
                assert extend.data_ptr() == prefix.data_ptr()
            else:
                expected = torch.tensor(
                    [0] + list(np.cumsum(reaches + [0] * (width - live))),
                    device="cuda",
                    dtype=torch.int32,
                )
                torch.testing.assert_close(extend, expected, rtol=0, atol=0)
