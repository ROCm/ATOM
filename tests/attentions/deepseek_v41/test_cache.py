# SPDX-License-Identifier: MIT
"""Paging, sparse causality and exact state recovery across request lifetimes."""

from dataclasses import replace

import numpy as np
import pytest
import torch
from atom.model_ops.attentions.deepseek_v41.cache import (
    PagedAttentionCache,
)
from atom.model_ops.attentions.deepseek_v41.checkpoints import StateCopies
from atom.model_ops.attentions.deepseek_v41.metadata import RequestSpan
from tests.attentions.deepseek_v41.helpers import geometry

from atom.model_engine.page_unit_checkpoint import (
    CheckpointRestoreOp,
    CheckpointStoreOp,
    PagedStateCheckpointSpec,
)


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

    geo = V41PoolGeometry(1, ((0, 2),), 32, 4, 512, 32, speculative_tokens=1)
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
    # assertions below would hold for destinations that ignore the question.
    assert live.any() and not live.all()
    rows = plan.compress_plan_gpu[:, 2] // 2
    per_page = geo.rows_per_page(2)
    pages, offsets, resolved = cache._plan_destinations(step, rows, 2, per_page)
    assert resolved.tolist() == live.tolist()
    # Negative, and negative after the writers' own `page * per_page + offset`
    # too: that is the row index V4's kernels skip, and the reason the fp8
    # path needs no destination of its own for a row that has no value.
    assert (pages[~live] * per_page + offsets[~live] < 0).all()
    # The live rows still resolve through the request's own PAGE table.
    table = step.block_tables[plan.compress_plan_gpu[:, 1].long()]
    expected = table[live, (rows[live] // per_page)]
    assert pages[live].tolist() == expected.tolist()
    assert offsets[live].tolist() == (rows[live] % per_page).tolist()
