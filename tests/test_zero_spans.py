# SPDX-License-Identifier: MIT

"""`zero_spans`: the plan names exactly the bytes `index_fill_(1, ...)` writes.

The kernel itself needs a GPU; what can go wrong without one is the arithmetic
that turns views into (address, stride, length) spans and tiles, and the
refusals in front of it. Those are checked here against the views' own
addresses, on CPU tensors.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from atom.model_ops.attentions.pool_layout import zero_spans as zs


def _chunk(view: torch.Tensor, layer: int, entry: int) -> tuple[int, int]:
    piece = view[layer, entry]
    return piece.data_ptr(), piece.numel() * piece.element_size()


def _pool_views():
    # Two fields of one arena the way `LayerMajorArena.view` cuts them:
    # [layers, entries, ...] over one uint8 buffer, layer-major.
    layers, entries = 3, 7
    buf = torch.zeros(layers * entries * (256 + 64), dtype=torch.uint8)
    k = buf[: layers * entries * 256].view(layers, entries, 256)
    s = buf[layers * entries * 256 :].view(torch.float32).view(layers, entries, 16)
    return buf, [k, s]


def test_spans_are_the_chunks_index_fill_writes():
    _, views = _pool_views()
    plan = zs.ZeroSpanPlan(views)
    assert plan.num_entries == 7
    assert plan.spans.shape == (6, 3)  # one per (view, layer)
    row = 0
    for view in views:
        for layer in range(view.shape[0]):
            base, stride, length = plan.spans[row]
            for entry in (0, 3, 6):
                assert (base + entry * stride, length) == _chunk(view, layer, entry)
            row += 1
    assert plan.bytes_per_entry == 3 * 256 + 3 * 64


def test_tiles_cover_every_span_once():
    k = torch.zeros(2, 4, 3 * zs._TILE_BYTES + 512, dtype=torch.uint8)
    plan = zs.ZeroSpanPlan([k])
    for span, length in enumerate(plan.spans[:, 2]):
        starts = plan.tile_start[plan.span_of_tile == span]
        assert starts.tolist() == list(range(0, int(length), zs._TILE_BYTES))
    assert plan.num_tiles == 2 * 4


def test_header_holds_the_three_tables_at_their_offsets():
    _, views = _pool_views()
    plan = zs.ZeroSpanPlan(views)
    _, at_tile, at_start = plan.header_offsets()
    h = plan.header
    assert h.dtype == np.int64 and h.size % 2 == 0
    assert (h[: plan.spans.size] == plan.spans.reshape(-1)).all()
    assert (h[at_tile : at_tile + plan.num_tiles] == plan.span_of_tile).all()
    assert (h[at_start : at_start + plan.num_tiles] == plan.tile_start).all()
    assert at_tile % 2 == 0 and at_start % 2 == 0  # 16 B aligned sections


def test_a_slice_of_the_conv_window_is_one_chunk_per_slot():
    # GDN's dead conv tail: k[:, :, live:, :] of [layers, slots, window, dim]
    conv = torch.zeros(2, 5, 6, 8, dtype=torch.bfloat16)
    dead = conv[:, :, 3:, :]
    plan = zs.ZeroSpanPlan([dead])
    base, stride, length = plan.spans[1]
    assert (base + 4 * stride, length) == _chunk(dead, 1, 4)
    assert length == 3 * 8 * 2


def test_refuses_views_whose_elements_are_not_one_chunk():
    k = torch.zeros(2, 4, 8, 8, dtype=torch.float32)
    with pytest.raises(ValueError, match="not dense"):
        zs.ZeroSpanPlan([k[:, :, :, :4]])
    with pytest.raises(ValueError, match="multiples of 16"):
        zs.ZeroSpanPlan([torch.zeros(2, 4, 3, dtype=torch.uint8)])
    with pytest.raises(ValueError):
        zs.ZeroSpanPlan([])


def test_zeroer_refuses_entries_outside_the_views_before_any_launch():
    _, views = _pool_views()
    zeroer = zs.SpanZeroer()
    with pytest.raises(IndexError):
        zeroer(views, [0, 7])
    with pytest.raises(IndexError):
        zeroer(views, np.array([-1, 2]))
    zeroer(views, ())  # nothing to do, nothing launched


def test_plans_are_rebuilt_for_a_reallocated_pool():
    _, views = _pool_views()
    zeroer = zs.SpanZeroer()
    first = zeroer.plan(views)
    assert zeroer.plan(views) is first
    _, other = _pool_views()
    assert zeroer.plan(other) is not first


def test_fused_zero_fill_keeps_index_fill_for_views_off_the_gpu(monkeypatch):
    pytest.importorskip("aiter", reason="needs the AITER GPU kernel library")
    from atom.model_ops.attentions.backends import AttentionMetadataBuilder

    monkeypatch.setenv("ATOM_FUSED_ZERO_FILL", "1")
    k = torch.full((3, 5, 2, 4), float("inf"))
    builder = SimpleNamespace(
        kv_pools={0: SimpleNamespace(_views={"k": k})}, device=torch.device("cpu")
    )
    builder._zero_spans = lambda views, entries: (
        AttentionMetadataBuilder._zero_spans(builder, views, entries)
    )
    AttentionMetadataBuilder.zero_kv_blocks(builder, (1, 3))
    assert (k[:, [1, 3]] == 0).all()
    assert not torch.isfinite(k[:, [0, 2, 4]]).any()
