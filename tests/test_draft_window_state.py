# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""A sliding-window draft's ring (`draft_window.py`) and its builder."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from atom.model_engine.kv_block import STATE_SLOT_CLASS
from atom.model_ops.attentions.mha_kv_pool import MhaKvPool
from atom.model_ops.attentions.pool_layout.sub_pool_spec import Pool
from atom.spec_decode import draft_window as dw
from atom.spec_decode.draft_window import DraftWindow
from atom.spec_decode.draft_window_state import (
    DRAFT_STATE_GROUP,
    DraftWindowStateBuilder,
)
from atom.utils import CpuGpuBuffer

BLOCK = 16
WINDOW = DraftWindow(
    window=40, read_page=4, lookahead=4, block_size=BLOCK, anchor_margin=2 * BLOCK
)  # 48 tokens a reader needs, 80 with the anchor margin: 5 ring entries


def _pool():
    return MhaKvPool(
        layers=2,
        block_size=BLOCK,
        num_kv_heads=1,
        head_dim=16,
        kv_dtype=torch.bfloat16,
    )


class _Group:
    def __init__(self):
        self.counts, self.published = {}, 0

    def set_count(self, buffer, count):
        self.counts[id(buffer)] = count

    def publish(self):
        self.published += 1


def _runner(max_bs=4, cols=12, tokens=64):
    def buffer(*shape, dtype):
        return CpuGpuBuffer(*shape, dtype=dtype, device="cpu", pin_memory=False)

    forward_vars = {
        "block_tables": buffer(max_bs, cols, dtype=torch.int32),
        "slot_mapping": buffer(tokens, dtype=torch.int64),
        "positions": buffer(tokens, dtype=torch.int64),
        "batch_id_per_q_token": buffer(tokens, dtype=torch.int32),
    }
    return SimpleNamespace(
        config=SimpleNamespace(decode_context_parallel_size=1, max_num_seqs=max_bs),
        device=torch.device("cpu"),
        block_size=BLOCK,
        forward_vars=forward_vars,
        h2d_groups={DRAFT_STATE_GROUP: _Group()},
    )


def _builder(runner=None, *, pool=True):
    runner = runner or _runner()
    builder = DraftWindowStateBuilder(
        runner, (lambda **_: None) if pool else None, WINDOW
    )
    if pool:
        builder._kv_pool = _pool()
    return builder


def test_the_ring_is_a_state_class_and_no_page_share():
    builder = _builder()
    [spec] = builder.sub_pool_specs()
    assert spec.pool is Pool.STATE and spec.name == STATE_SLOT_CLASS
    assert spec.entry_bytes == builder.kv_pool.entry_bytes * WINDOW.ring_blocks
    assert builder.paged_pool_bytes(100) == 0


def test_a_stage_without_the_draft_declares_the_same_state_at_no_bytes():
    builder = _builder(pool=False)
    [spec] = builder.sub_pool_specs()
    assert spec.pool is Pool.STATE and spec.entry_bytes == 0
    assert builder.allocate_per_req_cache({STATE_SLOT_CLASS: 3}) == {}
    assert builder.num_state_slots == 3
    assert builder.slot_regions() == [] and builder.state_entry_views(1) == []
    assert builder.checkpoint_image_bytes() is None
    assert builder.state_transfer() == _builder().state_transfer()


def test_the_transfer_rebuilds_a_window_and_trails_an_anchor():
    transfer = _builder().state_transfer()
    assert transfer.copies
    assert transfer.rebuild_tokens == WINDOW.rebuild_tokens == 48
    assert transfer.trailing_slack == WINDOW.trailing_slack == 5 * BLOCK - 48


def test_rings_are_allocated_past_a_null_entry():
    builder = _builder()
    builder.allocate_per_req_cache({STATE_SLOT_CLASS: 3})
    assert builder.kv_pool.entries == 1 + 3 * WINDOW.ring_blocks


def test_slot_regions_address_each_slot_ring():
    builder = _builder()
    builder.allocate_per_req_cache({STATE_SLOT_CLASS: 3})
    regions = builder.slot_regions()
    layout = builder.kv_pool.region_layout()
    assert len(regions) == len(layout)
    for region, (base, entry_bytes) in zip(regions, layout):
        assert region.unit_bytes == WINDOW.ring_blocks * entry_bytes
        assert region.unit_addr(2) == base + (1 + 2 * WINDOW.ring_blocks) * entry_bytes
    assert sum(r.unit_bytes for r in regions) == builder.checkpoint_image_bytes()


def test_relocating_a_slot_moves_its_ring_and_only_its_ring():
    builder = _builder()
    builder.allocate_per_req_cache({STATE_SLOT_CLASS: 3})
    for _, tensor in builder.kv_pool.region_tensors():
        tensor.view(torch.uint8).fill_(0)
    views = builder.state_entry_views(0)
    for view in views:
        view.fill_(7)
    builder.relocate_state_slots([(0, 2)])
    assert all(bool((v == 7).all()) for v in builder.state_entry_views(2))
    assert all(not v.any() for v in builder.state_entry_views(1))


def test_a_step_publishes_the_ring_table_and_each_rows_ring_slot():
    runner = _runner()
    builder = _builder(runner)
    builder.allocate_per_req_cache({STATE_SLOT_CLASS: 3})
    builder.declare_step_buffers(runner.forward_vars)
    var = runner.forward_vars
    # Two requests in slots 2 and 0: 6 rows ending at 100, 3 rows ending at 10.
    positions = [94, 95, 96, 97, 98, 99, 7, 8, 9, 0]
    rows = [0, 0, 0, 0, 0, 0, 1, 1, 1, -1]
    var["positions"].np[: len(positions)] = positions
    var["batch_id_per_q_token"].np[: len(rows)] = rows
    batch = SimpleNamespace(total_seqs_num=2, state_slots_committed=[2, 0])
    builder.stage_step(batch, running_bs=3, running_tokens=len(positions))

    table = var["draft_block_tables"].np
    assert table[0, 0] == 1 + 2 * WINDOW.ring_blocks and not table[2].any()
    slots = var["draft_ring_slots"].np[: len(positions)]
    expected = WINDOW.context_slots(
        np.array(positions),
        np.array(rows),
        np.array([100, 10, 0]),
        table[:3],
    )
    assert (slots == expected).all() and slots[-1] == -1
    assert runner.h2d_groups[DRAFT_STATE_GROUP].published == 1


def test_dcp_is_refused():
    runner = _runner()
    runner.config.decode_context_parallel_size = 2
    with pytest.raises(ValueError, match="DCP"):
        _builder(runner)


class _Binding:
    """A draft attention module as the pool walk sees it."""

    def __init__(self, layer_num, page_tokens):
        self.layer_num, self.page_tokens = layer_num, page_tokens


def test_a_wider_decode_tp_takes_its_heads_of_every_page():
    """The cut a producer makes (`slot_region_tp_segments`, applied as in
    `_execute_block_slot_transfer`) hands decode rank r its heads of every page
    the writer views an entry in: here two 8-token pages of a 16-token block."""
    from atom.kv_transfer.disaggregation.types import KVTransferTensors
    from atom.spec_decode.draft_kv import DRAFT_KV_ROWS

    heads, page = 2, 8
    builder = _builder()
    builder._kv_pool = MhaKvPool(
        layers=2,
        block_size=BLOCK,
        num_kv_heads=heads,
        head_dim=16,
        kv_dtype=torch.bfloat16,
    )
    builder._pool_rows = {DRAFT_KV_ROWS: {_Binding(n, page): n for n in range(2)}}
    builder.allocate_per_req_cache({STATE_SLOT_CLASS: 3})
    for layer in range(2):
        k, v = builder.kv_pool.kv_views(layer)
        n, nh, hdx, bs, x = k.shape
        k_pages = k.view(n * (bs // page), nh, hdx, page, x)
        v_pages = v.view(n * (bs // page), nh, page // x, v.shape[3], x)
        for h in range(heads):  # through the writer's page view
            k_pages[:, h] = h + 1
            v_pages[:, h] = h + 1

    tensors = builder.get_kv_transfer_tensors()
    assert tensors.num_slots == 3 and not any(tensors.slot_region_tp_replicated)
    rows = builder._slot_rows(1)
    for (_, tensor), segments in zip(
        builder.kv_pool.region_tensors(), tensors.slot_region_tp_segments
    ):
        slot = tensor[rows].reshape(-1)
        assert sum(segments) == slot.numel()
        for rank in range(heads):
            got, offset = [], 0
            for seg in segments:
                piece = seg // heads
                got.append(slot[offset + rank * piece : offset + (rank + 1) * piece])
                offset += seg
            values = torch.cat(got).view(torch.bfloat16)
            assert bool((values == rank + 1).all()), (rank, values.unique())
    target = KVTransferTensors()
    target.merge_pages(tensors)
    assert target.slot_regions == tensors.slot_regions and target.num_slots == 3
    assert target.slot_region_tp_segments == tensors.slot_region_tp_segments


def test_an_image_is_read_from_the_targets_units_in_copy_order():
    from atom.kv_transfer.disaggregation.types import KVTransferRegion

    builder = _builder()
    builder.allocate_per_req_cache({STATE_SLOT_CLASS: 3})
    image = builder.checkpoint_image_bytes()
    unit = 3 * image // 4  # so an image takes two units, the second partly
    regions = [
        KVTransferRegion(
            base_addr=1000, total_bytes=10 * unit // 2, unit_bytes=unit // 2
        ),
        KVTransferRegion(
            base_addr=90000, total_bytes=10 * unit // 2, unit_bytes=unit // 2
        ),
    ]
    builder._target_regions_cache = regions
    builder.model_runner.state_runtime = SimpleNamespace(
        checkpoint_spec=SimpleNamespace(image_bytes=image, page_unit_bytes=unit)
    )
    ptrs, sizes = builder.page_unit_segments([5, 7, 9])
    # The state tier's packer takes int64 arrays (`pack_segments`).
    assert ptrs.dtype == sizes.dtype == np.int64
    assert int(sizes.sum()) == image
    assert ptrs[:3].tolist() == [
        regions[0].unit_addr(5),
        regions[1].unit_addr(5),
        regions[0].unit_addr(7),
    ]


M31 = dw.DraftWindow(
    window=4096, read_page=16, lookahead=14, block_size=128, anchor_margin=256
)


def test_the_m31_ring():
    assert M31.rebuild_tokens == 4126
    assert M31.ring_blocks == 35
    assert M31.trailing_slack == 35 * 128 - 4126


def test_a_ring_table_wraps_and_pads_to_the_null_entry():
    table = M31.ring_table(np.array([0, 2, -1]), cols=40)
    assert table[0, 0] == 1 and table[0, 34] == 35 and table[0, 35] == 1
    assert table[1, 0] == 1 + 2 * 35
    assert not table[2].any()


def test_a_window_holds_distinct_entries():
    table = M31.ring_table(np.array([3]), cols=200)[0]
    span = M31.ring_blocks
    for first in range(200 - span):
        assert len(set(table[first : first + span])) == span


def _sequences_for(ends, lengths):
    """Rows of a step: each request's last `length` positions up to `end`."""
    positions, row_seq = [], []
    for seq, (end, length) in enumerate(zip(ends, lengths)):
        positions += list(range(end - length, end))
        row_seq += [seq] * length
    return np.array(positions), np.array(row_seq)


def test_context_slots_drop_rows_below_the_ring_and_padding():
    ends = np.array([10_000, 300])
    positions, row_seq = _sequences_for(ends, [6_000, 300])
    positions = np.append(positions, 0)
    row_seq = np.append(row_seq, -1)
    table = M31.ring_table(np.array([0, 1]), cols=100)
    slots = M31.context_slots(positions, row_seq, ends, table)
    first = slots[:6_000]
    kept = first >= 0
    assert kept.sum() == M31.ring_tokens
    assert positions[:6_000][kept].min() == 10_000 - M31.ring_tokens
    # Every kept row of one launch lands on its own slot.
    assert len(set(first[kept])) == kept.sum()
    assert (slots[6_000:6_300] >= 0).all()
    assert slots[-1] == -1


def test_a_kept_row_lands_at_its_block_and_offset():
    table = M31.ring_table(np.array([1]), cols=100)
    slots = M31.context_slots(
        np.array([4_500]), np.array([0]), np.array([4_501]), table
    )
    block = 1 + 35 + (4_500 // 128) % 35
    assert slots[0] == block * 128 + 4_500 % 128


def _config(arch="DSparkMiniMaxDraftModel", window=4096):
    draft = SimpleNamespace(architectures=[arch], sliding_window=window)
    spec = SimpleNamespace(draft_model_hf_config=draft, num_speculative_tokens=7)
    return SimpleNamespace(speculative_config=spec, kv_cache_block_size=128)


def test_the_m31_draft_has_a_window():
    assert dw.draft_window(_config()) == M31


@pytest.mark.parametrize(
    "config",
    [
        _config(arch="K3DSparkModel"),
        _config(window=0),
        _config(arch=None),
        SimpleNamespace(speculative_config=None, kv_cache_block_size=128),
    ],
)
def test_other_drafts_have_none(config):
    assert dw.draft_window(config) is None
