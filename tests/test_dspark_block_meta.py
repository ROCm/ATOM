# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""The DSpark draft block's paged metadata in one launch (`dspark_block_meta`)
gives the torch path's numbers, element for element: the proposer's slots and
context lengths (no DCP) and MiniMax-M3's `prepare_paged_block` tables."""

from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

T = 7


def _draft(window):
    from torch import nn

    from atom.models.minimax_m3.dspark import MiniMaxM3DSpark

    d = MiniMaxM3DSpark.__new__(MiniMaxM3DSpark)
    nn.Module.__init__(d)
    d.max_num_seqs, d.sliding_window, d._bt16 = 16, window, None
    return d


def _torch_path(d, bt, anchors, offsets, pages, sched, entry, pad_len):
    from atom.spec_decode.dspark_proposer import DSparkProposer

    rb = bt.shape[0]
    pos = anchors.view(rb, 1) + offsets.view(1, T)
    page_idx, in_table = DSparkProposer._block_page_idx(pos, entry, bt)
    slots = torch.gather(bt, 1, page_idx).to(torch.int64) * entry + pos % entry
    slots.masked_fill_(~in_table, -1)
    ctx = (anchors + 1 + T).to(torch.int32)
    slots[sched:] = -1
    ctx[sched:] = pad_len
    md = SimpleNamespace()
    d.prepare_paged_block(md, bt, pos, slots, ctx, pages, sched, entry)
    return pos, slots, ctx, md.dspark_bt16_rows.clone(), md.dspark_ctx_rows.clone()


@pytest.mark.parametrize("window", [0, 512])
@pytest.mark.parametrize("rb,sched", [(1, 1), (4, 3), (8, 8), (12, 9)])
def test_one_launch_matches_the_torch_path(window, rb, sched):
    g = torch.Generator().manual_seed(rb * 31 + sched + window)
    entry, W = 128, 40
    bt = torch.randint(0, 5000, (rb, W), generator=g, dtype=torch.int32).cuda()
    # anchors near the table end, past the allocation, and inside it
    anchors = torch.randint(0, W * entry, (rb,), generator=g).cuda()
    anchors[0] = W * entry - 3
    pages = torch.randint(0, W + 1, (sched,), generator=g).tolist()
    offsets = torch.arange(1, T + 1, dtype=torch.int64, device="cuda")
    want = _torch_path(_draft(window), bt, anchors, offsets, pages, sched, entry, 8)
    d = _draft(window)
    pos = torch.empty(rb, T, dtype=torch.int64, device="cuda")
    slots = torch.empty(rb, T, dtype=torch.int64, device="cuda")
    ctx = torch.empty(rb, dtype=torch.int32, device="cuda")
    md = SimpleNamespace()
    d.prepare_paged_block_fused(
        md, bt, anchors, offsets, pos, slots, ctx, pages, sched, entry, 8
    )
    got = (pos, slots, ctx, md.dspark_bt16_rows, md.dspark_ctx_rows)
    for name, a, b in zip(
        ("positions", "slots", "ctx_lens", "tables", "lens"), want, got
    ):
        assert torch.equal(a, b), name
