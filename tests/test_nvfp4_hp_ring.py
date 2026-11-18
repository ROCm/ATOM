# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The NVFP4 BF16 tail is one 16-token tile per request per layer."""

import torch

from atom.model_ops.attentions.nvfp4_hp_ring import (
    HP_BLOCK_SIZE,
    Nvfp4HpRing,
    tile_spans,
)

HEAD = 576
PAGE = 64


def _ring(**kwargs):
    args = {"num_reqs": 2, "layers": 3, "head_dim": HEAD, "rewind": 0}
    args.update(kwargs)
    return Nvfp4HpRing(**args)


def test_bytes_follow_the_ring_not_the_page():
    ring = _ring()
    assert ring.ring_size == HP_BLOCK_SIZE
    assert ring.nbytes() == 2 * 3 * HP_BLOCK_SIZE * HEAD * 2


def test_partial_tile_is_returned_ready_to_quantize():
    ring = _ring()
    token = torch.arange(HEAD, dtype=torch.float32)
    tiles, valid, pages, tile_ids = ring.append_tokens(
        1,
        torch.tensor([0]),
        torch.tensor([17]),
        token.to(torch.bfloat16).unsqueeze(0),
        PAGE,
    )
    assert valid.tolist() == [2]
    assert pages.tolist() == [0]
    assert tile_ids.tolist() == [1]
    assert tiles.shape == (1, 16, HEAD)
    assert torch.all(tiles[0, 2:] == 0)


def test_decode_rereads_the_bf16_tail():
    ring = _ring()
    first = torch.full((HEAD,), 1.25).to(torch.bfloat16)
    second = torch.full((HEAD,), -0.5).to(torch.bfloat16)
    ring.store_tail(0, 1, 16, first.unsqueeze(0))
    tiles, valid, pages, tile_ids = ring.append_tokens(
        0,
        torch.tensor([1]),
        torch.tensor([17]),
        second.unsqueeze(0),
        PAGE,
    )
    assert (valid.tolist(), pages.tolist(), tile_ids.tolist()) == ([2], [0], [1])
    assert torch.equal(tiles[0, 0], first)
    assert torch.equal(tiles[0, 1], second)
    assert torch.all(tiles[0, 2:] == 0)


def test_a_finished_tile_is_not_carried_into_the_next():
    ring = _ring()
    old = torch.full((16, HEAD), 3.0).to(torch.bfloat16)
    ring.store_tail(0, 0, 0, old)
    new = torch.full((HEAD,), 0.25).to(torch.bfloat16)
    tiles, valid, pages, tile_ids = ring.append_tokens(
        0,
        torch.tensor([0]),
        torch.tensor([16]),
        new.unsqueeze(0),
        PAGE,
    )
    assert (valid.item(), pages.item(), tile_ids.item()) == (1, 0, 1)
    assert torch.equal(tiles[0, 0], new)
    assert torch.all(tiles[0, 1:] == 0)


def test_position_65_opens_tile_0_of_the_next_page():
    ring = _ring()
    ring.store_tail(0, 0, 64, torch.ones(1, HEAD, dtype=torch.bfloat16))
    tiles, valid, pages, tile_ids = ring.append_tokens(
        0,
        torch.tensor([0]),
        torch.tensor([65]),
        torch.full((1, HEAD), 2.0, dtype=torch.bfloat16),
        PAGE,
    )
    assert (valid.item(), pages.item(), tile_ids.item()) == (2, 1, 0)
    assert tiles[0, 0, 0].item() == 1
    assert tiles[0, 1, 0].item() == 2


def test_requests_do_not_share_slots():
    ring = _ring()
    ring.append_tokens(
        0,
        torch.tensor([0]),
        torch.tensor([3]),
        torch.ones(1, HEAD, dtype=torch.bfloat16),
        PAGE,
    )
    tiles, valid, _, _ = ring.append_tokens(
        0,
        torch.tensor([1]),
        torch.tensor([3]),
        torch.full((1, HEAD), 4.0, dtype=torch.bfloat16),
        PAGE,
    )
    assert valid.item() == 4
    assert tiles[0, 3, 0].item() == 4
    assert torch.all(tiles[0, :3] == 0)


def test_prefill_spans_include_the_short_tail():
    assert tile_spans(17, PAGE) == [(0, 16, 0, 0), (16, 1, 0, 1)]
    assert tile_spans(64, PAGE) == [
        (0, 16, 0, 0),
        (16, 16, 0, 1),
        (32, 16, 0, 2),
        (48, 16, 0, 3),
    ]
    assert tile_spans(65, PAGE)[-1] == (64, 1, 1, 0)
    assert tile_spans(0, PAGE) == []
