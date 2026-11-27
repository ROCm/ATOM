# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""The NVFP4 BF16 tail is one 16-token tile per request per layer."""

import torch

from atom.model_ops.attentions.nvfp4_hp_ring import HP_BLOCK_SIZE, Nvfp4HpRing

HEAD = 576
PAGE = 64


def _ring():
    return Nvfp4HpRing(num_reqs=2, layers=3, head_dim=HEAD)


def _append(ring, layer, req, positions, value):
    """Tokens of one request at ``positions``, every element ``value``."""
    positions = torch.as_tensor(positions)
    return ring.append_tokens(
        layer,
        torch.full_like(positions, req),
        positions,
        torch.full((positions.numel(), HEAD), value, dtype=torch.bfloat16),
        PAGE,
    )


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
    _append(ring, 0, 1, [16], 1.25)
    tiles, valid, pages, tile_ids = _append(ring, 0, 1, [17], -0.5)
    assert (valid.tolist(), pages.tolist(), tile_ids.tolist()) == ([2], [0], [1])
    assert torch.all(tiles[0, 0] == 1.25)
    assert torch.all(tiles[0, 1] == -0.5)
    assert torch.all(tiles[0, 2:] == 0)


def test_a_finished_tile_is_not_carried_into_the_next():
    ring = _ring()
    _append(ring, 0, 0, list(range(16)), 3.0)
    tiles, valid, pages, tile_ids = _append(ring, 0, 0, [16], 0.25)
    assert (valid.item(), pages.item(), tile_ids.item()) == (1, 0, 1)
    assert torch.all(tiles[0, 0] == 0.25)
    assert torch.all(tiles[0, 1:] == 0)


def test_position_65_opens_tile_0_of_the_next_page():
    ring = _ring()
    _append(ring, 0, 0, [64], 1.0)
    tiles, valid, pages, tile_ids = _append(ring, 0, 0, [65], 2.0)
    assert (valid.item(), pages.item(), tile_ids.item()) == (2, 1, 0)
    assert tiles[0, 0, 0].item() == 1
    assert tiles[0, 1, 0].item() == 2


def test_requests_do_not_share_slots():
    ring = _ring()
    _append(ring, 0, 0, [3], 1.0)
    tiles, valid, _, _ = _append(ring, 0, 1, [3], 4.0)
    assert valid.item() == 4
    assert tiles[0, 3, 0].item() == 4
    assert torch.all(tiles[0, :3] == 0)
