# SPDX-License-Identifier: MIT
"""Cutting a TP1 producer's per-head KV units into a TP-sharded consumer's."""

import pytest
import torch

from atom.kv_transfer.disaggregation.mha_head_transfer import consumer_head_slices
from atom.kv_transfer.disaggregation.page_region import page_region
from atom.kv_transfer.disaggregation.types import KVTransferTensors


@pytest.mark.parametrize("runs", [1, 8])
@pytest.mark.parametrize("heads, tp", [(4, 4), (8, 4), (4, 8), (2, 4), (4, 1)])
def test_each_rank_receives_exactly_its_own_heads(runs, heads, tp):
    width = 64
    src = torch.arange(runs * heads * width).remainder(251).to(torch.uint8)
    src = src.view(runs, heads, width)
    dst_heads = max(1, heads // tp)
    for rank in range(tp):
        dst = torch.full((runs, dst_heads, width), 255, dtype=torch.uint8)
        for s, d, n in consumer_head_slices(
            src.numel(), dst.numel(), (runs, heads), (runs, dst_heads), rank, tp
        ):
            dst.view(-1)[d : d + n] = src.view(-1)[s : s + n]
        first = rank * dst_heads if heads >= tp else rank // (tp // heads)
        assert torch.equal(dst, src[:, first : first + dst_heads])


def test_a_unit_that_is_not_per_head_moves_whole():
    assert consumer_head_slices(18432, 18432, None, None, 3, 4) == [(0, 0, 18432)]


@pytest.mark.parametrize(
    "args",
    [
        (128, 128, None, (1, 1), 0, 4),  # per head on one end only
        (128, 64, None, None, 0, 4),  # not per head, sized differently
        (128, 32, (8, 8), (1, 2), 0, 4),  # run counts disagree
        (128, 64, (1, 8), (1, 2), 0, 4),  # a head is wider on one end
        (128, 32, (1, 8), (1, 2), 4, 4),  # rank outside the consumer's TP
        (4 * 64, 64, (1, 4), (1, 1), 0, 3),  # neither shards nor replicates
    ],
)
def test_geometry_the_two_ends_cannot_share_is_refused(args):
    with pytest.raises(ValueError):
        consumer_head_slices(*args)


def test_one_stage_local_contributor_makes_roles_unpairable():
    def tensors(global_roles):
        page = page_region(torch.zeros(2, 64, dtype=torch.uint8), semantic_role="r")
        return KVTransferTensors(pages=[page], roles_are_global=global_roles)

    both, mixed = tensors(True), tensors(True)
    both.merge_pages(tensors(True))
    mixed.merge_pages(tensors(False))
    assert both.roles_are_global and not mixed.roles_are_global
