# SPDX-License-Identifier: MIT
"""PAGE units as an MHA pool describes them, paired by spec, cut by KV head.

CPU only: the pool, `pair_units` and the head cut import no aiter.
"""

from dataclasses import replace

import numpy as np
import pytest
import torch

from atom.kv_transfer.disaggregation.sharded_transfer import (
    consumer_head_slices,
    kv_heads_of_rank,
)
from atom.kv_transfer.disaggregation.types import pair_units
from atom.model_ops.attentions.mha_kv_pool import MhaKvPool
from atom.model_ops.attentions.pool_layout.entry_arena import EntryField

BLOCK, HEAD_DIM, LAYERS, MTP = 128, 128, 8, 8
SPARSE = (0, 2, 6)  # layers with an indexer, viewed in page-16 SHUFFLE
INDEX = (128, torch.float8_e4m3fn)
PARTITIONS = [(2, 2, 2, 2), (3, 2, 2, 1)]


def stage_pages(layers, heads, kv_dtype=torch.float8_e4m3fn, draft=False, index=INDEX):
    """A stage's (or an unsplit model's) PAGE units: target pool, then draft."""
    sparse = [n for n in layers if n in SPARSE]
    dim, dtype = index
    pool = MhaKvPool(
        layers=len(layers),
        block_size=BLOCK,
        num_kv_heads=heads,
        head_dim=HEAD_DIM,
        kv_dtype=kv_dtype,
        extra_fields=[
            EntryField(
                "index", len(sparse), (BLOCK * dim * dtype.itemsize,), torch.uint8
            )
        ],
    )
    pool.allocate(4, "cpu")
    pages = pool.transfer_pages(
        "mha",
        "mha.kv",
        list(layers),
        [16 if n in SPARSE else BLOCK for n in layers],
        {"index": (sparse, dim, dtype)},
    )
    if draft:
        pool = MhaKvPool(
            layers=1,
            block_size=BLOCK,
            num_kv_heads=heads,
            head_dim=HEAD_DIM,
            kv_dtype=kv_dtype,
        )
        pool.allocate(4, "cpu")
        pages += pool.transfer_pages("draft", "draft", [LAYERS + 1])
    return pages


def stage_layers(partition, rank):
    start = sum(partition[:rank])
    layers = list(range(start, start + partition[rank]))
    last = rank == len(partition) - 1
    return layers + [MTP] * last, start, None if last else start + partition[rank]


def specs(pages):
    return [p.unit_spec for p in pages]


def test_the_pool_names_each_unit_by_its_layer_and_splits_it_by_head():
    pages = stage_pages(range(LAYERS), heads=4)
    units = {(p.unit_spec.field, p.unit_spec.layer): p for p in pages}
    # The roles are the pool's own rows; only the spec names the layer.
    assert [p.region.semantic_role for p in pages[:2]] == [
        "mha.kv.k.layer_0",
        "mha.kv.k.layer_1",
    ]
    assert units["k", 0].head_major_runs == units["v_scale", 6].head_major_runs
    assert units["k", 0].head_major_runs == (BLOCK // 16, 4)  # page-16
    assert units["v", 1].head_major_runs == (1, 4)  # one page-128 run
    assert units["index", 6].head_major_runs is None  # the same on every rank
    assert (units["k", 3].unit_spec.dim, units["k_scale", 3].unit_spec.dim) == (128, 1)
    index = units["index", 2].unit_spec
    assert (index.dim, index.dtype) == (128, "torch.float8_e4m3fn")


@pytest.mark.parametrize("heads", [1, 4])
@pytest.mark.parametrize("partition", PARTITIONS)
@pytest.mark.parametrize("draft", [False, True])
def test_every_stage_pairs_into_the_unsplit_consumer(heads, partition, draft):
    """An MTP layer sharing the target pool and a draft pool ride the last stage."""
    consumer = specs(stage_pages([*range(LAYERS), MTP], 1, draft=draft))
    claimed = []
    for rank in range(len(partition)):
        layers, first, end = stage_layers(partition, rank)
        last = rank == len(partition) - 1
        ours = specs(stage_pages(layers, heads, draft=draft and last))
        cmap = pair_units(ours, consumer, first, end)
        assert [consumer[c] for c in cmap] == ours
        claimed += cmap
    assert sorted(claimed) == list(range(len(consumer)))


def test_a_draft_only_the_consumer_holds_is_refused_on_the_last_stage():
    consumer = specs(stage_pages([*range(LAYERS), MTP], 1, draft=True))
    layers, first, end = stage_layers((2, 2, 2, 2), 3)
    with pytest.raises(ValueError, match="not held here"):
        pair_units(specs(stage_pages(layers, 4)), consumer, first, end)


@pytest.mark.parametrize(
    "change",
    [
        {"dim": 64},
        {"dtype": "torch.bfloat16"},
        {"layer": 100},
        {"namespace": "draft"},
    ],
)
def test_a_unit_the_consumer_holds_differently_is_refused(change):
    consumer = specs(stage_pages(range(LAYERS), 1))
    ours = specs(stage_pages(range(LAYERS), 4))
    ours[-1] = replace(ours[-1], **change)
    with pytest.raises(ValueError, match="mismatch|no unit"):
        pair_units(ours, consumer)


def test_equal_bytes_do_not_hide_another_index_format():
    consumer = specs(stage_pages(range(LAYERS), 1))
    ours = specs(stage_pages(range(LAYERS), 1, index=(64, torch.bfloat16)))
    with pytest.raises(ValueError, match="mismatch"):
        pair_units(ours, consumer)


def test_a_consumer_naming_one_unit_twice_is_refused():
    consumer = specs(stage_pages(range(LAYERS), 1))
    with pytest.raises(ValueError, match="twice"):
        pair_units(consumer, consumer + consumer[:1])


@pytest.mark.parametrize("runs", [1, 8])
@pytest.mark.parametrize("heads, tp", [(4, 4), (8, 4), (4, 8), (2, 4), (4, 1)])
def test_each_rank_receives_exactly_its_own_heads(runs, heads, tp):
    width = 64
    src = (np.arange(runs * heads * width) % 251).astype(np.uint8)
    for rank in range(tp):
        first, count = kv_heads_of_rank(rank, tp, heads)
        dst = np.full(runs * count * width, 255, np.uint8)
        cuts = consumer_head_slices(
            src.size, dst.size, (runs, heads), (runs, count), rank, tp, "k"
        )
        for s, d, n in cuts:
            dst[d : d + n] = src[s : s + n]
        want = src.reshape(runs, heads, width)[:, first : first + count]
        assert np.array_equal(dst.reshape(runs, count, width), want)


def test_a_unit_that_is_not_per_head_moves_whole():
    assert consumer_head_slices(18432, 18432, None, None, 3, 4, "index").tolist() == [
        [0, 0, 18432]
    ]


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
def test_geometry_the_two_ends_cannot_share_is_refused_by_name(args):
    with pytest.raises(ValueError, match="v_scale|KV heads"):
        consumer_head_slices(*args, "v_scale")
