# SPDX-License-Identifier: Apache-2.0

"""The GLM-5 fused indexer's selection tree, checked on the CPU.

``atom/models/glm52/mono/kernels/layout.py`` sizes the tree a BUILD carries and
``glm/kernel.py``'s ``tree_*`` size the part a TOKEN runs.  The kernel's half is
written against FlyDSL and cannot be imported without a GPU toolchain, so this
file mirrors it on Python ints -- the same expressions, line for line -- and
pins the mirror to a brute-force top-k.  What that buys:

* the tree's answer IS the global top-k, on distinct and on heavily tied scores;
* every level's output is dense and ascending by key index, which is the
  invariant the next level's flat addressing (``entry_pair``) depends on;
* exactly one level is a token's final one, and it is the one that publishes.
"""

import random

import pytest

from atom.models.glm52.mono.kernels.layout import (
    INDEX_SEG,
    index_fan,
    index_levels,
    index_lists,
    index_seg,
    index_select_capacity,
    index_select_stage,
    indexer_uses_tiles,
    layout,
    stage_tasks,
)

TOPK = 2048


# --------------------------------------------------------------- build sizing


def test_two_levels_reach_the_declared_max_position():
    """GLM-5.2 declares ``max_position_embeddings = 1048576``; the leaf is fixed
    at 16384 and the fan is 8, so exactly two merge levels cover it."""

    assert INDEX_SEG == 16384
    assert index_fan(1048576, TOPK) == 8
    assert index_levels(1048576, TOPK) == 2
    assert INDEX_SEG * 8**2 == 1048576


@pytest.mark.parametrize(
    ("index_max_seq", "depth", "slots"),
    [
        (4096, 0, [1]),
        (16384, 0, [1]),
        (65536, 1, [4, 1]),
        (131072, 1, [8, 1]),
        (1048576, 2, [64, 8, 1]),
    ],
)
def test_level_shape(index_max_seq, depth, slots):
    assert index_levels(index_max_seq, TOPK) == depth
    assert [
        index_lists(index_max_seq, level, TOPK) for level in range(depth + 1)
    ] == slots
    # The leaf never grows: that is the whole point, and it is what keeps
    # ``kernel.py``'s ``index_lds_words`` constant at every context.
    assert index_seg(index_max_seq) == min(index_max_seq, INDEX_SEG)


def test_tiles_are_the_context_parallel_feature_not_a_capacity_one():
    """Without CP there is nothing to shard across, so a long context goes
    through the tree; with CP the tiled radix still owns it, and the local
    selector stays a bare leaf because its score plane is one leaf wide."""

    assert not indexer_uses_tiles(1048576, False)
    assert indexer_uses_tiles(1048576, True)
    assert not indexer_uses_tiles(16384, True)
    assert index_select_capacity(1048576, False) == 1048576
    assert index_select_capacity(1048576, True) == INDEX_SEG
    assert index_levels(index_select_capacity(1048576, True), TOPK) == 0


def test_final_level_keeps_the_stage_name_it_always_had():
    assert index_select_stage(0, 0) == "index_select"
    assert [index_select_stage(level, 2) for level in range(3)] == [
        "index_sel0",
        "index_sel1",
        "index_select",
    ]


@pytest.mark.parametrize("indexer_cp", [False, True])
def test_a_one_leaf_build_adds_nothing(indexer_cp):
    """At ``index_max_seq <= INDEX_SEG`` the tree is zero levels deep, so the
    scratch map and the stage table must be exactly what they were."""

    scratch, _ = layout(8, 16, 4, TOPK, True, INDEX_SEG, indexer_cp=indexer_cp)
    tasks = stage_tasks(8, 16, TOPK, True, INDEX_SEG, indexer_cp=indexer_cp, npes=4)
    assert [name for name in scratch if name.startswith("isel")] == []
    assert [task for task in tasks if task[0].startswith("index_sel")] == [
        ("index_select", 8)
    ]


def test_a_deep_build_adds_one_region_and_one_stage_per_merge_level():
    scratch, _ = layout(8, 16, 4, TOPK, True, 1048576)
    tasks = dict(stage_tasks(8, 16, TOPK, True, 1048576, npes=4))
    assert [name for name in scratch if name.startswith("isel")] == ["isel0", "isel1"]
    # 64 lists then 8, each holding ``topk`` (score, key index) pairs.
    assert tasks["index_sel0"] == 8 * 64
    assert tasks["index_sel1"] == 8 * 8
    assert tasks["index_select"] == 8 * 1


# ------------------------------------------------- the token's half, mirrored


def _cdiv(a, b):
    return -(-a // b)


def tree_lists(bound, level, seg, fan):
    n = _cdiv(max(bound, 0), seg)
    for _ in range(level):
        n = _cdiv(n, fan)
    return n


def tree_inputs(bound, level, j, seg, fan, k):
    if level == 0:
        return max(min(bound - j * seg, seg), 0)
    below = tree_lists(bound, level - 1, seg, fan)
    first = j * fan
    count = min(below - first, fan)
    last = first + count - 1
    valid = min(tree_inputs(bound, level - 1, last, seg, fan, k), k)
    return max(count - 1, 0) * k + valid


def tree_count(bound, level, seg, fan):
    if level == 0:
        return max(tree_lists(bound, 0, seg, fan), 1)
    return (
        tree_lists(bound, level, seg, fan)
        if tree_lists(bound, level - 1, seg, fan) > 1
        else 0
    )


def tree_final(bound, level, seg, fan):
    return tree_lists(bound, level, seg, fan) <= 1


def _run_tree(scores, depth, seg, fan, k):
    """Return ``(publishing level, published key indices)``, exactly as the
    kernel's level loop does: rank, emit dense and index-ordered, merge."""

    bound = len(scores)
    below, published = None, None
    for level in range(depth + 1):
        if tree_count(bound, level, seg, fan) == 0:
            # The token already reached a single list below; the kernel's
            # ``j < tree_count`` gate is what stops this level from running.
            continue
        emitted = []
        for j in range(tree_count(bound, level, seg, fan)):
            n_in = tree_inputs(bound, level, j, seg, fan, k)
            if level == 0:
                values = [(scores[j * seg + i], j * seg + i) for i in range(n_in)]
            else:
                values = []
                for sub in range(fan):
                    source = j * fan + sub
                    if source < len(below):
                        values += below[source]
                # Dense: every list but a level's last holds exactly ``k``.
                assert len(values) == n_in
            ranked = sorted(
                range(len(values)), key=lambda t: (-values[t][0], values[t][1])
            )
            winners = sorted(ranked[: min(k, len(values))], key=lambda t: values[t][1])
            emitted.append([values[t] for t in winners])
        if tree_final(bound, level, seg, fan):
            assert published is None, "two levels claimed to be final"
            assert len(emitted) == 1
            published = (level, [ident for _, ident in emitted[0]])
        below = emitted
    assert published is not None
    return published


def _bounds(index_max_seq, depth, seg, fan):
    """Structural edges of the tree, plus random interior points."""

    out = {1, 2, TOPK, TOPK + 1, index_max_seq}
    for level in range(depth + 1):
        for multiple in (1, 2, 3):
            edge = seg * fan**level * multiple
            out |= {edge - 1, edge, edge + 1}
    rng = random.Random(index_max_seq)
    out |= {rng.randint(1, index_max_seq) for _ in range(8)}
    return sorted(b for b in out if 1 <= b <= index_max_seq)


@pytest.mark.parametrize("index_max_seq", [16384, 65536, 131072, 1048576])
@pytest.mark.parametrize("ties", [False, True])
def test_the_tree_is_the_exact_global_top_k(index_max_seq, ties):
    seg, fan = index_seg(index_max_seq), index_fan(index_max_seq, TOPK)
    depth = index_levels(index_max_seq, TOPK)
    for bound in _bounds(index_max_seq, depth, seg, fan):
        rng = random.Random(bound * 7 + int(ties))
        if ties:
            # Four distinct values over up to 1M keys: the threshold digit is a
            # tie for nearly every candidate, which is where a selector that
            # mishandles ``need_eq`` emits a short or an overlong row.
            scores = [rng.randint(0, 3) for _ in range(bound)]
        else:
            scores = rng.sample(range(10 * bound + 10), bound)
        _, got = _run_tree(scores, depth, seg, fan, TOPK)
        want = sorted(
            sorted(range(bound), key=lambda i: (-scores[i], i))[: min(TOPK, bound)]
        )
        assert got == want
        # Ascending by key index, and exactly ``min(topk, bound)`` long.
        assert got == sorted(set(got))
        assert len(got) == min(TOPK, bound)


@pytest.mark.parametrize("index_max_seq", [16384, 65536, 131072, 1048576])
def test_a_short_token_ends_the_tree_early(index_max_seq):
    """One build serves every context: a token below the build's maximum runs
    fewer lists and publishes from a lower level, on its traced ``bound``."""

    seg, fan = index_seg(index_max_seq), index_fan(index_max_seq, TOPK)
    depth = index_levels(index_max_seq, TOPK)
    for bound in sorted({1, seg, min(seg + 1, index_max_seq), index_max_seq}):
        ran = [
            level
            for level in range(depth + 1)
            if tree_count(bound, level, seg, fan) > 0
        ]
        finals = [level for level in ran if tree_final(bound, level, seg, fan)]
        assert len(finals) == 1, "exactly one level a token runs is its final one"
        # It is the depth this token's own context would have been built for.
        assert finals[0] == index_levels(bound, TOPK)
        assert ran == list(range(finals[0] + 1))
