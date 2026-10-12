# SPDX-License-Identifier: MIT
"""I3 (DESIGN_v3 4.9) for the V4.1 mono indexer: for any token bound, every
mailbox cell a stage polls is one its producer writes this step. The functions
under test are the ones the kernel calls (``mono/index_plan.py``)."""

import random

import numpy as np
import pytest

from atom.models.deepseek_v41.mono import index_plan as ip

EDGES = [0, 1, 2, ip.CHUNK - 1, ip.CHUNK, ip.CHUNK + 1, ip.TOPK - 1, ip.TOPK,
         ip.TOPK + 1, ip.SEG - 1, ip.SEG, ip.SEG + 1, 2 * ip.SEG - 1, 2 * ip.SEG,
         2 * ip.SEG + 1, 1 << 20, (1 << 21) - 1]  # fmt: skip


def bounds():
    rng = random.Random(0)
    return EDGES + [rng.randrange(0, 1 << 21) for _ in range(300)]


def written_chunks(bound, chunks_fn=ip.chunks):
    return {ch for ch in range(chunks_fn(bound) + 1) if ip.chunk_written(bound, ch)}


def polled_chunks(bound, seg_chunks=ip.seg_chunks):
    return {
        ip.CHUNKS_PER_SEG * s + c
        for s in range(ip.segments(bound))
        for c in range(seg_chunks(bound, s))
    }


@pytest.mark.parametrize("k", [ip.TOPK, ip.TOPK_BLOCKS])
def test_every_polled_cell_is_written(k):
    for bound in bounds():
        assert polled_chunks(bound) == written_chunks(bound), bound
        for level in range(1, ip.levels(bound, k) + 1):
            below = ip.lists(bound, level - 1, k)
            for j in range(ip.lists(bound, level, k)):
                # list j reads its children end to end: entry i is child
                # fan j + i // k, slot i % k; each child must hold every slot
                # read from it as a valid, written entry
                n = ip.list_inputs(bound, level, j, k)
                for c in range(ip.fan(k)):
                    read = min(k, max(n - c * k, 0))
                    child = j * ip.fan(k) + c
                    if read:
                        assert child < below, (bound, level, j, c)
                        assert read <= ip.list_valid(bound, level - 1, child, k)


@pytest.mark.parametrize("k", [ip.TOPK, ip.TOPK_BLOCKS])
def test_every_list_but_a_levels_last_is_full(k):
    for bound in bounds():
        for level in range(ip.levels(bound, k) + 1):
            n = ip.lists(bound, level, k)
            for j in range(n - 1):
                assert ip.list_valid(bound, level, j, k) == k, (bound, level, j)


@pytest.mark.parametrize("k", [ip.TOPK, ip.TOPK_BLOCKS])
def test_the_final_list_holds_the_selection(k):
    depth_max = ip.levels(max(bounds()), k)
    for bound in bounds():
        top = ip.final_level(bound, depth_max, k)
        assert ip.lists(bound, top, k) <= 1, bound
        assert ip.list_valid(bound, top, 0, k) == ip.selected(bound, k), bound


def test_positive_control_floor_chunks_leave_logits_unpolled():
    """A segment that polls floor(len / CHUNK) chunks misses a partial chunk: the
    property must report it (so the tests above can fail)."""

    def floor_seg_chunks(bound, s):
        return ip.seg_len(bound, s) // ip.CHUNK

    missed = [
        b for b in bounds() if polled_chunks(b, floor_seg_chunks) != written_chunks(b)
    ]
    assert missed and ip.CHUNK + 1 in missed


def test_int32_guard():
    with pytest.raises(AssertionError):
        ip.i32(1 << 31)


def test_positive_control_assuming_full_children_reads_invalid_entries():
    """A merge that takes every child as full (count x k inputs) reads past a
    partial last child's valid entries: the check above must catch it."""
    k = ip.TOPK
    caught = []
    for bound in bounds():
        for level in range(1, ip.levels(bound, k) + 1):
            below = ip.lists(bound, level - 1, k)
            for j in range(ip.lists(bound, level, k)):
                count = min(below - j * ip.fan(k), ip.fan(k))
                last = j * ip.fan(k) + count - 1
                if count * k > (count - 1) * k + ip.list_valid(
                    bound, level - 1, last, k
                ):
                    caught.append(bound)
    assert caught


def test_block_segments_poll_written_chunks_only():
    for bound in bounds():
        blocks = ip.blocks(bound)
        polled = set()
        for j in range(ip.segments(blocks)):
            first, count = ip.block_chunks(bound, j)
            polled |= set(range(first, first + count))
        assert polled == written_chunks(bound), bound


def test_flat_tasks_cover_every_token_index_once():
    rng = random.Random(1)
    for _ in range(300):
        counts = [rng.choice([0, 1, 2, rng.randrange(0, 70)]) for _ in range(6)]
        seen = [ip.flat_task(counts, task) for task in range(ip.flat_total(counts))]
        expected = [(t, j) for t, n in enumerate(counts) for j in range(n)]
        assert seen == expected, counts


CTAS = 256


def ragged_step(rng, chunks_max):
    """(bounds, owners): up to 8 requests of 1..6 adjacent tokens each, a few
    pad tokens (owner 0, bound 0) after them."""
    owners = []
    for request in range(1, rng.randrange(1, 9) + 1):
        owners += [request * 1000] * rng.randrange(1, 7)
    owners += [0] * rng.randrange(0, 3)
    bounds = [0 if o == 0 else rng.randrange(0, chunks_max * ip.CHUNK) for o in owners]
    return bounds, owners


def scored_cells(bounds, owners, width_log2_max, score_tasks=None):
    """Every (token, chunk) the score stage writes, a task's chunks for each of
    its tokens as the kernel runs them (``index_score.stage_iscore``)."""
    plan = score_tasks or ip.score_tasks
    counts, spans, width_log2 = plan(bounds, owners, CTAS, width_log2_max)
    assert width_log2 <= width_log2_max
    width, cells = 1 << width_log2, []
    for task in range(ip.flat_total(counts)):
        lead, i = ip.flat_task(counts, task)
        ch0 = i << width_log2
        for row in range(lead, lead + spans[lead]):
            assert owners[row] == owners[lead], (lead, row)
            for m in range(1 << width_log2_max):
                if ip.task_writes(bounds[row], ch0, width, m):
                    cells.append((row, ch0 + m))
    return cells


@pytest.mark.parametrize("width_log2_max", [0, 2, 3])
@pytest.mark.parametrize("chunks_max", [5, 8, 64, 2048, 8192])
def test_score_tasks_write_every_chunk_once(chunks_max, width_log2_max):
    rng = random.Random(chunks_max)
    for _ in range(100):
        bounds, owners = ragged_step(rng, chunks_max)
        expected = sorted(
            (t, ch) for t, b in enumerate(bounds) for ch in written_chunks(b)
        )
        cells = scored_cells(bounds, owners, width_log2_max)
        assert sorted(cells) == expected, (bounds, owners)


def test_score_tasks_share_and_widen_only_with_a_task_for_every_cta():
    owners = [1] * 6 + [2] * 3 + [3]
    # 3 runs x 256 chunks: shared; 2 chunks a task keeps 384 >= 256, 4 would not
    counts, spans, width_log2 = ip.score_tasks([CTAS * ip.CHUNK] * 10, owners, CTAS, 2)
    assert spans[0] == 6 and spans[6] == 3 and spans[9] == 1
    assert width_log2 == 1 and ip.flat_total(counts) == 3 * CTAS // 2
    # just past a take-all selection, 5 chunks a token: 50 tasks, nothing shared
    # or widened
    counts, spans, width_log2 = ip.score_tasks([ip.TOPK + 1] * 10, owners, CTAS, 2)
    assert set(spans) == {1} and width_log2 == 0
    assert ip.flat_total(counts) == 50
    # 1M columns: the widest group the build allows
    _, _, width_log2 = ip.score_tasks([1 << 20] * 10, owners, CTAS, 2)
    assert width_log2 == 2


def test_a_selection_of_every_column_is_not_scored():
    # bound <= TOPK: the top-k takes every column in column order, no score read
    for bound in (0, 1, ip.TOPK - 1, ip.TOPK):
        assert ip.score_chunks(bound) == 0 and not written_chunks(bound)
        assert not polled_chunks(bound)
        assert ip.block_chunks(bound, 0)[1] == 0
    assert ip.score_chunks(ip.TOPK + 1) == ip.chunks(ip.TOPK + 1)
    assert polled_chunks(ip.TOPK + 1) == written_chunks(ip.TOPK + 1) != set()


def test_positive_control_fixed_runs_mix_requests():
    def fixed_six(bounds, owners, ctas, width_log2_max):
        own = list(range(len(bounds)))
        counts, _, width_log2 = ip.score_tasks(bounds, own, ctas, width_log2_max)
        return counts, [6] * len(bounds), width_log2

    owners = [1] * 3 + [2] * 6
    with pytest.raises(AssertionError):
        scored_cells([CTAS * ip.CHUNK] * len(owners), owners, 0, fixed_six)


@pytest.mark.parametrize("k", [ip.TOPK, ip.TOPK_BLOCKS])
def test_each_token_runs_exactly_its_tree(k):
    depth_max = ip.levels(max(bounds()), k)
    for bound in bounds():
        top = ip.final_level(bound, depth_max, k)
        for level in range(depth_max + 1):
            expected = max(ip.lists(bound, level, k), 1) if level <= top else 0
            assert ip.list_count(bound, level, k) == expected, (bound, level)


def irdy_publishes(bound, depth, k):
    """The levels at which a token's selection publishes IRDY
    (``index_score``: a list it runs that ``is_final``)."""
    return [
        level
        for level in range(depth + 1)
        if ip.list_count(bound, level, k) > 0 and ip.is_final(bound, level, k)
    ]


@pytest.mark.parametrize("k", [ip.TOPK, ip.TOPK_BLOCKS])
@pytest.mark.parametrize("bound_max", [ip.CHUNK, ip.SEG, 4 * ip.SEG, 1 << 20])
def test_irdy_is_published_once_at_the_final_level(k, bound_max):
    depth = ip.levels(bound_max, k)
    for bound in [b for b in bounds() if b <= bound_max] + [bound_max]:
        assert irdy_publishes(bound, depth, k) == [
            ip.final_level(bound, depth, k)
        ], bound


def test_a_bound_past_the_build_never_publishes_irdy():
    """Positive control: the consumer polls IRDY for every token, so a bound
    beyond ``index_bound_max`` (whose depth the build fixed) would hang it."""
    k, bound_max = ip.TOPK, ip.SEG
    depth = ip.levels(bound_max, k)
    assert irdy_publishes(64 * ip.SEG * ip.fan(k), depth, k) == []


def walked_cells(bounds, owners, tasks):
    """Every (token, chunk) the FP4 score stage writes, a task's chunks for
    each of its tokens as the kernel runs them (``index_score_fp4``)."""
    cells = []
    for lead, span, ch0, ch1 in tasks:
        assert 1 <= span <= ip.WALK_ROWS and ch0 < ch1, (lead, span, ch0, ch1)
        for row in range(lead, lead + span):
            assert owners[row] == owners[lead], (lead, row)
            cells += [
                (row, ch) for ch in range(ch0, ch1) if ip.chunk_written(bounds[row], ch)
            ]
    return cells


def walk_tasks(bounds, owners):
    """``fill_walk_plan``'s tasks: its rows before the first span-0 one, every
    row past them zero."""
    plan = np.full((CTAS, 4), -9, dtype=np.int32)
    ip.fill_walk_plan(
        np.asarray(bounds, dtype=np.int32), np.asarray(owners, dtype=np.int32), plan
    )
    live = int(np.argmin(plan[:, 1] > 0)) if (plan[:, 1] == 0).any() else CTAS
    assert (plan[live:] == 0).all(), plan
    return [tuple(row) for row in plan[:live].tolist()]


@pytest.mark.parametrize("chunks_max", [5, 8, 64, 2048, 8192])
def test_walk_tasks_write_every_chunk_once(chunks_max):
    rng = random.Random(chunks_max)
    steps = [ragged_step(rng, chunks_max) for _ in range(100)]
    # the most pieces a step has: every row its own request
    rows = 48
    steps.append(([ip.TOPK + 1] * rows, list(range(rows))))
    steps.append(([chunks_max * ip.CHUNK] * rows, list(range(rows))))
    for bounds, owners in steps:
        tasks = walk_tasks(bounds, owners)
        expected = sorted(
            (t, ch) for t, b in enumerate(bounds) for ch in written_chunks(b)
        )
        assert sorted(walked_cells(bounds, owners, tasks)) == expected, (bounds, owners)


def test_walk_tasks_take_the_narrowest_width_that_fits():
    # one request of 6 rows at 1024 chunks: 4 chunks a task fills the grid
    # exactly; the old ceil(total / (ctas - pieces)) gave 5 (205 tasks)
    tasks = walk_tasks([1024 * ip.CHUNK] * 6, [7] * 6)
    assert len(tasks) == CTAS and {ch1 - ch0 for *_, ch0, ch1 in tasks} == {4}
    # runs longer than WALK_ROWS split into pieces of at most WALK_ROWS
    tasks = walk_tasks([2 * ip.CHUNK * 64] * 9, [1] * 9)
    assert sorted({(lead, span) for lead, span, *_ in tasks}) == [(0, 6), (6, 3)]
    # nothing to score: no task
    assert walk_tasks([ip.TOPK] * 6, [1] * 6) == []


def test_positive_control_a_wider_piece_mixes_owners():
    owners = [1] * 3 + [2] * 3
    tasks = [(0, 6, 0, 8)]
    with pytest.raises(AssertionError):
        walked_cells([8 * ip.CHUNK] * 6, owners, tasks)


@pytest.mark.parametrize("tokens", [1, 2, 5, 6, 12, 24, 48, 64])
def test_reindex_split_covers_each_tokens_chunks_once(tokens):
    per = ip.reindex_split(tokens, CTAS, 4)
    assert tokens * per <= CTAS or per == 1
    ranges = [ip.reindex_chunks(i, per) for i in range(per)]
    covered = [ch for ch0, ch1 in ranges for ch in range(ch0, ch1)]
    assert covered == list(range(ip.REINDEX_CHUNKS))
    assert min(ch1 - ch0 for ch0, ch1 in ranges) >= 4
