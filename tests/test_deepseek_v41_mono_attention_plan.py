# SPDX-License-Identifier: MIT
"""I3 (DESIGN_v3 4.9) for the V4.1 mono attention: for any token's selection
bound and window count, every split the PV stage polls (AM / AL / AP) is one the
score stage writes this step, and the score tasks cover every split. The
functions under test are the ones the kernel calls (``mono/attention_plan.py``)."""

import random

import pytest

from atom.models.deepseek_v41.mono import attention_plan as aplan
from atom.models.deepseek_v41.mono import index_plan as ip
from atom.models.deepseek_v41.mono.config import ATTENTION_KEYS_MAX

# a token's window rows at most: what the selection leaves of the keys
WINDOW_MAX = ATTENTION_KEYS_MAX - ip.TOPK
EDGE_BOUNDS = [0, 1, aplan.BK - 1, aplan.BK, ip.TOPK - 1, ip.TOPK, ip.TOPK + 1, 1 << 20]
EDGE_COUNTS = [0, 1, aplan.BK - 1, aplan.BK, aplan.BK + 1, WINDOW_MAX - 1, WINDOW_MAX]


def tokens():
    """(bound, count) pairs: the edges crossed, then random ones. A graph's pad
    row is (0, 0)."""
    rng = random.Random(0)
    pairs = [(b, c) for b in EDGE_BOUNDS for c in EDGE_COUNTS]
    pairs += [
        (rng.randrange(0, 1 << 21), rng.randrange(0, WINDOW_MAX + 1))
        for _ in range(500)
    ]
    return pairs


def written(n, written_splits=aplan.written_splits):
    return set(range(written_splits(n)))


def polled(n, last_split=aplan.last_split):
    return {aplan.polled_split(i, last_split(n)) for i in range(aplan.SPLITS)}


@pytest.mark.parametrize("per", [1, 2, 4, 8])
def test_score_tasks_cover_every_split_once(per):
    splits = [
        aplan.score_split(sg, per, k, wave)
        for sg in range(per)
        for wave in range(aplan.SCORE_WAVES)
        for k in range(aplan.score_splits_a_wave(per))
    ]
    assert sorted(splits) == list(range(aplan.SPLITS))


def test_every_polled_split_is_written():
    for bound, count in tokens():
        n = aplan.kv_len(ip.selected(bound), count)
        assert aplan.written_splits(n) <= aplan.SPLITS, (bound, count)
        assert polled(n) <= written(n), (bound, count)


def test_without_the_pad_row_split_a_pad_row_polls_an_unwritten_split():
    """Positive control: the score's split count without its floor of one
    leaves a pad row's split 0, which the PV still polls, unwritten."""

    def unfloored(n):
        return ip.cdiv(n, aplan.BK)

    n = aplan.kv_len(ip.selected(0), 0)
    assert not polled(n) <= written(n, written_splits=unfloored)
