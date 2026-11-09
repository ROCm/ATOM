# SPDX-License-Identifier: MIT
"""The gfx1250 step kernel's indexer plan (kernels/index_plan.py) on Python ints: the selection tree covers every input
once, the task counts match the tree, the REINDEX key rows stay inside the token's bound, the scorer covers its context."""
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from _gfx1250_plan_backend import Z, load  # noqa: E402

IP = load("index_plan")
KINDS = {"sel": (IP.KSEL, IP.SEG // IP.KSEL), "blk": (IP.KBLK, IP.SEG // IP.KBLK)}
EDGES = [0, 1, 7, 8, 9, IP.TOPK - 1, IP.TOPK, IP.TOPK + 1, IP.SEG - 1, IP.SEG, IP.SEG + 1, 2 * IP.SEG,
         16384, 16385, 26624, 65535, 65536, 131072, 1 << 20]  # fmt: skip


def bounds():
    rng = random.Random(0)
    return EDGES + [rng.randrange(0, 1 << 20) for _ in range(200)]


def outputs(n, lev, j, k, fan):
    """Reference: the values list j at level lev holds (its inputs, kept to k)."""
    if lev == 0:
        return min(max(n - j * IP.SEG, 0), IP.SEG)
    kids = range(j * fan, min((j + 1) * fan, int(IP.lists(Z(n), lev - 1, fan))))
    return sum(min(outputs(n, lev - 1, c, k, fan), k) for c in kids)


@pytest.mark.parametrize("kind", KINDS)
def test_tree_inputs(kind):
    k, fan = KINDS[kind]
    for n in bounds():
        for lev in range(IP.NLMAX + 1):
            nl = int(IP.lists_rt(Z(n), Z(lev), Z(fan)))
            assert nl == int(IP.lists(Z(n), lev, fan))
            for j in {0, max(nl - 1, 0)}:
                if j < nl:
                    assert int(IP.list_inputs_rt(Z(n), Z(lev), Z(j), Z(k), Z(fan))) == outputs(n, lev, j, k, fan), (n, lev, j)


@pytest.mark.parametrize("kind", KINDS)
def test_task_counts(kind):
    k, fan = KINDS[kind]
    for n in bounds():
        counts = [int(IP.list_count_rt(Z(n), Z(lev), Z(fan))) for lev in range(IP.NLMAX + 1)]
        assert counts[0] == max(int(IP.lists(Z(n), 0, fan)), 1)
        for lev in range(1, IP.NLMAX + 1):
            below = int(IP.lists(Z(n), lev - 1, fan))
            assert counts[lev] == (int(IP.lists(Z(n), lev, fan)) if below > 1 else 0)
        if n <= (1 << 20) // (8 if kind == "blk" else 1):   # NLMAX levels reach one list (blk: blocks of a 1M bound)
            assert int(IP.lists(Z(n), IP.NLMAX, fan)) <= 1


def rx(bd, cand):
    kept = IP.rx_kept(Z(bd))
    last = Z(cand[max(int(kept) - 1, 0)]) if int(kept) else Z(0)
    return int(kept), int(IP.rx_context(kept, last, Z(bd)))


def test_reindex_key_rows_past_16384():
    """A REINDEX token keeps <= 2048 candidate blocks (<= 16384 compacted columns) but their real rows run to its bound:
    the key row is clamped to the bound, not to the compacted count (the bug fixed 2026-11-09)."""
    rng = random.Random(1)
    for bd in [100, 16383, 16384, 16385, 18053, 26624, 65536]:
        nblk = -(-bd // 8)
        cand = sorted(rng.sample(range(nblk), min(nblk, IP.NCAND)))
        kept, cx = rx(bd, cand)
        assert kept == min(nblk, IP.NCAND) and cx <= IP.NCAND * 8
        for col in range(0, cx, 7):
            krow = cand[col // 8] * 8 + col % 8
            row = int(IP.key_row(Z(krow), Z(bd)))
            assert row == min(krow, bd - 1) and row < bd
            if krow < bd:
                assert row == krow
        if bd > 16384:   # the old clamp (to cx - 1) would move some real rows
            assert any(cand[c // 8] * 8 + c % 8 > cx - 1 for c in range(cx))


def test_reindex_context():
    """The compacted columns: every kept block's 8 rows but the last block's rows past the bound."""
    for bd in [0, 1, 8, 9, 100, 16384, 20000]:
        nblk = -(-bd // 8)
        cand = list(range(nblk))[-IP.NCAND:]
        kept, cx = rx(bd, cand)
        exp = 0 if not kept else (kept - 1) * 8 + min(max(bd - cand[kept - 1] * 8, 0), 8)
        assert cx == exp


@pytest.mark.parametrize("TR", [6, 12, 18, 24, 30, 36, 42, 48])
def test_score_split(TR):
    sub = IP.score_sub(TR, 256)
    assert 1 <= sub <= IP.SUB and (TR * -(-16 // sub) <= 256 or sub == IP.SUB)
    tcols = sub * IP.CHUNK
    for cx in bounds():
        tasks = int(IP.score_tasks(Z(cx), Z(tcols)))
        if cx <= IP.TOPK:
            assert tasks == 0
        else:
            assert tasks * tcols >= cx > (tasks - 1) * tcols
