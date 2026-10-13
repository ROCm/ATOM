"""A restore touches only the rows that need one, and is judged after it runs.

Two defects this covers, both measured on a live engine:

  * restoring every row every step overwrote live state with an image
    nothing had written -- a request's second prefill chunk arrived at
    `computed=4096` with `cursor=0`, its first chunk erased;
  * restoring after the cache had staged and judged the batch could not
    help a hit at all, because the contract had already raised.
"""

import numpy as np
import pytest

from atom.model_ops.attentions.deepseek_v41 import cache as v41_cache


class _Req:
    def __init__(self, rid, slot):
        self.request_id = rid
        self.slot = slot


class _Step:
    def __init__(self, reqs):
        self.requests = reqs
        self.slots = None


class _Cache:
    _restore_stale_rows = v41_cache.PagedAttentionCache._restore_stale_rows


@pytest.fixture(autouse=True)
def _no_hook():
    yield
    v41_cache.install_image_restore_hook(None)


def _cursors(*vals):
    return np.asarray([[v, 0] for v in vals], dtype=np.int64)


def test_only_the_stale_rows_are_offered():
    seen = []
    v41_cache.install_image_restore_hook(lambda rows: seen.append(rows) or 0)
    step = _Step([_Req("a", 7), _Req("b", 9), _Req("c", 11)])
    # row 0 is where it should be, row 1 is fresh, row 2 is stale.
    # row 1 is fresh; rows 0 and 2 are resuming, and only row 2 is stale.
    _Cache()._restore_stale_rows(
        _cursors(4096, 0, 0), np.asarray([4096, 0, 2048]), ["a", "b", "c"],
        np.asarray([False, True, False]), step, 3
    )
    assert seen == [[(11, 2048, 2)]]


def test_a_batch_with_nothing_stale_never_calls_the_hook():
    called = []
    v41_cache.install_image_restore_hook(lambda rows: called.append(rows) or 1)
    step = _Step([_Req("a", 7)])
    _Cache()._restore_stale_rows(
        _cursors(4096), np.asarray([4096]), ["a"], np.asarray([False]), step, 1
    )
    assert called == []


def test_a_hook_that_raises_leaves_the_cursors_alone_and_counts():
    def boom(rows):
        raise RuntimeError("no")

    v41_cache.install_image_restore_hook(boom)
    before = v41_cache.restore_failures()
    step = _Step([_Req("a", 7)])
    out = _Cache()._restore_stale_rows(
        _cursors(0), np.asarray([2048]), ["a"], np.asarray([False]), step, 1
    )
    assert out[0, 0] == 0
    assert v41_cache.restore_failures() == before + 1


def test_a_hook_that_restores_nothing_does_not_trigger_a_reread():
    """Re-reading needs a real cache; restoring zero rows must not reach it."""
    v41_cache.install_image_restore_hook(lambda rows: 0)
    step = _Step([_Req("a", 7)])
    out = _Cache()._restore_stale_rows(
        _cursors(0), np.asarray([2048]), ["a"], np.asarray([False]), step, 1
    )
    assert out[0, 0] == 0
