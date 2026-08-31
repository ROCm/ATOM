# SPDX-License-Identifier: MIT
"""The packed DP all_gather's row layout.

Every per-step scalar a DP group must agree on rides one collective as a row of
one int32 tensor, so the rows are a wire format: pack and unpack are two pieces
of index arithmetic that have to stay in step. Nothing else checks them -- a
shifted row reads a plausible number out of the wrong field and shows up as a
hang on eight ranks, not as a failure here.

The collective itself is faked, so this runs on CPU with no DP group.
"""

import torch

from atom.utils.tbo.ubatching import sync_dp_metadata


class _FakeGroup:
    """Stands in for the ranks. Each entry is one rank's field vector."""

    def __init__(self, peers):
        self.peers = peers


def _sync(monkeypatch, *, peers, **kw):
    """Run `sync_dp_metadata` against `peers` other ranks' field vectors.

    `peers` is a list of callables taking this rank's packed tensor and
    returning that rank's -- built relative to ours so a test says only how a
    peer DIFFERS.
    """

    def fake_all_gather(out_list, local, group=None):
        out_list[0].copy_(local)
        for i, make in enumerate(peers, start=1):
            out_list[i].copy_(make(local))

    monkeypatch.setattr(torch.distributed, "all_gather", fake_all_gather)
    return sync_dp_metadata(
        dp_group=_FakeGroup(peers),
        dp_size=1 + len(peers),
        **kw,
    )


def _base(**kw):
    base = {
        "scheduled_tokens": 10,
        "scheduled_bs": 2,
        "is_prefill": False,
        "tbo_on": False,
    }
    return {**base, **kw}


def _with(idx, value):
    """A peer identical to us except in row `idx`."""

    def make(local):
        peer = local.clone()
        peer[idx] = value
        return peer

    return make


def test_the_batch_is_reduced_by_max_and_not_by_something_that_looks_like_it(
    monkeypatch,
):
    """A rank holding 2 sequences alongside one holding 7 must leave with 7.

    Chosen so no other reduction lands on the same answer: sum is 9, min is 2,
    ours is 2. Only MAX gives 7, so a row read out of the wrong field or
    reduced the wrong way cannot pass.
    """
    r = _sync(monkeypatch, peers=[_with(1, 7)], **_base(scheduled_bs=2))
    assert r.max_bs_across_dp == 7


def test_the_batch_and_the_token_count_do_not_read_each_other(monkeypatch):
    """They are the same step in two units and sit in adjacent rows.

    Given distinct values on both, a one-row shift would swap them and every
    downstream shape would be wrong by a factor of the per-seq token count --
    which on a decode step is a plausible-looking number.
    """
    r = _sync(
        monkeypatch,
        peers=[_with(0, 99)],
        **_base(scheduled_tokens=10, scheduled_bs=3),
    )
    assert int(r.num_tokens_across_dp.max()) == 99
    assert r.max_bs_across_dp == 3


def test_the_prefill_flag_still_reads_as_a_flag_after_the_batch_row(monkeypatch):
    """`is_prefill` moved down one row when the batch was folded in.

    It OR-reduces, so a row shift onto the batch would make it True whenever
    any rank had a non-empty batch -- which is every step, and it silently
    disables the uniform-decode path for the whole group.
    """
    assert not _sync(
        monkeypatch, peers=[], **_base(is_prefill=False)
    ).any_rank_has_prefill
    assert _sync(monkeypatch, peers=[], **_base(is_prefill=True)).any_rank_has_prefill
    # A peer prefilling turns it on for everyone; our own batch does not.
    r = _sync(
        monkeypatch, peers=[_with(2, 1)], **_base(is_prefill=False, scheduled_bs=5)
    )
    assert r.any_rank_has_prefill and r.max_bs_across_dp == 5


def test_the_tbo_rows_survive_the_batch_being_folded_in(monkeypatch):
    """TBO's four fields shifted down with everything else.

    `can_split` AND-reduces and `meets_min_tokens` OR-reduces, so reading each
    other's row inverts the gate. Pinned with the two disagreeing.
    """
    r = _sync(
        monkeypatch,
        peers=[],
        **_base(
            tbo_on=True,
            local_meets_min_tokens=True,
            local_can_split=True,
            local_ub_tokens=(4, 6),
        ),
    )
    assert r.tbo_collective_active
    assert r.ub_max_tokens_across_dp == (4, 6)

    # One rank that cannot split vetoes the whole group.
    vetoed = _sync(
        monkeypatch,
        peers=[_with(4, 0)],
        **_base(
            tbo_on=True,
            local_meets_min_tokens=True,
            local_can_split=True,
            local_ub_tokens=(4, 6),
        ),
    )
    assert not vetoed.tbo_collective_active


def test_the_dspark_block_starts_after_the_tbo_block_in_both_widths(monkeypatch):
    """DSpark's rows are placed at an offset computed from `tbo_on`.

    Both widths are exercised because the offset is the thing that moved, and
    with TBO off it is the only arithmetic between the fixed head and DSpark.
    The length is read off a PEER carrying a value no other row holds, so a
    shifted offset reads one of the head's fields and reports it.
    """
    for tbo_on in (False, True):
        head = 7 if tbo_on else 3
        r = _sync(
            monkeypatch,
            peers=[_with(head, 9)],
            **_base(tbo_on=tbo_on, max_seqlen_q=3),
        )
        assert r.max_seqlen_q_across_dp == 9, f"tbo_on={tbo_on}"
        alone = _sync(monkeypatch, peers=[], **_base(tbo_on=tbo_on, max_seqlen_q=3))
        assert alone.max_seqlen_q_across_dp == 3, f"tbo_on={tbo_on}"
