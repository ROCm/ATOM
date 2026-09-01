# SPDX-License-Identifier: MIT
"""`ForwardMode.decide`: the one place a step's shape is settled.

Single-rank steps take the early return and issue no collective, so most of
this runs with no DP group at all. The cases that need one fake the packed
all_gather -- see `tests/test_dp_sync_layout.py` for the wire format itself.

A shape that came out per-rank is how a DP group ends up with two collective
widths, and that failure shows up as a hang rather than as a wrong number.
"""

import types

import numpy as np
import pytest
import torch

from atom.utils.forward_context import ForwardMode

# What `ModelRunner.capture_sizes_np` is: ascending int32, sorted where it is
# built. `decide` searches it, so a test that passed a plain list would be
# exercising a type production never hands it.
LADDER = np.asarray([1, 2, 4, 8, 16, 32, 48, 64], dtype=np.int32)


def _batch(*, seqs, tokens=None, prefill_tokens=0, q=1):
    """The fields `decide` reads. Decode unless `prefill_tokens` is given.

    A decode-only batch has `total_tokens_num == total_tokens_num_decode`, as a
    real one does -- the two diverging is what a mixed batch means, and no test
    here builds one.
    """
    total = seqs * q if tokens is None else tokens
    return types.SimpleNamespace(
        total_seqs_num=seqs,
        total_tokens_num=total,
        total_tokens_num_prefill=prefill_tokens,
        total_tokens_num_decode=0 if prefill_tokens else total,
        # Not a real ScheduledBatch field any more -- kept so `_decide` can pass
        # this stub's per-seq length the way `prepare_model` passes the shrink's.
        max_seqlen_q=q,
        is_dummy_run=False,
    )


def _decide(batch=None, **kw):
    base = {
        "batch": batch if batch is not None else _batch(seqs=8, q=1),
        "dp_size": 1,
        "dp_group": None,
        "enforce_eager": False,
        "capture_sizes": LADDER,
        "captured_tokens": None,
        "is_block_drafter": False,
        "tbo_on": False,
        "local_tbo": (False, False, 0, 0),
    }
    merged = {**base, **kw}
    # The batch no longer carries it -- `decide` is handed the step's per-seq
    # length, so the stub's `q` travels the same way production's does.
    merged.setdefault("max_seqlen_q", merged["batch"].max_seqlen_q)
    return ForwardMode.decide(**merged)


def _fake_sync(monkeypatch, *, peer_tokens, peer_bs, peer_prefill=False):
    """One peer rank differing from ours, without a real DP group."""
    import atom.utils.tbo.ubatching as ub

    def fake_all_gather(out_list, local, group=None):
        out_list[0].copy_(local)
        peer = local.clone()
        peer[0], peer[1] = peer_tokens, peer_bs
        peer[2] = 1 if peer_prefill else 0
        out_list[1].copy_(peer)

    monkeypatch.setattr(torch.distributed, "all_gather", fake_all_gather)
    return ub


# --------------------------------------------------------------------------- #
# running_bs -- what everything per-sequence runs at.                          #
# --------------------------------------------------------------------------- #


def test_the_batch_is_the_reduction_on_the_ladder_on_every_kind_of_step():
    """One rule, and the kind of step is not part of it.

    Prefill, eager decode and a replayed decode all leave with the same batch:
    the group's, rounded on a ladder every rank shares. That is what lets a
    graph key, a draft pass and an attention plan read one field without
    knowing which kind of step produced it -- when this branched on dispatch,
    two of the four arms handed back a per-rank count and every consumer had
    to know which it was holding.
    """
    prefill = _decide(_batch(seqs=3, tokens=900, prefill_tokens=900))
    assert prefill.is_prefill and not prefill.use_cudagraph
    assert prefill.running_bs == 4, "rounded up, exactly like a decode step"

    decode = _decide(_batch(seqs=3, q=6))
    assert decode.use_cudagraph and decode.running_bs == 4
    assert prefill.running_bs == decode.running_bs


def test_a_batch_the_ladder_cannot_hold_keeps_the_reduction_and_replays_nothing():
    """Running off the end of the ladder is the same fact twice: there is no
    recording that fits, so the batch stays what the reduction said and the
    step cannot replay. One lookup has to answer both, or a step dispatches
    PIECEWISE at a width nothing recorded."""
    mode = _decide(_batch(seqs=100, q=1))
    assert mode.running_bs == 100 and not mode.use_cudagraph


def test_a_decode_step_that_fits_the_ladder_replays():
    mode = _decide(_batch(seqs=50, q=6))
    assert mode.use_cudagraph and mode.running_bs == 64


def test_forcing_eager_changes_the_dispatch_and_not_the_batch():
    """`enforce_eager` answers "does the target replay", which is a different
    question from "how wide is the batch". Under it the runner never runs
    `capture_cudagraph`, so the ladder stays at its `[0]` placeholder and the
    reduction stands unrounded on its own -- by having nothing to round to,
    not by a branch."""
    warmed = _decide(_batch(seqs=5, q=1), enforce_eager=True)
    assert not warmed.use_cudagraph and warmed.running_bs == 8

    unwarmed = _decide(
        _batch(seqs=5, q=1),
        enforce_eager=True,
        capture_sizes=np.asarray([0], dtype=np.int32),
    )
    assert not unwarmed.use_cudagraph and unwarmed.running_bs == 5


def test_the_batch_ignores_what_this_rank_alone_was_handed(monkeypatch):
    """The DP property, stated where it is decided.

    Two ranks arrive at one step with different local batches. They must leave
    with the same `running_bs`, because that is the width a recording holds its
    collective at. Asserting the equality rather than the value is what makes a
    future term that reads the local count fail here.
    """
    _fake_sync(monkeypatch, peer_tokens=40, peer_bs=40)
    small = _decide(_batch(seqs=1, q=1), dp_size=2, dp_group=object())
    _fake_sync(monkeypatch, peer_tokens=1, peer_bs=1)
    large = _decide(_batch(seqs=40, q=1), dp_size=2, dp_group=object())
    assert small.running_bs == large.running_bs == 48
    assert small.use_cudagraph == large.use_cudagraph


# --------------------------------------------------------------------------- #
# The rest of what one step settles.                                           #
# --------------------------------------------------------------------------- #


def test_tbo_survives_having_no_peer_to_ask():
    """One rank still gets an answer, and it is the local one.

    The gate used to be read off the DP sync result, which a single rank never
    produces -- so `--enable-tbo` was a dead switch on every TP-only deployment,
    silently, and PCP+TBO (which is always dp==1) could never fire at all.
    """
    on = _decide(tbo_on=True, local_tbo=(True, True, 4, 4))
    assert on.tbo_collective_active

    # Both bits are required: reaching the min-token bar is not enough if the
    # batch cannot be split, and vice versa.
    for local in ((True, False, 4, 4), (False, True, 4, 4)):
        assert not _decide(tbo_on=True, local_tbo=local).tbo_collective_active
    assert not _decide(tbo_on=False, local_tbo=(True, True, 4, 4)).tbo_collective_active


def test_a_peer_that_cannot_split_vetoes_tbo_for_the_group(monkeypatch):
    """With peers the answer is theirs, not ours -- the reduction decides."""
    _fake_sync(monkeypatch, peer_tokens=8, peer_bs=8)
    mode = _decide(
        _batch(seqs=8, q=1),
        dp_size=2,
        dp_group=object(),
        tbo_on=True,
        local_tbo=(True, True, 4, 4),
    )
    assert mode.tbo_collective_active == mode.sync.tbo_collective_active


# --------------------------------------------------------------------------- #
# running_tokens -- what everything per-row runs at. NOT the batch.            #
# --------------------------------------------------------------------------- #


def test_the_decode_height_exceeds_what_any_rank_scheduled():
    """Why the reduced token count cannot serve on its own.

    The batch is rounded onto the ladder AFTER the sync, so the graph runs rows
    nobody reported: 50 seqs of 6 is 300 tokens, but the tensor has 64*6 = 384
    rows. Padding to 300 leaves MoE 84 rows short.
    """
    mode = _decide(_batch(seqs=50, q=6))
    assert mode.running_bs == 64 and mode.running_tokens == 384


def test_a_prefill_step_runs_its_own_rows(monkeypatch):
    """Prompts are ragged: no `bs * q` recovers a height, and no group max
    stands in for one either.

    `running_bs` here is 8, which would give a plausible-looking 8*6=48 if the
    decode rule leaked in, and the peer reports 700, which would win if the
    group max did. The answer is this rank's own 100 -- a prefilling rank is
    exactly the case where the ranks are NOT level, so each runs what it has.
    """
    _fake_sync(monkeypatch, peer_tokens=700, peer_bs=2, peer_prefill=True)
    mode = _decide(
        _batch(seqs=8, tokens=100, prefill_tokens=100, q=6),
        dp_size=2,
        dp_group=object(),
    )
    assert mode.is_prefill and not mode.running_tokens_are_unified
    assert mode.running_tokens == 100


def test_the_variable_length_path_runs_this_rank_alone_in_both_units(monkeypatch):
    """A prefilling peer puts every rank on the variable-length gather, and
    that gather's contract is that nothing was padded -- so NEITHER unit may
    come from the group. Unifying the batch here while leaving the token count
    local is what cost 0.9530 -> 0.9447: every published width stayed
    self-consistent, just consistent with a batch this rank never scheduled.

    Reachable only with a mixed step, which no single-DP run can show."""
    _fake_sync(monkeypatch, peer_tokens=9999, peer_bs=48, peer_prefill=True)
    mode = _decide(_batch(seqs=3, q=6), dp_size=2, dp_group=object())
    assert not mode.running_tokens_are_unified
    assert mode.running_tokens == 18 and mode.scheduled_tokens == 18
    # ...while the BATCH is still the group's. The two units answer to
    # different consumers: an over-wide per-sequence array costs a few sentinel
    # rows, an over-wide height is what the gather asserts against.
    assert mode.running_bs == 48 and mode.scheduled_bs == 3


def test_a_prefilling_peer_makes_the_whole_group_ragged(monkeypatch):
    """The one meaning `running_tokens_are_unified` now has.

    It used to also be True whenever dp-attention was off -- one flag standing
    for two questions, which let a decoding rank answer "we are all level" while
    a peer prefilled. Everything downstream that reads it (MoE's fixed-size
    gather, the DP-sharded LM head) needs the strong claim, so the flag makes
    only the strong claim and a mixed step drops every rank to eager.
    """
    _fake_sync(monkeypatch, peer_tokens=5, peer_bs=5, peer_prefill=True)
    mixed = _decide(_batch(seqs=8, q=1), dp_size=2, dp_group=object())
    _fake_sync(monkeypatch, peer_tokens=5, peer_bs=5, peer_prefill=False)
    level = _decide(_batch(seqs=8, q=1), dp_size=2, dp_group=object())

    assert not mixed.running_tokens_are_unified and not mixed.use_cudagraph
    assert level.running_tokens_are_unified and level.use_cudagraph


def test_a_flat_step_takes_the_bucket_and_not_a_product():
    """A block drafter's rows are a packed run, so they are a bucket rather than
    `running_bs * q`. Chosen here and read back everywhere -- two searches is how
    the height and the replay came to disagree."""
    mode = _decide(
        _batch(seqs=10, q=6), captured_tokens=[48, 96, 192], is_block_drafter=True
    )
    assert mode.running_bs == 16, "the batch still rounds onto the seq ladder"
    assert mode.running_tokens == 96, "smallest q-divisible bucket >= 60"
    assert mode.piecewise_captured


def test_a_rectangular_layout_never_takes_a_bucket_below_its_own_rectangle():
    """Only the flat layout may shrink below `running_bs * q`.

    Every per-token buffer of a rectangular backend spans one row per
    (seq, query) and `slot_mapping[i * q + j]` addresses through it, so a bucket
    under that leaves the tail of the rectangle unwritten. Same batch as above,
    same buckets -- only the layout differs.
    """
    mode = _decide(
        _batch(seqs=10, q=6), captured_tokens=[48, 96, 192], is_block_drafter=False
    )
    assert mode.running_tokens == 16 * 6


def test_no_fitting_bucket_declines_the_graph_rather_than_claiming_one():
    """The miss must not be reported as captured.

    `running_bs * q` is the fallback precisely because no bucket covered the
    step, so it is not in the captured set either -- and dispatching PIECEWISE
    at a width nothing recorded makes the wrapper capture one mid-serve.
    """
    mode = _decide(
        _batch(seqs=10, q=6), captured_tokens=[12, 24], is_block_drafter=True
    )
    assert mode.running_tokens == 16 * 6
    assert not mode.piecewise_captured


def test_the_rectangle_replays_only_when_it_was_captured():
    fits = _decide(
        _batch(seqs=8, q=2), captured_tokens=[16, 32], is_block_drafter=False
    )
    assert fits.running_tokens == 16 and fits.piecewise_captured
    misses = _decide(_batch(seqs=8, q=2), captured_tokens=[24], is_block_drafter=False)
    assert misses.running_tokens == 16 and not misses.piecewise_captured


# --------------------------------------------------------------------------- #
# Which recorded width a packed run lands on. Asked through `decide`, there    #
# being no second entry point to ask it through.                               #
# --------------------------------------------------------------------------- #


def _packed(*, scheduled_tokens, q, captured):
    """A block-drafter step whose packed run is `scheduled_tokens` long, over
    `q`-wide sequences. `seqs=1` keeps the rectangle out of the answer's way."""
    return _decide(
        _batch(seqs=1, q=q, tokens=scheduled_tokens),
        captured_tokens=captured,
        is_block_drafter=True,
    )


def test_nothing_recorded_means_no_packed_width_to_land_on():
    """Under FULL (non-PIECEWISE) cudagraphs nothing flat is recorded, so the
    run has nowhere to land and the step stays rectangular."""
    for captured in ([], None):
        mode = _packed(scheduled_tokens=3, q=6, captured=captured)
        assert mode.running_tokens == 1 * 6 and not mode.piecewise_captured


def test_the_width_covers_the_run_and_tiles_it():
    captured = [6, 12, 24]
    assert _packed(scheduled_tokens=3, q=6, captured=captured).running_tokens == 6
    assert _packed(scheduled_tokens=6, q=6, captured=captured).running_tokens == 6
    assert (
        _packed(scheduled_tokens=7, q=6, captured=captured).running_tokens == 12
    ), "smallest that holds it"

    # Longer than anything recorded -> forwarded eagerly at its own length.
    over = _packed(scheduled_tokens=25, q=6, captured=captured)
    assert over.running_tokens == 25 and not over.piecewise_captured

    # Covers the run but does not tile the per-seq rows at stride q.
    untiled = _packed(scheduled_tokens=3, q=6, captured=[10])
    assert untiled.running_tokens == 1 * 6 and not untiled.piecewise_captured


# --------------------------------------------------------------------------- #
# assert_shape_contract -- the padded step, which is the one that can be wrong. #
# --------------------------------------------------------------------------- #


def _md(*, mode, slot_mapping_rows=None, cu_rows=None, slot_out_rows=None):
    """What a builder publishes, defaulting to what `mode` says it should.

    Defaulting rather than requiring each width keeps a case to the ONE it is
    about; a test that had to restate all three would pass while contradicting
    itself, which is how the pair this replaces came to be built on a
    combination no builder emits.
    """

    def _z(n):
        return torch.zeros(n, dtype=torch.int32)

    return types.SimpleNamespace(
        slot_mapping=_z(
            mode.running_tokens if slot_mapping_rows is None else slot_mapping_rows
        ),
        cu_seqlens_q=_z(mode.running_bs + 1 if cu_rows is None else cu_rows),
        state_slot_out=_z(mode.running_bs if slot_out_rows is None else slot_out_rows),
        max_seqlen_q=mode.max_seqlen_q,
    )


def _ids(mode):
    return torch.zeros(mode.scheduled_tokens, dtype=torch.int32)


def test_a_padded_step_whose_widths_all_agree_passes():
    """50 sequences of 6 run at 64: per-token structures span the rectangle,
    per-sequence ones the batch. The check has to admit this -- it is what
    every cudagraph decode step looks like."""
    mode = _decide(_batch(seqs=50, q=6))
    assert mode.running_bs == 64 and mode.running_tokens == 384
    mode.assert_shape_contract(_ids(mode), _md(mode=mode))


def test_per_token_structures_left_at_the_scheduled_rows_are_caught():
    """The defect this check exists for: the batch was widened and the token
    dimension was not. Publishing `slot_mapping` at `scheduled_tokens` on a
    step running `running_tokens` rows leaves the tail of the rectangle
    addressing memory the builder never wrote."""
    mode = _decide(_batch(seqs=50, q=6))
    with pytest.raises(AssertionError, match="slot_mapping"):
        mode.assert_shape_contract(
            _ids(mode), _md(mode=mode, slot_mapping_rows=mode.scheduled_tokens)
        )


def test_per_sequence_structures_left_at_the_scheduled_batch_are_caught():
    """The mirror image, and the one that stays silent at runtime: a
    `cu_seqlens_q` published at `scheduled_bs` while the step runs
    `running_bs` hands attention a boundary list short of its own batch."""
    mode = _decide(_batch(seqs=50, q=6))
    with pytest.raises(AssertionError, match="cu_seqlens_q"):
        mode.assert_shape_contract(
            _ids(mode), _md(mode=mode, cu_rows=mode.scheduled_bs + 1)
        )
    with pytest.raises(AssertionError, match="state_slot_out"):
        mode.assert_shape_contract(
            _ids(mode), _md(mode=mode, slot_out_rows=mode.scheduled_bs)
        )


def test_a_ragged_group_is_checked_on_widths_but_not_on_the_rectangle(monkeypatch):
    """A prefilling peer drops every rank to its own rows, so `scheduled_tokens`
    need not tile at `max_seqlen_q` and no rectangle bounds it. The width
    equalities still hold, and still have to be checked -- this is the step
    where the batch and the token count are reduced differently."""
    _fake_sync(monkeypatch, peer_tokens=9999, peer_bs=48, peer_prefill=True)
    mode = _decide(_batch(seqs=3, q=6), dp_size=2, dp_group=object())
    assert not mode.running_tokens_are_unified
    mode.assert_shape_contract(_ids(mode), _md(mode=mode))
    with pytest.raises(AssertionError, match="cu_seqlens_q"):
        mode.assert_shape_contract(_ids(mode), _md(mode=mode, cu_rows=2))


def test_a_backend_that_publishes_none_is_not_invented_a_width_for():
    """V4 carries no `slot_mapping` on decode; absent is not zero-length."""
    mode = _decide(_batch(seqs=50, q=6))
    md = _md(mode=mode)
    md.slot_mapping = None
    md.state_slot_out = None
    mode.assert_shape_contract(_ids(mode), md)
