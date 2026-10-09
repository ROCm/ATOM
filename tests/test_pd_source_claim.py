# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The P/D producer's claim on a finished request's source blocks.

`Scheduler` has no producer branch: it parks a finished request while any
connector claims its HBM (`should_defer_free`) and retires the send's claim
through `send_finished`. A claim never taken frees the source under a live
RDMA read; one never dropped parks the blocks forever.

Timing matters as much as existence: `Scheduler._is_preemptable` negates the
same predicate, so claiming at alloc would pin every running request.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from aiter_stub import stubbed_aiter

with stubbed_aiter():
    # `importorskip`, not a plain import: `moriio_connector` reaches
    # `disaggregation.utils`, which imports triton, and a CPU-only runner has
    # none of it. A bare import there is a *collection* error, which takes the
    # whole suite down rather than this module. Every test below is
    # parametrized over both backends, so the module is the right granularity.
    # Same guard as `test_transfer_engine.py` and `test_pd_pp.py`.
    _SKIP = "P/D backend deps (triton, mooncake) are absent on a CPU-only runner"
    MooncakeConnectorScheduler = pytest.importorskip(
        "atom.kv_transfer.disaggregation.mooncake.mooncake_connector",
        reason=_SKIP,
    ).MooncakeConnectorScheduler
    MoRIIOConnectorScheduler = pytest.importorskip(
        "atom.kv_transfer.disaggregation.moriio.moriio_connector",
        reason=_SKIP,
    ).MoRIIOConnectorScheduler


def _build(cls, is_producer, **extra):
    """Only what `update_state_after_alloc` / `request_finished` read -- the
    rest of either scheduler needs a live backend."""
    sched = cls.__new__(cls)
    sched.is_producer = is_producer
    sched._awaiting_send = set()
    for name in ("_reqs_need_save", "_reqs_need_recv", "request_id_to_transfer_id"):
        setattr(sched, name, {})
    sched.transfer_id_to_request_id = {}
    for name, value in dict(
        engine_id="e0",
        host_ip="10.0.0.1",
        handshake_port=6300,
        base_handshake_port=6301,
        tp_size=1,
        dp_rank=0,
        **extra,
    ).items():
        setattr(sched, name, value)
    return sched


def _mooncake(is_producer=True):
    return _build(
        MooncakeConnectorScheduler,
        is_producer,
        block_size=16,
        dcp_size=1,
        hash_block_size=16,
        pp_size=1,
    )


def _moriio(is_producer=True):
    return _build(MoRIIOConnectorScheduler, is_producer)


def _seq(seq_id=7, **params):
    return SimpleNamespace(
        id=seq_id,
        block_table=[1, 2, 3],
        kv_transfer_params=dict(params),
        output_tokens=[42],
        spec_token_ids=None,
        prefix_cache_hit_tokens=0,
    )


BACKENDS = [pytest.param(_mooncake, id="mooncake"), pytest.param(_moriio, id="moriio")]


@pytest.mark.parametrize("build", BACKENDS)
def test_alloc_takes_no_claim_so_a_running_request_stays_preemptable(build):
    sched = build()
    seq = _seq(do_remote_decode=True)

    sched.update_state_after_alloc(seq)

    assert (
        sched.should_defer_free(seq) is False
    ), "claiming here would leave `_preempt_one_running` no candidates"


# The aggregator may report a request id as str or int; the claim keys by str.
@pytest.mark.parametrize("as_str", [False, True])
@pytest.mark.parametrize("build", BACKENDS)
def test_claim_is_taken_at_publication_and_dropped_on_send(build, as_str):
    sched = build()
    seq = _seq(do_remote_decode=True)
    sched.update_state_after_alloc(seq)
    seq.leave_reason = "stop_sequence"

    sched.request_finished(seq)
    assert (
        sched.should_defer_free(seq) is True
    ), "the peer now has these addresses and the read outlives the call"
    assert seq.kv_transfer_params_output["do_remote_prefill"] is True
    assert seq.kv_transfer_params_output["remote_block_ids"] == seq.block_table

    sched.send_finished(str(seq.id) if as_str else seq.id)
    assert sched.should_defer_free(seq) is False


@pytest.mark.parametrize("build", BACKENDS)
def test_aborted_producer_neither_claims_nor_advertises_blocks(build):
    sched = build()
    seq = _seq(do_remote_decode=True)
    sched.update_state_after_alloc(seq)
    seq.leave_reason = "aborted"

    sched.request_finished(seq)

    assert (
        sched.should_defer_free(seq) is False
    ), "an abort never sends, so nothing would retire the claim"
    assert seq.kv_transfer_params_output is None
    assert not sched.build_connector_meta().reqs_to_save


@pytest.mark.parametrize("build", BACKENDS)
def test_nothing_is_claimed_when_nothing_will_be_sent(build):
    """A consumer instance, and (mooncake only) a request the proxy never
    marked `do_remote_decode` -- the gate that fills `_reqs_need_save`.

    The old unconditional producer branch parked both anyway, waiting for a
    send that was never issued. moriio has no such gate: it never reads the
    field and sends everything it prefills.
    """
    consumer = build(is_producer=False)
    seq = _seq()
    consumer.update_state_after_alloc(seq)
    consumer.request_finished(seq)
    assert consumer.should_defer_free(seq) is False

    if build is _mooncake:
        local_only = build()
        seq = _seq()
        seq.leave_reason = "stop_sequence"
        local_only.request_finished(seq)
        assert local_only.should_defer_free(seq) is False
