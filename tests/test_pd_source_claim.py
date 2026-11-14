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
    sched._pending_chunked_admissions = {}
    sched._chunked_waiting = {}
    sched._chunked_ready = set()
    sched._producer_admission_undo = {}
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
        state_slot=-1,
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


@pytest.mark.parametrize("previous_claim", [False, True])
def test_mooncake_withdrawn_admission_drops_only_its_new_producer_state(previous_claim):
    sched = _mooncake()
    sched.enable_chunked_transfer = True
    seq = _seq(chunked_transfer=True, do_remote_decode=True, transfer_id="t")
    seq.prompt_token_ids = [1, 2, 3]
    seq.num_prompt_tokens = 3
    older = (seq, [8, 9], -1)
    if previous_claim:
        sched._awaiting_send.add(str(seq.id))
        sched._reqs_need_save[seq.id] = older
    sched._chunked_ready.add(seq.id)
    sched.update_state_after_alloc(seq)
    assert seq.id in sched._reqs_need_save
    sched.cancel_preadmission(seq)
    sched.cancel_preadmission(seq)  # idempotent
    assert seq.id in sched._chunked_ready  # keep the already-ready D reservation
    if previous_claim:
        assert sched._reqs_need_save[seq.id] is older
        assert sched.should_defer_free(seq)
    else:
        assert seq.id not in sched._reqs_need_save
        assert not sched.should_defer_free(seq)


def test_mooncake_preadmission_cancel_cannot_release_a_dispatched_source():
    sched = _mooncake()
    sched.enable_chunked_transfer = True
    seq = _seq(chunked_transfer=True, do_remote_decode=True, transfer_id="t")
    seq.prompt_token_ids = [1, 2, 3]
    seq.num_prompt_tokens = 3
    sched.update_state_after_alloc(seq)
    sched.build_connector_meta()
    sched.cancel_preadmission(seq)
    assert sched.should_defer_free(seq)


def test_nonchunked_mooncake_publishes_only_a_completed_prefill():
    sched = _mooncake()
    seq = _seq(do_remote_decode=True)
    seq.output_tokens = []
    sched.update_state_after_alloc(seq)
    assert not sched.build_connector_meta().reqs_to_save
    # A later chunk changes both its final PAGE allocation and STATE slot.
    seq.block_table = [8, 9, 10, 11]
    seq.state_slot = 3
    sched.update_state_after_alloc(seq)
    assert not sched.build_connector_meta().reqs_to_save
    seq.output_tokens = [42]
    seq.leave_reason = "stop_sequence"
    sched.request_finished(seq)
    publication = sched.build_connector_meta().reqs_to_save[seq.id]
    assert publication.local_block_ids == seq.block_table
    assert publication.local_slot_index == 3
    assert sched.should_defer_free(seq)
    assert not sched.build_connector_meta().reqs_to_save


def test_nonchunked_mooncake_cancel_after_dispatch_never_advertises_source():
    sched = _mooncake()
    seq = _seq(do_remote_decode=True)
    seq.output_tokens = []
    sched.update_state_after_alloc(seq)
    # Store loading dispatches metadata before the P forward. No producer
    # address can survive cancellation if that prefill never completed.
    dispatched = sched.build_connector_meta()
    seq.leave_reason = "aborted"
    sched.request_finished(seq)
    assert not dispatched.reqs_to_save
    assert not sched.build_connector_meta().reqs_to_save
    assert not sched.should_defer_free(seq)
    assert seq.kv_transfer_params_output is None


def test_nonchunked_decode_waits_for_successful_publication(monkeypatch):
    import threading
    from concurrent.futures import ThreadPoolExecutor

    with stubbed_aiter():
        from atom.kv_transfer.disaggregation.mooncake.mooncake_connector import (
            MooncakeConnector,
        )

    sched = _mooncake()
    seq = _seq(do_remote_decode=True)
    seq.output_tokens = []
    seq.state_slot = 5
    sched.update_state_after_alloc(seq)
    worker = MooncakeConnector.__new__(MooncakeConnector)
    worker.is_producer = True
    worker._completed_prefills = {}
    worker._completed_prefills_lock = threading.Lock()
    worker._completed_prefills_cv = threading.Condition(worker._completed_prefills_lock)
    # The input contains no chunked requests; isolate its unrelated dispatcher.
    monkeypatch.setattr(worker, "_dispatch_ready_chunked_requests", lambda: None)
    worker.start_load_kv(sched.build_connector_meta())
    entered_wait = threading.Event()
    wait = worker._completed_prefills_cv.wait

    def observe_wait(timeout=None):
        entered_wait.set()
        return wait(timeout)

    monkeypatch.setattr(worker._completed_prefills_cv, "wait", observe_wait)
    with ThreadPoolExecutor(max_workers=1) as executor:
        reader = executor.submit(worker._wait_for_prefill_data, seq.id)
        assert entered_wait.wait(timeout=2)
        assert not reader.done()
        seq.output_tokens = [42]
        seq.leave_reason = "stop_sequence"
        seq.block_table = [10, 11]
        sched.request_finished(seq)
        # Last request publication must reach D through the real idle engine.
        with stubbed_aiter():
            from atom.model_engine.engine_core import EngineCore
        core = EngineCore.__new__(EngineCore)
        core.kv_transfer_enabled = True
        core.scheduler = SimpleNamespace(kv_connector=sched)
        core.runner_mgr = SimpleNamespace(
            call_func=lambda name, meta: worker.start_load_kv(meta)
        )
        core._dispatch_idle_offload_work()
        assert reader.result(timeout=2) == {
            "block_ids": [10, 11],
            "swa_block_ids": [5],
            "slot_index": 5,
        }
