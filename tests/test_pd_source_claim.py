# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The P/D producer's claim on a finished request's source blocks.

`Scheduler` has no producer branch: it parks a finished request while any
connector claims its HBM (`should_defer_free`) and retires the send's claim
through `send_finished`. A claim never taken frees the source under a live
RDMA read; one never dropped parks the blocks forever.

Timing matters as much as existence: `Scheduler._is_preemptable` negates the
same predicate, so claiming at alloc would pin every running request.

An abort after the request finished, while decode may still pull its blocks,
must retire the claim too, but never under a running write.
"""

from __future__ import annotations

import contextlib
import queue
import signal
import socket
import threading
from collections import OrderedDict, deque
from types import SimpleNamespace

import msgpack
import pytest
import zmq
from aiter_stub import stubbed_aiter
from conftest import MockConfig

from atom.kv_transfer.disaggregation.multi.multi_connector import (
    MultiConnectorScheduler,
)
from atom.kv_transfer.disaggregation.types import connector_metadata_has_work
from atom.model_engine.scheduler import ScheduledBatchOutput, Scheduler
from atom.model_engine.sequence import Sequence
from atom.sampling_params import SamplingParams

with stubbed_aiter():
    # `importorskip`, not a plain import: `moriio_connector` reaches
    # `disaggregation.utils`, which imports triton, and a CPU-only runner has
    # none of it. A bare import there is a *collection* error, which takes the
    # whole suite down rather than this module. Every test below is
    # parametrized over both backends, so the module is the right granularity.
    # Same guard as `test_transfer_engine.py` and `test_pd_pp.py`.
    _SKIP = "P/D backend deps (triton, mooncake) are absent on a CPU-only runner"
    # Kept as a module: `stubbed_aiter` drops it from `sys.modules`, so a later
    # import would be a second copy its classes never read.
    mc = pytest.importorskip(
        "atom.kv_transfer.disaggregation.mooncake.mooncake_connector",
        reason=_SKIP,
    )
    MooncakeConnectorScheduler = mc.MooncakeConnectorScheduler
    MoRIIOConnectorScheduler = pytest.importorskip(
        "atom.kv_transfer.disaggregation.moriio.moriio_connector",
        reason=_SKIP,
    ).MoRIIOConnectorScheduler
    from atom.model_engine.engine_core import EngineCore


def _build(cls, is_producer, **extra):
    """Only what `update_state_after_alloc` / `request_finished` read -- the
    rest of either scheduler needs a live backend."""
    sched = cls.__new__(cls)
    sched.is_producer = is_producer
    sched._awaiting_send = set()
    sched._pending_chunked_admissions = {}
    sched._chunked_waiting = {}
    sched._chunked_ready = set()
    for name in ("_reqs_need_save", "_reqs_need_recv", "request_id_to_transfer_id"):
        setattr(sched, name, {})
    sched.transfer_id_to_request_id = {}
    sched._abort_sockets = {}
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


def _mooncake(is_producer=True, pp_size=1):
    return _build(
        MooncakeConnectorScheduler,
        is_producer,
        block_size=16,
        dcp_size=1,
        hash_block_size=16,
        pp_size=pp_size,
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


# -- An abort between the prefill's finish and the decode's pull ------------


def _producer_rank():
    """A Mooncake producer rank holding only the state its send path reads."""
    rank = object.__new__(mc.MooncakeConnector)
    prefills_lock = threading.Lock()
    rank.__dict__.update(
        is_producer=True,
        tp_rank=0,
        tp_size=1,
        request_id_to_transfer_id={},
        _completion_lock=threading.Lock(),
        _completed_prefills={},
        _completed_prefills_lock=prefills_lock,
        _completed_prefills_cv=threading.Condition(prefills_lock),
        _kv_cache_ready_events={},
        _kv_cache_ready_batches=deque(),
        _kv_ready_expired={},
        _kv_ready_done_event=None,
        _transfer_refcount={},
        _transfer_refcount_lock=threading.Lock(),
        _writes_in_flight={},
        _aborted_transfers={},
        _chunked_prefills={},
        _chunked_local_ids={},
        _chunked_admissions={},
        _admission_results=set(),
        _retired_chunked_transfers=OrderedDict(),
        _pending_chunked_requests=[],
        _received_handoffs={},
        _pending_recv_stall_watch={},
        _pending_recv_acks={},
        _held_failures={},
        done_sending=set(),
        done_recving=set(),
        failed_recving=set(),
    )
    return rank


class _Workers:
    """The engine's workers: one Mooncake producer rank, which takes the
    metadata the engine dispatches and reports back at each poll, and a
    forward that samples token 7."""

    def __init__(self, rank):
        self.rank = rank
        self.reported: list[set] = []  # each poll's finished_sending

    def call_func(self, name, *args, wait_out=False):
        if name == "process_kvconnector_output":
            return self.rank.start_load_kv(*args)
        if name != "forward":
            return None
        (batch,) = args
        return ScheduledBatchOutput(
            req_ids=list(batch.req_ids),
            token_ids=[(7,)] * len(batch.req_ids),
            num_rejected=None,
            num_bonus=None,
            draft_token_ids=None,
        )

    def call_func_with_aggregation(self, name):
        assert name == "async_proc_aggregation", name
        output = self.rank.get_finished()
        self.reported.append(set(output.finished_sending))
        return output


def _engine(rank):
    """A prefill engine: a real scheduler with Mooncake's scheduler half,
    over `rank`."""
    core = EngineCore.__new__(EngineCore)
    core.label = "prefill"
    core.scheduler = Scheduler(MockConfig())
    core.scheduler.kv_connector = _mooncake()
    core.kv_transfer_enabled = True
    core.runner_mgr = _Workers(rank)
    core.input_queue = queue.Queue()
    core.utility_queue = queue.Queue()
    core.output_queue = queue.Queue()
    core.stream_output_queue = queue.Queue()
    core._kv_progress_event = threading.Event()
    return core


def _step(core):
    """One engine step, its idle drain due."""
    core._next_idle_kv_drain = 0.0
    core._process_engine_step_inner()


def _parked(core):
    """A request the router sends on, prefilled and finished by one engine
    step, and its rank's last."""
    seq = Sequence(
        [1, 2, 3, 4],
        4,
        sampling_params=SamplingParams(max_tokens=1),
        kv_transfer_params={"do_remote_decode": True},
    )
    core.scheduler.add(seq)
    _step(core)
    assert seq.id in core.scheduler.deferred_free_blocks
    assert seq.id in core.runner_mgr.rank._completed_prefills
    return seq


def _write_request(seq):
    return {"request_id": "decode-1", "transfer_id": seq.id}


def _refused(rank, request) -> bool:
    """Whether a send worker answers `request` with a failure and writes nothing."""
    calls = []
    rank._execute_transfer = lambda _request: calls.append("wrote")
    rank._notify_transfer_result = lambda _request, success: calls.append(success)
    rank._serve_write(request)
    return calls == [False]


def test_mooncake_reports_an_abort_before_any_write_request_at_once():
    """On an idle prefill: with nothing else to schedule, the abort still
    reaches the rank, in the metadata of the next step."""
    rank = _producer_rank()
    core = _engine(rank)
    seq = _parked(core)
    sched = core.scheduler

    assert sched.abort_request(seq.id) is True
    _step(core)

    assert sched.deferred_free_blocks == {}
    assert sched.is_finished()
    assert not seq.block_table
    assert _refused(rank, _write_request(seq)), "its blocks are free"


@pytest.mark.parametrize("wrote", [True, False], ids=["written", "failed"])
def test_mooncake_reports_an_abort_mid_write_once_the_write_ends(wrote):
    rank = _producer_rank()
    core = _engine(rank)
    seq = _parked(core)
    sched = core.scheduler
    started, finish = threading.Event(), threading.Event()

    def write(_request):
        started.set()
        finish.wait(5)
        if wrote:  # what a served write does on its way out
            rank.done_sending.add(seq.id)
            rank._completed_prefills.pop(seq.id)

    rank._execute_transfer = write
    worker = threading.Thread(target=rank._serve_write, args=(_write_request(seq),))
    worker.start()
    assert started.wait(5), "a send worker took it before the abort"

    assert sched.abort_request(seq.id) is True
    _step(core)
    assert seq.id in sched.deferred_free_blocks, "freed under a running write"
    assert _refused(rank, dict(_write_request(seq), request_id="decode-2"))

    finish.set()
    worker.join(5)
    _step(core)

    assert sched.deferred_free_blocks == {}
    assert sched.is_finished()
    reports = [seq.id in sent for sent in core.runner_mgr.reported]
    assert reports.count(True) == 1, "once, written or not"


def _chunked_mooncake():
    sched = _mooncake()
    sched.enable_chunked_transfer = True
    return sched


@pytest.mark.parametrize(
    "build, params",
    [
        pytest.param(lambda: _mooncake(pp_size=2), {}, id="mooncake-pp"),
        pytest.param(
            _chunked_mooncake,
            {"chunked_transfer": True, "transfer_id": "x"},
            id="mooncake-chunked",
        ),
        pytest.param(_moriio, {}, id="moriio"),
    ],
)
def test_a_backend_that_cannot_see_every_transfer_keeps_an_aborted_claim(build, params):
    """Under PP prefill a stage sees only its own writes; MoRIIO's producer
    cannot see a consumer's read at all. A chunked transfer's source is
    retired by its own `pd_source_safe` report."""
    sched = build()
    seq = _seq(do_remote_decode=True, **params)
    seq.leave_reason = "length"
    seq.num_prompt_tokens = 3
    if params:
        sched._awaiting_send.add(str(seq.id))  # chunked: claimed at alloc
    sched.request_finished(seq)
    assert sched.should_defer_free(seq) is True

    sched.abort_advertised_send(seq)

    assert sched.should_defer_free(seq) is True
    assert not sched.build_connector_meta().reqs_not_processed


def test_multi_hands_an_abort_to_its_producer_sub_as_work():
    """The engine drops metadata that reports no work, so an abort alone must
    count as some."""
    producer = _mooncake()
    multi = MultiConnectorScheduler.__new__(MultiConnectorScheduler)
    multi._connectors = [producer, SimpleNamespace(build_connector_meta=lambda: None)]
    seq = _seq(do_remote_decode=True)
    seq.leave_reason = "length"
    producer.request_finished(seq)

    multi.abort_advertised_send(seq)
    meta = multi.build_connector_meta()

    assert meta.metas[0].reqs_not_processed == {seq.id}
    assert connector_metadata_has_work(meta)


# -- Decode aborts a request before it asks for its write -------------------
#
# Prefill's leg of the request is over, so no abort reaches it, and decode never
# sent the write request whose completion would have retired prefill's send
# claim. Decode's scheduler tells every producer rank instead.


class _Ended(Exception):
    """The scripted listener socket has handed over every message."""


def _listen(rank, monkeypatch, messages) -> None:
    """Run producer `rank`'s listener over `messages`, as its ROUTER reads them."""
    messages = list(messages)

    def recv_multipart():
        if not messages:
            raise _Ended
        return messages.pop(0)

    sock = SimpleNamespace(
        poll=lambda timeout=None: True, recv_multipart=recv_multipart
    )
    monkeypatch.setattr(
        mc, "zmq_socket_ctx", lambda *_a, **_k: contextlib.nullcontext(sock)
    )
    rank._side_channel_port = 0
    rank._publish_listener_state = lambda exc=None: None
    with pytest.raises(_Ended):
        rank._write_listener()


def _unwritten(blob):
    """A request a consumer finishes before asking for its write."""
    return SimpleNamespace(
        id=5,
        kv_transfer_params={**blob, "do_remote_prefill": True},
        block_table=[],
        output_tokens=[],
        spec_token_ids=None,
        prefix_cache_hit_tokens=0,
    )


def test_a_decode_abort_before_the_write_request_frees_the_producer(monkeypatch):
    rank = _producer_rank()
    prefill = _engine(rank)
    x = _parked(prefill)
    listener = zmq.Context.instance().socket(zmq.ROUTER)
    listener.setsockopt(zmq.LINGER, 0)
    port = listener.bind_to_random_port("tcp://127.0.0.1")
    decode = _engine(None)
    decode.scheduler.kv_connector = consumer = _mooncake(is_producer=False)
    blob = dict(x.kv_transfer_params_output)
    blob.update(remote_host="127.0.0.1", remote_handshake_port=port)
    y = Sequence(
        [1, 2, 3, 4],
        4,
        sampling_params=SamplingParams(max_tokens=8),
        kv_transfer_params=blob,
    )
    decode.scheduler.add(y)

    try:
        assert decode.scheduler.abort_request(y.id) is True
        assert listener.poll(5000), "the producer's listener heard nothing"
        message = listener.recv_multipart()
        consumer.request_finished(y)
        assert not listener.poll(100), "told once"
    finally:
        listener.close()
        for sock in consumer._abort_sockets.values():
            sock.close()
    assert message[1:] == [mc.MSG_ABORT, msgpack.dumps({"transfer_id": x.id})]

    _listen(rank, monkeypatch, [message])
    _step(prefill)

    assert prefill.scheduler.deferred_free_blocks == {}
    assert prefill.scheduler.is_finished()
    assert _refused(rank, _write_request(x)), "its blocks may be another's now"


def test_an_unreadable_abort_does_not_stop_the_write_listener(monkeypatch):
    # Every later write request on the rank needs the listener.
    rank = _producer_rank()
    prefill = _engine(rank)
    x = _parked(prefill)
    notice = msgpack.dumps({"transfer_id": x.id})

    _listen(
        rank,
        monkeypatch,
        [
            [b"decode", mc.MSG_ABORT, b"\xc1"],  # not msgpack
            [b"decode", mc.MSG_ABORT, msgpack.dumps({})],  # no transfer_id
            [b"decode", mc.MSG_ABORT, notice],
        ],
    )
    _step(prefill)

    assert prefill.scheduler.deferred_free_blocks == {}


def test_every_rank_of_every_producer_stage_is_told(monkeypatch):
    """Each producer rank reports its own send, and the producer's KV
    aggregation waits for all of them, so each one is told: at the port its
    listener binds."""
    sent = []

    class Socket:
        def setsockopt(self, *_):
            pass

        def connect(self, addr):
            self.addr = addr

        def send_multipart(self, frames):
            sent.append(self.addr)

    context = SimpleNamespace(socket=lambda _kind: Socket())
    fake = SimpleNamespace(
        Context=SimpleNamespace(instance=lambda: context),
        DEALER=0,
        LINGER=0,
        SNDHWM=0,
    )
    monkeypatch.setattr(mc, "zmq", fake)
    blob = {
        "transfer_id": 9,
        "remote_host": "producer",
        "remote_handshake_port": 7000,
        "tp_size": 2,
        "remote_pp_size": 2,
        "remote_dp_rank": 1,
        "remote_dp_size": 2,
    }

    _mooncake(is_producer=False).request_finished(_unwritten(blob))

    listeners = [
        f"tcp://producer:{7000 + mc._port_offset(1, tp_rank, 2, pp_rank, 2, 2)}"
        for pp_rank in (0, 1)
        for tp_rank in (0, 1)
    ]
    assert sent == listeners


def test_a_chunked_decode_abort_sends_no_notice(monkeypatch):
    """A chunked producer frees an unread source at its own terminal timeout,
    and its writes never pass the check a notice arms."""
    monkeypatch.setattr(mc, "zmq", None)  # any socket use fails the test
    blob = {
        "transfer_id": 9,
        "remote_host": "producer",
        "remote_handshake_port": 7000,
        "tp_size": 1,
        "chunked_transfer": True,
    }
    seq = _unwritten(blob)

    _mooncake(is_producer=False).request_finished(seq)

    assert seq.kv_transfer_params["do_remote_prefill"] is True


def test_notices_to_an_unreachable_producer_rank_never_block_the_step_loop():
    """A producer rank that never answers keeps every notice sent to it queued,
    and decode's step loop, which sends them, goes on. At a DEALER's default
    high-water mark, the send after the thousandth queued one would block."""
    with socket.socket() as unbound:
        unbound.bind(("127.0.0.1", 0))
        port = unbound.getsockname()[1]
    blob = {
        "transfer_id": 9,
        "remote_host": "127.0.0.1",
        "remote_handshake_port": port,
        "tp_size": 1,
    }
    consumer = _mooncake(is_producer=False)

    def blocked(*_):
        raise TimeoutError("a notice blocked the step loop")

    previous = signal.signal(signal.SIGALRM, blocked)
    signal.alarm(20)
    try:
        for _ in range(1001):
            consumer.request_finished(_unwritten(blob))
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
        for sock in consumer._abort_sockets.values():
            sock.close()
