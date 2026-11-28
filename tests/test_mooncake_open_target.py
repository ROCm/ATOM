# SPDX-License-Identifier: MIT
"""A Mooncake producer opens each consumer's segment once, when the consumer is
first named, and never inside a transfer.

Mooncake opens a target on first use: a P2P handshake to the consumer that
fetches its buffer list. The open is driven here through the producer's real
write listener, fed by a scripted socket, and a stub transfer engine records
every open.
"""

import threading
from contextlib import nullcontext
from types import SimpleNamespace

import msgpack
import pytest
import zmq
from aiter_stub import stubbed_aiter

with stubbed_aiter():
    from atom.kv_transfer.disaggregation.mooncake import mooncake_connector as mc

CONSUMER = {
    "consumer_host": "10.0.0.2",
    "consumer_rpc_port": 42000,
    "consumer_ib_device": "rdma0",
}
TARGET = "10.0.0.2:42000"


class StubEngine:
    """Records opens (`get_first_buffer_address`) and writes, in order."""

    def __init__(self, first_buffer=0x1000):
        self.first_buffer = first_buffer
        self.calls = []

    def get_first_buffer_address(self, target):
        self.calls.append(("open", target))
        return self.first_buffer

    def batch_transfer_sync_write(self, target, sources, destinations, sizes):
        self.calls.append(("write", target))
        return 0


class StopListener(Exception):
    pass


class ScriptedSocket:
    """Hands the listener `frames` one at a time, then stops it."""

    def __init__(self, frames):
        self.frames = list(frames)
        self.sent = []

    def poll(self, timeout):
        if not self.frames:
            raise StopListener
        return zmq.POLLIN

    def recv_multipart(self):
        return self.frames.pop(0)

    def send_multipart(self, parts):
        self.sent.append(parts)


def make_producer(engine=None):
    p = mc.MooncakeConnector.__new__(mc.MooncakeConnector)
    p.is_producer = True
    p.pp_size, p.pp_rank = 1, 0
    p.tp_size = p.dcp_size = 1
    p._side_channel_port = 6301
    p._publish_listener_state = lambda exc=None: None
    p._dispatch_ready_chunked_requests = lambda: None
    p._encoder = SimpleNamespace(encode=lambda meta: b"producer-meta")
    p._local_metadata = None
    p._rail_pool = None
    p.transfer_engine = engine or StubEngine()
    p._opened_targets = set()
    p.opens = []
    p._target_opener = SimpleNamespace(submit=lambda fn, *a: p.opens.append((fn, a)))
    p.submitted = []
    p._send_executor = SimpleNamespace(
        submit=lambda fn, *a: p.submitted.append((fn, a))
    )
    return p


def run_opens(p):
    """Run what the listener handed `_target_opener`, as its thread would."""
    opens, p.opens = p.opens, []
    for fn, args in opens:
        fn(*args)


def listen(p, frames, monkeypatch):
    sock = ScriptedSocket(frames)
    monkeypatch.setattr(mc, "zmq_socket_ctx", lambda *a, **kw: nullcontext(sock))
    with pytest.raises(StopListener):
        p._write_listener()
    return sock


def write_request(**overrides):
    request = {
        **CONSUMER,
        "request_id": "decode-1",
        "transfer_id": "prefill-1",
        "dst_block_ids": [9],
        "notify_host": "10.0.0.2",
        "notify_port": 42001,
        "consumer_tp_size": 1,
        "write_nonce": 1,
    }
    request.update(overrides)
    return request


def get_meta(consumer=CONSUMER):
    return [b"decode", mc.MSG_GET_META, msgpack.dumps(consumer)]


def write(request):
    return [b"decode", mc.MSG_WRITE_REQUEST, msgpack.dumps(request)]


def test_the_handshake_opens_the_consumer_before_its_first_transfer(monkeypatch):
    p = make_producer()
    engine = p.transfer_engine

    sock = listen(p, [get_meta()], monkeypatch)
    # Answered as before, and the open handed to its own thread.
    assert sock.sent == [[b"decode", b"", b"producer-meta"]]
    assert engine.calls == []
    run_opens(p)
    assert engine.calls == [("open", TARGET)]

    listen(p, [write(write_request())], monkeypatch)
    assert p.opens == []
    ((fn, (request,)),) = p.submitted
    assert fn == p._execute_transfer

    # The first transfer itself opens nothing.
    p._wait_for_prefill_data = lambda transfer_id: {"block_ids": [3]}
    p._get_kv_cache_ready_event = lambda transfer_id: None
    p._notify_transfer_result = lambda *a, **kw: None
    p._transfer_refcount_lock = threading.Lock()
    p._transfer_refcount = {}
    p._completion_lock = threading.Lock()
    p.done_sending = set()
    p._drop_completed_prefill = lambda transfer_id: None

    def block_transfer(request, target, src, dst, req_id, event, *, engine):
        engine.batch_transfer_sync_write(target, [], [], [])
        return True

    p._execute_block_transfer = block_transfer
    fn(request)
    assert engine.calls == [("open", TARGET), ("write", TARGET)]
    assert p.done_sending == {"prefill-1"}


def test_a_consumer_without_a_handshake_is_opened_at_its_first_write_request(
    monkeypatch,
):
    p = make_producer()
    listen(
        p,
        [
            [b"decode", mc.MSG_GET_META],  # a consumer that predates the frame
            write(write_request()),
            write(write_request(request_id="decode-2", transfer_id="prefill-2")),
        ],
        monkeypatch,
    )
    run_opens(p)
    assert p.transfer_engine.calls == [("open", TARGET)]
    assert len(p.submitted) == 2


def test_each_target_and_rail_is_opened_once(monkeypatch):
    rails = {"rdma0": StubEngine(), "rdma1": StubEngine()}
    p = make_producer()
    p._rail_pool = SimpleNamespace(get=rails.__getitem__)
    other = {**CONSUMER, "consumer_rpc_port": 42002}
    listen(
        p,
        [
            get_meta(),
            get_meta(),
            write(write_request()),
            get_meta({**CONSUMER, "consumer_ib_device": "rdma1"}),
            get_meta(other),
        ],
        monkeypatch,
    )
    run_opens(p)
    assert rails["rdma0"].calls == [("open", TARGET), ("open", "10.0.0.2:42002")]
    assert rails["rdma1"].calls == [("open", TARGET)]


@pytest.mark.parametrize("fails", ["no_buffers", "raises"])
def test_a_failed_open_is_retried_when_the_consumer_is_next_named(monkeypatch, fails):
    engine = StubEngine(first_buffer=0)
    if fails == "raises":

        def refuse(target):
            engine.calls.append(("open", target))
            raise RuntimeError("segment not found")

        engine.get_first_buffer_address = refuse
    p = make_producer(engine)
    listen(p, [get_meta()], monkeypatch)
    run_opens(p)
    listen(p, [write(write_request())], monkeypatch)
    run_opens(p)
    assert engine.calls == [("open", TARGET), ("open", TARGET)]


def test_a_bad_consumer_frame_still_gets_its_metadata(monkeypatch):
    p = make_producer()
    sock = listen(
        p,
        [[b"decode", mc.MSG_GET_META, b"\xc1"], get_meta({"consumer_host": "x"})],
        monkeypatch,
    )
    assert len(sock.sent) == 2
    assert p.opens == []


def test_the_consumer_names_itself_in_its_handshake(monkeypatch):
    monkeypatch.setattr(mc, "PD_META_TIMEOUT_MS", 50)
    ctx = zmq.Context()
    router = ctx.socket(zmq.ROUTER)
    port = router.bind_to_random_port("tcp://127.0.0.1")
    try:
        c = mc.MooncakeConnector.__new__(mc.MooncakeConnector)
        c._has_slot_regions = True
        c.zmq_context = ctx
        c.local_ip, c.rpc_port, c.ib_device = "10.0.0.2", 42000, "rdma0"
        # No reply: the consumer leaves the slot check to the producer.
        c._verify_producer(f"tcp://127.0.0.1:{port}")
        assert router.poll(1000)
        _identity, msg_type, frame = router.recv_multipart()
    finally:
        router.close(linger=0)
        ctx.term()
    assert msg_type == mc.MSG_GET_META
    assert msgpack.loads(frame) == CONSUMER
