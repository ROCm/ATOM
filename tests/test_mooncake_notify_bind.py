# SPDX-License-Identifier: MIT
"""The consumer's write-done listener must hold its port before publishing it.

`get_open_port()` reports a port that was free a moment ago. When the listener
bound it later, another process could take it first; the listener thread died
on EADDRINUSE and that decode rank never finished a KV receive again.
"""

import contextlib
import threading
from types import MethodType, SimpleNamespace

import msgpack
import pytest
import zmq


class _Stop(BaseException):
    """Ends the listener loop once the test has seen what it needs."""


def _consumer(mc, record_write_done):
    conn = SimpleNamespace(
        is_producer=False,
        _notification_port=None,
        _listener_bound=threading.Event(),
        _listener_error=None,
        _record_write_done=record_write_done,
    )
    conn._publish_listener_state = MethodType(
        mc.MooncakeConnector._publish_listener_state, conn
    )
    return conn


def _start_listener(mc, conn, ends_with):
    def run():
        # The listener reports its own death through _publish_listener_state;
        # swallow only the exception it is expected to end with.
        with contextlib.suppress(ends_with):
            mc.MooncakeConnector._notification_listener(conn)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


def test_published_port_is_held_and_reaches_the_listener():
    from atom.kv_transfer.disaggregation.mooncake import mooncake_connector as mc

    received = []

    def record_write_done(req_id, *args, **kwargs):
        received.append(req_id)
        raise _Stop

    conn = _consumer(mc, record_write_done)
    thread = _start_listener(mc, conn, ends_with=_Stop)
    mc.MooncakeConnector._await_listener(conn, timeout=10)
    port = mc.MooncakeConnector.notification_port.fget(conn)

    ctx = zmq.Context()
    try:
        # Published only once held: nobody else can take it any more.
        squatter = ctx.socket(zmq.ROUTER)
        with pytest.raises(zmq.ZMQError) as e:
            squatter.bind(f"tcp://*:{port}")
        assert e.value.errno == zmq.EADDRINUSE

        producer = ctx.socket(zmq.DEALER)
        producer.connect(f"tcp://127.0.0.1:{port}")
        producer.send_multipart(
            [mc.MSG_WRITE_DONE, msgpack.dumps({"request_id": "req-0"})]
        )
        thread.join(timeout=10)
    finally:
        ctx.destroy(linger=0)

    assert not thread.is_alive()
    assert received == ["req-0"]


def test_registration_fails_loudly_when_the_listener_cannot_bind(monkeypatch):
    from atom.kv_transfer.disaggregation.mooncake import mooncake_connector as mc

    def refuse_bind(*args, **kwargs):
        raise zmq.ZMQError(zmq.EADDRINUSE)

    monkeypatch.setattr(mc, "zmq_socket_ctx", refuse_bind)
    conn = _consumer(mc, record_write_done=None)
    _start_listener(mc, conn, ends_with=zmq.ZMQError).join(timeout=10)

    with pytest.raises(RuntimeError, match="failed to bind"):
        mc.MooncakeConnector._await_listener(conn, timeout=10)
    assert conn._notification_port is None
    with pytest.raises(RuntimeError, match="not bound yet"):
        mc.MooncakeConnector.notification_port.fget(conn)
