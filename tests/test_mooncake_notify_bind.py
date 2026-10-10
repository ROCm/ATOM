# SPDX-License-Identifier: MIT
"""The consumer's write-done listener must hold its port before publishing it.

`get_open_port()` reports a port that was free a moment ago. When the listener
bound it later, another process could take it first; the listener thread died
on EADDRINUSE and that decode rank never finished a KV receive again.
"""

from types import SimpleNamespace

import pytest
import zmq


@pytest.fixture
def taken_port():
    """A port some other socket already holds, as on a busy node."""
    ctx = zmq.Context()
    squatter = ctx.socket(zmq.ROUTER)
    port = squatter.bind_to_random_port("tcp://*")
    yield port
    ctx.destroy(linger=0)


def test_bind_retries_a_port_taken_after_it_was_picked(monkeypatch, taken_port):
    from atom.kv_transfer.disaggregation.mooncake import mooncake_connector as mc

    free = mc.get_open_port()
    picks = iter([taken_port, free])
    monkeypatch.setattr(mc, "get_open_port", lambda: next(picks))
    conn = SimpleNamespace(_notification_port=None)

    ctx, sock = mc.MooncakeConnector._bind_notification_socket(conn)
    try:
        # Published only once held, and it is the port the socket holds.
        assert conn._notification_port == free
        endpoint = sock.getsockopt_string(zmq.LAST_ENDPOINT)
        assert endpoint.endswith(f":{free}")
    finally:
        ctx.destroy(linger=0)


def test_bind_gives_up_loudly_and_publishes_nothing(monkeypatch, taken_port):
    from atom.kv_transfer.disaggregation.mooncake import mooncake_connector as mc

    monkeypatch.setattr(mc, "get_open_port", lambda: taken_port)
    conn = SimpleNamespace(_notification_port=None)

    with pytest.raises(RuntimeError, match="could not bind"):
        mc.MooncakeConnector._bind_notification_socket(conn)
    assert conn._notification_port is None
