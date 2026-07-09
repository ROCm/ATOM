# SPDX-License-Identifier: MIT
# CPP A·P1b — PPStageTransport ZMQ round-trip (GPU-free).

import os
import tempfile
from dataclasses import dataclass

import zmq

from atom.distributed.pp_transport import PPStageTransport


@dataclass
class _FakeBatch:
    """Stand-in for ScheduledBatch — any picklable object works."""

    req_ids: tuple
    positions: list


def _addrs(tmpdir, pp_size):
    meta = [f"ipc://{os.path.join(tmpdir, f'meta{s}')}" for s in range(pp_size)]
    token = f"ipc://{os.path.join(tmpdir, 'token')}"
    return meta, token


def test_head_last_metadata_and_token_roundtrip():
    ctx = zmq.Context()
    with tempfile.TemporaryDirectory() as tmp:
        meta_addrs, token_addr = _addrs(tmp, 2)
        # Build last first so its bind exists; head connects (ipc tolerates any order).
        last = PPStageTransport(1, 2, meta_addrs, token_addr, ctx=ctx)
        head = PPStageTransport(0, 2, meta_addrs, token_addr, ctx=ctx)

        batch = _FakeBatch(req_ids=(1, 2, 3), positions=[0, 1, 2])
        head.send_metadata(batch)
        got = last.recv_metadata(timeout_ms=2000)
        assert got == batch

        tokens = {"req1": (42,), "req2": (7,)}
        last.send_tokens(tokens)
        back = head.recv_tokens(timeout_ms=2000)
        assert back == tokens

        head.close()
        last.close()
    ctx.term()


def test_metadata_fans_out_to_all_downstream():
    ctx = zmq.Context()
    with tempfile.TemporaryDirectory() as tmp:
        meta_addrs, token_addr = _addrs(tmp, 3)
        s1 = PPStageTransport(1, 3, meta_addrs, token_addr, ctx=ctx)
        s2 = PPStageTransport(2, 3, meta_addrs, token_addr, ctx=ctx)
        head = PPStageTransport(0, 3, meta_addrs, token_addr, ctx=ctx)

        batch = _FakeBatch(req_ids=(9,), positions=[5])
        head.send_metadata(batch)
        assert s1.recv_metadata(timeout_ms=2000) == batch
        assert s2.recv_metadata(timeout_ms=2000) == batch

        head.close()
        s1.close()
        s2.close()
    ctx.term()


def test_recv_timeout_returns_none():
    ctx = zmq.Context()
    with tempfile.TemporaryDirectory() as tmp:
        meta_addrs, token_addr = _addrs(tmp, 2)
        last = PPStageTransport(1, 2, meta_addrs, token_addr, ctx=ctx)
        head = PPStageTransport(0, 2, meta_addrs, token_addr, ctx=ctx)
        # nothing sent → recv should time out, not hang
        assert last.recv_metadata(timeout_ms=200) is None
        assert head.recv_tokens(timeout_ms=200) is None
        head.close()
        last.close()
    ctx.term()
