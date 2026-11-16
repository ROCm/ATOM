# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The DeepSeek-V4.1 decode host behind TileRT's PD router (``mono/hi/pd``: ``receive``, ``session``, ``server``), end
to end on the CPU with ``session.FakeEngine``: no GPU, no aiter, no FlyDSL.

The test plays both peers over real sockets: a prefill rank on the control plane (hello, admission, the landed
header standing in for the RDMA write, done) and the router on HTTP (``/pd/decode`` streamed as NDJSON, read as
TileRT's ``DecodeReader`` reads it: the ``t`` batches, then a ``done`` whose ``n`` must equal the tokens streamed).

- a streamed request: the first token alone, the accepted tokens in order, EOS ends it unstreamed; the sampling mode
  set then the admission (TileRT's slow path), the KV loaded, the slot freed; the next request with the same mode
  admits with the parameters (the fast path), ``ignore_eos`` streams EOS, ``max_tokens`` ends it (``length``);
- the KV landing before or after the decode call; a non-streamed request;
- refusals (busy, sampling, penalties, grammar, logprobs, malformed body, KV timeout), each releasing the request;
- ``/pd/cancel`` and a client gone mid-stream; the control plane's busy / tombstone / oversize rejects;
- the loop's chunking against the budget; greedy refused; the modules import without the GPU stack.

Runs under pytest or as a script (``python -B tests/test_deepseek_v41_mono_hi_pd_server.py``).
"""

import http.client
import json
import os
import socket
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from atom.models.deepseek_v41.mono.hi.pd import fanout as F
from atom.models.deepseek_v41.mono.hi.pd import receive as R
from atom.models.deepseek_v41.mono.hi.pd import server as SV
from atom.models.deepseek_v41.mono.hi.pd import session as S
from atom.models.deepseek_v41.mono.hi.pd import tkworker as TW
from atom.models.deepseek_v41.mono.hi.pd import wire as W
from atom.models.deepseek_v41.mono.hi.tk import host as H

MAX_SEQ = 1 << 20


class FakeTransport:
    name = "mooncake"

    def local_meta(self) -> dict:
        return {"session_id": "127.0.0.1:1"}


@contextmanager
def stack(**engine_kw):
    """A receive server, a decode server over ``FakeEngine(**engine_kw)``, and the landed buffer (its header)."""
    engine = S.FakeEngine(max_seq_len=MAX_SEQ, **engine_kw)
    buf = torch.zeros(W.HDR_BYTES, dtype=torch.uint8)
    recv = R.ReceiveServer(
        MAX_SEQ, 0x7F0000000000, FakeTransport(), port=0, host="127.0.0.1", buffer=buf
    )
    app = SV.DecodeServer(recv, S.Decoder(engine), poll_s=0.002, abandon_drain_s=0.3)
    http_port = app.start()
    try:
        yield engine, recv, app, http_port, buf
    finally:
        app.shutdown()
        recv.close()


# ---- the prefill rank


def land(buf, seq_len: int) -> None:
    """What P's RDMA write leaves in the header (the rest of the image is not read by ``FakeEngine``)."""
    h = buf[: W.HDR_BYTES].view(torch.int32)
    h.zero_()
    h[W.H_MAGIC], h[W.H_VER], h[W.H_N] = W.WIRE_MAGIC, W.WIRE_VERSION, seq_len


def send_kv(
    ctrl_port: int, rid: str, seq_len: int, buf, done: bool = True, sampling=None
):
    """P rank 0: hello, admission, the header written (the RDMA write's stand-in), done. ``(hello, reply, conn)``;
    ``conn`` stays open when ``done`` is False."""
    c = socket.create_connection(("127.0.0.1", ctrl_port), timeout=10)
    hello = W.recv_msg(c)
    msg = {
        "rid": rid,
        "rank": 0,
        "seq_len": seq_len,
        "last_prompt_token": 7,
        "sampling": sampling or {},
        "admission_window_s": 0.6,
    }
    W.send_msg(c, msg)
    reply = W.recv_msg(c)
    if reply.get("accepted") and done:
        land(buf, seq_len)
        W.send_msg(c, W.done_msg(rid, 0, reply["generation"]))
    if done or not reply.get("accepted"):
        c.close()
        c = None
    return hello, reply, c


# ---- the router


def post(port: int, path: str, body) -> tuple[int, dict]:
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    c.request(
        "POST",
        path,
        body=json.dumps(body),
        headers={"Content-Type": "application/json"},
    )
    r = c.getresponse()
    out = r.status, json.loads(r.read() or b"null")
    c.close()
    return out


def get(port: int, path: str) -> dict:
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    c.request("GET", path)
    r = c.getresponse()
    assert r.status == 200, r.status
    out = json.loads(r.read())
    c.close()
    return out


def stream(port: int, body: dict, on_line=None):
    """``(status, lines)`` of a streamed ``/pd/decode``; ``on_line(line, conn)`` after each line (may close ``conn``
    and return True to stop reading)."""
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    c.request(
        "POST",
        "/pd/decode",
        body=json.dumps({**body, "stream": True}),
        headers={"Content-Type": "application/json"},
    )
    r = c.getresponse()
    if r.status != 200:
        out = r.status, json.loads(r.read())
        c.close()
        return out
    assert r.getheader("Content-Type") == "application/x-ndjson"
    lines = []
    while True:
        raw = r.readline()
        if not raw:
            break
        lines.append(json.loads(raw))
        if on_line is not None and on_line(lines[-1], c):
            break
    c.close()
    return 200, lines


def read_reply(lines: list[dict]) -> tuple[list[int], dict]:
    """TileRT's ``DecodeReader``: the tokens of the ``t`` lines and the terminal line, which must be the last and
    declare every token streamed."""
    toks = [t for ln in lines if "t" in ln for t in ln["t"]]
    term = lines[-1]
    assert "done" in term, term
    assert term["n"] == len(toks), (term["n"], len(toks))
    assert all("t" in ln for ln in lines[:-1]), lines
    return toks, term


def expected(first: int, n: int, eos_at=None) -> list[int]:
    return [first] + [
        S.EOS_ID if j == eos_at else S.FakeEngine.token(first, j) for j in range(1, n)
    ]


def wait_idle(port: int, timeout: float = 10.0) -> dict:
    t = time.time() + timeout
    while time.time() < t:
        st = get(port, "/decode_status")
        if st["status"] == "idle":
            return st
        time.sleep(0.01)
    raise AssertionError(f"still busy: {st}")


# ---- tests


def test_stream_eos_then_fast_path_and_length():
    with stack(eos_at=10) as (eng, recv, _app, port, buf):
        assert get(port, "/health") == {"status": "ok"}
        caps = get(port, "/capabilities")
        assert caps["profile"] == "dsv41"
        assert caps["capabilities"] == {
            "penalties": False,
            "ignore_eos": True,
            "logprobs": False,
        }

        # the decode call first, the KV landing while it waits
        def late_p():
            time.sleep(0.2)
            hello, reply, _ = send_kv(recv.port, "r1", 1000, buf)
            late_p.hello, late_p.reply = hello, reply

        th = threading.Thread(target=late_p)
        th.start()
        body = {
            "rid": "r1",
            "first_token_id": 12345,
            "max_tokens": 256,
            "sampling": {"temperature": 1.0},
        }
        status, lines = stream(port, body)
        th.join()
        assert status == 200
        h = late_p.hello
        assert (h["magic"], h["protocol_version"], h["layout_version"]) == (
            "tilert-pd",
            2,
            2041,
        )
        assert (h["transport"], h["session_id"], h["max_seq_len"], h["busy"]) == (
            "mooncake",
            "127.0.0.1:1",
            MAX_SEQ,
            False,
        )
        assert (h["senders"], h["dsv41_base"], h["dsv41_wire"]) == (
            1,
            0x7F0000000000,
            1,
        )
        assert late_p.reply == {
            "accepted": True,
            "rid": "r1",
            "rank": 0,
            "generation": 1,
        }
        assert lines[0] == {"t": [12345]}  # the first token alone
        toks, term = read_reply(lines)
        assert toks == expected(12345, 10)  # EOS (index 10) ends the reply, unstreamed
        assert (term["finish_reason"], term["seq_len"]) == ("stop", 1000)
        assert term["timing_ms"]["mtp_verify_calls"] >= 3
        assert eng.calls[:3] == [
            ("set_sampling", True),
            ("admit", 12345, 1000, False),
            ("load_kv", "r1", 1000),
        ]
        assert (
            eng.params[1] == S.pack_params(S.SamplingParams())[1] == 0x3F8000003F800000
        )
        assert wait_idle(port)["current_rid"] is None
        assert recv.state_snapshot()["rid"] is None

        # the KV first, then the decode call; same sampling mode: the admission carries the parameters
        send_kv(recv.port, "r2", 2000, buf)
        eng.calls.clear()
        body = {
            "rid": "r2",
            "first_token_id": 777,
            "max_tokens": 23,
            "sampling": {"temperature": 1.0, "ignore_eos": True},
        }
        status, lines = stream(port, body)
        toks, term = read_reply(lines)
        assert toks == expected(
            777, 23, eos_at=10
        )  # ignore_eos: EOS streamed like any token
        assert term["finish_reason"] == "length"
        assert eng.calls[:2] == [("admit", 777, 2000, True), ("load_kv", "r2", 2000)]
        steps = [c[1] for c in eng.calls if c[0] == "step"]
        assert all(1 <= k <= S.AR_STEPS for k in steps)
        assert sum(steps) * 6 >= 22  # enough steps were asked for the budget
        wait_idle(port)


def test_non_stream_and_greedy_mode_switch():
    with stack() as (eng, recv, _app, port, buf):
        send_kv(recv.port, "r3", 64, buf)
        st, out = post(
            port,
            "/pd/decode",
            {
                "rid": "r3",
                "first_token_id": 5,
                "max_tokens": 9,
                "sampling": {"temperature": 0},
            },
        )
        assert st == 200, out
        assert out["rid"] == "r3" and out["seq_len"] == 64
        assert out["token_ids"] == expected(5, 9)
        assert out["timing_ms"]["finish_reason"] == "length"
        assert eng.calls[0] == ("set_sampling", False)  # greedy
        assert wait_idle(port)["status"] == "idle"


def test_refusals_release_the_request():
    with stack(step_s=0.01) as (_eng, recv, _app, port, buf):
        st, out = post(port, "/pd/decode", {"first_token_id": 1})
        assert st == 422 and "rid" in out["detail"]
        cases = [
            (
                {"sampling": {"temperature": 1.0, "top_p": 0.9}},
                501,
                "capability_unavailable",
            ),
            ({"sampling": {"repetition_penalty": 1.2}}, 501, "capability_unavailable"),
            ({"sampling": {"temperature": -1}}, 400, "invalid_parameter"),
            ({"grammar_spec": {"type": "json_object"}}, 501, "grammar_unsupported"),
            ({"top_logprobs": 2}, 501, "logprobs_unavailable"),
        ]
        for i, (extra, status, etype) in enumerate(cases):
            st, out = post(
                port, "/pd/decode", {"rid": f"x{i}", "first_token_id": 3, **extra}
            )
            assert (st, out.get("error_type")) == (status, etype), (extra, st, out)
            assert wait_idle(port)["current_rid"] is None
        # a refused request's late KV is refused too (tombstone), so it cannot land on the next request
        _, reply, _ = send_kv(recv.port, "x0", 10, buf)
        assert reply == {
            "accepted": False,
            "error": "cancelling",
            "rid": "x0",
            "detail": "no_consumer",
        }
        st, out = post(
            port, "/pd/decode", {"rid": "late", "first_token_id": 3, "timeout_s": 0.3}
        )
        assert (st, out) == (504, {"error": "kv_transfer_timeout", "rid": "late"})

        # busy while a request decodes
        send_kv(recv.port, "long", 100, buf)
        got = threading.Event()
        res = {}

        def first_line(line, conn):
            got.set()
            return False

        th = threading.Thread(
            target=lambda: res.update(
                r=stream(
                    port,
                    {"rid": "long", "first_token_id": 9, "max_tokens": 300},
                    first_line,
                )
            )
        )
        th.start()
        assert got.wait(10)
        st, out = post(port, "/pd/decode", {"rid": "other", "first_token_id": 3})
        assert (st, out) == (429, {"error": "busy", "current_rid": "long"})
        th.join()
        toks, term = read_reply(res["r"][1])
        assert len(toks) == 300 and term["finish_reason"] == "length"
        wait_idle(port)


def test_cancel_and_client_gone():
    with stack(step_s=0.005) as (eng, recv, _app, port, buf):
        send_kv(recv.port, "c1", 100, buf)

        def cancel_after_first(line, conn):
            if line == {"t": [42]}:
                assert post(port, "/pd/cancel", {"rid": "c1"}) == (
                    200,
                    {"cancelled": "c1"},
                )
            return False

        st, lines = stream(
            port,
            {"rid": "c1", "first_token_id": 42, "max_tokens": 100000},
            cancel_after_first,
        )
        assert st == 200
        toks, term = read_reply(lines)
        assert term["finish_reason"] == "cancelled"
        assert toks == expected(42, len(toks)) and len(toks) < 2000
        assert post(port, "/pd/cancel", {"rid": "c1"})[0] == 404
        wait_idle(port)

        # the client leaves mid-stream: the decode is cancelled and the request released
        send_kv(recv.port, "c2", 100, buf)

        def leave(line, conn):
            conn.sock.shutdown(socket.SHUT_RDWR)
            return True

        stream(port, {"rid": "c2", "first_token_id": 43, "max_tokens": 100000}, leave)
        assert wait_idle(port)["current_rid"] is None
        assert eng.next_index < 20000  # stopped well before the budget


def test_control_plane_rejects():
    with stack() as (_eng, recv, _app, port, buf):
        _, reply, _ = send_kv(recv.port, "big", MAX_SEQ + 1, buf)
        assert (
            reply["accepted"] is False
            and reply["error"] == "seq_len exceeds max_seq_len"
        )
        _, r1, held = send_kv(
            recv.port, "a", 10, buf, done=False
        )  # a writer still transferring
        assert r1["accepted"]
        hello, r2, _ = send_kv(recv.port, "b", 10, buf)
        assert hello["busy"] is True
        assert r2 == {
            "accepted": False,
            "error": "busy",
            "busy_rid": "a",
            "busy_state": "transferring",
        }
        land(buf, 10)
        W.send_msg(held, W.done_msg("a", 0, r1["generation"]))
        held.close()
        st, out = post(
            port, "/pd/decode", {"rid": "a", "first_token_id": 2, "max_tokens": 4}
        )
        assert st == 200 and out["token_ids"] == expected(2, 4)
        _, again, _ = send_kv(
            recv.port, "a", 10, buf
        )  # a late duplicate of a finished request
        assert again["accepted"] is False and again["error"] == "cancelling"
        _, r3, _ = send_kv(recv.port, "b", 10, buf)
        assert r3["accepted"] and r3["generation"] == r1["generation"] + 1


def test_decoder_chunks_and_unbegun_path():
    eng = S.FakeEngine(max_seq_len=4096, accepts=(6,))
    dec = S.Decoder(eng)
    rec = W.ReceivedRequest(
        rid="u", seq_len=4000, last_prompt_token=0, first_token_id=None, sampling=None
    )
    dec.inject(
        rec, None
    )  # no begin: decode sets the mode and admits after the first token
    got = []
    toks = dec.decode(11, 500, {"temperature": 1.0}, on_token=got.append)
    # budget min(500, 4096 - 4000 - 1) = 95; but a chunk only goes where the cache has room for all its rows with 8
    # positions of slack: (4096 - 4001 - 8) // 6 = 14 steps, so 8 + 6 steps, 84 tokens after the first
    assert toks == got == expected(11, len(toks))
    assert dec.last_stats["finish_reason"] == "length" and len(toks) == 85
    steps = [c[1] for c in eng.calls if c[0] == "step"]
    assert steps == [8, 6]
    assert eng.calls[0] == ("load_kv", "u", 4000) and eng.calls[1][0] == "set_sampling"
    assert eng.calls[2] == ("admit", 11, 4000, False)
    assert S.pack_params(S.SamplingParams(temperature=0.5, seed=-1))[0] == -1


# tkserve 11-16, rank 0, the 2nd dsv41_e2e_begin (m7/rec_admit.jsonl seq 52488) and its buffers (the same process's map)
REC_ADMIT = (
    "0000ae2fde7e0000400012000000000000003ca2df7e0000400010000000000000800c14557e000040001000000000000078"
    "7f18707f0000007a7f18707f0000007c7f18707f00000800000000000000007e7f18707f000000807f18707f00000600000000"
    "0000000000a04d057f000000006018707f0000ad1a0600150a0000000010000100000000867f18707f00002a06090000000000"
    "0000803f0000803f00000000000000000300000000000000"
)
REC_ADDR = {
    "temp:plain_acc": 0x7EDE2FAE0000,
    "temp:ar_acc": 0x7EDFA23C0000,
    "temp:ar_num": 0x7E55140C8000,
    "temp:ctl": 0x7F70187F7800,
    "temp:blk_tok": 0x7F70187F7A00,
    "temp:blk_pos": 0x7F70187F7C00,
    "temp:vtok": 0x7F70187F7E00,
    "temp:vpos": 0x7F70187F8000,
    "param:embed": 0x7F054DA00000,
    "temp:X": 0x7F7018600000,
    "temp:samp_params": 0x7F70187F8600,
}


def test_admit_args_rebuild_recorded_blob():
    # dsv41_e2e_begin(first 2581, pos 400045, pack_params(temperature 1, seed 591402)): the .so's sampling counter is
    # the 4th word (3: set_sampling made it 1, this is the 2nd begin)
    params = S.pack_params(S.SamplingParams(temperature=1.0, seed=591402))
    assert params[3] == 0
    blob = H.admit_args(REC_ADDR, 2581, 400045, True, params[:3] + [3])
    assert blob == bytes.fromhex(REC_ADMIT)
    reset = H.admit_args(
        REC_ADDR, 0, 0, False, None
    )  # _reset_streams: no parameters, flag 0
    assert reset[:0x78] == blob[:0x78] and reset[0x78:] == bytes(8) + (
        1 << 20
    ).to_bytes(4, "little") + bytes(44)


@contextmanager
def rank_stack(nranks=8, timeouts=None, **rank_kw):
    """The decode server over ``fanout.RankEngine`` and ``nranks`` spawned ``FakeRank`` workers."""
    group = F.RankGroup(
        nranks,
        "atom.models.deepseek_v41.mono.hi.pd.fanout:FakeRank",
        rank_kw,
        timeouts=timeouts,
    )
    try:
        eng = F.RankEngine(group)
        info = group.reports[0]
        recv = R.ReceiveServer(
            eng.max_seq_len,
            info["base_ptr"],
            R.PublishedTransport(info["transport"], {"session_id": info["session_id"]}),
            port=0,
            host="127.0.0.1",
        )
        app = SV.DecodeServer(recv, S.Decoder(eng), poll_s=0.002, abandon_drain_s=0.3)
        port = app.start()
        try:
            yield group, eng, recv, port
        finally:
            app.shutdown()
            recv.close()
    finally:
        group.close()


def test_fanout_eight_ranks_end_to_end():
    buf = torch.zeros(W.HDR_BYTES, dtype=torch.uint8)
    with rank_stack(eos_at=10) as (group, eng, recv, port):
        assert [r["rank"] for r in group.reports] == list(range(8))
        assert len({r["pid"] for r in group.reports}) == 8  # one process a rank
        send_kv(recv.port, "f1", 1000, buf)
        status, lines = stream(
            port,
            {
                "rid": "f1",
                "first_token_id": 12345,
                "max_tokens": 256,
                "sampling": {"temperature": 1.0},
            },
        )
        toks, term = read_reply(lines)
        assert (
            status == 200
            and toks == expected(12345, 10)
            and term["finish_reason"] == "stop"
        )
        send_kv(recv.port, "f2", 2000, buf)
        status, lines = stream(
            port,
            {
                "rid": "f2",
                "first_token_id": 777,
                "max_tokens": 23,
                "sampling": {"temperature": 1.0, "ignore_eos": True},
            },
        )
        toks, term = read_reply(lines)
        assert (
            toks == expected(777, 23, eos_at=10) and term["finish_reason"] == "length"
        )
        states = [a.json() for a in group.call("state")]
        logs = [st["log"] for st in states]
        assert all(lg == logs[0] for lg in logs)  # every rank got the same commands
        ops = [e[0] for e in logs[0]]
        # f1 stopped with a chunk in flight: every rank dropped it
        assert ops == [
            "set_sampling",
            "admit",
            "load_kv",
            "discard",
            "admit",
            "load_kv",
        ]
        assert all(st["in_flight"] == [] for st in states)
        (_, p1, on), (_, f, pos, mtp, p_none) = logs[0][0], logs[0][1]
        assert on and p1[3] == 1 and (f, pos, mtp, p_none) == (12345, 1000, True, None)
        _, f, pos, mtp, p2 = logs[0][
            4
        ]  # same sampling mode: the parameters ride with the admission
        assert (
            (f, pos, mtp) == (777, 2000, True) and p2[1:3] == p1[1:3] and p2[3] == 2
        )  # [0]: a fresh seed
        assert eng.counter == 2
        for st in states:  # each rank's AdmitArgs over its own buffers
            blob, addr = bytes.fromhex(st["admit_blob"]), st["addr"]
            assert blob == H.admit_args(addr, 777, 2000, True, p2)
            assert int.from_bytes(blob[0x88:0x90], "little") == addr["temp:samp_params"]
        assert (
            wait_idle(port)["current_rid"] is None
            and recv.state_snapshot()["rid"] is None
        )


def test_fanout_failures():
    # an op raising on one rank: RankError with that rank's traceback; the group stays usable
    g = F.RankGroup(
        2,
        "atom.models.deepseek_v41.mono.hi.pd.fanout:FakeRank",
        {"fail": {"op": "admit", "rank": 1}},
    )
    try:
        eng = F.RankEngine(g)
        try:
            eng.admit(5, 10, None)
        except F.RankError as e:
            assert (
                e.op == "admit"
                and sorted(e.errors) == [1]
                and "injected failure" in e.errors[1]
            )
        else:
            raise AssertionError("no RankError")
        assert len(g.call("state")) == 2 and g.broken is None
    finally:
        g.close()
    # an op not answered in time: RankTimeout, then the group is broken
    g = F.RankGroup(
        2,
        "atom.models.deepseek_v41.mono.hi.pd.fanout:FakeRank",
        {"hang": {"op": "step", "rank": 1, "s": 5}},
        timeouts={"step": 0.3},
    )
    try:
        t = time.monotonic()
        try:
            g.call("step", [1, 1])
        except F.RankTimeout as e:
            assert "[1]" in str(e) and time.monotonic() - t < 3
        else:
            raise AssertionError("no RankTimeout")
        try:
            g.call("state")
        except F.RankGroupBroken:
            pass
        else:
            raise AssertionError("a broken group answered")
    finally:
        g.close(graceful=False)
    # a worker gone: RankDied at once, not at the timeout
    g = F.RankGroup(
        2,
        "atom.models.deepseek_v41.mono.hi.pd.fanout:FakeRank",
        {"hang": {"op": "sync", "rank": 0, "s": 30}},
    )
    try:
        g.procs[1].kill()
        g.procs[1].join(5)
        t = time.monotonic()
        try:
            g.call("state")
        except F.RankDied as e:
            assert "[1]" in str(e) and time.monotonic() - t < 3
        else:
            raise AssertionError("no RankDied")
    finally:
        g.close(graceful=False)
    # a backend that cannot start: its traceback at startup
    try:
        F.RankGroup(
            2, "atom.models.deepseek_v41.mono.hi.pd.fanout:FakeRank", {"no_such_arg": 1}
        )
    except F.RankError as e:
        assert e.op == "hello" and "no_such_arg" in e.errors[0]
    else:
        raise AssertionError("bad backend started")


def _reply(lines):
    """``(tokens, terminal line)`` of a finished reply, or ``(None, the error line)``."""
    if lines and "error" in lines[-1]:
        return None, lines[-1]
    return read_reply(lines)


def test_fanout_stop_with_chunks_in_flight_then_many_requests():
    """A reply that ends at a stop id leaves the chunk enqueued after the one holding it (two in flight): every rank
    drops it at the reply's end (``discard``), so any number of such replies run back to back. Before: nothing popped
    it, so the fifth one left behind made the next request's first ``step`` fail on every rank ("more than 4 chunks in
    flight"). A reply ended by its budget leaves none."""
    buf = torch.zeros(W.HDR_BYTES, dtype=torch.uint8)
    replies, states = [], []
    with rank_stack(nranks=4, eos_at=10) as (group, _eng, recv, port):
        for i in range(9):
            rid, length = f"s{i}", i % 3 == 2
            send_kv(recv.port, rid, 1000 + i, buf)
            body = {"rid": rid, "first_token_id": 100 + i, "max_tokens": 40 if length else 256,
                    "sampling": {"temperature": 1.0, "ignore_eos": length}}  # fmt: skip
            replies.append((length, i, stream(port, body)[1]))
            wait_idle(port)
            states.append([a.json() for a in group.call("state")])
        for length, i, lines in replies:  # every reply finished
            toks, term = _reply(lines)
            assert toks is not None, (i, term)
            if length:
                assert term["finish_reason"] == "length" and len(toks) == 40
                assert "discarded" not in term["timing_ms"]
            else:
                assert term["finish_reason"] == "stop" and toks == expected(100 + i, 10)
                assert term["timing_ms"]["discarded"] == 1
        assert all(
            st["in_flight"] == [] for sts in states for st in sts
        )  # nothing carried over, on any rank
        logs = [st["log"] for st in states[-1]]
        assert all(lg == logs[0] for lg in logs)  # every rank got the same commands
        assert sum(e[0] == "discard" for e in logs[0]) == 6
        assert group.broken is None


def test_fanout_rank_failure_mid_reply_leaves_no_chunks():
    """An op failing on one rank mid-reply (here rank 0's ``wait_read``, before it pops the chunk, as a KeyError there
    would): the ranks' in-flight chunks differ, and the failed reply's ``discard`` empties every rank's; the error
    names the rank's last commands."""
    buf = torch.zeros(W.HDR_BYTES, dtype=torch.uint8)
    with rank_stack(nranks=2, fail={"op": "wait_read", "rank": 0}) as (
        group,
        _eng,
        recv,
        port,
    ):
        send_kv(recv.port, "x1", 500, buf)
        _, lines = stream(port, {"rid": "x1", "first_token_id": 9, "max_tokens": 256})
        toks, err = _reply(lines)
        assert toks is None and "wait_read failed on rank(s) [0]" in err["error"]
        assert "recent commands" in err["error"] and "step[" in err["error"]
        wait_idle(port)
        states = [a.json() for a in group.call("state")]
        assert [s["in_flight"] for s in states] == [[], []]
        (d0,), (d1,) = ([e for e in s["log"] if e[0] == "discard"] for s in states)
        # rank 0 still held the chunk it failed on, rank 1 had popped it; both dropped what they held
        assert d0[1] == d1[1] == d1[2] and len(d0[2]) == len(d1[2]) + 1
        assert group.broken is None


def test_fanout_cancel_reconnect_and_abandoned_request():
    """The sequence around a router restart, every request on every rank: a cancel after the first token (the chunks
    in flight still stream: nothing to drop), a client gone mid-stream, a request abandoned before its KV came (504;
    its late P refused), a P of a request this node never heard of (the old router's: admitted, landed, never asked
    for), then a request that stops early (its stale predecessor in the completed queue dropped, its own leftover chunk
    discarded). No rank keeps a chunk, every rank got the same commands, the group stays whole.
    """
    buf = torch.zeros(W.HDR_BYTES, dtype=torch.uint8)
    with rank_stack(nranks=4, step_s=0.002, eos_at=30) as (group, _eng, recv, port):

        def in_flight():
            wait_idle(port)
            return [a.json()["in_flight"] for a in group.call("state")]

        send_kv(recv.port, "c1", 100, buf)

        def cancel_after_first(line, conn):
            if line == {"t": [42]}:
                assert post(port, "/pd/cancel", {"rid": "c1"}) == (
                    200,
                    {"cancelled": "c1"},
                )
            return False

        _, lines = stream(port, {"rid": "c1", "first_token_id": 42, "max_tokens": 100000,
                                 "sampling": {"temperature": 1.0, "ignore_eos": True}}, cancel_after_first)  # fmt: skip
        toks, term = _reply(lines)
        assert (
            term["finish_reason"] == "cancelled"
            and "discarded" not in term["timing_ms"]
        )
        assert in_flight() == [[]] * 4

        send_kv(recv.port, "c2", 100, buf)

        def leave(line, conn):
            conn.sock.shutdown(socket.SHUT_RDWR)
            return True

        stream(port, {"rid": "c2", "first_token_id": 43, "max_tokens": 100000,
                      "sampling": {"temperature": 1.0, "ignore_eos": True}}, leave)  # fmt: skip
        assert in_flight() == [[]] * 4

        # abandoned: the decode call gives up on its KV; its P, late, is refused
        st, out = post(
            port, "/pd/decode", {"rid": "gone", "first_token_id": 1, "timeout_s": 0.3}
        )
        assert st == 504 and out["error"] == "kv_transfer_timeout"
        _, reply, _ = send_kv(recv.port, "gone", 100, buf)
        assert reply["error"] == "cancelling" and reply["detail"] == "no_consumer"
        # the old router's request, never asked for here: its P lands
        _, reply, _ = send_kv(recv.port, "orphan", 300, buf)
        assert reply["accepted"]
        # the next request: the orphan is dropped from the completed queue, the reply stops with a chunk in flight
        send_kv(recv.port, "n1", 4000, buf)
        _, lines = stream(port, {"rid": "n1", "first_token_id": 77, "max_tokens": 256})
        toks, term = _reply(lines)
        assert (
            toks == expected(77, 30)
            and term["finish_reason"] == "stop"
            and term["seq_len"] == 4000
        )
        assert term["timing_ms"]["discarded"] == 1
        assert in_flight() == [[]] * 4
        logs = [a.json()["log"] for a in group.call("state")]
        assert all(lg == logs[0] for lg in logs)
        assert [e[1] for e in logs[0] if e[0] == "load_kv"] == [100, 100, 4000]
        assert group.broken is None and recv.state_snapshot()["rid"] is None


def test_rank_group_one_caller_at_a_time():
    """A second thread's call while one is in flight is refused (``RankGroupBusy``) instead of overwriting the
    mailbox under ranks that have not read it yet; the call in flight completes and the group stays whole.
    """
    g = F.RankGroup(2, "atom.models.deepseek_v41.mono.hi.pd.fanout:FakeRank", {"hang": {"op": "set_sampling", "rank": 0, "s": 1.0}})  # fmt: skip
    try:
        out = {}

        def first():
            out["first"] = g.call("set_sampling", [1, 1, 2, 3, 4])

        th = threading.Thread(target=first, name="first")
        th.start()
        time.sleep(0.2)
        try:
            g.call("discard", [7])
        except F.RankGroupBusy as e:
            assert "('set_sampling', 'first')" in str(e)
        else:
            raise AssertionError("a concurrent call went through")
        th.join(10)
        assert len(out["first"]) == 2 and g.broken is None
        assert [a.res for a in g.call("discard", [])] == [[], []]
        logs = [a.json()["log"] for a in g.call("state")]
        # the refused call reached no rank
        assert logs[0] == logs[1] == [["set_sampling", [1, 2, 3, 4], True], ["discard", [], []]]  # fmt: skip
    finally:
        g.close()


def test_decoder_discards_what_a_reply_leaves():
    """``Decoder`` over ``FakeEngine``: a stop with a chunk in flight discards it; a budget end has none; a failure
    mid-reply discards and forgets the sampling mode, so the next request sets it again.
    """
    eng = S.FakeEngine(max_seq_len=1 << 16, eos_at=10)
    dec = S.Decoder(eng)

    def run(first, max_tokens, sampling, on_token=None):
        rec = W.ReceivedRequest(rid=str(first), seq_len=100, last_prompt_token=0, first_token_id=None, sampling=None)  # fmt: skip
        dec.begin(first, sampling)
        dec.inject(rec, None)
        eng.calls.clear()
        return dec.decode(first, max_tokens, sampling, on_token=on_token)

    assert run(5, 256, {"temperature": 1.0}) == expected(5, 10)
    assert eng.calls[-1] == ("discard", 1) and dec.last_stats["discarded"] == 1
    run(6, 20, {"temperature": 1.0, "ignore_eos": True})
    assert dec.last_stats["finish_reason"] == "length" and not any(
        c[0] == "discard" for c in eng.calls
    )

    def boom(tok):
        if tok != 7:
            raise ConnectionError("client gone")

    try:
        run(7, 256, {"temperature": 1.0, "ignore_eos": True}, on_token=boom)
    except ConnectionError:
        pass
    else:
        raise AssertionError("no error")
    assert eng.calls[-1][0] == "discard" and dec._samp_mode is None
    eng.calls.clear()
    dec.begin(8, {"temperature": 1.0})
    dec.inject(W.ReceivedRequest(rid="8", seq_len=100, last_prompt_token=0, first_token_id=None, sampling=None), None)  # fmt: skip
    assert eng.calls[0] == (
        "set_sampling",
        True,
    )  # not the parameters riding with the admission


def test_receive_slot_held_while_its_kv_is_read():
    """While the decode call reads a request's KV out of the buffer (``claim`` .. ``read_done``), a P of another
    request (one left behind by an abandoned request, landing late) is refused busy; after it, TileRT's rule.
    """
    buf = torch.zeros(W.HDR_BYTES, dtype=torch.uint8)
    recv = R.ReceiveServer(MAX_SEQ, 0, FakeTransport(), port=0, host="127.0.0.1", buffer=buf)  # fmt: skip
    try:
        send_kv(recv.port, "a", 100, buf)
        rec = recv.completed.get(timeout=5)
        assert recv.claim(rec)
        _, reply, _ = send_kv(recv.port, "late", 200, buf)
        assert reply["error"] == "busy" and reply["busy_rid"] == "a"
        recv.read_done("a")
        _, reply, _ = send_kv(recv.port, "late", 200, buf)
        assert reply["accepted"]
        assert not recv.claim(rec)  # the slot moved on: a's buffer may hold late's KV
        recv.release("a")
        assert recv.state_snapshot()["rid"] == "late"
    finally:
        recv.close()


def test_stream_release_waits_for_the_decode_worker():
    """A streamed decode whose worker outlives its stream (here a slow chunk when the client leaves) keeps the
    request until the worker stops: a decode call meanwhile is answered busy, not started beside it.
    """
    with stack(step_s=0.05) as (_eng, recv, app, port, buf):
        app.join_s = 0.01
        send_kv(recv.port, "w1", 100, buf)

        def leave(line, conn):
            conn.sock.shutdown(socket.SHUT_RDWR)
            return True

        stream(port, {"rid": "w1", "first_token_id": 3, "max_tokens": 100000}, leave)
        time.sleep(0.1)
        assert get(port, "/decode_status")["status"] == "busy"
        assert post(port, "/pd/decode", {"rid": "w2", "first_token_id": 4})[0] == 429
        assert wait_idle(port)["current_rid"] is None


def test_tk_worker_discard_and_missing_chunk():
    w = TW.TkRankWorker.__new__(TW.TkRankWorker)
    w.rank, w.events = 3, {5: object(), 6: object()}
    try:
        w.wait_read(9, 1, 8, 1)
    except RuntimeError as e:
        assert "rank 3: chunk 9 is not in flight (in flight: [5, 6])" in str(e)
    else:
        raise AssertionError("no error")
    assert w.discard([6]) == [5, 6] and w.events == {}


def test_tk_worker_greedy():
    eng = F.RankEngine.__new__(F.RankEngine)
    eng._S, eng.greedy = S, False
    try:
        eng.check_sampling(None)
    except S.CapabilityUnavailable as e:
        assert "mode-2" in str(e)
    else:
        raise AssertionError("greedy accepted")
    eng.check_sampling(S.SamplingParams())


def test_load_planes_file():
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        planes = {
            "main_kv": {2: (torch.zeros(3, 256, dtype=torch.uint8),)},
            "index_k": {},
            "win": [],
            "state": {},
        }
        torch.save(
            {
                "n": 6,
                "first": 9,
                "planes": {**planes, "hist": torch.zeros(16, dtype=torch.int32)},
            },
            f"{d}/ok.pt",
        )
        snap = TW.load_planes_file(f"{d}/ok.pt", 6)
        assert (snap["n"], snap["first"]) == (6, 9) and snap["planes"]["main_kv"][2][
            0
        ].shape == (3, 256)
        for bad, args in (
            ({"n": 6, "first": 9, "planes": planes}, (6,)),
            ({"n": 6}, (None,)),
        ):
            torch.save(bad, f"{d}/bad.pt")
            try:
                TW.load_planes_file(f"{d}/bad.pt", *args)
            except ValueError:
                pass
            else:
                raise AssertionError(f"accepted {sorted(bad)}")
        try:
            TW.load_planes_file(f"{d}/ok.pt", 7)
        except ValueError as e:
            assert "6 positions" in str(e)
        else:
            raise AssertionError("accepted a wrong length")


def _share(rank, port, q):
    import torch.distributed as dist

    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2
    )
    planes = None
    if rank == 0:
        win = (
            torch.full((2, 4), 0.5).to(torch.float8_e4m3fn),
            torch.ones(2, dtype=torch.uint8),
        )
        planes = {
            "main_kv": {2: (torch.arange(6, dtype=torch.uint8).view(2, 3),)},
            "win": [win],
        }
    got = TW.share_planes(planes)
    w = got["win"][0]
    q.put(
        (
            rank,
            got["main_kv"][2][0].tolist(),
            w[0].dtype == torch.float8_e4m3fn,
            w[0].float().tolist(),
            w[1].tolist(),
        )
    )
    dist.destroy_process_group()


def test_share_planes_over_gloo():
    """``kv_share="gloo"``: rank 0's converted planes reach every rank, dtypes kept."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    ctx = torch.multiprocessing.get_context("spawn")
    q = ctx.Queue()
    ps = [ctx.Process(target=_share, args=(r, port, q)) for r in range(2)]
    for p_ in ps:
        p_.start()
    got = sorted(q.get(timeout=120) for _ in ps)
    for p_ in ps:
        p_.join(30)
    assert [g[0] for g in got] == [0, 1]
    assert all(
        g[1:] == ([[0, 1, 2], [3, 4, 5]], True, [[0.5] * 4] * 2, [1, 1]) for g in got
    )


def test_server_flags_for_a_real_run():
    a = SV.build_parser().parse_args(["--engine", "tk", "--with-mtp", "--num-mtp", "5", "--accept", "recorded", "--kv-share", "ipc"])  # fmt: skip
    assert (a.accept, a.kv_share, a.max_seq_len, a.ctrl_port, a.http_port) == ("recorded", "ipc", 1 << 20, 5556, 5557)  # fmt: skip
    d = SV.build_parser().parse_args([])
    assert (d.accept, d.kv_share) == ("real", "gloo")


def test_imports_without_gpu_stack():
    code = (
        "import sys; import atom.models.deepseek_v41.mono.hi.pd.server, atom.models.deepseek_v41.mono.hi.pd.tkworker; "
        "print(sorted({m.split('.')[0] for m in sys.modules} & {'aiter', 'flydsl', 'triton'}))"
    )
    env = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": os.pathsep.join([str(ROOT), *sys.path]),
    }
    out = subprocess.run(
        [sys.executable, "-B", "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]", out.stdout


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            t = time.time()
            fn()
            print(f"{name}: ok ({time.time() - t:.2f}s)")
