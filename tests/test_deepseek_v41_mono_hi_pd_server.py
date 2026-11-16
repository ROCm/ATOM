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
- the loop's chunking against the budget; ``TkEngine``'s stubs; the modules import without the GPU stack.

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

from atom.models.deepseek_v41.mono.hi.pd import receive as R
from atom.models.deepseek_v41.mono.hi.pd import server as SV
from atom.models.deepseek_v41.mono.hi.pd import session as S
from atom.models.deepseek_v41.mono.hi.pd import wire as W

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


def test_tk_engine_stubs():
    tk = S.TkEngine()
    assert tk.max_seq_len == MAX_SEQ and tk.num_mtp == 5
    for call in (
        lambda: tk.admit(1, 2),
        lambda: tk.set_sampling([0] * 4, True),
        lambda: tk.load_kv(None, None),
        lambda: tk.step(1),
        S.build_toktab,
    ):
        try:
            call()
        except NotImplementedError as e:
            assert "TileRT" in str(e) or "toktab" in str(e)
        else:
            raise AssertionError("a stub did not raise")
    try:
        tk.check_sampling(None)
    except S.CapabilityUnavailable as e:
        assert "mode-2" in str(e)
    else:
        raise AssertionError("greedy accepted")
    tk.check_sampling(S.SamplingParams())


def test_imports_without_gpu_stack():
    code = (
        "import sys; import atom.models.deepseek_v41.mono.hi.pd.server, atom.models.deepseek_v41.mono.hi.pd.session; "
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
