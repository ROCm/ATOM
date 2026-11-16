# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Ports TileRT 0.1.7 (MIT, Copyright (c) TileRT-team) ``tilert/pd_vllm/decode_server.py`` (``build_app``, ``main``) and
# ``tilert/pd_vllm/dsv41_decode_server.py``: the decode node's HTTP side, as TileRT's ``pd_router`` calls it.
"""ATOM's decode ("D") host for DeepSeek-V4.1-Flash behind TileRT's PD router: the router-facing HTTP endpoints, over
the receive path (``receive``) and a decode loop (``session.Decoder``).

Endpoints (TileRT ``decode_server.py:42-301``), one request decoded at a time:

- ``GET /health`` -> ``{"status": "ok"}``; ``GET /capabilities`` -> ``{profile, engine, capabilities: {penalties,
  ignore_eos, logprobs}}`` (the router intersects them over its pool); ``GET /decode_status`` -> ``{status:
  busy | idle, current_rid}``;
- ``POST /pd/cancel {rid}`` -> ``{cancelled: rid}``, or 404 when ``rid`` is not the request in flight (the router posts
  it when it lets go of a request it dispatched, e.g. the client left or a stop string matched);
- ``POST /pd/decode {rid, first_token_id, max_tokens=256, sampling, timeout_s=120, stream=false, grammar_spec,
  enable_thinking=true, top_logprobs}``: 429 ``busy`` while another request decodes; refusals (logprobs 501,
  ``validate_sampling`` / ``check_sampling`` 400 / 501, a grammar 501) after the request's KV has landed or 30 s, so
  the slot is released; then the KV is awaited (``timeout_s``: 504 ``kv_transfer_timeout``; cancelled: 499
  ``request_cancelled``), admitted and loaded (500 on failure), and decoded:

  - ``stream=false``: ``{rid, token_ids, seq_len, timing_ms}``;
  - ``stream=true``: NDJSON lines, ``{"t": [first]}`` alone, then ``{"t": [...]}`` batches (what arrived within
    ``TILERT_DECODE_POLL_MS``, default 200 ms), then ``{"done": true, "n", "seq_len", "finish_reason", "timing_ms"}``
    (``n``: every token streamed) or an ``{"error", ...}`` line; ``{"error": "decode stalled"}`` after 600 s
    without progress. A client gone mid-stream cancels the decode.

Each finished, refused or failed request resets the decoder and releases the receive slot (the rid is tombstoned, so
a late P of it is refused). The HTTP server is the standard library's (threads, HTTP/1.1, chunked streaming), bound
dual-stack on ``::`` as TileRT's uvicorn is.

``python -m atom.models.deepseek_v41.mono.hi.pd.server``: ``--engine tk`` (8 ``tkworker.TkRankWorker`` processes behind
``fanout.RankEngine``; rank 0's worker owns the receive buffer and its Mooncake registration, this process only the
control plane and HTTP; needs ``--toktab-dir`` until the token tables are built here) or ``--engine fake``
(``session.FakeEngine``: the receive path, the KV header check and the router protocol with scripted tokens, to wire a
real P and router to this host before the decode is ready).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import queue
import socket
import socketserver
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import fanout as F
from . import receive as R
from . import session as S
from . import wire as W

logger = logging.getLogger(__name__)

DECODE_POLL_S = (
    max(0.0, float(os.environ.get("TILERT_DECODE_POLL_MS") or "200")) / 1000.0
)
ABANDON_DRAIN_S = 30.0
STALL_S = 600.0
STREAM = "stream"
_CANCELLED = object()


@dataclass
class DecodeRequest:
    """``POST /pd/decode``'s body (TileRT's ``DecodeBody``)."""

    rid: str
    first_token_id: int
    max_tokens: int = 256
    sampling: dict | None = None
    timeout_s: float = 120.0
    stream: bool = False
    grammar_spec: dict | None = None
    enable_thinking: bool = True
    top_logprobs: int | None = None


class BadRequest(Exception):
    """A ``/pd/decode`` body that does not parse (HTTP 422)."""


def _as_int(name: str, v) -> int:
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, int):
        return v
    if isinstance(v, float) and v.is_integer():
        return int(v)
    if isinstance(v, str) and v.strip().lstrip("-").isdigit():
        return int(v)
    raise BadRequest(f"{name}: an integer is required, got {v!r}")


def parse_decode_body(body) -> DecodeRequest:
    """The body as ``DecodeRequest``; ``BadRequest`` (HTTP 422) on a missing or mistyped field."""
    if not isinstance(body, dict):
        raise BadRequest("body: a JSON object is required")
    for f in ("rid", "first_token_id"):
        if f not in body:
            raise BadRequest(f"{f}: field required")
    if not isinstance(body["rid"], str):
        raise BadRequest("rid: a string is required")
    out = DecodeRequest(
        rid=body["rid"],
        first_token_id=_as_int("first_token_id", body["first_token_id"]),
    )
    if body.get("max_tokens") is not None:
        out.max_tokens = _as_int("max_tokens", body["max_tokens"])
    for f in ("sampling", "grammar_spec"):
        v = body.get(f)
        if v is not None and not isinstance(v, dict):
            raise BadRequest(f"{f}: an object is required")
        setattr(out, f, v)
    if body.get("timeout_s") is not None:
        try:
            out.timeout_s = float(body["timeout_s"])
        except (TypeError, ValueError):
            raise BadRequest("timeout_s: a number is required") from None
    for f in ("stream", "enable_thinking"):
        if body.get(f) is not None:
            if not isinstance(body[f], bool):
                raise BadRequest(f"{f}: a boolean is required")
            setattr(out, f, body[f])
    if body.get("top_logprobs") is not None:
        out.top_logprobs = _as_int("top_logprobs", body["top_logprobs"])
    return out


class DecodeServer:
    """The decode node's request handling (module docstring), apart from HTTP: each route returns ``(status,
    payload)``; ``/pd/decode`` with ``stream`` returns ``(STREAM, lines)``, a ``_DecodeStream`` (iterate its line
    objects once, then ``close`` it, which releases the request)."""

    def __init__(
        self,
        receive: R.ReceiveServer,
        decoder: S.Decoder,
        profile: str = W.NAME,
        poll_s: float = DECODE_POLL_S,
        abandon_drain_s: float = ABANDON_DRAIN_S,
        stall_s: float = STALL_S,
    ):
        self.receive, self.decoder, self.profile = receive, decoder, profile
        self.poll_s = max(poll_s, 5e-4)
        self.abandon_drain_s, self.stall_s = abandon_drain_s, stall_s
        self._lock = threading.Lock()
        self.current_rid: str | None = None
        self.cancel_event: threading.Event | None = None
        self._httpd = None

    # ---- simple routes
    def health(self):
        return 200, {"status": "ok"}

    def capabilities(self):
        caps = dict(
            self.decoder.capabilities(), logprobs=self.decoder.supports_logprobs()
        )
        engine = f"Decoder[{getattr(self.decoder.engine, 'name', '?')}]"
        return 200, {"profile": self.profile, "engine": engine, "capabilities": caps}

    def decode_status(self):
        busy = self._lock.locked()
        return 200, {
            "status": "busy" if busy else "idle",
            "current_rid": self.current_rid,
        }

    def cancel(self, body: dict):
        rid, ev = body.get("rid"), self.cancel_event
        if rid and rid == self.current_rid and ev is not None:
            ev.set()
            logger.info("cancel requested for %s", rid)
            return 200, {"cancelled": rid}
        return 404, {
            "error": "no matching in-flight request",
            "current_rid": self.current_rid,
        }

    # ---- the request's lifetime
    def _cleanup(self) -> None:
        try:
            self.decoder.reset()
        except Exception:
            logger.exception("engine reset failed")
        if self.current_rid:
            self.receive.release(self.current_rid)
        self.current_rid = None
        self.cancel_event = None
        self._lock.release()

    def _drain_own_kv(self, rid: str, timeout_s: float, cancel=None):
        deadline = time.time() + timeout_s
        while True:
            if cancel is not None and cancel.is_set():
                return _CANCELLED
            remaining = deadline - time.time()
            if remaining <= 0:
                return None
            try:
                cand = self.receive.completed.get(timeout=min(remaining, 0.5))
            except queue.Empty:
                continue
            if cand.rid == rid:
                return cand
            logger.warning(
                "dropping unmatched request %s (waiting for %s)", cand.rid, rid
            )
            self.receive.release(cand.rid)

    def _abandon_pending_kv(self, rid: str, cancel=None) -> None:
        """A refused request: wait (bounded) for its KV, so P's transfer ends before the slot is freed."""
        try:
            if self._drain_own_kv(rid, self.abandon_drain_s, cancel) is None:
                logger.warning(
                    "abandoned %s: its KV did not arrive within %.0fs",
                    rid,
                    self.abandon_drain_s,
                )
        except Exception:
            logger.exception("draining KV for abandoned %s failed", rid)
        finally:
            self._cleanup()

    def _log_reqstat(
        self, req: DecodeRequest, rec, n_tokens: int, timing: dict
    ) -> None:
        logger.info(
            "REQSTAT rid=%s seq=%d completion=%d %s",
            req.rid,
            rec.seq_len,
            n_tokens,
            " ".join(f"{k}={v}" for k, v in timing.items()),
        )

    def decode(self, body):
        try:
            req = parse_decode_body(body)
        except BadRequest as e:
            return 422, {"detail": str(e)}
        if not self._lock.acquire(blocking=False):
            return 429, {"error": "busy", "current_rid": self.current_rid}
        self.current_rid = req.rid
        self.receive.expect(req.rid)
        cancel = threading.Event()
        self.cancel_event = cancel
        dec = self.decoder
        t0 = time.time()
        if req.top_logprobs is not None and not dec.supports_logprobs():
            self._abandon_pending_kv(req.rid, cancel)
            return 501, {
                "error": "this decode node does not produce logprobs",
                "error_type": "logprobs_unavailable",
            }
        try:
            S.validate_sampling(req.sampling or {}, dec.capabilities())
            dec.check_sampling(req.sampling or {})
            grammar = dec.prepare_grammar(req.grammar_spec, req.enable_thinking)
        except S.RequestRefused as e:
            logger.info("refused %s: %s (%s)", req.rid, e, e.error_type)
            self._abandon_pending_kv(req.rid, cancel)
            return e.http_status, e.to_payload()
        except Exception as e:
            logger.exception("request checks failed for %s", req.rid)
            self._abandon_pending_kv(req.rid, cancel)
            return 500, {"error": str(e)}
        try:
            rec = self._drain_own_kv(req.rid, req.timeout_s, cancel)
            if rec is _CANCELLED:
                self._cleanup()
                return 499, {
                    "error": "cancelled during KV transfer",
                    "error_type": "request_cancelled",
                    "rid": req.rid,
                }
            if rec is None:
                self._cleanup()
                return 504, {"error": "kv_transfer_timeout", "rid": req.rid}
            t_recv = time.time()
            dec.begin(req.first_token_id, req.sampling)
            dec.inject(rec, self.receive.buffer)
            t_inj = time.time()
        except Exception as e:
            logger.exception("prepare failed for %s", req.rid)
            self._cleanup()
            return 500, {"error": str(e), "rid": req.rid}
        pre = {
            "wire_wait": round(1000 * (t_recv - t0), 1),
            "inject": round(1000 * (t_inj - t_recv), 1),
        }
        if req.stream:
            return STREAM, self._stream(req, rec, cancel, grammar, pre, t_inj)
        try:
            tokens = dec.decode(
                first_token_id=req.first_token_id,
                max_tokens=req.max_tokens,
                sampling=req.sampling,
                cancel_event=cancel,
                grammar_session=grammar,
            )
            timing = {
                **pre,
                "decode": round(1000 * (time.time() - t_inj), 1),
                **dec.last_stats,
            }
            self._log_reqstat(req, rec, len(tokens), timing)
            return 200, {
                "rid": req.rid,
                "token_ids": tokens,
                "seq_len": rec.seq_len,
                "timing_ms": timing,
            }
        except S.RequestRefused as e:
            return e.http_status, e.to_payload()
        except Exception as e:
            logger.exception("decode failed for %s", req.rid)
            return 500, {"error": str(e), "rid": req.rid}
        finally:
            self._cleanup()

    def _stream(
        self, req: DecodeRequest, rec, cancel, grammar, pre: dict, t_inj: float
    ):
        return _DecodeStream(self, req, rec, cancel, grammar, pre, t_inj)

    # ---- HTTP
    def serve(self, port: int, host: str = "::") -> None:
        """Serve on ``host:port`` until ``shutdown``."""
        self._httpd = _HTTPServer((host, port), self)
        logger.info(
            "decode server on :%d (profile=%s)",
            self._httpd.server_address[1],
            self.profile,
        )
        self._httpd.serve_forever()

    def start(self, port: int = 0, host: str = "127.0.0.1") -> int:
        """Serve in a thread; the bound port."""
        self._httpd = _HTTPServer((host, port), self)
        threading.Thread(
            target=self._httpd.serve_forever, name="pd-http", daemon=True
        ).start()
        return self._httpd.server_address[1]

    def shutdown(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()


class _DecodeStream:
    """A streamed decode's NDJSON lines (iterate once), its decode on a worker thread started by the iteration.
    ``close`` (always called, idempotent): cancel the decode, join the worker, release the request.
    """

    def __init__(
        self,
        app: DecodeServer,
        req: DecodeRequest,
        rec,
        cancel,
        grammar,
        pre: dict,
        t_inj: float,
    ):
        self.app, self.req, self.rec, self.cancel = app, req, rec, cancel
        self.grammar, self.pre, self.t_inj = grammar, pre, t_inj
        self.q: queue.Queue = queue.Queue()
        self.finished = threading.Event()
        self.worker = threading.Thread(target=self._run, name="pd-decode", daemon=True)
        self._closed = False

    def _run(self) -> None:
        dec, req, q = self.app.decoder, self.req, self.q
        try:
            tokens = dec.decode(
                first_token_id=req.first_token_id,
                max_tokens=req.max_tokens,
                sampling=req.sampling,
                on_token=q.put,
                cancel_event=self.cancel,
                grammar_session=self.grammar,
            )
            q.put(("done", tokens))
        except S.RequestRefused as e:
            q.put(("error", e.to_payload()))
        except Exception as e:
            logger.exception("stream decode failed for %s", req.rid)
            q.put(("error", {"error": str(e)}))
        finally:
            self.finished.set()

    def __iter__(self):
        app, q = self.app, self.q
        self.worker.start()
        done_msg = None
        last = time.time()
        while True:  # the first token goes out alone
            try:
                first = q.get(timeout=0.05)
            except queue.Empty:
                if time.time() - last > app.stall_s:
                    yield {"error": "decode stalled"}
                    return
                continue
            if isinstance(first, int):
                yield {"t": [first]}
            else:
                done_msg = first
            last = time.time()
            break
        while done_msg is None:
            batch, drained = [], False
            while True:
                try:
                    item = q.get_nowait()
                except queue.Empty:
                    break
                drained = True
                if isinstance(item, int):
                    batch.append(item)
                else:
                    done_msg = item
                    break
            if batch:
                yield {"t": batch}
            if done_msg is None:
                if drained:
                    last = time.time()
                elif time.time() - last > app.stall_s:
                    yield {"error": "decode stalled"}
                    return
                else:
                    self.finished.wait(app.poll_s)
        kind, payload = done_msg
        if kind != "done":
            yield payload
            return
        dec = app.decoder
        timing = {
            **self.pre,
            "decode": round(1000 * (time.time() - self.t_inj), 1),
            **dec.last_stats,
        }
        app._log_reqstat(self.req, self.rec, len(payload), timing)
        yield {
            "done": True,
            "n": len(payload),
            "seq_len": self.rec.seq_len,
            "finish_reason": timing.get("finish_reason", "stop"),
            "timing_ms": timing,
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.cancel.set()
        if self.worker.ident is not None:
            self.worker.join(120)
            if self.worker.is_alive():
                logger.error("decode worker failed to stop for %s", self.req.rid)
        self.app._cleanup()


class _HTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, addr, app: DecodeServer):
        self.address_family = socket.AF_INET6 if ":" in addr[0] else socket.AF_INET
        self.app = app
        super().__init__(addr, _Handler)

    def server_bind(
        self,
    ):  # no getfqdn (a reverse lookup) as HTTPServer.server_bind does
        if self.address_family == socket.AF_INET6:
            with contextlib.suppress(OSError):
                self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = (
            str(self.server_address[0]),
            self.server_address[1],
        )


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "atom-pd-decode"

    def log_message(self, fmt, *args) -> None:
        logger.debug("%s " + fmt, self.address_string(), *args)

    def _json(self, status: int, payload) -> None:
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _stream(self, lines) -> None:
        try:
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            for line in lines:
                data = (json.dumps(line) + "\n").encode()
                self.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except (
            OSError
        ) as e:  # the client went away: closing the generator cancels the decode
            logger.info("decode stream client gone: %s", e)
            self.close_connection = True
        finally:
            lines.close()

    def do_GET(self) -> None:
        app = self.server.app
        route = {
            "/health": app.health,
            "/capabilities": app.capabilities,
            "/decode_status": app.decode_status,
        }.get(self.path.split("?", 1)[0])
        if route is None:
            return self._json(404, {"detail": "Not Found"})
        self._json(*route())

    def do_POST(self) -> None:
        app = self.server.app
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n > 0 else b""
        try:
            body = json.loads(raw) if raw else None
        except ValueError:
            return self._json(422, {"detail": "the body is not JSON"})
        path = self.path.split("?", 1)[0]
        if path == "/pd/cancel":
            return self._json(*app.cancel(body if isinstance(body, dict) else {}))
        if path == "/pd/decode":
            out = app.decode(body)
            if out[0] == STREAM:
                return self._stream(out[1])
            return self._json(*out)
        self._json(404, {"detail": "Not Found"})


# ---------------------------------------------------------------------------------------------------------------
# the process (decode_server.main, dsv41_decode_server.main)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="ATOM decode host (D) for DeepSeek-V4.1-Flash behind TileRT's PD router"
    )
    ap.add_argument("--engine", choices=["tk", "fake"], default="tk")
    ap.add_argument("--model", default=W.NAME, help="model profile (dsv41 only)")
    ap.add_argument("--max-seq-len", type=int, default=1 << 20)
    ap.add_argument("--ctrl-port", type=int, default=5556)
    ap.add_argument("--http-port", type=int, default=5557)
    ap.add_argument("--model-weights-dir", default="")
    ap.add_argument("--with-mtp", action="store_true")
    ap.add_argument("--num-mtp", type=int, choices=(3, 5), default=3)
    ap.add_argument("--transport", choices=["mooncake"], default="mooncake")
    ap.add_argument("--kv-cache-dtype", default="fp8_ds_mla")
    ap.add_argument(
        "--pd-buffer-device",
        choices=["cuda", "cpu"],
        default="cuda",
        help="fake engine: cpu = plain host memory (TileRT's pinned huge pages are not ported)",
    )
    ap.add_argument("--co", help="tk: TileRT's code objects (*.co)")
    ap.add_argument("--swap", help="tk: a swap table of FlyDSL builds")
    ap.add_argument(
        "--toktab-dir", help="tk: the token tables, rank<r>.pt (TileRT's g.toktab[r])"
    )
    ap.add_argument(
        "--dist-port", type=int, default=29561, help="tk: the workers' gloo group"
    )
    return ap


def build_engine(args):
    """``(engine, receive server)``: the fake engine with its own buffer, or the rank group whose rank 0 owns it."""
    if args.engine == "fake":
        engine = S.FakeEngine(max_seq_len=args.max_seq_len)
        device = "cpu" if args.pd_buffer_device == "cpu" else "cuda:0"
        return engine, R.open_receive_server(
            args.max_seq_len, args.ctrl_port, args.transport, device=device
        )
    if not (args.model_weights_dir and args.co):
        raise SystemExit("--engine tk needs --model-weights-dir and --co")
    kw = {
        "ckpt": args.model_weights_dir,
        "co": args.co,
        "toktab_dir": args.toktab_dir,
        "swap": args.swap,
        "dist_addr": f"127.0.0.1:{args.dist_port}",
        "mooncake_host": W.local_ip(),
    }
    group = F.RankGroup(
        8, "atom.models.deepseek_v41.mono.hi.pd.tkworker:TkRankWorker", kw
    )
    engine = F.RankEngine(group)
    info = group.reports[0]
    published = R.PublishedTransport(
        info["transport"], {"session_id": info["session_id"]}
    )
    return engine, R.ReceiveServer(
        args.max_seq_len, info["base_ptr"], published, port=args.ctrl_port
    )


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    args = build_parser().parse_args(argv)
    if args.model not in (
        W.NAME,
        "dsv4.1",
        "deepseek_v4_1_flash",
        "ds_v4_1_flash",
        "dsv41flash",
    ):
        raise SystemExit(f"--model {args.model}: this host serves dsv41 only")
    if not args.with_mtp or args.num_mtp not in W.SUPPORTED_NUM_MTP:
        raise SystemExit(
            f"dsv41 decodes at TP8 with --with-mtp --num-mtp 5 (got with_mtp={args.with_mtp}, num_mtp={args.num_mtp})"
        )
    if args.kv_cache_dtype.lower() not in ("fp8_ds_mla", "auto", "fp8"):
        raise SystemExit(
            f"--kv-cache-dtype {args.kv_cache_dtype}: the prefill must run fp8_ds_mla"
        )
    try:
        engine, receive = build_engine(args)
    except (NotImplementedError, F.RankError) as e:
        raise SystemExit(str(e)) from None
    if engine.max_seq_len != args.max_seq_len:
        raise SystemExit(
            f"--max-seq-len {args.max_seq_len}: the engine serves {engine.max_seq_len}"
        )
    server = DecodeServer(receive, S.Decoder(engine, ar_steps=S.AR_STEPS))
    logger.info(
        "decode host: engine=%s, ctrl=:%d, http=:%d",
        engine.name,
        receive.port,
        args.http_port,
    )
    server.serve(args.http_port)


if __name__ == "__main__":
    main()
