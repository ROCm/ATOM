"""Run the real ingress HTTP implementation; fake only encoder/upstream boundaries.

This deliberately does not establish V4.1 tokenizer equivalence or GPU performance.
Usage: python http_lifecycle_probe.py --source DIR --output FILE
"""

import argparse
import asyncio
import copy
import hashlib
import importlib.util
import json
import logging
import platform
import sys
import threading
import time
from pathlib import Path

logging.disable(logging.CRITICAL)


class Encoder:
    def __init__(self, gate=None, fail=None):
        self.gate = gate
        self.fail = fail
        self.began = threading.Event()
        self.calls = []

    def prepare_body(self, body):
        self.began.set()
        self.calls.append(copy.deepcopy(body))
        if self.gate is not None:
            assert self.gate.wait(3), "probe failed to release native work"
        if self.fail is not None:
            raise self.fail
        out = copy.deepcopy(body)
        out.setdefault("prompt_token_ids", [13, 37])
        return out


class BodyStream:
    def __init__(self, httpx, chunks, fail=None):
        class Stream(httpx.AsyncByteStream):
            closes = 0

            async def __aiter__(self_):
                for chunk in chunks:
                    yield chunk
                    await asyncio.sleep(0)
                if fail is not None:
                    raise fail

            async def aclose(self_):
                self_.closes += 1

        self.stream = Stream()


class Call:
    def __init__(self, app, body=None, raw=None, headers=()):
        self.queue = asyncio.Queue()
        self.queue.put_nowait(
            {
                "type": "http.request",
                "body": raw if raw is not None else json.dumps(body or {}).encode(),
                "more_body": False,
            }
        )
        self.app = app
        self.messages = []
        self.cancel_on_start = False
        self.disconnect_on_body = False
        self.scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/v1/chat/completions",
            "raw_path": b"/v1/chat/completions",
            "root_path": "",
            "query_string": b"",
            "headers": [(b"content-type", b"application/json"), *headers],
            "client": ("127.0.0.1", 10000),
            "server": ("127.0.0.1", 24671),
        }

    async def receive(self):
        return await self.queue.get()

    async def send(self, message):
        self.messages.append(message)
        if self.cancel_on_start and message["type"] == "http.response.start":
            raise asyncio.CancelledError()
        if self.disconnect_on_body and message["type"] == "http.response.body":
            self.queue.put_nowait({"type": "http.disconnect"})

    def disconnect(self):
        self.queue.put_nowait({"type": "http.disconnect"})

    async def run(self):
        await self.app(self.scope, self.receive, self.send)
        return self

    @property
    def status(self):
        return next(
            m["status"] for m in self.messages if m["type"] == "http.response.start"
        )

    @property
    def body(self):
        return b"".join(
            m.get("body", b"")
            for m in self.messages
            if m["type"] == "http.response.body"
        )


async def until(condition, timeout=2):
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "probe condition timed out"
        await asyncio.sleep(0.002)


async def suite(module):
    import fastapi
    import httpx
    import starlette

    results = []

    async def case(name, fn):
        started = time.monotonic()
        try:
            detail = await fn()
            results.append(
                {
                    "id": name,
                    "status": "PASS",
                    "seconds": time.monotonic() - started,
                    "detail": detail,
                }
            )
        # The harness records cancellation failures as well as exceptions.
        except BaseException as exc:  # noqa: BLE001
            results.append(
                {
                    "id": name,
                    "status": "FAIL",
                    "seconds": time.monotonic() - started,
                    "error": repr(exc),
                }
            )

    async def with_app(encoder, handler, fn, **kw):
        app = module.create_app(encoder, "http://fake-upstream", **kw)
        async with app.router.lifespan_context(app):
            await app.state.client.aclose()
            app.state.client = httpx.AsyncClient(
                transport=httpx.MockTransport(handler), timeout=0.1
            )
            try:
                return await fn(app)
            finally:
                if encoder.gate is not None:
                    encoder.gate.set()

    def success(request):
        return httpx.Response(200, stream=BodyStream(httpx, [b'{"ok":true}']).stream)

    async def forwarding():
        streams, sent = [], []

        def handler(request):
            sent.append(request)
            stream = BodyStream(
                httpx, [b"data: ", b'{"id":1}\n\n', b"data: [DONE]\n\n"]
            ).stream
            streams.append(stream)
            return httpx.Response(
                202,
                headers={
                    "content-type": "text/event-stream",
                    "x-request-id": "up-id",
                    "connection": "x-secret",
                    "x-secret": "strip-me",
                    "content-length": "9876",
                },
                stream=stream,
            )

        async def check(app):
            original = {
                "model": "real-model",
                "stream": True,
                "messages": [{"role": "user", "content": "中文"}],
                "prompt_token_ids": [1, 2, 999],
                "tools": [{"type": "function", "function": {"name": "f"}}],
            }
            call = await Call(
                app,
                original,
                headers=[
                    (b"authorization", b"Bearer fake-review-only"),
                    (b"connection", b"x-remove"),
                    (b"x-remove", b"secret"),
                    (b"x-keep", b"yes"),
                    (b"content-length", b"87654"),
                ],
            ).run()
            assert (
                call.status == 202
                and call.body == b'data: {"id":1}\n\ndata: [DONE]\n\n'
            )
            hdr = dict(
                next(
                    m["headers"]
                    for m in call.messages
                    if m["type"] == "http.response.start"
                )
            )
            assert (
                hdr[b"x-request-id"] == b"up-id"
                and b"x-secret" not in hdr
                and b"content-length" not in hdr
                and b"connection" not in hdr
            )
            req = sent[0]
            assert json.loads(req.content) == original
            assert (
                req.headers["authorization"] == "Bearer fake-review-only"
                and req.headers["x-keep"] == "yes"
            )
            assert (
                "x-remove" not in req.headers
                and req.headers["accept-encoding"] == "identity"
            )
            assert (
                int(req.headers["content-length"]) == len(req.content)
                and req.url.path == "/v1/chat/completions"
            )
            assert streams[0].closes == 1 and app.state.capacity._value == 1280
            return {
                "payload_preserved": True,
                "raw_stream_bytes": len(call.body),
                "underlying_stream_closes": streams[0].closes,
            }

        return await with_app(Encoder(), handler, check)

    async def nonstream_and_errors():
        statuses = []
        streams = []
        for status in (200, 400, 429, 500):

            def handler(request, status=status):
                st = BodyStream(httpx, [b'{"error":"same bytes"}']).stream
                streams.append(st)
                return httpx.Response(
                    status, headers={"content-type": "application/json"}, stream=st
                )

            async def check(app, status=status):
                call = await Call(app, {"stream": False}).run()
                assert call.status == status and call.body == b'{"error":"same bytes"}'
                assert streams[-1].closes == 1
                statuses.append(call.status)

            await with_app(Encoder(), handler, check)
        return {"forwarded_statuses": statuses, "all_closed": True}

    async def invalid_body():
        async def check(app):
            for raw in (b"null", b"[]", b"not-json"):
                call = await Call(app, raw=raw).run()
                assert call.status == 400
            assert app.state.capacity._value == 1280
            return {"invalid_forms": 3, "capacity_restored": True}

        return await with_app(Encoder(), success, check)

    async def native_failure():
        encoder = Encoder(fail=ValueError("controlled validation failure"))

        async def check(app):
            call = await Call(app, {}).run()
            assert call.status == 400
            await until(lambda: app.state.capacity._value == 1)
            encoder.fail = None
            again = await Call(app, {}).run()
            assert again.status == 200
            return {"error_status": call.status, "subsequent_status": again.status}

        return await with_app(encoder, success, check, workers=1, queued=1)

    async def native_timeout():
        gate = threading.Event()
        encoder = Encoder(gate)

        async def check(app):
            call = await Call(app, {}).run()
            assert call.status == 408
            assert encoder.began.is_set() and app.state.capacity._value == 0
            gate.set()
            await until(lambda: app.state.capacity._value == 1)
            again = await Call(app, {}).run()
            assert again.status == 200
            return {
                "timeout_status": 408,
                "capacity_while_native_running": 0,
                "capacity_after_native_finishes": 1,
                "subsequent_status": 200,
            }

        return await with_app(
            encoder, success, check, workers=1, queued=1, tokenize_timeout=0.04
        )

    async def native_disconnect():
        gate = threading.Event()
        encoder = Encoder(gate)

        async def check(app):
            call = Call(app, {})
            task = asyncio.create_task(call.run())
            await until(encoder.began.is_set)
            call.disconnect()
            await task
            assert call.status == 499 and app.state.capacity._value == 0
            gate.set()
            await until(lambda: app.state.capacity._value == 1)
            assert (await Call(app, {}).run()).status == 200
            return {
                "disconnect_status": 499,
                "capacity_held_until_native_completion": True,
            }

        return await with_app(encoder, success, check, workers=1, queued=1)

    async def queued_timeout():
        gate = threading.Event()
        encoder = Encoder(gate)

        async def check(app):
            first = Call(app, {})
            first_task = asyncio.create_task(first.run())
            await until(encoder.began.is_set)
            second = await Call(app, {}).run()
            assert second.status == 408
            assert len(encoder.calls) == 1 and app.state.capacity._value == 0
            gate.set()
            await first_task
            await until(lambda: app.state.capacity._value == 1)
            assert first.status == 200
            return {
                "queue_timeout_status": 408,
                "native_submissions": 1,
                "capacity_after_first_completion": 1,
            }

        return await with_app(
            encoder, success, check, workers=1, queued=1, queue_timeout=0.04
        )

    async def queued_disconnect():
        gate = threading.Event()
        encoder = Encoder(gate)

        async def check(app):
            first = Call(app, {})
            t1 = asyncio.create_task(first.run())
            await until(encoder.began.is_set)
            second = Call(app, {})
            t2 = asyncio.create_task(second.run())
            await asyncio.sleep(0.02)
            second.disconnect()
            await t2
            assert (
                second.status == 499
                and app.state.capacity._value == 0
                and len(encoder.calls) == 1
            )
            gate.set()
            await t1
            await until(lambda: app.state.capacity._value == 1)
            return {
                "queued_disconnect_status": 499,
                "no_second_native_submission": True,
                "capacity_after_finish": 1,
            }

        return await with_app(encoder, success, check, workers=1, queued=1)

    async def submit_failure():
        async def check(app):
            app.state.executor.shutdown(wait=True)
            try:
                await Call(app, {}).run()
            except RuntimeError:
                pass
            else:
                raise AssertionError("shutdown executor should propagate RuntimeError")
            assert app.state.capacity._value == 1
            return {"submission_failure_propagates": True, "capacity_restored": 1}

        return await with_app(Encoder(), success, check, workers=1, queued=1)

    async def upstream_failures():
        out = []
        for error in (
            httpx.ConnectError("controlled connect failure"),
            httpx.ReadTimeout("controlled timeout"),
            httpx.PoolTimeout("controlled pool timeout"),
        ):

            def handler(request, error=error):
                raise error

            async def check(app, error=error):
                call = await Call(app, {}).run()
                assert call.status == 502 and app.state.capacity._value == 1280
                out.append(type(error).__name__)

            await with_app(Encoder(), handler, check)
        return {"mapped_to_502": out, "native_capacity_restored": True}

    async def request_deadline():
        cancelled = []

        async def handler(request):
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                cancelled.append(True)
                raise

        async def check(app):
            call = await Call(app, {}).run()
            assert call.status == 502 and cancelled == [True]
            return {"deadline_status": 502, "upstream_send_cancelled": True}

        return await with_app(Encoder(), handler, check, request_timeout=0.04)

    async def upstream_disconnect():
        started = asyncio.Event()
        cancelled = []

        async def handler(request):
            started.set()
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                cancelled.append(True)
                raise

        async def check(app):
            call = Call(app, {})
            task = asyncio.create_task(call.run())
            await started.wait()
            call.disconnect()
            await task
            assert call.status == 499 and cancelled == [True]
            return {"disconnect_status": 499, "send_cancelled": True}

        return await with_app(Encoder(), handler, check)

    async def completed_send_disconnect_race():
        streams = []
        active = []

        def handler(request):
            st = BodyStream(httpx, [b"never send to client"]).stream
            streams.append(st)
            active[0].disconnect()
            return httpx.Response(200, stream=st)

        async def check(app):
            for _ in range(20):
                call = Call(app, {})
                active[:] = [call]
                await call.run()
                assert call.status == 499 and streams[-1].closes == 1
            return {
                "race_repetitions": 20,
                "underlying_stream_closes": [s.closes for s in streams],
            }

        return await with_app(Encoder(), handler, check)

    async def downstream_cancel():
        streams = []

        def handler(request):
            st = BodyStream(httpx, [b"a", b"b"]).stream
            streams.append(st)
            return httpx.Response(200, stream=st)

        async def check(app):
            call = Call(app, {})
            call.cancel_on_start = True
            propagated = False
            try:
                await call.run()
            except asyncio.CancelledError:
                propagated = True
            # Starlette ASGI 2.3 task groups may consume a send cancellation;
            # ownership must still end whether cancellation propagates or not.
            assert streams[0].closes == 1
            assert (await Call(app, {}).run()).status == 200
            return {
                "cancel_propagates": propagated,
                "underlying_stream_closes": streams[0].closes,
                "subsequent_request_succeeds": True,
            }

        return await with_app(Encoder(), handler, check)

    async def externally_cancelled_stream():
        began = asyncio.Event()
        streams = []

        class WaitingStream(httpx.AsyncByteStream):
            closes = 0

            async def __aiter__(self):
                yield b"first"
                began.set()
                await asyncio.sleep(10)

            async def aclose(self):
                self.closes += 1

        def handler(request):
            stream = WaitingStream()
            streams.append(stream)
            return httpx.Response(200, stream=stream)

        async def check(app):
            call = Call(app, {})
            task = asyncio.create_task(call.run())
            await began.wait()
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            else:
                raise AssertionError("external cancellation must propagate")
            assert streams[0].closes == 1 and call.body == b"first"
            return {"external_cancel_propagates": True, "underlying_stream_closes": 1}

        return await with_app(Encoder(), handler, check)

    async def streaming_read_failure():
        streams = []

        def handler(request):
            st = BodyStream(
                httpx, [b"partial"], httpx.ReadError("controlled read failure")
            ).stream
            streams.append(st)
            return httpx.Response(200, stream=st)

        async def check(app):
            call = Call(app, {})
            try:
                await call.run()
            except httpx.ReadError:
                pass
            else:
                raise AssertionError("stream read error should propagate after headers")
            assert (
                streams[0].closes == 1
                and call.status == 200
                and call.body == b"partial"
            )
            return {
                "partial_bytes_preserved": True,
                "error_propagates_after_headers": True,
                "underlying_stream_closes": 1,
            }

        return await with_app(Encoder(), handler, check)

    for fn in (
        forwarding,
        nonstream_and_errors,
        invalid_body,
        native_failure,
        native_timeout,
        native_disconnect,
        queued_timeout,
        queued_disconnect,
        submit_failure,
        upstream_failures,
        request_deadline,
        upstream_disconnect,
        completed_send_disconnect_race,
        downstream_cancel,
        externally_cancelled_stream,
        streaming_read_failure,
    ):
        await case(fn.__name__, fn)
    await asyncio.sleep(0.01)
    pending = [
        repr(t.get_coro())
        for t in asyncio.all_tasks()
        if t is not asyncio.current_task() and not t.done()
    ]
    results.append(
        {
            "id": "no_orphan_async_tasks",
            "status": "PASS" if not pending else "FAIL",
            "detail": pending,
        }
    )
    return {
        "dependencies": {
            "httpx": httpx.__version__,
            "fastapi": fastapi.__version__,
            "starlette": starlette.__version__,
        },
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.source / "exact_tokenizer_ingress.py"
    spec = importlib.util.spec_from_file_location("review_actual_ingress", source)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    data = asyncio.run(suite(mod))
    data.update(
        {
            "source": str(source.resolve()),
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "python": sys.version,
            "platform": platform.platform(),
            "scope": "Actual create_app HTTP lifecycle; controlled encoder and upstream boundaries, no tokenizer/GPU equality claim",
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "source_sha256": data["source_sha256"],
                "passed": sum(r["status"] == "PASS" for r in data["results"]),
                "failed": [r for r in data["results"] if r["status"] == "FAIL"],
                "output": str(args.output),
            },
            indent=2,
        )
    )
    return int(any(r["status"] != "PASS" for r in data["results"]))


if __name__ == "__main__":
    raise SystemExit(main())
