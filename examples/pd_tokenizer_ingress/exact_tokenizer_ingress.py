"""Exact-tokenization ingress. Scheduling, transfer and stream ledger stay in Atomesh."""

import argparse
import asyncio
import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

HOP_HEADERS = {
    "host",
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}


def forward_headers(headers):
    dynamic = {
        name.strip().lower()
        for name in headers.get("connection", "").split(",")
        if name.strip()
    }
    return {
        k: v
        for k, v in headers.items()
        if k.lower() not in HOP_HEADERS | dynamic and k.lower() != "content-length"
    }


class ClientGone(Exception):
    pass


class OwnedStreamingResponse(StreamingResponse):
    def __init__(self, *args, owner, **kwargs):
        super().__init__(*args, **kwargs)
        self.owner = owner

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            await self.owner.aclose()


class DisconnectWatcher:
    def __init__(self, request):
        self.disconnected = False
        self.task = asyncio.create_task(self.watch(request))

    async def watch(self, request):
        while True:
            message = await request.receive()
            if message["type"] == "http.disconnect":
                self.disconnected = True
                return

    async def aclose(self):
        self.task.cancel()
        await asyncio.gather(self.task, return_exceptions=True)


def disconnect_watcher(request):
    watcher = getattr(request.state, "exact_disconnect_watcher", None)
    if watcher is None:
        # Called only after request.json() consumed the HTTP body.
        watcher = DisconnectWatcher(request)
        request.state.exact_disconnect_watcher = watcher
    return watcher


async def is_disconnected(request):
    watcher = disconnect_watcher(request)
    if watcher.task.done():
        watcher.task.result()  # Propagate receive errors into existing cleanup.
    return watcher.disconnected


async def await_connected(task, request, deadline):
    try:
        watcher = disconnect_watcher(request)
        if not task.done():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError()
            await asyncio.wait(
                {task, watcher.task},
                timeout=remaining,
                return_when=asyncio.FIRST_COMPLETED,
            )
        if await is_disconnected(request):
            raise ClientGone()
        if not task.done():
            raise TimeoutError()
        return task.result()
    except BaseException:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        raise


def create_app(
    encoder,
    backend,
    *,
    workers=4,
    queued=1280,
    queue_timeout=30,
    tokenize_timeout=120,
    request_timeout=1800,
    max_connections=2048,
    max_keepalive_connections=512,
):
    if max_connections < 1 or not 0 <= max_keepalive_connections <= max_connections:
        raise ValueError("invalid HTTP connection budget")

    @asynccontextmanager
    async def lifespan(app):
        app.state.executor = ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="exact-chat"
        )
        app.state.capacity = asyncio.Semaphore(queued)
        app.state.client = httpx.AsyncClient(
            timeout=httpx.Timeout(request_timeout, connect=10, pool=60),
            limits=httpx.Limits(
                max_connections=max_connections,
                max_keepalive_connections=max_keepalive_connections,
            ),
        )
        try:
            yield
        finally:
            await app.state.client.aclose()
            app.state.executor.shutdown(wait=True, cancel_futures=True)

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    async def health():
        return {
            "status": "ready",
            "tokenization": "actual-atom-custom-encoder",
            "backend": backend,
            "process_id": os.getpid(),
            "encoder_threads": workers,
            "native_capacity": {
                "total": queued,
                "available": app.state.capacity._value,
            },
            "http_limits": {
                "max_connections": max_connections,
                "max_keepalive_connections": max_keepalive_connections,
            },
        }

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        try:
            return await owned_chat(request)
        finally:
            watcher = getattr(request.state, "exact_disconnect_watcher", None)
            if watcher is not None:
                # StreamingResponse takes over ASGI receive after this return.
                # Never leave two tasks competing for http.disconnect events.
                await watcher.aclose()

    async def owned_chat(request: Request):
        started = time.monotonic()
        timing = {
            "request_id": request.headers.get("x-request-id"),
            "wall_start": time.time(),
        }
        try:
            body = await request.json()
            timing["body_json_s"] = time.monotonic() - started
            if not isinstance(body, dict):
                raise TypeError("request must be a JSON object")

            async def accounted_render():
                capacity_started = time.monotonic()
                await asyncio.wait_for(
                    app.state.capacity.acquire(), timeout=queue_timeout
                )
                timing["capacity_wait_s"] = time.monotonic() - capacity_started
                loop = asyncio.get_running_loop()
                submitted = time.monotonic()

                def measured_prepare():
                    began = time.monotonic()
                    timing["executor_queue_s"] = began - submitted
                    try:
                        return encoder.prepare_body(body)
                    finally:
                        timing["prepare_s"] = time.monotonic() - began

                try:
                    future = loop.run_in_executor(app.state.executor, measured_prepare)
                except BaseException:
                    app.state.capacity.release()
                    raise

                # Native work keeps its slot until actual completion even if the
                # HTTP task is cancelled. Shield prevents cancelling this future.
                def native_done(done):
                    app.state.capacity.release()
                    if not done.cancelled():
                        done.exception()  # Observe late failures after disconnect.

                future.add_done_callback(native_done)
                return await asyncio.shield(future)

            payload = await await_connected(
                asyncio.create_task(accounted_render()),
                request,
                started + tokenize_timeout,
            )
            timing["prepared_at_s"] = time.monotonic() - started
            if await is_disconnected(request):
                raise ClientGone()
        except ClientGone:
            return JSONResponse(
                {"error": {"type": "client_disconnected"}}, status_code=499
            )
        except TimeoutError:
            return JSONResponse(
                {"error": {"type": "tokenization_timeout"}}, status_code=408
            )
        except (ValueError, TypeError) as exc:
            return JSONResponse(
                {"error": {"type": "exact_tokenization_error", "message": str(exc)}},
                status_code=400,
            )
        headers = forward_headers(request.headers)
        headers["accept-encoding"] = "identity"
        upstream_request = app.state.client.build_request(
            "POST",
            backend + "/v1/chat/completions",
            json=payload,
            headers=headers,
        )
        send_task = asyncio.create_task(
            app.state.client.send(upstream_request, stream=True)
        )
        response = None
        try:
            response = await await_connected(
                send_task, request, started + request_timeout
            )
            if await is_disconnected(request):
                raise ClientGone()
        except BaseException as exc:
            # send may have won the cancellation race; close the response even
            # when ownership was never transferred to the streaming body.
            if (
                response is None
                and send_task.done()
                and not send_task.cancelled()
                and send_task.exception() is None
            ):
                response = send_task.result()
            if response is not None:
                await response.aclose()
            if isinstance(exc, ClientGone):
                return JSONResponse(
                    {"error": {"type": "client_disconnected"}}, status_code=499
                )
            if isinstance(exc, (httpx.HTTPError, TimeoutError)):
                return JSONResponse(
                    {"error": {"type": "pd_router_unavailable"}}, status_code=502
                )
            raise

        async def chunks():
            try:
                async for chunk in response.aiter_raw():
                    yield chunk
            finally:
                await response.aclose()

        return OwnedStreamingResponse(
            chunks(),
            owner=response,
            status_code=response.status_code,
            headers=forward_headers(response.headers),
        )

    return app


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--backend", required=True)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=24671)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--timeout-keep-alive", type=int, default=5)
    a = p.parse_args()
    if a.timeout_keep_alive < 0:
        p.error("--timeout-keep-alive must be nonnegative")
    from exact_atom_chat import ExactAtomChatEncoder
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    tok = AutoTokenizer.from_pretrained(a.model, use_fast=True, trust_remote_code=True)
    probe = "Hello world 你好"
    if tok.decode(tok.encode(probe), skip_special_tokens=True) != probe:
        tok = PreTrainedTokenizerFast.from_pretrained(a.model)
    encoder = ExactAtomChatEncoder(a.model, tok, defaults={})
    import uvicorn

    uvicorn.run(
        create_app(encoder, a.backend, workers=a.workers),
        host=a.host,
        port=a.port,
        access_log=False,
        timeout_keep_alive=a.timeout_keep_alive,
    )


if __name__ == "__main__":
    main()
