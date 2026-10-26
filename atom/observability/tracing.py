"""Opt-in, bounded request tracing for the standalone aggregate engine.

Only primitives cross the request socket. The scheduler records monotonic
boundaries and exports completed spans once per sequence, never per token.
No GPU operations, prompt text, generated text, or exception messages are read.
"""

from __future__ import annotations

import atexit
import logging
import os
import threading
import time
from contextvars import ContextVar
from dataclasses import dataclass

from atom.utils import envs

logger = logging.getLogger("atom")
_ENABLED = envs.ATOM_ENABLE_OTEL_TRACE
_provider = None
_cached_tracer = None
_provider_pid = None
_init_failed_pid = None
_lock = threading.Lock()
_warned = False
_request: ContextVar = ContextVar("atom_otel_request", default=None)


def _after_fork():
    global _provider, _cached_tracer, _provider_pid, _lock, _init_failed_pid
    _provider = None
    _cached_tracer = None
    _provider_pid = None
    _init_failed_pid = None
    _lock = threading.Lock()
    _request.set(None)


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork)


def _warn_once():
    global _warned
    if not _warned:
        _warned = True
        # Do not log exception text: exporter URLs can contain credentials.
        logger.warning("ATOM trace operation failed; inference continues")


def initialize():
    """Build process-local SDK state at startup, before accepting traffic."""
    global _provider, _cached_tracer, _provider_pid, _init_failed_pid
    pid = os.getpid()
    if not _ENABLED or _init_failed_pid == pid:
        return
    if _provider_pid != pid:
        with _lock:
            if _provider_pid != pid:
                try:
                    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                        OTLPSpanExporter,
                    )
                    from opentelemetry.sdk.resources import Resource
                    from opentelemetry.sdk.trace import TracerProvider
                    from opentelemetry.sdk.trace.export import BatchSpanProcessor
                    from opentelemetry.sdk.trace.sampling import ALWAYS_ON

                    endpoint = os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
                    if not endpoint:
                        raise ValueError("Explicit HTTP trace endpoint required")
                    provider = TracerProvider(
                        # Full engine spans feed spanmetrics BEFORE collector
                        # tail sampling, including upstream sampled=0 requests.
                        sampler=ALWAYS_ON,
                        resource=Resource.create(
                            {
                                "service.name": os.getenv("OTEL_SERVICE_NAME", "atom"),
                                "service.instance.id": f"{os.uname().nodename}:{pid}",
                                "atom.version": envs.ATOM_TRACE_VERSION,
                            }
                        ),
                        shutdown_on_exit=False,
                    )
                    provider.add_span_processor(
                        BatchSpanProcessor(
                            OTLPSpanExporter(endpoint=endpoint, timeout=1),
                            max_queue_size=8192,
                            max_export_batch_size=512,
                            schedule_delay_millis=1000,
                        )
                    )
                    _cached_tracer = provider.get_tracer("atom.standalone")
                    _provider = provider
                    _provider_pid = pid
                    atexit.register(shutdown)
                except Exception:  # noqa: BLE001 - telemetry must not fail inference
                    _init_failed_pid = pid
                    _warn_once()
                    return


def _tracer():
    # No imports, construction or locks on the event loop / scheduler path.
    if _ENABLED and _provider_pid == os.getpid():
        return _cached_tracer
    return None


def shutdown():
    """Flush only this process's provider, off any inference request path."""
    global _provider, _cached_tracer, _provider_pid
    if _provider is not None and _provider_pid == os.getpid():
        provider = _provider
        _provider = _cached_tracer = _provider_pid = None
        try:
            try:
                provider.force_flush(timeout_millis=2000)
            finally:
                provider.shutdown()
        except Exception:  # noqa: BLE001 - best-effort shutdown
            _warn_once()


@dataclass
class SequenceTiming:
    # Pure values: safe to pickle. Never an SDK span, tracer, lock or context.
    epoch_ns: int
    monotonic_ns: int
    forward_ns: int | None = None
    prefill_end_ns: int | None = None
    first_token_ns: int | None = None
    decode_ns: int | None = None
    done: bool = False

    def now(self):
        return self.epoch_ns + (time.perf_counter_ns() - self.monotonic_ns)


def bind_sequences(seqs, *, unsupported=None):
    """Runs on the API event-loop thread, before CoreManager serializes ADD."""
    if not _ENABLED:
        return
    request = _request.get()
    if request is None:
        return
    for seq in seqs:
        try:
            # P/D needs distinct span identities and transfer boundaries first.
            transfer = getattr(seq, "kv_transfer_params", None) or {}
            if (
                unsupported
                or transfer.get("do_remote_prefill")
                or transfer.get("do_remote_decode")
            ):
                request["span"].set_attribute(
                    "atom.trace.unsupported", unsupported or "pd"
                )
                _unbind(seq)
                continue
            seq._atom_trace_carrier = dict(request["carrier"])
            seq._atom_trace_model = request["model"]
            seq._atom_trace_request_id = request.get("request_id")
            seq._atom_trace_timing = None
        except Exception:  # noqa: BLE001 - an untraced request still runs
            _unbind(seq)
            _warn_once()


def _unbind(seq):
    # Also safe if an already-bound sequence is passed through again.
    seq.__dict__.pop("_atom_trace_carrier", None)
    seq.__dict__.pop("_atom_trace_timing", None)


def set_request_id(request_id):
    if not _ENABLED:
        return
    request = _request.get()
    if request is not None:
        request["request_id"] = str(request_id)
        try:
            request["span"].set_attribute("gen_ai.request.id", str(request_id))
        except Exception:  # noqa: BLE001 - request metadata is best effort
            _warn_once()


def enqueue(seq, *, received_at=None):
    if not _ENABLED or not getattr(seq, "_atom_trace_carrier", None):
        return
    if getattr(seq, "_atom_trace_timing", None) is not None:
        return
    try:
        mono = time.perf_counter_ns()
        epoch = time.time_ns()
        received = mono if received_at is None else int(received_at * 1e9)
        seq._atom_trace_timing = SequenceTiming(epoch - (mono - received), received)
    except Exception:  # noqa: BLE001 - telemetry must not fail admission
        _unbind(seq)
        _warn_once()


def enqueue_many(seqs, *, received_at=None):
    if _ENABLED:
        for seq in seqs:
            enqueue(seq, received_at=received_at)


def record_forward(batch, seqs):
    """Observe real dispatch independently of Prometheus metrics."""
    if not _ENABLED:
        return
    try:
        if batch.is_dummy_run:
            return
        for i, req_id in enumerate(batch.req_ids):
            forward(seqs[req_id], is_decode=i < batch.total_seqs_num_decode)
    except Exception:  # noqa: BLE001 - telemetry must not fail a forward
        _warn_once()


def forward(seq, *, is_decode):
    if not _ENABLED:
        return
    try:
        timing = getattr(seq, "_atom_trace_timing", None)
        if timing is None or timing.done:
            return
        if timing.forward_ns is None:
            timing.forward_ns = timing.now()
            if is_decode:
                timing.decode_ns = timing.forward_ns
    except Exception:  # noqa: BLE001 - telemetry must not fail a forward
        _unbind(seq)
        _warn_once()


def prefill_complete(seq):
    if not _ENABLED:
        return
    try:
        timing = getattr(seq, "_atom_trace_timing", None)
        if (
            timing is not None
            and timing.forward_ns is not None
            and timing.prefill_end_ns is None
        ):
            timing.prefill_end_ns = timing.now()
    except Exception:  # noqa: BLE001 - telemetry must not fail postprocess
        _unbind(seq)
        _warn_once()


def first_token(seq):
    if not _ENABLED:
        return
    try:
        timing = getattr(seq, "_atom_trace_timing", None)
        if timing is not None and timing.first_token_ns is None:
            timing.first_token_ns = timing.now()
    except Exception:  # noqa: BLE001 - telemetry must not fail postprocess
        _unbind(seq)
        _warn_once()


def _completed(tracer, name, parent, start, end, attributes, failed):
    from opentelemetry.trace import Status, StatusCode

    span = tracer.start_span(
        name, context=parent, start_time=start, attributes=attributes
    )
    if failed:
        span.set_status(Status(StatusCode.ERROR))
    span.end(end_time=max(start, end))
    return span


def _phases(timing, end, completion_tokens, failed):
    queue_end = timing.forward_ns if timing.forward_ns is not None else end
    phases = [
        ("prefill_waiting", timing.epoch_ns, queue_end, timing.forward_ns is None)
    ]
    if timing.forward_ns is not None and timing.decode_ns is None:
        prefill_end = (
            timing.prefill_end_ns if timing.prefill_end_ns is not None else end
        )
        phases.append(
            (
                "prefill_forward",
                timing.forward_ns,
                prefill_end,
                timing.prefill_end_ns is None,
            )
        )
    decode_start = (
        timing.decode_ns if timing.decode_ns is not None else timing.prefill_end_ns
    )
    if (
        decode_start is not None
        and decode_start < end
        and (completion_tokens > 1 or failed)
    ):
        phases.append(("decode_forward", decode_start, end, True))
    return phases


_PHASE_LATENCY = {
    "prefill_waiting": "gen_ai.latency.time_in_queue",
    "prefill_forward": "gen_ai.latency.time_in_model_prefill",
    "decode_forward": "gen_ai.latency.time_in_model_decode",
}


def finish(seq, *, reason, num_tokens=None):
    if not _ENABLED:
        return
    try:
        timing = getattr(seq, "_atom_trace_timing", None)
        if timing is None or timing.done:
            return
        timing.done = True
        end = timing.now()
        tracer = _tracer()
        if tracer is None:
            return
        from opentelemetry.propagators.textmap import DefaultGetter
        from opentelemetry.trace import set_span_in_context
        from opentelemetry.trace.propagation.tracecontext import (
            TraceContextTextMapPropagator,
        )

        context = TraceContextTextMapPropagator().extract(
            seq._atom_trace_carrier, getter=DefaultGetter()
        )
        prompt_tokens = int(seq.num_prompt_tokens)
        finalized = (
            num_tokens
            if num_tokens is not None
            else getattr(seq, "num_finalized_tokens", prompt_tokens)
        )
        request_id = (
            getattr(seq, "_atom_trace_request_id", None)
            or getattr(seq, "external_request_id", None)
            or getattr(seq, "parent_request_id", None)
            or str(seq.id)
        )
        failed = reason not in {
            "stop",
            "length",
            "eos",
            "max_tokens",
        } and not reason.startswith("stop_")
        normalized_reason = (
            "stop" if reason.startswith("stop_") else reason.split(":", 1)[0]
        )
        attributes = {
            "gen_ai.request.id": str(request_id),
            "atom.sequence.id": str(seq.id),
            "gen_ai.response.model": getattr(seq, "_atom_trace_model", "unknown"),
            "gen_ai.response.finish_reasons": [normalized_reason],
            "gen_ai.usage.prompt_tokens": prompt_tokens,
            "gen_ai.usage.completion_tokens": max(0, int(finalized) - prompt_tokens),
            "gen_ai.usage.cached_tokens": int(
                getattr(seq, "prefix_cache_hit_tokens", 0)
            ),
            "gen_ai.latency.e2e": (end - timing.epoch_ns) / 1e9,
        }
        if failed:
            attributes["error.type"] = reason.split(":", 1)[0]
        phases = _phases(timing, end, max(0, int(finalized) - prompt_tokens), failed)
        for name, start_ns, end_ns, _ in phases:
            attributes[_PHASE_LATENCY[name]] = (end_ns - start_ns) / 1e9
        if timing.first_token_ns is not None:
            attributes["gen_ai.latency.time_to_first_token"] = (
                timing.first_token_ns - timing.epoch_ns
            ) / 1e9
        # Completed spans preserve the actual boundaries while avoiding live
        # spans across scheduler iterations, preemption, and request pickling.
        root = tracer.start_span(
            "engine.request",
            context=context,
            start_time=timing.epoch_ns,
            attributes=attributes,
        )
        parent = set_span_in_context(root)
        # Phase spans carry correlation only. Request usage and aggregate
        # latencies belong to engine.request, so collectors count them once.
        phase_attributes = {
            key: attributes[key]
            for key in (
                "gen_ai.request.id",
                "atom.sequence.id",
                "gen_ai.response.model",
            )
        }
        for name, start_ns, end_ns, active in phases:
            phase_failed = failed and active
            _completed(
                tracer,
                name,
                parent,
                start_ns,
                end_ns,
                (
                    {**phase_attributes, "error.type": normalized_reason}
                    if phase_failed
                    else phase_attributes
                ),
                phase_failed,
            )
        if failed:
            from opentelemetry.trace import Status, StatusCode

            root.set_status(Status(StatusCode.ERROR))
        root.end(end_time=end)
    except Exception:  # noqa: BLE001 - telemetry must not fail inference
        _unbind(seq)
        _warn_once()


class TraceMiddleware:
    """Request lifetime includes SSE body delivery and disconnect teardown."""

    def __init__(self, app, model=lambda: "unknown"):
        self.app = app
        self.model = model

    async def __call__(self, scope, receive, send):
        if (
            not _ENABLED
            or scope["type"] != "http"
            or scope.get("method") != "POST"
            or scope.get("path")
            not in {
                "/v1/chat/completions",
                "/v1/completions",
                "/v1/messages",
                "/v1/responses",
            }
        ):
            await self.app(scope, receive, send)
            return
        span = None
        token = None
        response_complete = False
        try:
            tracer = _tracer()
            if tracer is not None:
                from opentelemetry.trace import SpanKind, set_span_in_context
                from opentelemetry.trace.propagation.tracecontext import (
                    TraceContextTextMapPropagator,
                )

                propagator = TraceContextTextMapPropagator()
                headers = {}
                for k, v in scope.get("headers", []):
                    key = k.decode("latin1").lower()
                    if key not in {"traceparent", "tracestate"}:
                        continue
                    value = v.decode("latin1")
                    # W3C: repeated tracestate lines form one combined list.
                    if key == "tracestate" and key in headers:
                        value = f"{headers[key]},{value}"
                    headers[key] = value
                model = str(self.model() or "unknown")
                span = tracer.start_span(
                    "atom.request",
                    context=propagator.extract(headers),
                    kind=SpanKind.SERVER,
                    attributes={
                        "http.request.method": "POST",
                        "http.route": scope["path"],
                        "gen_ai.response.model": model,
                    },
                )
                carrier = {}
                propagator.inject(carrier, context=set_span_in_context(span))
                token = _request.set({"span": span, "carrier": carrier, "model": model})
        except Exception:  # noqa: BLE001 - telemetry must not fail HTTP handling
            _warn_once()

        async def traced_send(message):
            nonlocal response_complete
            if span is not None and message["type"] == "http.response.start":
                try:
                    span.set_attribute("http.response.status_code", message["status"])
                    # OTel HTTP semconv: a SERVER span leaves 4xx unset.
                    if message["status"] >= 500:
                        from opentelemetry.trace import Status, StatusCode

                        span.set_status(Status(StatusCode.ERROR))
                except Exception:  # noqa: BLE001 - preserve response delivery
                    _warn_once()
            # Starlette can notify its receive listener of disconnect after
            # the final body. That is normal completion, not a client abort.
            # Set this before send: a disconnect may race with its await.
            if message["type"] == "http.response.body" and not message.get(
                "more_body", False
            ):
                response_complete = True
            await send(message)

        async def traced_receive():
            message = await receive()
            if (
                span is not None
                and message["type"] == "http.disconnect"
                and not response_complete
            ):
                _mark_error(span, "client_disconnect")
            return message

        try:
            await self.app(scope, traced_receive, traced_send)
        except BaseException as exc:
            if span is not None:
                _mark_error(span, type(exc).__name__)
            raise
        finally:
            if token is not None:
                _request.reset(token)
            if span is not None:
                try:
                    span.end()
                except Exception:  # noqa: BLE001 - never mask the app's outcome
                    _warn_once()


def _mark_error(span, error_type):
    try:
        from opentelemetry.trace import Status, StatusCode

        span.set_attribute("error.type", error_type)
        span.set_status(Status(StatusCode.ERROR))
    except Exception:  # noqa: BLE001 - never mask the app's outcome
        _warn_once()
