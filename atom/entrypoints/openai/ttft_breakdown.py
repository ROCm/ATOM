"""Per-request decode TTFT stage histograms (API process, Path A).

Stages are wall-clock slices that sum (approximately) to
``atom:time_to_first_token_seconds`` for streaming decode requests:

  api_preprocess  — middleware entry → preprocess() return (tokenize, validate)
  api_enqueue     — preprocess done → add_request() return (ZMQ to engine)
  forward_to_output — engine first forward → scheduler first generation emit
  output_to_callback — scheduler emit → stream_callback() in API output thread
  callback_to_sse — stream_callback → first generated SSE payload

``api_preprocess`` is also subdivided (still Path A, same bucket set) so the
decode-node tokenize cost can be inspected without replacing the existing
superset:

  api_body_parse       — middleware entry → chat handler entry (JSON + pydantic)
  api_chat_template    — apply_chat_template (0 when prompt_token_ids reused)
  api_tokenize         — tokenizer.encode (0 when input is already token ids)
  api_preprocess_wait  — run_in_executor wall − tokenize (queue + Sequence build)

``api_detokenize_chunk_seconds`` is per streaming chunk (ITL path), not a TTFT
slice.

Engine-side ``queue_time`` and ``pd_kv_transfer`` are separate scheduler metrics
and are NOT duplicated here.

TODO: wire the same subdivides on ``/v1/completions``, ``/v1/messages``, and
atomesh standalone — currently only ``/v1/chat/completions``.
"""

from __future__ import annotations

# Shared with request_timing middleware (same perf_counter domain).
_TTFT_BUCKETS = (
    0.001,
    0.005,
    0.010,
    0.025,
    0.050,
    0.100,
    0.250,
    0.500,
    1.0,
    2.5,
    5.0,
    10.0,
    15.0,
    30.0,
    45.0,
    60.0,
    90.0,
    120.0,
    180.0,
    240.0,
)

# Detokenize is per-chunk on the ITL path; steady-state should be ≪ 1 ms.
_DETOKENIZE_BUCKETS = (
    0.00005,
    0.0001,
    0.00025,
    0.0005,
    0.001,
    0.002,
    0.005,
    0.010,
    0.025,
    0.050,
    0.100,
    0.250,
    0.500,
    1.0,
)


class TtftBreakdownMetrics:
    def __init__(self, registry):
        from prometheus_client import Histogram

        self._api_preprocess = Histogram(
            "atom:ttft_api_preprocess_seconds",
            "Decode TTFT slice: API middleware entry through preprocess() return.",
            buckets=_TTFT_BUCKETS,
            registry=registry,
        )
        self._api_enqueue = Histogram(
            "atom:ttft_api_enqueue_seconds",
            "Decode TTFT slice: preprocess() return through add_request() return.",
            buckets=_TTFT_BUCKETS,
            registry=registry,
        )
        self._output_to_callback = Histogram(
            "atom:ttft_output_to_callback_seconds",
            "Decode TTFT slice: scheduler first generation emit through API stream_callback.",
            buckets=_TTFT_BUCKETS,
            registry=registry,
        )
        self._callback_to_sse = Histogram(
            "atom:ttft_callback_to_sse_seconds",
            "Decode TTFT slice: first stream_callback through first generated SSE payload.",
            buckets=_TTFT_BUCKETS,
            registry=registry,
        )
        self._api_body_parse = Histogram(
            "atom:api_body_parse_seconds",
            "API middleware entry through /v1/chat/completions handler entry: "
            "JSON body read, pydantic ChatCompletionRequest validation, FastAPI deps. "
            "Subset of ttft_api_preprocess_seconds.",
            buckets=_TTFT_BUCKETS,
            registry=registry,
        )
        self._api_chat_template = Histogram(
            "atom:api_chat_template_seconds",
            "Wall time of apply_chat_template (and custom message encoder). "
            "Observes 0 when the request already carries prompt_token_ids. "
            "Multimodal prepare is not included. Chat completions only.",
            buckets=_TTFT_BUCKETS,
            registry=registry,
        )
        self._api_tokenize = Histogram(
            "atom:api_tokenize_seconds",
            "Wall time of tokenizer.encode for the chat prompt. "
            "Observes 0 when the input is already token ids (PD prompt_token_ids "
            "reuse or pre-tokenized callers). Chat completions only.",
            buckets=_TTFT_BUCKETS,
            registry=registry,
        )
        self._api_preprocess_wait = Histogram(
            "atom:api_preprocess_wait_seconds",
            "run_in_executor wall for preprocess minus tokenizer.encode: "
            "thread-pool queue delay plus Sequence construction. "
            "Chat completions only.",
            buckets=_TTFT_BUCKETS,
            registry=registry,
        )
        self._api_detokenize_chunk = Histogram(
            "atom:api_detokenize_chunk_seconds",
            "Per streaming chunk wall time of IncrementalStreamDetokenizer.update. "
            "ITL delivery path, not a TTFT slice. Chat completions streaming only.",
            buckets=_DETOKENIZE_BUCKETS,
            registry=registry,
        )

    def observe_api_preprocess(self, seconds: float) -> None:
        if seconds >= 0:
            self._api_preprocess.observe(seconds)

    def observe_api_enqueue(self, seconds: float) -> None:
        if seconds >= 0:
            self._api_enqueue.observe(seconds)

    def observe_output_to_callback(self, seconds: float) -> None:
        if seconds >= 0:
            self._output_to_callback.observe(seconds)

    def observe_callback_to_sse(self, seconds: float) -> None:
        if seconds >= 0:
            self._callback_to_sse.observe(seconds)

    def observe_api_body_parse(self, seconds: float) -> None:
        if seconds >= 0:
            self._api_body_parse.observe(seconds)

    def observe_api_chat_template(self, seconds: float) -> None:
        if seconds >= 0:
            self._api_chat_template.observe(seconds)

    def observe_api_tokenize(self, seconds: float) -> None:
        if seconds >= 0:
            self._api_tokenize.observe(seconds)

    def observe_api_preprocess_wait(self, seconds: float) -> None:
        if seconds >= 0:
            self._api_preprocess_wait.observe(seconds)

    def observe_api_detokenize_chunk(self, seconds: float) -> None:
        if seconds >= 0:
            self._api_detokenize_chunk.observe(seconds)


# Two jobs, one map. Dedupe: the engine stamps the same ``scheduler_output_at``
# on every ``RequestOutput`` of a request, so this side cannot tell a first
# chunk from a later one without remembering. Handoff: the engine output thread
# writes the stamp and the request's event loop task reads it for
# ``callback_to_sse``, which a ContextVar cannot do because it is task-local.
# Entries live exactly as long as ``_request_start_times``: added by the first
# callback of a streaming request, dropped by ``discard_callback_tracking``
# from ``cleanup_request``.
_first_callback_at: dict[str, float] = {}


def mark_first_callback(
    request_id: str,
    *,
    scheduler_output_at: float | None,
    callback_at: float,
    callback_perf: float,
    metrics: TtftBreakdownMetrics | None,
    has_tokens: bool,
) -> None:
    if not has_tokens or request_id in _first_callback_at:
        return
    _first_callback_at[request_id] = callback_perf
    if metrics is not None and scheduler_output_at is not None:
        metrics.observe_output_to_callback(callback_at - scheduler_output_at)


def first_callback_perf(request_id: str) -> float | None:
    """Read the output thread's stamp, in this process's ``perf_counter`` domain."""
    return _first_callback_at.get(request_id)


def discard_callback_tracking(request_id: str) -> None:
    """Forget a request once its streams are torn down."""
    _first_callback_at.pop(request_id, None)
