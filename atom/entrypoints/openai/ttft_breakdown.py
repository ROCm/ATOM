"""Request preparation and optional first-output delivery diagnostics.

Preparation is observed on the single-sequence streaming path. Body handling
covers chat handler entry; template/tokenize record only operations that run.
These populations differ, so their means and quantiles must not be summed.
"""

from __future__ import annotations

import math

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


class TtftBreakdownMetrics:
    def __init__(self, registry, *, output_delivery_enabled: bool = False):
        from prometheus_client import Histogram

        self.output_delivery_enabled = output_delivery_enabled
        self._api_preprocess = Histogram(
            "atom:ttft_api_preprocess_seconds",
            "API middleware entry through preprocess() return, for single-sequence streaming requests.",
            buckets=_TTFT_BUCKETS,
            registry=registry,
        )
        self._api_body_parse = Histogram(
            "atom:api_body_parse_seconds",
            "API middleware entry through chat handler entry: body reception, JSON parsing, "
            "pydantic validation, FastAPI dependencies and intervening waits.",
            buckets=_TTFT_BUCKETS,
            registry=registry,
        )
        self._api_chat_template = Histogram(
            "atom:api_chat_template_seconds",
            "Wall time of actual chat template rendering. No observation for reused "
            "prompt token ids or multimodal preparation.",
            buckets=_TTFT_BUCKETS,
            registry=registry,
        )
        self._api_tokenize = Histogram(
            "atom:api_tokenize_seconds",
            "Wall time of actual tokenizer.encode calls in API preprocessing. "
            "No observation for pre-tokenized inputs; includes shared chat/completion paths.",
            buckets=_TTFT_BUCKETS,
            registry=registry,
        )
        self._output_delivery = (
            Histogram(
                "atom:ttft_output_delivery_seconds",
                "Scheduler first generation emit through the first generated SSE payload "
                "in the API generator, including IPC and frontend processing. Single-sequence "
                "streaming only; excludes transport to the client. Requires "
                "ATOM_ENABLE_METRICS_OUTPUT_DELIVERY=1 and synchronized wall clocks across hosts.",
                buckets=_TTFT_BUCKETS,
                registry=registry,
            )
            if output_delivery_enabled
            else None
        )

    def observe_api_preprocess(self, seconds: float) -> None:
        if seconds >= 0:
            self._api_preprocess.observe(seconds)

    def observe_api_body_parse(self, seconds: float) -> None:
        if seconds >= 0:
            self._api_body_parse.observe(seconds)

    def observe_api_chat_template(self, seconds: float) -> None:
        if seconds >= 0:
            self._api_chat_template.observe(seconds)

    def observe_api_tokenize(self, seconds: float) -> None:
        if seconds >= 0:
            self._api_tokenize.observe(seconds)

    def observe_output_delivery(self, seconds: float) -> None:
        if (
            self._output_delivery is not None
            and math.isfinite(seconds)
            and seconds >= 0
        ):
            self._output_delivery.observe(seconds)
