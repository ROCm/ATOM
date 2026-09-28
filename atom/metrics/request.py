"""Request and streaming latency instruments."""


class RequestMetrics:
    """Request latency instruments registered once for an API instance."""

    def __init__(self, registry):
        from prometheus_client import Histogram

        self._time_to_first_token = Histogram(
            "atom:time_to_first_token_seconds",
            "Local API request arrival to first output. Streaming observes the "
            "first generated SSE payload; non-streaming observes the first "
            "internal token delivery. One sample per request.",
            labelnames=("streaming",),
            buckets=(
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
            ),
            registry=registry,
        )

        # Expose zero-valued children before traffic so Prometheus can establish
        # a baseline for rate(). Registering labels does not record a sample.
        for streaming in ("true", "false"):
            self._time_to_first_token.labels(streaming=streaming)

    def observe_time_to_first_token(self, interval: float, streaming: bool) -> None:
        self._time_to_first_token.labels(streaming=str(streaming).lower()).observe(
            interval
        )


INTER_TOKEN_LATENCY_BUCKETS = (
    0.002,
    0.004,
    0.006,
    0.008,
    0.010,
    0.015,
    0.020,
    0.025,
    0.030,
    0.035,
    0.040,
    0.060,
    0.080,
    0.100,
    0.200,
    0.400,
    0.600,
    0.800,
    1.000,
    2.000,
    4.000,
    6.000,
    8.000,
)


class StreamMetrics:
    """Delivery metrics owned by the API stream dispatcher."""

    def __init__(self, registry):
        from atom.metrics.histogram import WeightedHistogram

        self._inter_token_latency = WeightedHistogram(
            "atom:inter_token_latency_seconds",
            "Frontend-observed streaming output interval divided by new token "
            "count, weighted by that count. Excludes the first output batch.",
            buckets=INTER_TOKEN_LATENCY_BUCKETS,
            registry=registry,
        )

    def observe_inter_token_latency(self, interval: float, num_new_tokens: int) -> None:
        """Record token-weighted intervals in one aggregation update.

        These cumulative observations are independent of the engine snapshot;
        neither refreshing that snapshot nor scraping resets the histogram.
        """
        self._inter_token_latency.observe_weighted(interval, num_new_tokens)
