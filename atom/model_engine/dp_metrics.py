"""Prometheus exposition of EngineCoreMgr's DP routing statistics."""


def collect_dp_metrics(snapshot):
    from prometheus_client.core import CounterMetricFamily, GaugeMetricFamily

    snapshot = snapshot or {}
    dp_router = snapshot.get("dp_router", {})
    for name, documentation, value in (
        (
            "atom:dp_affinity_new",
            (
                "Number of new sticky DP sessions assigned to a load-aware "
                "cache owner."
            ),
            dp_router.get("affinity_new_total", 0),
        ),
        (
            "atom:dp_affinity_owner_hit",
            "Number of requests routed to an existing session cache owner.",
            dp_router.get("affinity_owner_hit_total", 0),
        ),
        (
            "atom:dp_affinity_spill",
            (
                "Number of existing sessions moved off their cache owner; "
                "strict affinity keeps this zero."
            ),
            dp_router.get("affinity_spill_total", 0),
        ),
        (
            "atom:dp_affinity_parent_ignored",
            (
                "Number of new child sessions independently placed instead of "
                "inheriting a parent owner."
            ),
            dp_router.get("affinity_parent_ignored_total", 0),
        ),
        (
            "atom:dp_route_explicit",
            "Number of requests routed by an explicit data-parallel rank.",
            dp_router.get("explicit_total", 0),
        ),
        (
            "atom:dp_route_load_balanced",
            (
                "Number of sessionless requests routed by the configured load "
                "balancer."
            ),
            dp_router.get("load_balanced_total", 0),
        ),
    ):
        metric = CounterMetricFamily(name, documentation)
        metric.add_metric([], float(value))
        yield metric

    metric = CounterMetricFamily(
        "atom:dp_requests_routed",
        "Cumulative requests routed to each data-parallel rank.",
        labels=["rank"],
    )
    for rank, value in enumerate(dp_router.get("requests_per_rank", [])):
        metric.add_metric([str(rank)], float(value))
    yield metric

    for name, documentation, values in (
        (
            "atom:dp_inflight_requests",
            "Current in-flight requests charged to each data-parallel rank.",
            dp_router.get("inflight_requests_per_rank", []),
        ),
        (
            "atom:dp_queued_prefill_tokens",
            (
                "Current estimated uncached prefill-token debt per "
                "data-parallel rank; later sticky turns charge only positive "
                "prompt growth."
            ),
            dp_router.get("queued_prefill_tokens_per_rank", []),
        ),
        (
            "atom:dp_sessions",
            "Sticky sessions currently owned by each data-parallel rank.",
            dp_router.get("session_count_per_rank", []),
        ),
    ):
        metric = GaugeMetricFamily(name, documentation, labels=["rank"])
        for rank, value in enumerate(values):
            metric.add_metric([str(rank)], float(value))
        yield metric
