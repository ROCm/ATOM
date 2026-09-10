"""Prometheus exposition of the offload connector's cumulative statistics."""


def collect_offload_metrics(snapshot):
    from prometheus_client.core import CounterMetricFamily, GaugeMetricFamily

    snapshot = snapshot or {}
    offload = snapshot.get("offload", {})
    for name, documentation, value in (
        (
            "atom:lmcache_load_requests",
            "Number of completed LMCache load operations.",
            offload.get("load_requests", 0),
        ),
        (
            "atom:lmcache_loaded_tokens",
            "Number of tokens loaded from LMCache.",
            offload.get("loaded_tokens", 0),
        ),
        (
            "atom:lmcache_load_failures",
            "Number of failed LMCache load operations.",
            offload.get("load_failures", 0),
        ),
        (
            "atom:lmcache_save_requests",
            "Number of completed LMCache save operations.",
            offload.get("save_requests", 0),
        ),
        (
            "atom:lmcache_saved_tokens",
            "Number of tokens saved to LMCache.",
            offload.get("saved_tokens", 0),
        ),
    ):
        metric = CounterMetricFamily(name, documentation)
        metric.add_metric([], float(value))
        yield metric

    for name, documentation, value in (
        (
            "atom:lmcache_loads_pending",
            "Number of LMCache loads currently in flight.",
            offload.get("loads_pending", 0),
        ),
        (
            "atom:lmcache_saves_pending",
            "Number of LMCache saves currently in flight.",
            offload.get("saves_pending", 0),
        ),
    ):
        metric = GaugeMetricFamily(name, documentation)
        metric.add_metric([], float(value))
        yield metric
