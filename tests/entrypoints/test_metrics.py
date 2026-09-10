# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""What `/metrics` may cost, which here is a correctness property.

Rendering runs inline on the loop that delivers every open SSE stream, so a
metric whose source walks the heap turns the scrape interval into a periodic
inter-token latency spike. These pin the bound, not any value: a slow source is
invisible until someone profiles a scrape.
"""

from __future__ import annotations

import gc
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest
from prometheus_client import REGISTRY, CollectorRegistry, Gauge, generate_latest
from prometheus_client.core import GaugeMetricFamily
from prometheus_client.parser import text_string_to_metric_families

from atom.entrypoints.openai.metrics import AtomMetricsExporter, _gc_metrics
from atom.entrypoints.openai.metrics_setup import create_metrics_exporter


def _render() -> str:
    class _Collector:
        def collect(self):
            yield from _gc_metrics()

    registry = CollectorRegistry()
    registry.register(_Collector())
    return generate_latest(registry).decode()


def _series_names(exposition: str) -> set[str]:
    return {
        line.split("{")[0].split(" ")[0]
        for line in exposition.splitlines()
        if line and not line.startswith("#")
    }


def test_a_scrape_never_walks_the_heap(monkeypatch):
    """The two ways to get this wrong, named so that adding either fails here.

    `atom:gc_frozen_objects` was one of them and had to go. Caching the count
    in `gc_utils` is not the way back: `gc.collect()` moves it without going
    through that module, so any mirror drifts. See `_gc_metrics` for the cost.
    """
    walked: list[str] = []

    def watch(name, result):
        def stub(*_args, **_kwargs):
            walked.append(name)
            return result

        monkeypatch.setattr(gc, name, stub)

    watch("get_freeze_count", 0)
    watch("get_objects", [])

    _render()

    assert walked == [], f"a scrape walked the heap via {walked}"


def test_the_exported_names_are_what_the_docs_tell_operators_to_query():
    """`prometheus_client` appends `_total` to a counter and nothing to a
    gauge, so the name in the source is not the name in a PromQL rule. Every
    one of these is written out in `docs/environment_variables.md`; a rule
    copied from there returning no series is indistinguishable from a healthy
    process, which is the failure this pins.
    """
    assert _series_names(_render()) == {
        "atom:gc_collections_total",
        "atom:gc_collected_total",
        "atom:gc_uncollectable_total",
        "atom:gc_threshold",
    }


def test_every_generation_is_labelled_rather_than_summed():
    """Gen-2 is the stop-the-world one; a total that folded it in with gen-0
    would be dominated by the cheap generation and say nothing."""
    exposition = _render()

    for generation in ("0", "1", "2"):
        assert f'atom:gc_collections_total{{generation="{generation}"}}' in exposition


def _samples(exposition):
    return {
        (sample.name, tuple(sorted(sample.labels.items()))): sample.value
        for family in text_string_to_metric_families(exposition.decode())
        for sample in family.samples
    }


def test_new_component_registers_without_exporter_changes_and_reads_cached_data():
    exporter = AtomMetricsExporter()
    seen = []

    def collect(snapshot):
        seen.append(snapshot)
        yield GaugeMetricFamily(
            "test:component_depth",
            "Component queue depth.",
            value=(snapshot or {}).get("component", {}).get("depth", 0),
        )

    exporter.register_snapshot_collector(collect)
    assert seen == [None]  # Name discovery needs no runtime snapshot.
    source = {"enabled": True, "component": {"depth": 3}}
    exporter.update(source)
    source["component"]["depth"] = 99
    snapshot, _, _ = exporter.read()
    snapshot["component"]["depth"] = 88
    for _ in range(2):
        assert _samples(exporter.render())[("test:component_depth", ())] == 3
    assert exporter.read()[0]["component"]["depth"] == 3
    with pytest.raises(ValueError, match="Duplicated timeseries"):
        exporter.register_snapshot_collector(collect)


@pytest.mark.parametrize(
    "name",
    [
        "atom:request_queue_time_seconds_bucket",
        "atom:request_queue_time_seconds_count",
        "atom:request_queue_time_seconds_sum",
        "atom:prefix_cache_offload_tokens_total",  # Absent on legacy snapshots.
        "atom:time_to_first_token_seconds_count",
        "atom:inter_token_latency_seconds_sum",
    ],
)
def test_registration_reserves_optional_families_and_generated_series(name):
    exporter, _, _ = create_metrics_exporter()
    assert ("atom:prefix_cache_offload_tokens_total", ()) not in _samples(
        exporter.render()
    )
    with pytest.raises(ValueError, match="Duplicated timeseries"):
        Gauge(name, "Conflicting instrument", registry=exporter.registry)


def test_components_do_not_pollute_default_registry_or_other_api_instances():
    def default_names():
        return {metric.name for metric in REGISTRY.collect()}

    before = default_names()
    first, request_metrics, stream_metrics = create_metrics_exporter()
    second, _, _ = create_metrics_exporter()
    request_metrics.observe_time_to_first_token(0.5, True)
    stream_metrics.observe_inter_token_latency(0.020, 4)
    first.update({"enabled": True, "requests_running": 2})
    first.record_refresh_error()
    a, b = _samples(first.render()), _samples(second.render())
    for key, observed in (
        (("atom:time_to_first_token_seconds_count", (("streaming", "true"),)), 1),
        (("atom:inter_token_latency_seconds_count", ()), 4),
        (("atom:requests_running", ()), 2),
        (("atom:metrics_refresh_errors_total", ()), 1),
        (("atom:metrics_snapshot_available", ()), 1),
    ):
        assert a[key] == observed
        assert b[key] == 0
    assert default_names() == before


def test_concurrent_scrapes_pin_one_snapshot_each_while_refresh_continues():
    exporter = AtomMetricsExporter()
    entered, resume = Event(), Event()

    def first(snapshot):
        if snapshot is not None and snapshot["revision"] == 1:
            entered.set()
            assert resume.wait(5), "second scrape did not complete"
        yield GaugeMetricFamily(
            "test:first_revision",
            "First component's revision.",
            value=(snapshot or {}).get("revision", 0),
        )

    def second(snapshot):
        yield GaugeMetricFamily(
            "test:second_revision",
            "Second component's revision.",
            value=(snapshot or {}).get("revision", 0),
        )

    exporter.register_snapshot_collector(first)
    exporter.register_snapshot_collector(second)
    exporter.update({"enabled": True, "revision": 1})
    with ThreadPoolExecutor(max_workers=2) as pool:
        old_scrape = pool.submit(exporter.render)
        try:
            assert entered.wait(5), "first scrape did not start"
            exporter.update({"enabled": False, "revision": 2})
            exporter.record_refresh_error()
            fresh = _samples(pool.submit(exporter.render).result(timeout=5))
        finally:
            resume.set()
        old = _samples(old_scrape.result(timeout=5))
    for name in ("test:first_revision", "test:second_revision"):
        assert old[(name, ())] == 1
        assert fresh[(name, ())] == 2
    assert old[("atom:metrics_snapshot_available", ())] == 1
    assert fresh[("atom:metrics_snapshot_available", ())] == 0
    assert old[("atom:metrics_refresh_errors_total", ())] == 0
    assert fresh[("atom:metrics_refresh_errors_total", ())] == 1


def test_failed_scrape_releases_its_snapshot_context():
    exporter = AtomMetricsExporter()

    def collect(snapshot):
        if snapshot and snapshot["fail"]:
            exporter.update({"fail": False, "revision": 2})
            raise RuntimeError("collector failed")
        yield GaugeMetricFamily(
            "test:revision",
            "Snapshot revision.",
            value=(snapshot or {}).get("revision", 0),
        )

    exporter.register_snapshot_collector(collect)
    exporter.update({"fail": True, "revision": 1})
    with pytest.raises(RuntimeError, match="collector failed"):
        exporter.render()
    assert exporter.read()[0] == {"fail": False, "revision": 2}
    assert _samples(exporter.render())[("test:revision", ())] == 2
