# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""What `/metrics` may cost, which here is a correctness property.

Rendering runs in a thread, but still shares CPU and the GIL with SSE delivery.
Collectors must stay bounded, and their event-loop state must be captured
before rendering. Concurrent or cancelled scrapes must not multiply the work.
"""

from __future__ import annotations

import asyncio
import gc
import threading
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest
from prometheus_client import REGISTRY, CollectorRegistry, Gauge, generate_latest
from prometheus_client.parser import text_string_to_metric_families

from atom.entrypoints.openai.metrics_setup import create_metrics_exporter
from atom.metrics.exporter import AtomMetricsExporter, _gc_metrics


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


def test_state_snapshots_are_private_and_scrapes_do_not_mutate_them():
    exporter = AtomMetricsExporter()
    source = {"enabled": True, "requests_running": 3}
    exporter.update(source)
    source["requests_running"] = 99
    snapshot, _, _ = exporter.read()
    snapshot["requests_running"] = 88
    for _ in range(2):
        assert _samples(exporter.render())[("atom:requests_running", ())] == 3


def test_registration_never_reads_snapshots_or_live_process_metrics(monkeypatch):
    def unexpected_read(*args, **kwargs):
        raise AssertionError("registration must only describe metric names")

    monkeypatch.setattr(AtomMetricsExporter, "read", unexpected_read)
    monkeypatch.setattr(gc, "get_stats", unexpected_read)
    monkeypatch.setattr(gc, "get_threshold", unexpected_read)
    monkeypatch.setattr(
        "atom.entrypoints.openai.metrics_setup.longest_silence_seconds", unexpected_read
    )
    exporter, _, _ = create_metrics_exporter()
    # Optional snapshot families must reserve their names before any samples.
    for name in (
        "atom:requests_running",
        "atom:prefix_cache_offload_tokens_total",
        "atom:dp_requests_routed_total",
        "atom:mtp_decode_steps_total",
        "atom:lmcache_loaded_tokens_total",
        "atom:gc_collections_total",
    ):
        with pytest.raises(ValueError, match="Duplicated timeseries"):
            Gauge(name, "Duplicate", registry=exporter.registry)


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


def test_async_scrapes_keep_live_state_on_loop_and_do_not_cache_responses(monkeypatch):
    owner = threading.get_ident()
    silence = 1.25
    render_threads = []

    def live_silence():
        assert threading.get_ident() == owner
        return silence

    def thread_probe(snapshot):
        if snapshot is not None:
            render_threads.append(threading.get_ident())
        return []

    monkeypatch.setattr(
        "atom.entrypoints.openai.metrics_setup.longest_silence_seconds", live_silence
    )
    exporter, requests, streams = create_metrics_exporter()
    exporter.registry.register(
        type("Probe", (), {"collect": lambda _: thread_probe(exporter.read()[0])})()
    )
    exporter.update({"enabled": True, "requests_running": 3})
    requests.observe_time_to_first_token(0.5, True)
    streams.observe_inter_token_latency(0.020, 4)

    def stable(exposition):
        return {
            key: value
            for key, value in _samples(exposition).items()
            if not key[0].startswith("atom:gc_")
        }

    expected = stable(exporter.render())
    render_threads.clear()

    async def run():
        nonlocal silence
        assert stable(await exporter.render_async()) == expected
        # No sleep or refresh tick: the next GET must see new API samples.
        requests.observe_time_to_first_token(0.25, True)
        streams.observe_inter_token_latency(0.010, 2)
        silence = 0.0
        result = _samples(await exporter.render_async())
        assert result[("atom:stream_longest_silence_seconds", ())] == 0
        assert result[("atom:inter_token_latency_seconds_count", ())] == 6
        assert (
            result[("atom:time_to_first_token_seconds_count", (("streaming", "true"),))]
            == 2
        )
        await exporter.wait_for_render()

    asyncio.run(run())
    assert len(render_threads) == 2
    assert all(worker != owner for worker in render_threads)


@pytest.mark.parametrize("fail", [False, True])
def test_cancelled_scrapes_share_one_render_and_shutdown_drains_it(monkeypatch, fail):
    exporter = AtomMetricsExporter()
    entered, resume = Event(), Event()
    calls = []
    original = exporter.render

    def blocked_render(**kwargs):
        calls.append(1)
        entered.set()
        assert resume.wait(5), "API loop did not resume while rendering"
        if fail:
            raise RuntimeError("render failed")
        return original(**kwargs)

    monkeypatch.setattr(exporter, "render", blocked_render)

    async def run():
        first = asyncio.create_task(exporter.render_async())
        try:
            # Yield the actual event loop while the worker is blocked.
            async def wait_for_worker():
                while not entered.is_set():
                    await asyncio.sleep(0)

            await asyncio.wait_for(wait_for_worker(), 3)
            peers = [asyncio.create_task(exporter.render_async()) for _ in range(8)]
            await asyncio.sleep(0)
            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first
            # Even when all current HTTP waiters disconnect, a later one must
            # join the running thread rather than launch a second render.
            for peer in peers:
                peer.cancel()
            await asyncio.gather(*peers, return_exceptions=True)
            remaining = asyncio.create_task(exporter.render_async())
            drain = asyncio.create_task(exporter.wait_for_render())
            await asyncio.sleep(0)
            assert not drain.done()
            assert calls == [1]
        finally:
            resume.set()
        if fail:
            with pytest.raises(RuntimeError, match="render failed"):
                await remaining
        else:
            assert b"atom:metrics_snapshot_available" in await remaining
        await drain
        monkeypatch.setattr(exporter, "render", original)
        assert b"atom:metrics_snapshot_available" in await exporter.render_async()

    asyncio.run(run())


def test_failed_render_after_all_clients_disconnect_is_retrieved(monkeypatch):
    exporter = AtomMetricsExporter()
    entered, resume = Event(), Event()

    def failed_render(**kwargs):
        entered.set()
        assert resume.wait(5)
        raise RuntimeError("client already gone")

    monkeypatch.setattr(exporter, "render", failed_render)

    async def run():
        failures = []
        asyncio.get_running_loop().set_exception_handler(
            lambda loop, context: failures.append(context)
        )
        waiter = asyncio.create_task(exporter.render_async())
        try:

            async def wait_for_worker():
                while not entered.is_set():
                    await asyncio.sleep(0)

            await asyncio.wait_for(wait_for_worker(), 3)
            waiter.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiter
        finally:
            resume.set()

        async def wait_for_release():
            while exporter._render_task is not None:
                await asyncio.sleep(0)

        await asyncio.wait_for(wait_for_release(), 3)
        gc.collect()
        assert failures == []

    asyncio.run(run())


@pytest.mark.parametrize("operation", ["read", "update"])
def test_snapshot_copy_does_not_hold_publication_lock(monkeypatch, operation):
    import copy

    exporter = AtomMetricsExporter()
    exporter.update({"revision": 1})
    entered, resume = Event(), Event()
    original = copy.deepcopy

    def blocked_copy(value):
        entered.set()
        assert resume.wait(5), "snapshot copy held the publication lock"
        return original(value)

    monkeypatch.setattr("atom.metrics.exporter.copy.deepcopy", blocked_copy)
    with ThreadPoolExecutor(2) as pool:
        operation_future = pool.submit(
            exporter.read
            if operation == "read"
            else lambda: exporter.update({"revision": 2})
        )
        try:
            assert entered.wait(3)
            # Both update and error recording acquire the publication lock.
            pool.submit(exporter.record_refresh_error).result(timeout=2)
        finally:
            resume.set()
        result = operation_future.result(timeout=3)
    monkeypatch.setattr("atom.metrics.exporter.copy.deepcopy", original)
    if operation == "read":
        assert result == ({"revision": 1}, 0, exporter.read()[2])
    assert exporter.read()[1] == 1
