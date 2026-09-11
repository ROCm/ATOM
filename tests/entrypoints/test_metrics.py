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
from types import SimpleNamespace

from prometheus_client import CollectorRegistry, generate_latest

from atom.entrypoints.openai.metrics import AtomMetricsExporter, _gc_metrics


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


def _offload_snapshot(*offload_stats):
    from atom.model_engine.llm_engine import LLMEngine

    engine = SimpleNamespace(
        core_mgr=SimpleNamespace(
            latest_metrics={
                rank: {"enabled": True, "offload": stats}
                for rank, stats in enumerate(offload_stats)
            },
            get_dp_router_statistics=dict,
        )
    )
    return LLMEngine.get_metrics_statistics(engine)


def _offload_samples(exporter):
    from prometheus_client.parser import text_string_to_metric_families

    exposition = exporter.render().decode()
    families = {
        family.name: family
        for family in text_string_to_metric_families(exposition)
        if family.name.startswith("atom:lmcache_")
    }
    values = {
        sample.name: sample.value
        for family in families.values()
        for sample in family.samples
    }
    return families, values


def test_offload_admission_metrics_reach_prometheus_with_correct_dp_aggregation():
    """Exercise the runtime snapshot -> public metric contract, including units.

    Missing aggregator keys previously made valid scheduler counters invisible.
    Capacity counts sum across DP schedulers, while the oldest age must not.
    """
    exporter = AtomMetricsExporter()
    exporter.update(
        _offload_snapshot(
            {
                "save_ops_admitted": 10,
                "save_tokens_admitted": 1024,
                "save_ops_dropped": 2,
                "save_tokens_dropped": 256,
                "save_cancel_requests": 1,
                "save_ops_unretired": 2,
                "save_pending_bytes": 4096,
                "source_blocks_reserved": 12,
                "source_blocks_leased": 5,
                "save_oldest_age_ms": 1750.5,
            },
            {
                "save_ops_admitted": 4,
                "save_tokens_admitted": 512,
                "save_ops_dropped": 3,
                "save_tokens_dropped": 128,
                "save_cancel_requests": 2,
                "save_ops_unretired": 1,
                "save_pending_bytes": 2048,
                "source_blocks_reserved": 4,
                "source_blocks_leased": 2,
                "save_oldest_age_ms": 2500.25,
            },
        )
    )
    families, samples = _offload_samples(exporter)
    expected_counters = {
        "save_ops_admitted": 14,
        "save_tokens_admitted": 1536,
        "save_ops_dropped": 5,
        "save_tokens_dropped": 384,
        "save_cancel_requests": 3,
    }
    expected_gauges = {
        "save_ops_unretired": 3,
        "save_pending_bytes": 6144,
        "source_blocks_reserved": 16,
        "source_blocks_leased": 7,
        "save_oldest_age_seconds": 2.50025,
    }
    for suffix, value in expected_counters.items():
        name = f"atom:lmcache_{suffix}"
        assert families[name].type == "counter"
        assert samples[f"{name}_total"] == value
    for suffix, value in expected_gauges.items():
        name = f"atom:lmcache_{suffix}"
        assert families[name].type == "gauge"
        assert samples[name] == value

    exporter.update(_offload_snapshot({"saved_tokens": 64}, {}))
    _, idle_samples = _offload_samples(exporter)
    for suffix in expected_gauges:
        assert idle_samples[f"atom:lmcache_{suffix}"] == 0


def test_offload_metrics_keep_legacy_snapshots_and_explain_store_counts():
    exporter = AtomMetricsExporter()
    exporter.update(_offload_snapshot({"save_requests": 2, "saved_tokens": 128}))
    families, samples = _offload_samples(exporter)
    assert samples["atom:lmcache_save_requests_total"] == 2
    assert samples["atom:lmcache_saved_tokens_total"] == 128
    assert samples["atom:lmcache_save_ops_admitted_total"] == 0
    assert samples["atom:lmcache_save_oldest_age_seconds"] == 0
    assert "candidate tokens" in families["atom:lmcache_saved_tokens"].documentation
    assert (
        "not actual persisted tokens"
        in families["atom:lmcache_saved_tokens"].documentation
    )
    assert (
        "source is already safe" in families["atom:lmcache_saves_pending"].documentation
    )
