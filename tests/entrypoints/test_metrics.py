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
from prometheus_client.parser import text_string_to_metric_families

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


def test_lmcache_save_admission_and_budget_metrics_are_exported():
    exporter = AtomMetricsExporter()
    exporter.update(
        {
            "enabled": True,
            "offload": {
                "save_admitted": 7,
                "save_candidates": 3,
                "save_candidates_finished": 1,
                "save_committed": 2,
                "save_dropped_capacity": 4,
                "save_dropped_tokens_capacity": 4096,
                "save_pin_budget_blocks": 100,
                "save_reserved_blocks": 20,
                "save_pinned_blocks": 12,
                "save_pinned_ratio": 0.32,
                "save_budget_available_blocks": 68,
                "save_budget_rejected": 5,
                "save_budget_rejected_blocks": 40,
                "save_budget_evicted": 2,
                "save_budget_evicted_blocks": 16,
                "save_oversized": 1,
                "save_pinned_tokens": 3072,
                "deferred_free_requests": 2,
            },
        }
    )

    samples = [
        sample
        for family in text_string_to_metric_families(exporter.render().decode())
        for sample in family.samples
        if sample.name.startswith("atom:lmcache_save_")
        or sample.name == "atom:lmcache_deferred_free_requests"
    ]
    by_name_and_labels = {
        (sample.name, tuple(sorted(sample.labels.items()))): sample.value
        for sample in samples
    }
    assert by_name_and_labels[("atom:lmcache_save_admitted_total", ())] == 7
    assert by_name_and_labels[("atom:lmcache_save_candidates", ())] == 3
    assert by_name_and_labels[("atom:lmcache_save_committed", ())] == 2
    assert (
        by_name_and_labels[
            ("atom:lmcache_save_dropped_total", (("reason", "capacity"),))
        ]
        == 4
    )
    assert (
        by_name_and_labels[
            (
                "atom:lmcache_save_dropped_tokens_total",
                (("reason", "capacity"),),
            )
        ]
        == 4096
    )
    assert by_name_and_labels[("atom:lmcache_save_pinned_blocks", ())] == 12
    assert by_name_and_labels[("atom:lmcache_save_reserved_blocks", ())] == 20
    assert by_name_and_labels[("atom:lmcache_save_pin_budget_blocks", ())] == 100
    assert by_name_and_labels[("atom:lmcache_save_pinned_ratio", ())] == 0.32
    assert by_name_and_labels[("atom:lmcache_save_budget_rejected_total", ())] == 5
    assert by_name_and_labels[("atom:lmcache_save_budget_evicted_total", ())] == 2
    assert by_name_and_labels[("atom:lmcache_deferred_free_requests", ())] == 2


def test_lmcache_save_budget_aggregation_sums_dp_pools_but_uses_hottest_ratio():
    from atom.model_engine.llm_engine import LLMEngine

    engine = LLMEngine.__new__(LLMEngine)
    engine.core_mgr = SimpleNamespace(
        latest_metrics={
            0: {
                "enabled": True,
                "offload": {
                    "save_pin_budget_blocks": 20,
                    "save_reserved_blocks": 4,
                    "save_pinned_blocks": 2,
                    "save_budget_available_blocks": 14,
                    "save_pinned_ratio": 0.06,
                },
            },
            1: {
                "enabled": True,
                "offload": {
                    "save_pin_budget_blocks": 20,
                    "save_reserved_blocks": 6,
                    "save_pinned_blocks": 3,
                    "save_budget_available_blocks": 11,
                    "save_pinned_ratio": 0.09,
                },
            },
        },
        get_dp_router_statistics=dict,
    )

    offload = engine.get_metrics_statistics()["offload"]
    assert offload["save_pin_budget_blocks"] == 40
    assert offload["save_reserved_blocks"] == 10
    assert offload["save_pinned_blocks"] == 5
    assert offload["save_budget_available_blocks"] == 25
    assert offload["save_pinned_ratio"] == 0.09
