"""Weighted observations preserve the standard histogram's public behavior."""

import pytest
from prometheus_client import CollectorRegistry, Histogram, generate_latest
from prometheus_client.parser import text_string_to_metric_families

from atom.utils.histogram import WeightedHistogram


def samples(registry):
    return {
        (sample.name, tuple(sorted(sample.labels.items()))): sample.value
        for family in text_string_to_metric_families(generate_latest(registry).decode())
        for sample in family.samples
        if not sample.name.endswith("_created")
    }


def test_weighted_and_ordinary_observations_share_standard_labels_and_buckets():
    weighted_registry, reference_registry = CollectorRegistry(), CollectorRegistry()
    kwargs = {
        "name": "test_latency",
        "documentation": "Latency",
        "labelnames": ("phase",),
        "buckets": (0, 0.005, 0.020),
    }
    weighted = WeightedHistogram(**kwargs, registry=weighted_registry)
    reference = Histogram(**kwargs, registry=reference_registry)
    for phase in ("prefill", "decode"):
        target, expected = weighted.labels(phase), reference.labels(phase)
        for interval, count in ((0, 2), (0.020, 4), (0.06, 3), (0.125, 2)):
            target.observe_weighted(interval, count)
            for _ in range(count):
                expected.observe(interval / count)
        target.observe(0.004)
        expected.observe(0.004)
        target.observe_weighted(100, 0)
        target.observe_weighted(100, -1)
    assert samples(weighted_registry) == pytest.approx(samples(reference_registry))
    with pytest.raises(ValueError, match="label"):
        weighted.observe_weighted(0.02, 4)
    with pytest.raises(ValueError, match="Duplicated timeseries"):
        WeightedHistogram(**kwargs, registry=weighted_registry)
