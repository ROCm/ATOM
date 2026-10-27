"""Read standard Histogram samples for lifecycle assertions."""

from prometheus_client import Histogram


def histogram_values(histogram):
    samples = next(iter(histogram.collect())).samples
    return {
        "buckets": [
            (float(s.labels["le"]), s.value)
            for s in samples
            if s.name.endswith("_bucket")
        ],
        "sum": next(s.value for s in samples if s.name.endswith("_sum")),
    }


def histogram_values_by_name(owner):
    if hasattr(owner, "poll"):
        owner.poll()
    return {
        "pd_kv_transfer" if name == "pd_transfer" else name: histogram_values(value)
        for name, value in vars(owner).items()
        if isinstance(value, Histogram)
    }
