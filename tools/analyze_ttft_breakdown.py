#!/usr/bin/env python3
"""Print decode TTFT stage histograms from Prometheus /metrics text."""

from __future__ import annotations

import argparse
import re
import sys
import urllib.request


def _fetch(url: str) -> str:
    with urllib.request.urlopen(url, timeout=30) as resp:
        return resp.read().decode()


def _parse_histograms(text: str) -> dict[str, dict[str, float]]:
    """Return {metric_name: {bucket_le: count}} for cumulative histograms."""
    out: dict[str, dict[str, float]] = {}
    for line in text.splitlines():
        if line.startswith("#") or "_bucket{" not in line:
            continue
        if (
            line.startswith("atom:time_to_first_token_seconds_")
            and 'streaming="true"' not in line
        ):
            continue
        m = re.search(r"^(\S+)_bucket\{.*le=\"([^\"]+)\".*\} (\S+)", line)
        if not m:
            continue
        name, le = m.group(1), m.group(2)
        # Buckets are summed across every label set (dp_rank, engine_role, ...).
        by_le = out.setdefault(name, {})
        by_le[le] = by_le.get(le, 0.0) + float(m.group(3))
    return out


def _parse_sum_count(text: str) -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        if (
            line.startswith("atom:time_to_first_token_seconds_")
            and 'streaming="true"' not in line
        ):
            continue
        m = re.search(r"^(\S+)_(sum|count)(?:\{[^}]*\})?\s+(\S+)", line)
        if not m:
            continue
        name, kind, val = m.group(1), m.group(2), float(m.group(3))
        s, c = out.get(name, (0.0, 0.0))
        if kind == "sum":
            s += val
        else:
            c += val
        out[name] = (s, c)
    return out


def _median_from_buckets(buckets: dict[str, float], total: float) -> float | None:
    if total <= 0:
        return None
    target = total / 2.0
    items = sorted(
        buckets.items(), key=lambda kv: float(kv[0]) if kv[0] != "+Inf" else 1e18
    )
    prev = 0.0
    prev_le = 0.0
    for le, cum in items:
        if cum >= target:
            upper = float(le) if le != "+Inf" else prev_le
            if le == "+Inf" or cum == prev:
                return upper
            frac = (target - prev) / (cum - prev)
            return prev_le + frac * (upper - prev_le)
        prev = cum
        prev_le = float(le) if le != "+Inf" else prev_le
    return None


def _row(
    name: str,
    label: str,
    sums: dict[str, tuple[float, float]],
    buckets: dict[str, dict[str, float]],
) -> float:
    """Print one stage and return its mean in seconds (0.0 when unsampled)."""
    s, c = sums.get(name, (0.0, 0.0))
    med = _median_from_buckets(buckets.get(name, {}), c)
    mean = s / c if c else 0.0
    med_s = f"{med * 1000:.1f}ms" if med is not None else "n/a"
    mean_s = f"{mean * 1000:.1f}ms" if c else "n/a"
    print(f"  {label:<28} count={int(c):>5}  p50≈{med_s:>8}  mean={mean_s:>8}")
    return mean


# Different populations and uncovered boundaries prevent an additive identity.
STAGES = (
    ("atom:ttft_api_preprocess_seconds", "api_preprocess"),
    ("atom:request_queue_time_seconds", "queue_time (engine)"),
    ("atom:ttft_forward_to_output_seconds", "forward_to_output"),
    ("atom:ttft_output_delivery_seconds", "output_delivery (optional)"),
    ("atom:api_body_parse_seconds", "body reception / handling"),
    ("atom:api_chat_template_seconds", "actual chat template calls"),
    ("atom:api_tokenize_seconds", "actual tokenize calls"),
)
SUBSET_STAGES = (
    ("atom:pd_kv_transfer_seconds", "  pd_kv (part of queue_time)"),
    ("atom:gpu_forward_seconds", "  gpu_forward (part of fwd)"),
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("metrics_url", help="e.g. http://127.0.0.1:8020/metrics")
    args = p.parse_args()

    text = _fetch(args.metrics_url)
    buckets = _parse_histograms(text)
    sums = _parse_sum_count(text)

    print("Decode TTFT stage breakdown (API + engine histograms):\n")
    for name, label in STAGES:
        _row(name, label, sums, buckets)
    print()
    for name, label in SUBSET_STAGES:
        _row(name, label, sums, buckets)
    print()
    _row("atom:time_to_first_token_seconds", "ttft_total (streaming)", sums, buckets)

    print(
        "\nStages cover different request populations and incomplete boundaries; "
        "means/quantiles are not summed. Unsampled stages are n/a."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
