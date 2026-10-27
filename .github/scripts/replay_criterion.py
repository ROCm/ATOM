#!/usr/bin/env python3
"""Replay both criteria over per-night deltas, and report what each would alert.

The point is not a one-off analysis. It turns "how often does this criterion
fire on real history" into something a test can assert, so a later change to a
constant cannot quietly walk the alert volume back up.

`nightly_noise.py` (outside this repo) builds the deltas from the public
dashboard; this module takes them already built so it stays testable with no
network.
"""

import collections
import sys
from pathlib import Path

# Add .github/scripts to path to import perf_criterion
sys.path.insert(0, str(Path(__file__).resolve().parent))

from perf_criterion import group_families, judge_family

THROUGHPUT_REGRESSION_PCT = -5.0  # summarize.py, as shipped
LATENCY_REGRESSION_PCT = 10.0

# Metrics from the delta shape: tput_pct (total_token_throughput), tpot_pct, ttft_pct.
# summarize.py's TRACKED_METRICS has four entries (adds output_throughput); see docstring
# of shipped_alerts() for why only these three are counted here.
_SHIPPED_METRICS = (
    ("tput_pct", True),
    ("tpot_pct", False),
    ("ttft_pct", False),
)


def shipped_alerts(deltas):
    """Reproduce summarize.py's criterion over the metrics the delta shape carries.

    This counts any one (configuration, metric) exceeding the regression threshold.
    Note: summarize.py tracks four metrics (TRACKED_METRICS), including
    output_throughput, but the delta dict carries only three (_SHIPPED_METRICS):
    tput_pct (mapped from total_token_throughput), tpot_pct, and ttft_pct. The
    output_throughput field is omitted because judge_family judges only on the
    single throughput field. Adding a second throughput metric would require
    extending the delta contract with a field no criterion reads.

    Consequently, shipped_total is a lower bound on what summarize.py would report
    from the same underlying data. The error is conservative: undercounting makes
    the shipped criterion appear safer than it actually is.
    """
    hits = 0
    for delta in deltas:
        for field, higher_is_better in _SHIPPED_METRICS:
            pct = delta.get(field)
            if pct is None:
                continue
            if (higher_is_better and pct <= THROUGHPUT_REGRESSION_PCT) or (
                not higher_is_better and pct >= LATENCY_REGRESSION_PCT
            ):
                hits += 1
    return hits


def linkage_alerts(deltas):
    """Family linkage: how many families trip."""
    return sum(
        1
        for levels in group_families(deltas).values()
        if judge_family(levels)["status"] == "triggered"
    )


def replay(deltas):
    """Both criteria, totalled and per night.

    ``deltas`` may carry a ``run`` key; entries without one are treated as a
    single night.
    """
    by_night = collections.defaultdict(list)
    for delta in deltas:
        by_night[delta.get("run", "_")].append(delta)

    shipped = {night: shipped_alerts(rows) for night, rows in by_night.items()}
    linkage = {night: linkage_alerts(rows) for night, rows in by_night.items()}
    return {
        "nights": len(by_night),
        "shipped_total": sum(shipped.values()),
        "linkage_total": sum(linkage.values()),
        "shipped_nights_alerting": sum(1 for v in shipped.values() if v),
        "linkage_nights_alerting": sum(1 for v in linkage.values() if v),
        "shipped_per_night": shipped,
        "linkage_per_night": linkage,
    }
