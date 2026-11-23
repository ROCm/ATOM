# SPDX-License-Identifier: MIT
"""RoundStats: the per-step tally of a decode round's routing decisions."""

import logging

from atom.model_engine.round_stats import MAX_BUCKET, RoundStats
from atom.utils.forward_context import RoundService


def test_counts_by_kind_and_requests_and_logs_each_interval(caplog):
    now = [0.0]
    stats = RoundStats(interval_s=60, clock=lambda: now[0])
    for _ in range(3):
        stats.record(is_prefill=False, requests=4, service=RoundService.DEVICE)
    stats.record(is_prefill=False, requests=12, service=RoundService.NONE)
    stats.record(is_prefill=False, requests=40, service=RoundService.NONE)
    stats.record(is_prefill=True, requests=2, service=RoundService.NONE)
    assert stats.total[("decode", 4)] == [3, 3]
    assert stats.total[("decode", MAX_BUCKET + 1)] == [0, 1]
    text = RoundStats.summary(stats.total)
    assert "decode served 3/5" in text and "4: 3/3" in text and "17+: 0/1" in text
    assert "prefill served 0/1" in text
    now[0] = 61.0
    with caplog.at_level(logging.INFO, logger="atom"):
        stats.record(is_prefill=False, requests=1, service=RoundService.FORWARD)
    assert any("decode round (last 60s)" in r.message for r in caplog.records)
    assert stats.window == {} and stats.total[("decode", 1)] == [1, 1]
