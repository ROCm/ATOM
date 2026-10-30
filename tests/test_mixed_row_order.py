# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Code outside the mixed path that splits a batch into prefill and decode
rows. A mixed batch is `[prefill | decode]`; a pure batch has one kind, so the
same bounds describe both."""

from types import SimpleNamespace

import pytest

from atom.metrics.scheduler import RequestQueueTiming, SchedulerMetrics
from atom.observability import tracing


class _Recorder:
    def __init__(self):
        self.values = []

    def observe(self, value):
        self.values.append(value)


def _mixed_batch():
    # Row 0: a single-chunk prefill of 8 prompt tokens. Rows 1-2: decodes.
    return SimpleNamespace(
        req_ids=[10, 20, 30],
        is_dummy_run=False,
        total_seqs_num_prefill=1,
        total_seqs_num_decode=2,
        total_tokens_num_prefill=8,
        num_cached_tokens=[0, 40, 50],
        context_lens=[8, 41, 51],
    )


def _seq(prompt_tokens):
    return SimpleNamespace(
        num_prompt_tokens=prompt_tokens,
        queue_timing=RequestQueueTiming(received_at=0.0),
    )


def test_scheduler_metrics_split_a_mixed_batch_prefill_first(monkeypatch):
    metrics = SchedulerMetrics.__new__(SchedulerMetrics)
    for name in (
        "queue_time",
        "decode_batch_size",
        "decode_context_tokens",
        "prefill_batch_tokens",
        "prefill_request_tokens",
        "prefill_context_tokens",
    ):
        setattr(metrics, name, _Recorder())
    metrics.decode_request_context_tokens = "decode"
    metrics.prefill_request_context_tokens = "prefill"
    contexts = []
    monkeypatch.setattr(
        metrics,
        "_record_request_context",
        lambda metric, req_id, seq, tokens: contexts.append((metric, req_id, tokens)),
    )
    seqs = {10: _seq(8), 20: _seq(40), 30: _seq(50)}

    metrics.record_forward(_mixed_batch(), seqs)

    # The prefill row is the request whose prompt is measured; the decode
    # rows are the ones whose decode context is recorded.
    assert metrics.prefill_request_tokens.values == [8]
    assert contexts == [("decode", 20, 41), ("decode", 30, 51), ("prefill", 10, 8)]
    assert metrics.prefill_context_tokens.values == [8]
    assert metrics.decode_context_tokens.values == [41 + 51]
    assert metrics.decode_batch_size.values == [2]
    assert not seqs[10].queue_timing.decode_context_observed


def test_trace_marks_only_a_mixed_batchs_decode_rows_as_decode(monkeypatch):
    calls = []
    monkeypatch.setattr(tracing, "_ENABLED", True)
    monkeypatch.setattr(
        tracing, "forward", lambda seq, *, is_decode: calls.append((seq, is_decode))
    )
    tracing.record_forward(_mixed_batch(), {10: "p", 20: "d1", 30: "d2"})
    assert calls == [("p", False), ("d1", True), ("d2", True)]


@pytest.mark.parametrize("request_context_enabled", [False, True])
def test_a_real_mixed_step_records_each_rows_own_metric(request_context_enabled):
    """A mixed batch the scheduler really builds, recorded by a real
    `SchedulerMetrics`: the single-chunk prefill lands in the prefill metrics
    and the decode row's context in the decode ones. Aggregate metrics remain
    available with per-request context gauges either disabled or enabled."""
    from conftest import MockConfig
    from prometheus_client import CollectorRegistry

    from atom.model_engine.scheduler import Scheduler
    from atom.model_engine.sequence import Sequence
    from atom.sampling_params import SamplingParams

    sched = Scheduler(
        MockConfig(
            enable_mixed_prefill_decode=True,
            num_kvcache_blocks=20,
            max_num_seqs=4,
            max_num_batched_tokens=256,
        )
    )
    decode = Sequence([1, 2, 3, 4, 5, 6], 4, sampling_params=SamplingParams())
    sched.add(decode)
    sched.schedule()
    decode.num_cached_tokens = decode.num_prompt_tokens
    decode.append_token(99)
    prefill = Sequence(list(range(10, 23)), 4, sampling_params=SamplingParams())
    sched.add(prefill)
    batch, _ = sched.schedule()
    assert batch.is_mixed and batch.req_ids == [prefill.id, decode.id]

    registry = CollectorRegistry()
    metrics = SchedulerMetrics(
        registry=registry, request_context_enabled=request_context_enabled
    )
    for seq in (prefill, decode):
        seq.queue_timing = None
        metrics.enqueue(seq)
    decode.queue_timing.prefill_observed = True  # its prefill was an earlier step
    metrics.record_forward(batch, {prefill.id: prefill, decode.id: decode})

    def value(name, **labels):
        labels = {"dp_rank": "0", "engine_role": "default", **labels}
        return registry.get_sample_value(name, labels)

    # The 13-token prompt is the request whose prefill is measured...
    assert value("atom:prefill_request_tokens_count") == 1
    assert value("atom:prefill_request_tokens_sum") == 13
    assert value("atom:prefill_context_tokens_sum") == 13
    # ...and the decode row's 7-token context is the decode context.
    assert value("atom:decode_context_tokens_sum") == 7
    assert value("atom:decode_batch_size_sum") == 1
    assert decode.queue_timing.decode_context_observed
    assert not prefill.queue_timing.decode_context_observed
    gauges = {
        (s.name, s.labels["sequence_id"]): s.value
        for metric in registry.collect()
        for s in metric.samples
        if s.name.endswith("request_context_tokens")
    }
    if request_context_enabled:
        assert gauges == {
            ("atom:decode_request_context_tokens", str(decode.id)): 7,
            ("atom:prefill_request_context_tokens", str(prefill.id)): 13,
        }
    else:
        assert gauges == {}


def test_eplb_commits_a_mixed_passs_load_and_counts_its_step(monkeypatch):
    """EPLB's two gates agree on a mixed pass: the step it advances on carries
    load into the window. Recording only pure-prefill passes would leave the
    window empty under mixed batching while the rebalance schedule ran on."""
    from atom.model_ops import eplb

    calls = []
    monitor = SimpleNamespace(
        on_forward_start=lambda: None,
        on_forward_end=lambda dummy, carries: calls.append(("record", carries)),
    )
    manager = SimpleNamespace(
        monitor=monitor,
        live_metadata=object(),
        on_forward_pass_end=lambda local, dp_any: calls.append(("step", local)),
    )
    monkeypatch.setattr(eplb, "_MANAGER", manager)
    forward = eplb.with_eplb_forward_monitor(lambda self, batch: None)
    runner = SimpleNamespace()
    for n_prefill_tokens, n_decode_tokens in ((40, 3), (40, 0), (0, 3)):
        forward(
            runner,
            SimpleNamespace(
                is_dummy_run=False,
                total_tokens_num_prefill=n_prefill_tokens,
                total_tokens_num_decode=n_decode_tokens,
            ),
        )
    assert calls == [
        ("record", True),  # mixed
        ("step", True),
        ("record", True),  # pure prefill
        ("step", True),
        ("record", False),  # pure decode
        ("step", False),
    ]
