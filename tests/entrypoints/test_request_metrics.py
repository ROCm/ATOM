"""No GPU: compatibility, streaming timing and bounded event persistence."""

import ast
import asyncio
import contextlib
import json
import logging
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from prometheus_client import CollectorRegistry, generate_latest
from prometheus_client.parser import text_string_to_metric_families

from atom.entrypoints.openai.metrics import AtomMetricsExporter
from atom.entrypoints.openai.request_metrics import RequestMetrics, RequestObservation


def read_metrics(registry):
    return {
        (s.name, tuple(sorted(s.labels.items()))): s.value
        for family in text_string_to_metric_families(generate_latest(registry).decode())
        for s in family.samples
    }


def value(registry, name, role="decode"):
    return read_metrics(registry).get((name, (("model", "test"), ("role", role))), 0)


def output(tokens, finished=False):
    return SimpleNamespace(
        output_tokens=tokens,
        finished=finished,
        finish_reason="stop" if finished else None,
        num_cached_tokens=8,
    )


def test_streaming_mtp_intervals_and_completed_request_are_distinct(
    monkeypatch, tmp_path
):
    registry = CollectorRegistry()
    metrics = RequestMetrics(registry)
    events = tmp_path / "decode.jsonl"
    metrics.configure(model="test", role="decode", run_id="42")
    clock = iter([100, 101, 101.12, 101.32, 101.4])
    monkeypatch.setattr(
        "atom.entrypoints.openai.request_metrics.time.monotonic", lambda: next(clock)
    )
    # Avoid the background writer's flush clock consuming this synthetic timeline.
    recorder = []
    metrics.writer = SimpleNamespace(put=recorder.append)
    observation = RequestObservation(metrics, "stream-a", choice_index=1)
    observation.num_prompt_tokens = 32
    observation.on_output(output([1]))
    observation.on_output(output([2, 3, 4]))
    observation.on_output(output([5]))
    observation.on_output(output([], finished=True))
    observation.on_output(output([999], finished=True))  # duplicate completion ignored
    assert value(registry, "atom:ttft_seconds_sum") == pytest.approx(1)
    assert value(registry, "atom:tpot_seconds_sum") == pytest.approx(0.32 / 4)
    assert value(registry, "atom:e2e_latency_seconds_sum") == pytest.approx(1.4)
    assert value(registry, "atom:output_chunk_interval_seconds_count") == 2
    assert value(registry, "atom:output_chunk_interval_seconds_sum") == pytest.approx(
        0.32
    )
    assert [e["token_count"] for e in recorder if e["event"] == "output_chunk"] == [
        1,
        3,
        1,
    ]
    assert recorder[-1]["isl"] == 32
    assert recorder[-1]["osl"] == 5
    assert recorder[-1]["choice_index"] == 1
    # Restore the real clock before stopping the background thread.
    monkeypatch.undo()
    metrics.writer = None
    metrics.configure(model="test", role="decode", events_path=events, run_id="42")
    for event in recorder:
        metrics.writer.put(event)
    metrics.close()
    assert [json.loads(line) for line in events.read_text().splitlines()] == recorder


def test_prefill_one_token_has_no_fake_tpot_or_itl():
    registry = CollectorRegistry()
    metrics = RequestMetrics(registry)
    metrics.configure(model="test", role="prefill")
    RequestObservation(metrics, "prefill").on_output(output([1], finished=True))
    assert value(registry, "atom:ttft_seconds_count", "prefill") == 1
    assert value(registry, "atom:tpot_seconds_count", "prefill") == 0
    assert value(registry, "atom:output_chunk_interval_seconds_count", "prefill") == 0
    assert value(registry, "atom:e2e_latency_seconds_count", "decode") == 0


def test_unfinished_request_does_not_enter_completion_histograms():
    registry = CollectorRegistry()
    metrics = RequestMetrics(registry)
    metrics.configure(model="test", role="decode")
    RequestObservation(metrics, "cancelled").on_output(output([1]))
    assert value(registry, "atom:e2e_latency_seconds_count") == 0


def test_scrapes_do_not_reset_counts_or_change_existing_metric_samples():
    exporter = AtomMetricsExporter()
    exporter.update(
        {
            "enabled": True,
            "requests_running": 3,
            "prompt_tokens": 27,
            "cache": {"cached_tokens": 12},
        }
    )
    before = read_metrics(exporter._registry)
    exporter.requests.configure(model="test", role="decode")
    RequestObservation(exporter.requests, "req").on_output(
        output([1, 2], finished=True)
    )
    first, second = read_metrics(exporter._registry), read_metrics(exporter._registry)
    assert first == second
    added_prefixes = (
        "atom:ttft_",
        "atom:tpot_",
        "atom:e2e_latency_",
        "atom:output_chunk_interval_",
        "atom:request_events_",
    )
    legacy = lambda data: {
        k: v for k, v in data.items() if not k[0].startswith(added_prefixes)
    }
    assert legacy(before) == legacy(first)
    assert first[("atom:requests_running", ())] == 3
    assert first[("atom:prompt_tokens_total", ())] == 27


@pytest.mark.parametrize(
    "name",
    [
        "generate_async",
        "generate_async_multimodal",
        "generate_async_fanout",
        "setup_streaming_request",
        "setup_streaming_request_fanout",
    ],
)
def test_all_generation_paths_record_callbacks_before_delivery(name):
    # Execute the actual entrypoint function with a tiny engine double. Loading
    # its AST avoids requiring transformers/aiter just to test callback wiring.
    path = Path(__file__).parents[2] / "atom/entrypoints/openai/api_server.py"
    tree = ast.parse(path.read_text())
    fn = next(
        node
        for node in tree.body
        if isinstance(node, ast.AsyncFunctionDef) and node.name == name
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            fn,
        ],
        type_ignores=[],
    )
    registry = CollectorRegistry()
    metrics = RequestMetrics(registry)
    metrics.configure(model="test", role="decode")
    callbacks, requests = {}, {}

    def preprocess(*args, stream_callback, **kwargs):
        seq = SimpleNamespace(id=len(callbacks), num_prompt_tokens=4)
        callbacks[seq.id], requests[seq.id] = stream_callback, seq
        return seq

    def preprocess_fanout(*args, stream_callbacks, **kwargs):
        return [preprocess(stream_callback=cb) for cb in stream_callbacks]

    def add(seqs):
        for seq in seqs:
            callbacks[seq.id](output([1]))
            callbacks[seq.id](output([2, 3], finished=True))

    def delivery(*args):
        if args[0].finished:
            assert value(registry, "atom:e2e_latency_seconds_count") > 0

    namespace = dict(  # noqa: C408 - mirrors named module globals
        asyncio=asyncio,
        contextlib=contextlib,
        time=time,
        logger=logging.getLogger("test"),
        RequestObservation=RequestObservation,
        _metrics_exporter=SimpleNamespace(requests=metrics),
        engine=SimpleNamespace(
            io_processor=SimpleNamespace(
                preprocess=preprocess,
                preprocess_fanout=preprocess_fanout,
                requests=requests,
            ),
            core_mgr=SimpleNamespace(add_request=add),
        ),
        new_token_ids=list,
        delivered_text=lambda tokens: "ok",
        _validate_sequence_context_length=lambda seq: None,
        StreamOutputCollector=lambda rid: None,
        _stream_loops={},
        _request_start_times={},
        _seq_id_to_request_id={},
        _stream_batch_dispatcher=SimpleNamespace(new_state=lambda: None),
        _send_stream_chunk_direct=delivery,
        _send_stream_chunk_tagged=delivery,
    )
    exec(  # noqa: S102 - execute only the trusted repository function
        compile(ast.fix_missing_locations(module), str(path), "exec"), namespace
    )

    async def run():
        args = (
            ([1, 2, 3, 4], {}, SimpleNamespace(n=2), "r")
            if name == "generate_async_multimodal"
            else ("prompt", SimpleNamespace(n=2), "r")
        )
        result = namespace[name](*args)
        if name in ("generate_async", "generate_async_multimodal"):
            async for _ in result:
                pass
        else:
            await result

    asyncio.run(run())
    expected = 2 if "fanout" in name else 1
    assert value(registry, "atom:e2e_latency_seconds_count") == expected
    assert value(registry, "atom:output_chunk_interval_seconds_count") == expected
