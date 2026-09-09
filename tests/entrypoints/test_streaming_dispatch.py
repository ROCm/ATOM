import asyncio
import logging

import pytest
from prometheus_client import CollectorRegistry, Histogram, generate_latest
from prometheus_client.parser import text_string_to_metric_families

from atom.entrypoints.openai import streaming_dispatch
from atom.entrypoints.openai.metrics import AtomMetricsExporter
from atom.entrypoints.openai.streaming_dispatch import (
    IncrementalStreamDetokenizer,
    StreamBatchDispatcher,
    StreamOutputCollector,
    enable_delta_reuse,
    merge_chunk,
)


class _Utf8ByteTokenizer:
    # One token per byte, so the delta-reuse probe's sampled-vocabulary
    # streams reach every id this double can produce.
    vocab_size = 256

    def encode(self, text, add_special_tokens=False):
        return list(text.encode())

    def decode(self, token_ids, skip_special_tokens=True):
        # `bytes(x)` of an `array("i")` copies its buffer -- four bytes per id
        # -- where from a list it takes the values. A real tokenizer reads ids,
        # so this double has to as well; keep the `list`.
        return bytes(list(token_ids)).decode("utf-8", errors="replace")


class _ImmediateLoop:
    def __init__(self):
        self.calls = []

    def call_soon_threadsafe(self, callback, *args):
        self.calls.append((callback, args))
        callback(*args)

    call_soon = call_soon_threadsafe


class _RecordingLoop:
    """Loop stub that defers callbacks so each round is one loop iteration."""

    def __init__(self):
        self.pending = []

    def call_soon_threadsafe(self, callback, *args):
        self.pending.append((callback, args))

    call_soon = call_soon_threadsafe

    def run(self):
        rounds = 0
        while self.pending:
            batch, self.pending = self.pending, []
            for callback, args in batch:
                callback(*args)
            rounds += 1
        return rounds


def _resolve(coro):
    """Drive a coroutine that must complete without ever suspending."""
    try:
        coro.send(None)
    except StopIteration as stop:
        return stop.value
    coro.close()
    raise AssertionError("coroutine suspended when it should have had a value ready")


def _itl_samples(exporter):
    return {
        (sample.name, sample.labels.get("le")): sample.value
        for family in text_string_to_metric_families(exporter.render().decode())
        for sample in family.samples
        if sample.name.startswith("atom:inter_token_latency_seconds_")
        and not sample.name.endswith("_created")
    }


def test_weighted_itl_matches_histogram_buckets_and_handles_large_batches():
    exporter = AtomMetricsExporter()
    registry = CollectorRegistry()
    reference = Histogram(
        "atom:inter_token_latency_seconds",
        "reference",
        buckets=exporter._inter_token_latency._bounds,
        registry=registry,
    )
    for interval, tokens in ((0.0, 3), (0.008, 4), (0.09, 3), (32.0, 2)):
        exporter.observe_inter_token_latency(interval, tokens)
        for _ in range(tokens):
            reference.observe(interval / tokens)
    expected = {
        (s.name, s.labels.get("le")): s.value
        for f in text_string_to_metric_families(generate_latest(registry).decode())
        for s in f.samples
        if not s.name.endswith("_created")
    }
    assert _itl_samples(exporter) == pytest.approx(expected)
    exporter.observe_inter_token_latency(5000.0, 10_000_000)
    exporter.observe_inter_token_latency(1.0, 0)
    samples = _itl_samples(exporter)
    prefix = "atom:inter_token_latency_seconds"
    assert samples[(prefix + "_count", None)] == 10_000_012
    assert samples[(prefix + "_bucket", "0.002")] == 10_000_007
    assert samples[(prefix + "_sum", None)] == pytest.approx(5032.098)


def test_itl_preserves_token_weighted_intervals_when_stream_chunks_coalesce(
    monkeypatch,
):
    exporter = AtomMetricsExporter()
    tokenizer = _CountingTokenizer()
    dispatcher = StreamBatchDispatcher(
        tokenizer,
        observe_inter_token_latency=exporter.observe_inter_token_latency,
    )
    state = dispatcher.new_state()
    collector = StreamOutputCollector("itl")
    loop = _ImmediateLoop()
    timestamps = iter((1.0, 1.125, 1.125, 1.375, 1.375))
    monkeypatch.setattr(
        "atom.entrypoints.openai.streaming_dispatch.time.perf_counter",
        lambda: next(timestamps),
    )
    # Four speculative tokens after 125 ms contribute four 31.25 ms
    # observations; the next token contributes one 250 ms observation.
    # An empty terminal output must not contribute another sample.
    for tokens in ([65], [66, 67, 68, 69], [70], []):
        dispatcher.enqueue(
            loop=loop,
            collector=collector,
            state=state,
            chunk={"token_ids": tokens, "finished": not tokens},
        )
        dispatcher.flush()

    # ITL still samples each delivered batch while decoding waits for a read.
    assert tokenizer.calls == 0
    samples = _itl_samples(exporter)
    assert _resolve(collector.get())["text"] == "ABCDEF"
    assert tokenizer.calls == 2
    assert _itl_samples(exporter) == samples
    prefix = "atom:inter_token_latency_seconds"
    assert samples[(prefix + "_count", None)] == 5
    assert samples[(prefix + "_sum", None)] == pytest.approx(0.375)
    assert samples[(prefix + "_bucket", "0.03")] == 0
    assert samples[(prefix + "_bucket", "0.035")] == 4
    assert samples[(prefix + "_bucket", "+Inf")] == 5

    # The 5-second runtime snapshot and repeated scrapes must neither reset
    # these streaming samples nor count them again.
    exporter.update({"enabled": True})
    exporter.update({"enabled": True, "requests_running": 0})
    assert _itl_samples(exporter) == samples
    assert _itl_samples(exporter) == samples


@pytest.mark.parametrize("collector_type", [asyncio.Queue, StreamOutputCollector])
def test_itl_keeps_independent_clocks_for_interleaved_fanout_choices(
    monkeypatch, collector_type
):
    exporter = AtomMetricsExporter()
    dispatcher = StreamBatchDispatcher(
        _Utf8ByteTokenizer(),
        observe_inter_token_latency=exporter.observe_inter_token_latency,
    )
    states = [dispatcher.new_state(), dispatcher.new_state()]
    loop = _ImmediateLoop()
    queue = collector_type()
    timestamps = iter((1.0, 2.0, 2.5, 2.5, 3.0, 3.0))
    monkeypatch.setattr(
        "atom.entrypoints.openai.streaming_dispatch.time.perf_counter",
        lambda: next(timestamps),
    )
    for tag, token in ((0, 65), (1, 66), (1, 67), (0, 68)):
        dispatcher.enqueue(
            loop=loop,
            collector=queue,
            state=states[tag],
            chunk={"token_ids": [token], "finished": False},
            tag=tag,
        )
    dispatcher.flush()
    samples = _itl_samples(exporter)
    prefix = "atom:inter_token_latency_seconds"
    assert samples[(prefix + "_count", None)] == 2
    assert samples[(prefix + "_sum", None)] == pytest.approx(2.5)
    assert samples[(prefix + "_bucket", "0.6")] == 1
    assert samples[(prefix + "_bucket", "2.0")] == 2


def test_itl_includes_frontend_queueing_but_excludes_observation_work(monkeypatch):
    exporter = AtomMetricsExporter()
    clock = [0.0]
    monkeypatch.setattr(
        "atom.entrypoints.openai.streaming_dispatch.time.perf_counter",
        lambda: clock[0],
    )

    def observe(interval, tokens):
        exporter.observe_inter_token_latency(interval, tokens)
        clock[0] += 0.125

    dispatcher = StreamBatchDispatcher(
        _Utf8ByteTokenizer(), observe_inter_token_latency=observe
    )
    loop = _RecordingLoop()
    collector = StreamOutputCollector("frontend-itl")
    state = dispatcher.new_state()
    prefix = "atom:inter_token_latency_seconds"

    for arrival, delivery, tokens in (
        (1.0, 2.0, [65, 66, 67, 68]),  # Entire first batch is excluded.
        (3.0, 4.5, [69, 70, 71, 72]),  # 2.5 seconds / 4 new tokens.
        (5.0, 5.5, []),  # Empty output does not advance the last-token clock.
        (6.0, 6.625, [73]),  # 2 seconds since observation ended at 4.625.
    ):
        before = _itl_samples(exporter)
        clock[0] = arrival
        dispatcher.enqueue(
            loop=loop,
            collector=collector,
            state=state,
            chunk={"token_ids": tokens, "finished": tokens == [73]},
        )
        dispatcher.flush()
        assert _itl_samples(exporter) == before
        clock[0] = delivery
        loop.run()

    samples = _itl_samples(exporter)
    assert samples[(prefix + "_count", None)] == 5
    assert samples[(prefix + "_sum", None)] == pytest.approx(4.5)
    assert samples[(prefix + "_bucket", "0.6")] == 0
    assert samples[(prefix + "_bucket", "0.8")] == 4
    assert _resolve(collector.get())["text"] == "ABCDEFGHI"


def test_incremental_detokenizer_holds_incomplete_utf8():
    detokenizer = IncrementalStreamDetokenizer(_Utf8ByteTokenizer())

    assert detokenizer.update([0xE4], finished=False) == ""
    assert detokenizer.update([0xBD, 0xA0], finished=False) == "你"
    assert detokenizer.update([ord("!")], finished=True) == "!"


def test_dispatcher_batches_direct_and_tagged_chunks_per_loop():
    dispatcher = StreamBatchDispatcher(_Utf8ByteTokenizer())
    loop = _ImmediateLoop()
    direct_queue = StreamOutputCollector("direct")
    tagged_queue = StreamOutputCollector("fanout")

    dispatcher.enqueue(
        loop=loop,
        collector=direct_queue,
        state=dispatcher.new_state(),
        chunk={"token_ids": [ord("A")], "finished": True},
    )
    dispatcher.enqueue(
        loop=loop,
        collector=tagged_queue,
        state=dispatcher.new_state(),
        chunk={"token_ids": [ord("B")], "finished": True},
        tag=0,
    )
    dispatcher.flush()

    assert len(loop.calls) == 1
    assert _resolve(direct_queue.get())["text"] == "A"
    sibling_index, chunk = _resolve(tagged_queue.get())
    assert sibling_index == 0
    assert chunk["text"] == "B"


def test_dispatcher_keeps_fanout_detokenizer_state_separate():
    dispatcher = StreamBatchDispatcher(_Utf8ByteTokenizer())
    loop = _ImmediateLoop()
    queue = StreamOutputCollector("fanout")
    sibling_0, sibling_1 = dispatcher.new_state(), dispatcher.new_state()

    dispatcher.enqueue(
        loop=loop,
        collector=queue,
        state=sibling_0,
        chunk={"token_ids": [0xE4], "finished": False},
        tag=0,
    )
    dispatcher.enqueue(
        loop=loop,
        collector=queue,
        state=sibling_1,
        chunk={"token_ids": [ord("X")], "finished": True},
        tag=1,
    )
    dispatcher.flush()

    assert _resolve(queue.get())[1]["text"] == ""
    assert _resolve(queue.get())[1]["text"] == "X"

    # Sibling 0's half character survives sibling 1 finishing in between.
    dispatcher.enqueue(
        loop=loop,
        collector=queue,
        state=sibling_0,
        chunk={"token_ids": [0xBD, 0xA0], "finished": True},
        tag=0,
    )
    dispatcher.flush()

    assert _resolve(queue.get())[1]["text"] == "你"


def test_a_fresh_stream_does_not_inherit_a_half_decoded_character():
    """Two fresh states in the same batch cannot share a partial character."""
    dispatcher = StreamBatchDispatcher(_Utf8ByteTokenizer())
    loop = _ImmediateLoop()
    partial = StreamOutputCollector("partial")
    fresh = StreamOutputCollector("fresh")
    partial_state = dispatcher.new_state()
    dispatcher.enqueue(
        loop=loop,
        collector=partial,
        state=partial_state,
        chunk={"token_ids": [0xE4], "finished": False},
    )
    dispatcher.enqueue(
        loop=loop,
        collector=fresh,
        state=dispatcher.new_state(),
        chunk={"token_ids": [ord("A")], "finished": True},
    )
    dispatcher.flush()

    assert len(loop.calls) == 1
    assert _resolve(partial.get())["text"] == ""
    assert _resolve(fresh.get())["text"] == "A"
    dispatcher.enqueue(
        loop=loop,
        collector=partial,
        state=partial_state,
        chunk={"token_ids": [0xBD, 0xA0], "finished": True},
    )
    dispatcher.flush()
    assert _resolve(partial.get())["text"] == "你"


def test_collector_decodes_a_lone_chunk_without_waiting_for_more():
    """The shipped dispatcher path is immediately readable without a timer."""
    tokenizer = _CountingTokenizer()
    dispatcher = StreamBatchDispatcher(tokenizer)
    collector = StreamOutputCollector("request-1")
    dispatcher.enqueue(
        loop=_ImmediateLoop(),
        collector=collector,
        state=dispatcher.new_state(),
        chunk={"token_ids": [ord("a")], "finished": False},
    )
    dispatcher.flush()

    assert tokenizer.calls == 0
    assert _resolve(collector.get()) == {
        "token_ids": [ord("a")],
        "text": "a",
        "finished": False,
    }
    assert tokenizer.calls == 2


def test_collector_merges_a_backlog_into_one_chunk():
    collector = StreamOutputCollector("request-1")
    collector.put_nowait({"token_ids": [1], "text": "he", "finished": False})
    collector.put_nowait(
        {"token_ids": [2, 3], "text": "ll", "finished": False, "num_cached_tokens": 7}
    )
    collector.put_nowait(
        {
            "token_ids": [4],
            "text": "o",
            "finished": True,
            "finish_reason": "stop",
            "kv_transfer_params": {"a": 1},
        }
    )

    chunk = _resolve(collector.get())

    assert chunk["token_ids"] == [1, 2, 3, 4]
    assert chunk["text"] == "hello"
    assert chunk["finished"] is True
    assert chunk["finish_reason"] == "stop"
    assert chunk["kv_transfer_params"] == {"a": 1}
    # Landed on a middle chunk, so a naive "take the last one" would drop it.
    assert chunk["num_cached_tokens"] == 7


def test_collector_carries_trailing_fields_from_earlier_chunks():
    collector = StreamOutputCollector("request-1")
    collector.put_nowait(
        {"token_ids": [1], "text": "a", "kv_transfer_params": {"a": 1}}
    )
    collector.put_nowait({"token_ids": [2], "text": "b", "finished": True})

    chunk = _resolve(collector.get())

    assert chunk["kv_transfer_params"] == {"a": 1}


def test_collector_merges_fanout_siblings_independently():
    collector = StreamOutputCollector("request-1")
    collector.put_nowait((0, {"token_ids": [1], "text": "a"}))
    collector.put_nowait((1, {"token_ids": [9], "text": "x"}))
    collector.put_nowait((0, {"token_ids": [2], "text": "b", "finished": True}))

    first_tag, first = _resolve(collector.get())
    second_tag, second = _resolve(collector.get())

    assert (first_tag, first["text"], first["token_ids"]) == (0, "ab", [1, 2])
    assert (second_tag, second["text"], second["token_ids"]) == (1, "x", [9])


def test_collector_waits_only_when_nothing_is_pending():
    async def scenario():
        collector = StreamOutputCollector("request-1")
        getter = asyncio.ensure_future(collector.get())
        await asyncio.sleep(0)
        assert not getter.done()

        collector.put_nowait({"token_ids": [1], "text": "a"})
        assert (await getter)["text"] == "a"

        # Drained again: the readiness flag must have been cleared, or the next
        # get() would spin instead of waiting.
        again = asyncio.ensure_future(collector.get())
        await asyncio.sleep(0)
        assert not again.done()
        again.cancel()

    asyncio.run(scenario())


def _stream_through_collector(payload: bytes, drain_every: int):
    """Feed ``payload`` one byte per engine step, draining every N steps."""
    dispatcher = StreamBatchDispatcher(_Utf8ByteTokenizer())
    loop = _ImmediateLoop()
    collector = StreamOutputCollector("request-1")
    state = dispatcher.new_state()

    texts = []
    token_ids = []
    terminal = 0
    for index, byte in enumerate(payload):
        last = index == len(payload) - 1
        dispatcher.enqueue(
            loop=loop,
            collector=collector,
            state=state,
            chunk={"token_ids": [byte], "finished": last},
        )
        dispatcher.flush()
        if (index + 1) % drain_every == 0 or last:
            chunk = _resolve(collector.get())
            texts.append(chunk["text"])
            token_ids.extend(chunk["token_ids"])
            terminal += bool(chunk.get("finished"))
    return "".join(texts), token_ids, terminal


def test_merging_is_identical_to_unmerged_delivery():
    payload = "你好, world! 🎉".encode()
    reference_text, reference_tokens, reference_terminal = _stream_through_collector(
        payload, 1
    )

    assert reference_text == payload.decode()
    assert reference_tokens == list(payload)
    assert reference_terminal == 1

    for drain_every in (2, 3, 5, len(payload) * 2):
        text, tokens, terminal = _stream_through_collector(payload, drain_every)
        assert text == reference_text
        assert tokens == reference_tokens
        # Merging must never duplicate or swallow the end of the stream.
        assert terminal == 1


def test_a_step_is_delivered_in_a_single_loop_callback():
    """Splitting a step across callbacks lets the next step overtake its tail.

    The output thread schedules each step with call_soon_threadsafe. If a
    delivery re-armed itself for the rest of the step, step N+1 could be run
    first, so a collector would see N+1's chunk before N's leftovers -- folding
    would then concatenate deltas out of order and an end-of-stream landing
    before a straggler would be overwritten by it.
    """
    dispatcher = StreamBatchDispatcher(_Utf8ByteTokenizer())
    loop = _RecordingLoop()
    collectors = [StreamOutputCollector(f"request-{i}") for i in range(300)]

    for index, collector in enumerate(collectors):
        dispatcher.enqueue(
            loop=loop,
            collector=collector,
            state=dispatcher.new_state(),
            chunk={"token_ids": [ord("A")], "finished": True},
        )
    dispatcher.flush()

    assert len(loop.pending) == 1
    assert loop.run() == 1
    for collector in collectors:
        assert _resolve(collector.get())["text"] == "A"


def test_two_steps_keep_their_order_within_one_stream():
    """The end of a stream must never be overtaken by an earlier step's chunk."""
    dispatcher = StreamBatchDispatcher(_Utf8ByteTokenizer())
    loop = _RecordingLoop()
    collector = StreamOutputCollector("request-1")
    state = dispatcher.new_state()

    for byte, finished in ((ord("a"), False), (ord("b"), True)):
        dispatcher.enqueue(
            loop=loop,
            collector=collector,
            state=state,
            chunk={"token_ids": [byte], "finished": finished},
        )
        dispatcher.flush()
    loop.run()

    chunk = _resolve(collector.get())
    assert chunk["text"] == "ab"
    assert chunk["finished"] is True


def test_merge_keeps_end_of_stream_and_never_extends_the_producers_list():
    """A swallowed terminal flag hangs its client; a mutated list corrupts the producer."""
    produced = [1]
    collector = StreamOutputCollector("request-1")
    collector.put_nowait({"token_ids": produced, "text": "a", "finished": True})
    collector.put_nowait({"token_ids": [2], "text": "b", "finished": False})

    chunk = _resolve(collector.get())

    assert chunk["finished"] is True
    assert chunk["text"] == "ab"
    assert chunk["token_ids"] == [1, 2]
    assert produced == [1]


def test_put_nowait_gives_merge_chunk_a_list_it_may_extend():
    """`merge_chunk` extends in place; it used to create this key itself."""
    collector = StreamOutputCollector("request-1")
    collector.put_nowait({"text": "a"})

    assert collector._pending[None]["token_ids"] == []


def test_merge_extends_in_place_instead_of_rebuilding():
    """Rebuilding walks the whole accumulation, which is what a stall grows."""
    collector = StreamOutputCollector("request-1")
    collector.put_nowait({"token_ids": [1], "text": "a"})
    accumulating = collector._pending[None]["token_ids"]
    for token in (2, 3, 4):
        collector.put_nowait({"token_ids": [token], "text": "b"})

    assert collector._pending[None]["token_ids"] is accumulating
    assert accumulating == [1, 2, 3, 4]


@pytest.mark.parametrize("depth", (1, 2, 17, 500))
def test_a_merged_stream_reads_the_same_as_an_unmerged_one(depth):
    """A consumer must not be able to tell how far behind it fell."""
    tokens = list(range(depth + 1))
    unmerged = StreamOutputCollector("request-1")
    merged = StreamOutputCollector("request-2")
    delivered = []
    for index, token in enumerate(tokens):
        finished = index == len(tokens) - 1
        unmerged.put_nowait(
            {"token_ids": [token], "text": f"{token} ", "finished": finished}
        )
        delivered.append(_resolve(unmerged.get()))
        merged.put_nowait(
            {"token_ids": [token], "text": f"{token} ", "finished": finished}
        )

    folded = _resolve(merged.get())

    assert folded["text"] == "".join(chunk["text"] for chunk in delivered)
    assert folded["token_ids"] == tokens
    assert folded["finished"] is True


def test_merge_keeps_the_text_it_had_when_the_delta_is_not_a_string():
    """The text is popped to keep its refcount at one; a raise must not drop it."""
    into = {"token_ids": [1], "text": "acc"}
    with pytest.raises(TypeError):
        merge_chunk(into, {"token_ids": [2], "text": None})

    assert into["text"] == "acc"


def test_dispatcher_keeps_no_per_stream_state():
    """Finished callbacks may go away while collectors still own unread state."""
    dispatcher = StreamBatchDispatcher(_Utf8ByteTokenizer())
    loop = _ImmediateLoop()
    collectors = [StreamOutputCollector(str(i)) for i in range(64)]
    for collector in collectors:
        dispatcher.enqueue(
            loop=loop,
            collector=collector,
            state=dispatcher.new_state(),
            chunk={"token_ids": [ord("A")], "finished": True},
        )
    dispatcher.flush()

    assert vars(dispatcher).keys() == {
        "tokenizer",
        "synthetic_text",
        "_observe_inter_token_latency",
        "_thread_local",
    }
    for collector in collectors:
        assert _resolve(collector.get())["text"] == "A"


def test_each_stream_gets_its_own_detokenizer():
    dispatcher = StreamBatchDispatcher(_Utf8ByteTokenizer())

    first, second = dispatcher.new_state(), dispatcher.new_state()

    assert first is not second
    assert first.detokenizer is not second.detokenizer
    assert first.timing is not second.timing
    assert not first.detokenizer.tokens and not second.detokenizer.tokens


class _CountingTokenizer(_Utf8ByteTokenizer):
    def __init__(self):
        self.calls = 0

    def decode(self, token_ids, skip_special_tokens=True):
        self.calls += 1
        return super().decode(token_ids, skip_special_tokens)


def test_backlogged_tokens_are_merged_before_decoding():
    tokenizer = _CountingTokenizer()
    dispatcher = StreamBatchDispatcher(tokenizer)
    loop = _RecordingLoop()
    collector = StreamOutputCollector("slow-reader")
    state = dispatcher.new_state()
    payload = ("你好 🎉 " * 40).encode()
    for i, byte in enumerate(payload):
        dispatcher.enqueue(
            loop=loop,
            collector=collector,
            state=state,
            chunk={
                "token_ids": [byte],
                "finished": i == len(payload) - 1,
                "finish_reason": "length" if i == len(payload) - 1 else None,
                "num_cached_tokens": 7 if i == 0 else 0,
            },
        )
        dispatcher.flush()
    loop.run()

    # Producers and delivery callbacks must not spend time decoding a backlog
    # the consumer will fold into a single response anyway.
    assert tokenizer.calls == 0
    chunk = _resolve(collector.get())
    assert chunk["text"] == payload.decode()
    assert chunk["token_ids"] == list(payload)
    assert chunk["finished"] is True
    assert chunk["finish_reason"] == "length"
    assert chunk["num_cached_tokens"] == 7
    assert "_detokenizer" not in chunk
    assert tokenizer.calls <= 2


def test_deferred_decode_keeps_fanout_and_partial_unicode_separate():
    dispatcher = StreamBatchDispatcher(_Utf8ByteTokenizer())
    loop = _ImmediateLoop()
    collector = StreamOutputCollector("fanout")
    states = [dispatcher.new_state(), dispatcher.new_state()]

    def send(tag, ids, finished=False):
        dispatcher.enqueue(
            loop=loop,
            collector=collector,
            state=states[tag],
            chunk={"token_ids": ids, "finished": finished},
            tag=tag,
        )
        dispatcher.flush()

    send(0, [0xE4])
    assert _resolve(collector.get())[1]["text"] == ""
    send(1, list(b"other"), True)
    send(0, [0xBD])
    send(0, [0xA0], True)
    assert _resolve(collector.get()) == (
        1,
        {"token_ids": list(b"other"), "text": "other", "finished": True},
    )
    assert _resolve(collector.get()) == (
        0,
        {"token_ids": [0xBD, 0xA0], "text": "你", "finished": True},
    )


def test_deferred_synthetic_stream_preserves_token_count():
    tokenizer = _CountingTokenizer()
    dispatcher = StreamBatchDispatcher(tokenizer, synthetic_text="synthetic ")
    loop = _ImmediateLoop()
    collector = StreamOutputCollector("synthetic")
    state = dispatcher.new_state()
    for byte in b"abc":
        dispatcher.enqueue(
            loop=loop,
            collector=collector,
            state=state,
            chunk={"token_ids": [byte], "finished": byte == ord("c")},
        )
        dispatcher.flush()
    assert tokenizer.calls == 0
    chunk = _resolve(collector.get())
    assert chunk["text"] == "synthetic " * 3
    assert chunk["token_ids"] == list(b"abc")
    assert chunk["finished"] is True
    assert tokenizer.calls == 2  # one real update for three accumulated tokens


def test_concurrent_output_threads_preserve_stream_contents_and_termination():
    async def scenario():
        dispatcher = StreamBatchDispatcher(_Utf8ByteTokenizer())
        loop = asyncio.get_running_loop()
        collectors = [StreamOutputCollector(str(i)) for i in range(16)]
        states = [dispatcher.new_state() for _ in collectors]
        payloads = [(f"stream {i}: 你好 🎉 " * 5).encode() for i in range(16)]

        def produce(rank):
            for step in range(max(map(len, payloads))):
                for i in range(rank, len(collectors), 4):
                    if step >= len(payloads[i]):
                        continue
                    dispatcher.enqueue(
                        loop=loop,
                        collector=collectors[i],
                        state=states[i],
                        chunk={
                            "token_ids": [payloads[i][step]],
                            "finished": step == len(payloads[i]) - 1,
                        },
                    )
                dispatcher.flush()

        async def consume(i):
            text = ""
            tokens = []
            while True:
                chunk = await collectors[i].get()
                text += chunk["text"]
                tokens.extend(chunk["token_ids"])
                if chunk["finished"]:
                    break
                await asyncio.sleep(0)
            assert text == payloads[i].decode()
            assert tokens == list(payloads[i])
            assert not collectors[i]._pending

        await asyncio.wait_for(
            asyncio.gather(
                *(asyncio.to_thread(produce, rank) for rank in range(4)),
                *(consume(i) for i in range(len(collectors))),
            ),
            timeout=5,
        )

    asyncio.run(scenario())


class _FailOnceTokenizer(_CountingTokenizer):
    def __init__(self, fail_on):
        super().__init__()
        self.fail_on = fail_on

    def decode(self, token_ids, skip_special_tokens=True):
        if self.calls + 1 == self.fail_on:
            self.calls += 1
            raise ValueError("injected decode failure")
        return super().decode(token_ids, skip_special_tokens)


@pytest.mark.parametrize("fail_on", (3, 4), ids=("prefix", "new-text"))
def test_decode_failure_keeps_tokens_for_a_later_update(fail_on, caplog):
    # Each update decodes the prefix and then the new text. Either call can
    # fail after tokens.extend(), before the prefix/read offsets advance.
    dispatcher = StreamBatchDispatcher(_FailOnceTokenizer(fail_on))
    collector = StreamOutputCollector("recoverable")
    state = dispatcher.new_state()
    loop = _ImmediateLoop()

    def send(byte, finished=False):
        dispatcher.enqueue(
            loop=loop,
            collector=collector,
            state=state,
            chunk={"token_ids": [byte], "finished": finished},
        )
        dispatcher.flush()
        return _resolve(collector.get())

    assert send(ord("A"))["text"] == "A"
    failed = send(ord("B"))
    assert failed == {"token_ids": [ord("B")], "text": "", "finished": False}
    recovered = send(ord("C"), finished=True)
    assert recovered == {"token_ids": [ord("C")], "text": "BC", "finished": True}
    assert list(state.detokenizer.tokens) == list(b"ABC")
    assert "Error detokenizing stream recoverable (tag=None)" in caplog.text
    assert "injected decode failure" in caplog.text


@pytest.mark.parametrize("fail_on", (1, 2), ids=("prefix", "new-text"))
def test_failed_terminal_decode_preserves_metadata_and_fanout(fail_on, caplog):
    dispatcher = StreamBatchDispatcher(_FailOnceTokenizer(fail_on))
    collector = StreamOutputCollector("fanout-error")
    loop = _ImmediateLoop()
    states = [dispatcher.new_state(), dispatcher.new_state()]
    for tag, ids, finished in ((0, [65], False), (0, [66], True), (1, [90], True)):
        dispatcher.enqueue(
            loop=loop,
            collector=collector,
            state=states[tag],
            chunk={
                "token_ids": ids,
                "finished": finished,
                "finish_reason": "length" if finished else None,
                "num_cached_tokens": 7,
                "kv_transfer_params": {"source": "prefill"},
            },
            tag=tag,
        )
    dispatcher.flush()

    tag, failed = _resolve(collector.get())
    assert tag == 0
    assert failed == {
        "token_ids": [65, 66],
        "text": "",
        "finished": True,
        "finish_reason": "length",
        "num_cached_tokens": 7,
        "kv_transfer_params": {"source": "prefill"},
    }
    tag, healthy = _resolve(collector.get())
    assert tag == 1
    assert healthy["text"] == "Z"
    assert healthy["token_ids"] == [90]
    assert healthy["finished"] is True
    assert not collector._pending
    assert not collector._ready.is_set()
    assert "Error detokenizing stream fanout-error (tag=0)" in caplog.text
    assert "injected decode failure" in caplog.text


class _PositionSensitiveTokenizer(_Utf8ByteTokenizer):
    """A tokenizer whose span text depends on where the window starts.

    What SentencePiece does with a leading space, in miniature. No such
    tokenizer is installed here, so the probe's ability to reject one has to be
    modelled or it is only ever exercised on tokenizers that pass.
    """

    def decode(self, token_ids, skip_special_tokens=True):
        text = super().decode(token_ids, skip_special_tokens)
        return text.lstrip(" ") if text.startswith(" ") else text


@pytest.fixture(autouse=True)
def _leave_delta_reuse_as_found():
    """Process-wide state; a test that left it on would change how every later
    test decodes."""
    from dataclasses import replace as _replace

    before = _replace(streaming_dispatch._DELTA_REUSE)
    yield
    for f, v in vars(before).items():
        setattr(streaming_dispatch._DELTA_REUSE, f, v)


def test_reuse_is_off_until_a_tokenizer_has_been_checked():
    """Default-off, so a path that forgets to probe is slow rather than wrong."""
    assert streaming_dispatch._DeltaReuse().enabled is False


def test_a_well_behaved_tokenizer_is_accepted_and_skips_a_decode():
    tokenizer = _CountingTokenizer()
    assert enable_delta_reuse(tokenizer, "auto", audit_every=10**9) is True

    tokenizer.calls = 0
    state = IncrementalStreamDetokenizer(tokenizer)
    text = "".join(state.update([b], b == ord("!")) for b in b"hello world!")

    assert text == "hello world!"
    # 12 tokens: without reuse each update decodes twice. The first update
    # is audited (calls starts at zero), so one extra.
    assert tokenizer.calls == 13


def test_a_position_sensitive_tokenizer_is_rejected():
    """The control the installed tokenizers cannot provide: the probe has to
    say no to something, or "it passed" carries no information."""
    assert enable_delta_reuse(_PositionSensitiveTokenizer(), "auto") is False
    assert streaming_dispatch._DELTA_REUSE.enabled is False


def test_output_is_identical_with_and_without_reuse():
    """The property that matters. Same streams, same merge depths, both paths."""
    tokenizer = _Utf8ByteTokenizer()
    payload = ("你好 🎉 world ゆ\t x  " * 3).encode()
    ids = list(payload)

    def run(chunk):
        state = IncrementalStreamDetokenizer(tokenizer)
        return "".join(
            state.update(ids[i : i + chunk], i + chunk >= len(ids))
            for i in range(0, len(ids), chunk)
        )

    for chunk in (1, 2, 3, 7, 64):
        enable_delta_reuse(tokenizer, "off")
        plain = run(chunk)
        enable_delta_reuse(tokenizer, "on")
        assert run(chunk) == plain == payload.decode(), f"chunk {chunk}"


def test_an_audit_mismatch_corrects_the_output_and_turns_reuse_off():
    """A wrong delta must not reach the client. The audit that catches it also
    has the decoded value in hand, so this call is answered correctly and only
    later calls lose the shortcut."""
    tokenizer = _Utf8ByteTokenizer()
    enable_delta_reuse(tokenizer, "on", audit_every=1)
    state = IncrementalStreamDetokenizer(tokenizer)
    state.update(list(b"ab"), False)
    state.last_delta = "wrong"

    delta = state.update(list(b"cd"), True)

    assert delta == "cd"
    assert streaming_dispatch._DELTA_REUSE.enabled is False
    assert streaming_dispatch._DELTA_REUSE.mismatches == 1


def test_disabling_reaches_streams_already_in_flight():
    """The flag is read per call, not captured per stream: four thousand
    streams are open when a mismatch is found, and they must all stop trusting
    reuse, not just the next one to arrive."""
    tokenizer = _Utf8ByteTokenizer()
    enable_delta_reuse(tokenizer, "on", audit_every=10**9)
    inflight = IncrementalStreamDetokenizer(tokenizer)
    inflight.update(list(b"xy"), False)

    streaming_dispatch._DELTA_REUSE.enabled = False
    counting = _CountingTokenizer()
    inflight.tokenizer = counting
    counting.calls = 0
    inflight.update(list(b"z"), True)

    assert counting.calls == 2, "still trusting the delta after reuse was off"


def test_an_empty_update_does_not_disturb_the_stream():
    """`update()` documents this as legal, and delta reuse is why it now
    matters: an empty call that touched `last_delta` would leave it naming a
    span it does not describe, and every later delta would be cut at the wrong
    length. Nothing would raise."""
    tokenizer = _Utf8ByteTokenizer()
    enable_delta_reuse(tokenizer, "on", audit_every=10**9)
    state = IncrementalStreamDetokenizer(tokenizer)

    assert state.update(list(b"ab"), False) == "ab"
    carried = state.last_delta
    assert state.update([], False) == ""
    assert state.last_delta == carried
    assert state.update(list(b"cd"), True) == "cd"


@pytest.mark.parametrize("spelling", ["off", "OFF", "Off", " off ", "  OfF"])
def test_the_kill_switch_is_not_case_sensitive(spelling):
    """Whoever sets this has just read the mismatch ERROR. A spelling that
    silently means `auto` hands them back the behaviour they were disabling,
    and `_Utf8ByteTokenizer` passes the probe, so `auto` here means on."""
    assert enable_delta_reuse(_Utf8ByteTokenizer(), spelling) is False
    assert streaming_dispatch._DELTA_REUSE.enabled is False


@pytest.mark.parametrize("spelling", ["0", "false", "no", "disabled", "", "atuo"])
def test_an_unreadable_mode_leaves_reuse_off(spelling, caplog):
    """`auto` is the wrong fallback for a value nobody can parse: every
    plausible misspelling here is someone reaching for off, and the tokenizer
    that would then be trusted was never the thing in doubt."""
    with caplog.at_level(logging.WARNING):
        assert enable_delta_reuse(_Utf8ByteTokenizer(), spelling) is False
    assert "unknown delta reuse mode" in caplog.text


@pytest.mark.parametrize("spelling", ["on", "ON", " On "])
def test_the_mode_still_pins_reuse_on(spelling):
    """The negative control for the two above: normalising must not have made
    every value mean off."""
    assert enable_delta_reuse(_Utf8ByteTokenizer(), spelling) is True


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("", streaming_dispatch.DEFAULT_AUDIT_EVERY),
        (None, streaming_dispatch.DEFAULT_AUDIT_EVERY),
        ("abc", streaming_dispatch.DEFAULT_AUDIT_EVERY),
        ("1_000ms", streaming_dispatch.DEFAULT_AUDIT_EVERY),
        ("0", streaming_dispatch.DEFAULT_AUDIT_EVERY),
        ("-5", streaming_dispatch.DEFAULT_AUDIT_EVERY),
        ("7", 7),
        (250, 250),
    ],
)
def test_an_unusable_audit_interval_falls_back_instead_of_raising(value, expected):
    """This is read as an argument at a callsite that has already loaded the
    weights onto eight GPUs, so a typo in an optional tuning knob must not be
    what ends the process -- and `0`, which reads like "never audit", would
    divide by zero."""
    assert enable_delta_reuse(_Utf8ByteTokenizer(), "on", value) is True
    assert streaming_dispatch._DELTA_REUSE.audit_every == expected


@pytest.mark.parametrize("value", ["abc", "1_000ms", "", "0", "-5"])
def test_the_audit_interval_env_survives_a_typo(value, monkeypatch):
    """The raise this guards against is in `envs`, one frame above
    `enable_delta_reuse`, where its own try/except cannot reach it. Its sibling
    `ATOM_GC_THRESHOLD` is text for the same reason."""
    from atom.utils import envs

    monkeypatch.setenv("ATOM_DETOKENIZER_AUDIT_EVERY", value)

    assert enable_delta_reuse(
        _Utf8ByteTokenizer(), "on", envs.ATOM_DETOKENIZER_AUDIT_EVERY
    )


@pytest.mark.parametrize(
    ("raw", "normalized"), [("OFF", "off"), (" Auto ", "auto"), ("On", "on")]
)
def test_the_mode_env_is_normalized_where_it_is_read(raw, normalized, monkeypatch):
    """Normalising in both places would be redundant; normalising in neither is
    what shipped. `envs` owns it, and `enable_delta_reuse` keeps its own guard
    because it is also called directly."""
    from atom.utils import envs

    monkeypatch.setenv("ATOM_DETOKENIZER_DELTA_REUSE", raw)

    assert envs.ATOM_DETOKENIZER_DELTA_REUSE == normalized
