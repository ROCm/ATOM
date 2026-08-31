import asyncio

import pytest

from atom.entrypoints.openai.streaming_dispatch import (
    IncrementalStreamDetokenizer,
    StreamBatchDispatcher,
    StreamOutputCollector,
    merge_chunk,
)


class _Utf8ByteTokenizer:
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


def test_incremental_detokenizer_holds_incomplete_utf8():
    detokenizer = IncrementalStreamDetokenizer(_Utf8ByteTokenizer())

    assert detokenizer.update([0xE4], finished=False) == ""
    assert detokenizer.update([0xBD, 0xA0], finished=False) == "你"
    assert detokenizer.update([ord("!")], finished=True) == "!"


def test_dispatcher_batches_direct_and_tagged_chunks_per_loop():
    dispatcher = StreamBatchDispatcher(_Utf8ByteTokenizer())
    loop = _ImmediateLoop()
    direct_queue = asyncio.Queue()
    tagged_queue = asyncio.Queue()

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
    assert direct_queue.get_nowait()["text"] == "A"
    sibling_index, chunk = tagged_queue.get_nowait()
    assert sibling_index == 0
    assert chunk["text"] == "B"


def test_dispatcher_keeps_fanout_detokenizer_state_separate():
    dispatcher = StreamBatchDispatcher(_Utf8ByteTokenizer())
    loop = _ImmediateLoop()
    queue = asyncio.Queue()
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

    assert queue.get_nowait()[1]["text"] == ""
    assert queue.get_nowait()[1]["text"] == "X"

    # Sibling 0's half character survives sibling 1 finishing in between.
    dispatcher.enqueue(
        loop=loop,
        collector=queue,
        state=sibling_0,
        chunk={"token_ids": [0xBD, 0xA0], "finished": True},
        tag=0,
    )
    dispatcher.flush()

    assert queue.get_nowait()[1]["text"] == "你"


def test_a_fresh_stream_does_not_inherit_a_half_decoded_character():
    """Each stream's detokenizer is its own object, so bytes cannot leak over."""
    dispatcher = StreamBatchDispatcher(_Utf8ByteTokenizer())
    loop = _ImmediateLoop()
    queue = asyncio.Queue()

    for _ in range(2):
        dispatcher.enqueue(
            loop=loop,
            collector=queue,
            state=dispatcher.new_state(),
            chunk={"token_ids": [0xE4], "finished": False},
        )
    dispatcher.flush()

    for _ in range(2):
        dispatcher.enqueue(
            loop=loop,
            collector=queue,
            state=dispatcher.new_state(),
            chunk={"token_ids": [ord("A")], "finished": True},
        )
    dispatcher.flush()

    assert queue.get_nowait()["text"] == ""
    assert queue.get_nowait()["text"] == ""
    assert queue.get_nowait()["text"] == "A"
    assert queue.get_nowait()["text"] == "A"


def test_collector_hands_over_a_lone_chunk_untouched():
    """A consumer that keeps up must see exactly what the queue used to give."""
    collector = StreamOutputCollector("request-1")
    chunk = {"token_ids": [1], "text": "a", "finished": False}
    collector.put_nowait(chunk)

    assert _resolve(collector.get()) is chunk


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
    """The dispatcher must stay stateless between streams.

    Detokenizers used to live in a dict here: first behind a lock that cost 27%
    of the API server's CPU, then lock-free with an index two threads had to
    keep in agreement and teardown had to remember to clear -- draining 8192
    streams cost 894 ms of scanning, and a missed removal leaked a detokenizer
    whose token list grows without bound. Now each one belongs to the engine
    callback that feeds it, so there is nothing here to leak or to race on.
    """
    dispatcher = StreamBatchDispatcher(_Utf8ByteTokenizer())
    loop = _ImmediateLoop()
    queue = asyncio.Queue()

    for _ in range(64):
        dispatcher.enqueue(
            loop=loop,
            collector=queue,
            state=dispatcher.new_state(),
            chunk={"token_ids": [ord("A")], "finished": True},
        )
    dispatcher.flush()

    assert vars(dispatcher).keys() == {"tokenizer", "_thread_local"}


def test_each_stream_gets_its_own_detokenizer():
    dispatcher = StreamBatchDispatcher(_Utf8ByteTokenizer())

    first, second = dispatcher.new_state(), dispatcher.new_state()

    assert first is not second
    assert not first.tokens and not second.tokens
