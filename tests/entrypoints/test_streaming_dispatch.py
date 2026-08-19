import time

from atom.entrypoints.openai.streaming_dispatch import (
    IncrementalStreamDetokenizer,
    StreamBatchDispatcher,
)


class _Utf8ByteTokenizer:
    def decode(self, token_ids, skip_special_tokens=True):
        return bytes(token_ids).decode("utf-8", errors="replace")


class _BatchUtf8ByteTokenizer(_Utf8ByteTokenizer):
    is_fast = True

    def __init__(self):
        self.batch_decode_calls = []
        self.decode_calls = []

    def decode(self, token_ids, skip_special_tokens=True):
        self.decode_calls.append(list(token_ids))
        return super().decode(token_ids, skip_special_tokens)

    def batch_decode(self, token_ids, skip_special_tokens=True):
        batches = [list(ids) for ids in token_ids]
        self.batch_decode_calls.append(batches)
        return [
            bytes(ids).decode("utf-8", errors="replace") for ids in batches
        ]


class _ImmediateLoop:
    def __init__(self):
        self.calls = []

    def call_soon_threadsafe(self, callback, *args):
        self.calls.append((callback, args))
        callback(*args)


class _ImmediateCollector:
    def __init__(self, request_id=""):
        self.request_id = request_id
        self.items = []

    def put_nowait(self, payload):
        self.items.append(payload)

    def get_nowait(self):
        return self.items.pop(0)

    def empty(self):
        return not self.items


def _wait_for_collector(collector, timeout=5.0):
    deadline = time.monotonic() + timeout
    while collector.empty() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not collector.empty(), "timed out waiting for detokenizer process"


def test_incremental_detokenizer_holds_incomplete_utf8():
    detokenizer = IncrementalStreamDetokenizer(_Utf8ByteTokenizer())

    assert detokenizer.update([0xE4], finished=False) == ""
    assert detokenizer.update([0xBD, 0xA0], finished=False) == "你"
    assert detokenizer.update([ord("!")], finished=True) == "!"


def test_dispatcher_batch_decodes_across_streams():
    tokenizer = _BatchUtf8ByteTokenizer()
    dispatcher = StreamBatchDispatcher(tokenizer)
    loop = _ImmediateLoop()
    collector_1 = _ImmediateCollector("request-1")
    collector_2 = _ImmediateCollector("request-2")
    state_1 = dispatcher.new_state()
    state_2 = dispatcher.new_state()

    for collector, state, token in (
        (collector_1, state_1, ord("A")),
        (collector_2, state_2, ord("B")),
    ):
        dispatcher.enqueue(
            loop=loop,
            collector=collector,
            state=state,
            chunk={"token_ids": [token], "finished": False},
        )
    dispatcher.flush()

    assert tokenizer.batch_decode_calls == [[[ord("A")], [ord("B")]]]
    assert tokenizer.decode_calls == []
    assert collector_1.get_nowait()["text"] == "A"
    assert collector_2.get_nowait()["text"] == "B"

    tokenizer.batch_decode_calls.clear()
    for collector, state, token in (
        (collector_1, state_1, ord("C")),
        (collector_2, state_2, ord("D")),
    ):
        dispatcher.enqueue(
            loop=loop,
            collector=collector,
            state=state,
            chunk={"token_ids": [token], "finished": True},
        )
    dispatcher.flush()

    assert tokenizer.batch_decode_calls == [
        [[ord("A")], [ord("B")]],
        [[ord("A"), ord("C")], [ord("B"), ord("D")]],
    ]
    assert tokenizer.decode_calls == []
    assert collector_1.get_nowait()["text"] == "C"
    assert collector_2.get_nowait()["text"] == "D"


def test_dispatcher_batch_decode_preserves_incomplete_utf8():
    tokenizer = _BatchUtf8ByteTokenizer()
    dispatcher = StreamBatchDispatcher(tokenizer)
    loop = _ImmediateLoop()
    collector_1 = _ImmediateCollector("request-1")
    collector_2 = _ImmediateCollector("request-2")
    state_1 = dispatcher.new_state()
    state_2 = dispatcher.new_state()

    dispatcher.enqueue(
        loop=loop,
        collector=collector_1,
        state=state_1,
        chunk={"token_ids": [0xE4], "finished": False},
    )
    dispatcher.enqueue(
        loop=loop,
        collector=collector_2,
        state=state_2,
        chunk={"token_ids": [ord("X")], "finished": True},
    )
    dispatcher.flush()

    assert collector_1.get_nowait()["text"] == ""
    assert collector_2.get_nowait()["text"] == "X"

    dispatcher.enqueue(
        loop=loop,
        collector=collector_1,
        state=state_1,
        chunk={"token_ids": [0xBD, 0xA0], "finished": True},
    )
    dispatcher.flush()

    assert collector_1.get_nowait()["text"] == "你"


def test_dispatcher_can_disable_batch_decode():
    tokenizer = _BatchUtf8ByteTokenizer()
    dispatcher = StreamBatchDispatcher(tokenizer, disable_batch_decode=True)
    loop = _ImmediateLoop()
    collector_1 = _ImmediateCollector("request-1")
    collector_2 = _ImmediateCollector("request-2")

    for collector, state, token in (
        (collector_1, dispatcher.new_state(), ord("A")),
        (collector_2, dispatcher.new_state(), ord("B")),
    ):
        dispatcher.enqueue(
            loop=loop,
            collector=collector,
            state=state,
            chunk={"token_ids": [token], "finished": True},
        )
    dispatcher.flush()

    assert tokenizer.batch_decode_calls == []
    assert tokenizer.decode_calls == [[ord("A")], [ord("B")]]
    assert collector_1.get_nowait()["text"] == "A"
    assert collector_2.get_nowait()["text"] == "B"


def test_dispatcher_can_detokenize_in_independent_process():
    dispatcher = StreamBatchDispatcher(
        _BatchUtf8ByteTokenizer(),
        use_process=True,
        process_count=2,
        process_start_method="spawn",
    )
    loop = _ImmediateLoop()
    collector_1 = _ImmediateCollector("request-1")
    collector_2 = _ImmediateCollector("request-2")
    state_1 = dispatcher.new_state()
    state_2 = dispatcher.new_state()
    try:
        assert len(dispatcher._processes) == 2
        assert all(process.is_alive() for process in dispatcher._processes)
        dispatcher.enqueue(
            loop=loop,
            collector=collector_1,
            state=state_1,
            chunk={"token_ids": [0xE4], "finished": False},
        )
        dispatcher.enqueue(
            loop=loop,
            collector=collector_2,
            state=state_2,
            chunk={"token_ids": [ord("X")], "finished": True},
        )
        dispatcher.flush()
        _wait_for_collector(collector_1)
        _wait_for_collector(collector_2)

        assert collector_1.get_nowait()["text"] == ""
        assert collector_2.get_nowait()["text"] == "X"

        dispatcher.enqueue(
            loop=loop,
            collector=collector_1,
            state=state_1,
            chunk={"token_ids": [0xBD, 0xA0], "finished": True},
        )
        dispatcher.flush()
        _wait_for_collector(collector_1)

        assert collector_1.get_nowait()["text"] == "你"
    finally:
        dispatcher.close()


def test_dispatcher_batches_direct_and_tagged_chunks_per_loop():
    dispatcher = StreamBatchDispatcher(_Utf8ByteTokenizer())
    loop = _ImmediateLoop()
    direct_collector = _ImmediateCollector("request-1")
    tagged_collector = _ImmediateCollector("request-2")

    dispatcher.enqueue(
        loop=loop,
        collector=direct_collector,
        state=dispatcher.new_state(),
        chunk={"token_ids": [ord("A")], "finished": True},
    )
    dispatcher.enqueue(
        loop=loop,
        collector=tagged_collector,
        state=dispatcher.new_state(),
        chunk={"token_ids": [ord("B")], "finished": True},
        tag=0,
    )
    dispatcher.flush()

    assert len(loop.calls) == 1
    assert direct_collector.get_nowait()["text"] == "A"
    sibling_index, chunk = tagged_collector.get_nowait()
    assert sibling_index == 0
    assert chunk["text"] == "B"


def test_dispatcher_keeps_fanout_detokenizer_state_separate():
    dispatcher = StreamBatchDispatcher(_Utf8ByteTokenizer())
    loop = _ImmediateLoop()
    collector = _ImmediateCollector("request")
    state_0 = dispatcher.new_state()
    state_1 = dispatcher.new_state()

    dispatcher.enqueue(
        loop=loop,
        collector=collector,
        state=state_0,
        chunk={"token_ids": [0xE4], "finished": False},
        tag=0,
    )
    dispatcher.enqueue(
        loop=loop,
        collector=collector,
        state=state_1,
        chunk={"token_ids": [ord("X")], "finished": True},
        tag=1,
    )
    dispatcher.flush()

    assert collector.get_nowait()[1]["text"] == ""
    assert collector.get_nowait()[1]["text"] == "X"

    dispatcher.enqueue(
        loop=loop,
        collector=collector,
        state=state_0,
        chunk={"token_ids": [0xBD, 0xA0], "finished": True},
        tag=0,
    )
    dispatcher.flush()

    assert collector.get_nowait()[1]["text"] == "你"


def test_dispatcher_merges_multiple_pending_chunks_per_stream():
    dispatcher = StreamBatchDispatcher(_Utf8ByteTokenizer())
    loop = _ImmediateLoop()
    collector = _ImmediateCollector("request")
    state = dispatcher.new_state()

    dispatcher.enqueue(
        loop=loop,
        collector=collector,
        state=state,
        chunk={"token_ids": [ord("A")], "finished": False},
    )
    dispatcher.enqueue(
        loop=loop,
        collector=collector,
        state=state,
        chunk={"token_ids": [ord("B")], "finished": True},
    )
    dispatcher.flush()

    chunk = collector.get_nowait()
    assert chunk["token_ids"] == [ord("A"), ord("B")]
    assert chunk["text"] == "AB"
    assert chunk["finished"] is True
