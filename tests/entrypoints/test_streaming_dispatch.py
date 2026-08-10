import asyncio

from atom.entrypoints.openai.streaming_dispatch import (
    IncrementalStreamDetokenizer,
    StreamBatchDispatcher,
)


class _Utf8ByteTokenizer:
    def decode(self, token_ids, skip_special_tokens=True):
        return bytes(token_ids).decode("utf-8", errors="replace")


class _ImmediateLoop:
    def __init__(self):
        self.calls = []

    def call_soon_threadsafe(self, callback, *args):
        self.calls.append((callback, args))
        callback(*args)


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
        queue=direct_queue,
        state_key="request-1",
        chunk={"token_ids": [ord("A")], "finished": True},
    )
    dispatcher.enqueue(
        loop=loop,
        queue=tagged_queue,
        state_key=("request-2", 0),
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

    dispatcher.enqueue(
        loop=loop,
        queue=queue,
        state_key=("request", 0),
        chunk={"token_ids": [0xE4], "finished": False},
        tag=0,
    )
    dispatcher.enqueue(
        loop=loop,
        queue=queue,
        state_key=("request", 1),
        chunk={"token_ids": [ord("X")], "finished": True},
        tag=1,
    )
    dispatcher.flush()

    assert queue.get_nowait()[1]["text"] == ""
    assert queue.get_nowait()[1]["text"] == "X"

    dispatcher.enqueue(
        loop=loop,
        queue=queue,
        state_key=("request", 0),
        chunk={"token_ids": [0xBD, 0xA0], "finished": True},
        tag=0,
    )
    dispatcher.flush()

    assert queue.get_nowait()[1]["text"] == "你"


def test_discard_request_drops_partial_direct_and_fanout_state():
    dispatcher = StreamBatchDispatcher(_Utf8ByteTokenizer())
    loop = _ImmediateLoop()
    queue = asyncio.Queue()

    for state_key in ("request", ("request", 0)):
        dispatcher.enqueue(
            loop=loop,
            queue=queue,
            state_key=state_key,
            chunk={"token_ids": [0xE4], "finished": False},
        )
    dispatcher.flush()
    dispatcher.discard_request("request")

    for state_key in ("request", ("request", 0)):
        dispatcher.enqueue(
            loop=loop,
            queue=queue,
            state_key=state_key,
            chunk={"token_ids": [ord("A")], "finished": True},
        )
    dispatcher.flush()

    assert queue.get_nowait()["text"] == ""
    assert queue.get_nowait()["text"] == ""
    assert queue.get_nowait()["text"] == "A"
    assert queue.get_nowait()["text"] == "A"
