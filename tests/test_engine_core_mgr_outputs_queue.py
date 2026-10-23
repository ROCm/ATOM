# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Finished sequences go on `outputs_queue` only where something reads it.

`LLMEngine.step()` / `generate()` and asyncio mode read the queue. The API
server, the one caller that wires the stream-flush hook, delivers through the
per-sequence callbacks and never does. The queue used to be filled anyway, so
online every finished `Sequence` stayed reachable for the life of the process
and each gen-2 GC pass of the API process grew with the requests served.
"""

import pickle
import uuid
from types import SimpleNamespace

import pytest
import zmq

from atom.model_engine.engine_core_mgr import CoreManager
from atom.model_engine.engine_core_protocol import EngineCoreRequestType


def _queued_after_add(*, flush_hook, asyncio_mode):
    """Ids of what a real output thread queues for one ADD of finished seqs."""
    mgr = CoreManager.__new__(CoreManager)
    mgr._init_shared_state(
        SimpleNamespace(
            dp_load_balance="round_robin",
            tp_world_size=1,
            prefill_context_parallel_size=1,
            asyncio_mode=asyncio_mode,
        ),
        label="test-mgr",
        local_engine_count=1,
    )
    mgr._flush_stream_batch_fn = flush_hook
    tag = uuid.uuid4().hex
    out_addr, stop_addr = f"inproc://out-{tag}", f"inproc://stop-{tag}"
    pull = mgr.ctx.socket(zmq.PULL)
    pull.bind(out_addr)
    push = mgr.ctx.socket(zmq.PUSH)
    push.connect(out_addr)
    stop = mgr.ctx.socket(zmq.PAIR)
    stop.connect(stop_addr)
    thread = mgr._create_output_thread(0, pull, stop_addr)
    thread.start()
    try:
        seqs = [SimpleNamespace(id=1), SimpleNamespace(id=2)]
        push.send(pickle.dumps((EngineCoreRequestType.ADD, seqs)))
        # The socket is FIFO: once this reply is routed, the ADD was handled.
        with mgr._rpc_router.register("after") as replies:
            push.send(
                pickle.dumps(
                    (
                        EngineCoreRequestType.UTILITY_RESPONSE,
                        {"cmd": "clear_kv_cache", "request_id": "after", "result": 1},
                    )
                )
            )
            replies.get(timeout=5)
        queued = []
        while not mgr.outputs_queue.empty():
            queued.append([seq.id for seq in mgr.outputs_queue.get_nowait()])
        return queued
    finally:
        try:
            stop.send(b"", zmq.NOBLOCK)
        except zmq.Again:
            pass  # the thread already ended
        thread.join(timeout=10)
        mgr.ctx.destroy(linger=0)


@pytest.mark.parametrize(
    "flush_hook,asyncio_mode,expected",
    [
        (None, False, [[1, 2]]),  # offline: LLMEngine.step() / generate()
        (None, True, [[1, 2]]),
        (lambda: None, True, [[1, 2]]),  # asyncio mode drains the queue
        (lambda: None, False, []),  # API server: nothing reads it
    ],
    ids=["offline", "offline-asyncio", "online-asyncio", "online"],
)
def test_finished_seqs_are_queued_only_for_a_reader(flush_hook, asyncio_mode, expected):
    assert (
        _queued_after_add(flush_hook=flush_hook, asyncio_mode=asyncio_mode) == expected
    )
