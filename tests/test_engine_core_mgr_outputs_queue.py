# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Finished sequences go on `outputs_queue` only where something reads it.

`LLMEngine.step()` / `generate()` and asyncio mode read the queue. The OpenAI
server and the atomesh standalone service deliver through the per-sequence
callbacks and never do. The queue used to be filled anyway, so online every
finished `Sequence` stayed reachable for the life of the process and each
gen-2 GC pass grew with the requests served.

Whether the queue has a reader is declared by `consumes_outputs_queue`, which
the callback-only entrypoints clear. It used to be inferred from the API
server's stream-flush hook, which atomesh never sets, so atomesh still leaked.
"""

import asyncio
import pickle
import uuid
from types import SimpleNamespace

import pytest
import zmq

from atom.entrypoints.atomesh.atom_standalone_service import AtomEngineService
from atom.entrypoints.openai import api_server
from atom.model_engine.engine_core_mgr import CoreManager
from atom.model_engine.engine_core_protocol import EngineCoreRequestType
from atom.model_engine.llm_engine import LLMEngine


def _bare_manager(*, asyncio_mode=False):
    """A manager with its shared state set up and no engine processes spawned."""
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
    return mgr


def _queued_after_add(mgr):
    """Ids of what a real output thread queues for one ADD of finished seqs."""
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


def test_llm_engine_gets_its_finished_seqs():
    """Offline: nothing opts in, and LLMEngine.step() reads what was queued."""
    mgr = _bare_manager()
    assert _queued_after_add(mgr) == [[1, 2]]

    mgr.outputs_queue.put_nowait([SimpleNamespace(id=3)])
    engine = LLMEngine.__new__(LLMEngine)
    engine.core_mgr = mgr
    assert [seq.id for seq in engine.step()] == [3]


def test_asyncio_mode_gets_its_finished_seqs():
    mgr = _bare_manager(asyncio_mode=True)
    assert _queued_after_add(mgr) == [[1, 2]]


def test_api_server_wiring_queues_nothing():
    mgr = _bare_manager()
    api_server._deliver_through_stream_callbacks(mgr)
    assert mgr._flush_stream_batch_fn is api_server.flush_stream_batch
    assert _queued_after_add(mgr) == []


def test_atomesh_engine_service_queues_nothing():
    """atomesh delivers through callbacks too, and never sets the flush hook."""
    mgr = _bare_manager()
    service = AtomEngineService(SimpleNamespace(core_mgr=mgr), tokenizer=None)
    try:
        assert mgr._flush_stream_batch_fn is None
        assert _queued_after_add(mgr) == []
    finally:
        service.close()


def test_reading_a_queue_nothing_fills_fails_instead_of_blocking():
    """`generate()` / `step()` behind a callback-only entrypoint used to hang.

    Something is queued already, so a read that does not check the flag
    returns it here instead of blocking the test forever.
    """
    mgr = _bare_manager()
    mgr.consumes_outputs_queue = False
    mgr.outputs_queue.put_nowait([SimpleNamespace(id=1)])
    with pytest.raises(RuntimeError, match="consumes_outputs_queue"):
        mgr.get_output()

    mgr.async_output_queue = asyncio.Queue()
    mgr.async_output_queue.put_nowait([SimpleNamespace(id=1)])
    mgr._output_handler_task = None
    with pytest.raises(RuntimeError, match="consumes_outputs_queue"):
        asyncio.run(mgr.get_output_async())
    mgr.ctx.term()
