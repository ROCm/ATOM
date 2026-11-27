# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""Last PP stage over the IPC boundary: completions keep FIFO order and a
deferred output is collected only after the next forward is launched."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from aiter_stub import stubbed_aiter

with stubbed_aiter():
    from atom.model_engine.pp_engine_core import PPEngineCoreProc

PENDING = "PP_OUTPUT_PENDING"


def _batch(rid, output=True, ack=False):
    return SimpleNamespace(
        req_ids=[rid],
        connector_meta_output=None,
        produces_output=lambda: output,
        pp_completion_ack=ack,
    )


def _run(batches):
    """Drive the last stage's loop over ``batches``; the events it produced."""
    proc = PPEngineCoreProc.__new__(PPEngineCoreProc)
    proc.is_last = True
    proc.pp_last_async_out = True
    proc.kv_transfer_enabled = False
    proc.scheduler = MagicMock()
    proc.scheduler.kv_connector = None
    proc.utility_handler = MagicMock()
    proc.utility_queue = MagicMock()
    proc._is_engine_held = lambda: False
    feed, events, pending = list(batches), [], []
    # Stop a few idle polls after the feed is drained.
    polls = iter(range(len(batches) + 4))
    proc.pull_and_process_input_queue = lambda: next(polls) >= len(batches) + 3

    def recv_metadata(timeout_ms):
        events.append(("poll", timeout_ms, len(pending)))
        return feed.pop(0) if feed else None

    proc.pp_transport = MagicMock()
    proc.pp_transport.recv_metadata.side_effect = recv_metadata
    proc.pp_transport.send_completion.side_effect = lambda ids, out=None: events.append(
        ("done", ids[0], out)
    )

    def call_func(name, *args, wait_out=False):
        if name == "forward":
            events.append(("forward", args[0].req_ids[0]))
            if args[0].produces_output():
                pending.append(args[0].req_ids[0])
                return PENDING
            return SimpleNamespace(empty=True)
        if name == "pp_collect_output":
            events.append(("collect",))
            return f"out{pending.pop(0)}"
        return True

    proc.runner_mgr = MagicMock()
    proc.runner_mgr.call_func.side_effect = call_func
    proc._downstream_busy_loop()
    return events


def test_deferred_output_collected_after_next_forward():
    events = _run([_batch(1), _batch(2), _batch(3)])
    assert [e for e in events if e[0] != "poll"] == [
        ("forward", 1),
        ("forward", 2),
        ("collect",),
        ("done", 1, "out1"),
        ("forward", 3),
        ("collect",),
        ("done", 2, "out2"),
        ("collect",),
        ("done", 3, "out3"),
    ]
    # A deferred output never waits behind a blocking metadata poll.
    assert all(e[1] == 0 for e in events if e[0] == "poll" and e[2])


def test_middle_chunk_ack_waits_behind_deferred_output():
    events = _run([_batch(1), _batch(2, output=False, ack=True), _batch(3)])
    assert [e[1] for e in events if e[0] == "done"] == [1, 2, 3]
