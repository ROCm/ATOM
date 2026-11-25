# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""The engine loop hands the GIL to its process's other threads (`yield_gil`).

`EngineCore.busy_loop` blocks nowhere while it has nothing to schedule, and
nothing blocks between queuing a step's outputs and the loop's next blocking
call. A thread ready to run meanwhile -- the output sender with a finished
request's last chunk -- otherwise waits out CPython's switch interval (5 ms).
"""

import queue
from types import SimpleNamespace

from aiter_stub import stubbed_aiter

with stubbed_aiter():
    from atom.model_engine import engine_core
    from atom.model_engine.engine_core import EngineCore


def _idle_engine(passes):
    return SimpleNamespace(
        label="test",
        utility_queue=None,
        utility_handler=SimpleNamespace(
            process_queue=lambda *args: None, push_metrics=lambda **kw: None
        ),
        pull_and_process_input_queue=lambda: next(passes),
        _is_engine_held=lambda: False,
        has_pending_kv_work=lambda: False,
        _drain_kv_work_at_exit=lambda: None,
        scheduler=SimpleNamespace(
            heartbeat_throughput=lambda _: None,
            is_finished=lambda: True,  # nothing to schedule: every pass is idle
            engine_stats=SimpleNamespace(note_idle=lambda _: None),
            publish_kv_events=lambda: None,
            shutdown_kv_events=lambda: None,
            close_kv_connector=lambda: None,
        ),
    )


def test_every_idle_pass_yields_the_gil(monkeypatch):
    yields = []
    monkeypatch.setattr(engine_core, "yield_gil", lambda: yields.append(True))

    EngineCore.busy_loop(_idle_engine(iter([False, False, True])))

    assert yields == [True, True], "one yield per idle pass, none at shutdown"


def test_postprocess_yields_after_queuing_its_outputs(monkeypatch):
    events = []
    monkeypatch.setattr(engine_core, "yield_gil", lambda: events.append("yield"))

    def postprocess(seqs, fwd_out, stream_output_queue, batch):
        stream_output_queue.put([("seq", "chunk")])
        return ["finished seq"]

    engine = SimpleNamespace(
        _report_backlog=False,
        stream_output_queue=queue.Queue(),
        output_queue=SimpleNamespace(
            put_nowait=lambda item: events.append(
                item[0] if isinstance(item, tuple) else "finished"
            )
        ),
        _release_request_resources=lambda seqs: None,
        scheduler=SimpleNamespace(postprocess=postprocess),
    )

    EngineCore._postprocess(engine, [], None, None)

    # The last chunk and the finished list are queued before the sender is
    # handed the GIL.
    assert events == ["STREAM", "finished", "yield"]
