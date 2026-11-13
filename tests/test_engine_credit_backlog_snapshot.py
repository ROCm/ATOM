# SPDX-License-Identifier: MIT
import sys
from queue import Queue
from types import ModuleType, SimpleNamespace

import pytest

from atom.model_engine.engine_core_protocol import EngineCoreRequestType


@pytest.mark.parametrize("core_class", ["EngineCore", "DecodeEngineCore"])
def test_credit_ack_snapshot_does_not_suppress_finished_backlog(
    monkeypatch, core_class
):
    # Backlog IPC does not construct the AITER-backed worker RPC manager.
    # Keep the same import boundary as the engine scheduling tests.
    async_proc = ModuleType("atom.model_engine.async_proc")
    async_proc.AsyncIOProcManager = object
    monkeypatch.setitem(sys.modules, "atom.model_engine.async_proc", async_proc)
    from atom.model_engine import engine_core

    cls = getattr(engine_core, core_class)
    engine = cls.__new__(cls)
    engine.input_queue = Queue()
    engine.output_queue = Queue()
    engine._ingested_seqs = 2
    engine._last_backlog_report = (2, 0)
    engine._next_backlog_report = 0
    seq = SimpleNamespace(num_completion_tokens=0)
    engine.scheduler = SimpleNamespace(
        waiting=[],
        running=[seq],
        pending_prefill_tokens=lambda seq, estimate: 512,
        correct_prefix_estimates=lambda updates: None,
        allocate_waiting=list,
    )
    engine.input_queue.put((EngineCoreRequestType.DP_PREFIX_CREDIT, [(2, 0)]))
    assert engine.pull_and_process_input_queue() is False
    assert engine.output_queue.get_nowait() == ("PREFILL_BACKLOG", (2, 512, [(2, 0)]))
    seq.num_completion_tokens = 1
    engine._report_prefill_backlog(1.0)
    assert engine.output_queue.get_nowait() == ("PREFILL_BACKLOG", (2, 0))
