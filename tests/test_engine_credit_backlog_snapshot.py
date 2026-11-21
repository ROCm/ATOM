# SPDX-License-Identifier: MIT
import sys
from queue import Queue
from types import ModuleType, SimpleNamespace

from atom.model_engine.engine_core_protocol import EngineCoreRequestType


def test_credit_ack_snapshot_does_not_suppress_finished_backlog(monkeypatch):
    # Backlog IPC does not construct the AITER-backed worker RPC manager.
    # Keep the same import boundary as the engine scheduling tests.
    async_proc = ModuleType("atom.model_engine.async_proc")
    async_proc.AsyncIOProcManager = object
    monkeypatch.setitem(sys.modules, "atom.model_engine.async_proc", async_proc)
    from atom.model_engine import engine_core

    engine = engine_core.EngineCore.__new__(engine_core.EngineCore)
    engine.input_queue = Queue()
    engine.output_queue = Queue()
    engine._ingested_seqs = 2
    engine._last_backlog_report = (2, 0)
    engine._next_backlog_report = 0
    seq = SimpleNamespace(num_completion_tokens=0, is_partial_prefill=False)
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
    assert engine.output_queue.get_nowait() == ("PREFILL_BACKLOG", (2, 0, ()))


def test_a_rejected_request_counts_as_ingested(monkeypatch):
    """The CoreManager's ledger matches the ingest count against every request
    it sent: one the Core rejects at admission must count too, or it stays
    a phantom backlog on that rank."""
    async_proc = ModuleType("atom.model_engine.async_proc")
    async_proc.AsyncIOProcManager = object
    monkeypatch.setitem(sys.modules, "atom.model_engine.async_proc", async_proc)
    from atom.model_engine import engine_core

    engine = engine_core.EngineCore.__new__(engine_core.EngineCore)
    engine.label = "test"
    engine.input_queue = Queue()
    engine.output_queue = Queue()
    engine._ingested_seqs = 0
    engine._report_backlog = True
    engine._kv_progress_event = SimpleNamespace(set=lambda: None)
    engine._offer_tier_lookups = lambda seqs: None
    built = SimpleNamespace(id=1, status=None)

    def build(spec, context):
        if spec.seq_id == 2:
            raise ValueError("refused")
        return built

    monkeypatch.setattr(engine_core, "build_sequence", build)
    monkeypatch.setattr(engine_core.tracing, "enqueue_many", lambda *a, **k: None)
    monkeypatch.setattr(engine_core.SchedulerMetrics, "enqueue", lambda *a, **k: None)
    engine._admission = None
    extended = []
    engine.scheduler = SimpleNamespace(extend=extended.extend)
    specs = [SimpleNamespace(seq_id=1), SimpleNamespace(seq_id=2)]
    engine._admit(specs, [(3, "undecodable", 0)], received_at=0.0)
    assert engine.pull_and_process_input_queue() is False
    assert extended == [built]
    assert engine._ingested_seqs == 3
    kind, [(seq_two, *_), (seq_three, *_)] = engine.output_queue.get_nowait()
    assert (kind, sorted([seq_two, seq_three])) == ("REJECTED", [2, 3])


def test_without_backlog_reports_a_rejection_puts_nothing_on_the_input_queue(
    monkeypatch,
):
    # RapidServe's DecodeEngineCore reads the input queue as Sequence lists.
    async_proc = ModuleType("atom.model_engine.async_proc")
    async_proc.AsyncIOProcManager = object
    monkeypatch.setitem(sys.modules, "atom.model_engine.async_proc", async_proc)
    from atom.model_engine import engine_core

    engine = engine_core.EngineCore.__new__(engine_core.EngineCore)
    engine.label = "test"
    engine.input_queue = Queue()
    engine.output_queue = Queue()
    engine._admit([], [(3, "undecodable", 0)], received_at=0.0)
    assert engine.input_queue.empty()
    assert engine.output_queue.get_nowait()[0] == "REJECTED"
