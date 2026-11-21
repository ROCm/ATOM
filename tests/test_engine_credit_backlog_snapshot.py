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


def _input_thread_queue(monkeypatch, *, report_backlog):
    """What the Core's input thread queues for one ADD of a request it admits
    and one it refuses (it does not encode: sent refused), then SHUTDOWN."""
    from contextlib import nullcontext

    from aiter_stub import stubbed_aiter

    with stubbed_aiter():
        from atom.model_engine import engine_core as core
    from atom.model_engine.admission import AdmissionContext
    from atom.model_engine.sequence import Sequence
    from atom.model_engine.zmq_protocol.codec import encode_input

    good = Sequence([1, 3, 4], block_size=4, id=1)
    bad = Sequence([1, 3, 4], block_size=4, id=2, dp_session_id="s\ud800")
    frames = iter(
        [
            encode_input(EngineCoreRequestType.ADD, [good, bad]),
            encode_input(EngineCoreRequestType.SHUTDOWN),
        ]
    )
    sock = SimpleNamespace(send=lambda _: None, recv=lambda **_: next(frames))
    monkeypatch.setattr(core, "make_zmq_socket", lambda *a, **kw: nullcontext(sock))
    monkeypatch.setattr(
        core.zmq,
        "Poller",
        lambda: SimpleNamespace(
            register=lambda *a: None, poll=lambda: [(sock, core.zmq.POLLIN)]
        ),
    )
    engine = core.EngineCore.__new__(core.EngineCore)
    engine._kv_progress_event = core.threading.Event()
    engine.label = "backlog-test"
    engine.input_queue = Queue()
    engine.output_queue = Queue()
    engine._report_backlog = report_backlog
    engine._admission = AdmissionContext(
        SimpleNamespace(
            kv_cache_block_size=4,
            speculative_config=None,
            hf_config=SimpleNamespace(model_type="llama"),
            end_token_ids=(2,),
            enable_dp_attention=False,
            kv_transfer_config=None,
            enable_rapidserve=False,
        )
    )
    engine.process_input_sockets("input", "control")
    queued = []
    while not engine.input_queue.empty():
        queued.append(engine.input_queue.get_nowait())
    kind, [rejected] = engine.output_queue.get_nowait()
    assert (kind, rejected.seq_id, rejected.rejected) == ("REJECTED", 2, True)
    return core, engine, queued


def test_a_rejected_request_counts_as_ingested(monkeypatch):
    """The CoreManager's ledger matches the ingest count against every request
    it sent: one the Core rejects at admission must count too, or it stays
    a phantom backlog on that rank. It counts in queue order, after the
    batch's admitted requests."""
    core, engine, queued = _input_thread_queue(monkeypatch, report_backlog=True)
    admitted, marker, _shutdown = queued
    assert [seq.id for seq in admitted] == [1]
    assert marker == (core.INGEST_REJECTED, 1)
    extended = []
    engine.scheduler = SimpleNamespace(extend=extended.extend)
    engine._ingested_seqs = 0
    engine.input_queue.put(admitted)
    engine.input_queue.put(marker)
    assert engine.pull_and_process_input_queue() is False
    assert [seq.id for seq in extended] == [1]
    assert engine._ingested_seqs == 2


def test_without_backlog_reports_a_rejection_puts_nothing_on_the_input_queue(
    monkeypatch,
):
    # RapidServe's DecodeEngineCore reads the input queue as Sequence lists.
    _, _, queued = _input_thread_queue(monkeypatch, report_backlog=False)
    assert [type(item) for item in queued] == [list, list]
