# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""What the engine wire protocol's tests share: a Core's admission context,
and an EngineCore input thread fed messages by hand."""

from contextlib import nullcontext
from queue import Queue
from types import SimpleNamespace

from atom.model_engine.admission import AdmissionContext


def core_context(**config) -> AdmissionContext:
    """A Core's `AdmissionContext` over a minimal config; `config` overrides."""
    fields = {
        "kv_cache_block_size": 16,
        "speculative_config": None,
        "enable_dp_attention": False,
        "hf_config": SimpleNamespace(model_type="llama"),
        "end_token_ids": (2,),
        "kv_transfer_config": None,
        "enable_rapidserve": False,
        **config,
    }
    return AdmissionContext(SimpleNamespace(**fields))


def run_input_thread(monkeypatch, core, messages, *, context=None, **attrs):
    """`EngineCore.process_input_sockets` over `messages`, as one socket hands
    them over (end them with SHUTDOWN); the engine whose queues it filled.
    `core` is the `engine_core` module, `attrs` more engine attributes."""
    feed = iter(messages)
    sock = SimpleNamespace(send=lambda _: None, recv=lambda **_: next(feed))
    monkeypatch.setattr(core, "make_zmq_socket", lambda *a, **kw: nullcontext(sock))
    poller = SimpleNamespace(
        register=lambda *a: None, poll=lambda: [(sock, core.zmq.POLLIN)]
    )
    monkeypatch.setattr(core.zmq, "Poller", lambda: poller)
    engine = core.EngineCore.__new__(core.EngineCore)
    engine._kv_progress_event = core.threading.Event()
    engine.label = "test"
    engine.input_queue = Queue()
    engine.output_queue = Queue()
    engine._admission = context or core_context(kv_cache_block_size=4)
    for name, value in attrs.items():
        setattr(engine, name, value)
    engine.process_input_sockets("input", "control")
    return engine
