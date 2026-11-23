# SPDX-License-Identifier: MIT
"""A RapidServe decode process runs its model once before it reports READY.

Decode builds its model on meta and skips the runner's warmup; its weights and
KV cache arrive later from the prefill process. Without CUDA graph capture
(``--enforce-eager``) nothing else ran the model before the first request, so
torch.compile traced a one-token decode: a size-1 token dim ``mark_dynamic``
cannot keep dynamic, the graph compiled for one token, and the next forward
raised ``IndexError`` in the piecewise backend (jgong5/ATOM#1598).

``DecodeEngineCore.__init__`` is driven for real here; only what it would
reach across processes is replaced: the base engine's runner start-up, the
prefill process's kvcache bundle (a ZMQ PUSH this test sends), and the
runner process behind ``call_func``.
"""

import pickle
import threading
from types import SimpleNamespace

import pytest
import zmq
from aiter_stub import atom_without_aiter

READY = "READY"


@pytest.fixture(scope="module")
def modules():
    with atom_without_aiter() as load:
        yield SimpleNamespace(
            engine_core=load("atom.model_engine.engine_core"),
            model_runner=load("atom.model_engine.model_runner"),
        )


class _RunnerProcess:
    """``AsyncIOProcManager.call_func`` as the engine sees it. The worker
    sends a reply only for a return that is not None, so a ``wait_out=True``
    call of a method returning None never comes back: raised here instead of
    hanging the test, as it hung the decode engine before its first request."""

    def __init__(self, runner, log):
        self.runner, self.log = runner, log

    def call_func(self, func_name, *args, wait_out=False):
        self.log.append(func_name)
        out = getattr(self.runner, func_name)(*args)
        if wait_out and out is None:
            raise AssertionError(
                f"{func_name} returned None: the worker sends no reply, so "
                "call_func(wait_out=True) would wait forever"
            )
        return out


def _decode_runner(model_runner, log):
    """A decode ``RapidServeModelRunner`` whose GPU work is stubbed; the
    stubs return what the real methods do."""
    runner = model_runner.RapidServeModelRunner.__new__(
        model_runner.RapidServeModelRunner
    )
    runner.import_kv_cache_ipc_handle = lambda paths, num_blocks: True
    runner.create_decode_stream_pool = lambda: True
    runner.capture_cudagraph = lambda: (0.0, [1, 2], 0)
    # The real warmup returns None; it runs the dummy forward torch.compile
    # first traces.
    runner.warmup_model = lambda: log.append("model forward (warmup)")
    return runner


def _start_decode(modules, monkeypatch, tmp_path, enforce_eager):
    """Run ``DecodeEngineCore.__init__`` and return the runner calls and
    READY, in the order they happened."""
    engine_core = modules.engine_core
    log = []
    runner = _decode_runner(modules.model_runner, log)

    def base_init(self, config, input_address, output_address):
        # What the base engine leaves behind on the decode side: the runner
        # process (weights already imported), no scheduler yet.
        self.runner_mgr = _RunnerProcess(runner, log)
        self.state_runtime = None
        self.utility_handler = SimpleNamespace(scheduler=None)

    monkeypatch.setattr(engine_core.EngineCore, "__init__", base_init)
    monkeypatch.setattr(
        engine_core.EngineCore, "_send_ready_signal", lambda self: log.append(READY)
    )
    monkeypatch.setattr(
        engine_core.EngineCore, "_freeze_after_startup", lambda self: None
    )
    monkeypatch.setattr(
        engine_core, "DecodeScheduler", lambda *a, **k: SimpleNamespace()
    )

    kv_addr = f"ipc://{tmp_path}/kvcache"
    config = SimpleNamespace(
        disagg_d2p_addr=f"ipc://{tmp_path}/d2p",
        disagg_p2d_addr=f"ipc://{tmp_path}/p2d",
        disagg_weight_ipc_addr=f"ipc://{tmp_path}/weights",
        disagg_weight_ack_addr=f"ipc://{tmp_path}/ack",
        disagg_kvcache_ipc_addr=kv_addr,
        disagg_cu_shm_name=None,
        enforce_eager=enforce_eager,
        num_kvcache_blocks=None,
    )

    # The prefill process's side of round 2: the kvcache bundle.
    ctx = zmq.Context()
    push = ctx.socket(zmq.PUSH)
    push.bind(kv_addr)
    bundle = pickle.dumps({"kvcache_args": ["handles"], "num_kvcache_blocks": 8})
    sender = threading.Thread(target=push.send, args=(bundle,), daemon=True)
    sender.start()
    core = None
    try:
        core = engine_core.DecodeEngineCore(config, "in", "out")
    finally:
        sender.join(timeout=5)
        push.close(linger=0)
        ctx.term()
        if core is not None:
            core._disagg_ctx.destroy(linger=0)
    assert config.num_kvcache_blocks == 8
    return log


def test_eager_decode_compiles_its_model_before_ready(modules, monkeypatch, tmp_path):
    log = _start_decode(modules, monkeypatch, tmp_path, enforce_eager=True)
    assert READY in log, log
    before_ready = log[: log.index(READY)]
    # The warmup the runner skipped at construction runs once the KV cache is
    # imported, on the decode stream pool its forward needs.
    assert "model forward (warmup)" in before_ready, (
        f"an eager decode engine reported READY without running its model, "
        f"so the first request's one-token decode is what torch.compile "
        f"traces: {log}"
    )
    assert before_ready.index("import_kv_cache_ipc_handle") < before_ready.index(
        "create_decode_stream_pool"
    )
    assert before_ready.index("create_decode_stream_pool") < before_ready.index(
        "model forward (warmup)"
    )
    assert "capture_cudagraph" not in log


def test_graph_decode_captures_before_ready_and_runs_no_warmup(
    modules, monkeypatch, tmp_path
):
    log = _start_decode(modules, monkeypatch, tmp_path, enforce_eager=False)
    assert READY in log, log
    before_ready = log[: log.index(READY)]
    assert "capture_cudagraph" in before_ready
    assert "model forward (warmup)" not in log
