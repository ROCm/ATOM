"""Real 81198 CPU utility dispatcher and MessageQueue callable round trip."""

import importlib.util
import os
import queue
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
REPRO = Path("/app/test_scripts/dspark_pd/prefix-read-repro-81198-20261014")
sys.path[:0] = [
    str(ROOT / ".github/scripts/atomesh"),
    str(REPRO / "candidate"),
    str(REPRO),
    "/app/test_scripts/dspark_pd/prefix-read-observer-81198-20261014",
]
os.environ["PREFIX_READ_VARIANT"] = "candidate"
import pd_prefix_runtime as runtime
from test_integration import roundtrip
from vllm.v1.engine.core import EngineCoreRequestType, EngineShutdownState
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder


@pytest.mark.parametrize("role", ["prefill", "decode"])
def test_real_utility_dispatch_and_worker_callable_codec(monkeypatch, tmp_path, role):
    path = ROOT / ".github/benchmark/prefix-read-fixed/runtime/core.py"
    spec = importlib.util.spec_from_file_location("prefix_read_core_overlay", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    engine = object.__new__(module.EngineCoreProc)
    engine.shutdown_state = EngineShutdownState.RUNNING
    engine.output_queue = queue.Queue()
    events = []

    class Adapter:
        observer = SimpleNamespace(dropped=0, invalid=[])

        def stop_sampling(self):
            events.append("scheduler-stopped")

        def close_scheduler(self):
            events.append("scheduler-restored")
            return {"status": "READY", "restored": True, "metadata": []}

        def close_worker(self):
            assert events[:2] == ["scheduler-stopped", "scheduler-restored"]
            events.append("worker-restored")
            return {"status": "READY", "restored": True, "evidence": {"plans": []}}

    def mark(child):
        child._prefix_read_installation = (
            "SKIPPED_PRODUCER_NO_READ" if role == "prefill" else "ARMED"
        )
        if role == "decode":
            child._prefix_read_adapter = Adapter()

    scheduler_owner = SimpleNamespace(is_producer=role == "prefill")
    child = type("MoRIIOConnector", (), {})()
    child.connector_scheduler = scheduler_owner
    child.connector_worker = None
    mark(child)
    engine.scheduler = SimpleNamespace(
        connector=SimpleNamespace(sub_connectors=[child]), has_requests=lambda: False
    )
    monkeypatch.setenv("C48_DIAG_ROLE", role)
    monkeypatch.setenv("C48_DIAG_DIR", str(tmp_path / "model-manifests"))

    def collective(method, timeout):
        assert timeout == 30
        # Existing MessageQueue actual codec must preserve the worker callable.
        restored = roundtrip(method)
        assert restored is runtime.worker_close
        rows = []
        for rank in range(8):
            worker_child = type("MoRIIOConnector", (), {})()
            worker_child.connector_scheduler = None
            worker_child.connector_worker = SimpleNamespace(
                is_producer=role == "prefill", tp_rank=rank
            )
            mark(worker_child)
            import vllm.distributed.kv_transfer.kv_transfer_state as state

            monkeypatch.setattr(state, "_KV_CONNECTOR_AGENT", worker_child)
            rows.append(restored(SimpleNamespace(rank=rank)))
        return rows

    engine.model_executor = SimpleNamespace(collective_rpc=collective)
    packet = (0, 1234, "prefix_read_close", ())
    decoded = MsgpackDecoder().decode(MsgpackEncoder().encode(packet))
    engine._handle_client_request(EngineCoreRequestType.UTILITY, decoded)
    _, output = engine.output_queue.get_nowait()
    assert output.utility_output.failure_message is None
    # UtilityResult is the real msgspec transport wrapper, not a fake return.
    encoded = MsgpackEncoder().encode(output)
    result = MsgpackDecoder(module.EngineCoreOutputs).decode(encoded)
    payload = result.utility_output.result.result
    runtime.validate_role(payload)
    assert payload["status"] == "READY"
    assert len(payload["workers"]) == 8
    if role == "decode":
        assert (
            events
            == ["scheduler-stopped", "scheduler-restored"] + ["worker-restored"] * 8
        )
    else:
        assert events == []
