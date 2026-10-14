"""CPU full phase ordering and dedicated scheduler-to-worker RPC contract."""

import asyncio
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / ".github/scripts/atomesh"
sys.path.insert(0, str(SCRIPTS))
import pd_prefix_gate as gate
import pd_prefix_runtime as runtime


def ready(role):
    def row(rank, owner):
        return {
            "role": role,
            "rank": rank,
            "owner": owner,
            "status": "READY" if role == "decode" else "SKIPPED",
            "restored": True,
            "installed": role == "decode",
            "dropped": 0,
            "invalid": [],
            "artifact": {"sha256": "saved", "path": "evidence"},
        }

    return {
        "role": role,
        "status": "READY",
        "scheduler": row(None, "scheduler"),
        "workers": [row(rank, "worker") for rank in range(8)],
    }


def test_barrier_waits_for_all_ranks():
    calls = []

    async def fetch(role):
        calls.append(role)
        result = ready(role)
        if role == "decode" and calls.count(role) == 1:
            result["status"] = "NOT_READY"
            result["workers"][7]["status"] = "NOT_READY"
            result["workers"][7]["restored"] = False
        return result

    result = asyncio.run(gate.close_all(fetch, timeout=1, interval=0))
    assert result["attempts"] == 2
    assert calls == ["prefill", "decode", "prefill", "decode"]


@pytest.mark.parametrize(
    "fault", ["drop", "invalid", "missing", "duplicate", "unrestored", "skipped_decode"]
)
def test_barrier_never_accepts_incomplete_or_invalid_evidence(fault):
    async def fetch(role):
        result = ready(role)
        if role == "decode":
            if fault == "drop":
                result["workers"][0]["dropped"] = 1
            elif fault == "invalid":
                result["workers"][0]["invalid"] = ["unknown"]
            elif fault == "missing":
                result["workers"].pop()
            elif fault == "duplicate":
                result["workers"][7]["rank"] = 0
            elif fault == "unrestored":
                result["workers"][0]["restored"] = False
            else:
                result["workers"][0]["status"] = "SKIPPED"
        return result

    with pytest.raises((RuntimeError, ValueError)):
        asyncio.run(gate.close_all(fetch, timeout=1, interval=0))


def test_barrier_timeout_is_not_success():
    async def fetch(role):
        return dict(ready(role), status="NOT_READY")

    with pytest.raises(TimeoutError):
        asyncio.run(gate.close_all(fetch, timeout=0.005, interval=0.001))


def test_multiconnector_unique_mori():
    child = type("MoRIIOConnector", (), {})()
    assert (
        runtime.mori_child(SimpleNamespace(sub_connectors=[object(), child])) is child
    )
    with pytest.raises(RuntimeError):
        runtime.mori_child(SimpleNamespace(sub_connectors=[child, child]))


def test_real_aiperf_phase_orchestration_orders_gate(monkeypatch):
    # Run with the existing pinned AIPerf venv. Import the real patched module,
    # replace only runner work, and execute the actual _execute_phases method.
    from aiperf.common.enums import CreditPhase
    from aiperf.timing import phase_orchestrator as module

    assert "/prefix-read-fixed/client/" in module.__file__
    events = []

    class Runner:
        def __init__(self, config, **kwargs):
            self.phase = config.phase

        async def run(self, is_final_phase):
            events.append(str(self.phase))

    monkeypatch.setattr(module, "PhaseRunner", Runner)

    async def close():
        events.append("all-ranks-restored")

    monkeypatch.setattr(gate, "post_warmup_gate", close)
    instance = object.__new__(module.PhaseOrchestrator)
    instance._ordered_phase_configs = [
        SimpleNamespace(phase=phase, seamless=False, request_cancellation=None)
        for phase in (CreditPhase.WARMUP, CreditPhase.PROFILING)
    ]
    for field in (
        "conversation_source",
        "phase_publisher",
        "credit_router",
        "concurrency_manager",
        "callback_handler",
        "url_sampler",
        "run",
        "session_tree_registry",
    ):
        setattr(instance, "_" + field, None)
    instance._active_runners = []
    monkeypatch.setattr(module, "RequestCancellationSimulator", lambda x: None)
    asyncio.run(instance._execute_phases())
    assert events == [
        str(CreditPhase.WARMUP),
        "all-ranks-restored",
        str(CreditPhase.PROFILING),
    ]
    events.clear()

    async def fail():
        raise RuntimeError("rank7 NOT_READY")

    monkeypatch.setattr(gate, "post_warmup_gate", fail)
    instance.error = lambda *a: None

    async def cancel():
        events.append("cancelled")

    instance.cancel = cancel
    with pytest.raises(RuntimeError, match="rank7"):
        asyncio.run(instance._execute_phases())
    assert str(CreditPhase.PROFILING) not in events


@pytest.mark.parametrize(
    "fault", [None, "hash", "missing-rank", "bad-ranges", "no-coverage"]
)
def test_full_d8_mechanism_gate(tmp_path, fault, monkeypatch):
    import hashlib
    import json

    root = tmp_path / "prefix-read-observer"
    root.mkdir()
    spec = importlib.util.spec_from_file_location(
        "observer_fixture",
        "/app/test_scripts/dspark_pd/prefix-read-observer-81198-20261014/test_observer.py",
    )
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    result = {"roles": [ready("prefill"), ready("decode")]}
    scheduler_path = root / "scheduler.json"
    scheduler_raw = json.dumps(
        {"status": "READY", "restored": True, "metadata": []}
    ).encode()
    scheduler_path.write_bytes(scheduler_raw)
    scheduler_path.chmod(0o444)
    result["roles"][1]["scheduler"]["artifact"] = {
        "path": str(scheduler_path),
        "bytes": len(scheduler_raw),
        "sha256": hashlib.sha256(scheduler_raw).hexdigest(),
    }
    for rank, row in enumerate(result["roles"][1]["workers"]):
        plan = fixture.plan(ranks=list(range(8)))
        plan.update(rank=rank, page_tokens=16)
        ranges = (
            None if fault != "bad-ranges" or rank != 7 else [([32], [48], [16], "attn")]
        )
        monkeypatch.setattr(fixture.Worker, "tp_rank", rank)
        observer, _, _ = fixture.execute(plan, ranges=ranges)
        evidence = observer.close()
        if fault == "no-coverage":
            evidence["plans"][0]["external"] = 0
        path = root / f"rank-{rank}.json"
        encoded = (json.dumps({"evidence": evidence}) + "\n").encode()
        path.write_bytes(encoded)
        path.chmod(0o444)
        row["artifact"] = {
            "path": str(path),
            "bytes": len(encoded),
            "sha256": hashlib.sha256(encoded).hexdigest(),
        }
    if fault == "hash":
        result["roles"][1]["workers"][0]["artifact"]["sha256"] = "bad"
    if fault == "missing-rank":
        result["roles"][1]["workers"].pop()
    if fault is None:
        report = gate.mechanism_gate(result, tmp_path)
        assert report["submission"] == report["completion"] == "PASS"
        assert report["common_eligible_transfers"] == ["xfer"]
    else:
        with pytest.raises((RuntimeError, ValueError)):
            gate.mechanism_gate(result, tmp_path)


def test_http_uses_scheduler_utility_and_auth(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    calls = []

    async def utility(method):
        calls.append(method)
        return ready("decode")

    monkeypatch.setenv("PREFIX_READ_CONTROL_TOKEN", "test-only-token")
    app = FastAPI()
    app.state.engine_client = SimpleNamespace(
        engine_core=SimpleNamespace(call_utility_async=utility)
    )
    runtime.attach_router(app)
    client = TestClient(app)
    assert client.post("/_prefix_read/close").status_code == 403
    assert calls == []
    response = client.post(
        "/_prefix_read/close", headers={"x-prefix-read-token": "test-only-token"}
    )
    assert response.status_code == 200
    assert calls == ["prefix_read_close"]
