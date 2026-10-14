"""Experiment-only scheduler utility and serialized worker observation closure."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import secrets
from pathlib import Path

PAYLOAD = Path(__file__).resolve().parents[2] / "benchmark/prefix-read-fixed"


def mori_child(connector):
    if type(connector).__name__ == "MoRIIOConnector":
        return connector
    matches = []
    for child in getattr(connector, "sub_connectors", []):
        if type(child).__name__ == "MoRIIOConnector":
            matches.append(child)
    if len(matches) != 1:
        raise RuntimeError("expected exactly one MoRI child")
    return matches[0]


def install(connector):
    child = mori_child(connector)
    owner = child.connector_scheduler or child.connector_worker
    role = os.environ["C48_DIAG_ROLE"]
    if owner is None or bool(owner.is_producer) != (role == "prefill"):
        raise RuntimeError("MoRI role mismatch")
    if role == "prefill":
        child._prefix_read_installation = "SKIPPED_PRODUCER_NO_READ"
        return
    if role != "decode":
        raise RuntimeError("unsupported serving role")
    spec = importlib.util.spec_from_file_location(
        "prefix_read_adapter", PAYLOAD / "adapter.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.runtime_install(child) is None:
        raise RuntimeError("decode observer was not installed")
    child._prefix_read_installation = "ARMED"


def persist(name: str, evidence: dict) -> dict:
    root = Path(os.environ["C48_DIAG_DIR"]).parent / "prefix-read-observer"
    root.mkdir(parents=True, exist_ok=True)
    path = root / name
    encoded = (json.dumps(evidence, sort_keys=True) + "\n").encode()
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    path.chmod(0o444)
    return {
        "path": str(path),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "bytes": len(encoded),
    }


def close_owner(child, owner_kind: str, rank: int | None = None) -> dict:
    saved = getattr(child, "_prefix_read_closed", None)
    if saved is not None:
        return saved
    role = os.environ["C48_DIAG_ROLE"]
    installation = getattr(child, "_prefix_read_installation", None)
    common = {"role": role, "owner": owner_kind, "rank": rank}
    if installation == "SKIPPED_PRODUCER_NO_READ":
        if role != "prefill" or hasattr(child, "_prefix_read_adapter"):
            raise RuntimeError("invalid producer skip")
        result = dict(
            common,
            status="SKIPPED",
            restored=True,
            installed=False,
            reason="producer has no READ path; no observer installed",
            dropped=0,
            invalid=[],
        )
    elif installation == "ARMED":
        adapter = child._prefix_read_adapter
        if owner_kind == "scheduler":
            adapter.stop_sampling()
            raw = adapter.close_scheduler()
        else:
            raw = adapter.close_worker()
        result = dict(
            common,
            status=raw["status"],
            restored=raw.get("restored", False),
            installed=True,
            dropped=adapter.observer.dropped,
            invalid=list(adapter.observer.invalid),
        )
        if (
            result["status"] != "READY"
            or not result["restored"]
            or result["dropped"]
            or result["invalid"]
        ):
            return dict(result, status="NOT_READY")
        result["artifact"] = persist(f"{role}-{owner_kind}-{rank}.json", raw)
    else:
        raise RuntimeError("observer installation evidence missing")
    child._prefix_read_closed = result
    return result


def worker_close(worker):
    from vllm.distributed.kv_transfer.kv_transfer_state import get_kv_transfer_group

    child = mori_child(get_kv_transfer_group())
    if child.connector_worker is None:
        raise RuntimeError("worker RPC has no worker connector")
    rank = int(worker.rank)
    if rank != int(child.connector_worker.tp_rank):
        raise RuntimeError("TP rank does not match worker rank")
    return close_owner(child, "worker", rank)


def engine_close(engine):
    child = mori_child(engine.scheduler.connector)
    if child.connector_scheduler is None:
        raise RuntimeError("utility RPC has no scheduler connector")
    if engine.scheduler.has_requests():
        return {
            "role": os.environ["C48_DIAG_ROLE"],
            "status": "NOT_READY",
            "workers": [],
            "scheduler": {
                "status": "NOT_READY",
                "reason": "existing scheduler not idle",
            },
        }
    scheduler = close_owner(child, "scheduler")
    if scheduler["status"] not in ("READY", "SKIPPED"):
        return {
            "role": os.environ["C48_DIAG_ROLE"],
            "scheduler": scheduler,
            "workers": [],
            "status": "NOT_READY",
        }
    workers = engine.model_executor.collective_rpc(worker_close, timeout=30)
    result = {
        "role": os.environ["C48_DIAG_ROLE"],
        "scheduler": scheduler,
        "workers": workers,
    }
    try:
        validate_role(result)
    except (ValueError, KeyError, TypeError):
        result["status"] = "NOT_READY"
    else:
        result["status"] = "READY"
    persist(f"{result['role']}-closure.json", result)
    return result


def validate_role(result: dict) -> None:
    role = result["role"]
    if role not in ("prefill", "decode"):
        raise ValueError("unknown role")
    rows = result["workers"]
    if len(rows) != 8 or {row["rank"] for row in rows} != set(range(8)):
        raise ValueError("missing or duplicate worker rank")
    if (
        result["scheduler"]["owner"] != "scheduler"
        or result["scheduler"]["rank"] is not None
    ):
        raise ValueError("invalid scheduler owner")
    if any(row["owner"] != "worker" for row in rows):
        raise ValueError("invalid worker owner")
    expected = "SKIPPED" if role == "prefill" else "READY"
    for row in [result["scheduler"], *rows]:
        if (
            row["role"] != role
            or row["status"] != expected
            or row["restored"] is not True
        ):
            raise ValueError("not all owners restored")
        if (
            row["dropped"] != 0
            or row["invalid"]
            or row["installed"] != (role == "decode")
        ):
            raise ValueError("invalid/dropped observation or unsupported skip")
        if role == "decode" and not row.get("artifact", {}).get("sha256"):
            raise ValueError("missing saved observation")


def attach_router(app):
    from fastapi import HTTPException, Request

    # A dedicated token avoids enabling arbitrary dev RPC / serialized callables.
    async def close(request: Request):
        token = os.environ.get("PREFIX_READ_CONTROL_TOKEN", "")
        supplied = request.headers.get("x-prefix-read-token", "")
        if not token or not secrets.compare_digest(token, supplied):
            raise HTTPException(
                status_code=403, detail="invalid experiment control token"
            )
        client = request.app.state.engine_client
        return await client.engine_core.call_utility_async("prefix_read_close")

    # Request is local to this function; postponed annotation must be resolved.
    close.__annotations__["request"] = Request
    app.add_api_route("/_prefix_read/close", close, methods=["POST"])
