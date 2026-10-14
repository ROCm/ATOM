"""Await restoration after the existing warmup drain, before profiling starts."""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import os
import time
from pathlib import Path

from pd_prefix_runtime import validate_role


async def close_all(fetch, timeout=90.0, interval=0.5):
    started = time.monotonic()
    attempts = []
    async with asyncio.timeout(timeout):
        while True:
            results = await asyncio.gather(fetch("prefill"), fetch("decode"))
            attempts.append(results)
            if {row.get("role") for row in results} != {"prefill", "decode"}:
                raise RuntimeError("missing/duplicate serving role")
            for result in results:
                owners = [result.get("scheduler", {}), *result.get("workers", [])]
                if any(row.get("dropped", 0) or row.get("invalid") for row in owners):
                    raise RuntimeError("observer dropped or invalid events")
            if all(row.get("status") == "READY" for row in results):
                for result in results:
                    validate_role(result)
                return {
                    "status": "READY",
                    "wait_seconds": time.monotonic() - started,
                    "attempts": len(attempts),
                    "roles": results,
                }
            if any(row.get("status") not in ("READY", "NOT_READY") for row in results):
                raise RuntimeError("unknown closure state")
            # Only repeat serialized close-status queries. Never issue model work,
            # drain credits, poll transfers, or reset state from the barrier.
            await asyncio.sleep(interval)


def mechanism_gate(result, run_dir):
    root = Path(run_dir).resolve(strict=True)
    parts = []
    eligible = []
    decode = next(row for row in result["roles"] if row["role"] == "decode")
    validate_role(decode)
    for row in [decode["scheduler"], *decode["workers"]]:
        artifact = row["artifact"]
        path = Path(artifact["path"]).resolve(strict=True)
        if not path.is_relative_to(root) or "prefix-read-observer" not in path.parts:
            raise RuntimeError("observer artifact escapes shared run directory")
        if path.stat().st_mode & 0o222:
            raise RuntimeError("observer artifact was not finalized readonly")
        raw = path.read_bytes()
        if (
            len(raw) != artifact["bytes"]
            or hashlib.sha256(raw).hexdigest() != artifact["sha256"]
        ):
            raise RuntimeError("observer evidence hash mismatch")
        document = json.loads(raw)
        if row["owner"] == "scheduler":
            if document["status"] != "READY" or document["restored"] is not True:
                raise RuntimeError("scheduler sidecar not closed")
            continue
        evidence = document["evidence"]
        if any(plan["rank"] != row["rank"] for plan in evidence["plans"]):
            raise RuntimeError("artifact rank mismatch")
        selected = {
            plan["transfer"]
            for plan in evidence["plans"]
            if plan["external"] > 0
            and plan["prompt"] - 1 - plan["external"] >= plan["page_tokens"]
        }
        eligible.append(selected)
        parts.append(evidence)
    shared = set.intersection(*eligible)
    if not shared:
        raise RuntimeError(
            "no complete-cache-hit external transfer observed on all D8 ranks"
        )
    path = (
        Path(__file__).resolve().parents[2] / "benchmark/prefix-read-fixed/observer.py"
    )
    spec = importlib.util.spec_from_file_location("prefix_read_mechanism", path)
    observer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(observer)
    verdict = observer.check(observer.combine(parts), expected_ranks=list(range(8)))
    if verdict["submission"] != "PASS" or verdict["completion"] != "PASS":
        raise RuntimeError("prefix READ mechanism failed: " + json.dumps(verdict))
    return dict(verdict, common_eligible_transfers=sorted(shared))


def write_result(output, result):
    temporary = output.with_suffix(output.suffix + f".{os.getpid()}.tmp")
    with temporary.open("xb") as stream:
        stream.write((json.dumps(result, indent=2) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, output)


async def post_warmup_gate():
    import aiohttp

    endpoints = json.loads(os.environ["PREFIX_READ_ENDPOINTS"])
    if set(endpoints) != {"prefill", "decode"}:
        raise RuntimeError("expected exactly two role endpoints")
    token = os.environ["PREFIX_READ_CONTROL_TOKEN"]
    output = Path(os.environ["PREFIX_READ_GATE_OUTPUT"])
    started = time.monotonic()
    timeout = aiohttp.ClientTimeout(total=35)
    async with aiohttp.ClientSession(timeout=timeout) as session:

        async def fetch(role):
            async with session.post(
                endpoints[role] + "/_prefix_read/close",
                headers={"x-prefix-read-token": token},
            ) as response:
                response.raise_for_status()
                return await response.json()

        try:
            async with asyncio.timeout(120):
                result = await close_all(fetch)
                result["mechanism"] = await asyncio.to_thread(
                    mechanism_gate, result, os.environ["RUN_DIR"]
                )
                result["wait_seconds"] = time.monotonic() - started
            write_result(output, result)
        except BaseException as exc:
            write_result(
                output,
                {
                    "status": "FAILED",
                    "wait_seconds": time.monotonic() - started,
                    "error": type(exc).__name__,
                    "detail": str(exc),
                },
            )
            raise
