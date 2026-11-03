#!/usr/bin/env python3
"""Closed-loop synthetic PoC load; localhost only, same manifest for OFF/ON."""

import argparse
import asyncio
import hashlib
import json
import random
import statistics
import time
from pathlib import Path
from urllib.parse import urlsplit

import httpx


def percentile(values, quantile):
    values = sorted(values)
    if not values:
        return None
    position = (len(values) - 1) * quantile
    lo = int(position)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def distribution(values):
    return {
        "mean": statistics.mean(values),
        "p50": percentile(values, 0.5),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
    }


def manifest(args):
    path = Path(args.manifest)
    config = {
        "input_tokens": args.input_tokens,
        "output_tokens": args.output_tokens,
        "requests": args.requests,
        "seed": 20261020,
        "point_namespace": f"i{args.input_tokens}-c{args.concurrency}",
    }
    if path.exists():
        data = json.loads(path.read_text())
        if data["config"] != config:
            raise ValueError("Existing manifest has a different workload")
    else:
        data = {"config": config, "prompts": []}
        for number in range(args.requests):
            rng = random.Random(
                f"{config['seed']}:{config['point_namespace']}:{number}"
            )
            data["prompts"].append(
                [rng.randrange(100, 10000) for _ in range(args.input_tokens)]
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, separators=(",", ":")) + "\n")
    return data, hashlib.sha256(path.read_bytes()).hexdigest()


async def request(client, args, prompt, request_number):
    started = time.perf_counter()
    first = last = None
    usage = None
    terminal = False
    response_id = None
    text_hash = hashlib.sha256()
    finished = False
    trace_id = hashlib.sha256(
        f"{args.arm}:{args.concurrency}:{args.input_tokens}:{request_number}".encode()
    ).hexdigest()[:32]
    body = {
        "model": args.model,
        "prompt": prompt,
        "max_tokens": args.output_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    try:
        async with client.stream(
            "POST",
            args.url + "/v1/completions",
            json=body,
            headers={
                "traceparent": f"00-{trace_id}-00f067aa0ba902b7-01",
                "tracestate": "nsdye=1",
            },
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                if terminal:
                    raise ValueError("Data after DONE")
                payload = line[5:].strip()
                if payload == "[DONE]":
                    terminal = True
                    continue
                item = json.loads(payload)
                if item.get("id"):
                    if response_id is not None and response_id != item["id"]:
                        raise ValueError("Response ID changed within stream")
                    response_id = item["id"]
                if "error" in item:
                    raise ValueError("SSE error event")
                if item.get("usage"):
                    usage = item["usage"]
                for choice in item.get("choices", []):
                    if choice.get("finish_reason") is not None:
                        finished = True
                    text = choice.get("text") or ""
                    if text:
                        now = time.perf_counter()
                        first = now if first is None else first
                        last = now
                        text_hash.update(text.encode())
        ended = time.perf_counter()
        if not terminal or not finished or usage is None or first is None:
            raise ValueError("Incomplete streaming response")
        if usage["completion_tokens"] != args.output_tokens or usage[
            "prompt_tokens"
        ] != len(prompt):
            raise ValueError("Token count differs from fixed workload")
        return {
            "id": request_number,
            "trace_id": trace_id,
            "response_id": response_id,
            "ok": True,
            "ttft_s": first - started,
            "e2e_s": ended - started,
            "tpot_s": (last - first) / max(1, usage["completion_tokens"] - 1),
            "usage": usage,
            "output_sha256": text_hash.hexdigest(),
        }
    except Exception as exc:  # noqa: BLE001 - preserve each failed request
        return {
            "id": request_number,
            "trace_id": trace_id,
            "ok": False,
            "error_type": type(exc).__name__,
            "e2e_s": time.perf_counter() - started,
        }


async def run(args, data, digest):
    if urlsplit(args.url).hostname not in {"localhost", "127.0.0.1"}:
        raise ValueError("PoC benchmark accepts localhost endpoints only")
    args.url = args.url.rstrip("/")
    limits = httpx.Limits(
        max_connections=max(4, args.concurrency),
        max_keepalive_connections=max(4, args.concurrency),
    )
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(180), limits=limits, trust_env=False
    ) as client:
        # Distinct warmup prompts; none of the timed prompts is primed.
        warmup = []
        for start in range(0, args.warmup, args.concurrency):
            batch = []
            for n in range(start, min(start + args.concurrency, args.warmup)):
                prompt = list(data["prompts"][n % len(data["prompts"])])
                prompt[0] = 10001 + n
                batch.append(request(client, args, prompt, -n - 1))
            warmup.extend(await asyncio.gather(*batch))
        if not all(row["ok"] for row in warmup):
            return {
                "arm": args.arm,
                "valid": False,
                "warmup": warmup,
                "performance": None,
            }
        pending = asyncio.Queue()
        for number, prompt in enumerate(data["prompts"]):
            pending.put_nowait((number, prompt))
        rows = []

        async def worker():
            while not pending.empty():
                number, prompt = pending.get_nowait()
                rows.append(await request(client, args, prompt, number))

        started = time.perf_counter()
        try:
            await asyncio.wait_for(
                asyncio.gather(*(worker() for _ in range(args.concurrency))), 900
            )
        except asyncio.TimeoutError:
            completed = {row["id"] for row in rows}
            rows.extend(
                {"id": number, "ok": False, "error_type": "PointDeadline"}
                for number in range(args.requests)
                if number not in completed
            )
        wall = time.perf_counter() - started
    valid = len(rows) == args.requests and all(row["ok"] for row in rows)
    performance = None
    if valid:
        performance = {
            "wall_s": wall,
            "requests_per_s": len(rows) / wall,
            "output_tokens_per_s": args.output_tokens * len(rows) / wall,
            "ttft_s": distribution([r["ttft_s"] for r in rows]),
            "tpot_s": distribution([r["tpot_s"] for r in rows]),
            "e2e_s": distribution([r["e2e_s"] for r in rows]),
        }
    return {
        "arm": args.arm,
        "mode": "closed_loop",
        "concurrency": args.concurrency,
        "manifest_sha256": digest,
        "workload": data["config"],
        "valid": valid,
        "failures": sum(not r["ok"] for r in rows),
        "performance": performance,
        "warmup": warmup,
        "requests": sorted(rows, key=lambda r: r["id"]),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:30380")
    parser.add_argument("--model", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--input-tokens", type=int, default=2048)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--requests", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=4)
    args = parser.parse_args()
    if (
        min(args.concurrency, args.input_tokens, args.output_tokens, args.requests) < 1
        or args.warmup < 0
    ):
        parser.error("workload sizes must be positive; warmup must be nonnegative")
    if Path(args.output).exists():
        parser.error("output already exists; use a fresh arm/run identifier")
    data, digest = manifest(args)
    summary = asyncio.run(run(args, data, digest))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in summary.items() if k not in {"requests", "warmup"}}
        )
    )
    raise SystemExit(0 if summary["valid"] else 1)
