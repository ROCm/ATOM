# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Prove a deployment works: a real request, tool calling, and a quick load test.

The load test is a sanity number (threads + streaming, stdlib only), not a
replacement for ``python -m atom.benchmarks.benchmark_serving``; the result
says how to run that one for publishable figures.
"""

from __future__ import annotations

import random
import statistics
import threading
import time

from atom.agent.brain import BrainError, chat_completion, list_models

_WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}
_WORDS = [
    "atom", "rocm", "instinct", "kernel", "tensor", "cache", "token", "batch",
    "graph", "decode", "prefill", "expert", "router", "memory", "bandwidth",
    "latency", "throughput", "cluster", "node", "stream", "wave",
]  # fmt: skip


def _model_id(base_url: str) -> str | None:
    models = list_models(base_url)
    return models[0] if models else None


def smoke_test(
    base_url: str,
    prompt: str = "In one sentence, what is an AMD Instinct GPU?",
    max_tokens: int = 128,
    check_tool_calling: bool = True,
) -> dict:
    model = _model_id(base_url)
    if model is None:
        return {"ok": False, "error": f"{base_url}/models did not answer"}
    out: dict = {"model": model}
    # The first request after startup pays one-time costs (lazy kernel
    # builds, allocator growth); time it separately so the reported TTFT is
    # what clients will see, and the cold number is not misread.
    try:
        cold = chat_completion(
            base_url,
            model,
            [{"role": "user", "content": "ping"}],
            max_tokens=1,
            temperature=0,
        )
        out["first_request_ttft_ms"] = round((cold.ttft_s or cold.total_s) * 1000, 1)
    except BrainError as e:
        return {"ok": False, "model": model, "error": str(e)}
    try:
        r = chat_completion(
            base_url,
            model,
            [{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0,
        )
    except BrainError as e:
        return {"ok": False, "model": model, "error": str(e)}
    toks = (r.usage or {}).get("completion_tokens") or 0
    decode_s = (r.chunk_times[-1] - r.ttft_s) if r.ttft_s and r.chunk_times else 0
    out["chat"] = {
        "reply": r.content.strip()[:600],
        # TTFT counts the first streamed token of any kind, reasoning included.
        "reasoning_chars": len(r.reasoning),
        "finish_reason": r.finish_reason,
        "ttft_ms": round(r.ttft_s * 1000, 1) if r.ttft_s else None,
        "completion_tokens": toks,
        "decode_tok_per_s": (
            round((toks - 1) / decode_s, 1) if toks > 1 and decode_s else None
        ),
    }
    ok = bool(r.content.strip() or r.reasoning.strip())
    if check_tool_calling:
        try:
            t = chat_completion(
                base_url,
                model,
                [
                    {
                        "role": "user",
                        "content": "What's the weather in Austin? Use the tool.",
                    }
                ],
                tools=[_WEATHER_TOOL],
                max_tokens=max(512, max_tokens),
                temperature=0,
            )
            calls = [
                {"name": c.name, "arguments": c.arguments[:200]} for c in t.tool_calls
            ]
            out["tool_calling"] = {
                "works": any(c["name"] == "get_weather" for c in calls),
                "calls": calls,
                "text_if_no_call": None if calls else t.content.strip()[:300],
            }
        except BrainError as e:
            out["tool_calling"] = {"works": False, "error": str(e)[:500]}
    out["ok"] = ok
    return out


def _quiet(fn, *args, **kwargs) -> None:
    try:
        fn(*args, **kwargs)
    except BrainError:
        pass


def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    k = (len(values) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def benchmark(
    base_url: str,
    input_len: int = 1024,
    output_len: int = 256,
    concurrency: int = 8,
    num_requests: int | None = None,
    timeout_s: float = 900,
) -> dict:
    model = _model_id(base_url)
    if model is None:
        return {"ok": False, "error": f"{base_url}/models did not answer"}
    num_requests = num_requests or concurrency * 4
    rng = random.Random(0)
    # Distinct prompts so prefix caching does not flatter TTFT.
    prompts = [
        f"[{i}] " + " ".join(rng.choice(_WORDS) for _ in range(input_len))
        for i in range(num_requests)
    ]
    results: list[dict] = []
    errors: list[str] = []
    lock = threading.Lock()
    queue = list(range(num_requests))
    deadline = time.monotonic() + timeout_s

    def worker() -> None:
        while time.monotonic() < deadline:
            with lock:
                if not queue:
                    return
                i = queue.pop()
            try:
                r = chat_completion(
                    base_url,
                    model,
                    [{"role": "user", "content": prompts[i]}],
                    max_tokens=output_len,
                    temperature=0,
                    extra={"ignore_eos": True},
                    timeout=timeout_s,
                )
            except BrainError as e:
                with lock:
                    errors.append(str(e)[:300])
                continue
            usage = r.usage or {}
            out_toks = usage.get("completion_tokens") or len(r.chunk_times)
            tpot = (
                (r.chunk_times[-1] - r.ttft_s) / (out_toks - 1)
                if r.ttft_s and out_toks > 1
                else None
            )
            with lock:
                results.append(
                    {
                        "ttft": r.ttft_s,
                        "tpot": tpot,
                        "e2e": r.total_s,
                        "in": usage.get("prompt_tokens") or 0,
                        "out": out_toks,
                    }
                )

    # Warm up first, as benchmark_serving does: the first requests at a new
    # concurrency hit cold paths (first prefill shapes, allocator growth)
    # and would otherwise land in the measured tail.
    warm = [
        threading.Thread(
            target=lambda i=i: _quiet(
                chat_completion,
                base_url,
                model,
                [
                    {
                        "role": "user",
                        "content": f"[warmup {i}] " + prompts[i % len(prompts)],
                    }
                ],
                max_tokens=8,
                temperature=0,
                timeout=timeout_s,
            ),
            daemon=True,
        )
        for i in range(concurrency)
    ]
    for t in warm:
        t.start()
    for t in warm:
        t.join(timeout_s)

    t0 = time.perf_counter()
    threads = [threading.Thread(target=worker, daemon=True) for _ in range(concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(max(deadline - time.monotonic(), 0) + 5)
    wall = time.perf_counter() - t0

    def ms(values, p):
        v = _percentile([x for x in values if x is not None], p)
        return round(v * 1000, 1) if v is not None else None

    ttft = [r["ttft"] for r in results]
    tpot = [r["tpot"] for r in results]
    total_in = sum(r["in"] for r in results)
    total_out = sum(r["out"] for r in results)
    return {
        "ok": bool(results),
        "model": model,
        "requests_ok": len(results),
        "requests_failed": len(errors),
        "errors": errors[:3],
        "concurrency": concurrency,
        "warmup_requests": concurrency,
        "mean_input_tokens": round(total_in / len(results)) if results else None,
        "mean_output_tokens": round(total_out / len(results)) if results else None,
        "wall_s": round(wall, 1),
        "ttft_ms": {"p50": ms(ttft, 0.5), "p90": ms(ttft, 0.9), "p99": ms(ttft, 0.99)},
        "tpot_ms": {"p50": ms(tpot, 0.5), "p90": ms(tpot, 0.9)},
        "mean_tpot_ms": (
            round(statistics.mean([x for x in tpot if x]) * 1000, 2)
            if any(tpot)
            else None
        ),
        "output_tok_per_s": round(total_out / wall, 1) if wall else None,
        "total_tok_per_s": round((total_in + total_out) / wall, 1) if wall else None,
        "note": (
            "Quick sanity load (stdlib client). For publishable numbers run "
            "python -m atom.benchmarks.benchmark_serving (see "
            ".claude/commands/benchmark-guide.md); report Total Token throughput."
        ),
    }
