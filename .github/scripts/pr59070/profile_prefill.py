"""Sequential, identical-input TP8 prefill and prefix-cache diagnostic client."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import httpx
from prometheus_client.parser import text_string_to_metric_families


METRICS = (
    "vllm:request_prefill_kv_computed_tokens",
    "vllm:request_prefill_time_seconds",
    "vllm:request_queue_time_seconds",
    "vllm:time_to_first_token_seconds",
    "vllm:prefix_cache_hits",
    "vllm:prefix_cache_queries",
)


def metric_snapshot(client, output, tag):
    response = client.get("/metrics")
    response.raise_for_status()
    raw = response.text
    (output / f"{tag}.metrics.txt").write_text(raw)
    values = {}
    for family in text_string_to_metric_families(raw):
        for sample in family.samples:
            if sample.name.startswith(METRICS) and not sample.name.endswith(
                ("_bucket", "_created")
            ):
                values[sample.name] = values.get(sample.name, 0.0) + sample.value
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:2584")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with httpx.Client(base_url=args.url, timeout=600) as client:
        client.get("/health").raise_for_status()
        text = (
            "Analyze the following engineering record. Preserve the numbered facts "
            "and give a brief explanation after reading the complete record.\n"
            + "\n".join(
                f"Record {i}: The service processes tasks in order. Each task has "
                "an input buffer, a computation stage, and a result. The operator "
                "checks timings and reuses unchanged intermediate results."
                for i in range(2400)
            )
        )
        response = client.post("/tokenize", json={"model": "Kimi-K3", "prompt": text})
        response.raise_for_status()
        tokens = response.json()["tokens"]
        assert len(tokens) > 40000
        alternate = client.post(
            "/tokenize",
            json={
                "model": "Kimi-K3",
                "prompt": " Describe the alternative schedule and its timing.",
            },
        )
        alternate.raise_for_status()
        tail = alternate.json()["tokens"]
        assert tail
        fork_tail = (tail * (8192 // len(tail) + 1))[:8192]
        sibling = client.post(
            "/tokenize",
            json={
                "model": "Kimi-K3",
                "prompt": " Evaluate a different buffer allocation and its costs.",
            },
        )
        sibling.raise_for_status()
        sibling_tail = sibling.json()["tokens"]
        sibling_tail = (sibling_tail * (8192 // len(sibling_tail) + 1))[:8192]
        cases = [
            ("cold_32768", tokens[:32768]),
            ("exact_repeat_32768", tokens[:32768]),
            ("append_1024", tokens[:33792]),
            ("fork_at_12032", tokens[:12032] + fork_tail),
            ("fork_repeat", tokens[:12032] + fork_tail),
            ("fork_append_1024", tokens[:12032] + fork_tail + tokens[35000:36024]),
            ("sibling_at_12032", tokens[:12032] + sibling_tail),
            ("sibling_repeat", tokens[:12032] + sibling_tail),
        ]
        workload = {
            "cases": [
                {
                    "name": n,
                    "prompt_token_ids": t,
                    "sha256": hashlib.sha256(json.dumps(t).encode()).hexdigest(),
                }
                for n, t in cases
            ],
            "scope": "synthetic deterministic prefix chain; max_tokens=1",
        }
        (args.output / "workload.json").write_text(json.dumps(workload) + "\n")

        def request(name, prompt):
            before = metric_snapshot(client, args.output, f"{name}-before")
            count_key = "vllm:request_prefill_kv_computed_tokens_count"
            assert count_key in before, "Per-request prefill metrics are unavailable"
            wall_start_ns = time.time_ns()
            start = time.perf_counter()
            response = client.post(
                "/v1/completions",
                json={
                    "model": "Kimi-K3",
                    "prompt": prompt,
                    "max_tokens": 1,
                    "temperature": 0,
                    "stream": False,
                },
                headers={"X-Request-Id": f"p8-profile-{name}"},
            )
            elapsed = time.perf_counter() - start
            wall_end_ns = time.time_ns()
            response.raise_for_status()
            body = response.json()
            assert body["usage"]["prompt_tokens"] == len(prompt), body
            assert body["usage"]["completion_tokens"] == 1, body
            after = metric_snapshot(client, args.output, f"{name}-after")
            deadline = time.monotonic() + 20
            while after[count_key] < before[count_key] + 1:
                assert time.monotonic() < deadline, "Metrics did not record completion"
                time.sleep(0.1)
                after = metric_snapshot(client, args.output, f"{name}-after")
            assert after[count_key] == before[count_key] + 1, (
                "Another request contaminated the sequential measurement"
            )
            row = {
                "name": name,
                "prompt_tokens": len(prompt),
                "client_elapsed_s": elapsed,
                "wall_start_ns": wall_start_ns,
                "wall_end_ns": wall_end_ns,
                "usage": body["usage"],
                "response_id": body["id"],
                "choices": body["choices"],
                "metrics_delta": {
                    k: after.get(k, 0.0) - before.get(k, 0.0)
                    for k in set(before) | set(after)
                },
            }
            (args.output / f"{name}.json").write_text(json.dumps(row, indent=2) + "\n")
            print(json.dumps(row), flush=True)
            return row

        # Trigger shape-specific compilation before measurements/profiling.
        for length in [512, 4096, 16384, 32768]:
            request(f"warmup_{length}", tokens[-length:])

        def reset_cache():
            reset = client.post("/reset_prefix_cache")
            reset.raise_for_status()
            assert reset.json()["success"], reset.text

        reset_cache()
        for name, prompt in cases:
            request(f"warmup_chain_{name}", prompt)

        assert args.repeats > 0
        rows = []
        for repeat in range(args.repeats):
            reset_cache()
            for name, prompt in cases:
                row = request(f"timing_{repeat}_{name}", prompt)
                rows.append({**row, "case": name, "repeat": repeat})
        (args.output / "timing-summary.json").write_text(
            json.dumps(rows, indent=2) + "\n"
        )

        # Profile a separate replay of exactly the same history. Timings above
        # remain free of profiler overhead. Manual stop also runs after failure.
        reset_cache()
        client.post("/start_profile").raise_for_status()
        try:
            for name, prompt in cases:
                request(f"profile_{name}", prompt)
        finally:
            response = client.post("/stop_profile")
            response.raise_for_status()


if __name__ == "__main__":
    main()
