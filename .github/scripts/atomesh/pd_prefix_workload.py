# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Controlled D-prefix workload through the real PD router, without speculation."""

import argparse
import asyncio
import hashlib
import json
import statistics
import time
from pathlib import Path

import httpx


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


def same_output(left, right):
    if left["output_token_ids"] and right["output_token_ids"]:
        return left["output_token_ids"] == right["output_token_ids"]
    return bool(left["text"]) and left["text"] == right["text"]


def workload_geometry(page_tokens):
    assert page_tokens > 0
    mechanism_pages = max(8, (8192 + page_tokens - 1) // page_tokens)
    performance_pages = max(10, (32768 + page_tokens - 1) // page_tokens)
    return {
        "page_tokens": page_tokens,
        "mechanism_end": mechanism_pages * page_tokens,
        "mechanism_hit": (mechanism_pages * 3 // 4) * page_tokens,
        "performance_length": performance_pages * page_tokens + 1,
        "performance_hit": (performance_pages * 9 // 10) * page_tokens,
    }


class Trace:
    def __init__(self, root):
        self.root = root
        self.offsets = {}
        self.records = []

    def poll(self):
        for path in self.root.glob("*.jsonl"):
            with path.open() as stream:
                stream.seek(self.offsets.get(path, 0))
                while line := stream.readline():
                    if not line.endswith("\n"):
                        break
                    self.records.append(json.loads(line))
                    self.offsets[path] = stream.tell()

    async def request(self, names):
        for _ in range(120):
            self.poll()
            plans = [
                r
                for r in self.records
                if r["event"] == "plan"
                and any(name and name in r["request_id"] for name in names)
            ]
            if plans:
                plan = plans[-1]
                admissions = [
                    r
                    for r in self.records
                    if r["event"] == "admission"
                    and r["request_id"] == plan["request_id"]
                ]
                completions = [
                    r
                    for r in self.records
                    if r["event"] == "read_complete"
                    and r["request_id"] == plan["request_id"]
                ]
                completed = {
                    (r["rank"], r["kind"]) for r in completions if r["success"]
                }
                expected = {
                    (rank, kind) for rank in range(8) for kind in ("attention", "kda")
                }
                if admissions and completed == expected:
                    return plan, admissions[-1], completions
            await asyncio.sleep(0.5)
        raise AssertionError(
            f"Missing admission/plan/eight-rank READ completion for {names}"
        )


async def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    trace = Trace(args.trace)
    serial = 0
    records = []
    requests_file = args.output / "requests.jsonl"
    async with httpx.AsyncClient(timeout=600, trust_env=False) as client:

        async def post(url, path, body=None):
            response = await client.post(url + path, json=body)
            response.raise_for_status()
            return response.json()

        async def reset_decode():
            for _ in range(60):
                result = await post(args.decode, "/reset_prefix_cache")
                if result.get("success") is True:
                    return
                await asyncio.sleep(0.5)
            raise AssertionError(f"D prefix reset did not succeed: {result}")

        async def tokenize(text):
            result = await post(
                args.prefill,
                "/tokenize",
                {
                    "model": args.model,
                    "prompt": text,
                    "add_special_tokens": False,
                },
            )
            return result["tokens"]

        async def complete(tokens, tag, *, url=None, count=128, observe=True):
            nonlocal serial
            serial += 1
            name = f"prefix-{args.variant}-{tag}-{serial:06d}"
            body = {
                "model": args.model,
                "prompt": tokens,
                "max_tokens": count,
                "temperature": 0,
                "seed": 42,
                "ignore_eos": True,
                "stream": True,
                "stream_options": {"include_usage": True},
                "return_token_ids": True,
            }
            start = time.perf_counter()
            first = None
            arrivals, output_tokens, output_text = [], [], []
            response_id = None
            usage = None
            async with client.stream(
                "POST",
                (url or args.router) + "/v1/completions",
                json=body,
                headers={"X-Request-Id": name},
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    chunk = json.loads(line[6:])
                    response_id = chunk.get("id", response_id)
                    if chunk.get("usage"):
                        usage = chunk["usage"]
                    for choice in chunk.get("choices", []):
                        text = choice.get("text", "")
                        ids = choice.get("token_ids") or []
                        if text or ids:
                            now = time.perf_counter()
                            if first is None:
                                first = now
                            arrivals.append(now - start)
                            output_tokens.extend(ids)
                            output_text.append(text)
            elapsed = time.perf_counter() - start
            assert first is not None and usage, (name, response_id, usage)
            assert usage["prompt_tokens"] == len(tokens), (len(tokens), usage)
            record = {
                "name": name,
                "tag": tag,
                "id": response_id,
                "prompt_hash": digest(tokens),
                "prompt_tokens": len(tokens),
                "usage": usage,
                "ttft_seconds": first - start,
                "latency_seconds": elapsed,
                "chunk_arrivals_seconds": arrivals,
                "output_token_ids": output_tokens,
                "text": "".join(output_text),
            }
            if observe:
                plan, admission, completions = await trace.request([name, response_id])
                assert all(r["success"] for r in completions), name
                record.update(plan=plan, admission=admission, completions=completions)
                page = plan["page_tokens"]
                local = admission["local_tokens"]
                remote_end = plan["prompt_tokens"] - 1
                external = admission["external_tokens"]
                if external:
                    assert local + external == remote_end, (name, admission)
                    source = plan["attention_full"]
                    cached = set(source[: local // page])
                    duplicate = sorted(cached.intersection(plan["local"][0]))
                    record["cached_read_destinations"] = duplicate
                    if args.variant == "B":
                        assert not duplicate, (name, duplicate)
                        assert (
                            len(plan["local"][0])
                            == (remote_end + page - 1) // page - local // page
                        )
                    elif local >= page:
                        assert duplicate, (name, "A did not reproduce redundant READ")
            with requests_file.open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            records.append(record)
            return record

        base = await tokenize(
            "Analyze these engineering records.\n"
            + "\n".join(
                f"Record {i}: a worker receives a buffer, computes a result, and caches unchanged data."
                for i in range(6000)
            )
        )
        assert len(base) > 33000
        branch = await tokenize(
            "Alternative continuation: summarize the prior records carefully. " * 128
        )
        tail = await tokenize(
            "\nSummarize the sequence of operations and their purpose."
        )
        workloads = []

        async def pair(length, hit, group, item):
            prefix = await tokenize(f"Experiment {group}, item {item}.\n")
            target = (prefix + base)[: length - len(tail)] + tail
            suffix = branch if branch[0] != target[hit] else branch[1:]
            assert suffix[0] != target[hit]
            seed = target[:hit] + suffix[:129]
            with (args.output / "workload.jsonl").open("a") as stream:
                stream.write(
                    json.dumps(
                        {
                            "group": group,
                            "item": item,
                            "hit": hit,
                            "seed": seed,
                            "target": target,
                        }
                    )
                    + "\n"
                )
            workloads.append(
                {
                    "group": group,
                    "item": item,
                    "hit": hit,
                    "seed_hash": digest(seed),
                    "target_hash": digest(target),
                    "target_length": len(target),
                }
            )
            (args.output / "workload.json").write_text(json.dumps(workloads))
            return seed, target

        async def prepare(pool, condition):
            # Warm P through actual PD requests and wait for real completion/ACK.
            for _, target in pool:
                await complete(target, "prepare-target", count=1)
            await reset_decode()
            if condition != "cold":
                for seed, _ in pool:
                    await complete(seed, "prepare-seed", count=1)

        warmup = await complete(base[:1025], "mechanism-warmup", count=8)
        geometry = workload_geometry(warmup["plan"]["page_tokens"])
        assert geometry["performance_length"] < 1048576
        base *= (geometry["performance_length"] + len(base) - 1) // len(base)
        (args.output / "geometry.json").write_text(json.dumps(geometry, indent=2))
        correctness = []
        for length in range(geometry["mechanism_end"], geometry["mechanism_end"] + 3):
            seed, target = await pair(
                length, geometry["mechanism_hit"], f"mechanism-{length}", 0
            )
            reference = await complete(
                target, "correctness-local-p", url=args.prefill, count=32, observe=False
            )
            for condition in ("cold", "partial"):
                await prepare([(seed, target)], condition)
                actual = await complete(
                    target, f"mechanism-{length}-{condition}", count=32
                )
                assert same_output(actual, reference), (
                    length,
                    condition,
                    "output mismatch",
                )
                local = actual["admission"]["local_tokens"]
                assert local == (
                    0 if condition == "cold" else geometry["mechanism_hit"]
                )
                correctness.append(
                    {"length": length, "condition": condition, "passed": True}
                )
            repeated = await complete(target, f"mechanism-{length}-repeat", count=32)
            assert same_output(repeated, reference), (
                length,
                "repeat output mismatch",
            )
        (args.output / "mechanism.json").write_text(json.dumps(correctness, indent=2))

        summaries = []
        for condition, hit in (("cold", 0), ("high", geometry["performance_hit"])):
            active_seconds = 0.0
            measured = []
            segments = []
            cohort = 0
            while active_seconds < args.duration or len(measured) < args.min_samples:
                pool = [
                    await pair(
                        geometry["performance_length"], hit, f"{condition}-{cohort}", i
                    )
                    for i in range(args.pool_size)
                ]
                await prepare(pool, condition)
                start = time.perf_counter()
                cohort_records = []
                for _, target in pool:
                    record = await complete(target, f"measure-{condition}-{cohort}")
                    local = record["admission"]["local_tokens"]
                    ratio = local / (len(target) - 1)
                    assert local == 0 if condition == "cold" else 0.8 <= ratio <= 0.95
                    cohort_records.append(record)
                # Trace harvesting follows each response, so service-rate
                # accounting uses the measured HTTP intervals, not this wall span.
                seconds = sum(r["latency_seconds"] for r in cohort_records)
                segments.append(
                    {
                        "wall_seconds": time.perf_counter() - start,
                        "http_seconds": seconds,
                        "requests": len(cohort_records),
                    }
                )
                active_seconds += seconds
                measured.extend(cohort_records)
                cohort += 1
            ttft = sorted(r["ttft_seconds"] for r in measured)
            summary = {
                "condition": condition,
                "samples": len(measured),
                "observation_seconds": active_seconds,
                "segments": segments,
                "denominator": "sum of serial HTTP request intervals; preparation excluded",
                "ttft_mean": statistics.mean(ttft),
                "ttft_p50": statistics.median(ttft),
                "ttft_p90": ttft[int(0.9 * (len(ttft) - 1))],
                "output_tokens_per_second": sum(
                    r["usage"]["completion_tokens"] for r in measured
                )
                / active_seconds,
                "requests_per_second": len(measured) / active_seconds,
            }
            summaries.append(summary)
            (args.output / "performance.json").write_text(
                json.dumps(summaries, indent=2)
            )
        (args.output / "complete.json").write_text(
            json.dumps(
                {
                    "variant": args.variant,
                    "geometry": geometry,
                    "correctness": correctness,
                    "performance": summaries,
                    "workload_hash": digest(workloads),
                    "speculative_decoding": False,
                    "concurrent_requests": 1,
                    "wire_bytes_measured": False,
                },
                indent=2,
            )
        )
        print(f"Prefix experiment complete: {args.output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("prefill", "decode", "router", "model"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--variant", choices=("A", "B"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=300)
    parser.add_argument("--min-samples", type=int, default=16)
    parser.add_argument("--pool-size", type=int, default=4)

    async def bounded():
        async with asyncio.timeout(5400):
            await run(parser.parse_args())

    asyncio.run(bounded())
