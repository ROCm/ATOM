#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Length-based full-model candidate; run only through ATOMesh benchmark CI."""

import argparse
import ast
import gzip
import hashlib
import http.client
import json
import os
import random
import re
import stat
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlsplit

# Original prefill log, run 38664663067 attempt 1, physical line 3329.
# The last request was still chunking: its full prompt is 170398, not 164992.
# For 68, 131072 is the frontier at the failed step, not its initial cache hit.
TARGETS = (
    (68, 142015, 131072, 10943),
    (65, 149072, 146688, 2384),
    (69, 165518, 165376, 142),
    (70, 178412, 176000, 2412),
    (73, 172112, 171776, 336),
    (71, 175618, 171648, 3970),
    (72, 203129, 190720, 12409),
    (74, 170398, 164864, 128),
)
WORDS = """river mountain ocean forest cloud light blue green quiet clear paper
window table road city garden stone field morning evening water wind rain snow
read write build check run walk turn open close send return count compare find
small large first second last next previous value number list map tree line
request response memory cache token model test result data process thread time
one two three four five six seven eight nine ten alpha beta gamma delta""".split()  # noqa: SIM905


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()


def save_gzip(path, data):
    path.write_bytes(gzip.compress(data, mtime=0))


def shared_prefix(a, b):
    for i, (left, right) in enumerate(zip(a, b)):
        if left != right:
            return i
    return min(len(a), len(b))


def build_pair(tokenizer, render, original_id, full_length, prefix_length):
    """Fit chat text and verify the *rendered* token prefix, including framing."""
    rng = random.Random(42 + original_id)
    text = f"Replay candidate request {original_id}. " + " ".join(
        rng.choices(WORDS, k=full_length + 1024)
    )
    content_ids = tokenizer.encode(text, add_special_tokens=False)

    def candidate(count):
        content = tokenizer.decode(
            content_ids[:count],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        messages = [{"role": "user", "content": content}]
        # The server renders a string and calls encode() with its defaults.
        return messages, tokenizer.encode(render(messages))

    count = full_length - len(
        tokenizer.encode(render([{"role": "user", "content": ""}]))
    )
    tried = set()
    while count not in tried:
        if not 1 <= count <= len(content_ids):
            raise ValueError("Unable to fit full prompt into generated content")
        tried.add(count)
        messages, full_ids = candidate(count)
        delta = full_length - len(full_ids)
        if delta == 0:
            break
        count += delta
    else:
        raise ValueError(f"Tokenizer cannot fit exact prompt length {full_length}")

    # Ending a chat inserts template suffix tokens. Do not mistake those for
    # shared content. Aim just inside the desired 128-token cache block edge.
    warm_count = min(count - 1, prefix_length)
    tried = set()
    while warm_count not in tried:
        if not 1 <= warm_count < count:
            raise ValueError("Unable to fit cache primer")
        tried.add(warm_count)
        warm_messages, warm_ids = candidate(warm_count)
        common = shared_prefix(warm_ids, full_ids)
        if common // 128 * 128 == prefix_length:
            return messages, full_ids, warm_messages, warm_ids, common
        warm_count += prefix_length + 1 - common
    raise ValueError(f"Tokenizer cannot fit cache prefix {prefix_length}")


def make_payload(model, messages):
    return {
        "model": model,
        "messages": messages,
        "chat_template_kwargs": {"thinking_mode": "enabled"},
        "max_tokens": 1,
        "temperature": 0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }


def send_request(base_url, body, expected_tokens, timeout, response_path, barrier=None):
    """No retries: an HTTP error, incomplete SSE, or wrong token count fails."""
    url = urlsplit(base_url)
    if url.scheme != "http":
        raise ValueError("Replay expects the local HTTP mesh endpoint")
    conn = http.client.HTTPConnection(url.hostname, url.port, timeout=timeout)
    started = time.monotonic()
    usage = None
    finished = done = False
    try:
        conn.connect()
        if barrier is not None:
            barrier.wait(timeout=30)
        started = time.monotonic()
        conn.request(
            "POST",
            "/v1/chat/completions",
            body=body,
            headers={"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}: {response.read(4096)!r}")
        with response_path.open("wb") as output:
            while True:
                remaining = timeout - (time.monotonic() - started)
                if remaining <= 0:
                    raise TimeoutError("Replay request deadline expired")
                if conn.sock is not None:
                    conn.sock.settimeout(remaining)
                line = response.readline()
                if not line:
                    break
                output.write(line)
                output.flush()
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    done = True
                    break
                chunk = json.loads(data)
                if "error" in chunk:
                    raise RuntimeError(f"Streaming error: {chunk['error']}")
                usage = chunk.get("usage") or usage
                finished |= any(
                    c.get("finish_reason") is not None for c in chunk.get("choices", [])
                )
        if not done or not finished:
            raise RuntimeError("Incomplete response: missing finish_reason or [DONE]")
        if not usage or usage.get("prompt_tokens") != expected_tokens:
            raise RuntimeError(
                f"Prompt length mismatch: wanted {expected_tokens}, usage={usage}"
            )
        if usage.get("completion_tokens") != 1:
            raise RuntimeError(f"Expected one output token, usage={usage}")
        return {
            "status": "completed",
            "usage": usage,
            "seconds": time.monotonic() - started,
        }
    except (OSError, ValueError, RuntimeError, http.client.HTTPException) as exc:
        # A broken start barrier must not leave the other seven clients waiting.
        if barrier is not None:
            barrier.abort()
        return {
            "status": "failed",
            "error": str(exc),
            "seconds": time.monotonic() - started,
        }
    finally:
        conn.close()


def extract_batches(log_path):
    pattern = re.compile(
        r"Scheduled prefill batch: (\d+) reqs, (\d+) new tokens "
        r"\(done: (\[[^]]*\]), new: (\[[^]]*\])\), req_ids: (\([^)]*\))"
    )
    batches = []
    for line_number, line in enumerate(
        log_path.read_bytes().decode(errors="replace").split("\n"), 1
    ):
        match = pattern.search(line)
        if match:
            done, new, ids = (ast.literal_eval(match[i]) for i in (3, 4, 5))
            batches.append(
                {
                    "line": line_number,
                    "batch_size": int(match[1]),
                    "done": done,
                    "new": new,
                    "request_ids": ids,
                    "total_kv": sum(done) + sum(new),
                }
            )
    return batches


def write_control(path, payload):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(path)


def trigger_dumps(directory, stop, record):
    # Give a stalled batch 30 seconds, then request Flight Recorder dumps while
    # native faulthandler independently captures all Python thread stacks.
    if stop.wait(30):
        return
    outcomes = []
    for path in directory.rglob("nccl-pipe-*"):
        try:
            if not stat.S_ISFIFO(path.stat().st_mode):
                continue
            fd = os.open(path, os.O_WRONLY | os.O_NONBLOCK)
            try:
                os.write(fd, b"1")
            finally:
                os.close(fd)
            outcomes.append({"pipe": str(path), "status": "requested"})
        except OSError as exc:
            outcomes.append({"pipe": str(path), "status": "failed", "error": str(exc)})
    record(
        {
            "phase": "diagnostics",
            "status": "dump_requested",
            "pipes": outcomes,
            "note": "An empty pipe list means this PyTorch build did not expose the configured FIFO.",
        }
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--prefill-log", required=True, type=Path)
    parser.add_argument("--request-timeout", type=float, default=660)
    parser.add_argument("--gate-dir", type=Path)
    parser.add_argument("--nccl-pipe-dir", type=Path)
    parser.add_argument("--target-timeout", type=float, default=75)
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    if args.gate_dir is not None:
        args.gate_dir.mkdir(parents=True, exist_ok=True)
    # CPU tokenizer only. No model or GPU operation is run by this client.
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, use_fast=True, trust_remote_code=True
    )
    probe = "Hello world 你好"
    if tokenizer.decode(tokenizer.encode(probe), skip_special_tokens=True) != probe:
        tokenizer = PreTrainedTokenizerFast.from_pretrained(args.model)

    def render(messages):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            thinking_mode="enabled",
        )

    template_prefix = shared_prefix(
        tokenizer.encode(render([{"role": "user", "content": "alpha"}])),
        tokenizer.encode(render([{"role": "user", "content": "beta"}])),
    )
    template_cache_tokens = template_prefix // 128 * 128
    manifest = {
        "kind": "synthetic_lengths_full_model",
        "seed": 42,
        "shared_template_cache_tokens": template_cache_tokens,
        "source_run": "https://github.com/ROCm/ATOM/actions/runs/38664663067/attempts/1",
        "source_line": 3329,
        "original_batch_total_kv": 1350868,
        "original_inputs_recovered": False,
        "note": "Eight fresh primers replace original history; the optional CI gate orders target admission and validates the first batch.",
        "gated": args.gate_dir is not None,
        "requests": [],
    }
    prepared = []
    for original_id, length, prefix, original_new in TARGETS:
        messages, ids, warm_messages, warm_ids, common = build_pair(
            tokenizer, render, original_id, length, prefix
        )
        for previous in prepared:
            if shared_prefix(ids, previous["ids"]) // 128 * 128 > template_cache_tokens:
                raise ValueError(
                    "Unexpected shared content blocks beyond the system template"
                )
        full_body = compact(make_payload(args.model, messages))
        warm_body = compact(make_payload(args.model, warm_messages))
        for phase, body, token_ids in (
            ("warm", warm_body, warm_ids),
            ("target", full_body, ids),
        ):
            save_gzip(output / f"{phase}-{original_id}.request.json.gz", body)
            save_gzip(
                output / f"{phase}-{original_id}.tokens.json.gz", compact(token_ids)
            )
        manifest["requests"].append(
            {
                "original_id": original_id,
                "full_prompt_tokens": len(ids),
                "primer_prompt_tokens": len(warm_ids),
                "shared_prefix_tokens": common,
                "expected_cache_hit_tokens": prefix,
                "original_step_new_tokens": original_new,
                "prompt_token_ids_sha256": hashlib.sha256(compact(ids)).hexdigest(),
            }
        )
        prepared.append(
            {
                "original_id": original_id,
                "ids": ids,
                "warm_ids": warm_ids,
                "warm_body": warm_body,
                "full_body": full_body,
            }
        )
        print(
            f"[replay] prepared original_id={original_id} full={len(ids)} shared={common} cached_blocks={prefix}",
            flush=True,
        )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    lock = threading.Lock()

    def record(event):
        with lock, (output / "events.jsonl").open("a") as stream:
            event["unix_time"] = time.time()
            stream.write(json.dumps(event) + "\n")
            print("[replay] " + json.dumps(event), flush=True)

    def run_one(item, phase, barrier=None):
        original_id = item["original_id"]
        record({"phase": phase, "original_id": original_id, "status": "started"})
        is_warm = phase == "warm"
        result = send_request(
            args.base_url,
            item["warm_body" if is_warm else "full_body"],
            len(item["warm_ids" if is_warm else "ids"]),
            args.target_timeout
            if args.gate_dir is not None and not is_warm
            else args.request_timeout,
            output / f"{phase}-{original_id}.response.sse",
            barrier,
        )
        record({"phase": phase, "original_id": original_id, **result})
        return result["status"] == "completed"

    ok = False
    dump_stop = threading.Event()
    dump_thread = None
    try:
        # Sequential primers avoid target batch pressure while building caches.
        for item in prepared:
            if not run_one(item, "warm"):
                return 1
        if args.gate_dir is not None:
            args.gate_dir.mkdir(parents=True, exist_ok=True)
            if (args.gate_dir / "armed.json").exists() or (
                args.gate_dir / "client-done.json"
            ).exists():
                raise RuntimeError("Refusing to reuse a stale replay gate directory")
            write_control(
                args.gate_dir / "armed.json",
                {
                    "lengths": [target[1] for target in TARGETS],
                    "done": [target[2] for target in TARGETS],
                    "new": [target[3] for target in TARGETS],
                },
            )
            record({"phase": "gate", "status": "armed"})
            dump_thread = threading.Thread(
                target=trigger_dumps,
                args=(args.nccl_pipe_dir or args.gate_dir, dump_stop, record),
                daemon=True,
            )
            dump_thread.start()
        barrier = threading.Barrier(len(prepared))
        with ThreadPoolExecutor(max_workers=len(prepared)) as pool:
            futures = [
                pool.submit(run_one, item, "target", barrier) for item in prepared
            ]
            outcomes = [future.result() for future in futures]
            ok = all(outcomes)
        if args.gate_dir is not None:
            gate_file = args.gate_dir / "gate-batch.json"
            matched = (
                gate_file.exists()
                and json.loads(gate_file.read_text()).get("matched") is True
            )
            ok = ok and matched
            record({"phase": "gate", "status": "verified" if matched else "failed"})
        return 0 if ok else 1
    finally:
        dump_stop.set()
        if dump_thread is not None:
            dump_thread.join(timeout=2)
        if args.gate_dir is not None:
            write_control(
                args.gate_dir / "client-done.json",
                {"requests_completed": ok, "unix_time": time.time()},
            )
            # Let each process query its final GPU markers before service cleanup.
            time.sleep(2)
        batches = extract_batches(args.prefill_log) if args.prefill_log.exists() else []
        desired_done = [target[2] for target in TARGETS]
        desired_new = [target[3] for target in TARGETS]
        summary = {
            "requests_completed": ok,
            "classification": "candidate_completed_no_hang"
            if ok
            else "request_failed_inspect_service_logs",
            "exact_original_batch_shape_seen": any(
                b["done"] == desired_done and b["new"] == desired_new for b in batches
            ),
            "max_batch_total_kv": max((b["total_kv"] for b in batches), default=0),
            "scheduled_batches": batches,
        }
        (output / "replay-summary.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
        print(
            "[replay] "
            + json.dumps(
                {k: v for k, v in summary.items() if k != "scheduled_batches"}
            ),
            flush=True,
        )


if __name__ == "__main__":
    raise SystemExit(main())
