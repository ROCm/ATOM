#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Profile a single exact-length token-ID request after unprofiled warmup."""

import argparse
import json
import time
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--tokens", type=int, choices=[2048, 131072], default=2048)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.warmups < 1:
        parser.error("At least one warmup is required")

    def request(path, body):
        req = urllib.request.Request(
            f"http://127.0.0.1:{args.port}" + path,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=1800) as response:
            return json.load(response)

    prompt = {
        "model": args.model,
        "prompt_token_ids": [1] * args.tokens,
        "max_tokens": 1,
        "temperature": 0,
    }
    for _ in range(args.warmups):
        request("/v1/completions", prompt)
    request("/start_profile", {})
    start = time.time()
    try:
        result = request("/v1/completions", prompt)
    finally:
        profile = request("/stop_profile", {})
    args.output.write_text(
        json.dumps(
            {
                "start_time": start,
                "end_time": time.time(),
                "tokens": args.tokens,
                "result": result,
                "profile": profile,
                "scope": "Repeated-token diagnostic input; not representative serving throughput",
            },
            indent=2,
        )
        + "\n"
    )
    if result.get("usage", {}).get("prompt_tokens") != args.tokens:
        raise RuntimeError("Server prompt length differs from requested token count")
    if not profile.get("traces") or not all(
        t.get("trace_dir") for t in profile["traces"]
    ):
        raise RuntimeError("No per-rank traces; launch with --profile-dir")


if __name__ == "__main__":
    main()
