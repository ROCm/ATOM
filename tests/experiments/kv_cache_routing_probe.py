# SPDX-License-Identifier: MIT
"""Probe already-running native ATOM P/D executions; run inside the runtime image.

This records requests and catalog/load facts, not a calibrated benchmark. Ports
must already be configured. Repeated outputs are compared without normalizing
away differences. No environment or model is installed by this script.
"""

import argparse
import itertools
import json
import time
from pathlib import Path
from urllib.request import Request, urlopen


def call(url, path, body=None):
    data = None if body is None else json.dumps(body).encode()
    request = Request(
        url.rstrip("/") + path,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=180) as response:
        return json.load(response)


def probe(args):
    endpoints = args.prefill + args.decode
    info = {url: call(url, "/kv_transfer_info") for url in endpoints}
    result = {
        "discovery": info,
        "catalog_info": {url: call(url, "/v1/cache/info") for url in endpoints},
        "pairs": [],
    }
    body = {
        "model": args.model,
        "prompt": args.prompt,
        "max_tokens": args.max_tokens,
        "temperature": 0,
        "top_k": 1,
        "stream": False,
    }
    for repeat in range(args.repeat):
        for p, d in itertools.product(args.prefill, args.decode):
            before = {url: call(url, "/v1/cache/load") for url in (p, d)}
            started = time.perf_counter()
            prefill = call(
                p,
                "/v1/completions",
                {
                    **body,
                    "max_tokens": 1,
                    "kv_transfer_params": {
                        "do_remote_decode": True,
                        "do_remote_prefill": False,
                    },
                },
            )
            p_ms = (time.perf_counter() - started) * 1000
            transfer = dict(prefill["kv_transfer_params"])
            transfer["remote_dp_size"] = info[p]["dp_size"]
            transfer["remote_tp_size"] = info[p]["tp_size"]
            if "dp_rank" in transfer:
                transfer["remote_dp_rank"] = transfer["dp_rank"]
            started = time.perf_counter()
            response = call(
                d, "/v1/completions", {**body, "kv_transfer_params": transfer}
            )
            d_ms = (time.perf_counter() - started) * 1000
            time.sleep(1.2)  # Let the existing engine metrics sampler publish.
            row = {
                "repeat": repeat,
                "prefill": p,
                "decode": d,
                "prefill_http_ms": p_ms,
                "decode_http_ms": d_ms,
                "response": response,
                "before": before,
                "after": {url: call(url, "/v1/cache/load") for url in (p, d)},
            }
            result["pairs"].append(row)
            Path(args.output).write_text(json.dumps(result, indent=2))
            print(
                json.dumps(
                    {k: v for k, v in row.items() if k not in ("before", "after")}
                ),
                flush=True,
            )
    result["identical_text"] = (
        len({row["response"]["choices"][0]["text"] for row in result["pairs"]}) == 1
    )
    result["catalogs"] = {url: call(url, "/v1/cache/snapshot") for url in endpoints}
    Path(args.output).write_text(json.dumps(result, indent=2))
    print("identical_text:", result["identical_text"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefill", action="append", required=True)
    parser.add_argument("--decode", action="append", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--output", default="/tmp/kv-routing-probe.json")
    parser.add_argument(
        "--prompt",
        default="A cache stores results so repeated work can be avoided. " * 32
        + "\nQuestion: What does a cache store?\nAnswer:",
    )
    probe(parser.parse_args())


if __name__ == "__main__":
    main()
