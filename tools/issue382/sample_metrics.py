"""Persist P/D Prometheus snapshots for the issue #382 experiment only."""

import argparse
import concurrent.futures
import gzip
import http.client
import json
import signal
import threading
import time
import urllib.request
from pathlib import Path


def fetch(endpoint):
    role, url = endpoint
    result = {"role": role, "timestamp": time.time(), "url": url}
    try:
        with urllib.request.urlopen(url, timeout=3) as response:
            result["metrics"] = response.read().decode()
    except (OSError, UnicodeError, http.client.HTTPException) as exc:
        result["error"] = str(exc)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefill", required=True)
    parser.add_argument("--decode", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=5)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("interval must be positive")
    endpoints = [("prefill", args.prefill), ("decode", args.decode)]
    stopped = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stopped.set())
    signal.signal(signal.SIGINT, lambda *_: stopped.set())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        if args.check:
            results = list(pool.map(fetch, endpoints))
            for result in results:
                if "error" in result:
                    raise RuntimeError(result)
            for metric in (
                "atom:pd_admission_pending_requests",
                "atom:pd_admission_pending_kv_bytes",
            ):
                if metric not in results[1]["metrics"]:
                    raise RuntimeError(f"decode metrics missing {metric}")
            print("[issue382] P/D metrics reachable; admission series present")
            return
        with gzip.open(args.output, "wt", encoding="utf-8") as output:
            while not stopped.is_set():
                started = time.monotonic()
                for result in pool.map(fetch, endpoints):
                    output.write(json.dumps(result) + "\n")
                output.flush()
                stopped.wait(max(0, args.interval - (time.monotonic() - started)))


if __name__ == "__main__":
    main()
