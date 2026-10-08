# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Keep a node-local LMCache MP server alive for one ATOM prefill worker."""

import argparse
import json
import os
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path


def stop(process):
    if process is None:
        return
    # ATOM spawns engine workers; terminate the entire owned session even when
    # its leader has already exited. Never match unrelated processes by name.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def interrupted(signum, _frame):
    raise SystemExit(128 + signum)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("an ATOM server command is required")
    if os.environ.get("ATOMESH_PD_WORKER_LAYOUT", "multi_node") != "multi_node":
        parser.error("LMCache MP currently requires one worker container per node")
    extra = json.loads(os.environ["ATOM_KV_OFFLOAD_EXTRA_CONFIG"])
    port = int(os.environ.get("LMCACHE_MP_PORT", "5555"))
    http_port = int(os.environ.get("LMCACHE_MP_HTTP_PORT", "19555"))
    if (
        extra.get("lmcache.mp.host") != "tcp://127.0.0.1"
        or int(extra.get("lmcache.mp.port", 0)) != port
    ):
        parser.error("the ATOM connector must point to this node's MP server")
    if not 1 <= port <= 65535 or not 1 <= http_port <= 65535 or port == http_port:
        parser.error("MP RPC and HTTP ports must be distinct valid ports")
    # Do not attach to a stale or unrelated MP server on the host network.
    for candidate in (port, http_port):
        with socket.socket() as check:
            try:
                check.bind(("127.0.0.1", candidate))
            except OSError as error:
                kind = "RPC" if candidate == port else "HTTP metrics"
                raise RuntimeError(
                    f"LMCache MP {kind} port 127.0.0.1:{candidate} is unavailable; "
                    "check the port owner and clean up the previous worker"
                ) from error
    mp_command = [
        "lmcache",
        "server",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--chunk-size",
        os.environ.get("LMCACHE_CHUNK_SIZE", "256"),
        "--null-block-id",
        "-1",
        "--separate-object-groups",
        "--supported-transfer-mode",
        "lmcache_driven",
        "--l1-size-gb",
        os.environ.get("LMCACHE_MP_L1_SIZE_GB", "1000"),
        "--l1-use-lazy",
        "--l1-read-ttl-seconds",
        os.environ.get("LMCACHE_MP_L1_READ_TTL_SECONDS", "900"),
        "--eviction-policy",
        "LRU",
        "--eviction-trigger-watermark",
        os.environ.get("LMCACHE_MP_EVICTION_WATERMARK", "0.98"),
        "--http-host",
        "127.0.0.1",
        "--http-port",
        str(http_port),
    ]
    args.log_dir.mkdir(parents=True, exist_ok=True)
    (args.log_dir / "command.json").write_text(json.dumps(mp_command, indent=2))
    mp = engine = None
    metrics_url = f"http://127.0.0.1:{http_port}/metrics"
    timeout = float(os.environ.get("LMCACHE_MP_STARTUP_TIMEOUT", "300"))
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, interrupted)
    try:
        with (args.log_dir / "server.log").open("w") as log:
            mp = subprocess.Popen(
                mp_command,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        deadline = time.monotonic() + timeout
        while True:
            if mp.poll() is not None:
                raise RuntimeError(f"LMCache MP exited during startup: {mp.returncode}")
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=1):
                    pass
                with urllib.request.urlopen(metrics_url, timeout=2) as response:
                    response.read()
                break
            except (OSError, urllib.error.URLError):
                if time.monotonic() >= deadline:
                    raise TimeoutError("LMCache MP did not become ready")
                time.sleep(1)
        print(
            f"[lmcache-mp] ready pid={mp.pid} rpc={port} metrics={metrics_url}",
            flush=True,
        )
        engine = subprocess.Popen(command, start_new_session=True)
        next_sample = 0.0
        while engine.poll() is None:
            if mp.poll() is not None:
                raise RuntimeError(
                    f"LMCache MP exited while ATOM was running: {mp.returncode}"
                )
            if time.monotonic() >= next_sample:
                try:
                    with urllib.request.urlopen(metrics_url, timeout=2) as response:
                        (args.log_dir / f"metrics-{time.time_ns()}.prom").write_bytes(
                            response.read()
                        )
                except OSError as error:
                    print(f"[lmcache-mp] metrics: {error}", flush=True)
                next_sample = time.monotonic() + 30
            time.sleep(0.5)
        if mp.poll() is not None:
            raise RuntimeError("LMCache MP exited before the ATOM worker completed")
        return engine.returncode
    finally:
        stop(engine)
        stop(mp)


if __name__ == "__main__":
    raise SystemExit(main())
