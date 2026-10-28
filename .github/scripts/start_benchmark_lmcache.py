"""Start the opt-in, container-owned LMCache MP service for benchmark cells."""

import os
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path


def main():
    if os.environ.get("ATOM_BENCHMARK_LMCACHE_MP", "0") != "1":
        return

    port = int(os.environ.get("ATOM_BENCHMARK_LMCACHE_PORT", "24850"))
    http_port = int(os.environ.get("ATOM_BENCHMARK_LMCACHE_HTTP_PORT", "24851"))
    l1_gb = float(os.environ.get("ATOM_BENCHMARK_LMCACHE_L1_GB", "256"))
    if not (0 < port < 65536 and 0 < http_port < 65536 and port != http_port):
        raise ValueError("LMCache requires two distinct valid ports")
    if not 0 < l1_gb <= 256:
        raise ValueError("LMCache benchmark L1 must be in (0, 256] GiB")
    # Fail on an occupied endpoint; never reuse or stop another cache service.
    for endpoint in (port, http_port):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", endpoint))

    command = [
        "lmcache",
        "server",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--http-host",
        "127.0.0.1",
        "--http-port",
        str(http_port),
        "--chunk-size",
        "256",
        "--null-block-id",
        "-1",
        "--separate-object-groups",
        "--supported-transfer-mode",
        "lmcache_driven",
        "--l1-size-gb",
        str(l1_gb),
        "--l1-read-ttl-seconds",
        "900",
        "--eviction-policy",
        "LRU",
        "--eviction-trigger-watermark",
        "0.98",
    ]
    log_path = Path("/tmp/atom_lmcache_mp.log")
    with log_path.open("w") as log:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            env={**os.environ, "LMCACHE_USAGE_STATS_ENABLED": "False"},
            start_new_session=True,
        )
    # The enclosing benchmark container owns the detached process and tears it
    # down on success, failure and cancellation. Readiness checks its engine,
    # not merely the listening HTTP socket.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic() + 180
    try:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"LMCache exited with code {process.returncode}")
            try:
                with opener.open(
                    f"http://127.0.0.1:{http_port}/healthcheck", timeout=2
                ) as response:
                    if response.status == 200:
                        print(f"LMCache MP ready on {port}; L1={l1_gb:g} GiB")
                        return
            except (urllib.error.URLError, TimeoutError):
                pass
            time.sleep(1)
        raise TimeoutError("LMCache MP did not become ready within 180 seconds")
    except BaseException:
        if process.poll() is None:
            process.terminate()
        print(log_path.read_text(errors="replace")[-12000:])
        raise


if __name__ == "__main__":
    main()
