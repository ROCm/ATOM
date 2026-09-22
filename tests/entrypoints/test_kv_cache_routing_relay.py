# SPDX-License-Identifier: MIT
"""Run the built Rust router against real HTTP catalogs and controlled P/D servers."""

import json
import os
import socket
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

import pytest

from atom.cache_routing.catalog import CacheCatalog
from atom.cache_routing.config import CacheRoutingConfig
from atom.cache_routing.keys import content_keys, root_key
from atom.cache_routing.server import CatalogServer
from atom.distributed.kv_events import BlockRemoved, BlockStored

BINARY = os.environ.get("ATOMESH_BINARY", "")
pytestmark = pytest.mark.skipif(
    not BINARY or not Path(BINARY).is_file(), reason="requires a built ATOMESH_BINARY"
)
NS = "01" * 32
TOKENS = list(range(1024))


def _port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _request(url, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = Request(url, data=data, headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=5) as response:
        return json.load(response)


class Execution:
    def __init__(self, name, role):
        self.name, self.role = name, role
        self.requests = []
        self.renders = 0
        self.cfg = CacheRoutingConfig(name, "http://127.0.0.1:0", NS)
        self.catalog = CacheCatalog(
            self.cfg, f"{role}-layout", 16 if role == "prefill" else 64, 1
        )
        self.catalog_server = CatalogServer(self.catalog)
        base = f"http://127.0.0.1:{self.catalog_server.server.server_port}/v1/cache"
        self.catalog.info = {
            "protocol_version": 1,
            "execution_id": name,
            "content_namespace": NS,
            "canonical_hash": "sha256-prefix-u32le-v1",
            "canonical_block_size_tokens": 16,
            "hash_block_size_tokens": self.catalog.hash_span,
            "lmcache_chunk_size_tokens": 256,
            "layout_id": f"{role}-layout",
            "cpu_layout_id": f"{role}-layout",
            "storage_domain_id": name,
            "min_load_tokens": 0,
            "catalog_http": base,
            "capabilities": {
                "exact_prefix_reuse": True,
                "cache_load_policy_hint": True,
                "pd_delta_receive": True,
            },
            "costs": {
                "prefill_curves": [
                    {
                        "max_context_tokens": 16384,
                        "points": [[16384, 176 if name == "p2" else 160]],
                    }
                ],
                "h2d_curve": [[16384, 20]],
                "decode_step_ms": 2.1 if name == "d2" else 2.0,
            },
            "transfer_paths": [
                {
                    "verified": True,
                    "protocol": "mooncake",
                    "destination_execution_id": d,
                    "source_layout_id": "prefill-layout",
                    "destination_layout_id": "decode-layout",
                    "transfer_curve": [[16384, 80]],
                }
                for d in ("d1", "d2")
            ],
        }
        execution = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def reply(self, data):
                body = json.dumps(data).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path == "/v1/models":
                    self.reply({"data": [{"id": "harness"}]})
                elif self.path in ("/server_info", "/get_server_info"):
                    self.reply(
                        {
                            "model_id": "harness",
                            "served_model_name": "harness",
                            "tp_size": 1,
                            "dp_size": 1,
                            "cache_routing": {"catalog_http": base},
                        }
                    )
                elif self.path == "/kv_transfer_info":
                    self.reply(
                        {
                            "tp_size": 1,
                            "dp_size": 1,
                            "kv_role": (
                                "kv_producer" if role == "prefill" else "kv_consumer"
                            ),
                        }
                    )
                else:
                    self.reply({"status": "healthy"})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self.path.endswith("/render"):
                    execution.renders += 1
                    self.reply([{"token_ids": TOKENS}])
                    return
                execution.requests.append(body)
                dispatch = body.get("routing_hints", {}).get("dispatch_id")
                if dispatch:
                    execution.catalog.accepted_dispatches.append(dispatch)
                result = {
                    "id": "test",
                    "object": "text_completion",
                    "model": "harness",
                    "choices": [{"text": name, "index": 0, "finish_reason": "stop"}],
                    "usage": {},
                }
                if role == "prefill":
                    result["kv_transfer_params"] = {
                        "remote_engine_id": name,
                        "remote_host": "127.0.0.1",
                        "remote_port": 1234,
                    }
                self.reply(result)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def sample(self):
        with self.catalog.lock:
            self.catalog.load = {
                "sample_time_unix_ns": str(time.time_ns()),
                "pending_prefill_tokens": 0,
                "requests_waiting": 0,
                "kv_blocks_free": 65536,
                "offload": {"loads_pending": 0},
                "accepted_dispatch_ids": list(self.catalog.accepted_dispatches),
            }

    def warm_cpu(self):
        keys = content_keys(NS, TOKENS, 16)
        events = [
            {
                "readable": True,
                "chunk_id": keys[end // 16 - 1],
                "token_start": end - 256,
                "token_end": end,
                "content_keys": keys[(end - 256) // 16 : end // 16],
                "size_bytes": 4096,
                "parent_key": (
                    keys[(end - 256) // 16 - 1] if end > 256 else root_key(NS, 16).hex()
                ),
            }
            for end in range(256, 1025, 256)
        ]
        self.catalog.cpu_update(
            {
                "rank": 0,
                "source_epoch": "cpu",
                "seq": "1",
                "after_seq": "0",
                "snapshot": True,
                "layout_id": "prefill-layout",
                "chunk_size": 256,
                "events": events,
            }
        )

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.catalog_server.close()


def test_exact_cpu_hbm_pair_and_eviction_through_rust_relay(tmp_path):
    executions = [
        Execution(name, role)
        for name, role in [
            ("p1", "prefill"),
            ("p2", "prefill"),
            ("d1", "decode"),
            ("d2", "decode"),
        ]
    ]
    p1, p2, d1, d2 = executions
    p2.warm_cpu()
    d2.catalog.hbm_events(
        [
            BlockStored(
                block_hashes=list(range(1, 17)),
                parent_block_hash=None,
                token_ids=TOKENS,
                block_size=64,
            )
        ]
    )
    stop = threading.Event()

    def samples():
        while not stop.is_set():
            for execution in executions:
                execution.sample()
            # Heartbeat the unchanged native source; no lookup or pin.
            p2.catalog.cpu_update(
                {
                    "rank": 0,
                    "source_epoch": "cpu",
                    "seq": "1",
                    "after_seq": "1",
                    "snapshot": False,
                    "layout_id": "prefill-layout",
                    "chunk_size": 256,
                    "events": [],
                }
            )
            stop.wait(0.1)

    sampler = threading.Thread(target=samples, daemon=True)
    sampler.start()
    port = _port()
    metrics_port = _port()
    command = [
        BINARY,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--backend",
        "atom",
        "--pd-disaggregation",
        "--policy",
        "kv_cache_aware",
        "--prefill-policy",
        "kv_cache_aware",
        "--decode-policy",
        "kv_cache_aware",
        "--prometheus-port",
        str(metrics_port),
        "--prefill",
        p1.url,
        "--prefill",
        p2.url,
        "--decode",
        d1.url,
        "--decode",
        d2.url,
    ]
    log_path = tmp_path / "router.log"
    with log_path.open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if process.poll() is not None:
                pytest.fail(log_path.read_text())
            try:
                _request(f"http://127.0.0.1:{port}/health")
                break
            except (URLError, json.JSONDecodeError):
                time.sleep(0.1)
        time.sleep(0.7)  # Allow the background index's first snapshot/replay.
        result = _request(
            f"http://127.0.0.1:{port}/v1/completions",
            {"model": "harness", "prompt": "test", "max_tokens": 1},
        )
        assert result["choices"][0]["text"] == "d2", log_path.read_text()
        assert not p1.requests and len(p2.requests) == 1
        assert p2.requests[0]["max_tokens"] == 1
        assert p2.requests[0]["routing_hints"]["cache_load_policy"] == "auto"
        assert d2.requests[0]["kv_transfer_params"]["remote_engine_id"] == "p2"
        assert sum(e.renders for e in executions) == 1
        # Removing the first CPU chunk leaves a hole; no HBM+CPU union is allowed.
        p2.catalog.cpu_update(
            {
                "rank": 0,
                "source_epoch": "cpu",
                "seq": "2",
                "after_seq": "1",
                "snapshot": False,
                "layout_id": "prefill-layout",
                "chunk_size": 256,
                "events": [
                    {
                        "chunk_id": content_keys(NS, TOKENS[:256], 16)[-1],
                        "readable": False,
                    }
                ],
            }
        )
        d2.catalog.hbm_events([BlockRemoved(block_hashes=list(range(1, 17)))])
        time.sleep(0.4)
        result = _request(
            f"http://127.0.0.1:{port}/v1/completions",
            {"model": "harness", "prompt": "test", "max_tokens": 1},
        )
        assert result["choices"][0]["text"] == "d1", log_path.read_text()
        assert (
            len(p1.requests) == 1
            and p1.requests[0]["routing_hints"]["cache_load_policy"] == "skip"
        )
        assert d1.requests[0]["kv_transfer_params"]["remote_engine_id"] == "p1"
        # CI distinguishes actual calibrated selections from the new policy's
        # load fallback. Check the real exported counters, including fallback.
        with urlopen(f"http://127.0.0.1:{metrics_port}/metrics", timeout=5) as response:
            metrics = response.read().decode()
        assert (
            'atomesh_kv_cache_routing_decisions_total{outcome="selected"} 2' in metrics
        )
        for execution in (p1, p2):
            with execution.catalog.lock:
                execution.catalog.info["costs"] = {}
        time.sleep(0.4)
        _request(
            f"http://127.0.0.1:{port}/v1/completions",
            {"model": "harness", "prompt": "test", "max_tokens": 1},
        )
        with urlopen(f"http://127.0.0.1:{metrics_port}/metrics", timeout=5) as response:
            metrics = response.read().decode()
        assert (
            'atomesh_kv_cache_routing_decisions_total{outcome="fallback"} 1' in metrics
        )
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        stop.set()
        sampler.join(timeout=2)
        for execution in executions:
            execution.close()
