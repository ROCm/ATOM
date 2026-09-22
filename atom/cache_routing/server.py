# SPDX-License-Identifier: MIT
"""HTTP control plane for bounded catalog recovery and worker residency reports."""

from __future__ import annotations

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen

from atom.cache_routing.catalog import CacheCatalog, SnapshotRequired
from atom.cache_routing.config import CacheRoutingConfig

logger = logging.getLogger(__name__)
MAX_BODY_BYTES = 16 * 1024 * 1024


def read_catalog(path: str) -> dict:
    """Proxy metadata from the scheduler's execution-specific provider."""
    cfg = CacheRoutingConfig.from_env()
    if cfg is None:
        raise ValueError("cache routing is disabled")
    with urlopen(cfg.catalog_url.rstrip("/") + path, timeout=1) as response:
        return json.load(response)


def post_cpu_report(url: str, report: dict) -> None:
    data = json.dumps(report, separators=(",", ":")).encode()
    request = Request(
        url.rstrip("/") + "/v1/cache/cpu",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=1) as response:
        response.read()


class CatalogServer:
    def __init__(self, catalog: CacheCatalog):
        self.catalog = catalog
        endpoint = urlparse(catalog.config.catalog_url)

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def respond(self, status, data):
                payload = json.dumps(data, separators=(",", ":")).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_GET(self):
                try:
                    parsed = urlparse(self.path)
                    query = parse_qs(parsed.query)
                    if parsed.path == "/v1/cache/snapshot":
                        result = catalog.snapshot(
                            query.get("snapshot_id", [None])[0],
                            int(query.get("page_token", ["0"])[0]),
                        )
                    elif parsed.path == "/v1/cache/events":
                        result = catalog.events(
                            query["source_epoch"][0], int(query["after_seq"][0])
                        )
                    elif parsed.path == "/v1/cache/load":
                        with catalog.lock:
                            result = dict(catalog.load)
                    elif parsed.path == "/v1/cache/info":
                        with catalog.lock:
                            result = {**catalog.info, "source_epoch": catalog.epoch}
                    else:
                        self.respond(404, {"error": "unknown catalog endpoint"})
                        return
                    self.respond(200, result)
                except SnapshotRequired as exc:
                    self.respond(
                        410, {"error": "snapshot_required", "detail": str(exc)}
                    )
                except (KeyError, ValueError, TypeError) as exc:
                    self.respond(400, {"error": str(exc)})

            def do_POST(self):
                if self.path != "/v1/cache/cpu":
                    self.respond(404, {"error": "unknown catalog endpoint"})
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if not 0 < length <= MAX_BODY_BYTES:
                        self.respond(
                            413, {"error": "CPU report exceeds metadata budget"}
                        )
                        return
                    update = json.loads(self.rfile.read(length))
                    if update["content_namespace"] != catalog.config.content_namespace:
                        raise ValueError("CPU namespace differs from execution")
                    catalog.cpu_update(update)
                    self.respond(200, {"ok": True})
                except SnapshotRequired as exc:
                    self.respond(410, {"error": str(exc)})
                except (KeyError, ValueError, TypeError) as exc:
                    self.respond(400, {"error": str(exc)})

        self.server = ThreadingHTTPServer((endpoint.hostname, endpoint.port), Handler)
        self.thread = threading.Thread(
            target=self.server.serve_forever, name="cache-catalog", daemon=True
        )
        self.thread.start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
