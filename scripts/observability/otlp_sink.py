#!/usr/bin/env python3
"""Loopback-only OTLP/HTTP trace receiver for isolated PoC evidence."""

import argparse
import gzip
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
    ExportTraceServiceRequest,
)


def value(item):
    kind = item.WhichOneof("value")
    if kind == "array_value":
        return [value(v) for v in item.array_value.values]
    if kind == "kvlist_value":
        return {v.key: value(v.value) for v in item.kvlist_value.values}
    if kind == "bytes_value":
        return item.bytes_value.hex()
    return getattr(item, kind) if kind else None


def decode(body):
    request = ExportTraceServiceRequest.FromString(body)
    for resource in request.resource_spans:
        attrs = {v.key: value(v.value) for v in resource.resource.attributes}
        for scope in resource.scope_spans:
            for span in scope.spans:
                yield {
                    "trace_id": span.trace_id.hex(),
                    "span_id": span.span_id.hex(),
                    "parent_span_id": span.parent_span_id.hex(),
                    "name": span.name,
                    "start_ns": span.start_time_unix_nano,
                    "end_ns": span.end_time_unix_nano,
                    "status_code": span.status.code,
                    "trace_state": span.trace_state,
                    "attributes": {v.key: value(v.value) for v in span.attributes},
                    "resource": attrs,
                }


def receiver(output, port=0):
    path = Path(output)
    if path.exists():
        raise FileExistsError("Use a fresh output for each receiver run")
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path != "/v1/traces":
                self.send_error(404)
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 16 * 1024 * 1024:
                    raise ValueError("Invalid body size")
                body = self.rfile.read(size)
                if self.headers.get("Content-Encoding") == "gzip":
                    body = gzip.decompress(body)
                rows = list(decode(body))
                with lock, path.open("a") as stream:
                    for row in rows:
                        stream.write(json.dumps(row, sort_keys=True) + "\n")
                self.send_response(200)
                self.send_header("Content-Type", "application/x-protobuf")
                self.send_header("Content-Length", "0")
                self.end_headers()
            except Exception:  # noqa: BLE001 - reject malformed telemetry with 400
                self.send_error(400)

        def log_message(self, *args):
            pass

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--port", type=int, default=43180)
    args = parser.parse_args()
    server = receiver(args.output, args.port)
    print(json.dumps({"listening": server.server_address}), flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
