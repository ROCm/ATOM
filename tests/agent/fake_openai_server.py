# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""A tiny OpenAI-compatible server for agent tests (no GPU, stdlib only).

In-process: ``port, stop = start()``. As a process it accepts the ATOM
server's ``--server-port`` so deploy tests can launch it in place of
``atom.entrypoints.openai_server``.
"""

from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL = "fake-model"

# Fault injection for retry tests: each POST consumes one entry, if any.
#   "503"      -> HTTP 503 before any data
#   "stall"    -> one chunk, then silence longer than the client timeout
#   "truncate" -> one chunk, then close without finishing
FAULTS: list[str] = []
REQUESTS: list[dict] = []


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # keep test output quiet
        pass

    def _fault(self) -> bool:
        if not FAULTS:
            return False
        fault = FAULTS.pop(0)
        if fault == "503":
            self._json({"error": "overloaded"}, 503)
            return True
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        self._sse({"choices": [{"index": 0, "delta": {"content": "Hel"}}]})
        if fault == "stall":
            import time

            time.sleep(3)
        return True  # truncate: return without [DONE]; the connection closes

    def _event(self, name, data):
        self.wfile.write(f"event: {name}\ndata: {json.dumps(data)}\n\n".encode())
        self.wfile.flush()

    def _anthropic(self, body):
        """/v1/messages: text, or a tool_use split across input_json_deltas."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        last = body["messages"][-1]
        text = json.dumps(last["content"])
        self._event(
            "message_start",
            {"type": "message_start", "message": {"usage": {"input_tokens": 9}}},
        )
        if body.get("tools") and "weather" in text and "tool_result" not in text:
            self._event(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "get_weather",
                    },
                },
            )
            for part in ['{"ci', 'ty": "Austin"}']:
                self._event(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "input_json_delta", "partial_json": part},
                    },
                )
            stop = "tool_use"
        else:
            self._event(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text"},
                },
            )
            for word in ["Hello", " from", " messages."]:
                self._event(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "text_delta", "text": word},
                    },
                )
            stop = "end_turn"
        self._event("content_block_stop", {"type": "content_block_stop", "index": 0})
        self._event(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": stop},
                "usage": {"output_tokens": 3},
            },
        )
        self._event("message_stop", {"type": "message_stop"})

    def do_GET(self):
        if self.path.rstrip("/") == "/v1/models":
            self._json({"object": "list", "data": [{"id": MODEL, "object": "model"}]})
        else:
            self.send_error(404)

    def _json(self, obj, code=200):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _sse(self, chunk):
        self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
        self.wfile.flush()

    def do_POST(self):
        path = self.path.rstrip("/")
        if path not in ("/v1/chat/completions", "/v1/messages"):
            self.send_error(404)
            return
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        REQUESTS.append({"path": path, "body": body, "headers": dict(self.headers)})
        if self._fault():
            return
        if path == "/v1/messages":
            self._anthropic(body)
            return
        last = body["messages"][-1]
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()

        def delta(d, finish=None):
            return {"choices": [{"index": 0, "delta": d, "finish_reason": finish}]}

        wants_tool = body.get("tools") and "weather" in str(last.get("content", ""))
        if wants_tool:
            # Tool call split across chunks, as real servers stream it.
            self._sse(delta({"reasoning_content": "need the tool"}))
            self._sse(
                delta(
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"ci',
                                },
                            }
                        ]
                    }
                )
            )
            self._sse(
                delta(
                    {
                        "tool_calls": [
                            {"index": 0, "function": {"arguments": 'ty": "Austin"}'}}
                        ]
                    }
                )
            )
            self._sse(delta({}, "tool_calls"))
        else:
            for word in ["Hello", " from", " fake", "."]:
                self._sse(delta({"content": word}))
            self._sse(delta({}, "stop"))
        self._sse(
            {
                "choices": [],
                "usage": {
                    "prompt_tokens": 7,
                    "completion_tokens": 4,
                    "total_tokens": 11,
                },
            }
        )
        self.wfile.write(b"data: [DONE]\n\n")


def start(port: int = 0):
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server.server_address[1], server.shutdown


if __name__ == "__main__":
    port = int(sys.argv[sys.argv.index("--server-port") + 1])
    print("Server started successfully and ready to accept requests", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
