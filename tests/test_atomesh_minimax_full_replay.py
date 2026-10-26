# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU checks for request construction and error classification, no model run."""

import importlib.util
import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / ".github/scripts/atomesh/minimax_full_model_replay.py"
)
SPEC = importlib.util.spec_from_file_location("minimax_full_model_replay", SCRIPT)
replay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(replay)


class CharacterTokenizer:
    def encode(self, text, **kwargs):
        return list(text.encode())

    def decode(self, tokens, **kwargs):
        return bytes(tokens).decode()


class ReplayTest(unittest.TestCase):
    def test_cache_primer_excludes_template_suffix(self):
        tokenizer = CharacterTokenizer()
        render = lambda messages: (
            "<user>" + messages[0]["content"] + "</user><assistant><think>"
        )
        messages, ids, warm, warm_ids, common = replay.build_pair(
            tokenizer, render, 68, 777, 512
        )
        self.assertEqual(len(ids), 777)
        self.assertEqual(common // 128 * 128, 512)
        self.assertLess(common, len(warm_ids))
        self.assertEqual(ids[:512], warm_ids[:512])
        self.assertTrue(messages[0]["content"].startswith(warm[0]["content"]))
        self.assertEqual(replay.build_pair(tokenizer, render, 68, 777, 512)[1], ids)
        self.assertLess(
            replay.shared_prefix(
                ids, replay.build_pair(tokenizer, render, 65, 777, 512)[1]
            ),
            128,
        )

    def test_full_input_is_not_confused_with_partial_step(self):
        self.assertEqual(
            sum(prefix + new for _, _, prefix, new in replay.TARGETS), 1350868
        )
        self.assertEqual(replay.TARGETS[-1], (74, 170398, 164864, 128))
        self.assertGreater(170398, 164864 + 128)

    def test_sse_success_and_failures_are_distinguished(self):
        cases = [
            (200, 777, True, True, "completed"),
            (200, 776, True, True, "failed"),
            (200, 777, False, True, "failed"),
            (200, 777, True, False, "failed"),
            (500, 777, True, True, "failed"),
        ]
        for status, length, done, finished, expected in cases:
            with self.subTest(
                status=status, length=length, done=done, finished=finished
            ):
                seen = []

                class Handler(BaseHTTPRequestHandler):
                    def log_message(self, *args):
                        pass

                    def do_POST(
                        self,
                        seen=seen,
                        status=status,
                        length=length,
                        done=done,
                        finished=finished,
                    ):
                        seen.append(
                            json.loads(
                                self.rfile.read(int(self.headers["Content-Length"]))
                            )
                        )
                        self.send_response(status)
                        self.end_headers()
                        body = {
                            "choices": [
                                {"finish_reason": "length" if finished else None}
                            ],
                            "usage": {"prompt_tokens": length, "completion_tokens": 1},
                        }
                        self.wfile.write(
                            b"data: " + json.dumps(body).encode() + b"\n\n"
                        )
                        if done:
                            self.wfile.write(b"data: [DONE]\n\n")

                server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
                worker = threading.Thread(target=server.serve_forever)
                worker.start()
                try:
                    with tempfile.TemporaryDirectory() as directory:
                        payload = replay.make_payload(
                            "test", [{"role": "user", "content": "hello"}]
                        )
                        result = replay.send_request(
                            f"http://127.0.0.1:{server.server_port}",
                            replay.compact(payload),
                            777,
                            5,
                            Path(directory) / "response.sse",
                        )
                        self.assertEqual(result["status"], expected)
                        self.assertEqual(seen, [payload])
                finally:
                    server.shutdown()
                    worker.join()
                    server.server_close()

    def test_batch_extraction_uses_physical_lines(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prefill.log"
            path.write_bytes(
                b"progress\rprogress\n[atom] Scheduled prefill batch: 2 reqs, 10 new tokens (done: [128, 256], new: [6, 4]), req_ids: (8, 9)\n"
            )
            batches = replay.extract_batches(path)
            self.assertEqual(len(batches), 1)
            self.assertEqual(batches[0]["line"], 2)
            self.assertEqual(batches[0]["total_kv"], 394)


if __name__ == "__main__":
    unittest.main()
