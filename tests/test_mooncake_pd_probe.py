"""Keep the offload gate from treating HTTP success as Store reuse."""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT = (
    Path(__file__).resolve().parents[1] / ".github/scripts/atomesh/pd_mooncake_probe.py"
)
SPEC = importlib.util.spec_from_file_location("mooncake_probe", SCRIPT)
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


class Response:
    def __init__(self, body=None, text=""):
        self.body = body
        self.text = text
        self.content = b"json" if body is not None else b""

    def raise_for_status(self):
        pass

    def json(self):
        return self.body


class Session:
    def __init__(self, reuse=False, reset=False):
        self.reuse = reuse
        self.reset_refused = reset
        self.generations = 0

    def post(self, url, **kwargs):
        if url.endswith("/tokenize"):
            return Response({"tokens": [1, 2]})
        if url.endswith("/reset_prefix_cache"):
            return Response({"success": not self.reset_refused})
        self.generations += 1
        return Response({"choices": [{"text": "hello"}]})

    def get(self, url, **kwargs):
        count = self.generations if self.reuse else 0
        return Response(
            text=(
                'vllm:mooncake_store_operation_bytes_total{operation="load_get",status="ok"} '
                f"{count * 4096}\n"
                f"vllm:external_prefix_cache_hits_total {count * 1024}\n"
            )
        )


class ProbeTest(unittest.TestCase):
    def run_probe(self, session, error=None):
        with tempfile.TemporaryDirectory() as output:
            argv = [
                str(SCRIPT),
                "--prefill",
                "http://p",
                "--decode",
                "http://d",
                "--router",
                "http://r",
                "--model",
                "test",
                "--output",
                output,
            ]
            with (
                patch.object(sys, "argv", argv),
                patch.object(probe.requests, "Session", return_value=session),
                patch.object(probe.time, "sleep"),
            ):
                if error:
                    with self.assertRaisesRegex(RuntimeError, error):
                        probe.main()
                else:
                    probe.main()
            return json.loads((Path(output) / "result.json").read_text())

    def test_http_success_without_store_bytes_is_rejected(self):
        result = self.run_probe(Session(), "No CPU Store reuse")
        self.assertFalse(result["store_reuse_observed"])

    def test_positive_store_bytes_and_external_tokens_pass(self):
        result = self.run_probe(Session(reuse=True))
        self.assertTrue(result["store_reuse_observed"])
        self.assertEqual(len(result["cases"]), 2)

    def test_refused_gpu_reset_is_not_reported_as_offload(self):
        result = self.run_probe(Session(reset=True), "reset was refused")
        self.assertEqual(result["cases"], [])

    def test_error_operations_are_not_counted_as_successful_loads(self):
        text = 'vllm:mooncake_store_operation_bytes_total{operation="load_get",status="error"} 4096\n'
        self.assertEqual(
            probe.counter(
                text,
                "vllm:mooncake_store_operation_bytes_total",
                operation="load_get",
                status="ok",
            ),
            0,
        )


if __name__ == "__main__":
    unittest.main()
