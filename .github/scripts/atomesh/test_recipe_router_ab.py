import json
from pathlib import Path
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
import tempfile
import unittest
from unittest.mock import patch

from pd_recipe_router_ab import cancellation_probe, request, worker_plan


ROOT = Path(__file__).resolve().parents[3]
CONTRACT = json.loads(
    (ROOT / ".github/benchmark/pr5090-router-ab/contract.json").read_text()
)
ENV = {
    "MORI_SOCKET_IFNAME": "rdma0",
    "NCCL_IB_HCA": "ionic_0,ionic_1",
    "NCCL_IB_GID_INDEX": "1",
    "NCCL_IB_TC": "104",
    "CUDA_VISIBLE_DEVICES": "7",
}


class RecipeWorkerTest(unittest.TestCase):
    def test_preserves_all_source_role_arguments_and_natural_speculation(self):
        for role in ("prefill", "decode"):
            argv, env = worker_plan(
                CONTRACT, role, "/model", 6100, "10.0.0.1", "0,1,2,3", ENV
            )
            self.assertEqual(argv[:3], ["vllm", "serve", "/model"])
            for key, value in CONTRACT["roles"][role]["args"].items():
                flag = "--" + key.replace("_", "-")
                self.assertEqual(argv.count(flag), 1, flag)
                if value is not True:
                    self.assertEqual(argv[argv.index(flag) + 1], str(value))
            spec = json.loads(argv[argv.index("--speculative-config") + 1])
            self.assertEqual(spec["rejection_sample_method"], "block")
            self.assertNotIn("synthetic_acceptance_length", spec)
            self.assertNotIn("CUDA_VISIBLE_DEVICES", env)
            self.assertEqual(env["HIP_VISIBLE_DEVICES"], "0,1,2,3")
            self.assertEqual(env["VLLM_HOST_IP"], "10.0.0.1")
            self.assertEqual(env["MC_GID_INDEX"], "1")
            self.assertEqual(env["HF_HOME"], "/hf_hub_cache")
            self.assertEqual(
                env.get("VLLM_MOONCAKE_BOOTSTRAP_PORT"),
                "7200" if role == "prefill" else None,
            )

    def test_router_arm_does_not_change_worker_plan(self):
        left = worker_plan(
            CONTRACT,
            "decode",
            "/model",
            6101,
            "10.0.0.2",
            "4,5,6,7",
            ENV | {"ATOMESH_EXECUTION_PHASE": "baseline"},
        )
        right = worker_plan(
            CONTRACT,
            "decode",
            "/model",
            6101,
            "10.0.0.2",
            "4,5,6,7",
            ENV | {"ATOMESH_EXECUTION_PHASE": "atomesh"},
        )
        self.assertEqual(left[0], right[0])
        for env in (left[1], right[1]):
            env.pop("ATOMESH_EXECUTION_PHASE")
        self.assertEqual(left[1], right[1])


class StreamTest(unittest.TestCase):
    def test_cancellation_checks_waiting_requests_as_well_as_running(self):
        class Handler(BaseHTTPRequestHandler):
            waiting = 0

            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'data: {"choices":[]}\n\n')

            def do_GET(self):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(
                    f"vllm:num_requests_running 0\nvllm:num_requests_waiting {self.waiting}\n".encode()
                )

            def log_message(self, *args):
                pass

        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        url = f"http://127.0.0.1:{server.server_port}"
        try:
            with (
                tempfile.TemporaryDirectory() as temporary,
                patch("pd_recipe_router_ab.time.sleep"),
            ):
                output = Path(temporary) / "cancel.json"
                cancellation_probe(CONTRACT, url, [url], output)
                self.assertEqual(
                    json.loads(output.read_text())[
                        "running_and_waiting_after_disconnect"
                    ],
                    [0, 0],
                )
                Handler.waiting = 1
                with patch(
                    "pd_recipe_router_ab.time.monotonic", side_effect=[0, 1, 31]
                ):
                    with self.assertRaises(AssertionError):
                        cancellation_probe(CONTRACT, url, [url], output)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_sse_requires_done_and_rejects_error_frames(self):
        class Handler(BaseHTTPRequestHandler):
            payload = b""

            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
                self.send_response(200)
                self.end_headers()
                self.wfile.write(self.payload)

            def log_message(self, *args):
                pass

        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        url = f"http://127.0.0.1:{server.server_port}/v1/chat/completions"
        try:
            Handler.payload = (
                b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\ndata: [DONE]\n\n'
            )
            self.assertEqual(len(request(url, {}, True)), 1)
            Handler.payload = b'data: {"choices":[]}\n\n'
            with self.assertRaisesRegex(AssertionError, "incomplete SSE"):
                request(url, {}, True)
            Handler.payload = b'data: {"error":"upstream failed"}\n\ndata: [DONE]\n\n'
            with self.assertRaises(AssertionError):
                request(url, {}, True)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == "__main__":
    unittest.main()
