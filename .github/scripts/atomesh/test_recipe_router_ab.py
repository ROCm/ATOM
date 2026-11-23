import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pd_recipe_router_ab import (
    benchmark,
    cancellation_probe,
    functional_probe,
    prefill_failure_probe,
    request,
    transferred_bytes,
    verify_kv_transfer,
    worker_plan,
)

ROOT = Path(__file__).resolve().parents[3]
CONTRACT = json.loads(
    (ROOT / ".github/benchmark/pr5090-router-ab/contract.json").read_text()
)
SINGLE_NODE = json.loads(
    (
        ROOT / ".github/benchmark/pr5090-router-ab/single-node-functional.json"
    ).read_text()
)
ENV = {
    "MORI_SOCKET_IFNAME": "rdma0",
    "NCCL_IB_HCA": "ionic_0,ionic_1",
    "NCCL_IB_GID_INDEX": "1",
    "NCCL_IB_TC": "104",
    "CUDA_VISIBLE_DEVICES": "7",
}


class RecipeWorkerTest(unittest.TestCase):
    def test_single_node_workers_use_disjoint_tp4_devices(self):
        for role, gpus in (("prefill", "0,1,2,3"), ("decode", "4,5,6,7")):
            argv, env = worker_plan(
                SINGLE_NODE, role, "/model", 6100, "10.0.0.1", "0,1,2,3", ENV
            )
            self.assertEqual(env["HIP_VISIBLE_DEVICES"], gpus)
            self.assertEqual(argv[argv.index("--tensor-parallel-size") + 1], "4")
            self.assertEqual(
                SINGLE_NODE["roles"][role]["args"], CONTRACT["roles"][role]["args"]
            )
            self.assertEqual(
                SINGLE_NODE["roles"][role]["env"], CONTRACT["roles"][role]["env"]
            )

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


class FunctionalOnlyTest(unittest.TestCase):
    def test_functional_mode_skips_client_install_and_performance(self):
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch("pd_recipe_router_ab.scrape"),
            patch("pd_recipe_router_ab.functional_probe") as probe,
            patch("pd_recipe_router_ab.cancellation_probe") as cancel,
            patch(
                "pd_recipe_router_ab.verify_kv_transfer", return_value={"passed": True}
            ),
            patch("pd_recipe_router_ab.shutil.which", return_value=None),
            patch(
                "pd_recipe_router_ab.prefill_failure_probe",
                return_value=[{"passed": True}],
            ),
            patch("pd_recipe_router_ab.subprocess.run") as run,
            patch("pd_recipe_router_ab.subprocess.check_output") as check_output,
        ):
            args = SimpleNamespace(
                output=str(Path(temporary) / "functional"),
                router="http://router",
                workers=["p", "d"],
            )
            self.assertEqual(benchmark(SINGLE_NODE, args), 0)
            probe.assert_called_once()
            cancel.assert_called_once()
            run.assert_not_called()
            check_output.assert_not_called()
            status = json.loads(
                (Path(temporary) / "functional" / "functional-status.json").read_text()
            )
            self.assertEqual(status["http_probes"], "passed")
            self.assertFalse(status["performance_run"])

    def test_failed_probe_does_not_publish_success(self):
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch("pd_recipe_router_ab.scrape"),
            patch(
                "pd_recipe_router_ab.functional_probe",
                side_effect=AssertionError("bad response"),
            ),
        ):
            args = SimpleNamespace(
                output=temporary, router="http://router", workers=["p", "d"]
            )
            with self.assertRaises(AssertionError):
                benchmark(SINGLE_NODE, args)
            self.assertFalse((Path(temporary) / "functional-status.json").exists())

    def test_single_node_probes_include_context_longer_than_a_kv_block(self):
        calls = []

        def respond(url, payload=None, stream=False):
            if payload is None:
                return {"data": [{"id": SINGLE_NODE["model"]}]}
            calls.append(payload)
            return {"choices": [{}], "usage": {"prompt_tokens": 1024}}

        with (
            tempfile.TemporaryDirectory() as temporary,
            patch("pd_recipe_router_ab.request", side_effect=respond),
        ):
            functional_probe(
                SINGLE_NODE, "http://router", Path(temporary) / "functional.json"
            )
        self.assertEqual(len(calls), 5)
        self.assertGreater(len(calls[-1]["prompt"].split()), 128)


class PrefillFailureProbeTest(unittest.TestCase):
    def test_requires_prefill_error_idle_workers_and_recovery(self):
        class Handler(BaseHTTPRequestHandler):
            error_code = "prefill_server_error"

            def do_POST(self):
                payload = json.loads(
                    self.rfile.read(int(self.headers["Content-Length"]))
                )
                fault = "min_tokens" in payload
                stream = fault and payload["stream"]
                self.send_response(200 if stream or not fault else 400)
                self.end_headers()
                body = (
                    {"error": {"code": self.error_code}}
                    if fault
                    else {"choices": [{"text": "hello"}]}
                )
                text = json.dumps(body)
                self.wfile.write(
                    (f"data: {text}\n\ndata: [DONE]\n\n" if stream else text).encode()
                )

            def do_GET(self):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(
                    b"vllm:num_requests_running 0\nvllm:num_requests_waiting 0\n"
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
                output = Path(temporary) / "failure.json"
                records = prefill_failure_probe(SINGLE_NODE, url, [url], output)
                self.assertEqual(
                    [record["http_status"] for record in records], [400, 200]
                )
                self.assertTrue(
                    all(record["recovery_response"]["choices"] for record in records)
                )
                Handler.error_code = "decode_server_error"
                with self.assertRaises(AssertionError):
                    prefill_failure_probe(SINGLE_NODE, url, [url], output)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


class TransferTest(unittest.TestCase):
    def test_histogram_sum_handles_multiple_tp_labels(self):
        self.assertEqual(
            transferred_bytes(
                'vllm:mooncake_bytes_transferred_sum{engine="0"} 1e3\nvllm:mooncake_bytes_transferred_sum{engine="1"} 2000\n'
            ),
            3000,
        )
        with self.assertRaises(ValueError):
            transferred_bytes("vllm:num_requests_running 0\n")

    def test_transfer_requires_a_positive_delta(self):
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch("pd_recipe_router_ab.time.sleep"),
        ):
            output = Path(temporary)
            (output / "before-probe-worker-0.prom").write_text(
                "vllm:mooncake_bytes_transferred_sum 100\n"
            )

            def capture(workers, directory, tag):
                (directory / f"{tag}-worker-0.prom").write_text(
                    "vllm:mooncake_bytes_transferred_sum 100\n"
                )

            with (
                patch("pd_recipe_router_ab.scrape", side_effect=capture),
                self.assertRaises(AssertionError),
            ):
                verify_kv_transfer(["p", "d"], output)
            self.assertFalse(
                json.loads((output / "kv-transfer.json").read_text())["passed"]
            )
            (output / "after-transfer-worker-0.prom").write_text(
                "vllm:mooncake_bytes_transferred_sum 300\n"
            )
            with patch("pd_recipe_router_ab.scrape"):
                self.assertEqual(
                    verify_kv_transfer(["p", "d"], output)["transferred_bytes_delta"],
                    200,
                )


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
                with (
                    patch("pd_recipe_router_ab.time.monotonic", side_effect=[0, 1, 31]),
                    self.assertRaises(AssertionError),
                ):
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
