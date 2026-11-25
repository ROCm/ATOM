"""Scope guards and first-failure correlation for the timeout diagnostic."""

import copy
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pd_native_stacks import capture_stacks
from pd_timeout_probe import failure_evidence, main, validate_probe


class TimeoutProbeTests(unittest.TestCase):
    def setUp(self):
        self.contract = json.loads(
            (
                Path(__file__).resolve().parents[2]
                / "benchmark/pr5090-router-ab/single-node-pp4-dep4-timeout.json"
            ).read_text()
        )

    def test_original_concurrency_and_deadline(self):
        validate_probe(self.contract)
        bad = copy.deepcopy(self.contract)
        kv = json.loads(bad["roles"]["prefill"]["args"]["kv-transfer-config"])
        kv["kv_connector_extra_config"]["connectors"][0]["kv_connector_extra_config"][
            "num_workers"
        ] = 1
        bad["roles"]["prefill"]["args"]["kv-transfer-config"] = json.dumps(kv)
        with self.assertRaisesRegex(ValueError, "ten"):
            validate_probe(bad)
        for key in ("MC_TRANSFER_TIMEOUT", "MC_SLICE_TIMEOUT"):
            bad = copy.deepcopy(self.contract)
            bad["roles"]["prefill"]["env"][key] = "60"
            with self.assertRaisesRegex(ValueError, "native timeout"):
                validate_probe(bad)

    def test_only_correlate_same_process_and_native_thread(self):
        n = {
            "event": "before_free_uncached_batch",
            "pid": 10,
            "tid": 20,
            "monotonic_ns": 20,
        }
        f = {
            "event": "failed_native_write",
            "pid": 10,
            "tid": 21,
            "native_call_start_monotonic_ns": 10,
            "monotonic_ns": 30,
        }
        self.assertIsNone(failure_evidence([n, f]))
        f["tid"] = 20
        self.assertEqual(failure_evidence([n, f]), {"native": n, "connector": f})
        f["pid"] = 11
        self.assertIsNone(failure_evidence([n, f]))
        f["pid"] = 10
        f["native_call_start_monotonic_ns"] = 21
        self.assertIsNone(failure_evidence([n, f]))

    def test_capture_stops_only_owned_child_and_preserves_failed_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            run = Path(tmp) / "run"
            context = repo / "context"
            context.mkdir(parents=True)
            run.mkdir()
            (repo / ".sa-srt-runtime").mkdir()
            contract = copy.deepcopy(self.contract)
            contract["timeout_probe"]["context"] = "context"
            if contract["timeout_probe"].get("capture_native_stacks"):
                (context / "debugger").mkdir()
                real_context = (
                    Path(__file__).resolve().parents[2]
                    / "benchmark/pr5090-router-ab/diagnostics/mooncake-timeout"
                )
                (context / "debugger/Dockerfile").write_bytes(
                    (real_context / "debugger/Dockerfile").read_bytes()
                )
            (repo / "contract.json").write_text(json.dumps(contract))
            manifest = {
                "base_image": contract["image"],
                "files": {},
                "engine_sha256": "native",
                "instrumented_connector_sha256": "python",
            }
            (context / "manifest.json").write_text(json.dumps(manifest))

            def command(argv, **kwargs):
                if argv[:2] == ["docker", "build"]:
                    Path(argv[argv.index("--iidfile") + 1]).write_text(
                        "sha256:" + "a" * 64
                    )
                return SimpleNamespace(returncode=0)

            def output(argv, **kwargs):
                if "--version" in argv:
                    return "GNU gdb fixture\n"
                return json.dumps(
                    {"engine_sha256": "native", "connector_sha256": "python"}
                )

            class Child:
                returncode = None
                terminated = 0

                def poll(self):
                    return self.returncode

                def terminate(self):
                    self.terminated += 1
                    self.returncode = 143

                def wait(self, **kwargs):
                    return self.returncode

            child = Child()

            def stacks(*args):
                self.assertEqual(child.terminated, 0)
                self.assertIsNone(child.returncode)
                return {"frames_present": True, "sidecars_remaining": []}

            events = [
                {
                    "event": "before_free_uncached_batch",
                    "pid": 1,
                    "tid": 2,
                    "monotonic_ns": 20,
                },
                {
                    "event": "failed_native_write",
                    "pid": 1,
                    "tid": 2,
                    "native_call_start_monotonic_ns": 10,
                    "monotonic_ns": 30,
                },
            ]
            env = {
                "REPO_ROOT": str(repo),
                "RUN_DIR": str(run),
                "ATOMESH_ENV_ATOMESH_PR5090_CONTRACT": "contract.json",
                "SRTCTL_JOB_ID": "owned",
            }
            with (
                patch.dict(os.environ, env),
                patch("pd_timeout_probe.subprocess.run", side_effect=command),
                patch(
                    "pd_timeout_probe.subprocess.check_output",
                    side_effect=output,
                ),
                patch("pd_timeout_probe.subprocess.Popen", return_value=child),
                patch("pd_timeout_probe.records", return_value=events),
                patch("pd_timeout_probe.signal.signal"),
                patch("pd_timeout_probe.capture_stacks", side_effect=stacks) as capture,
            ):
                self.assertEqual(main(), 86)
                if contract["timeout_probe"].get("capture_native_stacks"):
                    capture.assert_called_once()
            self.assertEqual(child.terminated, 1)
            self.assertEqual(child.returncode, 143)
            self.assertEqual(
                json.loads((run / "timeout-probe-result.json").read_text())["status"],
                "captured_failure_stopping",
            )

    def test_debugger_timeout_cleans_sidecar_without_killing_worker(self):
        target, sidecar = "a" * 64, "b" * 64
        selected = [
            {
                "Id": target,
                "Config": {
                    "Image": "worker-image",
                    "Labels": {"srtctl.step": "prefill_0_fixture"},
                },
            }
        ]
        removed = []

        def run(argv, **kwargs):
            if argv[:2] == ["docker", "run"]:
                raise subprocess.TimeoutExpired(argv, 25)
            if argv[:3] == ["docker", "rm", "-f"]:
                removed.append(argv[3])
            return SimpleNamespace(returncode=0)

        with (
            tempfile.TemporaryDirectory() as temp,
            patch(
                "pd_native_stacks.subprocess.check_output",
                side_effect=[target, json.dumps(selected), sidecar, ""],
            ),
            patch("pd_native_stacks.subprocess.run", side_effect=run),
        ):
            result = capture_stacks(
                "owned-job", 123, "worker-image", "debugger-image", Path(temp) / "out"
            )
        self.assertTrue(result["timed_out"])
        self.assertEqual(removed, [sidecar])
        self.assertEqual(result["sidecars_remaining"], [])


if __name__ == "__main__":
    unittest.main()
