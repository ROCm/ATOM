import json
from pathlib import Path
import tempfile
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pd_agentic_profile import Capture, inspect_trace
from pd_recipe_router_ab import profile_command, validate_client_contract


class ParallelTraceCoverageTest(unittest.TestCase):
    def capture(self, decode_axis="dp", duplicate=False):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = SimpleNamespace(
                output=root / "control",
                traces=root / "traces",
                prefill="http://p",
                decode="http://d",
                ranks=4,
                seconds=0.001,
                prefill_rank_axis="pp",
                decode_rank_axis=decode_axis,
            )
            capture = Capture(args)

            def service(path, window):
                if path != "/start_profile":
                    return
                for role in ("prefill", "decode"):
                    folder = args.traces / role
                    folder.mkdir(parents=True)
                    for rank in range(4):
                        name = (
                            f"dp0_pp{rank}_tp0_dcp0_ep0_rank{rank}"
                            if role == "prefill"
                            else f"dp{rank if not duplicate else 0}_pp0_tp0_dcp0_ep{rank}_rank0"
                        )
                        (folder / f"{name}.1.pt.trace.json").write_text(
                            json.dumps(
                                {
                                    "traceEvents": [
                                        {"cat": "kernel", "ts": 1, "dur": 2},
                                        {"cat": "cpu_op"},
                                    ]
                                }
                            )
                        )

            with (
                patch.object(capture, "metrics"),
                patch.object(capture, "both", side_effect=service),
                patch("pd_agentic_profile.time.monotonic", side_effect=[0, 1]),
            ):
                inventory = capture.window(1, SimpleNamespace(poll=lambda: None))
            self.assertEqual(len(inventory), 8)
            self.assertEqual(
                {r["parallel_ranks"]["dp"] for r in inventory if r["role"] == "decode"},
                set(range(4)),
            )

    def test_pp_and_dp_axes_cover_four_workers_with_repeated_global_rank(self):
        self.capture()

    def test_global_rank_cannot_replace_dp_coverage(self):
        with self.assertRaisesRegex(ValueError, "decode: expected 4 unique rank"):
            self.capture(decode_axis="rank")

    def test_duplicate_dp_rank_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "decode: expected 4 unique dp"):
            self.capture(duplicate=True)

    def test_missing_gpu_activity_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            trace = Path(temporary) / "worker_rank0.1.pt.trace.json"
            trace.write_text(json.dumps({"traceEvents": [{"cat": "cpu_op"}]}))
            with self.assertRaisesRegex(ValueError, "Missing GPU kernels"):
                inspect_trace(trace)


class ProfileCommandTest(unittest.TestCase):
    def test_diagnostic_contract_wraps_agentx_and_reaches_both_roles(self):
        path = (
            Path(__file__).resolve().parents[3]
            / ".github/benchmark/pr5090-router-ab/single-node-pp4-dep4-c80-profile.json"
        )
        contract = json.loads(path.read_text())
        validate_client_contract(contract)
        command = profile_command(
            contract,
            ["agentx", "profile"],
            ["http://p", "http://d"],
            Path("/logs/benchmark"),
        )
        self.assertEqual(command[-3:], ["--", "agentx", "profile"])
        self.assertEqual(command[command.index("--decode") + 1], "http://d")
        self.assertEqual(command[command.index("--decode-rank-axis") + 1], "dp")
        self.assertEqual(command[command.index("--prefill-rank-axis") + 1], "pp")
        for role in ("prefill", "decode"):
            cfg = json.loads(contract["roles"][role]["args"]["profiler-config"])
            self.assertEqual(cfg["torch_profiler_dir"], f"/logs/torch-traces/{role}")
            self.assertTrue(cfg["ignore_frontend"])
        for change in ({"diagnostic_only": False}, {"duration": 601}):
            with self.assertRaises(ValueError):
                validate_client_contract(contract | change)

    def test_nonprofiling_command_is_unchanged(self):
        argv = ["agentx", "profile"]
        self.assertIs(profile_command({}, argv, [], Path("/logs")), argv)


class WorkloadLifecycleTest(unittest.TestCase):
    def run_capture(self, failure):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            log = root / "aiperf.log"
            log.write_text("Phase profiling (profiling) started\n")
            args = SimpleNamespace(
                prefill="http://p",
                decode="http://d",
                output=root / "control",
                aiperf_log=log,
                phase_timeout=2,
                finish_timeout=5,
                offsets=[0],
                ranks=1,
                inherit_stdout=True,
                stop_on_capture_error=True,
            )
            capture = Capture(args)
            with patch.object(
                capture,
                "window",
                return_value=[{}, {}],
                side_effect=ValueError("missing DP trace") if failure else None,
            ):
                rc = capture.run(
                    [
                        sys.executable,
                        "-c",
                        (
                            "import time; time.sleep(60)"
                            if failure
                            else "print('AgentX output remains visible')"
                        ),
                    ]
                )
            result = json.loads((args.output / "result.json").read_text())
            self.assertEqual(rc, 1 if failure else 0)
            self.assertEqual(result["complete"], not failure)
            if failure:
                self.assertNotEqual(result["workload_rc"], 0)
                self.assertIn("missing DP trace", result["errors"][0])

    def test_capture_failure_stops_owned_workload(self):
        self.run_capture(True)

    def test_success_preserves_workload_completion(self):
        self.run_capture(False)


if __name__ == "__main__":
    unittest.main()
