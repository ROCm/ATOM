# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU-only tests for ATOMesh dashboard result processing."""

import importlib.util
import json
import sys
import tempfile
import unittest
import urllib.parse
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[1] / ".github/scripts/atomesh/process_result.py"
)
# The script imports its siblings by bare name, as it runs from its own directory.
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("atomesh_process_result", SCRIPT)
process_result = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(process_result)

# The launcher writes one worker per role next to the results even when a
# single aggregated server serves both, which is what makes the name or the
# explicit flag the only honest source for the cell's size.
DOCKER_ENV = """PREFILL_WORKERS=1
DECODE_WORKERS=1
PREFILL_TP=8
DECODE_TP=8
PREFILL_DCP_SIZE=8
DECODE_DCP_SIZE=8
SLURM_SUBMIT_RUNNER=atomesh-cicd
"""

RESULT = {
    "benchmark_kind": "agentic",
    "total_token_throughput": 102000.0,
    "input_throughput": 100000.0,
    "output_throughput": 2000.0,
    "median_tpot_ms": 28.0,
}


def env_payload(topology, **overrides):
    return {
        "topology": topology,
        "display_topology": topology.upper() + "-TP8-DCP8",
        "prefill_workers": "1",
        "decode_workers": "1",
        "prefill_tp": "8",
        "decode_tp": "8",
        **overrides,
    }


def write_cell(root, topology, disaggregated):
    results = root / topology / "benchmark_results"
    results.mkdir(parents=True)
    (root / topology / "docker.env").write_text(DOCKER_ENV, encoding="utf-8")
    name = f"pd-atom-Kimi-K3-MXFP4-{topology}-isl262144-osl1024-conc64-0.8.json"
    payload = {
        **RESULT,
        "topology": topology,
        "display_topology": topology.upper() + "-TP8-DCP8",
        "disaggregated": disaggregated,
    }
    (results / name).write_text(json.dumps(payload), encoding="utf-8")
    return results / name


def perf_points(entries):
    points = {}
    for entry in entries:
        _, _, encoded = entry.get("extra", "").partition("perf_point=")
        if encoded:
            point = json.loads(urllib.parse.unquote(encoded))
            points[point["config_label"].split("_")[3]] = point
    return points


class AggregatedResourcesTest(unittest.TestCase):
    """An aggregated cell runs both phases on one server's GPUs."""

    def test_aggregated_topology_is_not_billed_as_two_services(self):
        resources = process_result.topology_resources(env_payload("agg"), {})
        self.assertFalse(resources["disaggregated"])
        self.assertEqual(resources["total_gpu"], 8)
        self.assertIsNone(resources["num_prefill_gpu"])
        self.assertIsNone(resources["num_decode_gpu"])
        self.assertIsNone(resources["prefill_workers"])
        self.assertIsNone(resources["decode_workers"])

    def test_split_topologies_still_sum_both_roles(self):
        for topology in ("1p1d", "1p1d_dpa", "2p1d"):
            with self.subTest(topology=topology):
                resources = process_result.topology_resources(env_payload(topology), {})
                self.assertTrue(resources["disaggregated"])
                self.assertEqual(resources["total_gpu"], 16)

    def test_launcher_flag_outranks_the_topology_name(self):
        for declared, total_gpu in ((True, 16), (False, 8), ("false", 8), ("1", 16)):
            with self.subTest(disaggregated=declared):
                resources = process_result.topology_resources(
                    env_payload("1p1d", disaggregated=declared), {}
                )
                self.assertEqual(resources["total_gpu"], total_gpu)

    def test_per_gpu_throughput_compares_aggregated_against_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = [
                write_cell(root, "agg", False),
                write_cell(root, "1p1d", True),
            ]
            entries, rows = process_result.collect_dashboard_entries(
                paths, None, {}, None
            )
        points = perf_points(entries)
        self.assertEqual(points["agg"]["total_gpu"], 8)
        self.assertEqual(points["1p1d"]["total_gpu"], 16)
        self.assertEqual(points["agg"]["tput_per_gpu"], 12750.0)
        self.assertEqual(points["1p1d"]["tput_per_gpu"], 6375.0)
        self.assertEqual(points["agg"]["output_tput_per_gpu"], 250.0)
        self.assertEqual(points["1p1d"]["output_tput_per_gpu"], 125.0)
        # The summary reads the per-GPU columns off the enriched rows.
        for row in rows:
            self.assertIn("tput_per_gpu", row)

    def test_eval_scores_attach_to_an_aggregated_run(self):
        for tag, topology in (
            ("20260922_gsm8k_agg_c64", "agg"),
            ("20260922_gsm8k_1p1d_c64", "1p1d"),
            ("20260922_swebench_lite_agg_c8", "agg"),
        ):
            with self.subTest(tag=tag):
                path = Path("/runs") / tag / "results.json"
                self.assertEqual(process_result.eval_topology(path), topology)
                self.assertEqual(process_result.topology_key(topology), topology)



class NonFiniteBenchmarkMetricsTest(unittest.TestCase):
    """Missing/unmeasured benchmark metrics must not become chart numbers."""

    def test_nonfinite_number_is_missing_and_valid_fallback_survives(self):
        for invalid in (float("nan"), float("inf"), -float("inf"), "NaN", "Infinity"):
            with self.subTest(invalid=str(invalid)):
                self.assertIsNone(process_result.number(invalid))
                self.assertEqual(process_result.number(invalid, "17.25"), 17.25)
                self.assertIsNone(process_result.round_or_none(invalid))
                self.assertIsNone(process_result.int_value(invalid))
                self.assertEqual(process_result.fmt(invalid), "--")
                self.assertIsNone(
                    process_result.metric_entry("TTFT", "ms", invalid, None)
                )
        # A real measured zero must not be treated as missing.
        self.assertEqual(process_result.number(0, 17.25), 0.0)
        self.assertEqual(
            process_result.metric_entry("TTFT", "ms", 0, None)["value"], 0.0
        )

    def test_all_nonfinite_throughput_has_no_dashboard_point(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = write_cell(root, "agg", False)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload.update(
                total_token_throughput="NaN",
                input_throughput="Infinity",
                output_throughput="-Infinity",
                mean_ttft_ms="NaN",
                median_tpot_ms="Infinity",
            )
            path.write_text(json.dumps(payload), encoding="utf-8")
            entries, rows = process_result.collect_dashboard_entries(
                [path], None, {}, None
            )
            self.assertEqual(entries, [])
            self.assertIsNone(rows[0]["tput_per_gpu"])
            summary = root / "summary.md"
            process_result.write_summary(rows, summary)
            self.assertNotIn("nan", summary.read_text(encoding="utf-8").lower())
            self.assertNotIn("inf", summary.read_text(encoding="utf-8").lower())

    def test_nonfinite_ttft_omitted_when_throughput_is_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = write_cell(root, "agg", False)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["mean_ttft_ms"] = float("nan")
            payload["total_token_throughput"] = "NaN"
            payload["input_throughput"] = float("inf")
            path.write_text(json.dumps(payload), encoding="utf-8")
            entries, _ = process_result.collect_dashboard_entries(
                [path], None, {}, None
            )
            point = perf_points(entries)["agg"]
            self.assertNotIn("ttft_ms", point)
            self.assertNotIn("total_tput", point)
            self.assertEqual(point["output_tput_per_gpu"], 250.0)
            # Strict JSON prohibits non-finite chart values.
            json.dumps(entries, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
