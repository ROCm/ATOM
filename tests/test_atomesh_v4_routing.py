# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Verify V4 DPA routing from the real catalog through the shell command builder."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / ".github/scripts/atomesh"


class V4RoutingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = dict(
            os.environ,
            ATOMESH_SLURM_SUBMIT_RUNNER="atomesh-cicd",
            ATOMESH_SLURM_ACCOUNT="",
            ATOMESH_SLURM_PARTITION="",
            ATOMESH_MODEL_ROOT="/models",
            ATOMESH_LOG_ROOT="/logs",
            ATOMESH_NODE_POOL="pit2-p03-g13,pit2-p03-g27",
            ATOMESH_1P1D_NODES="",
            ATOMESH_PD_RANK_MAPPING_POLICY="none",
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "matrix.json"
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS / "pd_matrix.py"),
                    "--suite",
                    "weekly",
                    "--model",
                    "DeepSeek-V4-Pro-0813",
                    "--output",
                    str(output),
                ],
                cwd=ROOT,
                env=cls.env,
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            cls.cells = json.loads(output.read_text())["include"]
        source = (SCRIPTS / "pd_server_atom.sh").read_text()
        functions = []
        for name in ("has_cli_flag", "is_agentic_dpa", "start_router"):
            start = source.index(f"{name}() {{")
            functions.append(source[start : source.index("\n}\n", start) + 3])
        cls.script = "\n".join(functions) + """
dump_launch_info() { :; }
start_logged_process() { shift 2; printf 'ARG:%s\\n' "$@"; }
prefill_args=(--prefill http://prefill:8010)
decode_args=(--decode http://decode:8020)
start_router
"""

    def router_args(self, cell, policy=None, kind=None):
        service = cell["service"]
        result = subprocess.run(
            ["bash", "-eu", "-c", self.script],
            cwd=ROOT,
            env=dict(
                self.env | cell["env"]["common"],
                ROUTER_POLICY=policy or service["router"]["policy"],
                ATOMESH_MESH_BINARY="/test/atomesh",
                ATOM_PD_RANK_MAPPING_POLICY="none",
                BENCHMARK_KIND=kind or cell["benchmark"]["kind"],
                PREFILL_EXTRA_SERVER_ARGS=service["prefill"]["extra_args"],
                DECODE_EXTRA_SERVER_ARGS=service["decode"]["extra_args"],
                ROUTER_PORT="8000",
                PROMETHEUS_PORT="29100",
                RUNTIME_LOG_DIR="/tmp",
            ),
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return [
            line[4:] for line in result.stdout.splitlines() if line.startswith("ARG:")
        ]

    def test_all_v4_concurrencies_launch_cache_aware_on_both_roles(self):
        self.assertEqual(
            {tuple(cell["concurrency"]) for cell in self.cells},
            {(1,), (2,), (16,), (32,), (128,), (192,), (256,)},
        )
        expected = {
            "--policy": "cache_aware",
            "--prefill-policy": "cache_aware",
            "--decode-policy": "cache_aware",
            "--cache-threshold": "0.8",
            "--balance-rel-threshold": "2.0",
            "--eviction-interval": "300",
            "--atom-pd-rank-mapping-policy": "none",
        }
        for cell in self.cells:
            with self.subTest(concurrency=cell["concurrency"]):
                expected["--balance-abs-threshold"] = (
                    "40" if cell["concurrency"] == [256] else "20"
                )
                self.assertEqual(cell["service"]["router"]["policy"], "cache_aware")
                args = self.router_args(cell)
                self.assertIn("--dp-aware", args)
                self.assertNotIn("dp_sticky", args)
                for flag, value in expected.items():
                    self.assertEqual(args.count(flag), 1)
                    self.assertEqual(args[args.index(flag) + 1], value)

    def test_weekly_service_capacity_and_performance_settings(self):
        for cell in self.cells:
            with self.subTest(concurrency=cell["concurrency"]):
                concurrency = cell["concurrency"][0]
                slots = max(32, concurrency * 2)
                self.assertEqual(cell["suite"], "weekly")
                self.assertEqual(cell["num_nodes"], 2)
                self.assertEqual(cell["server_args"]["max_num_seqs"], slots)
                self.assertEqual(cell["server_args"]["decode_max_num_seqs"], slots)
                self.assertEqual(
                    cell["server_args"]["spec_decode_acceptance_length"], 3.01
                )
                self.assertFalse(cell["run_eval"])
                self.assertFalse(cell["eval_only"])
                for role in ("prefill", "decode"):
                    service = cell["service"][role]
                    self.assertEqual(service["workers"], 1)
                    self.assertEqual(service["tp"], 8)
                    self.assertIn("--enable-dp-attention", service["extra_args"])
                    connector = cell["env"][role][f"{role.upper()}_KV_TRANSFER_CONFIG"]
                    self.assertIn('"kv_connector":"mooncake"', connector)
                    self.assertNotIn("lmcache", connector.lower())
                self.assertEqual(
                    json.loads(cell["service"]["decode"]["cudagraph"]),
                    list(range(1, slots // 8 + 1)),
                )

    def test_existing_agentic_dpa_default_remains_sticky(self):
        args = self.router_args(self.cells[0], policy="random")
        self.assertEqual(args[args.index("--policy") + 1], "dp_sticky")
        self.assertIn("--dp-aware", args)

    def test_non_agentic_policy_is_preserved(self):
        args = self.router_args(self.cells[0], policy="random", kind="random")
        self.assertEqual(args[args.index("--policy") + 1], "random")
        self.assertNotIn("dp_sticky", args)


if __name__ == "__main__":
    unittest.main()
