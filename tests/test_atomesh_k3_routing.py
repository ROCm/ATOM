# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Verify Kimi-K3 2P1D prefill routing from the real catalog through the shell
command builder."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / ".github/scripts/atomesh"
CACHE_AWARE_CASE = "kimi-k3-mxfp4-2p1d-tp8-dcp8-dspark3-agentic-lmcache-1m-c72"
STICKY_CASE = "kimi-k3-mxfp4-2p1d-tp8-dcp8-agentic-lmcache-1m-c72"


class K3MultiPrefillRoutingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = dict(
            os.environ,
            ATOMESH_SLURM_SUBMIT_RUNNER="atomesh-cicd",
            ATOMESH_SLURM_ACCOUNT="",
            ATOMESH_SLURM_PARTITION="",
            ATOMESH_MODEL_ROOT="/models",
            ATOMESH_LOG_ROOT="/logs",
            ATOMESH_NODE_POOL="",
            ATOMESH_1P1D_NODES="",
            ATOMESH_2P1D_NODES="",
            ATOMESH_PD_RANK_MAPPING_POLICY="none",
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "matrix.json"
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS / "pd_matrix.py"),
                    "--suite",
                    "nightly",
                    "--case",
                    f"{CACHE_AWARE_CASE},{STICKY_CASE}",
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
            cls.cells = {
                cell["name"]: cell for cell in json.loads(output.read_text())["include"]
            }
        source = (SCRIPTS / "pd_server_atom.sh").read_text()
        functions = []
        for name in (
            "has_cli_flag",
            "is_agentic_dpa",
            "is_agentic_multi_prefill",
            "start_router",
        ):
            start = source.index(f"{name}() {{")
            functions.append(source[start : source.index("\n}\n", start) + 3])
        cls.script = "\n".join(functions) + """
dump_launch_info() { :; }
start_logged_process() { shift 2; printf 'ARG:%s\\n' "$@"; }
prefill_args=(--prefill http://prefill-0:8010 --prefill http://prefill-1:8010)
decode_args=(--decode http://decode:8020)
start_router
"""

    def router_args(self, cell):
        service = cell["service"]
        result = subprocess.run(
            ["bash", "-eu", "-c", self.script],
            cwd=ROOT,
            env=dict(
                self.env,
                xP=str(service["prefill"]["workers"]),
                ROUTER_POLICY=service["router"].get("policy", "random"),
                ROUTER_BALANCE_ABS_THRESHOLD="",
                ATOMESH_MESH_BINARY="/test/atomesh",
                ATOM_PD_RANK_MAPPING_POLICY="none",
                BENCHMARK_KIND=cell["benchmark"]["kind"],
                PREFILL_EXTRA_SERVER_ARGS=service["prefill"].get("extra_args", ""),
                DECODE_EXTRA_SERVER_ARGS=service["decode"].get("extra_args", ""),
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

    def test_explicit_cache_aware_routes_prefill_by_prefix(self):
        cell = self.cells[CACHE_AWARE_CASE]
        self.assertEqual(cell["service"]["prefill"]["workers"], 2)
        args = self.router_args(cell)
        self.assertNotIn("dp_sticky", args)
        self.assertNotIn("--dp-aware", args)
        expected = {
            "--policy": "cache_aware",
            "--prefill-policy": "cache_aware",
            "--decode-policy": "cache_aware",
            "--cache-threshold": "0.8",
            "--balance-abs-threshold": "20",
            "--balance-rel-threshold": "2.0",
        }
        for flag, value in expected.items():
            self.assertEqual(args.count(flag), 1, flag)
            self.assertEqual(args[args.index(flag) + 1], value, flag)

    def test_multi_prefill_default_stays_session_sticky(self):
        cell = self.cells[STICKY_CASE]
        self.assertEqual(cell["service"]["prefill"]["workers"], 2)
        args = self.router_args(cell)
        self.assertEqual(args.count("--prefill-policy"), 1)
        self.assertEqual(args[args.index("--prefill-policy") + 1], "dp_sticky")
        self.assertEqual(args[args.index("--policy") + 1], "random")


if __name__ == "__main__":
    unittest.main()
