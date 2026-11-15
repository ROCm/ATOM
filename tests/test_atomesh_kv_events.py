# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Verify the kv_events cells through the router and prefill command builders."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / ".github/scripts/atomesh"


def shell_functions(*names):
    source = (SCRIPTS / "pd_server_atom.sh").read_text()
    functions = []
    for name in names:
        start = source.index(f"{name}() {{")
        functions.append(source[start : source.index("\n}\n", start) + 3])
    return "\n".join(functions)


class KvEventsCellsTest(unittest.TestCase):
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
                    "kv_events",
                    "--model",
                    "GLM-5.2-MXFP4",
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
        cls.router_script = (
            shell_functions(
                "has_cli_flag",
                "is_agentic_dpa",
                "is_agentic_multi_prefill",
                "start_router",
            )
            + """
dump_launch_info() { :; }
start_logged_process() { shift 2; printf 'ARG:%s\\n' "$@"; }
prefill_args=(--prefill http://prefill:8010)
decode_args=(--decode http://decode:8020)
start_router
"""
        )
        cls.prefill_script = shell_functions("start_prefill") + """
apply_role_env() { :; }
reset_lmcache_disk() { :; }
start_mooncake_store() { :; }
mooncake_store_running() { false; }
build_server_cache_env() { :; }
kv_transfer_config() { echo '{}'; }
build_torch_profiler_args() { :; }
dump_launch_info() { :; }
start_logged_process() { shift 2; printf 'ARG:%s\\n' "$@"; }
ATOM_SOURCE_ENV=()
server_common=()
prefill_parallel=()
prefill_cudagraph_args=()
start_prefill prefill-rank-0 "${SERVER_PORT}"
"""

    def run_script(self, script, **env):
        result = subprocess.run(
            ["bash", "-eu", "-c", script],
            cwd=ROOT,
            env=dict(self.env, **env),
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return [
            line[4:] for line in result.stdout.splitlines() if line.startswith("ARG:")
        ]

    def router_args(self, cell, **env):
        service = cell["service"]
        return self.run_script(
            self.router_script,
            **cell["env"]["common"]
            | {
                "ROUTER_POLICY": service["router"]["policy"],
                "ATOMESH_MESH_BINARY": "/test/atomesh",
                "ATOM_PD_RANK_MAPPING_POLICY": "none",
                "BENCHMARK_KIND": cell["benchmark"]["kind"],
                "PREFILL_EXTRA_SERVER_ARGS": service["prefill"]["extra_args"],
                "DECODE_EXTRA_SERVER_ARGS": service["decode"]["extra_args"],
                "xP": str(service["prefill"]["workers"]),
                "ROUTER_PORT": "8000",
                "PROMETHEUS_PORT": "29100",
                "RUNTIME_LOG_DIR": "/tmp",
                "ATOMESH_MOONCAKE_METRICS_PORT": "26090",
            }
            | env,
        )

    def test_cells_route_cache_aware_by_the_prefill_events(self):
        self.assertEqual(
            {
                (cell["display_topology"], tuple(cell["concurrency"]))
                for cell in self.cells
            },
            {
                (f"1P1D-CPP4-DCP4-TP1-TP4-{tier}-KVEV", (concurrency,))
                for tier in ("MCSTORE", "LMCACHE")
                for concurrency in (56, 72, 96)
            },
        )
        for cell in self.cells:
            with self.subTest(cell=cell["name"]):
                self.assertTrue(cell["service"]["router"]["kv_events"])
                args = self.router_args(cell, ROUTER_KV_EVENTS="true")
                self.assertEqual(args[args.index("--policy") + 1], "cache_aware")
                self.assertEqual(args.count("--kv-events"), 1)
                self.assertEqual(args[args.index("--mooncake-admin-port") + 1], "26090")
                self.assertEqual(
                    args[args.index("--prepare-queue-timeout-ms") + 1], "5000"
                )
                self.assertEqual(
                    args[args.index("--prepare-prefix-cache-mb") + 1], "4096"
                )

    def test_router_follows_no_events_without_the_switch(self):
        args = self.router_args(self.cells[0])
        self.assertNotIn("--kv-events", args)
        self.assertNotIn("--mooncake-admin-port", args)
        self.assertNotIn("--prepare-queue-timeout-ms", args)

    def test_store_cells_keep_the_checkpoint_cached_while_workers_load(self):
        script = (
            shell_functions("mooncake_setting", "start_mooncake_page_cache_dropper")
            + """
setsid() { printf 'ARG:%s\\n' "$@"; }
mooncake_page_cache_dropper_running() { true; }
ATOMESH_SCRIPT_DIR=/scripts
MODEL_PATH=/models/m
start_mooncake_page_cache_dropper
"""
        )
        self.assertEqual(
            self.run_script(script),
            [
                "python3",
                "/scripts/drop_page_cache.py",
                "--every",
                "45",
                "--for",
                "1800",
                "/models/m",
            ],
        )
        stores = [
            cell
            for cell in self.cells
            if cell["env"]["prefill"].get("MOONCAKE_STORE") == "1"
        ]
        self.assertEqual(len(stores), 3)
        for cell in stores:
            with self.subTest(cell=cell["name"]):
                drop_for = cell["env"]["prefill"]["MOONCAKE_STORE_PAGE_CACHE_DROP_FOR"]
                self.assertEqual(
                    self.run_script(
                        script,
                        ATOMESH_PREFILL_ENV_MOONCAKE_STORE_PAGE_CACHE_DROP_FOR=drop_for,
                    ),
                    [],
                )

    def test_each_prefill_worker_of_a_node_publishes_on_its_own_ports(self):
        env = {
            "PREFILL_PORT": "8010",
            "ATOMESH_KV_EVENTS_PORT": "5557",
            "HANDSHAKE_PORT": "6301",
            "PREFILL_DP_MASTER_PORT": "29500",
            "PREFILL_DP_BASE_PORT": "29600",
            "USE_EXPLICIT_DP_PORTS": "0",
            "PREFILL_KV_TRANSFER_CONFIG": "{}",
            "MAX_NUM_SEQS": "512",
            "PREFILL_SERVER_ARGS": "",
            "RUNTIME_LOG_DIR": "/tmp",
            "NODE_RANK": "0",
            "host_name": "node",
            "host_ip": "10.0.0.1",
            "HIP_VISIBLE_DEVICES": "0",
        }
        self.assertFalse(
            [
                arg
                for arg in self.run_script(
                    self.prefill_script, SERVER_PORT="8010", **env
                )
                if arg.startswith("ATOM_KV_EVENTS")
            ]
        )
        args = self.run_script(
            self.prefill_script, SERVER_PORT="8011", ROUTER_KV_EVENTS="true", **env
        )
        self.assertIn("ATOM_KV_EVENTS_ENABLE=1", args)
        self.assertIn("ATOM_KV_EVENTS_ENDPOINT=tcp://*:5589", args)
        self.assertIn("ATOM_KV_EVENTS_REPLAY_ENDPOINT=tcp://*:5605", args)


if __name__ == "__main__":
    unittest.main()
