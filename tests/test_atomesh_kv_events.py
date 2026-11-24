# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Verify the kv_events cells through pd_submit.sh's exports and the router and
prefill command builders."""

import json
import os
import shlex
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


MATRIX_ENV = {
    "ATOMESH_SLURM_SUBMIT_RUNNER": "atomesh-cicd",
    "ATOMESH_SLURM_ACCOUNT": "",
    "ATOMESH_SLURM_PARTITION": "",
    "ATOMESH_MODEL_ROOT": "/models",
    "ATOMESH_LOG_ROOT": "/logs",
    "ATOMESH_NODE_POOL": "pit2-p03-g13,pit2-p03-g27",
    "ATOMESH_1P1D_NODES": "",
    "ATOMESH_PD_RANK_MAPPING_POLICY": "none",
}


def router_script():
    return (
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


class KvEventsCellsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = dict(os.environ, **MATRIX_ENV)
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
        cls.router_script = router_script()
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
                for tier, concurrencies in (
                    ("MCSTORE", (56, 72, 96)),
                    ("LMCACHE", (56, 72)),
                )
                for concurrency in concurrencies
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
                self.assertEqual(args[args.index("--fastokens-models") + 1], "*")
                self.assertEqual(
                    args[args.index("--fastokens-segment-cache-mb") + 1], "1024"
                )

    def test_submit_exports_the_switch(self):
        source = (SCRIPTS / "pd_submit.sh").read_text()
        start = source.index("python3 - <<'PY'\n") + len("python3 - <<'PY'\n")
        exports = source[start : source.index("\nPY\n", start)]
        plain = json.loads(json.dumps(self.cells[0]))
        del plain["service"]["router"]["kv_events"]
        for cell, switch in ((self.cells[0], "true"), (plain, "false")):
            with self.subTest(switch=switch):
                result = subprocess.run(
                    [sys.executable, "-c", exports],
                    env=dict(self.env, CELL_JSON=json.dumps(cell)),
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                self.assertIn(
                    f"export ROUTER_KV_EVENTS={switch}", result.stdout.splitlines()
                )

    def test_router_follows_no_events_without_the_switch(self):
        args = self.router_args(self.cells[0])
        self.assertNotIn("--kv-events", args)
        self.assertNotIn("--mooncake-admin-port", args)
        self.assertNotIn("--prepare-queue-timeout-ms", args)
        self.assertNotIn("--fastokens-models", args)

    def test_a_kv_events_router_saves_the_sources_it_follows(self):
        script = shell_functions("save_router_kv_sources") + """
curl() { [[ "${@: -1}" == */kv_events/* ]] || return 22; printf '[{"url":"%s"}]' "${@: -1}"; }
save_router_kv_sources "${OUT}/router-after"
"""
        with tempfile.TemporaryDirectory() as directory:
            for switch in ("true", "false"):
                with self.subTest(switch=switch):
                    out = Path(directory) / switch
                    out.mkdir()
                    result = subprocess.run(
                        ["bash", "-eu", "-c", script],
                        env=dict(
                            self.env,
                            OUT=str(out),
                            ROUTER_PORT="8000",
                            ROUTER_KV_EVENTS=switch,
                        ),
                        check=True,
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                    saved = {p.name: json.loads(p.read_text()) for p in out.iterdir()}
                    if switch == "false":
                        self.assertEqual(saved, {})
                        continue
                    self.assertEqual(
                        saved,
                        {
                            "router-after-kv_events-sources.json": [
                                {"url": "http://127.0.0.1:8000/kv_events/sources"}
                            ]
                        },
                    )
                    self.assertIn("/kv_storage/sources snapshot failed", result.stderr)

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


class DeepSeekV41ShareCacheCellsTest(unittest.TestCase):
    """The recipe's router on the baseline, and the agentic prefill, least_load
    decode and KV events with the Store's hits on the other cell."""

    BASELINE = "ds-v41-flash-8p16d-p2ep2-d4noep-dspark5-agentic-mcstore-c1228"
    run_script = KvEventsCellsTest.run_script
    router_args = KvEventsCellsTest.router_args

    @classmethod
    def setUpClass(cls):
        cls.env = dict(
            os.environ,
            **MATRIX_ENV,
            ATOMESH_2P1D_NODES="pit2-p03-g27,pit2-p03-g48,pit2-p03-g53",
        )
        cls.router_script = router_script()
        cls.cells = []
        with tempfile.TemporaryDirectory() as directory:
            for suite in ("mooncake_store", "kv_events"):
                output = Path(directory) / f"{suite}.json"
                subprocess.run(
                    [
                        sys.executable,
                        str(SCRIPTS / "pd_matrix.py"),
                        "--suite",
                        suite,
                        "--model",
                        "DeepSeek-V4.1-Flash-1017",
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
                cls.cells += json.loads(output.read_text())["include"]

    def submitted_exports(self, cell):
        source = (SCRIPTS / "pd_submit.sh").read_text()
        start = source.index("python3 - <<'PY'\n") + len("python3 - <<'PY'\n")
        result = subprocess.run(
            [sys.executable, "-c", source[start : source.index("\nPY\n", start)]],
            env=dict(self.env, CELL_JSON=json.dumps(cell)),
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return dict(
            shlex.split(line.removeprefix("export "))[0].split("=", 1)
            for line in result.stdout.splitlines()
            if line.startswith("export ")
        )

    def submitted_router_args(self, cell):
        exports = {
            key: value
            for key, value in self.submitted_exports(cell).items()
            if key.startswith("ROUTER_")
        }
        return self.router_args(cell, **exports)

    def test_cells_run_the_recipes_scenario_and_warmup(self):
        for cell in self.cells:
            with self.subTest(cell=cell["name"]):
                exports = self.submitted_exports(cell)
                self.assertEqual(exports["AIPERF_SCENARIO"], "inferencex-agentx-mvp")
                self.assertEqual(exports["AIPERF_WARMUP_REQUESTS_PER_LANE"], "10")

    def test_every_mooncake_process_runs_the_recipes_settings_on_the_lossless_class(
        self,
    ):
        settings = {
            "MC_NUM_QP_PER_EP": "1",
            "MC_IB_TC": "104",
            "MC_WORKERS_PER_CTX": None,
            "MC_MAX_WR": None,
            "MC_IB_PCI_RELAXED_ORDERING": None,
            "MC_ENABLE_DEST_DEVICE_AFFINITY": None,
        }
        for cell in self.cells:
            with self.subTest(cell=cell["name"]):
                exports = self.submitted_exports(cell)
                self.assertEqual(
                    {key: exports.get(f"ATOMESH_ENV_{key}") for key in settings},
                    settings,
                )

    def test_each_decode_node_owner_serves_on_all_of_its_numa_nodes_nics(self):
        for cell in self.cells:
            with self.subTest(cell=cell["name"]):
                exports = self.submitted_exports(cell)
                owners = exports["ATOMESH_PREFILL_ENV_MOONCAKE_STORE_DECODE_OWNERS"]
                self.assertEqual(
                    [entry.split(":")[2] for entry in owners.split(";")],
                    ["rdma0,rdma1,rdma2,rdma3", "rdma4,rdma5,rdma6,rdma7"],
                )

    def test_cells_launch_the_recipe_router_or_the_cache_aware_flow(self):
        self.assertEqual(
            {cell["name"] for cell in self.cells},
            {self.BASELINE, self.BASELINE.replace("-mcstore-", "-mcstore-kvev-")},
        )
        timeouts = {
            "--request-timeout-secs": "1800",
            "--prepare-queue-timeout-ms": "5000",
            "--prepare-timeout-ms": "15000",
        }
        for cell in self.cells:
            with self.subTest(cell=cell["name"]):
                self.assertEqual(cell["num_nodes"], 3)
                self.assertEqual(cell["concurrency"], [1228])
                args = self.submitted_router_args(cell)
                flags = [arg for arg in args if arg.startswith("--")]
                self.assertEqual(len(flags), len(set(flags)), flags)
                self.assertIn("--dp-aware", args)
                value = {flag: args[args.index(flag) + 1] for flag in flags}
                for flag, expected in timeouts.items():
                    self.assertEqual(value[flag], expected)
                if cell["name"] == self.BASELINE:
                    # Above a 1.0 cache hit ratio and with no load imbalance
                    # allowed, cache_aware routes by load alone.
                    self.assertEqual(value["--prefill-policy"], "cache_aware")
                    self.assertEqual(value["--decode-policy"], "cache_aware")
                    self.assertEqual(float(value["--cache-threshold"]), 1.0)
                    self.assertEqual(float(value["--balance-abs-threshold"]), 0)
                    self.assertEqual(float(value["--balance-rel-threshold"]), 1.0)
                    self.assertNotIn("--kv-events", flags)
                    continue
                self.assertEqual(value["--prefill-policy"], "agentic")
                self.assertEqual(value["--decode-policy"], "least_load")
                self.assertIn("--kv-events", flags)
                self.assertEqual(float(value["--kv-storage-weight"]), 0.87)


if __name__ == "__main__":
    unittest.main()
