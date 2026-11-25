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
start_umbp_pool_server() { :; }
umbp_pool_requested() { false; }
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

    def test_a_cells_store_load_cost_reaches_the_router(self):
        cell = next(cell for cell in self.cells if cell["name"] != self.BASELINE)
        self.assertNotIn("--kv-storage-load-cost", self.submitted_router_args(cell))
        router = dict(cell["service"]["router"], kv_storage_load_cost=65536)
        service = dict(cell["service"], router=router)
        args = self.submitted_router_args(dict(cell, service=service))
        self.assertEqual(args[args.index("--kv-storage-load-cost") + 1], "65536")


class DeepSeekV41UmbpPoolCellsTest(unittest.TestCase):
    """The share-cache cells with a UMBP pool in the Store's place: both routers
    on three nodes, both with two prefill workers on a single node, and the
    cache-aware one's smoke there."""

    STORE = DeepSeekV41ShareCacheCellsTest.BASELINE
    NAMES = {
        "umbp_pool": STORE.replace("-mcstore-", "-umbp-"),
        "umbp_kv_events": STORE.replace("-mcstore-", "-umbp-kvev-"),
        "umbp_smoke": "ds-v41-flash-1p1d-p2ep2-d4noep-dspark5-agentic-umbp-kvev-c16",
        "umbp_2p1d": "ds-v41-flash-2p1d-p2ep2-d4noep-dspark5-agentic-umbp-c64",
        "umbp_2p1d_kv_events": (
            "ds-v41-flash-2p1d-p2ep2-d4noep-dspark5-agentic-umbp-kvev-c64"
        ),
        "umbp_kv_events_wait": STORE.replace("-mcstore-", "-umbp-kvev-wait-"),
        "umbp_2p1d_kv_events_wait": (
            "ds-v41-flash-2p1d-p2ep2-d4noep-dspark5-agentic-umbp-kvev-wait-c64"
        ),
        "umbp_kv_events_opt": STORE.replace("-mcstore-", "-umbp-kvev-opt-"),
        "umbp_2p1d_kv_events_opt": (
            "ds-v41-flash-2p1d-p2ep2-d4noep-dspark5-agentic-umbp-kvev-opt-c64"
        ),
    }
    # The cells whose prefill and decode share their one node.
    ONE_NODE = (
        "umbp_smoke",
        "umbp_2p1d",
        "umbp_2p1d_kv_events",
        "umbp_2p1d_kv_events_wait",
        "umbp_2p1d_kv_events_opt",
    )
    # Each cell whose pass waits for the copy it starts, and the cell it is.
    WAIT = {
        "umbp_kv_events_wait": "umbp_kv_events",
        "umbp_2p1d_kv_events_wait": "umbp_2p1d_kv_events",
    }
    # Each wait cell with larger prefill pools, and the wait cell it is.
    OPT = {
        "umbp_kv_events_opt": "umbp_kv_events_wait",
        "umbp_2p1d_kv_events_opt": "umbp_2p1d_kv_events_wait",
    }
    # Each pool cell and the Store cell it is.
    STORE_SUITE = {"umbp_pool": "mooncake_store", "umbp_kv_events": "kv_events"}
    run_script = KvEventsCellsTest.run_script
    router_args = KvEventsCellsTest.router_args
    submitted_exports = DeepSeekV41ShareCacheCellsTest.submitted_exports
    submitted_router_args = DeepSeekV41ShareCacheCellsTest.submitted_router_args

    @classmethod
    def setUpClass(cls):
        cls.env = (
            dict(os.environ)
            | MATRIX_ENV
            | {
                "ATOMESH_1P1D_NODES": "pit2-p03-g27",
                "ATOMESH_2P1D_NODES": "pit2-p03-g27,pit2-p03-g48,pit2-p03-g53",
            }
        )
        cls.router_script = router_script()
        cls.suites = {}
        with tempfile.TemporaryDirectory() as directory:
            for suite in (*cls.NAMES, *cls.STORE_SUITE.values()):
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
                cls.suites[suite] = json.loads(output.read_text())["include"]
        cls.cells = {suite: cells[0] for suite, cells in cls.suites.items()}

    def prefill_exports(self, suite):
        return {
            key: value
            for key, value in self.submitted_exports(self.cells[suite]).items()
            if key.startswith("ATOMESH_PREFILL_ENV_")
        }

    def test_each_suite_holds_its_one_cell(self):
        self.assertEqual(
            {
                suite: [cell["name"] for cell in self.suites[suite]]
                for suite in self.NAMES
            },
            {suite: [name] for suite, name in self.NAMES.items()},
        )
        self.assertEqual(
            {suite: self.cells[suite]["display_topology"] for suite in self.NAMES},
            {
                "umbp_pool": "8P16D-P2EP2-D4NOEP-DSPARK5-UMBP",
                "umbp_kv_events": "8P16D-P2EP2-D4NOEP-DSPARK5-UMBP-KVEV",
                "umbp_smoke": "1P1D-P2EP2-D4NOEP-DSPARK5-UMBP-KVEV",
                "umbp_2p1d": "2P1D-P2EP2-D4NOEP-DSPARK5-UMBP",
                "umbp_2p1d_kv_events": "2P1D-P2EP2-D4NOEP-DSPARK5-UMBP-KVEV",
                "umbp_kv_events_wait": "8P16D-P2EP2-D4NOEP-DSPARK5-UMBP-KVEV-WAIT",
                "umbp_2p1d_kv_events_wait": (
                    "2P1D-P2EP2-D4NOEP-DSPARK5-UMBP-KVEV-WAIT"
                ),
                "umbp_kv_events_opt": "8P16D-P2EP2-D4NOEP-DSPARK5-UMBP-KVEV-OPT",
                "umbp_2p1d_kv_events_opt": "2P1D-P2EP2-D4NOEP-DSPARK5-UMBP-KVEV-OPT",
            },
        )

    def test_the_three_node_cells_are_the_store_cells_with_the_pool_in_its_place(self):
        for suite, store_suite in self.STORE_SUITE.items():
            with self.subTest(suite=suite):
                cell, store = self.cells[suite], self.cells[store_suite]
                changed = {key for key in store if cell[key] != store[key]}
                self.assertEqual(
                    changed - {"service"},
                    {"id", "suite", "name", "display_topology", "env"},
                )
                # The Store cell's router, less what it counts the Store's hits at.
                router = dict(store["service"]["router"])
                router.pop("kv_storage_weight", None)
                self.assertEqual(cell["service"], dict(store["service"], router=router))
                self.assertEqual(
                    {
                        role
                        for role in store["env"]
                        if cell["env"][role] != store["env"][role]
                    },
                    {"prefill"},
                )
                prefill, store_prefill = cell["env"]["prefill"], store["env"]["prefill"]
                self.assertEqual(
                    {
                        key
                        for key in prefill.keys() | store_prefill.keys()
                        if prefill.get(key) != store_prefill.get(key)
                    },
                    {"MOONCAKE_STORE", "UMBP_POOL", "PREFILL_KV_TRANSFER_CONFIG"},
                )

    def test_the_launcher_reads_the_pool_on_and_the_store_off(self):
        script = (
            shell_functions("mooncake_store_requested", "umbp_pool_requested") + """
if mooncake_store_requested; then echo "ARG:store"; fi
if umbp_pool_requested; then echo "ARG:pool"; fi
"""
        )
        for suite in self.cells:
            with self.subTest(suite=suite):
                self.assertEqual(
                    self.run_script(script, **self.prefill_exports(suite)),
                    ["pool" if suite in self.NAMES else "store"],
                )

    def test_each_prefill_worker_names_a_pool_of_its_own_beside_the_p_d_connector(
        self,
    ):
        source = (SCRIPTS / "pd_server_atom.sh").read_text()
        script = shell_functions("apply_prefixed_env", "kv_transfer_config") + """
apply_prefixed_env "ATOMESH_PREFILL_ENV_" 10.0.0.1
printf 'ARG:%s\\n' "${PREFILL_KV_TRANSFER_CONFIG}"
printf 'ARG:%s\\n' "$(kv_transfer_config kv_producer "${PREFILL_KV_TRANSFER_CONFIG}" 10.0.0.1 "${HANDSHAKE_PORT}")"
"""
        for suite in self.NAMES:
            # packed_nodes exports each worker's handshake port, 6301 + its first GPU.
            for port in (6301, 6303, 6305, 6307):
                with self.subTest(suite=suite, port=port):
                    substituted, passed = self.run_script(
                        script, HANDSHAKE_PORT=str(port), **self.prefill_exports(suite)
                    )
                    # The launcher hands the case's JSON to the worker as it is.
                    self.assertEqual(passed, substituted)
                    config = json.loads(passed)
                    self.assertEqual(config["kv_connector"], "multi")
                    p_d, pool = config["connectors"]
                    self.assertEqual(
                        p_d,
                        {
                            "kv_role": "kv_producer",
                            "kv_connector": "mooncake",
                            "proxy_ip": "10.0.0.1",
                            "handshake_port": port,
                            "protocol": "rdma",
                        },
                    )
                    self.assertEqual(pool["kv_connector"], "umbp_pool")
                    extra = pool["kv_connector_extra_config"]
                    self.assertEqual(extra["namespace"], f"p{port}")
                    waits = suite in self.WAIT or suite in self.OPT
                    self.assertEqual(extra.get("fetch_wait_ms"), 50 if waits else None)
                    # Where the launcher's server listens unless the case moves it.
                    self.assertIn(
                        f'"${{UMBP_POOL_ADDRESS:-{extra["address"]}}}"', source
                    )

    def test_cells_launch_the_recipe_router_or_route_by_the_events(self):
        for suite in ("umbp_pool", "umbp_2p1d"):
            with self.subTest(suite=suite):
                self.assertEqual(
                    self.submitted_router_args(self.cells[suite]),
                    self.submitted_router_args(self.cells["mooncake_store"]),
                )
        # The Store cell's flow without the Store's hits: the router counts what
        # a rank's pool group holds at its own --kv-share-weight.
        store = self.submitted_router_args(self.cells["kv_events"])
        weight = store.index("--kv-storage-weight")
        for suite in (
            "umbp_kv_events",
            "umbp_smoke",
            "umbp_2p1d_kv_events",
            *self.WAIT,
            *self.OPT,
        ):
            with self.subTest(suite=suite):
                args = self.submitted_router_args(self.cells[suite])
                self.assertEqual(args, store[:weight] + store[weight + 2 :])
                self.assertEqual(args[args.index("--prefill-policy") + 1], "agentic")
                self.assertEqual(args[args.index("--decode-policy") + 1], "least_load")
                self.assertIn("--kv-events", args)
                self.assertNotIn("--kv-share-weight", args)

    def test_a_cells_share_weight_reaches_the_router_with_the_events(self):
        for suite, events in (("umbp_kv_events", True), ("umbp_pool", False)):
            with self.subTest(suite=suite):
                cell = self.cells[suite]
                self.assertEqual(
                    self.submitted_exports(cell)["ROUTER_KV_SHARE_WEIGHT"], ""
                )
                router = dict(cell["service"]["router"], kv_share_weight=0.5)
                service = dict(cell["service"], router=router)
                args = self.submitted_router_args(dict(cell, service=service))
                if events:
                    self.assertEqual(args[args.index("--kv-share-weight") + 1], "0.5")
                else:
                    self.assertNotIn("--kv-share-weight", args)
        for suite in self.STORE_SUITE.values():
            with self.subTest(suite=suite):
                self.assertNotIn(
                    "--kv-share-weight", self.submitted_router_args(self.cells[suite])
                )

    def test_the_smoke_is_one_prefill_and_one_decode_worker_on_one_node(self):
        smoke, cell = self.cells["umbp_smoke"], self.cells["umbp_kv_events"]
        self.assertEqual(
            (smoke["topology"], smoke["pd_worker_layout"], smoke["nodes"]),
            ("1p1d_dpa", "single_node", ["pit2-p03-g27"]),
        )
        self.assertEqual(smoke["num_nodes"], 1)
        self.assertEqual(smoke["concurrency"], [16])
        self.assertEqual(
            smoke["benchmark"], dict(cell["benchmark"], benchmark_duration=300)
        )
        self.assertEqual(smoke["service"]["router"], cell["service"]["router"])
        # single_node starts each role on the devices its role env names: as
        # many as its TP, and none of the other role's.
        devices = {"prefill": "0,1", "decode": "4,5,6,7"}
        for role, tp in (("prefill", 2), ("decode", 4)):
            with self.subTest(role=role):
                self.assertEqual(
                    smoke["service"][role], dict(cell["service"][role], workers=1)
                )
                self.assertEqual(smoke["service"][role]["tp"], tp)
                self.assertEqual(len(devices[role].split(",")), tp)
                self.assertEqual(
                    smoke["env"][role],
                    dict(
                        cell["env"][role],
                        MC_DISABLE_HIP="1",
                        HIP_VISIBLE_DEVICES=devices[role],
                    ),
                )

    def test_two_prefill_workers_and_the_decode_pack_one_node(self):
        smoke = self.cells["umbp_smoke"]
        for suite, three_nodes in (
            ("umbp_2p1d", "umbp_pool"),
            ("umbp_2p1d_kv_events", "umbp_kv_events"),
        ):
            with self.subTest(suite=suite):
                packed, cell = self.cells[suite], self.cells[three_nodes]
                self.assertEqual(
                    (packed["topology"], packed["pd_worker_layout"], packed["nodes"]),
                    ("2p1d_dpa", "packed_nodes", ["pit2-p03-g27"]),
                )
                # Two GPUs a prefill worker and the decode's four: one node's eight.
                self.assertEqual(packed["num_nodes"], 1)
                self.assertEqual(packed["concurrency"], [64])
                self.assertEqual(packed["benchmark"], smoke["benchmark"])
                self.assertEqual(
                    packed["service"],
                    dict(
                        cell["service"],
                        prefill=dict(cell["service"]["prefill"], workers=2),
                        decode=dict(cell["service"]["decode"], workers=1),
                    ),
                )
                # packed_nodes gives each worker its devices: the role env names none.
                for role in ("prefill", "decode"):
                    self.assertEqual(
                        packed["env"][role], dict(cell["env"][role], MC_DISABLE_HIP="1")
                    )

    def test_a_wait_cell_is_its_cell_with_the_wait_in_the_pools_settings(self):
        for suite, base_suite in self.WAIT.items():
            with self.subTest(suite=suite):
                cell, base = self.cells[suite], self.cells[base_suite]
                self.assertEqual(
                    {key for key in base if cell[key] != base[key]},
                    {"id", "suite", "name", "display_topology", "env"},
                )
                self.assertEqual(
                    {
                        role
                        for role in base["env"]
                        if cell["env"][role] != base["env"][role]
                    },
                    {"prefill"},
                )
                prefill, base_prefill = cell["env"]["prefill"], base["env"]["prefill"]
                self.assertEqual(
                    {key for key in base_prefill if prefill[key] != base_prefill[key]},
                    {"PREFILL_KV_TRANSFER_CONFIG"},
                )
                # Substituted as the launcher does: the port is a bare number.
                config, base_config = (
                    json.loads(
                        env["PREFILL_KV_TRANSFER_CONFIG"].replace(
                            "${HANDSHAKE_PORT}", "6301"
                        )
                    )
                    for env in (prefill, base_prefill)
                )
                pool = base_config["connectors"][1]["kv_connector_extra_config"]
                pool["fetch_wait_ms"] = 50
                self.assertEqual(config, base_config)

    def test_an_opt_cell_is_its_wait_cell_with_larger_prefill_pools(self):
        for suite, wait_suite in self.OPT.items():
            with self.subTest(suite=suite):
                cell, wait = self.cells[suite], self.cells[wait_suite]
                self.assertEqual(
                    {key for key in wait if cell[key] != wait[key]},
                    {"id", "suite", "name", "display_topology", "service"},
                )
                service, wait_service = cell["service"], wait["service"]
                self.assertEqual(
                    {key for key in wait_service if service[key] != wait_service[key]},
                    {"prefill"},
                )
                prefill, wait_prefill = service["prefill"], wait_service["prefill"]
                self.assertEqual(
                    {key for key in wait_prefill if prefill[key] != wait_prefill[key]},
                    {"extra_args"},
                )
                # After the recipe's 85% among the common arguments: the last wins.
                self.assertEqual(
                    prefill["extra_args"],
                    wait_prefill["extra_args"] + " --gpu-memory-utilization 0.95",
                )

    def test_only_the_cells_of_one_node_take_mooncake_off_hip_ipc(self):
        # A prefill worker's two devices cannot open the memory of the third
        # and fourth of a decode on its node, which HIP IPC would have it do.
        for suite in self.cells:
            with self.subTest(suite=suite):
                self.assertEqual(
                    self.submitted_exports(self.cells[suite]).get(
                        "ATOMESH_ENV_MC_DISABLE_HIP"
                    ),
                    "1" if suite in self.ONE_NODE else None,
                )


if __name__ == "__main__":
    unittest.main()
