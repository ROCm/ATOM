# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU-only tests for ATOMesh workers that span nodes (nodes_per_worker > 1).

Each test runs the relevant slice of pd_server_atom.sh in bash, so what is
checked is the launcher itself: which role a node rank takes, the DP flags it
passes ATOM, the endpoints the router dials, and what each node does.
"""

import os
import subprocess
import unittest
from pathlib import Path

SERVER_SCRIPT = (
    Path(__file__).resolve().parents[1] / ".github/scripts/atomesh/pd_server_atom.sh"
)
SOURCE = SERVER_SCRIPT.read_text()


def script_section(start: str, end: str | None = None) -> str:
    begin = SOURCE.index(start)
    return SOURCE[begin : SOURCE.index(end, begin)] if end else SOURCE[begin:]


LAYOUT = script_section(
    "# A worker may span nodes", 'PREFILL_PORT="${PREFILL_PORT:-8010}"'
)
DP_PORTS = script_section(
    'PREFILL_DP_MASTER_PORT="${PREFILL_DP_MASTER_PORT:-29500}"',
    "# Mooncake Store (MOONCAKE_STORE=1)",
)
EXPLICIT_DP_PORTS = script_section(
    "USE_EXPLICIT_DP_PORTS=0", 'KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"'
)
ENDPOINTS = script_section(
    "IFS=',' read -r -a IP_ARRAY <<< \"${IPADDRS}\"",
    "# AgentX captures every query-token count",
)
HAS_CLI_FLAG = script_section("has_cli_flag() {", "is_agentic_dpa() {")
HOST_PROXY = script_section(
    "# Cross-node MoRI EP goes through the host proxy", "# GPU timing is opt-in"
)
MAIN_FLOW = script_section(
    "# Every branch installs its EXIT trap before its first start"
)

# Stand-ins for everything the main flow starts or waits on.
STUBS = """
start_prefill() { echo "start_prefill $1"; server_pid=1; }
start_decode() { echo "start_decode ${1:-default}"; server_pid=1; }
start_aggregated() { echo "start_aggregated $1"; server_pid=1; }
start_router() { echo "start_router"; router_pid=2; }
wait_http() { echo "wait_http $1"; }
wait_router_closed() { echo "wait_router_closed"; }
run_benchmark_and_eval() { echo "run_benchmark_and_eval"; }
cleanup_processes() { :; }
kill() { :; }
"""

BASE_ENV = {
    "AGGREGATED": "0",
    "ATOMESH_PD_WORKER_LAYOUT": "multi_node",
    "SINGLE_NODE_PD": "0",
    "PACKED_NODES_PD": "0",
    "PREFILL_SINGLE_NODE_PD": "0",
    "DECODE_SINGLE_NODE_PD": "0",
    "xP": "1",
    "yD": "1",
    "PREFILL_PORT": "8010",
    "DECODE_PORT": "8020",
    "ROUTER_PORT": "8000",
    "PREFILL_TP_SIZE": "8",
    "DECODE_TP_SIZE": "8",
    "PREFILL_DCP_SIZE": "1",
    "DECODE_DCP_SIZE": "1",
    "PREFILL_ENABLE_DP": "false",
    "DECODE_ENABLE_DP": "false",
    "PREFILL_SERVER_ARGS": "",
    "DECODE_SERVER_ARGS": "",
    "NODE0_ADDR": "10.0.0.1",
    "WAIT_SERVER_TIMEOUT": "60",
    "WAIT_ROUTER_TIMEOUT": "60",
}
TWO_BY_TWO = {"PREFILL_NODES_PER_WORKER": "2", "DECODE_NODES_PER_WORKER": "2"}


def ips(count: int) -> str:
    return ",".join(f"10.0.0.{index}" for index in range(1, count + 1))


def run_bash(body: str, **env) -> subprocess.CompletedProcess:
    inherited = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("PREFILL_", "DECODE_", "MORI_"))
    }
    return subprocess.run(
        ["bash", "-c", "set -euo pipefail\n" + body],
        env={
            **inherited,
            **BASE_ENV,
            **{key: str(value) for key, value in env.items()},
        },
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


class NodeRoleTest(unittest.TestCase):
    """Prefill workers take the first nodes, decode workers the rest."""

    def roles(self, ranks, **env):
        found = {}
        for rank in ranks:
            result = run_bash(
                LAYOUT + 'echo "${NODE_ROLE} ${NODE_WORKER} ${NODE_IN_WORKER}"',
                NODE_RANK=rank,
                **env,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            found[rank] = result.stdout.splitlines()[-1]
        return found

    def test_two_nodes_per_worker(self):
        self.assertEqual(
            self.roles(range(4), **TWO_BY_TWO),
            {0: "prefill 0 0", 1: "prefill 0 1", 2: "decode 0 0", 3: "decode 0 1"},
        )

    def test_one_node_per_worker_keeps_the_old_mapping(self):
        self.assertEqual(
            self.roles(range(3), xP=2),
            {0: "prefill 0 0", 1: "prefill 1 0", 2: "decode 0 0"},
        )

    def test_several_spanning_prefill_workers(self):
        self.assertEqual(
            self.roles(range(5), xP=2, PREFILL_NODES_PER_WORKER=2),
            {
                0: "prefill 0 0",
                1: "prefill 0 1",
                2: "prefill 1 0",
                3: "prefill 1 1",
                4: "decode 0 0",
            },
        )

    def test_an_aggregated_server_counts_its_nodes_from_zero(self):
        self.assertEqual(
            self.roles(range(2), AGGREGATED=1, DECODE_NODES_PER_WORKER=2),
            {0: "server 0 0", 1: "server 0 1"},
        )

    def test_spanning_needs_the_multi_node_layout(self):
        result = run_bash(
            LAYOUT,
            NODE_RANK=0,
            ATOMESH_PD_WORKER_LAYOUT="single_node",
            DECODE_NODES_PER_WORKER=2,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("needs pd_worker_layout multi_node", result.stderr)


class DpPortTest(unittest.TestCase):
    """Every node of a spanning worker must agree on ports ATOM cannot share."""

    def ports(self, **env):
        result = run_bash(
            DP_PORTS
            + 'echo "${PREFILL_DP_MASTER_PORT} ${PREFILL_DP_BASE_PORT}'
            + ' ${DECODE_DP_MASTER_PORT} ${DECODE_DP_BASE_PORT}"',
            **{"PREFILL_NODES_PER_WORKER": 1, "DECODE_NODES_PER_WORKER": 1, **env},
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return [int(port) for port in result.stdout.split()]

    def test_one_node_roles_keep_their_ports(self):
        self.assertEqual(self.ports(), [29500, 29600, 29700, 29800])

    def test_spanning_roles_move_the_store_off_the_engine_sockets(self):
        ports = self.ports(**TWO_BY_TWO)
        self.assertEqual(ports, [29500, 29900, 29700, 30100])
        for master, base in (ports[:2], ports[2:]):
            # ATOM's TCP engine sockets: master + 100, three per rank, 16 ranks.
            self.assertNotIn(base, range(master + 100, master + 100 + 3 * 16))

    def test_an_explicit_base_port_wins(self):
        self.assertEqual(self.ports(PREFILL_DP_BASE_PORT=31000, **TWO_BY_TWO)[1], 31000)

    def test_spanning_roles_pass_explicit_ports(self):
        for nodes, expected in (("1", "0"), ("2", "1")):
            with self.subTest(nodes_per_worker=nodes):
                result = run_bash(
                    EXPLICIT_DP_PORTS + 'echo "${USE_EXPLICIT_DP_PORTS}"',
                    PREFILL_NODES_PER_WORKER=1,
                    DECODE_NODES_PER_WORKER=nodes,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.strip(), expected)


class EndpointAndDpFlagTest(unittest.TestCase):
    """The router dials coordinators only; every node gets its slice of the group."""

    def launch(self, rank, nodes, **env):
        result = run_bash(
            LAYOUT
            + ENDPOINTS
            + 'echo "router ${prefill_args[*]} ${decode_args[*]}"\n'
            + 'echo "prefill ${prefill_parallel[*]}"\n'
            + 'echo "decode ${decode_parallel[*]}"\n',
            NODE_RANK=rank,
            IPADDRS=ips(nodes),
            **env,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        lines = result.stdout.splitlines()[-3:]
        return {line.split(" ", 1)[0]: line.split(" ", 1)[1] for line in lines}

    def test_two_by_two_reaches_the_router_as_one_prefill_and_one_decode(self):
        for rank in range(4):
            with self.subTest(rank=rank):
                self.assertEqual(
                    self.launch(rank, 4, **TWO_BY_TWO)["router"],
                    "--prefill http://10.0.0.1:8010 --decode http://10.0.0.3:8020",
                )

    def test_two_by_two_dp_flags_per_node(self):
        local = "-tp 8 --decode-context-parallel-size 1"
        for rank, role, dp_rank, coordinator in (
            (0, "prefill", 0, "10.0.0.1"),
            (1, "prefill", 1, "10.0.0.1"),
            (2, "decode", 0, "10.0.0.3"),
            (3, "decode", 1, "10.0.0.3"),
        ):
            with self.subTest(rank=rank):
                launch = self.launch(rank, 4, **TWO_BY_TWO)
                self.assertEqual(
                    launch[role],
                    f"{local} -dp 2 --data-parallel-size-local 1 "
                    f"--data-parallel-rank {dp_rank} "
                    f"--data-parallel-master-ip {coordinator}",
                )
                other = "decode" if role == "prefill" else "prefill"
                self.assertEqual(launch[other], local)

    def test_one_node_workers_are_unchanged(self):
        launch = self.launch(1, 2)
        self.assertEqual(
            launch["router"],
            "--prefill http://10.0.0.1:8010 --decode http://10.0.0.2:8020",
        )
        self.assertNotIn("-dp", launch["prefill"] + launch["decode"])

    def test_a_second_spanning_prefill_worker_coordinates_on_its_first_node(self):
        launch = self.launch(3, 5, xP=2, PREFILL_NODES_PER_WORKER=2)
        self.assertEqual(
            launch["router"],
            "--prefill http://10.0.0.1:8010 --prefill http://10.0.0.3:8010 "
            "--decode http://10.0.0.5:8020",
        )
        self.assertTrue(
            launch["prefill"].endswith(
                "--data-parallel-rank 1 --data-parallel-master-ip 10.0.0.3"
            )
        )

    def test_an_aggregated_server_spans_nodes_from_node_zero(self):
        launch = self.launch(1, 2, AGGREGATED=1, DECODE_NODES_PER_WORKER=2)
        self.assertTrue(
            launch["decode"].endswith(
                "-dp 2 --data-parallel-size-local 1 --data-parallel-rank 1 "
                "--data-parallel-master-ip 10.0.0.1"
            )
        )

    def test_too_few_nodes_fail_before_anything_starts(self):
        result = run_bash(LAYOUT + ENDPOINTS, NODE_RANK=0, IPADDRS=ips(3), **TWO_BY_TWO)
        self.assertEqual(result.returncode, 2)
        self.assertIn("needs 4 nodes, got 3", result.stderr)


class HostProxyDefaultTest(unittest.TestCase):
    """MoRI EP starts on the host proxy unless the cell picks a transport."""

    def proxy(self, **env):
        result = run_bash(
            HAS_CLI_FLAG + HOST_PROXY + 'echo "${MORI_ENABLE_HOST_PROXY:-unset}"', **env
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout.strip()

    def test_expert_parallel_turns_it_on(self):
        self.assertEqual(
            self.proxy(
                DECODE_SERVER_ARGS="--enable-dp-attention --enable-expert-parallel"
            ),
            "1",
        )

    def test_the_cell_setting_wins(self):
        self.assertEqual(
            self.proxy(
                PREFILL_SERVER_ARGS="--enable-expert-parallel", MORI_ENABLE_HOST_PROXY=0
            ),
            "0",
        )

    def test_no_expert_parallel_leaves_it_alone(self):
        self.assertEqual(
            self.proxy(DECODE_SERVER_ARGS="--enable-dp-attention"), "unset"
        )


class MainFlowTest(unittest.TestCase):
    """What each node runs, with every server start and wait stubbed out."""

    def flow(self, rank, nodes, **env):
        result = run_bash(
            LAYOUT + ENDPOINTS + STUBS + MAIN_FLOW,
            NODE_RANK=rank,
            IPADDRS=ips(nodes),
            **env,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return [line for line in result.stdout.splitlines() if not line.startswith("[")]

    def test_two_by_two(self):
        router_up = ["wait_http http://10.0.0.1:8000/health", "wait_router_closed"]
        expected = {
            0: [
                "start_prefill prefill-rank-0",
                "wait_http http://10.0.0.1:8010/health",
                "wait_http http://10.0.0.3:8020/health",
                "start_router",
                "wait_http http://127.0.0.1:8000/v1/models",
                "run_benchmark_and_eval",
            ],
            1: ["start_prefill prefill-rank-1", *router_up],
            2: ["start_decode default", *router_up],
            3: ["start_decode default", *router_up],
        }
        for rank, steps in expected.items():
            with self.subTest(rank=rank):
                self.assertEqual(self.flow(rank, 4, **TWO_BY_TWO), steps)

    def test_aggregated_across_two_nodes(self):
        env = {"AGGREGATED": 1, "DECODE_NODES_PER_WORKER": 2}
        self.assertEqual(
            self.flow(0, 2, **env),
            [
                "start_aggregated server-rank-0",
                "wait_http http://127.0.0.1:8000/health",
                "run_benchmark_and_eval",
            ],
        )
        self.assertEqual(
            self.flow(1, 2, **env),
            [
                "start_aggregated server-rank-1",
                "wait_http http://10.0.0.1:8000/health",
                "wait_router_closed",
            ],
        )

    def test_a_one_node_aggregated_server_idles_the_other_nodes(self):
        result = run_bash(
            LAYOUT + ENDPOINTS + STUBS + MAIN_FLOW,
            NODE_RANK=1,
            IPADDRS=ips(2),
            AGGREGATED=1,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("idle", result.stdout)
        self.assertNotIn("start_aggregated", result.stdout)


if __name__ == "__main__":
    unittest.main()
