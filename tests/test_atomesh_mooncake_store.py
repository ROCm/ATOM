# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Exercise the launcher's Mooncake Store: owner planning, lifecycle and the
prefill workers' mooncake_store offload env.

No Mooncake process starts: a ``python3`` stub on PATH stands in for
numa_exec.py (and so for mooncake_master / mooncake_client), drop_page_cache.py
and numa_memory_budget.py, and a ``curl`` stub for the master's metadata and
metrics servers. The stub master listens on its RPC port, as the readiness
check needs; each owner the stub starts adds its segment to the capacity the
metrics stub reports. The stub also answers the image check's
``python3 -c 'import mooncake.store'``.
"""

import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

SERVER_SCRIPT = (
    Path(__file__).resolve().parents[1] / ".github/scripts/atomesh/pd_server_atom.sh"
)

PYTHON_STUB = textwrap.dedent("""\
    #!/usr/bin/env bash
    if [[ "${1:-}" == "-c" && "${2:-}" == "import mooncake.store" ]]; then
      exit "${STUB_STORE_IMPORT_RC:-0}"
    fi
    echo "$$ HIP=${HIP_VISIBLE_DEVICES-unset} QP=${MC_NUM_QP_PER_EP-unset} THP=${GLIBC_TUNABLES-unset} BIND=${MC_TCP_BIND_ADDRESS-unset} MR=${MC_MAX_MR_SIZE-unset} $*" >> "${STUB_CALLS}"
    case " $* " in
      *" mooncake_master "*)
        touch "${STUB_DIR}/master-up"
        port="$(sed -n 's/.* --rpc_port=\\([0-9]*\\) .*/\\1/p' <<< " $* ")"
        exec "${REAL_PYTHON}" -c 'import socket, sys, time
    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("127.0.0.1", int(sys.argv[1])))
    s.listen(64)
    time.sleep(60)' "${port}"
        ;;
      *" mooncake_client "*)
        port="$(sed -n 's/.* --port=\\([0-9]*\\) .*/\\1/p' <<< " $* ")"
        if [[ "${port}" == "${STUB_DIE_OWNER_PORT:-}" ]]; then
          exit 5
        fi
        bytes="$(sed -n 's/.* --global_segment_size=\\([0-9]*\\) .*/\\1/p' <<< " $* ")"
        master="$(sed -n 's/.* --master_server_address=[^ ]*:\\([0-9]*\\) .*/\\1/p' <<< " $* ")"
        echo "${master} ${bytes}" >> "${STUB_DIR}/segments"
        exec sleep 60
        ;;
      *" --every "*) exec sleep 60 ;;
      *"/numa_memory_budget.py "*) exit "${STUB_BUDGET_RC:-0}" ;;
    esac
    exit 0
    """)

CURL_STUB = textwrap.dedent("""\
    #!/usr/bin/env bash
    url="${*: -1}"
    case "${url}" in
      */metadata*) [[ -e "${STUB_DIR}/master-up" ]] ;;
      */metrics)
        # A pool's metrics count the owners that joined its master.
        port="$(sed -n 's#^http://[^/]*:\\([0-9]*\\)/metrics$#\\1#p' <<< "${url}")"
        master=$(( ATOMESH_MOONCAKE_MASTER_PORT + port - ATOMESH_MOONCAKE_METRICS_PORT ))
        total=0
        if [[ -e "${STUB_DIR}/segments" ]]; then
          while read -r owner_master bytes; do
            if [[ "${owner_master}" == "${master}" ]]; then total=$(( total + bytes )); fi
          done < "${STUB_DIR}/segments"
        fi
        echo "# HELP master_total_capacity_bytes"
        echo "master_total_capacity_bytes ${total}"
        ;;
      *) exit 7 ;;
    esac
    """)

# The image check only looks the Store binaries up on PATH; the launcher runs
# them through numa_exec.py, which the python3 stub stands in for.
STORE_BINARIES = ("mooncake_master", "mooncake_client")

GIB = 1024**3
HOST = "127.0.0.1"
P_D_PRODUCER = (
    '{"kv_connector":"mooncake","kv_role":"kv_producer",'
    '"proxy_ip":"127.0.0.1","handshake_port":6301,"protocol":"rdma"}'
)


def free_port():
    with socket.socket() as sock:
        sock.bind((HOST, 0))
        return sock.getsockname()[1]


def prefixed_lines(stdout, prefix):
    """``<prefix>NAME=value`` lines as a dict, values kept verbatim."""
    return dict(
        line.removeprefix(prefix).split("=", 1)
        for line in stdout.splitlines()
        if line.startswith(prefix)
    )


class MooncakeStoreTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="atomesh-mooncake ")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        (self.root / "logs").mkdir()
        self.bin_dir = self.root / "bin"
        self.bin_dir.mkdir()
        stubs = [("python3", PYTHON_STUB), ("curl", CURL_STUB)]
        stubs += [(name, "#!/usr/bin/env bash\nexit 99\n") for name in STORE_BINARIES]
        for name, text in stubs:
            stub = self.bin_dir / name
            stub.write_text(text)
            stub.chmod(0o755)
        # rdma0-3 on NUMA0, rdma4-7 on NUMA1, every port ACTIVE.
        self.ib = self.root / "infiniband"
        for index in range(8):
            device = self.ib / f"rdma{index}"
            (device / "ports/1").mkdir(parents=True)
            (device / "ports/1/state").write_text("4: ACTIVE\n")
            (device / "device").mkdir()
            (device / "device/numa_node").write_text(f"{index // 4}\n")
        self.master_port = free_port()
        self.calls = self.root / "calls"
        self.calls.touch()
        self.path = f"{self.bin_dir}:{os.environ['PATH']}"
        source = SERVER_SCRIPT.read_text()
        self.functions = (
            source[
                source.index("apply_prefixed_env() {") : source.index('host_ip="$(echo')
            ]
            + source[
                source.index("process_is_running()") : source.index(
                    "write_metadata() {"
                )
            ]
        )
        self.addCleanup(self.kill_stubs)

    def kill_stubs(self):
        for pid in self.stub_pids():
            try:
                os.killpg(pid, 9)
            except (ProcessLookupError, PermissionError):
                pass

    def stub_pids(self):
        return [int(line.split()[0]) for line in self.calls.read_text().splitlines()]

    def store_calls(self, binary):
        return [
            line
            for line in self.calls.read_text().splitlines()
            if f" {binary} " in line
        ]

    def run_shell(self, script, expect_rc=0, **env):
        base = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith(
                ("MC_", "MOONCAKE_", "ATOMESH_", "ATOM_KV_OFFLOAD", "OFFLOAD_")
            )
            and key != "PREFILL_KV_TRANSFER_CONFIG"
        }
        result = subprocess.run(
            [
                "bash",
                "-c",
                "set -euo pipefail\ndump_launch_info() { :; }\n"
                + self.functions
                + script,
            ],
            env={
                **base,
                "PATH": self.path,
                "STUB_CALLS": str(self.calls),
                "STUB_DIR": str(self.root),
                "REAL_PYTHON": sys.executable,
                "HANDSHAKE_PORT": "6301",
                "RUNTIME_LOG_DIR": str(self.root / "logs"),
                "NODE_RANK": "0",
                "host_ip": HOST,
                "ATOMESH_SCRIPT_DIR": "/scripts",
                "ATOMESH_IB_SYSFS_ROOT": str(self.ib),
                "MODEL_PATH": "/models/m",
                "HIP_VISIBLE_DEVICES": "0,1,2,3",
                "ATOMESH_MOONCAKE_MASTER_PORT": str(self.master_port),
                "ATOMESH_MOONCAKE_METADATA_PORT": "50180",
                "ATOMESH_MOONCAKE_METRICS_PORT": "50190",
                "ATOMESH_MOONCAKE_OWNER_PORT": "50152",
                "MOONCAKE_STORE": "1",
                **env,
            },
            cwd=self.root,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        self.assertEqual(result.returncode, expect_rc, result.stdout + result.stderr)
        return result

    def plan(self, **env):
        result = self.run_shell(
            """
plan_mooncake_store_owners
for i in "${!mooncake_owner_plan_numa[@]}"; do
  echo "${mooncake_owner_plan_numa[i]}|${mooncake_owner_plan_gib[i]}|${mooncake_owner_plan_devices[i]}"
done
echo "DEVICES $(mooncake_owner_devices_csv)"
""",
            **env,
        )
        lines = result.stdout.splitlines()
        return [line.split("|") for line in lines[:-1]], lines[-1].split()[1]

    def start_and_stop(self, **env):
        """Starts the Store, prints the prefill workers' env, and stops it."""
        result = self.run_shell(
            """
start_mooncake_store
printf 'ENV %s\\n' "${mooncake_store_prefill_env[@]}"
stop_mooncake_store
""",
            **env,
        )
        return result, prefixed_lines(result.stdout, "ENV ")

    def test_default_plan_is_one_owner_per_numa_on_its_own_numa1_nic_pair(self):
        owners, devices = self.plan()
        self.assertEqual(
            owners, [["0", "768", "rdma4,rdma5"], ["1", "768", "rdma6,rdma7"]]
        )
        self.assertEqual(devices, "rdma4,rdma5,rdma6,rdma7")

    def test_owner_entries_may_name_their_own_nics(self):
        # An entry without its own NICs serves on rdma4-rdma7.
        owners, devices = self.plan(MOONCAKE_STORE_OWNERS="0:64:rdma0,rdma1;1:96")
        self.assertEqual(
            owners,
            [["0", "64", "rdma0,rdma1"], ["1", "96", "rdma4,rdma5,rdma6,rdma7"]],
        )
        self.assertEqual(devices, "rdma0,rdma1,rdma4,rdma5,rdma6,rdma7")

    def test_invalid_owner_specs_are_refused(self):
        for env in (
            {"MOONCAKE_STORE_OWNERS": "0:0"},
            {"MOONCAKE_STORE_OWNERS": "x:10"},
            {"MOONCAKE_STORE_OWNERS": "0:10;;1:10"},
            {"MOONCAKE_STORE_OWNERS": "0:10:rdma 4"},
            {"MOONCAKE_STORE_OWNERS": ";"},
        ):
            with self.subTest(env=env):
                self.run_shell("plan_mooncake_store_owners\n", expect_rc=2, **env)

    def test_start_waits_for_the_capacity_and_stop_ends_everything(self):
        port = self.master_port
        result = self.run_shell(
            """
dump_launch_info() { echo "LAUNCH $*"; }
start_mooncake_store
mooncake_store_running
printf 'ENV %s\\n' "${mooncake_store_prefill_env[@]}"
stop_mooncake_store
! mooncake_store_running
""",
            MOONCAKE_STORE_OWNERS="0:64;1:96",
        )
        # The page cache goes before anything allocates, then the budget of the
        # owners' pins.
        drop, budget = [
            line.split(" ", 1)[1]
            for line in self.calls.read_text().splitlines()
            if "drop_page_cache.py /" in line or "numa_memory_budget.py" in line
        ]
        self.assertTrue(drop.endswith("/scripts/drop_page_cache.py /models/m"), drop)
        self.assertIn(
            "/scripts/numa_memory_budget.py --reserve-gib 128 --compact "
            "--compact-timeout 600 0:64 1:96",
            budget,
        )
        (master,) = self.store_calls("mooncake_master")
        self.assertIn("HIP=-1 QP=1 ", master)
        self.assertIn("/scripts/numa_exec.py 1 mooncake_master", master)
        # The launch info shows each Store process's own environment.
        self.assertIn(
            "LAUNCH MOONCAKE_OWNER HIP_VISIBLE_DEVICES=-1 CUDA_VISIBLE_DEVICES=-1 "
            "MC_NUM_QP_PER_EP=1 GLIBC_TUNABLES=glibc.malloc.hugetlb=1 "
            f"MC_MAX_MR_SIZE=68719476736 MC_TCP_BIND_ADDRESS={HOST} python3 ",
            result.stdout,
        )
        self.assertIn(
            "LAUNCH MOONCAKE_MASTER HIP_VISIBLE_DEVICES=-1 CUDA_VISIBLE_DEVICES=-1 "
            "MC_NUM_QP_PER_EP=1 python3 ",
            result.stdout,
        )
        for flag in (
            f"--rpc_port={port}",
            "--enable_http_metadata_server=true",
            f"--http_metadata_server_host={HOST}",
            "--http_metadata_server_port=50180",
            "--metrics_port=50190",
            "--default_kv_lease_ttl=10000",
            "--eviction_high_watermark_ratio=0.90",
            "--eviction_ratio=0.05",
            "--memory_allocator=offset",
        ):
            self.assertIn(f" {flag} ", f"{master} ")
        owners = sorted(self.store_calls("mooncake_client"))
        self.assertEqual(len(owners), 2)
        for call, numa, owner_port, size in zip(
            sorted(owners, key=lambda c: "--port=50153" in c),
            ("0", "1"),
            ("50152", "50153"),
            (64 * GIB, 96 * GIB),
        ):
            self.assertIn(
                f"HIP=-1 QP=1 THP=glibc.malloc.hugetlb=1 BIND={HOST} "
                "MR=68719476736 ",
                call,
            )
            self.assertIn(f"/scripts/numa_exec.py {numa} mooncake_client ", call)
            for flag in (
                f"--port={owner_port}",
                f"--host={HOST}",
                f"--master_server_address={HOST}:{port}",
                f"--metadata_server=http://{HOST}:50180/metadata",
                "--protocol=rdma",
                "--device_names=rdma4,rdma5,rdma6,rdma7",
                f"--global_segment_size={size}",
                "--local_buffer_size=0",
            ):
                self.assertIn(f" {flag} ", f"{call} ")
        # One shared master; ATOM keeps the stages off the owners' NICs.
        env = prefixed_lines(result.stdout, "ENV ")
        self.assertEqual(
            json.loads(env["ATOM_KV_OFFLOAD_EXTRA_CONFIG"]),
            {
                "mooncake_store.local_hostname": HOST,
                "mooncake_store.protocol": "rdma",
                "mooncake_store.master": f"{HOST}:{port}",
                "mooncake_store.metadata": f"http://{HOST}:50180/metadata",
                "mooncake_store.owner_rdma_devices": "rdma4,rdma5,rdma6,rdma7",
            },
        )
        metrics = (self.root / "logs/mooncake-master-rank-0.metrics").read_text()
        self.assertIn(f"master_total_capacity_bytes {160 * GIB}", metrics)
        self.assertTrue((self.root / "logs/mooncake-owner-rank-0-1.log").exists())
        for pid in self.stub_pids():
            self.assertFalse(self.alive(pid), pid)

    def test_the_servers_start_with_the_store_env_on_their_command_lines(self):
        source = SERVER_SCRIPT.read_text()
        servers = source[
            source.index("start_prefill() {") : source.index("start_router() {")
        ]
        result = self.run_shell(
            servers + """
# Records each server's command line, one argument per line.
start_logged_process() {
  printf '%s\\n' "${@:3}" > "${STUB_DIR}/argv-$(basename "$2" .log)"
}
build_server_cache_env() { local -n out="$3"; out=("HOME=/cache/$1"); }
server_common=(--model /models/m)
prefill_parallel=(-tp 4)
decode_parallel=(-tp 4)
prefill_cudagraph_args=()
decode_cudagraph_args=()
start_prefill prefill-rank-0
start_decode decode-rank-0
echo "EXPORTED=${ATOM_KV_OFFLOAD:-none}"
stop_mooncake_store
""",
            MOONCAKE_STORE_OWNERS="0:8;1:8",
            PREFILL_KV_TRANSFER_CONFIG=P_D_PRODUCER,
            DECODE_KV_TRANSFER_CONFIG="",
            PREFILL_PORT="8010",
            PREFILL_DP_MASTER_PORT="29500",
            PREFILL_DP_BASE_PORT="29600",
            DECODE_PORT="8020",
            DECODE_DP_MASTER_PORT="29700",
            DECODE_DP_BASE_PORT="29800",
            USE_EXPLICIT_DP_PORTS="0",
            MAX_NUM_SEQS="8",
            DECODE_MAX_NUM_SEQS="",
            DECODE_MAX_NUM_BATCHED_TOKENS="",
            BENCH_MAX_CONCURRENCY="8",
            ISL_LIST="8192",
            OSL="1024",
            PREFILL_SERVER_ARGS="",
            DECODE_SERVER_ARGS="",
            host_name="node0",
        )

        def command_line(log_name):
            argv = (self.root / f"argv-{log_name}").read_text().splitlines()
            server = argv.index("python3")
            return argv[:server], argv[server:]

        # The prefill's env assignments precede its command: the offload
        # connector and its Store, beside the P/D connector of the case.
        env, server = command_line("prefill-rank-0")
        self.assertEqual(env[0], "env")
        settings = dict(entry.split("=", 1) for entry in env[1:])
        extra = settings.pop("ATOM_KV_OFFLOAD_EXTRA_CONFIG", None)
        self.assertEqual(
            settings,
            {
                "HOME": "/cache/prefill",
                "ATOM_KV_OFFLOAD": "mooncake_store",
                "OFFLOAD_LOAD_WORKERS": "1",
                "MC_NUM_QP_PER_EP": "1",
                "MC_MAX_MR_SIZE": "1073741824",
                "MC_TCP_BIND_ADDRESS": HOST,
            },
        )
        extra = json.loads(extra)
        self.assertEqual(extra["mooncake_store.master"], f"{HOST}:{self.master_port}")
        self.assertEqual(extra["mooncake_store.protocol"], "rdma")
        self.assertEqual(
            server[:3], ["python3", "-m", "atom.entrypoints.openai_server"]
        )
        self.assertEqual(server[server.index("--kv-transfer-config") + 1], P_D_PRODUCER)
        # The decode gets one QP per endpoint, as the prefill's P/D engine
        # has, and nothing of the offload: it was never exported.
        env, server = command_line("decode-rank-0")
        self.assertEqual(env, ["env", "HOME=/cache/decode", "MC_NUM_QP_PER_EP=1"])
        self.assertFalse([arg for arg in server if "ATOM_KV_OFFLOAD" in arg])
        self.assertIn("EXPORTED=none", result.stdout)

    def test_the_workers_load_workers_and_mr_size_follow_the_case(self):
        _, env = self.start_and_stop(
            MOONCAKE_STORE_OWNERS="0:8;1:8",
            OFFLOAD_LOAD_WORKERS="2",
            MC_MAX_MR_SIZE="4294967296",
        )
        self.assertEqual(env["OFFLOAD_LOAD_WORKERS"], "2")
        self.assertEqual(env["MC_MAX_MR_SIZE"], "4294967296")

    def pools(self, **env):
        result = self.run_shell(
            """
plan_mooncake_store_pools
for i in "${!mooncake_pool_devices[@]}"; do
  echo "${mooncake_pool_devices[i]}|${mooncake_pool_capacity[i]}|$(mooncake_pool_port 50000 "${i}")$(mooncake_pool_suffix "${i}")"
done
""",
            **env,
        )
        return [line.split("|") for line in result.stdout.splitlines()]

    def test_one_shared_pool_counts_the_owners_of_every_node(self):
        self.assertEqual(
            self.pools(
                MOONCAKE_STORE_OWNERS="0:64;1:96",
                MOONCAKE_STORE_DECODE_OWNERS="0:32:rdma0",
            ),
            [["", str(192 * GIB), "50000"]],
        )

    def test_per_nic_pools_number_the_nics_of_both_nodes_alike(self):
        # Each node lists its owners in its own order; a decode node reads the
        # settings from their prefill-prefixed copies.
        self.assertEqual(
            self.pools(
                MOONCAKE_STORE_POOLS="per_nic",
                MOONCAKE_STORE_OWNERS="0:192:rdma1;0:192:rdma0;0:16:rdma10",
                ATOMESH_PREFILL_ENV_MOONCAKE_STORE_DECODE_OWNERS=(
                    "0:240:rdma0;0:240:rdma1"
                ),
            ),
            [
                ["rdma0", str(432 * GIB), "50000-pool0"],
                ["rdma1", str(432 * GIB), "50100-pool1"],
                ["rdma10", str(16 * GIB), "50200-pool2"],
            ],
        )

    def test_invalid_pool_settings_are_refused(self):
        per_nic = {"MOONCAKE_STORE_POOLS": "per_nic"}
        for env, message in (
            (
                {**per_nic, "MOONCAKE_STORE_OWNERS": "0:64:rdma0,rdma1"},
                "needs exactly one NIC; 0:64:rdma0,rdma1 names several",
            ),
            # Without a list of its own an owner takes rdma4-rdma7.
            ({**per_nic, "MOONCAKE_STORE_OWNERS": "0:64"}, "needs exactly one NIC"),
            ({"MOONCAKE_STORE_POOLS": "ring"}, "neither shared nor per_nic"),
            (
                {
                    **per_nic,
                    "MOONCAKE_STORE_OWNERS": "0:8:rdma0;0:8:rdma1",
                    "ATOMESH_MOONCAKE_METRICS_PORT": "65500",
                },
                "port ATOMESH_MOONCAKE_METRICS_PORT + 100 x 1 = 65600",
            ),
        ):
            with self.subTest(env=env):
                result = self.run_shell(
                    "plan_mooncake_store_pools\n", expect_rc=2, **env
                )
                self.assertIn(message, result.stderr)

    def test_decode_owners_need_their_own_nodes_and_one_prefill_node(self):
        decode_owners = {"MOONCAKE_STORE_DECODE_OWNERS": "0:8:rdma0"}
        for env in (
            {"ATOMESH_PD_WORKER_LAYOUT": "single_node", "xP": "1"},
            {"ATOMESH_PD_WORKER_LAYOUT": "multi_node", "xP": "2"},
            # Every decode node would start the owners the pools count once.
            {"ATOMESH_PD_WORKER_LAYOUT": "multi_node", "xP": "1", "yD": "2"},
        ):
            with self.subTest(env=env):
                result = self.run_shell(
                    "check_mooncake_store_settings\n",
                    expect_rc=2,
                    **decode_owners,
                    **env,
                )
                self.assertIn("needs the decode on nodes of its own", result.stderr)
        self.run_shell(
            "check_mooncake_store_settings\n",
            **decode_owners,
            ATOMESH_PD_WORKER_LAYOUT="multi_node",
            xP="1",
            yD="1",
        )

    def test_per_nic_store_gives_each_nic_a_master_and_each_stage_its_pool(self):
        port = self.master_port
        result = self.run_shell(
            """
start_mooncake_store
printf 'ENV %s\\n' "${mooncake_store_prefill_env[@]}"
# As cleanup_processes runs them.
save_mooncake_store_metrics
stop_mooncake_store
""",
            MOONCAKE_STORE_POOLS="per_nic",
            MOONCAKE_STORE_MASTER_NUMA="0",
            MOONCAKE_STORE_OWNERS="0:64:rdma1;0:32:rdma0",
        )
        masters = sorted(self.store_calls("mooncake_master"))
        self.assertEqual(len(masters), 2)
        for pool, master in enumerate(
            sorted(masters, key=lambda c: f"--rpc_port={port + 100}" in c)
        ):
            self.assertIn("/scripts/numa_exec.py 0 mooncake_master", master)
            for flag in (
                f"--rpc_port={port + 100 * pool}",
                f"--http_metadata_server_port={50180 + 100 * pool}",
                f"--metrics_port={50190 + 100 * pool}",
            ):
                self.assertIn(f" {flag} ", f"{master} ")
        owners = self.store_calls("mooncake_client")
        for device, pool, size in (("rdma1", 1, 64 * GIB), ("rdma0", 0, 32 * GIB)):
            (call,) = [c for c in owners if f"--device_names={device} " in f"{c} "]
            for flag in (
                f"--master_server_address={HOST}:{port + 100 * pool}",
                f"--metadata_server=http://{HOST}:{50180 + 100 * pool}/metadata",
                f"--global_segment_size={size}",
            ):
                self.assertIn(f" {flag} ", f"{call} ")
        env = prefixed_lines(result.stdout, "ENV ")
        # The stage on a NIC reads that NIC's pool, whose owners share the NIC
        # by design here: no single master and no NICs to stay off.
        self.assertEqual(
            json.loads(env["ATOM_KV_OFFLOAD_EXTRA_CONFIG"]),
            {
                "mooncake_store.local_hostname": HOST,
                "mooncake_store.protocol": "rdma",
                "mooncake_store.pools": {
                    "rdma0": {
                        "master": f"{HOST}:{port}",
                        "metadata": f"http://{HOST}:50180/metadata",
                    },
                    "rdma1": {
                        "master": f"{HOST}:{port + 100}",
                        "metadata": f"http://{HOST}:50280/metadata",
                    },
                },
            },
        )
        for pool, size in ((0, 32 * GIB), (1, 64 * GIB)):
            metrics = self.root / f"logs/mooncake-master-rank-0-pool{pool}.metrics"
            self.assertIn(f"master_total_capacity_bytes {size}", metrics.read_text())
        for pid in self.stub_pids():
            self.assertFalse(self.alive(pid), pid)

    def test_decode_node_owners_join_the_prefill_nodes_masters(self):
        port = self.master_port
        # The prefill node runs the masters and has mounted its own owners.
        (self.root / "segments").write_text(
            f"{port} {64 * GIB}\n{port + 100} {64 * GIB}\n"
        )
        result = self.run_shell(
            f"""
for p in {port} {port + 100}; do
  setsid python3 /scripts/numa_exec.py 0 mooncake_master --rpc_port=$p \
    >/dev/null 2>&1 &
done
start_mooncake_store_decode_owners
mooncake_store_running
echo "MASTERS=${{mooncake_store_master_count}}"
# As cleanup_processes runs them.
save_mooncake_store_metrics
stop_mooncake_store
""",
            NODE_RANK="1",
            NODE0_ADDR=HOST,
            host_ip="127.0.0.2",
            HIP_VISIBLE_DEVICES="0,1,2,3",
            MOONCAKE_STORE="0",
            ATOMESH_PREFILL_ENV_MOONCAKE_STORE="1",
            ATOMESH_PREFILL_ENV_MOONCAKE_STORE_POOLS="per_nic",
            ATOMESH_PREFILL_ENV_MOONCAKE_STORE_OWNERS="0:64:rdma0;0:64:rdma1",
            ATOMESH_PREFILL_ENV_MOONCAKE_STORE_DECODE_OWNERS="0:96:rdma1;0:96:rdma0",
        )
        self.assertIn("MASTERS=0", result.stdout)
        # No prefill GPU here: the budget counts the owners only.
        (budget,) = [
            line
            for line in self.calls.read_text().splitlines()
            if "numa_memory_budget.py" in line
        ]
        self.assertIn(
            "--reserve-gib 128 --compact --compact-timeout 600 0:96 0:96", budget
        )
        owners = self.store_calls("mooncake_client")
        self.assertEqual(len(owners), 2)
        for device, pool in (("rdma0", 0), ("rdma1", 1)):
            (call,) = [c for c in owners if f"--device_names={device} " in f"{c} "]
            self.assertIn("BIND=127.0.0.2 ", call)
            self.assertIn("/scripts/numa_exec.py 0 mooncake_client ", call)
            for flag in (
                "--host=127.0.0.2",
                f"--master_server_address={HOST}:{port + 100 * pool}",
                f"--metadata_server=http://{HOST}:{50180 + 100 * pool}/metadata",
                f"--global_segment_size={96 * GIB}",
            ):
                self.assertIn(f" {flag} ", f"{call} ")
        # Its owners are stopped; the masters are the prefill node's to stop.
        owner_pids = {int(c.split()[0]) for c in owners}
        for pid in owner_pids:
            self.assertFalse(self.alive(pid), pid)
        self.assertFalse(list((self.root / "logs").glob("*.metrics")))

    def test_decode_owners_give_up_on_masters_that_never_answer(self):
        # A decode node: the Store settings come prefixed.
        result = self.run_shell(
            "start_mooncake_store_decode_owners\n",
            expect_rc=1,
            NODE_RANK="1",
            NODE0_ADDR=HOST,
            host_ip="127.0.0.2",
            MOONCAKE_STORE="0",
            ATOMESH_PREFILL_ENV_MOONCAKE_STORE="1",
            ATOMESH_PREFILL_ENV_MOONCAKE_STORE_OWNERS="0:8:rdma0",
            ATOMESH_PREFILL_ENV_MOONCAKE_STORE_DECODE_OWNERS="0:8:rdma1;0:8:rdma2",
            ATOMESH_PREFILL_ENV_MOONCAKE_STORE_WAIT_TIMEOUT="4",
            ATOMESH_PREFILL_ENV_MOONCAKE_STORE_COMPACT_TIMEOUT="1",
        )
        # The wait, plus the prefill node's compaction of two NUMA nodes.
        self.assertIn("prefill-node masters not ready after 6s", result.stderr)
        self.assertEqual(self.store_calls("mooncake_client"), [])
        # The page cache dropper is stopped with the rest.
        for pid in self.stub_pids():
            self.assertFalse(self.alive(pid), pid)

    def test_the_prefill_node_allows_for_the_decode_nodes_compaction(self):
        # The decode node mounts its owners only once it has compacted its
        # memory: here 5 s into a 2 s wait, inside the 2 x 10 s it allows for
        # that compaction.
        self.run_shell(
            f"""
(
  until [[ "$(wc -l < "${{STUB_DIR}}/segments" 2>/dev/null)" -ge 2 ]]; do
    sleep 0.2
  done
  sleep 5
  echo "{self.master_port} {8 * GIB}" >> "${{STUB_DIR}}/segments"
) >/dev/null 2>&1 &
start_mooncake_store
stop_mooncake_store
""",
            MOONCAKE_STORE_OWNERS="0:8;1:8",
            MOONCAKE_STORE_DECODE_OWNERS="0:8:rdma1",
            MOONCAKE_STORE_WAIT_TIMEOUT="2",
            MOONCAKE_STORE_COMPACT_TIMEOUT="10",
        )
        self.assertEqual(len(self.store_calls("mooncake_client")), 2)
        for pid in self.stub_pids():
            self.assertFalse(self.alive(pid), pid)

    def test_several_prefill_workers_on_a_node_are_refused_before_anything_starts(
        self,
    ):
        # The Store's owners are placed around one prefill worker's GPUs, and
        # a second worker in the same shell would reach it only once the first
        # was up.
        for script, env, workers in (
            ("", {"ATOMESH_PD_WORKER_LAYOUT": "prefill_single_node", "xP": "2"}, 2),
            (
                "prefill_nodes=(0 0)\n",
                {"ATOMESH_PD_WORKER_LAYOUT": "packed_nodes", "xP": "2"},
                2,
            ),
            (
                "prefill_nodes=(0 1 1)\n",
                {"ATOMESH_PD_WORKER_LAYOUT": "packed_nodes", "xP": "3"},
                2,
            ),
        ):
            with self.subTest(env=env, script=script):
                result = self.run_shell(
                    script + "validate_mooncake_store_settings\n", expect_rc=2, **env
                )
                self.assertIn(
                    f"{env['ATOMESH_PD_WORKER_LAYOUT']} starts {workers} prefill "
                    "workers on one node",
                    result.stderr,
                )
        self.assertEqual(self.calls.read_text(), "")
        for script, env in (
            ("", {"ATOMESH_PD_WORKER_LAYOUT": "prefill_single_node", "xP": "1"}),
            ("prefill_nodes=(0 1)\n", {"ATOMESH_PD_WORKER_LAYOUT": "packed_nodes"}),
            ("", {"ATOMESH_PD_WORKER_LAYOUT": "multi_node", "xP": "2"}),
        ):
            with self.subTest(env=env, script=script):
                self.run_shell(script + "check_mooncake_store_settings\n", **env)

    def test_cleanup_saves_the_metrics_before_it_stops_a_worker(self):
        # A decode node stops its owners once the router closes, and the
        # router closes once the prefill worker stops.
        result = self.run_shell(
            """
start_mooncake_store
setsid bash -c 'trap "ls \\"$RUNTIME_LOG_DIR\\" > \\"$STUB_DIR/at-term\\"; exit 0" TERM
  for _ in $(seq 300); do sleep 0.2; done' &
cleanup_processes $!
""",
            MOONCAKE_STORE_OWNERS="0:8;1:8",
        )
        at_term = (self.root / "at-term").read_text().split()
        self.assertIn("mooncake-master-rank-0.metrics", at_term, result.stdout)
        for pid in self.stub_pids():
            self.assertFalse(self.alive(pid), pid)

    def test_an_owner_dying_before_ready_stops_the_store(self):
        self.run_shell(
            "start_mooncake_store\n",
            expect_rc=5,
            MOONCAKE_STORE_OWNERS="0:8;1:8",
            STUB_DIE_OWNER_PORT="50153",
            MOONCAKE_STORE_WAIT_TIMEOUT="30",
        )
        self.assertEqual(len(self.store_calls("mooncake_client")), 2)
        for pid in self.stub_pids():
            self.assertFalse(self.alive(pid), pid)

    def test_an_owner_losing_its_segment_fails_the_stop(self):
        # client_ttl unmounts an owner whose heartbeats lapse, and it keeps
        # running: only the master's capacity at shutdown shows it.
        result = self.run_shell(
            """
start_mooncake_store
head -n 1 "${STUB_DIR}/segments" > "${STUB_DIR}/segments.left"
mv "${STUB_DIR}/segments.left" "${STUB_DIR}/segments"
if stop_mooncake_store; then echo "STOP rc=0"; else echo "STOP rc=$?"; fi
""",
            MOONCAKE_STORE_OWNERS="0:8;1:8",
        )
        self.assertIn("STOP rc=1", result.stdout)
        self.assertIn(
            f"counts {8 * GIB} of its owners' {16 * GIB} bytes", result.stderr
        )
        for pid in self.stub_pids():
            self.assertFalse(self.alive(pid), pid)

    def test_the_master_is_ready_only_once_its_rpc_port_listens(self):
        # Its HTTP metadata server answers first; the owners need the RPC one.
        self.run_shell(
            'plan_mooncake_store_pools\ntouch "${STUB_DIR}/master-up"\n'
            '! mooncake_masters_ready "${host_ip}"\n'
        )

    def test_pins_that_do_not_fit_stop_the_start_before_the_master(self):
        result = self.run_shell(
            "start_mooncake_store\n", expect_rc=2, STUB_BUDGET_RC="2"
        )
        self.assertIn("must fit their NUMA nodes", result.stderr)
        self.assertEqual(self.store_calls("mooncake_master"), [])

    def test_page_cache_drop_covers_the_configured_directories(self):
        self.run_shell(
            "prepare_mooncake_store_memory\n",
            MOONCAKE_STORE_PAGE_CACHE_DROP_DIRS="/share/models:/data/cache",
            # A fragmented node may need longer than the default 600 s.
            MOONCAKE_STORE_COMPACT_TIMEOUT="1800",
        )
        calls = self.calls.read_text()
        self.assertIn("/scripts/drop_page_cache.py /share/models /data/cache\n", calls)
        self.assertIn(
            "--reserve-gib 128 --compact --compact-timeout 1800\n",
            calls,
        )

    def test_owner_devices_must_be_active_and_on_one_numa_node(self):
        (self.ib / "rdma6/ports/1/state").write_text("1: DOWN\n")
        for owners, message in (
            ("0:8:rdma5,rdma6", "rdma6 has no ACTIVE port"),
            ("0:8:rdma3,rdma4", "sit on NUMA nodes 0 1"),
            ("0:8:rdma9", "rdma9 is not in"),
        ):
            with self.subTest(owners=owners):
                result = self.run_shell(
                    "plan_mooncake_store_owners\ncheck_mooncake_owner_devices\n",
                    expect_rc=2,
                    MOONCAKE_STORE_OWNERS=owners,
                )
                self.assertIn(message, result.stderr)
        # NUMA0 memory served by NUMA1's NICs is fine: one node of NICs.
        self.run_shell(
            "plan_mooncake_store_owners\ncheck_mooncake_owner_devices\n",
            MOONCAKE_STORE_OWNERS="0:8:rdma4,rdma5;1:8:rdma7",
        )

    def test_conflicting_settings_are_refused_before_anything_starts(self):
        for env, message in (
            ({"ATOM_KV_OFFLOAD": "lmcache"}, "ATOM_KV_OFFLOAD is set by the launcher"),
            (
                {"ATOM_KV_OFFLOAD_EXTRA_CONFIG": "{}"},
                "ATOM_KV_OFFLOAD_EXTRA_CONFIG is set by the launcher",
            ),
            # A worker runs one offload connector; ATOM would refuse a second
            # one only after the Store had started.
            (
                {
                    "PREFILL_KV_TRANSFER_CONFIG": (
                        '{"kv_connector":"multi","connectors":['
                        f"{P_D_PRODUCER},"
                        '{"kv_connector":"lmcache_offload","kv_role":"offload"}]}'
                    )
                },
                "names the offload connector lmcache_offload",
            ),
            # Aliases match in any case, as ATOM matches them.
            (
                {
                    "PREFILL_KV_TRANSFER_CONFIG": (
                        '{"kv_connector": " LMCacheMPConnector ", "kv_role": "offload"}'
                    )
                },
                "names the offload connector lmcachempconnector",
            ),
            (
                {
                    "PREFILL_KV_TRANSFER_CONFIG": (
                        '{"kv_connector":"mooncake_store","kv_role":"offload"}'
                    )
                },
                "names the offload connector mooncake_store",
            ),
            ({"MC_NUM_QP_PER_EP": "2"}, "MC_NUM_QP_PER_EP=2"),
            ({"MC_MS_AUTO_DISC": "1"}, "MC_MS_AUTO_DISC=1 makes the Store's"),
            # Each would be misread later: by the compaction's timeout, bash
            # arithmetic in the waits, or a Store process.
            *(
                ({f"MOONCAKE_STORE_{name}": value}, f"{name}={value} is not a {kind}")
                for name, value, kind in (
                    ("COMPACT_TIMEOUT", "0", "positive number"),
                    ("COMPACT_TIMEOUT", "-600", "positive number"),
                    ("COMPACT_TIMEOUT", "nan", "positive number"),
                    ("WAIT_TIMEOUT", "20m", "positive integer number"),
                    ("MASTER_NUMA", "-1", "integer number"),
                )
            ),
        ):
            with self.subTest(env=env):
                result = self.run_shell("start_mooncake_store\n", expect_rc=2, **env)
                self.assertIn(message, result.stderr)
        self.assertEqual(self.store_calls("mooncake_master"), [])

    def test_the_p_d_connector_alone_passes(self):
        for config in (
            P_D_PRODUCER,
            '{"kv_connector":"multi","connectors":[' + P_D_PRODUCER + "]}",
        ):
            with self.subTest(config=config):
                self.run_shell(
                    "check_mooncake_store_settings\n",
                    PREFILL_KV_TRANSFER_CONFIG=config,
                )

    def test_validation_reads_each_role_env_before_anything_starts(self):
        # As the launcher's main block runs it: from role-prefixed settings,
        # before the prefill or the Store start.
        ok = {
            "MOONCAKE_STORE": "0",
            "ATOMESH_PREFILL_ENV_MOONCAKE_STORE": "1",
        }
        for env, message in (
            (
                {
                    "ATOMESH_PREFILL_ENV_PREFILL_KV_TRANSFER_CONFIG": (
                        '{"kv_connector":"lmcache_offload","kv_role":"offload"}'
                    )
                },
                "names the offload connector lmcache_offload",
            ),
            ({"ATOMESH_ENV_MC_NUM_QP_PER_EP": "4"}, "MC_NUM_QP_PER_EP=4"),
            # The decode node's owners start from the decode role env.
            ({"ATOMESH_DECODE_ENV_MC_MS_AUTO_DISC": " 1"}, "MC_MS_AUTO_DISC=1"),
            # Only the decode node would plan its pools with it.
            (
                {"ATOMESH_DECODE_ENV_MOONCAKE_STORE_DECODE_OWNERS": "0:8:rdma0"},
                "belong to the prefill role env",
            ),
        ):
            with self.subTest(env=env):
                result = self.run_shell(
                    "validate_mooncake_store_settings\n", expect_rc=2, **ok, **env
                )
                self.assertIn(message, result.stderr)
        # A decode-only switch would give the decode one QP and the prefill none.
        result = self.run_shell(
            "validate_mooncake_store_settings\n",
            expect_rc=2,
            MOONCAKE_STORE="0",
            ATOMESH_DECODE_ENV_MOONCAKE_STORE="1",
        )
        self.assertIn("belong to the prefill role env", result.stderr)
        result = self.run_shell(
            "validate_mooncake_store_settings\n"
            'echo "LEAKED=${MOONCAKE_STORE_OWNERS:-none}"\n',
            **ok,
            ATOMESH_PREFILL_ENV_MOONCAKE_STORE_OWNERS="0:8",
            ATOMESH_PREFILL_ENV_PREFILL_KV_TRANSFER_CONFIG=P_D_PRODUCER,
        )
        self.assertIn("LEAKED=none", result.stdout)
        self.assertEqual(self.calls.read_text(), "")

    def test_an_image_without_the_store_is_refused_before_anything_starts(self):
        ok = {"MOONCAKE_STORE": "0", "ATOMESH_PREFILL_ENV_MOONCAKE_STORE": "1"}
        result = self.run_shell(
            "validate_mooncake_store_settings\n",
            expect_rc=2,
            STUB_STORE_IMPORT_RC="1",
            **ok,
        )
        self.assertIn("cannot import mooncake.store", result.stderr)
        for binary in STORE_BINARIES:
            with self.subTest(binary=binary):
                if shutil.which(binary):
                    self.skipTest(f"this host has a {binary} on PATH")
                stub = self.bin_dir / binary
                text = stub.read_text()
                stub.unlink()
                try:
                    result = self.run_shell(
                        "validate_mooncake_store_settings\n", expect_rc=2, **ok
                    )
                finally:
                    stub.write_text(text)
                    stub.chmod(0o755)
                self.assertIn(f"{binary} is not on PATH", result.stderr)
        # Without the switch, nothing is asked of the image.
        self.run_shell(
            "validate_mooncake_store_settings\n",
            STUB_STORE_IMPORT_RC="1",
            MOONCAKE_STORE="0",
        )
        self.assertEqual(self.calls.read_text(), "")

    def test_disabled_without_the_prefill_switch(self):
        self.run_shell(
            "start_mooncake_store\n! mooncake_store_running\nstop_mooncake_store\n",
            MOONCAKE_STORE="0",
        )
        self.assertEqual(self.calls.read_text(), "")

    def test_decode_on_any_node_reads_the_prefill_switch(self):
        self.run_shell(
            "mooncake_store_requested\nrequire_mooncake_qp_per_endpoint\n",
            MOONCAKE_STORE="0",
            ATOMESH_PREFILL_ENV_MOONCAKE_STORE="1",
            MC_NUM_QP_PER_EP="1",
        )
        self.run_shell(
            "! mooncake_store_requested\n",
            MOONCAKE_STORE="0",
            ATOMESH_PREFILL_ENV_MOONCAKE_STORE="0",
        )
        self.run_shell(
            "require_mooncake_qp_per_endpoint\n", expect_rc=2, MC_NUM_QP_PER_EP="2"
        )

    def test_stop_from_an_exit_trap_stops_live_processes(self):
        # cleanup runs from the EXIT trap after a failure; process_is_running
        # must not take the exit status for its own answer there.
        result = self.run_shell(
            """
trap 'stop_mooncake_store; echo "STOPPED rc=$?"' EXIT
start_mooncake_store
exit 3
""",
            expect_rc=3,
            MOONCAKE_STORE_OWNERS="0:8",
        )
        self.assertIn("STOPPED rc=0", result.stdout)
        for pid in self.stub_pids():
            self.assertFalse(self.alive(pid), pid)

    @staticmethod
    def alive(pid):
        try:
            state = Path(f"/proc/{pid}/stat").read_text().split()[2]
        except (FileNotFoundError, IndexError):
            return False
        return state != "Z"


class LauncherWiringTest(unittest.TestCase):
    """The parts outside the function block: ports and the role env wiring."""

    source = SERVER_SCRIPT.read_text()

    def test_store_ports_shift_with_the_offset_and_bound_it_only_when_used(self):
        for name in (
            "ATOMESH_MOONCAKE_MASTER_PORT",
            "ATOMESH_MOONCAKE_METADATA_PORT",
            "ATOMESH_MOONCAKE_METRICS_PORT",
            "ATOMESH_MOONCAKE_OWNER_PORT",
        ):
            self.assertIn(
                f"{name}=$(({name} + ATOMESH_SERVICE_PORT_OFFSET))", self.source
            )
        ports = self.source[
            self.source.index(
                'PREFILL_PORT="${PREFILL_PORT:-8010}"'
            ) : self.source.index("unset -f validate_shifted_port")
        ]
        base = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith(("MOONCAKE_", "ATOMESH_"))
        }
        # 60051 + 6000 is no port, but only the Store uses it.
        for store, expect_rc in (("0", 0), ("1", 2)):
            result = subprocess.run(
                ["bash", "-c", "set -euo pipefail\n" + ports],
                env={
                    **base,
                    "ATOMESH_SERVICE_PORT_OFFSET": "6000",
                    "ATOMESH_MOONCAKE_MASTER_PORT": "60051",
                    "ATOMESH_PREFILL_ENV_MOONCAKE_STORE": store,
                },
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            self.assertEqual(result.returncode, expect_rc, result.stderr)
        self.assertIn("ATOMESH_MOONCAKE_MASTER_PORT=66051 is outside", result.stderr)

    def test_store_steps_are_ordered_and_the_waits_watch_the_store(self):
        # MooncakeStoreTest runs the server functions and cleanup_processes;
        # these orderings and the waits are outside what it runs.
        decode = self.source.split("start_decode() {")[1].split("\n}\n")[0]
        # The decode node's owners mount before its workers load.
        self.assertLess(
            decode.index("start_mooncake_store_decode_owners"),
            decode.index("start_logged_process"),
        )
        # Every setting is checked before any role starts.
        self.assertLess(
            self.source.index("\nvalidate_mooncake_store_settings\nwrite_metadata\n"),
            self.source.index('  start_prefill "prefill-rank-0"'),
        )
        for waiter in ("wait_http() {", "wait_router_closed() {"):
            body = self.source.split(waiter)[1].split("\n}\n")[0]
            self.assertIn("exit_if_mooncake_store_died", body)

    def test_every_role_branch_traps_exit_before_it_starts_a_server(self):
        # start_prefill and start_decode start the Store's masters and owners
        # before their server: a branch whose trap came after a start would
        # leave them running when the rest of that start fails under set -e.
        dispatch = self.source[
            self.source.index("\nvalidate_mooncake_store_settings\nwrite_metadata\n") :
        ]
        branches = re.split(
            r"^(?:if|elif) .*; then$|^else$", dispatch, flags=re.MULTILINE
        )
        branches = branches[1:]
        self.assertEqual(len(branches), 8)
        for branch in branches:
            starts = re.search(r"^\s*start_(?:prefill|decode)\b", branch, re.MULTILINE)
            trap = re.search(
                r"^\s*trap 'cleanup_processes [^']*' EXIT$", branch, re.MULTILINE
            )
            self.assertIsNotNone(starts, branch[:200])
            self.assertIsNotNone(trap, branch[:200])
            self.assertLess(trap.start(), starts.start(), branch[:200])


if __name__ == "__main__":
    unittest.main()
