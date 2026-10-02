# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Exercise the launcher's Mooncake Store L2: owner planning and lifecycle.

No Mooncake process starts: a ``python3`` stub on PATH stands in for
numa_exec.py (and so for mooncake_master / mooncake_client), drop_page_cache.py
and numa_memory_budget.py, and a ``curl`` stub for the master's metadata and
metrics servers. The stub master listens on its RPC port, as the readiness
check needs; each owner the stub starts adds its segment to the capacity the
metrics stub reports.
"""

import json
import os
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
    echo "$$ HIP=${HIP_VISIBLE_DEVICES-unset} QP=${MC_NUM_QP_PER_EP-unset} THP=${GLIBC_TUNABLES-unset} BIND=${MC_TCP_BIND_ADDRESS-unset} MR=${MC_MAX_MR_SIZE-unset} $*" >> "${STUB_CALLS}"
    case " $* " in
      *" mooncake_master "*)
        touch "${STUB_DIR}/master-up"
        [[ -n "${STUB_MASTER_NO_RPC:-}" ]] && exec sleep 60
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

GIB = 1024**3
HOST = "127.0.0.1"


def free_port():
    with socket.socket() as sock:
        sock.bind((HOST, 0))
        return sock.getsockname()[1]


class MooncakeStoreTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="atomesh-mooncake ")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        (self.root / "logs").mkdir()
        bin_dir = self.root / "bin"
        bin_dir.mkdir()
        for name, text in (("python3", PYTHON_STUB), ("curl", CURL_STUB)):
            stub = bin_dir / name
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
        self.path = f"{bin_dir}:{os.environ['PATH']}"
        source = SERVER_SCRIPT.read_text()
        self.functions = (
            source[
                source.index("apply_prefixed_env() {") : source.index('host_ip="$(echo')
            ]
            + source[
                source.index("process_is_running()") : source.index(
                    "cleanup_processes()"
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
            if not key.startswith(("LMCACHE_", "MC_", "MOONCAKE_", "ATOMESH_"))
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
                "LMCACHE_MOONCAKE_L2": "1",
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

    def test_default_plan_is_one_owner_per_numa_on_its_own_numa1_nic_pair(self):
        owners, devices = self.plan()
        self.assertEqual(
            owners, [["0", "768", "rdma4,rdma5"], ["1", "768", "rdma6,rdma7"]]
        )
        self.assertEqual(devices, "rdma4,rdma5,rdma6,rdma7")

    def test_owner_entries_may_name_their_own_nics(self):
        owners, devices = self.plan(
            LMCACHE_MOONCAKE_OWNERS="0:64:rdma0,rdma1;1:96",
            LMCACHE_MOONCAKE_OWNER_RDMA_DEVICES="rdma6,rdma7",
        )
        self.assertEqual(
            owners, [["0", "64", "rdma0,rdma1"], ["1", "96", "rdma6,rdma7"]]
        )
        self.assertEqual(devices, "rdma0,rdma1,rdma6,rdma7")

    def test_invalid_owner_specs_are_refused(self):
        for env in (
            {"LMCACHE_MOONCAKE_OWNERS": "0:0"},
            {"LMCACHE_MOONCAKE_OWNERS": "x:10"},
            {"LMCACHE_MOONCAKE_OWNERS": "0:10;;1:10"},
            {"LMCACHE_MOONCAKE_OWNERS": "0:10:rdma 4"},
            {"LMCACHE_MOONCAKE_OWNERS": ";"},
            {"LMCACHE_MOONCAKE_OWNER_RDMA_DEVICES": "rdma4,,rdma5"},
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
# A second prefill worker of this shell reuses the Store.
start_mooncake_store
printf 'ENV %s\\n' "${mooncake_l2_prefill_env[@]}"
stop_mooncake_store
! mooncake_store_running
""",
            LMCACHE_MOONCAKE_OWNERS="0:64;1:96",
        )
        # The page cache goes before anything allocates, then the budget.
        drop, budget = [
            line.split(" ", 1)[1]
            for line in self.calls.read_text().splitlines()
            if "drop_page_cache.py /" in line or "numa_memory_budget.py" in line
        ]
        self.assertTrue(drop.endswith("/scripts/drop_page_cache.py /models/m"), drop)
        self.assertIn(
            "/scripts/numa_memory_budget.py --reserve-gib 128 --gpus 0,1,2,3 "
            "--per-gpu-gib 48 --compact 0:64 1:96",
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
        env = dict(
            line.removeprefix("ENV ").split("=", 1)
            for line in result.stdout.splitlines()
            if line.startswith("ENV ")
        )
        self.assertEqual(env["LMCACHE_REMOTE_URL"], f"mooncakestore://{HOST}:{port}/")
        self.assertEqual(env["MC_NUM_QP_PER_EP"], "1")
        self.assertEqual(env["MC_MAX_MR_SIZE"], "1073741824")
        self.assertEqual(env["MC_TCP_BIND_ADDRESS"], HOST)
        self.assertEqual(env["LMCACHE_BLOCKING_TIMEOUT_SECS"], "60")
        self.assertEqual(env["LMCACHE_MAX_LOCAL_CPU_SIZE"], "48")
        self.assertEqual(env["OFFLOAD_LOAD_WORKERS"], "1")
        self.assertEqual(
            env["ATOM_LMCACHE_MOONCAKE_OWNER_RDMA_DEVICES"], "rdma4,rdma5,rdma6,rdma7"
        )
        extra = json.loads(env["LMCACHE_EXTRA_CONFIG"])
        self.assertEqual(
            extra,
            {
                "save_chunk_meta": False,
                "transfer_timeout": 60,
                "use_exists_sync": True,
                "remote_enable_mla_worker_id_as0": False,
                "mooncake_local_hostname": HOST,
                "mooncake_metadata_server": f"http://{HOST}:50180/metadata",
                "mooncake_master_server_addr": f"{HOST}:{port}",
                "mooncake_protocol": "rdma",
                "mooncake_global_segment_size": "0",
                "mooncake_local_buffer_size": "67108864",
            },
        )
        self.assertNotIn("mooncake_rdma_devices", extra)  # ATOM sets it per rank
        metrics = (self.root / "logs/mooncake-master-rank-0.metrics").read_text()
        self.assertIn(f"master_total_capacity_bytes {160 * GIB}", metrics)
        self.assertTrue((self.root / "logs/mooncake-owner-rank-0-1.log").exists())
        for pid in self.stub_pids():
            self.assertFalse(self.alive(pid), pid)

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
                LMCACHE_MOONCAKE_OWNERS="0:64;1:96",
                LMCACHE_MOONCAKE_DECODE_OWNERS="0:32:rdma0",
            ),
            [["", str(192 * GIB), "50000"]],
        )

    def test_per_nic_pools_number_the_nics_of_both_nodes_alike(self):
        # Each node lists its owners in its own order; a decode node reads the
        # settings from their prefill-prefixed copies.
        self.assertEqual(
            self.pools(
                LMCACHE_MOONCAKE_POOLS="per_nic",
                LMCACHE_MOONCAKE_OWNERS="0:192:rdma1;0:192:rdma0;0:16:rdma10",
                ATOMESH_PREFILL_ENV_LMCACHE_MOONCAKE_DECODE_OWNERS=(
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
        per_nic = {"LMCACHE_MOONCAKE_POOLS": "per_nic"}
        for env, message in (
            (
                {**per_nic, "LMCACHE_MOONCAKE_OWNERS": "0:64:rdma0,rdma1"},
                "needs exactly one NIC; 0:64:rdma0,rdma1 names several",
            ),
            # Without a list of its own an owner takes rdma4-rdma7.
            ({**per_nic, "LMCACHE_MOONCAKE_OWNERS": "0:64"}, "needs exactly one NIC"),
            ({"LMCACHE_MOONCAKE_POOLS": "ring"}, "neither shared nor per_nic"),
            (
                {
                    **per_nic,
                    "LMCACHE_MOONCAKE_OWNERS": "0:8:rdma0;0:8:rdma1",
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
        decode_owners = {"LMCACHE_MOONCAKE_DECODE_OWNERS": "0:8:rdma0"}
        for env in (
            {"ATOMESH_PD_WORKER_LAYOUT": "single_node", "xP": "1"},
            {"ATOMESH_PD_WORKER_LAYOUT": "multi_node", "xP": "2"},
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
        )

    def test_per_nic_store_gives_each_nic_a_master_and_each_stage_its_pool(self):
        port = self.master_port
        result = self.run_shell(
            """
start_mooncake_store
printf 'ENV %s\\n' "${mooncake_l2_prefill_env[@]}"
# As cleanup_processes runs them.
save_mooncake_store_metrics
stop_mooncake_store
""",
            LMCACHE_MOONCAKE_POOLS="per_nic",
            LMCACHE_MOONCAKE_MASTER_NUMA="0",
            LMCACHE_MOONCAKE_OWNERS="0:64:rdma1;0:32:rdma0",
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
        env = dict(
            line.removeprefix("ENV ").split("=", 1)
            for line in result.stdout.splitlines()
            if line.startswith("ENV ")
        )
        self.assertEqual(
            json.loads(env["ATOM_LMCACHE_MOONCAKE_POOLS"]),
            {
                "rdma0": {
                    "master": f"{HOST}:{port}",
                    "metadata": f"http://{HOST}:50180/metadata",
                },
                "rdma1": {
                    "master": f"{HOST}:{port + 100}",
                    "metadata": f"http://{HOST}:50280/metadata",
                },
            },
        )
        # Owners share the stages' NICs by design here.
        self.assertNotIn("ATOM_LMCACHE_MOONCAKE_OWNER_RDMA_DEVICES", env)
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
            LMCACHE_MOONCAKE_L2="0",
            ATOMESH_PREFILL_ENV_LMCACHE_MOONCAKE_L2="1",
            ATOMESH_PREFILL_ENV_LMCACHE_MOONCAKE_POOLS="per_nic",
            ATOMESH_PREFILL_ENV_LMCACHE_MOONCAKE_OWNERS="0:64:rdma0;0:64:rdma1",
            ATOMESH_PREFILL_ENV_LMCACHE_MOONCAKE_DECODE_OWNERS=(
                "0:96:rdma1;0:96:rdma0"
            ),
        )
        self.assertIn("MASTERS=0", result.stdout)
        # No L1 here: the budget counts the owners only.
        (budget,) = [
            line
            for line in self.calls.read_text().splitlines()
            if "numa_memory_budget.py" in line
        ]
        self.assertIn("--gpus  --per-gpu-gib 48 --compact 0:96 0:96", budget)
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

    def test_an_owner_dying_before_ready_stops_the_store(self):
        self.run_shell(
            "start_mooncake_store\n",
            expect_rc=5,
            LMCACHE_MOONCAKE_OWNERS="0:8;1:8",
            STUB_DIE_OWNER_PORT="50153",
            LMCACHE_MOONCAKE_WAIT_TIMEOUT="30",
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
            LMCACHE_MOONCAKE_OWNERS="0:8;1:8",
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
            "start_mooncake_store\n",
            expect_rc=1,
            STUB_MASTER_NO_RPC="1",
            LMCACHE_MOONCAKE_MASTER_WAIT_TIMEOUT="4",
        )
        self.assertEqual(len(self.store_calls("mooncake_master")), 1)
        self.assertEqual(self.store_calls("mooncake_client"), [])
        for pid in self.stub_pids():
            self.assertFalse(self.alive(pid), pid)

    def test_pins_that_do_not_fit_stop_the_start_before_the_master(self):
        result = self.run_shell(
            "start_mooncake_store\n", expect_rc=2, STUB_BUDGET_RC="2"
        )
        self.assertIn("must fit their NUMA nodes", result.stderr)
        self.assertEqual(self.store_calls("mooncake_master"), [])

    def test_page_cache_drop_covers_the_configured_directories(self):
        self.run_shell(
            "prepare_mooncake_store_memory\n",
            LMCACHE_MOONCAKE_PAGE_CACHE_DROP_DIRS="/share/models:/data/cache",
            LMCACHE_MOONCAKE_NODE_RESERVE_GIB="64",
            LMCACHE_MAX_LOCAL_CPU_SIZE="32",
        )
        calls = self.calls.read_text()
        self.assertIn("/scripts/drop_page_cache.py /share/models /data/cache\n", calls)
        self.assertIn(
            "--reserve-gib 64 --gpus 0,1,2,3 --per-gpu-gib 32 --compact\n", calls
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
                    LMCACHE_MOONCAKE_OWNERS=owners,
                )
                self.assertIn(message, result.stderr)
        # NUMA0 memory served by NUMA1's NICs is fine: one node of NICs.
        self.run_shell(
            "plan_mooncake_store_owners\ncheck_mooncake_owner_devices\n",
            LMCACHE_MOONCAKE_OWNERS="0:8:rdma4,rdma5;1:8:rdma7",
        )

    def test_conflicting_settings_are_refused_before_anything_starts(self):
        for env in (
            {"LMCACHE_MP_SERVER": "1"},
            {"LMCACHE_EXTRA_CONFIG": "{}"},
            {"LMCACHE_REMOTE_URL": "mooncakestore://x:1/"},
            {"MC_NUM_QP_PER_EP": "2"},
            # The default owners name their own NICs; this one takes the list.
            {
                "LMCACHE_MOONCAKE_OWNERS": "0:8",
                "LMCACHE_MOONCAKE_OWNER_RDMA_DEVICES": "rdma9",
            },
        ):
            with self.subTest(env=env):
                self.run_shell("start_mooncake_store\n", expect_rc=2, **env)
        self.assertEqual(self.store_calls("mooncake_master"), [])

    def test_validation_reads_each_role_env_before_anything_starts(self):
        # As the launcher's main block runs it: from role-prefixed settings,
        # before the prefill (and its MP servers) or the Store start.
        ok = {
            "LMCACHE_MOONCAKE_L2": "0",
            "ATOMESH_PREFILL_ENV_LMCACHE_MOONCAKE_L2": "1",
        }
        for env, message in (
            (
                {"ATOMESH_PREFILL_ENV_LMCACHE_MP_SERVER": "1"},
                "cannot be combined with LMCACHE_MP_SERVER",
            ),
            ({"ATOMESH_DECODE_ENV_MC_NUM_QP_PER_EP": "2"}, "MC_NUM_QP_PER_EP=2"),
            ({"ATOMESH_ENV_MC_NUM_QP_PER_EP": "4"}, "MC_NUM_QP_PER_EP=4"),
            (
                {"ATOMESH_PREFILL_ENV_LMCACHE_MOONCAKE_OWNERS": "0:0"},
                "is not <numa>:<GiB>",
            ),
        ):
            with self.subTest(env=env):
                result = self.run_shell(
                    "validate_mooncake_l2_settings\n", expect_rc=2, **ok, **env
                )
                self.assertIn(message, result.stderr)
        result = self.run_shell(
            "validate_mooncake_l2_settings\n"
            'echo "LEAKED=${LMCACHE_MOONCAKE_OWNERS:-none}"\n',
            **ok,
            ATOMESH_PREFILL_ENV_LMCACHE_MOONCAKE_OWNERS="0:8",
        )
        self.assertIn("LEAKED=none", result.stdout)
        self.assertEqual(self.calls.read_text(), "")

    def test_disabled_without_the_prefill_flag(self):
        self.run_shell(
            "start_mooncake_store\n! mooncake_store_running\nstop_mooncake_store\n",
            LMCACHE_MOONCAKE_L2="0",
        )
        self.assertEqual(self.calls.read_text(), "")

    def test_decode_on_any_node_reads_the_prefill_flag(self):
        self.run_shell(
            "mooncake_l2_requested\nrequire_mooncake_qp_per_endpoint\n",
            LMCACHE_MOONCAKE_L2="0",
            ATOMESH_PREFILL_ENV_LMCACHE_MOONCAKE_L2="1",
            MC_NUM_QP_PER_EP="1",
        )
        self.run_shell(
            "! mooncake_l2_requested\n",
            LMCACHE_MOONCAKE_L2="0",
            ATOMESH_PREFILL_ENV_LMCACHE_MOONCAKE_L2="0",
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
            LMCACHE_MOONCAKE_OWNERS="0:8",
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
            if not key.startswith(("LMCACHE_", "ATOMESH_"))
        }
        # 50051 + 20000 is no port, but only the Store uses it.
        for l2, expect_rc in (("0", 0), ("1", 2)):
            result = subprocess.run(
                ["bash", "-c", "set -euo pipefail\n" + ports],
                env={
                    **base,
                    "ATOMESH_SERVICE_PORT_OFFSET": "20000",
                    "ATOMESH_PREFILL_ENV_LMCACHE_MOONCAKE_L2": l2,
                },
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            self.assertEqual(result.returncode, expect_rc, result.stderr)
        self.assertIn("ATOMESH_MOONCAKE_MASTER_PORT=70051 is outside", result.stderr)

    def test_prefill_gets_the_store_env_and_decode_one_qp(self):
        prefill = self.source.split("start_prefill() {")[1].split("\n}\n")[0]
        self.assertIn("start_mooncake_store", prefill)
        self.assertIn('prefill_offload_env+=("${mooncake_l2_prefill_env[@]}")', prefill)
        decode = self.source.split("start_decode() {")[1].split("\n}\n")[0]
        self.assertIn('decode_mooncake_env=("MC_NUM_QP_PER_EP=1")', decode)
        self.assertIn('"${decode_mooncake_env[@]}" "${decode_cmd[@]}"', decode)
        # Every setting is checked before any role starts.
        self.assertLess(
            self.source.index("\nvalidate_mooncake_l2_settings\nwrite_metadata\n"),
            self.source.index('  start_prefill "prefill-rank-0"'),
        )
        cleanup = self.source.split("cleanup_processes() {")[1].split("\n}\n")[0]
        self.assertIn("stop_mooncake_store", cleanup)
        for waiter in ("wait_http() {", "wait_router_closed() {"):
            body = self.source.split(waiter)[1].split("\n}\n")[0]
            self.assertIn("exit_if_mooncake_store_died", body)


if __name__ == "__main__":
    unittest.main()
