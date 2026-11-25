# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Exercise the launcher's UMBP pool: its server's lifecycle, and what each
worker of the DeepSeek-V4.1 umbp cells starts with.

No UMBP server starts: a stub stands in for the binary. It records its
environment and arguments and listens on the unix socket of its address, as
the readiness check needs. A test can have it exit before it listens
(``STUB_SERVER_MODE=die``), never listen (``stall``), or exit later with the
status in ``${STUB_DIR}/server-exit``, after sixty lines of log.
"""

import json
import os
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / ".github/scripts/atomesh"
SOURCE = (SCRIPTS / "pd_server_atom.sh").read_text()

SERVER_STUB = textwrap.dedent("""\
    #!/usr/bin/env bash
    echo "$$ HIP=${HIP_VISIBLE_DEVICES-unset} LEVEL=${MORI_UMBP_LOG_LEVEL-unset} EXTRA=${STUB_EXTRA-unset} $*" >> "${STUB_CALLS}"
    echo "umbp stub starts on $1"
    case "${STUB_SERVER_MODE:-serve}" in
      die) echo "E stub cannot open its device"; exit 7 ;;
      stall) exec sleep 60 ;;
    esac
    exec "${REAL_PYTHON}" -c 'import os, socket, sys, time
    server = socket.socket(socket.AF_UNIX)
    server.bind(sys.argv[1].removeprefix("unix://"))
    server.listen(8)
    trigger = os.path.join(os.environ["STUB_DIR"], "server-exit")
    for _ in range(600):
        if os.path.exists(trigger) and (status := open(trigger).read().strip()):
            for line in range(60):
                print(f"stub log line {line}", flush=True)
            sys.exit(int(status))
        time.sleep(0.1)' "$1"
    """)

# Waits until the server the launcher started has exited by itself.
AWAIT_EXIT = """
for _ in $(seq 100); do
  process_is_running "${umbp_pool_server_pid}" || break
  sleep 0.1
done
"""


def section(start, end=None):
    begin = SOURCE.index(start)
    return SOURCE[begin : SOURCE.index(end, begin)] if end else SOURCE[begin:]


# The launcher's functions around the server, as the Store's tests take them.
FUNCTIONS = section("apply_prefixed_env() {", 'host_ip="$(echo') + section(
    "process_is_running()", "write_metadata() {"
)
WAITS = section("wait_http() {", "start_logged_process() {")
SERVERS = section("start_prefill() {", "start_router() {")
# Records a worker's devices and command line, one word per line, and gives the
# launcher a process to stop in its place.
RECORD_WORKERS = """
start_logged_process() {
  {
    echo "HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES}"
    printf '%s\\n' "${@:3}"
  } > "${STUB_DIR}/argv-$(basename "$2" .log)"
  setsid sleep 120 >/dev/null 2>&1 &
  printf -v "$1" '%s' "$!"
}
build_server_cache_env() { local -n out="$3"; out=(); }
"""


def alive(pid):
    try:
        state = Path(f"/proc/{pid}/stat").read_text().split()[2]
    except (FileNotFoundError, IndexError):
        return False
    return state != "Z"


class UmbpPoolTestCase(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="atomesh-umbp "))
        self.addCleanup(self.remove_root)
        (self.root / "logs").mkdir()
        bin_dir = self.root / "bin"
        bin_dir.mkdir()
        self.server = bin_dir / "umbp_standalone_server"
        self.server.write_text(SERVER_STUB)
        self.server.chmod(0o755)
        curl = bin_dir / "curl"
        curl.write_text('#!/usr/bin/env bash\nexit "${STUB_CURL_RC:-7}"\n')
        curl.chmod(0o755)
        # Its directory is the launcher's to make.
        self.socket = self.root / "run/server.sock"
        self.address = f"unix://{self.socket}"
        self.calls = self.root / "calls"
        self.calls.touch()
        self.stubs = {
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "STUB_CALLS": str(self.calls),
            "STUB_DIR": str(self.root),
            "REAL_PYTHON": sys.executable,
        }
        self.addCleanup(self.kill_stubs)

    def remove_root(self):
        # A temporary directory on NFS keeps a log for a moment after the
        # server that wrote it exited: removing it then leaves a .nfs file
        # behind, and its directory not empty, until the client closes it.
        for _ in range(50):
            try:
                shutil.rmtree(self.root)
                return
            except FileNotFoundError:
                return
            except OSError:
                time.sleep(0.1)
        shutil.rmtree(self.root, ignore_errors=True)

    def kill_stubs(self):
        for call in self.server_calls():
            try:
                os.killpg(int(call.split()[0]), 9)
            except (ProcessLookupError, PermissionError):
                pass

    def server_calls(self):
        return self.calls.read_text().splitlines()

    def workers(self):
        """Each recorded worker's devices, env words and server command."""
        found = {}
        for path in sorted(self.root.glob("argv-*")):
            devices, env, *words = path.read_text().splitlines()
            self.assertEqual(env, "env")
            command = words.index("python3")
            found[path.name.removeprefix("argv-")] = (
                devices.removeprefix("HIP_VISIBLE_DEVICES="),
                dict(word.split("=", 1) for word in words[:command]),
                words[command:],
            )
        return found


class UmbpPoolServerTest(UmbpPoolTestCase):
    def run_shell(self, script, expect_rc=0, **env):
        """Runs `script` after the launcher's functions, on these variables
        alone; None drops one."""
        settings = {
            **self.stubs,
            "HANDSHAKE_PORT": "6301",
            "RUNTIME_LOG_DIR": str(self.root / "logs"),
            "RUN_DIR": str(self.root / "run_dir"),
            "NODE_RANK": "0",
            "host_ip": "127.0.0.1",
            # A prefill worker's, which the server must not inherit.
            "HIP_VISIBLE_DEVICES": "2,3",
            "UMBP_POOL": "1",
            "UMBP_POOL_SERVER": str(self.server),
            "UMBP_POOL_ADDRESS": self.address,
            **env,
        }
        result = subprocess.run(
            [
                "bash",
                "-c",
                "set -euo pipefail\ndump_launch_info() { :; }\n" + FUNCTIONS + script,
            ],
            env={key: value for key, value in settings.items() if value is not None},
            cwd=self.root,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        self.assertEqual(result.returncode, expect_rc, result.stdout + result.stderr)
        return result

    def test_a_node_starts_one_server_that_sees_every_gpu_and_stops_it(self):
        result = self.run_shell("""
dump_launch_info() { printf 'LAUNCH'; printf ' %q' "$@"; echo; }
start_umbp_pool_server
# The node's next prefill worker.
start_umbp_pool_server
test -S "${UMBP_POOL_ADDRESS#unix://}"
echo "PID=${umbp_pool_server_pid}"
stop_umbp_pool_server
# cleanup_processes runs again from the EXIT trap.
stop_umbp_pool_server
""")
        (call,) = self.server_calls()
        pid, *recorded = call.split(" ", 4)
        self.assertEqual(
            recorded, ["HIP=unset", "LEVEL=info", "EXTRA=unset", self.address]
        )
        self.assertIn(f"PID={pid}\n", result.stdout)
        # The launch info shows the server's own environment and command.
        launch = shlex.split(result.stdout.split("LAUNCH ", 1)[1].splitlines()[0])
        self.assertEqual(
            launch,
            [
                "UMBP_SERVER",
                "env",
                "-u",
                "HIP_VISIBLE_DEVICES",
                "MORI_UMBP_LOG_LEVEL=info",
                "setsid",
                str(self.server),
                self.address,
            ],
        )
        log = self.root / "logs/umbp-server-rank-0.log"
        self.assertEqual(log.read_text(), f"umbp stub starts on {self.address}\n")
        self.assertEqual(result.stdout.count("[umbp-pool] server stopped"), 1)
        self.assertFalse(alive(pid), pid)

    def test_the_cases_words_join_the_servers_environment(self):
        self.run_shell(
            "start_umbp_pool_server\nstop_umbp_pool_server\n",
            UMBP_POOL_SERVER_ENV="STUB_EXTRA=a=b  MORI_UMBP_LOG_LEVEL=debug",
        )
        (call,) = self.server_calls()
        self.assertIn(" HIP=unset LEVEL=debug EXTRA=a=b ", call)

    def test_a_stale_socket_does_not_pass_for_a_server_that_never_listens(self):
        # What an earlier server left behind at the address.
        self.socket.parent.mkdir()
        with socket.socket(socket.AF_UNIX) as stale:
            stale.bind(str(self.socket))
        self.assertTrue(self.socket.is_socket())
        result = self.run_shell(
            "start_umbp_pool_server\n",
            expect_rc=1,
            STUB_SERVER_MODE="stall",
            # Two seconds in decimal, not an octal number.
            UMBP_POOL_START_TIMEOUT="0002",
        )
        self.assertIn(
            f"umbp-pool server has no socket {self.socket} after 2s", result.stderr
        )
        self.assertIn(f"umbp stub starts on {self.address}", result.stderr)
        self.assertFalse(self.socket.exists())
        (call,) = self.server_calls()
        self.assertFalse(alive(call.split()[0]), call)

    def test_a_server_that_exits_before_it_listens_fails_the_start(self):
        result = self.run_shell(
            "start_umbp_pool_server\n", expect_rc=7, STUB_SERVER_MODE="die"
        )
        self.assertIn("E stub cannot open its device", result.stderr)
        self.assertIn(
            "[umbp-pool][FAIL] the UMBP server exited unexpectedly rc=7, see "
            f"{self.root}/logs/umbp-server-rank-0.log",
            result.stderr,
        )

    def test_the_waits_fail_once_the_server_exited(self):
        waits = {
            'wait_http "http://127.0.0.1:9/health" prefill 600': "7",
            "wait_router_closed": "0",
        }
        # A server that exits cleanly leaves its workers without it as well.
        for status, expect_rc in (("3", 3), ("0", 1)):
            for wait, curl_rc in waits.items():
                with self.subTest(wait=wait, status=status):
                    (self.root / "server-exit").unlink(missing_ok=True)
                    result = self.run_shell(
                        WAITS
                        + "start_umbp_pool_server\n"
                        + f'echo {status} > "${{STUB_DIR}}/server-exit"\n'
                        + AWAIT_EXIT
                        + wait
                        + "\n",
                        expect_rc=expect_rc,
                        STUB_CURL_RC=curl_rc,
                        NODE0_ADDR="127.0.0.1",
                        ROUTER_PORT="9",
                    )
                    self.assertIn(
                        f"the UMBP server exited unexpectedly rc={status}",
                        result.stderr,
                    )
                    self.assertIn("stub log line 59\n", result.stderr)

    def test_cleanup_stops_the_server_after_the_workers(self):
        # A worker detaches from the server as it exits: here it finds the
        # server still up when it is told to.
        result = self.run_shell("""
start_umbp_pool_server
SERVER="${umbp_pool_server_pid}" setsid bash -c '
  trap "kill -0 \\"${SERVER}\\" && touch \\"${STUB_DIR}/server-up-at-term\\"; exit 0" TERM
  for _ in $(seq 300); do sleep 0.2; done' &
sleep 1
cleanup_processes $!
""")
        self.assertTrue((self.root / "server-up-at-term").exists(), result.stderr)
        self.assertIn("[umbp-pool] server stopped", result.stdout)
        (call,) = self.server_calls()
        self.assertFalse(alive(call.split()[0]), call)

    def test_a_server_that_exited_by_itself_fails_the_cleanup_unless_it_exited_0(
        self,
    ):
        script = (
            "start_umbp_pool_server\n"
            + 'echo "${STATUS}" > "${STUB_DIR}/server-exit"\n'
            + AWAIT_EXIT
            + 'if cleanup_processes; then echo "CLEANUP rc=0"; else echo "CLEANUP rc=$?"; fi\n'
            # Again from the EXIT trap, after the launcher exited with that status.
            + 'if cleanup_processes; then echo "TRAP rc=0"; else echo "TRAP rc=$?"; fi\n'
        )
        result = self.run_shell(script, STATUS="5")
        self.assertIn("CLEANUP rc=5\nTRAP rc=5\n", result.stdout)
        self.assertEqual(
            result.stderr.count(
                "[umbp-pool][FAIL] the UMBP server exited by itself rc=5 before it was stopped"
            ),
            1,
        )
        # The last 40 of the log's 61 lines.
        self.assertIn("stub log line 20\n", result.stderr)
        self.assertNotIn("stub log line 19\n", result.stderr)
        (self.root / "server-exit").unlink()
        result = self.run_shell(script, STATUS="0")
        self.assertIn("CLEANUP rc=0\nTRAP rc=0\n", result.stdout)
        self.assertIn("[umbp-pool] server had exited by itself rc=0", result.stdout)
        self.assertNotIn("stub log line", result.stderr)

    def test_settings_the_server_cannot_start_with_are_refused_before_anything_starts(
        self,
    ):
        # As the launcher's main block reads them: role-prefixed, before the
        # prefill starts.
        role = "ATOMESH_PREFILL_ENV_"
        ok = {
            "UMBP_POOL": None,
            "UMBP_POOL_SERVER": None,
            "UMBP_POOL_ADDRESS": None,
            role + "UMBP_POOL": "1",
            role + "UMBP_POOL_SERVER": str(self.server),
        }
        for env, message in (
            (
                {role + "UMBP_POOL_SERVER": str(self.root / "missing")},
                "/missing is not an executable file",
            ),
            (
                {role + "UMBP_POOL_SERVER": str(self.calls)},
                "/calls is not an executable file",
            ),
            (
                {role + "UMBP_POOL_SERVER": None},
                "UMBP_POOL_SERVER=/opt/umbp-l1/umbp_standalone_server is not",
            ),
            # A worker given the Store's connector and the pool's refuses to
            # start, long after the Store has.
            ({role + "MOONCAKE_STORE": "1"}, "UMBP_POOL=1 with MOONCAKE_STORE=1"),
            (
                {role + "UMBP_POOL_ADDRESS": "127.0.0.1:50051"},
                "UMBP_POOL_ADDRESS=127.0.0.1:50051 is not unix://<absolute path>",
            ),
            (
                {role + "UMBP_POOL_ADDRESS": "unix:server.sock"},
                "is not unix://<absolute path>",
            ),
            (
                {role + "UMBP_POOL_START_TIMEOUT": "2m"},
                "UMBP_POOL_START_TIMEOUT=2m is not a positive integer",
            ),
            (
                {role + "UMBP_POOL_START_TIMEOUT": "0"},
                "UMBP_POOL_START_TIMEOUT=0 is not a positive integer",
            ),
            # env would run it in the server's place.
            (
                {role + "UMBP_POOL_SERVER_ENV": "A=1 sleep"},
                "UMBP_POOL_SERVER_ENV word 'sleep' is not NAME=VALUE",
            ),
            ({"AGGREGATED": "1"}, "an aggregated cell starts none"),
        ):
            with self.subTest(env=env):
                if (
                    "/opt/umbp-l1" in message
                    and Path("/opt/umbp-l1/umbp_standalone_server").exists()
                ):
                    self.skipTest("this host has the UMBP standalone server")
                result = self.run_shell(
                    "validate_umbp_pool_settings\n", expect_rc=2, **{**ok, **env}
                )
                self.assertIn(message, result.stderr)
        result = self.run_shell(
            'validate_umbp_pool_settings\necho "LEAKED=${UMBP_POOL_SERVER:-none}"\n',
            **ok,
            **{
                role + "UMBP_POOL_ADDRESS": self.address,
                role + "UMBP_POOL_START_TIMEOUT": "0090",
                role + "UMBP_POOL_SERVER_ENV": "A=1 B=x=y",
                role + "MOONCAKE_STORE": "0",
            },
        )
        self.assertIn("LEAKED=none", result.stdout)
        self.assertEqual(self.server_calls(), [])
        self.assertFalse(self.socket.parent.exists())

    def test_nothing_starts_without_the_prefill_switch(self):
        self.run_shell(
            "validate_umbp_pool_settings\nstart_umbp_pool_server\n"
            "exit_if_umbp_pool_server_died\ncleanup_processes\n",
            UMBP_POOL="0",
            UMBP_POOL_SERVER=str(self.root / "missing"),
        )
        self.assertEqual(self.server_calls(), [])
        self.assertFalse(self.socket.parent.exists())

    def test_each_prefill_worker_is_told_the_servers_ordinals_of_its_devices(self):
        # Two prefill workers and a decode worker of one node, from one shell,
        # with the settings where the launcher finds them: the prefill role env.
        role = "ATOMESH_PREFILL_ENV_"
        self.run_shell(
            SERVERS + RECORD_WORKERS + """
build_torch_profiler_args() { local -n out="$2"; out=(); }
ATOM_SOURCE_ENV=()
server_common=(--model /models/m)
prefill_parallel=(-tp 2)
decode_parallel=(-tp 4)
prefill_cudagraph_args=()
decode_cudagraph_args=()
pids=()
start_prefill prefill-worker-0 8010 6301 29500 29600 "0,1"
pids+=("${server_pid}")
start_prefill prefill-worker-1 8011 6303 29900 30000 "2,3"
pids+=("${server_pid}")
start_decode decode-worker-0 8020 6305 29700 29800 "4,5,6,7"
pids+=("${server_pid}")
cleanup_processes "${pids[@]}"
""",
            **{
                "UMBP_POOL": None,
                "UMBP_POOL_SERVER": None,
                "UMBP_POOL_ADDRESS": None,
                role + "UMBP_POOL": "1",
                role + "UMBP_POOL_SERVER": str(self.server),
                role + "UMBP_POOL_ADDRESS": self.address,
            },
            PREFILL_KV_TRANSFER_CONFIG="",
            DECODE_KV_TRANSFER_CONFIG="",
            PREFILL_PORT="8010",
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
        workers = self.workers()
        self.assertEqual(
            {name: (devices, env) for name, (devices, env, _) in workers.items()},
            {
                "prefill-worker-0": ("0,1", {"ATOM_UMBP_DEVICE_ORDINALS": "0,1"}),
                "prefill-worker-1": ("2,3", {"ATOM_UMBP_DEVICE_ORDINALS": "2,3"}),
                "decode-worker-0": ("4,5,6,7", {}),
            },
        )
        # One server for the node, started under the first worker's devices.
        (call,) = self.server_calls()
        self.assertIn(f" HIP=unset LEVEL=info EXTRA=unset {self.address}", call)
        self.assertFalse(alive(call.split()[0]), call)


class UmbpPoolCellsLaunchTest(UmbpPoolTestCase):
    """What the launcher starts on each node of the DeepSeek-V4.1 umbp cells:
    the script from its settings to its main flow, on a cell's exports, with
    stand-ins for the workers, the router and the benchmark."""

    MATRIX_ENV = {
        "ATOMESH_SLURM_SUBMIT_RUNNER": "atomesh-cicd",
        "ATOMESH_SLURM_ACCOUNT": "",
        "ATOMESH_SLURM_PARTITION": "",
        "ATOMESH_MODEL_ROOT": "/models",
        "ATOMESH_LOG_ROOT": "/logs",
        "ATOMESH_1P1D_NODES": "pit2-p03-g27",
        "ATOMESH_2P1D_NODES": "pit2-p03-g27,pit2-p03-g48,pit2-p03-g53",
        "ATOMESH_PD_RANK_MAPPING_POLICY": "none",
    }
    LAUNCHER = (
        "set -euo pipefail\nATOM_SOURCE_ENV=()\n"
        + section(
            'MODEL_NAME="${MODEL_NAME:?MODEL_NAME is required}"',
            "rm -rf /root/.cache/atom",
        )
        + section("dump_launch_info() {", "# Where a worker's FlyDSL cache lives")
        + section(
            "server_common=(",
            "# Every branch installs its EXIT trap before its first start",
        )
        + RECORD_WORKERS
        + """
start_router() { router_pid=""; }
wait_http() { :; }
wait_router_closed() { :; }
run_benchmark_and_eval() { echo "BENCHMARK"; }
"""
        + section("# Every branch installs its EXIT trap before its first start")
    )

    @classmethod
    def setUpClass(cls):
        env = dict(os.environ, **cls.MATRIX_ENV)
        submit = (SCRIPTS / "pd_submit.sh").read_text()
        start = submit.index("python3 - <<'PY'\n") + len("python3 - <<'PY'\n")
        exports = submit[start : submit.index("\nPY\n", start)]
        cls.exports = {}
        with tempfile.TemporaryDirectory() as directory:
            for suite in ("umbp_kv_events", "umbp_smoke", "umbp_2p1d_kv_events"):
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
                    env=env,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                (cell,) = json.loads(output.read_text())["include"]
                result = subprocess.run(
                    [sys.executable, "-c", exports],
                    env=dict(env, CELL_JSON=json.dumps(cell)),
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                cls.exports[suite] = dict(
                    shlex.split(line.removeprefix("export "))[0].split("=", 1)
                    for line in result.stdout.splitlines()
                    if line.startswith("export ")
                )

    def launch(self, suite, rank, nodes):
        """Runs the launcher on node `rank` of `nodes`, up to its cleanup."""
        exports = self.exports[suite]
        ips = [f"10.0.0.{index}" for index in range(1, nodes + 1)]
        # Longer than one argument of a command line may be.
        launcher = self.root / "launcher.sh"
        launcher.write_text(self.LAUNCHER)
        result = subprocess.run(
            ["bash", str(launcher)],
            env={
                **self.stubs,
                **exports,
                # As pd_slurm_job.sh names them for the container.
                "xP": exports["PREFILL_WORKERS"],
                "yD": exports["DECODE_WORKERS"],
                "PREFILL_TP_SIZE": exports["PREFILL_TP"],
                "DECODE_TP_SIZE": exports["DECODE_TP"],
                "NODE_RANK": str(rank),
                "NODE0_ADDR": ips[0],
                "IPADDRS": ",".join(ips),
                "RUN_DIR": str(self.root / "run_dir"),
                "ATOMESH_SCRIPT_DIR": str(SCRIPTS),
                # The stub, on a socket of this test's.
                "ATOMESH_PREFILL_ENV_UMBP_POOL_SERVER": str(self.server),
                "ATOMESH_PREFILL_ENV_UMBP_POOL_ADDRESS": self.address,
            },
            cwd=self.root,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("BENCHMARK" if rank == 0 else "[layout]", result.stdout)
        return self.workers()

    @staticmethod
    def flag(command, name):
        return command[command.index(name) + 1]

    def assert_one_server_served_and_stopped(self):
        (call,) = self.server_calls()
        self.assertIn(f" HIP=unset LEVEL=info EXTRA=unset {self.address}", call)
        self.assertFalse(alive(call.split()[0]), call)

    def test_the_smoke_runs_both_workers_on_devices_of_their_own_of_one_node(self):
        workers = self.launch("umbp_smoke", rank=0, nodes=1)
        self.assertEqual(sorted(workers), ["decode-rank-0", "prefill-rank-0"])
        devices, env, command = workers["prefill-rank-0"]
        self.assertEqual(devices, "0,1")
        self.assertEqual(
            env,
            {
                "ATOM_DP_MASTER_PORT": "29500",
                "ATOM_DP_BASE_PORT": "29600",
                "ATOM_UMBP_DEVICE_ORDINALS": "0,1",
                "ATOM_KV_EVENTS_ENABLE": "1",
                "ATOM_KV_EVENTS_ENDPOINT": "tcp://*:5557",
                "ATOM_KV_EVENTS_REPLAY_ENDPOINT": "tcp://*:5573",
            },
        )
        self.assertEqual(self.flag(command, "--server-port"), "8010")
        self.assertEqual(self.flag(command, "-tp"), "2")
        self.assertEqual(
            json.loads(self.flag(command, "--kv-transfer-config")),
            {
                "kv_connector": "multi",
                "connectors": [
                    {
                        "kv_role": "kv_producer",
                        "kv_connector": "mooncake",
                        "proxy_ip": "10.0.0.1",
                        "handshake_port": 6301,
                        "protocol": "rdma",
                    },
                    {
                        "kv_connector": "umbp_pool",
                        "kv_connector_extra_config": {
                            "address": "unix:///tmp/umbp-l1/server.sock",
                            "namespace": "p6301",
                        },
                    },
                ],
            },
        )
        devices, env, command = workers["decode-rank-0"]
        self.assertEqual(devices, "4,5,6,7")
        self.assertEqual(
            env, {"ATOM_DP_MASTER_PORT": "29700", "ATOM_DP_BASE_PORT": "29800"}
        )
        self.assertEqual(self.flag(command, "--server-port"), "8020")
        self.assertEqual(self.flag(command, "-tp"), "4")
        self.assertEqual(
            json.loads(self.flag(command, "--kv-transfer-config"))["kv_connector"],
            "mooncake",
        )
        self.assert_one_server_served_and_stopped()

    def test_two_prefill_workers_share_the_decodes_node_and_one_server(self):
        workers = self.launch("umbp_2p1d_kv_events", rank=0, nodes=1)
        self.assertEqual(
            sorted(workers),
            [
                "decode-rank-0-worker-0",
                "prefill-rank-0-worker-0",
                "prefill-rank-0-worker-1",
            ],
        )
        for index in range(2):
            with self.subTest(worker=index):
                devices, env, command = workers[f"prefill-rank-0-worker-{index}"]
                self.assertEqual(devices, f"{2 * index},{2 * index + 1}")
                self.assertEqual(env["ATOM_UMBP_DEVICE_ORDINALS"], devices)
                self.assertEqual(self.flag(command, "--server-port"), str(8010 + index))
                p_d, pool = json.loads(self.flag(command, "--kv-transfer-config"))[
                    "connectors"
                ]
                self.assertEqual(p_d["handshake_port"], 6301 + 2 * index)
                self.assertEqual(
                    pool["kv_connector_extra_config"]["namespace"],
                    f"p{6301 + 2 * index}",
                )
        devices, env, command = workers["decode-rank-0-worker-0"]
        self.assertEqual(devices, "4,5,6,7")
        self.assertNotIn("ATOM_UMBP_DEVICE_ORDINALS", env)
        self.assertEqual(
            json.loads(self.flag(command, "--kv-transfer-config"))["handshake_port"],
            6305,
        )
        self.assert_one_server_served_and_stopped()

    def test_only_the_cells_of_one_node_take_mooncake_off_hip_ipc(self):
        # Where the prefill and the decode share a node, a prefill worker's two
        # devices cannot open the memory of the decode's third and fourth.
        self.assertEqual(
            {
                suite: exports.get("ATOMESH_ENV_MC_DISABLE_HIP")
                for suite, exports in self.exports.items()
            },
            {"umbp_kv_events": None, "umbp_smoke": "1", "umbp_2p1d_kv_events": "1"},
        )

    def test_the_prefill_node_runs_one_server_for_its_four_workers(self):
        workers = self.launch("umbp_kv_events", rank=0, nodes=3)
        self.assertEqual(
            sorted(workers), [f"prefill-rank-0-worker-{index}" for index in range(4)]
        )
        namespaces = set()
        for index in range(4):
            with self.subTest(worker=index):
                devices, env, command = workers[f"prefill-rank-0-worker-{index}"]
                # Two GPUs each, and the handshake port of the first one's rank.
                self.assertEqual(devices, f"{2 * index},{2 * index + 1}")
                self.assertEqual(env["ATOM_UMBP_DEVICE_ORDINALS"], devices)
                self.assertEqual(self.flag(command, "--server-port"), str(8010 + index))
                self.assertEqual(
                    env["ATOM_KV_EVENTS_ENDPOINT"], f"tcp://*:{5557 + 32 * index}"
                )
                self.assertEqual(env["ATOM_DP_MASTER_PORT"], str(29500 + 400 * index))
                p_d, pool = json.loads(self.flag(command, "--kv-transfer-config"))[
                    "connectors"
                ]
                self.assertEqual(p_d["handshake_port"], 6301 + 2 * index)
                self.assertEqual(
                    pool["kv_connector_extra_config"]["namespace"],
                    f"p{6301 + 2 * index}",
                )
                namespaces.add(pool["kv_connector_extra_config"]["namespace"])
        self.assertEqual(len(namespaces), 4)
        self.assert_one_server_served_and_stopped()

    def test_a_decode_node_starts_no_server(self):
        workers = self.launch("umbp_kv_events", rank=1, nodes=3)
        self.assertEqual(
            {
                name: (devices, env.get("ATOM_UMBP_DEVICE_ORDINALS"))
                for name, (devices, env, _) in workers.items()
            },
            {
                "decode-rank-1-worker-0": ("0,1,2,3", None),
                "decode-rank-1-worker-1": ("4,5,6,7", None),
            },
        )
        self.assertEqual(self.server_calls(), [])
        self.assertFalse(self.socket.parent.exists())


if __name__ == "__main__":
    unittest.main()
