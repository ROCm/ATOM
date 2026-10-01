# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Exercise the launcher's LMCache MP server planning and lifecycle.

No GPU service starts: a ``python3`` stub on PATH stands in for the LMCache
server, numa_exec.py and drop_page_cache.py, and records each invocation.
"""

import json
import os
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path

SERVER_SCRIPT = (
    Path(__file__).resolve().parents[1] / ".github/scripts/atomesh/pd_server_atom.sh"
)

PYTHON_STUB = textwrap.dedent("""\
    #!/usr/bin/env bash
    echo "$$ HIP=${HIP_VISIBLE_DEVICES-unset} CUDA=${CUDA_VISIBLE_DEVICES-unset} $*" >> "${STUB_CALLS}"
    case " $* " in
      *" lmcache.v1.multiprocess.server "*)
        if [[ -n "${STUB_DIE_PORT:-}" && " $* " == *" --port ${STUB_DIE_PORT} "* ]]; then
          exit 3
        fi
        echo "LMCache cache server is running..."
        exec sleep 60
        ;;
      *" --every "*) exec sleep 60 ;;
    esac
    exit 0
    """)


class LMCacheMPServersTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="atomesh-mp ")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        (self.root / "logs").mkdir()
        bin_dir = self.root / "bin"
        bin_dir.mkdir()
        stub = bin_dir / "python3"
        stub.write_text(PYTHON_STUB)
        stub.chmod(0o755)
        self.calls = self.root / "calls"
        self.calls.touch()
        self.path = f"{bin_dir}:{os.environ['PATH']}"
        source = SERVER_SCRIPT.read_text()
        self.functions = source[
            source.index("process_is_running()") : source.index("cleanup_processes()")
        ]
        self.addCleanup(self.kill_stubs)

    def kill_stubs(self):
        for pid in self.stub_pids():
            try:
                os.killpg(pid, 9)
            except (ProcessLookupError, PermissionError):
                pass

    def stub_pids(self):
        return [int(line.split()[0]) for line in self.calls.read_text().splitlines()]

    def server_calls(self):
        return [
            line
            for line in self.calls.read_text().splitlines()
            if "lmcache.v1.multiprocess.server" in line
        ]

    def run_shell(self, script, expect_rc=0, **env):
        base = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith(("LMCACHE_", "HIP_", "CUDA_", "PREFILL_"))
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
                "RUNTIME_LOG_DIR": str(self.root / "logs"),
                "NODE_RANK": "0",
                "ATOMESH_SCRIPT_DIR": "/scripts",
                "MODEL_PATH": "/models/m",
                "ATOMESH_LMCACHE_MP_PORT": "25555",
                "ATOMESH_LMCACHE_MP_PROMETHEUS_PORT": "29190",
                "LMCACHE_MP_SERVER": "1",
                "PREFILL_TP_SIZE": "1",
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
plan_lmcache_mp_servers
for i in "${!lmcache_mp_plan_numa[@]}"; do
  echo "${lmcache_mp_plan_numa[i]}|${lmcache_mp_plan_first_stage[i]}|${lmcache_mp_plan_last_stage[i]}|${lmcache_mp_plan_l1_gb[i]}"
done
echo "JSON $(lmcache_mp_extra_config_json)"
""",
            **env,
        )
        lines = result.stdout.splitlines()
        config = json.loads(lines[-1].removeprefix("JSON "))
        return [line.split("|") for line in lines[:-1]], config

    def test_stage_spec_plans_one_server_per_numa_group(self):
        servers, config = self.plan(
            LMCACHE_MP_STAGE_SERVERS="0:0-1:1000;1:2-3:1000",
            HIP_VISIBLE_DEVICES="0,1,2,3",
        )
        self.assertEqual(
            servers,
            [["0", "0", "1", "1000"], ["1", "2", "3", "1000"]],
        )
        self.assertEqual(
            config,
            {
                "lmcache.mp.stage_servers": [
                    {"url": "tcp://127.0.0.1:25555", "pp_ranks": [0, 1]},
                    {"url": "tcp://127.0.0.1:25556", "pp_ranks": [2, 3]},
                ],
                "lmcache.mp.l2": "none",
            },
        )

    def test_single_server_plan_and_config_are_unchanged(self):
        servers, config = self.plan(
            LMCACHE_MP_NUMA_NODE="0",
            LMCACHE_MP_L1_SIZE_GB="1024",
            HIP_VISIBLE_DEVICES="0,1,2,3",
        )
        self.assertEqual(servers, [["0", "", "", "1024"]])
        self.assertEqual(
            config,
            {
                "lmcache.mp.host": "tcp://127.0.0.1",
                "lmcache.mp.port": 25555,
                "lmcache.mp.l2": "none",
            },
        )

    def test_invalid_stage_specs_are_refused(self):
        for env in (
            {"LMCACHE_MP_NUMA_NODE": "0"},
            {"LMCACHE_MP_L1_SIZE_GB": "10"},
            {"LMCACHE_MP_STAGE_SERVERS": "0:0-3:1000"},
            {"LMCACHE_MP_STAGE_SERVERS": "0:0-0:10;1:2-3:10"},
            {"LMCACHE_MP_STAGE_SERVERS": "0:0-1:10;1:2-3:big"},
            # Stages past the pipeline are refused by the engine's own
            # lmcache.mp.stage_servers check, not by the launcher.
        ):
            env = {
                "LMCACHE_MP_STAGE_SERVERS": "0:0-1:10;1:2-3:10",
                "HIP_VISIBLE_DEVICES": "0,1,2,3",
                **env,
            }
            with self.subTest(env=env):
                self.run_shell("plan_lmcache_mp_servers\n", expect_rc=2, **env)

    def test_start_waits_for_every_server_and_stop_ends_them(self):
        result = self.run_shell(
            """
start_lmcache_mp_servers
lmcache_mp_servers_running
echo "CONFIG ${lmcache_mp_offload_extra_config}"
# A repeat start for the same prefill reuses the servers.
start_lmcache_mp_servers
stop_lmcache_mp_servers
! lmcache_mp_servers_running
""",
            LMCACHE_MP_STAGE_SERVERS="0:0-1:1000;1:2-3:500",
            HIP_VISIBLE_DEVICES="0,1,2,3",
            CUDA_VISIBLE_DEVICES="0,1,2,3",
        )
        # The servers start in the background, so their calls land in any order.
        servers = sorted(self.server_calls(), key=lambda call: "--port 25556" in call)
        self.assertEqual(len(servers), 2)
        for call, numa, port, prometheus, l1, devices in (
            (servers[0], "0", "25555", "29190", "1000", "0,1,2,3"),
            (servers[1], "1", "25556", "29191", "500", "0,1,2,3"),
        ):
            # Every server keeps the prefill's GPU numbering: HIP cannot open
            # an IPC handle of a GPU the importing process renumbers.
            self.assertIn(f"HIP={devices} CUDA={devices}", call)
            self.assertIn(f"/scripts/numa_exec.py {numa} python3 -m", call)
            self.assertIn(f"--port {port} ", call)
            self.assertIn(f"--prometheus-port {prometheus}", call)
            self.assertIn(f"--l1-size-gb {l1} ", call)
        self.assertIn('"lmcache.mp.stage_servers"', result.stdout)
        self.assertTrue((self.root / "logs/lmcache-mp-rank-0-s1.log").exists())
        for pid in self.stub_pids():
            self.assertFalse(self.alive(pid), pid)

    def test_one_server_dying_before_ready_stops_all(self):
        self.run_shell(
            "start_lmcache_mp_servers\n",
            expect_rc=3,
            LMCACHE_MP_STAGE_SERVERS="0:0-1:10;1:2-3:10",
            HIP_VISIBLE_DEVICES="0,1,2,3",
            STUB_DIE_PORT="25556",
        )
        self.assertEqual(len(self.server_calls()), 2)
        for pid in self.stub_pids():
            self.assertFalse(self.alive(pid), pid)

    @staticmethod
    def alive(pid):
        try:
            state = Path(f"/proc/{pid}/stat").read_text().split()[2]
        except (FileNotFoundError, IndexError):
            return False
        return state != "Z"


if __name__ == "__main__":
    unittest.main()
