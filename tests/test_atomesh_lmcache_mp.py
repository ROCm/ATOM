# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Exercise MP readiness, failure propagation and process cleanup without GPUs."""

import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / ".github/scripts/atomesh/pd_lmcache_mp.py"
FAKE_MP = r"""#!/usr/bin/env python3
import http.server, json, os, socket, sys, threading, time
from pathlib import Path
root = Path(os.environ["FAKE_ROOT"])
(root / "mp.pid").write_text(str(os.getpid()))
(root / "mp-argv.json").write_text(json.dumps(sys.argv[1:]))
mode = os.environ["FAKE_MODE"]
if mode == "startup-fail":
    sys.exit(7)
rpc = socket.socket()
rpc.bind(("127.0.0.1", int(sys.argv[sys.argv.index("--port") + 1])))
rpc.listen()
class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"lmcache_test_total 1\n")
    def log_message(self, *args):
        pass
http = http.server.HTTPServer(
    ("127.0.0.1", int(sys.argv[sys.argv.index("--http-port") + 1])), Handler
)
threading.Thread(target=http.serve_forever, daemon=True).start()
if mode == "runtime-fail":
    time.sleep(2.5)
    sys.exit(9)
while True:
    time.sleep(1)
"""


@unittest.skipUnless(os.name == "posix", "Linux process groups are required")
class LMCacheMPLifecycleTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        executable = self.root / "lmcache"
        executable.write_text(FAKE_MP)
        executable.chmod(0o755)
        with socket.socket() as rpc, socket.socket() as http:
            rpc.bind(("127.0.0.1", 0))
            http.bind(("127.0.0.1", 0))
            self.rpc = rpc.getsockname()[1]
            self.http = http.getsockname()[1]

    def launch(self, mode, engine_seconds=0.2):
        env = dict(
            os.environ,
            PATH=str(self.root) + os.pathsep + os.environ["PATH"],
            FAKE_ROOT=str(self.root),
            FAKE_MODE=mode,
            ATOMESH_PD_WORKER_LAYOUT="multi_node",
            ATOM_KV_OFFLOAD_EXTRA_CONFIG=json.dumps(
                {"lmcache.mp.host": "tcp://127.0.0.1", "lmcache.mp.port": self.rpc}
            ),
            LMCACHE_MP_PORT=str(self.rpc),
            LMCACHE_MP_HTTP_PORT=str(self.http),
            LMCACHE_MP_STARTUP_TIMEOUT="8",
        )
        engine = (
            "from pathlib import Path; import os,time; "
            "Path(os.environ['FAKE_ROOT'],'engine.pid').write_text(str(os.getpid())); "
            f"time.sleep({engine_seconds})"
        )
        process = subprocess.Popen(
            [
                sys.executable,
                str(WRAPPER),
                "--log-dir",
                str(self.root / "logs"),
                "--",
                sys.executable,
                "-c",
                engine,
            ],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        self.addCleanup(self.cleanup_process, process)
        return process

    @staticmethod
    def cleanup_process(process):
        if process.poll() is None:
            process.terminate()
            process.communicate(timeout=15)

    def assert_child_stopped(self, name):
        pid = int((self.root / f"{name}.pid").read_text())
        with self.assertRaises(ProcessLookupError):
            os.kill(pid, 0)

    def test_ready_server_runs_engine_collects_metrics_and_cleans_up(self):
        process = self.launch("healthy")
        output, _ = process.communicate(timeout=15)
        self.assertEqual(process.returncode, 0, output)
        self.assertIn("[lmcache-mp] ready", output)
        self.assertTrue(list((self.root / "logs").glob("metrics-*.prom")))
        args = json.loads((self.root / "mp-argv.json").read_text())
        self.assertIn("--separate-object-groups", args)
        self.assertEqual(
            args[args.index("--supported-transfer-mode") + 1], "lmcache_driven"
        )
        self.assert_child_stopped("mp")
        self.assert_child_stopped("engine")

    def test_mp_startup_failure_does_not_launch_engine(self):
        process = self.launch("startup-fail")
        output, _ = process.communicate(timeout=15)
        self.assertNotEqual(process.returncode, 0)
        self.assertIn("exited during startup", output)
        self.assertFalse((self.root / "engine.pid").exists())
        self.assert_child_stopped("mp")

    def test_occupied_port_reports_address_without_launching_children(self):
        with socket.socket() as occupied:
            occupied.bind(("127.0.0.1", self.rpc))
            occupied.listen()
            process = self.launch("healthy")
            output, _ = process.communicate(timeout=15)
        self.assertNotEqual(process.returncode, 0)
        self.assertIn(f"RPC port 127.0.0.1:{self.rpc} is unavailable", output)
        self.assertFalse((self.root / "mp.pid").exists())
        self.assertFalse((self.root / "engine.pid").exists())

    def test_mp_failure_stops_running_engine_and_fails_worker(self):
        process = self.launch("runtime-fail", 60)
        output, _ = process.communicate(timeout=15)
        self.assertNotEqual(process.returncode, 0)
        self.assertIn("exited while ATOM was running", output)
        self.assert_child_stopped("mp")
        self.assert_child_stopped("engine")

    def test_cancel_stops_both_owned_processes(self):
        process = self.launch("healthy", 60)
        deadline = time.monotonic() + 10
        while not (self.root / "engine.pid").exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertTrue((self.root / "engine.pid").exists())
        process.terminate()
        output, _ = process.communicate(timeout=15)
        self.assertEqual(process.returncode, 128 + signal.SIGTERM, output)
        self.assert_child_stopped("mp")
        self.assert_child_stopped("engine")


if __name__ == "__main__":
    unittest.main()
