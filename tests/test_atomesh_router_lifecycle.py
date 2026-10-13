# SPDX-License-Identifier: MIT
"""Exercise the real sidecar launcher against detached Docker failure states."""

import http.server
import os
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / ".github/scripts/atomesh/pd_slurm_job.sh"


class Health(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()

    def log_message(self, *args):
        pass


class RouterLifecycleTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.directory = Path(temp.name)
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Health)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.stop_http)
        self.docker = self.directory / "docker"
        self.docker.write_text("""#!/usr/bin/env python3
import json,os,signal,socket,sys,time
from pathlib import Path
args=sys.argv[1:]
root=Path(os.environ['TEST_DIRECTORY'])
mode=os.environ['TEST_MODE']
if args[0]=='run':
    if mode=='exit_during_workload':
        (root/'workload_pid').write_text(str(os.getpid()))
        time.sleep(30)
        sys.exit(0)
    if mode=='bind_failure':
        with socket.socket() as s:
            try:
                s.bind(('127.0.0.1',int(os.environ['ROUTER_PORT'])))
            except OSError as e:
                assert e.errno==98
                (root/'child.log').write_text(str(e))
            else:
                raise AssertionError('Expected real occupied-port failure')
    else:
        (root/'child.log').write_text('router started')
elif args[0]=='inspect':
    counter=root/'inspections'
    count=int(counter.read_text())+1 if counter.exists() else 1
    counter.write_text(str(count))
    if mode=='inspect_error':
        sys.exit(1)
    running=mode not in ('bind_failure','exited','exit_during_workload') and not (mode=='exit_after_http' and count>1)
    print(json.dumps({'Running':running,'ExitCode':0 if running else 1}) if 'json' in args[2] else str(running).lower())
elif args[0]=='logs':
    print((root/'child.log').read_text())
elif args[0]=='rm' and mode=='exit_during_workload':
    pid=root/'workload_pid'
    if pid.exists():
        os.kill(int(pid.read_text()),signal.SIGTERM)
elif args[0] not in ('pull','kill','rm'):
    raise AssertionError(args)
""")
        self.docker.chmod(0o755)

    def stop_http(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def launch(self, mode):
        source = SCRIPT.read_text()
        setup = source.split("# vLLM workers self-register", 1)[1].split(
            "\npeer_state()", 1
        )[0]
        shell = (
            "set -euo pipefail\nbounded_docker_rm() { :; }\n# vLLM workers self-register"
            + setup
        )
        shell += '\nstart_vllm_router owned-router "$TEST_DIRECTORY/router.log" 0\n'
        shell += "echo MODEL_START_ALLOWED\n"
        env = dict(os.environ)
        env.update(
            PATH=str(self.directory) + os.pathsep + env["PATH"],
            TEST_DIRECTORY=str(self.directory),
            TEST_MODE=mode,
            ROUTER_PORT=str(self.server.server_port),
            ROUTER_POLICY="consistent_hash",
            ATOMESH_VLLM_ROUTER_IMAGE="test-only",
            ATOMESH_VLLM_ROUTER_DISCOVERY_PORT="36367",
        )
        return subprocess.run(
            ["bash", "-c", shell],
            env=env,
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )

    def assert_rejected(self, mode):
        result = self.launch(mode)
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn("MODEL_START_ALLOWED", result.stdout)

    def test_real_bind_failure_not_masked_by_another_healthy_listener(self):
        self.assert_rejected("bind_failure")
        self.assertIn(
            "Address already in use", (self.directory / "child.log").read_text()
        )

    def test_exited_container_rejected(self):
        self.assert_rejected("exited")

    def test_failed_container_query_rejected(self):
        self.assert_rejected("inspect_error")

    def test_container_exit_during_http_probe_rejected(self):
        self.assert_rejected("exit_after_http")

    def test_live_container_with_liveness_response_accepted(self):
        result = self.launch("running")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("MODEL_START_ALLOWED", result.stdout)

    def test_sidecar_exit_stops_running_workload(self):
        source = SCRIPT.read_text()
        helper = (
            "vllm_router_running() {"
            + source.split("vllm_router_running() {", 1)[1].split("\n}", 1)[0]
            + "\n}\n"
        )
        monitor = (
            "  local docker_rc"
            + source.split("  local docker_rc\n", 1)[1].split(
                '\n  return "${docker_rc}"', 1
            )[0]
        )
        shell = "set -euo pipefail\n" + helper
        shell += """
peer_state() { return 1; }
bounded_docker_rm() { :; }
monitor() {
  local container=owned-workload router_container=owned-router
  local rank=0 execution_phase=combined phase_suffix= container_log=container.log
  local rank_dir="$TEST_DIRECTORY" RUN_DIR="$TEST_DIRECTORY"
  local -a docker_args=(run)
"""
        shell += monitor + '\nreturn "$docker_rc"\n}\nmonitor\n'
        result = subprocess.run(
            ["bash", "-c", shell],
            env={
                **os.environ,
                "PATH": str(self.directory) + os.pathsep + os.environ["PATH"],
                "TEST_MODE": "exit_during_workload",
                "TEST_DIRECTORY": str(self.directory),
            },
            capture_output=True,
            text=True,
            timeout=12,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("[router-failure]", result.stderr)
        self.assertIn("exited rc=1", result.stdout)


if __name__ == "__main__":
    unittest.main()
