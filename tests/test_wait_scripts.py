# SPDX-License-Identifier: MIT
"""CPU-only integration tests: execute the real shell scripts with fake I/O.

Run directly to avoid loading the model/GPU-oriented tests/conftest.py:
    python3 tests/test_wait_scripts.py -v
No server, container, real sleep, network connection or GPU is used.
"""

import ipaddress
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = "DeepSeek-V4.1-Flash"
MODEL_BODY = json.dumps(
    {"object": "list", "data": [{"id": MODEL, "max_model_len": 131072}]}
)
POST_OK = 'INFO: 127.0.0.1:1234 - "POST /v1/chat/completions HTTP/1.1" 200 OK\n'
NOISE = (
    "Avg prompt throughput: 0.0 tokens/s, Running: 1 reqs\n"
    "WARNING No available shared memory broadcast block\n"
    "WARNING Traceback (most recent call last):\n"
    "Engine Core: output send\n"
    'INFO: "GET /v1/models HTTP/1.1" 200 OK\n'
    'INFO: "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error\n'
    'INFO: "POST /v1/chat/completions HTTP/1.1" 2000 nope\n'
    'INFO: "POST /v1/chat/completions HTTP/1.1"\n'
)

# sleep advances a virtual poll clock and mutates only temporary fixture files.
# curl verifies that the script asks for a status code, then supplies a complete
# or deliberately bad body/status/exit code. pgrep is used only by legacy tests.
MOCK_TOOL = r"""
import json, os, pathlib, sys
root = pathlib.Path(os.environ["WAIT_TEST_ROOT"])
config = json.loads((root / "config.json").read_text())
name = pathlib.Path(sys.argv[0]).name
clock = root / "ticks"
tick = int(clock.read_text()) if clock.exists() else 0
with (root / (name + ".calls")).open("a") as stream:
    stream.write(json.dumps(sys.argv[1:]) + "\n")
if name == "sleep":
    tick += 1
    clock.write_text(str(tick))
    for dest, key in (("server.log", "server_noise"), ("client.log", "client_noise")):
        with (root / dest).open("a") as stream:
            stream.write(config.get(key, ""))
    for action in config.get("events", {}).get(str(tick), []):
        dest = root / action["path"]
        op = action["op"]
        if op == "remove":
            dest.unlink(missing_ok=True)
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open("a" if op == "append" else "w") as stream:
                stream.write(action["text"])
elif name == "curl":
    assert "-w" in sys.argv
    assert sys.argv[sys.argv.index("-w") + 1] == r"\n%{http_code}\n%{remote_ip}\n%{remote_port}"
    assert sys.argv[sys.argv.index("--noproxy") + 1] == "*"
    assert sys.argv[-1] == "http://localhost:8021/v1/models"
    if config.get("die_during_curl"):
        (root / "proc/4101/stat").unlink()
    if config.get("drop_listener_during_curl"):
        (root / "proc/4101/fd/9").unlink()
    if config.get("server_stat_during_curl"):
        (root / "proc/4101/stat").write_text(config["server_stat_during_curl"])
    sys.stdout.write("\n".join((config["body"], config["status"],
                              config.get("peer_ip", "127.0.0.1"),
                              config.get("peer_port", "8021"))))
    sys.exit(config.get("curl_rc", 0))
elif name == "pgrep":
    if not config.get("legacy"):
        raise AssertionError("explicit PID mode must not use global pgrep")
    server = "atom\\.entrypoints" in sys.argv[-1]
    until = config.get("server_live_until" if server else "client_live_until", 1)
    if tick > until:
        sys.exit(1)
    print("4101" if server else "4102 python -m lm_eval")
else:
    raise AssertionError(name)
"""


def stat_text(pid, state="S", ppid=1, starttime=1000):
    # Fields 3..52, with starttime at the real field 22 (index 19 after comm).
    # Spaces and nested closing parentheses are legal in Linux comm.
    fields = [state, str(ppid), *(["0"] * 17), str(starttime), *(["0"] * 30)]
    return f"{pid} (worker ) (with spaces)) {' '.join(fields)}\n"


def tcp_row(inode=90001, address="127.0.0.1", port=8021, state="0A"):
    packed = ipaddress.ip_address(address).packed
    encoded = "".join(
        f"{int.from_bytes(packed[i:i + 4], sys.byteorder):08X}"
        for i in range(0, len(packed), 4)
    )
    return (
        f"0: {encoded}:{port:04X} {'0' * len(encoded)}:0000 {state} "
        f"00000000:00000000 00:00000000 00000000 0 0 {inode} 1\n"
    )


class WaitScriptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="wait-script-test-")
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name)
        self.bin = self.work / "bin"
        self.bin.mkdir()
        for name in ("curl", "sleep", "pgrep"):
            path = self.bin / name
            path.write_text(f"#!{sys.executable}\n" + MOCK_TOOL)
            path.chmod(0o755)
        for name in ("server.log", "client.log"):
            (self.work / name).write_text("")
        self.set_pid(4101)
        self.set_pid(4102)
        fd = self.work / "proc/4101/fd"
        fd.mkdir()
        (fd / "1").symlink_to(self.work / "server.log")
        self.set_socket(4101)
        self.set_tcp("tcp", tcp_row())
        self.set_tcp("tcp6", "")
        self.config = {"body": MODEL_BODY, "status": "200", "events": {}}
        self.env = os.environ.copy()
        for name in (
            "SERVER_PID",
            "CLIENT_PID",
            "SERVER_LOG",
            "PROC_ROOT",
            "LOG_START_BYTES",
            "EXPECTED_MODEL_ID",
            "EXPECTED_MAX_MODEL_LEN",
        ):
            self.env.pop(name, None)
        self.env.update(
            PATH=f"{self.bin}:{self.env['PATH']}",
            WAIT_TEST_ROOT=str(self.work),
            PROC_ROOT=str(self.work / "proc"),
        )

    def set_pid(self, pid, state="S", ppid=1):
        path = self.work / f"proc/{pid}/stat"
        path.parent.mkdir(parents=True, exist_ok=True)
        if state is None:
            path.unlink(missing_ok=True)
        else:
            path.write_text(stat_text(pid, state, ppid))

    def set_socket(self, pid, inode=90001):
        path = self.work / f"proc/{pid}/fd/9"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.unlink(missing_ok=True)
        path.symlink_to(f"socket:[{inode}]")

    def set_tcp(self, table, rows):
        path = self.work / "proc/self/net" / table
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("sl local_address rem_address st tx_queue uid inode\n" + rows)

    def event(self, tick, path, text=None, op="append"):
        self.config["events"].setdefault(str(tick), []).append(
            {"path": path, "op": op, "text": text}
        )

    def run_script(self, script, args, env):
        # Reset virtual time/call history, but preserve scenario files/config.
        for name in ("ticks", "curl.calls", "sleep.calls", "pgrep.calls"):
            (self.work / name).unlink(missing_ok=True)
        (self.work / "config.json").write_text(json.dumps(self.config))
        return subprocess.run(
            ["bash", str(ROOT / "scripts" / script), *map(str, args)],
            env={**self.env, **env},
            text=True,
            capture_output=True,
            timeout=5,  # A broken loop must fail the test, not wedge the suite.
            check=False,
        )

    def ready(self, **env):
        return self.run_script(
            "wait_server_ready.sh",
            [8021, 1, 60, self.work / "server.log"],
            env,
        )

    def drain(self, *, explicit=True, startup=1, stuck=2, **env):
        options = {"SERVER_PID": "4101", "CLIENT_PID": "4102"} if explicit else {}
        return self.run_script(
            "wait_infer_drain.sh",
            [8021, 1, 1, self.work / "client.log", stuck, startup],
            {**options, **env},
        )

    def assert_exit(self, result, code):
        self.assertEqual(result.returncode, code, result.stdout + result.stderr)

    def ticks(self):
        path = self.work / "ticks"
        return int(path.read_text()) if path.exists() else 0

    def assert_no_global_probe(self):
        self.assertFalse((self.work / "pgrep.calls").exists())
        self.assertFalse((self.work / "curl.calls").exists())

    def test_ready_checks_full_models_response(self):
        # Expected entry is beyond the old head -c 60 truncation point.
        self.config["body"] = json.dumps(
            {
                "object": "list",
                "padding": "x" * 200,
                "data": [
                    {"id": "other", "max_model_len": 64},
                    {"id": MODEL, "max_model_len": 131072},
                ],
            }
        )
        result = self.ready(
            SERVER_PID="4101",
            EXPECTED_MODEL_ID=MODEL,
            EXPECTED_MAX_MODEL_LEN="131072",
        )
        self.assert_exit(result, 0)
        self.assertIn("Server READY", result.stdout)
        self.assertEqual(self.ticks(), 1)

    def test_ready_real_live_pid_without_listener_is_not_ready(self):
        # The unittest process is alive but owns no server listener. Valid mocked
        # HTTP must not manufacture an association with that unrelated process.
        result = self.ready(SERVER_PID=str(os.getpid()), PROC_ROOT="/proc")
        self.assert_exit(result, 1)
        self.assertIn("listener ownership not verified", result.stdout)
        self.assertNotIn("Server FAILED (PID", result.stdout)

    def test_ready_listener_owned_by_grandchild_or_shared_with_parent(self):
        self.set_pid(4201, ppid=4101)
        self.set_pid(4301, ppid=4201)
        self.set_socket(4301)
        # Unrelated PID disappeared during /proc enumeration: tolerate it.
        self.set_pid(9999, None)
        self.assert_exit(self.ready(SERVER_PID="4101"), 0)
        (self.work / "proc/4101/fd/9").unlink()
        self.assert_exit(self.ready(SERVER_PID="4101"), 0)

    def test_ready_rejects_listener_owned_by_unrelated_live_pid(self):
        (self.work / "proc/4101/fd/9").unlink()
        self.set_pid(9999)
        self.set_socket(9999)
        result = self.ready(SERVER_PID="4101")
        self.assert_exit(result, 1)
        self.assertIn("listener ownership not verified", result.stdout)
        self.assertNotIn("Server READY", result.stdout)

    def test_ready_zombie_or_broken_ancestry_cannot_own_listener(self):
        (self.work / "proc/4101/fd/9").unlink()
        self.set_socket(4301)
        for middle_state in (None, "Z"):
            with self.subTest(middle_state=middle_state):
                self.set_pid(4201, middle_state, ppid=4101)
                self.set_pid(4301, ppid=4201)
                self.assert_exit(self.ready(SERVER_PID="4101"), 1)
        self.set_pid(4301, "Z", ppid=4101)
        self.assert_exit(self.ready(SERVER_PID="4101"), 1)

    def test_ready_requires_listener_state_port_address_and_fd_inode(self):
        for row in (
            tcp_row(inode=12345),
            tcp_row(port=8022),
            tcp_row(state="01"),
            tcp_row(address="127.0.0.2"),
            "",
        ):
            with self.subTest(row=row):
                self.set_tcp("tcp", row)
                self.assert_exit(self.ready(SERVER_PID="4101"), 1)

    def test_ready_reuseport_listener_set_must_all_belong_to_server_tree(self):
        self.set_tcp("tcp", tcp_row() + tcp_row(inode=90002))
        self.set_pid(9999)
        self.set_socket(9999, 90002)
        self.assert_exit(self.ready(SERVER_PID="4101"), 1)
        self.set_pid(9999, ppid=4101)
        self.assert_exit(self.ready(SERVER_PID="4101"), 0)

    def test_ready_ipv4_ipv6_and_wildcard_listener_ownership(self):
        for peer, bind, table in (
            ("127.0.0.1", "0.0.0.0", "tcp"),
            ("::1", "::1", "tcp6"),
            ("::1", "::", "tcp6"),
            ("127.0.0.1", "::", "tcp6"),
            ("127.0.0.1", "::ffff:127.0.0.1", "tcp6"),
        ):
            with self.subTest(peer=peer, bind=bind):
                self.set_tcp("tcp", "")
                self.set_tcp("tcp6", "")
                self.set_tcp(table, tcp_row(address=bind))
                self.config["peer_ip"] = peer
                self.assert_exit(self.ready(SERVER_PID="4101"), 0)

    def test_ready_uses_actual_curl_peer_not_other_address_on_same_port(self):
        self.set_tcp("tcp", tcp_row())
        self.set_tcp("tcp6", tcp_row(inode=90002, address="::1"))
        self.config["peer_ip"] = "::1"
        self.assert_exit(self.ready(SERVER_PID="4101"), 1)
        self.set_pid(4201, ppid=4101)
        self.set_socket(4201, 90002)
        self.assert_exit(self.ready(SERVER_PID="4101"), 0)
        self.config["peer_port"] = "8022"
        self.assert_exit(self.ready(SERVER_PID="4101"), 1)

    def test_ready_missing_or_unreadable_network_evidence_fails_closed(self):
        table = self.work / "proc/self/net/tcp"
        table.unlink()
        self.assert_exit(self.ready(SERVER_PID="4101"), 1)
        # A directory causes an OSError even for privileged test runners.
        table.mkdir()
        self.assert_exit(self.ready(SERVER_PID="4101"), 1)
        table.rmdir()
        self.set_tcp("tcp", tcp_row())
        (self.work / "proc/4101/fd/9").unlink()
        self.assert_exit(self.ready(SERVER_PID="4101"), 1)

    def test_ready_listener_disappearing_during_curl_does_not_pass(self):
        self.config["drop_listener_during_curl"] = True
        self.assert_exit(self.ready(SERVER_PID="4101"), 1)

    def test_ready_rejects_dead_zombie_and_malformed_pid_before_sleep_or_curl(self):
        for state in (None, "Z", "X", "?"):
            with self.subTest(state=state):
                self.set_pid(4101, state)
                result = self.ready(SERVER_PID="4101")
                self.assert_exit(result, 1)
                self.assertIn("Server FAILED", result.stdout)
                self.assertEqual(self.ticks(), 0)
                self.assert_no_global_probe()
        self.assert_exit(self.ready(SERVER_PID="not-a-pid"), 1)

    def test_ready_detects_pid_death_during_sleep_and_curl(self):
        for phase in ("sleep", "curl"):
            with self.subTest(phase=phase):
                self.set_pid(4101)
                self.config["events"] = {}
                self.config["die_during_curl"] = phase == "curl"
                if phase == "sleep":
                    self.event(1, "proc/4101/stat", op="remove")
                result = self.ready(SERVER_PID="4101")
                self.assert_exit(result, 1)
                self.assertNotIn("Server READY", result.stdout)

    def test_ready_rejects_server_pid_reuse_during_sleep_or_curl(self):
        for phase in ("sleep", "curl"):
            with self.subTest(phase=phase):
                self.set_pid(4101)
                replacement = stat_text(4101, "R", starttime=2000)
                self.config["events"] = {}
                self.config.pop("server_stat_during_curl", None)
                if phase == "sleep":
                    self.event(1, "proc/4101/stat", replacement, op="write")
                else:
                    self.config["server_stat_during_curl"] = replacement
                result = self.ready(SERVER_PID="4101")
                self.assert_exit(result, 1)
                self.assertIn("Server FAILED", result.stdout)
                self.assertNotIn("Server READY", result.stdout)

    def test_ready_same_starttime_with_state_change_remains_alive(self):
        self.event(1, "proc/4101/stat", stat_text(4101, "R"), op="write")
        self.assert_exit(self.ready(SERVER_PID="4101"), 0)

    def test_pid_starttime_missing_or_malformed_fails_closed(self):
        for bad in ("4101 (worker) S 1 0\n", stat_text(4101, starttime="invalid")):
            with self.subTest(stat=bad):
                (self.work / "proc/4101/stat").write_text(bad)
                self.assert_exit(self.ready(SERVER_PID="4101"), 1)
                self.assertEqual(self.ticks(), 0)
                self.assert_exit(self.drain(), 2)
                self.assertEqual(self.ticks(), 0)

    def test_ready_rejects_http_errors_and_connection_refusal(self):
        cases = (
            ("404", '{"error":"not found"}', 0),
            ("500", MODEL_BODY, 0),
            ("302", MODEL_BODY, 0),
            ("000", "", 7),  # curl: connection refused
            ("200", MODEL_BODY, 18),  # partial transfer must not pass either
        )
        for status, body, rc in cases:
            with self.subTest(status=status, rc=rc):
                self.config.update(status=status, body=body, curl_rc=rc)
                result = self.ready()
                self.assert_exit(result, 1)
                self.assertNotIn("Server READY", result.stdout)
                self.assertEqual(self.ticks(), 1)

    def test_ready_rejects_invalid_json_or_missing_model_data(self):
        for body in (
            "<html>unavailable</html>",
            "",
            MODEL_BODY[:-1],
            MODEL_BODY + " trailing garbage",
            '{"error":"not ready"}',
            '{"data":[]}',
            '{"data":"not a list"}',
            '{"data":[{}]}',
            '{"data":[{"id":""}]}',
            "[]",
        ):
            with self.subTest(body=body):
                self.config["body"] = body
                self.assert_exit(self.ready(), 1)

    def test_ready_identity_and_length_must_match_same_entry(self):
        self.config["body"] = json.dumps(
            {
                "data": [
                    {"id": MODEL, "max_model_len": 64},
                    {"id": "other", "max_model_len": 131072},
                ]
            }
        )
        for expected in (
            {"EXPECTED_MODEL_ID": "wrong"},
            {"EXPECTED_MAX_MODEL_LEN": "128"},
            {"EXPECTED_MODEL_ID": MODEL, "EXPECTED_MAX_MODEL_LEN": "131072"},
        ):
            with self.subTest(expected=expected):
                self.assert_exit(self.ready(**expected), 1)
        self.config["body"] = json.dumps({"data": [{"id": MODEL}]})
        self.assert_exit(self.ready(EXPECTED_MAX_MODEL_LEN="131072"), 1)
        self.assert_exit(self.ready(EXPECTED_MODEL_ID=MODEL), 0)

    def test_ready_default_ignores_preexisting_log_but_zero_scans_it(self):
        (self.work / "server.log").write_text("vllm: error: bad flag\n")
        self.assert_exit(self.ready(), 0)
        result = self.ready(LOG_START_BYTES="0")
        self.assert_exit(result, 1)
        self.assertIn("errors detected", result.stdout)
        self.assertNotIn("Server READY", result.stdout)

    def test_ready_catches_appended_and_truncated_error_logs(self):
        for truncate in (False, True):
            with self.subTest(truncate=truncate):
                (self.work / "server.log").write_text("old log\n" * 100)
                self.config["events"] = {}
                self.event(
                    1,
                    "server.log",
                    "Engine core initialization failed\n",
                    op="write" if truncate else "append",
                )
                result = self.ready()
                self.assert_exit(result, 1)
                if truncate:
                    self.assertIn("log truncated", result.stdout)

    def test_ready_detects_argparse_and_engine_failures_not_warning_tracebacks(self):
        for error in (
            "vllm: error: unrecognized arguments: --bad",
            "openai_server.py: error: bad argument",
            "Engine core initialization failed",
            "EngineCore encountered an issue",
            "EngineCore encountered an error.",
            "EngineCore initialization failed",
            "EngineCore is dead",
            "EngineCore process died unexpectedly",
            "EngineDeadError: EngineCore is dead",
            "Fatal error in model execution",
        ):
            with self.subTest(error=error):
                (self.work / "server.log").write_text(error + "\n")
                self.assert_exit(self.ready(LOG_START_BYTES="0"), 1)
        (self.work / "server.log").write_text(NOISE)
        self.assert_exit(self.ready(LOG_START_BYTES="0"), 0)

    def test_enginecore_prefixed_recoverable_failures_are_not_fatal(self):
        warnings = (
            "(EngineCore_DP0 pid=123) WARNING LMCache lookup failed; recomputing\n"
            "(EngineCore_DP0 pid=123) WARNING LMCache encountered lookup miss\n"
            "(EngineCore_DP0 pid=123) WARNING transport deadline exceeded\n"
        )
        (self.work / "server.log").write_text(warnings)
        self.assert_exit(self.ready(SERVER_PID="4101", LOG_START_BYTES="0"), 0)
        self.config["server_noise"] = warnings
        result = self.drain(LOG_START_BYTES="0")
        self.assert_exit(result, 1)  # Actual no-progress hang, NOT fault exit 2.
        self.assertIn("HANG detected", result.stdout)
        self.assertEqual(self.ticks(), 2)

    def test_ready_explicit_server_log_overrides_positional_log(self):
        (self.work / "server.log").write_text("vllm: error: stale\n")
        self.assert_exit(
            self.ready(SERVER_LOG=str(self.work / "client.log"), LOG_START_BYTES="0"),
            0,
        )

    def test_drain_requires_paired_pids(self):
        for env in ({"SERVER_PID": "4101"}, {"CLIENT_PID": "4102"}):
            with self.subTest(env=env):
                self.assert_exit(self.drain(explicit=False, **env), 2)
                self.assertEqual(self.ticks(), 0)
                self.assert_no_global_probe()

    def test_drain_dead_server_is_failure_even_before_first_observation(self):
        for state in (None, "Z", "X"):
            with self.subTest(state=state):
                self.set_pid(4101, state)
                result = self.drain()
                self.assert_exit(result, 2)
                self.assertNotIn("DRAINED", result.stdout)
                self.assertEqual(self.ticks(), 0)
                self.assert_no_global_probe()

    def test_drain_server_dying_after_seen_is_not_success(self):
        self.event(2, "proc/4101/stat", stat_text(4101, "Z"), op="write")
        self.event(2, "proc/4102/stat", op="remove")
        result = self.drain()
        self.assert_exit(result, 2)
        self.assertNotIn("DRAINED", result.stdout)
        self.assertEqual(self.ticks(), 2)
        self.assert_no_global_probe()

    def test_drain_server_pid_reuse_is_failure_not_drain(self):
        self.event(
            1, "proc/4101/stat", stat_text(4101, "S", starttime=2000), op="write"
        )
        self.event(1, "proc/4102/stat", op="remove")
        result = self.drain()
        self.assert_exit(result, 2)
        self.assertNotIn("DRAINED", result.stdout)
        self.assertEqual(self.ticks(), 1)

    def test_drain_client_pid_reuse_means_original_client_exited(self):
        result_text = "INVALID: cold scored 0/16\n"
        (self.work / "client.log").write_text(result_text)
        self.event(
            1, "proc/4102/stat", stat_text(4102, "R", starttime=2000), op="write"
        )
        result = self.drain()
        self.assert_exit(result, 0)
        self.assertIn("harness must check its exit code and result", result.stdout)
        self.assertEqual(self.ticks(), 1)
        self.assertEqual((self.work / "client.log").read_text(), result_text)

    def test_drain_same_starttimes_with_state_changes_do_not_look_exited(self):
        self.event(1, "proc/4101/stat", stat_text(4101, "R"), op="write")
        self.event(1, "proc/4102/stat", stat_text(4102, "R"), op="write")
        result = self.drain()
        self.assert_exit(result, 1)
        self.assertIn("HANG detected", result.stdout)
        self.assertEqual(self.ticks(), 2)

    def test_drain_client_first_observed_later_is_also_pinned(self):
        self.set_pid(4102, None)
        self.event(1, "proc/4102/stat", stat_text(4102, starttime=5000), op="write")
        self.event(2, "proc/4102/stat", stat_text(4102, starttime=6000), op="write")
        result = self.drain()
        self.assert_exit(result, 0)
        self.assertEqual(self.ticks(), 2)

    def test_drain_never_seen_client_cannot_be_mistaken_for_completed(self):
        for state in (None, "Z"):
            with self.subTest(state=state):
                self.set_pid(4102, state)
                result = self.drain(startup=0)
                self.assert_exit(result, 4)
                self.assertIn("never observed alive", result.stdout)
                self.assertNotIn("DRAINED", result.stdout)

    def test_drain_can_observe_client_after_startup(self):
        self.set_pid(4102, None)
        self.event(2, "proc/4102/stat", stat_text(4102), op="write")
        self.event(3, "proc/4102/stat", op="remove")
        result = self.drain()
        self.assert_exit(result, 0)
        self.assertIn("waiting to observe client", result.stdout)
        self.assertEqual(self.ticks(), 3)

    def test_drain_observes_short_client_before_first_poll_and_preserves_result(self):
        result_text = "INVALID: cold scored 0/16; task is not solvable\n"
        (self.work / "client.log").write_text(result_text)
        self.event(1, "proc/4102/stat", stat_text(4102, "Z"), op="write")
        result = self.drain()
        self.assert_exit(result, 0)
        self.assertIn("harness must check its exit code and result", result.stdout)
        self.assertEqual((self.work / "client.log").read_text(), result_text)
        self.assertEqual(self.ticks(), 1)

    def test_drain_stats_warnings_get_and_error_post_do_not_hide_short_hang(self):
        self.config.update(server_noise=NOISE, client_noise="WARNING still waiting\n")
        result = self.drain()
        self.assert_exit(result, 1)
        self.assertIn("HANG detected", result.stdout)
        self.assertEqual(self.ticks(), 2)
        self.assert_no_global_probe()

    def test_drain_successful_post_resets_hang_counter(self):
        for endpoint in ("chat/completions", "completions"):
            with self.subTest(endpoint=endpoint):
                (self.work / "server.log").write_text("")
                self.config["events"] = {}
                self.event(
                    2, "server.log", POST_OK.replace("chat/completions", endpoint)
                )
                result = self.drain()
                self.assert_exit(result, 1)
                self.assertIn("http200=1 (+1)", result.stdout)
                self.assertEqual(self.ticks(), 4)

    def test_drain_arm_counts_completed_not_correct_or_duplicate_summaries(self):
        summary = "ARM cold  correct 0/16 prompt_tokens=123\n"
        self.event(1, "client.log", summary)
        self.event(2, "client.log", summary)
        result = self.drain()
        self.assert_exit(result, 1)
        self.assertIn("completed=16 (+16)", result.stdout)
        self.assertEqual(self.ticks(), 3)

    def test_drain_post_and_arm_progress_then_client_exit(self):
        self.config.update(server_noise=NOISE, client_noise="WARNING waiting\n")
        self.event(2, "server.log", POST_OK)
        self.event(4, "client.log", "ARM warm  correct 16/16 prompt_tokens=50\n")
        self.event(5, "proc/4102/stat", op="remove")
        result = self.drain()
        self.assert_exit(result, 0)
        self.assertIn("completed=16 (+16)", result.stdout)
        self.assertEqual(self.ticks(), 5)

    def test_drain_truncating_log_is_not_progress(self):
        (self.work / "server.log").write_text(POST_OK)
        self.event(2, "server.log", "WARNING log rotated\n", op="write")
        result = self.drain()
        self.assert_exit(result, 1)
        self.assertIn("http200=0 (+-1)", result.stdout)
        self.assertEqual(self.ticks(), 3)

    def test_drain_max_wait_remains_bounded_during_continuous_progress(self):
        self.config["server_noise"] = POST_OK
        result = self.run_script(
            "wait_infer_drain.sh",
            [8021, 1, 30, self.work / "client.log", 2, 1],
            {"SERVER_PID": "4101", "CLIENT_PID": "4102"},
        )
        self.assert_exit(result, 4)
        self.assertIn("MAX_WAIT", result.stdout)
        self.assertEqual(self.ticks(), 2)

    def test_drain_checks_fault_in_client_log_before_normal_client_exit(self):
        self.event(1, "client.log", "EngineDeadError: EngineCore is dead\n")
        self.event(1, "proc/4102/stat", op="remove")
        result = self.drain()
        self.assert_exit(result, 2)
        self.assertNotIn("DRAINED", result.stdout)

    def test_drain_rechecks_fatal_written_after_initial_fault_scan(self):
        # completed_questions runs after the initial scan and before client
        # liveness. Deterministically inject the late fault in precisely that gap.
        awk = self.bin / "awk"
        awk.write_text(
            f"#!{sys.executable}\n"
            "import os\n"
            "from pathlib import Path\n"
            "root = Path(os.environ['WAIT_TEST_ROOT'])\n"
            "with (root / 'server.log').open('a') as log:\n"
            "    log.write('EngineDeadError: EngineCore is dead\\n')\n"
            "(root / 'proc/4102/stat').unlink(missing_ok=True)\n"
            "print(0)\n"
        )
        awk.chmod(0o755)
        result = self.drain()
        self.assert_exit(result, 2)
        self.assertNotIn("DRAINED", result.stdout)
        self.assertEqual(self.ticks(), 1)
        self.assertIn("EngineDeadError", (self.work / "server.log").read_text())

    def test_drain_argparse_and_engine_failure_fail_while_pid_is_alive(self):
        for error in (
            "vllm: error: unrecognized arguments",
            "Engine core initialization failed",
            "EngineCore encountered an issue",
            "EngineCore encountered an error.",
            "EngineCore initialization failed",
            "EngineCore is dead",
            "EngineDeadError: EngineCore is dead",
            "Fatal error in model execution",
        ):
            with self.subTest(error=error):
                (self.work / "server.log").write_text(error + "\n")
                result = self.drain(LOG_START_BYTES="0")
                self.assert_exit(result, 2)
                self.assertNotIn("DRAINED", result.stdout)
                self.assertEqual(self.ticks(), 1)

    def test_drain_explicit_log_overrides_fd_discovery_and_honors_offset(self):
        (self.work / "server.log").write_text("vllm: error: from a different run\n")
        fresh = self.work / "fresh.log"
        fresh.write_text("vllm: error: deliberately skipped\n")
        result = self.drain(
            SERVER_LOG=str(fresh), LOG_START_BYTES=str(fresh.stat().st_size)
        )
        self.assert_exit(result, 1)  # no progress, not an old startup fault
        self.assertIn("HANG detected", result.stdout)
        self.assertEqual(self.ticks(), 2)
        self.assert_no_global_probe()

    def test_legacy_native_exit_and_startup_guard_are_unchanged(self):
        self.config.update(legacy=True, server_live_until=1)
        self.assert_exit(self.drain(explicit=False), 0)
        self.assertEqual(self.ticks(), 2)
        self.config["server_live_until"] = 0
        result = self.drain(explicit=False, startup=0)
        self.assert_exit(result, 4)
        self.assertIn("workload python never appeared", result.stdout)

    def test_legacy_native_stats_still_count_as_progress(self):
        self.config.update(
            legacy=True,
            server_live_until=3,
            client_live_until=3,
            server_noise="Avg prompt throughput: 0 tokens/s\n",
        )
        result = self.drain(explicit=False)
        self.assert_exit(result, 0)
        self.assertEqual(self.ticks(), 4)
        self.assertNotIn("HANG detected", result.stdout)


if __name__ == "__main__":
    unittest.main()
