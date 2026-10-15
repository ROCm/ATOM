# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only checks: VM writes and large allocations are replaced by fake steps."""

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / ".github/scripts/atomesh"
with patch.object(sys, "path", [str(SCRIPTS), *sys.path]):
    SPEC = importlib.util.spec_from_file_location(
        "host_memory_preparation", SCRIPTS / "prepare_lmcache_host_memory.py"
    )
    PREP = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(PREP)


class PreparationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.args = SimpleNamespace(
            output=self.root / "out",
            directories=[str(self.root)],
            gpus="0,1,2,3",
            per_gpu_gib=256,
            reserve_gib=128,
            attempts=3,
            timeout=120,
            step_timeout=35,
        )
        self.state = {
            "numa": {0: {"MemTotal": 1536 * PREP.GIB, "MemFree": 1400 * PREP.GIB}},
            "cgroup": [{"limit": None, "current": 0}],
        }
        self.calls = []
        self.direct_ok = False
        self.partial = 0
        self.now = 0

    def run_step(self, command, seconds, output):
        self.calls.append((command, seconds))
        text, rc = "", 0
        if command[0] == "sh":
            text, rc = ("", 0) if self.direct_ok else ("Permission denied", 1)
        elif any(str(x).endswith("numa_exec.py") for x in command):
            if self.partial:
                self.partial -= 1
                text = "[thp-compact][WARN] 1 of 1024 GiB came as huge pages"
            else:
                text = "[thp-compact] 1024 of 1024 GiB came as huge pages"
        output.write_text(text)
        return {"returncode": rc, "timed_out": False, "text": text}

    def prepare(self, run=None):
        with patch.object(PREP.budget, "plan_pins", return_value={0: 1024}):
            rc = PREP.prepare(
                self.args,
                run=run or self.run_step,
                take_snapshot=lambda planned: self.state,
                clock=lambda: self.now,
            )
        self.result = json.loads((self.args.output / "summary.json").read_text())
        return rc

    def test_direct_commands_have_no_sudo_and_stop_after_success(self):
        self.direct_ok = True
        self.assertEqual(self.prepare(), 0)
        self.assertEqual(self.result["method"], "direct_vm")
        self.assertEqual(len(self.result["attempts"]), 1)
        script = self.calls[0][0][2]
        self.assertNotIn("sudo", script)
        self.assertIn("/proc/sys/vm/drop_caches", script)
        self.assertIn("/proc/sys/vm/compact_memory", script)
        self.assertFalse(any("numa_exec.py" in str(cmd) for cmd, _ in self.calls))

    def test_permission_denial_falls_back_and_retries_partial_thp(self):
        self.partial = 1
        self.assertEqual(self.prepare(), 0)
        self.assertEqual(len(self.result["attempts"]), 2)
        self.assertEqual(self.result["method"], "file_cache_and_numa")
        self.assertEqual(sum(cmd[0] == "sh" for cmd, _ in self.calls), 1)
        compact = [cmd for cmd, _ in self.calls if "numa_exec.py" in str(cmd)]
        self.assertEqual([cmd[-1] for cmd in compact], ["1024", "1024"])
        self.assertTrue(all(timeout <= 35 for _, timeout in self.calls))

    def test_persistent_partial_thp_fails_after_three_attempts(self):
        self.partial = 10
        self.assertEqual(self.prepare(), 2)
        self.assertEqual(self.result["status"], "numa_preparation_incomplete")
        self.assertEqual(len(self.result["attempts"]), 3)

    def test_low_numa_free_or_cgroup_limit_never_starts_large_allocation(self):
        for kind in ("numa", "cgroup", "unknown_cgroup"):
            with self.subTest(kind=kind):
                if kind == "numa":
                    self.state["numa"][0]["MemFree"] = 800 * PREP.GIB
                else:
                    self.state["numa"][0]["MemFree"] = 1400 * PREP.GIB
                    self.state["cgroup"] = (
                        [{"limit": 512 * PREP.GIB, "current": 0}]
                        if kind == "cgroup"
                        else []
                    )
                self.calls.clear()
                self.assertEqual(self.prepare(), 2)
                self.assertFalse(
                    any("numa_exec.py" in str(cmd) for cmd, _ in self.calls)
                )

    def test_clean_file_cache_covers_reserve_but_dirty_pages_do_not(self):
        info = self.state["numa"][0]
        info.update(
            MemFree=1056 * PREP.GIB,
            **{"Active(file)": 72 * PREP.GIB, "Inactive(file)": 342 * PREP.GIB},
        )
        self.assertEqual(self.prepare(), 0)
        self.assertTrue(any("numa_exec.py" in str(cmd) for cmd, _ in self.calls))
        self.calls.clear()
        info.update(Dirty=330 * PREP.GIB, Writeback=10 * PREP.GIB)
        self.assertEqual(self.prepare(), 2)
        self.assertEqual(self.result["status"], "insufficient_numa_reclaimable_memory")
        self.assertFalse(any("numa_exec.py" in str(cmd) for cmd, _ in self.calls))

    def test_reclaims_cache_before_accepting_node_from_run_38745966243(self):
        info = self.state["numa"][0]
        info.update(MemFree=769 * PREP.GIB, **{"Inactive(file)": 701 * PREP.GIB})

        def reclaim(command, seconds, output):
            result = self.run_step(command, seconds, output)
            if "numa_exec.py" in str(command):
                info["MemFree"] = 1100 * PREP.GIB
                info["Inactive(file)"] = 370 * PREP.GIB
            return result

        self.assertEqual(self.prepare(reclaim), 0)
        self.assertEqual(len(self.result["attempts"]), 1)
        self.assertEqual(self.result["method"], "file_cache_and_numa")

    def test_full_thp_message_without_recovered_free_memory_is_not_ready(self):
        info = self.state["numa"][0]
        info.update(MemFree=769 * PREP.GIB, **{"Inactive(file)": 701 * PREP.GIB})
        self.assertEqual(self.prepare(), 2)
        self.assertEqual(self.result["status"], "insufficient_numa_free_memory")
        self.assertEqual(len(self.result["attempts"]), 3)
        self.assertEqual(sum("numa_exec.py" in str(cmd) for cmd, _ in self.calls), 3)

    def test_node_without_enough_reclaimable_memory_never_allocates(self):
        info = self.state["numa"][0]
        info.update(MemFree=470 * PREP.GIB, **{"Inactive(file)": 268 * PREP.GIB})
        self.assertEqual(self.prepare(), 2)
        self.assertEqual(self.result["status"], "insufficient_numa_reclaimable_memory")
        self.assertFalse(any("numa_exec.py" in str(cmd) for cmd, _ in self.calls))

    def test_agentic_prefill_requires_host_preparation(self):
        source = (SCRIPTS / "pd_server_atom.sh").read_text()
        start = source.index("start_prefill() {")
        setup = source[start : source.index("  start_mooncake_store\n", start)]
        script = (
            setup
            + """
  echo START_SERVICES
}
apply_role_env() { :; }
reset_lmcache_disk() { :; }
python3() { echo UNEXPECTED_CONTAINER_PREPARATION; return 99; }
start_prefill prefill-rank-0
"""
        )
        for enabled, prepared, rc in (
            ("true", "true", 0),
            ("true", "false", 2),
            ("false", "false", 0),
        ):
            with self.subTest(enabled=enabled, prepared=prepared):
                result = subprocess.run(
                    ["bash", "-euo", "pipefail", "-c", script],
                    env={
                        **os.environ,
                        "PREFILL_PORT": "8010",
                        "HANDSHAKE_PORT": "6301",
                        "PREFILL_DP_MASTER_PORT": "6200",
                        "PREFILL_DP_BASE_PORT": "6210",
                        "host_ip": "127.0.0.1",
                        "ATOMESH_PREPARE_LMCACHE_HOST_MEMORY": enabled,
                        "ATOMESH_LMCACHE_HOST_PREPARED": prepared,
                    },
                    text=True,
                    capture_output=True,
                    timeout=10,
                    check=False,
                )
                self.assertEqual(result.returncode, rc, result.stderr)
                self.assertNotIn("UNEXPECTED_CONTAINER_PREPARATION", result.stdout)
                self.assertEqual("START_SERVICES" in result.stdout, rc == 0)

    def test_deadline_prevents_starting_another_step(self):
        def run(*args):
            result = self.run_step(*args)
            self.now += 121
            return result

        self.assertEqual(self.prepare(run), 2)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.result["status"], "preparation_deadline")

    def test_child_that_will_not_exit_is_recorded_as_failure(self):
        def run(*args):
            raise RuntimeError("preparation child did not exit after SIGKILL")

        self.assertEqual(self.prepare(run), 2)
        self.assertIn("did not exit", self.result["error"])

    def test_real_small_child_timeout_is_reaped(self):
        result = PREP.run_step(
            [sys.executable, "-c", "import time; time.sleep(5)"],
            0.05,
            self.root / "child.log",
        )
        self.assertTrue(result["timed_out"])
        self.assertEqual(result["returncode"], -9)

    def test_cgroup_v2_includes_visible_ancestor_limit(self):
        root = self.root / "cgroup"
        child = root / "job/worker"
        child.mkdir(parents=True)
        for directory, limit in ((root, "max"), (child.parent, "100"), (child, "max")):
            (directory / "memory.max").write_text(limit)
            (directory / "memory.current").write_text("25")
        membership = self.root / "membership"
        membership.write_text("0::/job/worker\n")
        rows = PREP.cgroup_memory(root, membership)
        self.assertEqual(len(rows), 3)
        self.assertEqual(
            [row["limit"] for row in rows if row["limit"] is not None], [100]
        )


if __name__ == "__main__":
    unittest.main()
