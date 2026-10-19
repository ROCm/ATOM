# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only checks: VM writes and large allocations are replaced by fake steps."""

import errno
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
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

    def test_store_reclaims_both_numa_nodes_before_full_thp_without_vm_writes(self):
        # Failed gather CI 39203564775: each 768 GiB owner was started after
        # compaction timed out with only 280 / 303 GiB free and >1 TiB cache.
        self.args.gpus = ""
        self.args.per_gpu_gib = 0
        self.args.pins = ["0:768", "1:768"]
        self.args.skip_direct = True
        self.state["numa"] = {
            node: {
                "MemTotal": 1511 * PREP.GIB,
                "MemFree": free * PREP.GIB,
                "Inactive(file)": cache * PREP.GIB,
            }
            for node, free, cache in ((0, 280, 1185), (1, 303, 1180))
        }
        reclaimed, compacted = [], []

        def run(command, seconds, output):
            self.assertNotEqual(command[0], "sh")
            result = self.run_step(command, seconds, output)
            if "reclaim_file_cache.py" in str(command):
                node = int(command[2])
                reclaimed.append(node)
                self.state["numa"][node].update(
                    MemFree=1400 * PREP.GIB, **{"Inactive(file)": 65 * PREP.GIB}
                )
            if "compact_huge_pages.py" in str(command):
                node = int(command[2])
                self.assertIn(node, reclaimed)
                self.assertEqual(command[-1], "768.0")
                compacted.append(node)
                result["text"] = "[thp-compact] 768 of 768 GiB came as huge pages"
            return result

        # Use the real plan parser: Store owners need no GPU discovery.
        with patch.object(PREP.budget, "gpu_numa_nodes", side_effect=AssertionError):
            rc = PREP.prepare(
                self.args,
                run=run,
                take_snapshot=lambda planned: self.state,
                clock=lambda: self.now,
            )
        self.assertEqual(rc, 0)
        self.assertEqual(reclaimed, [0, 1])
        self.assertEqual(compacted, [0, 1])
        result = json.loads((self.args.output / "summary.json").read_text())
        self.assertEqual(result["planned_gib"], {"0": 768, "1": 768})
        self.assertEqual(result["method"], "file_cache_and_numa")

    def test_store_rejects_partial_thp_even_when_compaction_exits_zero(self):
        self.args.skip_direct = True
        self.partial = 10
        self.assertEqual(self.prepare(), 2)
        self.assertEqual(self.result["status"], "numa_preparation_incomplete")
        self.assertFalse(any(cmd[0] == "sh" for cmd, _ in self.calls))

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

    def test_progress_only_cannot_pass_full_thp_gate(self):
        def run(command, seconds, output):
            result = self.run_step(command, seconds, output)
            if "numa_exec.py" in str(command):
                result["text"] = (
                    '[thp-progress] {"touched_range_gib": 1024, "anon_huge_gib": 800}'
                )
            return result

        self.assertEqual(self.prepare(run), 2)
        self.assertEqual(self.result["status"], "numa_preparation_incomplete")
        self.assertEqual(len(self.result["attempts"]), 3)

    def test_file_advice_counts_success_separately_from_errors(self):
        spec = importlib.util.spec_from_file_location(
            "drop_page_cache", SCRIPTS / "drop_page_cache.py"
        )
        drop = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(drop)
        for name in ("weight1", "weight2"):
            (self.root / name).write_bytes(b"x" * 4096)
        (self.root / "broken-link").symlink_to(self.root / "missing")
        with (
            patch.object(
                drop.os,
                "posix_fadvise",
                side_effect=[None, PermissionError(errno.EACCES, "denied")],
            ),
            redirect_stdout(io.StringIO()) as log,
        ):
            result = drop.drop_page_cache(str(self.root), stats=True)
        self.assertEqual(result["files_seen"], 3)
        self.assertEqual(result["files_opened"], 2)
        self.assertEqual(result["files_advised"], 1)
        self.assertEqual(result["advised_file_bytes"], 4096)
        self.assertEqual(result["advice_errors"], 1)
        self.assertEqual(result["open_errors"], 1)
        self.assertEqual(
            {e["operation"] for e in result["error_examples"]}, {"open", "advice"}
        )
        self.assertIn('"event": "complete"', log.getvalue())
        with redirect_stdout(io.StringIO()):
            result = drop.drop_page_cache(str(self.root / "missing"), stats=True)
        self.assertEqual(result["walk_errors"], 1)
        self.assertEqual(result["files_advised"], 0)

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

    def test_reclaims_other_models_before_compacting_cache_saturated_node(self):
        self.args.cache_roots = [str(self.root)]
        info = self.state["numa"][0]
        info.update(MemFree=58 * PREP.GIB, **{"Inactive(file)": 1404 * PREP.GIB})
        broader_seen = False

        def reclaim(command, seconds, output):
            nonlocal broader_seen
            result = self.run_step(command, seconds, output)
            if "model-cache" in output.name:
                broader_seen = True
                info["MemFree"] = 1400 * PREP.GIB
                info["Inactive(file)"] = 62 * PREP.GIB
            if "numa_exec.py" in str(command):
                self.assertTrue(broader_seen)
            return result

        self.assertEqual(self.prepare(reclaim), 0)
        self.assertTrue(broader_seen)
        self.assertEqual(len(self.result["attempts"]), 1)
        self.assertIn("model_cache_delta", self.result["attempts"][0])

    def test_other_model_cache_is_untouched_when_ram_is_already_free(self):
        self.args.cache_roots = [str(self.root)]
        touched = []

        def record(command, seconds, output):
            touched.append(output.name)
            return self.run_step(command, seconds, output)

        self.assertEqual(self.prepare(record), 0)
        self.assertFalse(any("model-cache" in name for name in touched))

    def test_full_thp_message_without_recovered_free_memory_is_not_ready(self):
        info = self.state["numa"][0]
        info.update(MemFree=769 * PREP.GIB, **{"Inactive(file)": 701 * PREP.GIB})
        self.assertEqual(self.prepare(), 2)
        self.assertEqual(self.result["status"], "insufficient_numa_free_memory")
        self.assertEqual(len(self.result["attempts"]), 3)
        self.assertEqual(
            sum("compact_huge_pages.py" in str(cmd) for cmd, _ in self.calls), 3
        )

    def test_node_without_enough_reclaimable_memory_never_allocates(self):
        info = self.state["numa"][0]
        info.update(MemFree=470 * PREP.GIB, **{"Inactive(file)": 268 * PREP.GIB})
        self.assertEqual(self.prepare(), 2)
        self.assertEqual(self.result["status"], "insufficient_numa_reclaimable_memory")
        self.assertFalse(any("numa_exec.py" in str(cmd) for cmd, _ in self.calls))

    def test_base_page_plan_keeps_numa_and_cgroup_reserve(self):
        info = self.state["numa"][0]
        info.update(MemFree=748 * PREP.GIB, **{"Inactive(file)": 725 * PREP.GIB})
        self.assertEqual(PREP.base_page_reclaim_gib(self.state, 0, 1024, 128), 1345)
        self.state["cgroup"] = [{"limit": 1400 * PREP.GIB, "current": 100 * PREP.GIB}]
        self.assertEqual(PREP.base_page_reclaim_gib(self.state, 0, 1024, 128), 1172)
        info["Dirty"] = 725 * PREP.GIB
        self.assertEqual(PREP.base_page_reclaim_gib(self.state, 0, 1024, 128), 0)

    def test_fragmented_free_memory_also_needs_base_page_reclaim(self):
        info = self.state["numa"][0]
        info.update(MemFree=1156 * PREP.GIB, **{"Inactive(file)": 311 * PREP.GIB})
        self.state["buddyinfo"] = "Node 0, zone Normal 0 0 0 0 0 0 0 0 0 0 200000\n"
        self.assertEqual(PREP.base_page_reclaim_gib(self.state, 0, 1024, 128), 1339)
        self.state["buddyinfo"] = "Node 0, zone Normal 0 0 0 0 0 0 0 0 0 0 300000\n"
        self.assertEqual(PREP.base_page_reclaim_gib(self.state, 0, 1024, 128), 0)

    def test_base_page_reclaim_finishes_before_thp(self):
        info = self.state["numa"][0]
        info.update(MemFree=748 * PREP.GIB, **{"Inactive(file)": 725 * PREP.GIB})
        reclaimed = False

        def run(command, seconds, output):
            nonlocal reclaimed
            result = self.run_step(command, seconds, output)
            if "reclaim_file_cache.py" in str(command):
                reclaimed = True
                info["MemFree"] = 1350 * PREP.GIB
                info["Inactive(file)"] = 123 * PREP.GIB
            if "compact_huge_pages.py" in str(command):
                self.assertTrue(reclaimed)
            return result

        self.assertEqual(self.prepare(run), 0)
        self.assertTrue(reclaimed)
        self.assertIn("after_base_reclaim", self.result["attempts"][0])
        for command, timeout in self.calls:
            allowance = 30 if "reclaim_file_cache.py" in str(command) else 0
            self.assertLessEqual(timeout, self.args.step_timeout + allowance)
            self.assertLessEqual(timeout, self.args.timeout)

    def test_failed_base_page_reclaim_never_starts_thp(self):
        info = self.state["numa"][0]
        info.update(MemFree=748 * PREP.GIB, **{"Inactive(file)": 725 * PREP.GIB})

        def run(command, seconds, output):
            result = self.run_step(command, seconds, output)
            if "reclaim_file_cache.py" in str(command):
                result.update(returncode=-9, timed_out=True)
            return result

        self.assertEqual(self.prepare(run), 2)
        self.assertEqual(self.result["status"], "base_page_reclaim_failed")
        self.assertEqual(len(self.result["attempts"]), 3)
        self.assertFalse(
            any("compact_huge_pages.py" in str(cmd) for cmd, _ in self.calls)
        )

    def test_base_page_release_allowance_cannot_exceed_total_deadline(self):
        self.args.timeout = 50
        info = self.state["numa"][0]
        info.update(MemFree=748 * PREP.GIB, **{"Inactive(file)": 725 * PREP.GIB})
        self.assertEqual(self.prepare(), 2)  # Fake allocation does not recover RAM.
        limits = [
            seconds
            for cmd, seconds in self.calls
            if "reclaim_file_cache.py" in str(cmd)
        ]
        self.assertEqual(limits, [50])

    def test_small_base_page_allocation_is_resident_then_released(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPTS / "reclaim_file_cache.py"), "0.0078125"],
            text=True,
            capture_output=True,
            timeout=15,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        rows = [
            json.loads(line.split(" ", 1)[1]) for line in result.stdout.splitlines()
        ]
        populated = next(row for row in rows if row["event"] == "populated")
        self.assertEqual(populated["populated_gib"], 0.0078125)
        self.assertGreaterEqual(populated["rss_gib"], 0.0078125)
        self.assertEqual(populated["anon_huge_gib"], 0)
        self.assertEqual(rows[-1]["event"], "released")

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
            [
                sys.executable,
                "-c",
                "import time; print('progress before timeout', flush=True); time.sleep(5)",
            ],
            0.05,
            self.root / "child.log",
        )
        self.assertTrue(result["timed_out"])
        self.assertEqual(result["returncode"], -9)
        self.assertIn("progress before timeout", result["text"])
        self.assertTrue(result["heartbeats"])

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
