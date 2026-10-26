# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""HugeTLB lifecycle on fake sysfs; never allocate memory or write host VM state."""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / ".github/scripts/atomesh"
with patch.object(sys, "path", [str(SCRIPTS), *sys.path]):
    SPEC = importlib.util.spec_from_file_location(
        "lmcache_hugetlb_test", SCRIPTS / "lmcache_hugetlb.py"
    )
    HP = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(HP)


class HugeTLBTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.output = self.root / "output"
        self.pool = HP.pool_path(self.root, 0)
        self.pool.mkdir(parents=True)
        self.original = 368640  # 720 GiB, matching run 38738296856.
        self.set_pool(self.original)
        self.reserved = (
            self.root / "kernel/mm/hugepages/hugepages-2048kB/resv_hugepages"
        )
        self.reserved.parent.mkdir(parents=True)
        self.reserved.write_text("0")
        self.memory = {"MemFree": 470 * HP.GIB, "MemTotal": 1511 * HP.GIB}
        self.calls = []
        self.patch = patch.object(
            HP.budget, "node_meminfo", side_effect=lambda *args: self.memory.copy()
        )
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def set_pool(self, count):
        for name, value in (
            ("nr_hugepages", count),
            ("free_hugepages", count),
            ("surplus_hugepages", 0),
        ):
            (self.pool / name).write_text(str(value))

    def change(self, path, count, log):
        self.calls.append(count)
        previous = HP.read_pool(path)["nr_hugepages"]
        self.set_pool(count)
        self.memory["MemFree"] += (previous - count) * HP.PAGE_BYTES

    def prepare(self, change=None):
        HP.prepare(
            self.output,
            "8537",
            {0: 1024},
            sysfs=self.root,
            change=change or self.change,
        )

    def restore(self, change=None, job="8537"):
        HP.restore(self.output, job, sysfs=self.root, change=change or self.change)

    def test_release_only_shortfall_and_restore_original_pool(self):
        self.prepare()
        self.assertEqual(self.memory["MemFree"], 1152 * HP.GIB)
        self.assertGreater(HP.read_pool(self.pool)["nr_hugepages"], 0)
        self.restore()
        self.assertEqual(HP.read_pool(self.pool)["nr_hugepages"], self.original)
        self.assertEqual(self.memory["MemFree"], 470 * HP.GIB)
        self.restore()  # Idempotent after success.
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(
            json.loads((self.output / "hugetlb.json").read_text())["restore_status"],
            "restored",
        )

    def test_no_shrink_when_normal_memory_is_enough(self):
        self.memory["MemFree"] = 1200 * HP.GIB
        self.prepare()
        self.restore()
        self.assertEqual(self.calls, [])

    def test_in_use_pool_or_global_reservation_prevents_write(self):
        (self.pool / "free_hugepages").write_text(str(self.original - 1))
        with self.assertRaisesRegex(RuntimeError, "in use or reserved"):
            self.prepare()
        self.assertEqual(self.calls, [])
        (self.output / "hugetlb.json").unlink()
        self.set_pool(self.original)
        self.reserved.write_text("1")
        with self.assertRaisesRegex(RuntimeError, "in use or reserved"):
            self.prepare()
        self.assertEqual(self.calls, [])

    def test_permission_failure_preserves_pool_and_recovery_record(self):
        def deny(*args):
            raise RuntimeError("HugeTLB write failed: Permission denied")

        with self.assertRaisesRegex(RuntimeError, "Permission denied"):
            self.prepare(deny)
        self.restore()
        self.assertEqual(HP.read_pool(self.pool)["nr_hugepages"], self.original)
        self.assertEqual(self.calls, [])

    def test_noop_write_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "did not reach"):
            self.prepare(lambda *args: None)
        self.restore()
        self.assertEqual(self.calls, [])

    def test_no_memory_recovery_is_rejected_and_pool_restored(self):
        def no_recovery(path, count, log):
            self.set_pool(count)

        with self.assertRaisesRegex(RuntimeError, "did not recover"):
            self.prepare(no_recovery)
        self.restore()
        self.assertEqual(HP.read_pool(self.pool)["nr_hugepages"], self.original)

    def test_restore_failure_is_not_reported_as_success(self):
        self.prepare()
        with self.assertRaisesRegex(RuntimeError, "restore incomplete"):
            self.restore(lambda *args: None)
        self.assertEqual(
            json.loads((self.output / "hugetlb.json").read_text())["restore_status"],
            "failed",
        )

    def test_unreaped_writer_blocks_recovery_and_repeated_prepare(self):
        def unkillable(*args):
            raise RuntimeError("preparation child 123 did not exit after SIGKILL")

        with self.assertRaisesRegex(RuntimeError, "did not exit"):
            self.prepare(unkillable)
        with self.assertRaisesRegex(RuntimeError, "may still be running"):
            self.restore()
        with self.assertRaisesRegex(RuntimeError, "already exists"):
            self.prepare()
        self.assertEqual(self.calls, [])

    def test_bounded_writer_executes_only_against_fake_sysfs(self):
        HP.change_pool(self.pool, 123, self.root / "writer.log")
        self.assertEqual((self.pool / "nr_hugepages").read_text(), "123\n")

    def test_other_job_cannot_restore(self):
        self.prepare()
        with self.assertRaisesRegex(RuntimeError, "does not belong"):
            self.restore(job="999")
        self.assertEqual(len(self.calls), 1)


if __name__ == "__main__":
    unittest.main()
