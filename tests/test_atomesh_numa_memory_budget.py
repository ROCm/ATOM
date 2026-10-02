# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The launcher's per-NUMA-node check of the memory the Mooncake L2 pins.

sysfs is a temporary tree shaped like pit2-p03-g40: two CPU nodes first in the
KFD topology, then eight GPUs in KFD order, the first four on NUMA node 0.
"""

import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / ".github/scripts/atomesh/numa_memory_budget.py"
)
SPEC = importlib.util.spec_from_file_location("numa_memory_budget", SCRIPT)
BUDGET = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUDGET)

GIB_KB = 1024 * 1024
# KFD order of the GPUs' PCI buses, and their NUMA nodes.
GPU_BUSES = ["75", "05", "65", "15", "f5", "85", "e5", "95"]


class NumaMemoryBudgetTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.sysfs = Path(temporary.name)
        topology = self.sysfs / "class/kfd/kfd/topology/nodes"
        for index in range(2):
            node = topology / str(index)
            node.mkdir(parents=True)
            (node / "properties").write_text(
                "cpu_cores_count 128\nsimd_count 0\nlocation_id 0\ndomain 0\n"
            )
        for index, bus in enumerate(GPU_BUSES):
            node = topology / str(index + 2)
            node.mkdir(parents=True)
            (node / "properties").write_text(
                f"simd_count 1024\nlocation_id {int(bus, 16) << 8}\ndomain 0\n"
            )
            device = self.sysfs / f"bus/pci/devices/0000:{bus}:00.0"
            device.mkdir(parents=True)
            (device / "numa_node").write_text(f"{index // 4}\n")
        self.meminfo(0, total_gib=1511, free_gib=800, cache_gib=675)
        self.meminfo(1, total_gib=1511, free_gib=786, cache_gib=711)

    def meminfo(self, node, *, total_gib, free_gib, cache_gib):
        directory = self.sysfs / f"devices/system/node/node{node}"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "meminfo").write_text(
            f"Node {node} MemTotal:       {total_gib * GIB_KB} kB\n"
            f"Node {node} MemFree:        {free_gib * GIB_KB} kB\n"
            f"Node {node} FilePages:      {cache_gib * GIB_KB} kB\n"
            f"Node {node} HugePages_Total:     0\n"
        )

    def run_script(self, *args):
        return subprocess.run(
            [sys.executable, str(SCRIPT), "--sysfs", str(self.sysfs), *args],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

    def test_gpus_are_numbered_in_kfd_order(self):
        self.assertEqual(
            BUDGET.kfd_gpu_pci_addresses(self.sysfs)[:2],
            ["0000:75:00.0", "0000:05:00.0"],
        )
        self.assertEqual(BUDGET.gpu_numa_nodes(self.sysfs, [0, 3, 4, 7]), [0, 0, 1, 1])
        with self.assertRaises(SystemExit):
            BUDGET.gpu_numa_nodes(self.sysfs, [8])
        (self.sysfs / "bus/pci/devices/0000:65:00.0/numa_node").unlink()
        with self.assertRaisesRegex(SystemExit, "GPU 2's NUMA node"):
            BUDGET.gpu_numa_nodes(self.sysfs, [2])

    def test_owners_and_the_l1s_add_up_per_node(self):
        planned = BUDGET.plan_pins(["0:768", "1:960"], "0,1,2,3", 48, self.sysfs)
        self.assertEqual(planned, {0: 768 + 4 * 48, 1: 960})
        self.assertEqual(BUDGET.plan_pins(["0:8"], "", 48, self.sysfs), {0: 8})
        with self.assertRaises(SystemExit):
            BUDGET.plan_pins(["x:8"], "", 0, self.sysfs)

    def test_pins_beyond_free_memory_warn_and_beyond_the_node_fail(self):
        # pit2-p03-g40 with Kimi's weights cached: NUMA1 has 786 GiB free.
        l1 = ("--reserve-gib", "128", "--gpus", "0,1,2,3", "--per-gpu-gib")
        result = self.run_script(*l1, "48", "0:768", "1:960")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("NUMA node 0: pins 960 GiB", result.stderr)
        self.assertIn("[numa-budget][WARN] NUMA node 1: pins 960 GiB", result.stderr)
        self.assertIn("786 GiB free, 711 GiB page cache", result.stderr)
        # An L1 base still at 256 GiB cannot fit next to owner0.
        result = self.run_script(*l1, "256", "0:768", "1:960")
        self.assertEqual(result.returncode, 2)
        self.assertIn("[numa-budget][FAIL] NUMA node 0: pins 1792 GiB", result.stderr)

    def test_a_plan_that_fits_free_memory_passes_quietly(self):
        result = self.run_script("--reserve-gib", "128", "0:64", "1:64")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        self.assertIn("[numa-budget] NUMA node 1: pins 64 GiB", result.stdout)

    def test_a_missing_node_fails(self):
        result = self.run_script("2:8")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("cannot read NUMA node 2", result.stderr)


if __name__ == "__main__":
    unittest.main()
