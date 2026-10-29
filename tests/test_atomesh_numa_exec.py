# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""numa_exec.py CPU binding when the caller is pinned to another NUMA node."""

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SCRIPT = Path(__file__).resolve().parents[1] / ".github/scripts/atomesh/numa_exec.py"
SPEC = importlib.util.spec_from_file_location("numa_exec", SCRIPT)
NUMA = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(NUMA)

NODE_CPUS = {0: set(range(64)), 1: set(range(64, 128))}


class BindCpusTest(unittest.TestCase):
    def bind(self, node, cgroup_text, caller_affinity, init_status=None):
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            if cgroup_text is not None:
                path = Path(tmp) / "cpuset.cpus.effective"
                path.write_text(cgroup_text)
                paths.append(str(path))
            paths.append(str(Path(tmp) / "missing"))
            status = Path(tmp) / "status"
            if init_status is not None:
                status.write_text(init_status)
            with (
                mock.patch.object(NUMA, "CGROUP_CPUSETS", tuple(paths)),
                mock.patch.object(NUMA, "INIT_STATUS", str(status)),
                mock.patch.object(NUMA, "node_cpus", NODE_CPUS.__getitem__),
                mock.patch.object(
                    NUMA.os, "sched_getaffinity", return_value=caller_affinity
                ),
                mock.patch.object(NUMA.os, "sched_setaffinity") as setaff,
            ):
                NUMA.bind_cpus(node)
        return setaff.call_args.args[1]

    def test_caller_pinned_to_node_1_still_binds_node_0(self):
        self.assertEqual(self.bind(0, "0-127\n", NODE_CPUS[1]), NODE_CPUS[0])

    def test_container_cpuset_still_limits_the_node(self):
        self.assertEqual(self.bind(0, "0-31,64-127\n", NODE_CPUS[1]), set(range(32)))

    def test_node_outside_container_cpuset_is_refused(self):
        with self.assertRaises(SystemExit) as raised:
            self.bind(0, "64-127\n", NODE_CPUS[1])
        self.assertIn("no CPU of NUMA node 0", str(raised.exception))

    def test_without_cgroup_cpuset_uses_container_init_cpus(self):
        status = "Name:\tbash\nCpus_allowed_list:\t0-127\n"
        self.assertEqual(self.bind(0, None, NODE_CPUS[1], status), NODE_CPUS[0])

    def test_without_cgroup_cpuset_falls_back_to_affinity(self):
        self.assertEqual(self.bind(1, None, set(range(64, 96))), set(range(64, 96)))


if __name__ == "__main__":
    unittest.main()
