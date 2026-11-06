"""Standalone-store owners: one per NIC on its NUMA node, and a failed mount stops the launch."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import store_owners


class StoreOwnersTest(unittest.TestCase):
    def test_plan_and_failed_mount(self):
        case = {"mooncake": {"store_owners": {"devices": ["rdma0", "rdma4"], "segment_size": 1 << 30,
                                              "env": {"MC_MAX_MR_SIZE": "68719476736"}}},
                "roles": {"prefill": {"env": {"MC_IB_TC": "104", "MC_MAX_MR_SIZE": "1073741824",
                                              "GLIBC_TUNABLES": "glibc.malloc.hugetlb=1", "VLLM_X": "1"}}}}
        with tempfile.TemporaryDirectory() as tmp:
            for dev, numa in (("rdma0", 0), ("rdma4", 1)):
                (Path(tmp) / dev / "device").mkdir(parents=True)
                (Path(tmp) / dev / "device" / "numa_node").write_text(f"{numa}\n")
            real_path = store_owners.Path
            fake_path = lambda p: real_path(str(p).replace("/sys/class/infiniband", tmp))
            with mock.patch.object(store_owners, "Path", fake_path):
                plan = store_owners.owner_plan(case)
            env = {"MC_IB_TC": "104", "MC_MAX_MR_SIZE": "68719476736", "GLIBC_TUNABLES": "glibc.malloc.hugetlb=1",
                   "HIP_VISIBLE_DEVICES": "-1", "CUDA_VISIBLE_DEVICES": "-1"}
            self.assertEqual(plan, [("rdma0", 0, 1 << 30, env), ("rdma4", 1, 1 << 30, env)])

            client = Path(tmp) / "client.sh"
            # Still running: the failure line alone must stop the launch, and the owners with it.
            client.write_text("#!/bin/sh\necho 'E1106 Failed to mount segment: INVALID_PARAMS'\nexec sleep 60\n")
            client.chmod(0o755)
            case_path = Path(tmp) / "case.json"
            case_path.write_text(json.dumps(case))
            real_popen, procs = subprocess.Popen, []
            # Skip numa_exec.py: binding needs CAP_SYS_NICE.
            popen = lambda cmd, **kw: procs.append(real_popen(cmd[3:], **kw)) or procs[-1]
            with mock.patch.object(store_owners, "Path", fake_path), \
                    mock.patch.object(store_owners.subprocess, "Popen", popen), \
                    self.assertRaises(SystemExit) as stop:
                store_owners.main([str(case_path), str(client), "127.0.0.1", "1", "1", "1", tmp])
            self.assertRegex(str(stop.exception.code), r"owner rdma[04] \(rc=None\)")
            self.assertIn("Failed to mount segment", str(stop.exception.code))
            self.assertEqual([p.returncode for p in procs], [-9, -9])


if __name__ == "__main__":
    unittest.main()
