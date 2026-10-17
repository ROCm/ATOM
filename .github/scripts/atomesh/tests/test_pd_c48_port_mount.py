"""Check which pinned C48 overlays each case mounts."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[4]
SCRIPT = REPO / ".github/scripts/atomesh/pd_c48_port_mount.sh"
MANAGER = "/vllm/v1/simple_kv_offload/manager.py,readonly"


def mounts(case, root=REPO):
    env = dict(os.environ, ATOMESH_ENV_ATOMESH_C48_CASE=case)
    proc = subprocess.run(
        ["bash", "-c", f'docker_args=(); source "{SCRIPT}"; c48_append_port_mount "$1"; '
         'rc=$?; printf "%s\\n" "${docker_args[@]}"; exit $rc', "_", str(root)],
        env=env, capture_output=True, text=True,
    )
    return proc.returncode, [a for a in proc.stdout.splitlines() if a != "--mount"]


class PortMountTest(unittest.TestCase):
    def test_manager_overlay_only_for_cpuref(self):
        rc, base = mounts("V-QD")
        self.assertEqual((rc, len(base)), (0, 3))
        self.assertFalse(any(a.endswith(MANAGER) for a in base))
        rc, cpuref = mounts("V-QD-CPUREF")
        self.assertEqual(rc, 0)
        self.assertEqual(cpuref[:3], base)
        self.assertTrue(cpuref[3].endswith(MANAGER))

    def test_cpuref_rejects_changed_manager(self):
        with tempfile.TemporaryDirectory() as temp:
            gap = Path(temp) / ".github/benchmark/c48-model-gap"
            shutil.copytree(REPO / ".github/benchmark/c48-model-gap", gap)
            with open(gap / "cpu-lru-refresh/manager.py", "a") as f:
                f.write("\n")
            self.assertEqual(mounts("V-QD", temp)[0], 0)
            self.assertEqual(mounts("V-QD-CPUREF", temp)[0], 2)


if __name__ == "__main__":
    unittest.main()
