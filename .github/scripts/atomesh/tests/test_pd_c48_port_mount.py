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
CPUREF = {
    "V-QD-CPUREF": "cpu-lru-refresh",
    "V-QD-CPUREF2": "cpu-lru-refresh-v2",
    "V-QD-CPUREF2-LPT3072": "cpu-lru-refresh-v2",
    "V-QD-CPUREF2-LPT6144": "cpu-lru-refresh-v2",
    "V-QD-CPUREF2-FP8ATTN": "cpu-lru-refresh-v2",
}
FP8ATTN = "/vllm/v1/attention/backends/mla/prefill/aiter_flash_attn.py,readonly"


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
        self.assertEqual(mounts("V-QD-LPT3072"), (0, base))
        for case, fix_dir in CPUREF.items():
            rc, cpuref = mounts(case)
            self.assertEqual(rc, 0)
            self.assertEqual(cpuref[:3], base)
            self.assertTrue(cpuref[3].endswith(MANAGER))
            self.assertIn(f"/c48-model-gap/{fix_dir}/manager.py,", cpuref[3])
            fp8 = case.endswith("-FP8ATTN")
            self.assertEqual(len(cpuref), 5 if fp8 else 4)
            if fp8:
                self.assertTrue(cpuref[4].endswith(FP8ATTN))
                self.assertIn("/c48-model-gap/mla-prefill-fp8/", cpuref[4])

    def test_fp8attn_rejects_changed_overlay(self):
        with tempfile.TemporaryDirectory() as temp:
            gap = Path(temp) / ".github/benchmark/c48-model-gap"
            shutil.copytree(REPO / ".github/benchmark/c48-model-gap", gap)
            with open(gap / "mla-prefill-fp8/aiter_flash_attn.py", "a") as f:
                f.write("\n")
            self.assertEqual(mounts("V-QD-CPUREF2", temp)[0], 0)
            self.assertEqual(mounts("V-QD-CPUREF2-FP8ATTN", temp)[0], 2)

    def test_cpuref_rejects_changed_manager(self):
        for case, fix_dir in CPUREF.items():
            with tempfile.TemporaryDirectory() as temp:
                gap = Path(temp) / ".github/benchmark/c48-model-gap"
                shutil.copytree(REPO / ".github/benchmark/c48-model-gap", gap)
                with open(gap / fix_dir / "manager.py", "a") as f:
                    f.write("\n")
                self.assertEqual(mounts("V-QD", temp)[0], 0)
                self.assertEqual(mounts(case, temp)[0], 2)


if __name__ == "__main__":
    unittest.main()
