# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Run atom-test.yaml's `GPU host health check` step against rocm-smi tables CI
hosts printed, through a stand-in rocm-smi, without GPUs.

The tables are real: a healthy host, a host with one GPU whose SMU does not
answer (every job on it hung on that GPU), and a host whose GPUs all lack a
junction temperature and two of which report no power (jobs hung on those two).
"""

import os
import shutil
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

HEADER = """\
============================================= ROCm System Management Interface =============================================
======================================================= Concise Info =======================================================
Device  Node  IDs              Temp        Power     Partitions          SCLK    MCLK     Fan  Perf  PwrCap   VRAM%  GPU%
              (DID,     GUID)  (Junction)  (Socket)  (Mem, Compute, ID)
============================================================================================================================
"""
FOOTER = "=" * 124 + "\n"

HEALTHY = HEADER + """\
0       2     0x75b0,   29921  54.0°C      260.0W    NPS1, SPX, 0        149Mhz  2000Mhz  0%   auto  1000.0W  0%     0%
1       3     0x75b0,   45798  57.0°C      266.0W    NPS1, SPX, 0        150Mhz  2000Mhz  0%   auto  1000.0W  0%     0%
2       4     0x75b0,   47340  52.0°C      262.0W    NPS1, SPX, 0        149Mhz  2000Mhz  0%   auto  1000.0W  0%     0%
3       5     0x75b0,   32491  61.0°C      269.0W    NPS1, SPX, 0        150Mhz  2000Mhz  0%   auto  1000.0W  0%     0%
""" + FOOTER

ONE_DEAD_GPU = HEADER + """\
0       2     0x75b0,   29921  54.0°C      259.0W    NPS1, SPX, 0        169Mhz  2000Mhz  0%   auto  1000.0W  0%     0%
1       3     0x75b0,   45798  61.0°C      264.0W    NPS1, SPX, 0        173Mhz  2000Mhz  0%   auto  1000.0W  0%     0%
2       4     0x75b0,   47340  55.0°C      260.0W    NPS1, SPX, 0        174Mhz  2000Mhz  0%   auto  1000.0W  0%     0%
3       5     0x75b0,   32491  N/A         N/A       NPS1, SPX, 0        N/A     N/A      0%   auto  N/A      0%     0%
""" + FOOTER

# Edge, not Junction: none of this host's GPUs reports a junction temperature.
EDGE_HOST = """\
========================================== ROCm System Management Interface ==========================================
==================================================== Concise Info ====================================================
Device  Node  IDs              Temp    Power   Partitions          SCLK    MCLK     Fan  Perf  PwrCap   VRAM%  GPU%
              (DID,     GUID)  (Edge)  (Avg)   (Mem, Compute, ID)
======================================================================================================================
0       2     0x75b0,   29921  N/A     N/A     NPS1, SPX, 0        N/A     N/A      0%   auto  N/A      0%     0%
1       3     0x75b0,   45798  N/A     265.0W  NPS1, SPX, 0        155Mhz  2000Mhz  0%   auto  1000.0W  0%     0%
2       4     0x75b0,   47340  N/A     260.0W  NPS1, SPX, 0        154Mhz  2000Mhz  0%   auto  1000.0W  0%     0%
3       5     0x75b0,   32491  N/A     268.0W  NPS1, SPX, 0        154Mhz  2000Mhz  0%   auto  1000.0W  0%     0%
4       6     0x75b0,   44280  N/A     262.0W  NPS1, SPX, 0        155Mhz  2000Mhz  0%   auto  1000.0W  0%     0%
5       7     0x75b0,   27391  N/A     N/A     NPS1, SPX, 0        N/A     N/A      0%   auto  N/A      0%     0%
======================================================================================================================
"""

UNIQUE_IDS = "".join(f"GPU[{i}]\t\t: Unique ID: 0x{i:016x}\n" for i in range(8))


class GpuHostHealthCheckTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        workflow = yaml.safe_load(
            (ROOT / ".github/workflows/atom-test.yaml").read_text()
        )
        steps = workflow["jobs"]["atom-test"]["steps"]
        # It runs before anything else touches the host.
        assert steps[0].get("id") == "gpu_host_health", steps[0].get("name")
        cls.script = steps[0]["run"]

    def run_step(self, table=None, *, smi_exit=0):
        """Run the step with a stand-in rocm-smi printing ``table``; None: no
        rocm-smi on the host."""
        with tempfile.TemporaryDirectory() as directory:
            bin_dir = Path(directory)
            # Only the tools the step uses, so a real rocm-smi cannot leak in.
            for tool in ("awk", "grep", "sed", "wc", "timeout", "hostname"):
                target = shutil.which(tool)
                if target:
                    (bin_dir / tool).symlink_to(target)
            if table is not None:
                smi = bin_dir / "rocm-smi"
                cat = shutil.which("cat")
                (bin_dir / "table").write_text(table)
                (bin_dir / "ids").write_text(UNIQUE_IDS)
                smi.write_text(
                    "#!/bin/bash\n"
                    f'[ "${{1:-}}" = "--showuniqueid" ] && exec {cat} "{bin_dir}/ids"\n'
                    f'{cat} "{bin_dir}/table"\n'
                    f"exit {smi_exit}\n"
                )
                smi.chmod(smi.stat().st_mode | stat.S_IEXEC)
            env = dict(os.environ, PATH=str(bin_dir), RUNNER_NAME="runner-under-test")
            # GitHub runs `run:` with `bash -e {0}`.
            bash = shutil.which("bash")
            # The exit code is what each test asserts on.
            return subprocess.run(
                [bash, "-e", "-c", self.script],
                check=False,
                env=env,
                capture_output=True,
                text=True,
            )

    def test_a_healthy_host_passes(self):
        result = self.run_step(HEALTHY)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("All 4 GPUs on runner-under-test report", result.stdout)

    def test_a_gpu_whose_smu_does_not_answer_fails_the_host(self):
        result = self.run_step(ONE_DEAD_GPU)
        self.assertEqual(result.returncode, 1)
        self.assertIn("GPU(s) 3 report no power", result.stdout)
        # The GPU's unique id, for whoever drains the host.
        self.assertIn("GPU[3] Unique ID: 0x0000000000000003", result.stdout)

    def test_an_edge_only_host_fails_and_names_the_gpus_without_power(self):
        result = self.run_step(EDGE_HOST)
        self.assertEqual(result.returncode, 1)
        self.assertIn("GPU(s) 0 1 2 3 4 5 report N/A", result.stdout)
        self.assertIn("GPU(s) 0 5 report no power", result.stdout)
        self.assertNotIn("GPU[1] Unique ID", result.stdout)

    def test_a_host_without_rocm_smi_fails(self):
        result = self.run_step(None)
        self.assertEqual(result.returncode, 1)
        self.assertIn("rocm-smi not found on runner-under-test", result.stdout)

    def test_a_failing_rocm_smi_fails_the_host(self):
        result = self.run_step(HEALTHY, smi_exit=2)
        self.assertEqual(result.returncode, 1)
        self.assertIn("rocm-smi failed or timed out", result.stdout)

    def test_output_without_gpu_rows_fails_rather_than_passing(self):
        result = self.run_step(HEADER + FOOTER)
        self.assertEqual(result.returncode, 1)
        self.assertIn("no GPU rows in rocm-smi output", result.stdout)


if __name__ == "__main__":
    unittest.main()
