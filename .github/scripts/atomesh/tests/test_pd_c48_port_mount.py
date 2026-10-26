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
    "V-QD-CPUREF2-FP8ATTN-PROFILE": "cpu-lru-refresh-v2",
    "V-QD-CPUREF2-FP8ATTN-FQ": "cpu-lru-refresh-v2",
    "V-QD-CPUREF2-FP8ATTN-FQ2": "cpu-lru-refresh-v2",
    "V-QD-CPUREF2-FP8ATTN-FQ2-PQ": "cpu-lru-refresh-v2",
    "V-QD-CPUREF2-FP8ATTN-FQ2-PQ-LDFP4": "cpu-lru-refresh-v2",
    "V-QD-CPUREF2-FP8ATTN-FQ2-PQ-LDFP4-INPAD": "cpu-lru-refresh-v2",
    "V-QD-CPUREF2-FP8ATTN-FQ2-PQ-LDFP4-INPAD-WARM": "cpu-lru-refresh-v2",
    "V-QD-CPUREF2-FP8ATTN-FQ3": "cpu-lru-refresh-v2",
    "V-QD-CPUREF2-FP8ATTN-FQ2-WARM": "cpu-lru-refresh-v2",
    "V-QD-CPUREF2-FP8ATTN-FQ2-WARM1": "cpu-lru-refresh-v2",
    "V-QD-CPUREF2-FP8ATTN-FQ3-PQ-LDFP4-INPAD-WARM": "cpu-lru-refresh-v2",
    "V-QD-CPUREF3-FP8ATTN-FQ3-PQ-LDFP4-INPAD-WARM": "cpu-lru-refresh-v3",
    "V-QD-CPUREF3-FP8ATTN-FQ3-PQ-LDFP4-INPAD-WARM-DG5": "cpu-lru-refresh-v3",
    "V-QD-CPUREF3-FP8ATTN-FQ3-PQ-LDFP4-INPAD-WARM-PNT": "cpu-lru-refresh-v3",
    "V-QD-CPUREF3-FP8ATTN-FQ3-PQ-LDFP4-INPAD-WARM-DG5-PNT": "cpu-lru-refresh-v3",
    **{f"V-QD-CPUREF3-FP8ATTN-FQ3-PQ-LDFP4-INPAD-WARM-SW-C{c}": "cpu-lru-refresh-v3"
       for c in (1, 4, 16, 32, 48, 56, 72, 88)},
    **{f"V-QD-CPUREF3-FP8ATTN-FQ3-PQ-LDFP4-INPAD-WARM-DG5-PNT-SWB-{t}": "cpu-lru-refresh-v3"
       for t in ("C4D8", "C16", "C32", "C56", "C72", "C88", "C56-CPU26", "C72-CPU26", "C88-CPU26",
                 "C72-MLA96", "C72-MLA96-DPROF", "C72-MLA96-DLDFP4", "C72-DLDFP4")},
    "V-QD-CPUREF3-FP8ATTN-FQ3-PQ-LDFP4-INPAD-WARM-DG5-PNT-UPS": "cpu-lru-refresh-v3",
    "V-QD-CPUREF3-FP8ATTN-FQ3-PQ-LDFP4-INPAD-WARM-DG5-PNT-MLA96-GSM8K": "cpu-lru-refresh-v3",
}
VLLM = "/usr/local/lib/python3.12/dist-packages/vllm/"
MLA96 = {"mla-dcp96/rocm_aiter_mla.py": "v1/attention/backends/mla/rocm_aiter_mla.py"}
UPS = {
    "ups-pnt-warm/scheduler.py": "v1/core/sched/scheduler.py",
    "ups-pnt-warm/cache.py": "config/cache.py",
    "ups-pnt-warm/arg_utils.py": "engine/arg_utils.py",
    "ups-pnt-warm/kernel.py": "config/kernel.py",
    "ups-pnt-warm/kernel_warmup.py": "model_executor/warmup/kernel_warmup.py",
}


def extra_overlays(case):
    return {**(MLA96 if "-MLA96" in case else {}), **(UPS if case.endswith("-UPS") else {})}
FP8ATTN = "/vllm/v1/attention/backends/mla/prefill/aiter_flash_attn.py,readonly"
LDFP4_CASE = "V-QD-CPUREF2-FP8ATTN-FQ2-PQ-LDFP4"
LDFP4 = "/vllm/models/kimi_k3/amd/linear.py,readonly"
INPAD_CASE = "V-QD-CPUREF2-FP8ATTN-FQ2-PQ-LDFP4-INPAD"
FQ3_CASE = "V-QD-CPUREF2-FP8ATTN-FQ3"


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
        self.assertEqual(mounts("V-QD-LMC"), (0, base))
        for case, fix_dir in CPUREF.items():
            rc, cpuref = mounts(case)
            self.assertEqual(rc, 0)
            inpad = "-INPAD" in case
            kda_dir = "inproj-pad128" if inpad else "online-quant-fix"
            self.assertIn(f"/c48-model-gap/{kda_dir}/kda.py,", cpuref[1])
            self.assertEqual(cpuref[1].split(",")[2:], base[1].split(",")[2:])
            self.assertEqual([cpuref[0], cpuref[2]], [base[0], base[2]])
            self.assertTrue(cpuref[3].endswith(MANAGER))
            self.assertIn(f"/c48-model-gap/{fix_dir}/manager.py,", cpuref[3])
            fp8 = "-FP8ATTN" in case
            ldfp4 = "-PQ-LDFP4" in case
            extra = extra_overlays(case)
            self.assertEqual(len(cpuref), 4 + fp8 + ldfp4 + len(extra))
            if fp8:
                self.assertTrue(cpuref[4].endswith(FP8ATTN))
                fp8_dir = "mla-prefill-fp8-fusedq" if "-FQ3" in case else "mla-prefill-fp8"
                self.assertIn(f"/c48-model-gap/{fp8_dir}/", cpuref[4])
            if ldfp4:
                self.assertTrue(cpuref[5].endswith(LDFP4))
                self.assertIn("/c48-model-gap/latent-down-mxfp4/", cpuref[5])
            for arg, (source, target) in zip(cpuref[6:], extra.items()):
                self.assertIn(f"/c48-model-gap/{source},", arg)
                self.assertTrue(arg.endswith(f"target={VLLM}{target},readonly"), arg)

    def test_extra_overlays_reject_changed_file(self):
        for case in ("V-QD-CPUREF3-FP8ATTN-FQ3-PQ-LDFP4-INPAD-WARM-DG5-PNT-SWB-C72-MLA96",
                     "V-QD-CPUREF3-FP8ATTN-FQ3-PQ-LDFP4-INPAD-WARM-DG5-PNT-UPS"):
            for source in extra_overlays(case):
                with tempfile.TemporaryDirectory() as temp:
                    gap = Path(temp) / ".github/benchmark/c48-model-gap"
                    shutil.copytree(REPO / ".github/benchmark/c48-model-gap", gap)
                    with open(gap / source, "a") as f:
                        f.write("\n")
                    self.assertEqual(mounts("V-QD-CPUREF3-FP8ATTN-FQ3-PQ-LDFP4-INPAD-WARM-DG5-PNT", temp)[0], 0)
                    self.assertEqual(mounts(case, temp)[0], 2)

    def test_ldfp4_rejects_changed_overlay(self):
        with tempfile.TemporaryDirectory() as temp:
            gap = Path(temp) / ".github/benchmark/c48-model-gap"
            shutil.copytree(REPO / ".github/benchmark/c48-model-gap", gap)
            with open(gap / "latent-down-mxfp4/linear.py", "a") as f:
                f.write("\n")
            self.assertEqual(mounts("V-QD-CPUREF2-FP8ATTN-FQ2-PQ", temp)[0], 0)
            self.assertEqual(mounts(LDFP4_CASE, temp)[0], 2)

    def test_inpad_rejects_changed_kda(self):
        with tempfile.TemporaryDirectory() as temp:
            gap = Path(temp) / ".github/benchmark/c48-model-gap"
            shutil.copytree(REPO / ".github/benchmark/c48-model-gap", gap)
            with open(gap / "inproj-pad128/kda.py", "a") as f:
                f.write("\n")
            self.assertEqual(mounts(LDFP4_CASE, temp)[0], 0)
            self.assertEqual(mounts(INPAD_CASE, temp)[0], 2)

    def test_fp8attn_rejects_changed_overlay(self):
        with tempfile.TemporaryDirectory() as temp:
            gap = Path(temp) / ".github/benchmark/c48-model-gap"
            shutil.copytree(REPO / ".github/benchmark/c48-model-gap", gap)
            with open(gap / "mla-prefill-fp8/aiter_flash_attn.py", "a") as f:
                f.write("\n")
            self.assertEqual(mounts("V-QD-CPUREF2", temp)[0], 0)
            self.assertEqual(mounts("V-QD-CPUREF2-FP8ATTN", temp)[0], 2)

    def test_fq3_rejects_changed_overlay(self):
        with tempfile.TemporaryDirectory() as temp:
            gap = Path(temp) / ".github/benchmark/c48-model-gap"
            shutil.copytree(REPO / ".github/benchmark/c48-model-gap", gap)
            with open(gap / "mla-prefill-fp8-fusedq/aiter_flash_attn.py", "a") as f:
                f.write("\n")
            self.assertEqual(mounts("V-QD-CPUREF2-FP8ATTN-FQ2", temp)[0], 0)
            self.assertEqual(mounts(FQ3_CASE, temp)[0], 2)

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
