# SPDX-License-Identifier: MIT
"""Enforce the user's one-variable experiment against the saved CI matrix."""

import contextlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / ".github/scripts/atomesh/check_m3_gluon_c32.py"
spec = importlib.util.spec_from_file_location("gluon_c32_check", SCRIPT)
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


class SingleVariableTest(unittest.TestCase):
    def candidate(self):
        data = json.loads(SCRIPT.with_name("m3_gluon_c32_baseline.json").read_text())
        for role in ("common", "prefill", "decode", "router"):
            data["include"][0]["env"][role]["ATOM_AITER_SCATTER_REF"] = "1" * 40
        return data

    def test_only_pin_change_is_accepted(self):
        with contextlib.redirect_stdout(io.StringIO()):
            checker.compare(self.candidate())

    def test_draft_parallelism_cache_workload_changes_are_rejected(self):
        changes = [
            ("server_args", "draft_model", "Inferact/MiniMax-M3-EAGLE-GQA"),
            ("server_args", "num_speculative_tokens", 4),
            ("server_args", "gpu_memory_utilization", 0.90),
            ("benchmark", "benchmark_duration", 1800),
            ("benchmark", "mesh_source", "mounted"),
        ]
        for section, key, value in changes:
            with self.subTest(field=key):
                data = self.candidate()
                data["include"][0][section][key] = value
                with self.assertRaisesRegex(ValueError, "Non-kernel"):
                    checker.compare(data)
        data = self.candidate()
        data["include"][0]["service"]["prefill"]["tp"] = 1
        with self.assertRaisesRegex(ValueError, "Non-kernel"):
            checker.compare(data)
        data = self.candidate()
        data["include"][0]["env"]["prefill"]["LMCACHE_MAX_LOCAL_CPU_SIZE"] = "768"
        with self.assertRaisesRegex(ValueError, "Non-kernel"):
            checker.compare(data)

    def test_adapter_gate_uses_checkout_with_conflicting_editable_install(self):
        with tempfile.TemporaryDirectory() as temp:
            decoy = Path(temp)
            (decoy / "atom").mkdir()
            (decoy / "atom/__init__.py").write_text(
                'raise AssertionError("Imported the image ATOM")\n'
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT.with_name("check_m3_gluon_adapter.py")),
                    "--check-source",
                ],
                cwd=decoy,
                env={**os.environ, "PYTHONPATH": str(decoy)},
                capture_output=True,
                text=True,
                timeout=10,
                check=True,
            )
            self.assertIn(f"atom_source={ROOT}/atom/__init__.py", result.stdout)

    def test_baseline_runtime_instrumentation_still_applies(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            subprocess.run(
                [
                    "python3",
                    str(SCRIPT.with_name("minimax_runtime_watch.py")),
                    "--source",
                    str(ROOT),
                    "--runtime",
                    str(root / "runtime"),
                    "--output",
                    str(root / "diagnostics"),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            source = (root / "runtime/atom/model_ops/attention_mha.py").read_text()
            compile(source, "instrumented_attention_mha.py", "exec")
            self.assertTrue(
                (root / "runtime/atom/model_ops/gluon_prefill.py").is_file()
            )


if __name__ == "__main__":
    unittest.main()
