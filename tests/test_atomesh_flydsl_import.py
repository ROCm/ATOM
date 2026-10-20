# SPDX-License-Identifier: MIT
"""Exercise the real API probe under Python's fresh-process import semantics."""

import ast
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class FlyDSLImportTests(unittest.TestCase):
    def run_probe(self, import_first, supported=True):
        tree = ast.parse((ROOT / "atom/model_ops/base_attention.py").read_text())
        probe = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_flydsl_explicit_plan_api"
        )
        # Isolate this GPU-independent probe from the attention module's GPU imports.
        source = ast.unparse(probe)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for package in ("aiter", "aiter/ops", "aiter/ops/flydsl"):
                path = root / package
                path.mkdir(exist_ok=True)
                (path / "__init__.py").touch()
            package = root / "aiter/ops/flydsl"
            (package / "__init__.py").write_text(
                "from importlib import import_module\n"
                "def __getattr__(name):\n"
                "    if name != 'pa_decode': raise AttributeError(name)\n"
                "    value = import_module('.pa_decode', __name__).pa_decode\n"
                "    globals()[name] = value\n"
                "    return value\n"
            )
            plan_arg = "work_plan" if supported else "work_plan=None"
            (package / "pa_decode.py").write_text(
                f"def pa_decode(output, *, {plan_arg}, max_context_length=None): pass\n"
                "def plan_pa_decode(): pass\n"
            )
            script = (
                "import functools, importlib, logging\n"
                "logger = logging.getLogger('test')\n"
                + import_first
                + "\n"
                + source
                + "\n"
                + f"assert _flydsl_explicit_plan_api() is {supported}\n"
            )
            result = subprocess.run(
                [sys.executable, "-c", script],
                cwd=root,
                capture_output=True,
                check=False,
                text=True,
                timeout=10,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_lazy_function_export_before_probe(self):
        self.run_probe(
            "from aiter.ops.flydsl import pa_decode; assert callable(pa_decode)"
        )

    def test_probe_as_first_import(self):
        self.run_probe("")

    def test_submodule_import_before_probe(self):
        self.run_probe("importlib.import_module('aiter.ops.flydsl.pa_decode')")

    def test_old_optional_plan_api_is_rejected(self):
        self.run_probe(
            "from aiter.ops.flydsl import pa_decode; assert callable(pa_decode)",
            supported=False,
        )


if __name__ == "__main__":
    unittest.main()
