# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Launcher checks that fail a cell instead of publishing what it did not run."""

import importlib.util
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / ".github/scripts/atomesh"
SPEC = importlib.util.spec_from_file_location(
    "atomesh_pd_matrix", SCRIPTS / "pd_matrix.py"
)
pd_matrix = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pd_matrix)


def shell_function(name):
    source = (SCRIPTS / "pd_server_atom.sh").read_text()
    start = source.index(f"{name}() {{")
    return source[start : source.index("\n}\n", start) + 3]


class RequiredAtomEnvTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        package = Path(temp.name) / "atom"
        (package / "utils").mkdir(parents=True)
        (package / "__init__.py").write_text("")
        (package / "utils" / "envs.py").write_text(
            "import os\n"
            "environment_variables = {\n"
            '    "ATOM_PD_EARLY_T0": lambda: os.getenv("ATOM_PD_EARLY_T0", "0"),\n'
            "}\n"
        )
        self.atom_origin = str(package / "__init__.py")

    def check(self, required):
        return subprocess.run(
            [
                "bash",
                "-eu",
                "-c",
                shell_function("check_required_atom_env")
                + "check_required_atom_env\necho OK",
            ],
            env=dict(
                os.environ,
                atom_origin=self.atom_origin,
                ATOMESH_CELL_ID="cell-c1",
                ATOMESH_REQUIRED_ATOM_ENV=required,
            ),
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

    def test_defined_settings_pass(self):
        for required in ("", "ATOM_PD_EARLY_T0"):
            with self.subTest(required=required):
                result = self.check(required)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout, "OK\n")

    def test_an_undefined_setting_fails_the_cell(self):
        result = self.check("ATOM_PD_EARLY_T0,ATOM_PD_PROMPT_IDS_B64")
        self.assertEqual(result.returncode, 1)
        self.assertNotIn("OK", result.stdout)
        self.assertIn("cell-c1 needs ATOM settings", result.stderr)
        self.assertIn("ATOM_PD_PROMPT_IDS_B64", result.stderr)
        self.assertNotIn("ATOM_PD_EARLY_T0", result.stderr)

    def test_an_unreadable_envs_py_fails_the_cell(self):
        Path(self.atom_origin).with_name("utils").joinpath("envs.py").write_text(
            "raise ImportError('broken')\n"
        )
        result = self.check("ATOM_PD_EARLY_T0")
        self.assertEqual(result.returncode, 1)
        self.assertIn("cannot read the settings", result.stderr)

    def test_the_cell_carries_its_requirements(self):
        cfg = {
            "backends": {"atom": {"image": "img"}},
            "models": {
                "M": {
                    "backend": "atom",
                    "model_path": "/m",
                    "suites": {
                        "nightly": [
                            {
                                "name": "c1",
                                "topology": "1p1d",
                                "nodes": "n1,n2",
                                "isl": 1024,
                                "osl": 1024,
                                "concurrency": [1],
                                "requires_atom_env": ["ATOM_PD_EARLY_T0"],
                            },
                            {
                                "name": "c2",
                                "topology": "1p1d",
                                "nodes": "n1,n2",
                                "isl": 1024,
                                "osl": 1024,
                                "concurrency": [2],
                            },
                        ]
                    },
                }
            },
        }
        cells = pd_matrix.build_cells(
            cfg,
            suite="nightly",
            model_filter=None,
            case_filter=None,
            benchmark_kind_filter=None,
            override_image=None,
            override_benchmark_concurrency=None,
            override_eval_concurrency=None,
        )
        self.assertEqual(
            {cell["name"]: cell["requires_atom_env"] for cell in cells},
            {"c1": ["ATOM_PD_EARLY_T0"], "c2": []},
        )


if __name__ == "__main__":
    unittest.main()
