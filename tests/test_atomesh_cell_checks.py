# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Launcher checks that fail a cell instead of publishing what it did not run."""

import copy
import importlib.util
import json
import os
import shlex
import subprocess
import sys
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


def kimi_k3_cells(suite):
    """The real Kimi-K3 cells of `suite`, as pd_matrix.py writes them."""
    env = dict(
        os.environ,
        ATOMESH_SLURM_SUBMIT_RUNNER="atomesh-cicd",
        ATOMESH_SLURM_ACCOUNT="",
        ATOMESH_SLURM_PARTITION="",
        ATOMESH_MODEL_ROOT="/models",
        ATOMESH_LOG_ROOT="/logs",
        ATOMESH_1P1D_NODES="",
        ATOMESH_2P1D_NODES="",
        ATOMESH_PD_RANK_MAPPING_POLICY="none",
    )
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "matrix.json"
        subprocess.run(
            [
                sys.executable,
                str(SCRIPTS / "pd_matrix.py"),
                "--suite",
                suite,
                "--model",
                "Kimi-K3-MXFP4",
                "--output",
                str(output),
            ],
            cwd=ROOT,
            env=env,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return json.loads(output.read_text())["include"]


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

    def test_a_kimi_k3_cell_requires_each_newer_setting_it_sets(self):
        newer = {
            "ATOM_KV_POLL_FRESH_WAIT_MS",
            "ATOM_PD_PROMPT_IDS_B64",
            "ATOM_PD_EARLY_T0",
            "ATOM_PD_MLA_STAGING_CHUNK_PAGES",
            "ATOM_PD_MLA_STAGING_STREAM_PRIORITY",
            "ATOM_PP_MAX_INFLIGHT",
        }
        in_flight_cells = 0
        for suite in ("nightly", "profile"):
            for cell in kimi_k3_cells(suite):
                sets = {name for role in cell["env"].values() for name in role}
                with self.subTest(cell=cell["name"]):
                    self.assertLessEqual(sets & newer, set(cell["requires_atom_env"]))
                in_flight_cells += "ATOM_PP_MAX_INFLIGHT" in sets
        # C88's, which the env read above must have seen
        self.assertEqual(in_flight_cells, 1)


MONO_UP = ["Kimi-K3 mono decode armed", "Kimi-K3 mono decode on"]
MONO_UP_LOG = ["Kimi-K3 mono decode armed (TP8)", "Kimi-K3 mono decode on"]


class ExpectedLogTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.logs = Path(temp.name)

    def write(self, name, *lines):
        (self.logs / name).write_text(
            "".join(f"INFO {line}\n" for line in ("model loaded", *lines))
        )

    def check(self, aggregated="0", prefill=None, decode=None):
        return self.check_exported(
            aggregated,
            json.dumps(prefill) if prefill else "",
            json.dumps(decode) if decode else "",
        )

    def check_exported(self, aggregated, prefill, decode):
        return subprocess.run(
            [
                "bash",
                "-eu",
                "-c",
                shell_function("collect_role_logs")
                + shell_function("parse_log_lines")
                + shell_function("check_expected_logs")
                + "check_expected_logs\necho OK",
            ],
            env=dict(
                os.environ,
                RUNTIME_LOG_DIR=str(self.logs),
                AGGREGATED=aggregated,
                ATOMESH_EXPECT_LOG_TIMEOUT="0",
                PREFILL_EXPECT_LOG=prefill,
                DECODE_EXPECT_LOG=decode,
            ),
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

    def test_nothing_expected_reads_no_log(self):
        result = self.check()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "OK\n")

    def test_every_server_of_the_role_logged_every_line(self):
        self.write("prefill-rank-0.log", "prefill up")
        self.write("decode-rank-1.log", *MONO_UP_LOG)
        self.write("decode-rank-1-worker-1.log", *MONO_UP_LOG)
        result = self.check(decode=MONO_UP)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            "all 2 decode server log(s) show all 2 expected line(s)", result.stdout
        )
        self.assertTrue(result.stdout.endswith("OK\n"))

    def test_a_server_armed_but_never_on_fails_the_cell(self):
        self.write("decode-rank-1.log", *MONO_UP_LOG)
        self.write("decode-rank-1-worker-1.log", "Kimi-K3 mono decode armed (TP8)")
        result = self.check(decode=MONO_UP)
        self.assertEqual(result.returncode, 1)
        self.assertNotIn("OK", result.stdout)
        self.assertEqual(
            result.stderr,
            "[check][FAIL] "
            f"{self.logs}/decode-rank-1-worker-1.log never logged "
            "'Kimi-K3 mono decode on'\n",
        )

    def test_a_server_that_logged_neither_names_both(self):
        self.write("decode-rank-1.log", "Kimi-K3 mono decode off: why")
        result = self.check(decode=MONO_UP)
        self.assertEqual(result.returncode, 1)
        for line in MONO_UP:
            self.assertIn(f"decode-rank-1.log never logged '{line}'", result.stderr)

    def test_no_server_log_fails_the_cell(self):
        self.write("prefill-rank-0.log", *MONO_UP_LOG)
        result = self.check(decode=MONO_UP)
        self.assertEqual(result.returncode, 1)
        self.assertIn("no decode server log", result.stderr)

    def test_aggregated_reads_the_server_as_the_decode(self):
        self.write("server-rank-0.log", *MONO_UP_LOG)
        result = self.check(aggregated="1", decode=MONO_UP)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("all 1 decode server log(s)", result.stdout)

    def test_an_aggregated_prefill_line_fails_the_cell(self):
        self.write("server-rank-0.log", *MONO_UP_LOG)
        result = self.check(aggregated="1", prefill=MONO_UP)
        self.assertEqual(result.returncode, 1)
        self.assertNotIn("OK", result.stdout)
        self.assertIn("aggregated cell has no prefill server", result.stderr)

    def test_a_value_that_is_not_a_list_of_lines_fails_the_cell(self):
        # Read as lines, a bare string would be split char by char and an
        # empty list or entry would check nothing, so either passes this log.
        self.write("decode-rank-1.log", *MONO_UP_LOG)
        for value in (
            "Kimi-K3 mono decode armed",
            json.dumps("Kimi-K3 mono decode armed"),
            "[]",
            json.dumps(["Kimi-K3 mono decode armed", ""]),
            json.dumps(["Kimi-K3 mono decode armed", 1]),
        ):
            with self.subTest(value=value):
                result = self.check_exported("0", "", value)
                self.assertEqual(result.returncode, 1)
                self.assertNotIn("OK", result.stdout)
                self.assertIn(
                    "decode expect_log is not a non-empty JSON list of "
                    "non-empty strings",
                    result.stderr,
                )


MONO_GONE = [
    "Kimi-K3 mono decode off",
    "Kimi-K3 mono decode declined the",
]


class RejectedLogTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.logs = Path(temp.name)

    def write(self, name, *lines):
        (self.logs / name).write_text("".join(f"INFO {line}\n" for line in lines))

    def check(self, aggregated="0", prefill=None, decode=None):
        return subprocess.run(
            [
                "bash",
                "-eu",
                "-c",
                shell_function("collect_role_logs")
                + shell_function("parse_log_lines")
                + shell_function("check_rejected_logs")
                + "check_rejected_logs\necho OK",
            ],
            env=dict(
                os.environ,
                RUNTIME_LOG_DIR=str(self.logs),
                AGGREGATED=aggregated,
                PREFILL_REJECT_LOG=json.dumps(prefill) if prefill else "",
                DECODE_REJECT_LOG=json.dumps(decode) if decode else "",
            ),
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

    def test_nothing_rejected_reads_no_log(self):
        result = self.check()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "OK\n")

    def test_servers_that_kept_the_feature_pass(self):
        # A prefill turning mono down is its own role's business.
        self.write("prefill-rank-0.log", "Kimi-K3 mono decode off: expert parallel")
        self.write(
            "decode-rank-1.log",
            "Kimi-K3 mono decode armed (TP8)",
            "Kimi-K3 mono decode declined a step: prefill",
            "Kimi-K3 mono decode declined a step: 128 rows > the 16-row buffers",
            "Kimi-K3 mono decode on",
        )
        result = self.check(decode=MONO_GONE)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("no decode server log shows any of 2", result.stdout)
        self.assertTrue(result.stdout.endswith("OK\n"))

    def test_a_server_that_logged_one_fails_the_cell(self):
        # Whichever width it was: 8 rows is C1's step, 16 its capture's widest.
        for rows in (8, 16):
            with self.subTest(rows=rows):
                self.write("decode-rank-1.log", "Kimi-K3 mono decode armed (TP8)")
                self.write(
                    "decode-rank-1-worker-1.log",
                    "Kimi-K3 mono decode armed (TP8)",
                    f"Kimi-K3 mono decode declined the {rows}-row build; "
                    "other widths unaffected",
                )
                result = self.check(decode=MONO_GONE)
                self.assertEqual(result.returncode, 1)
                self.assertNotIn("OK", result.stdout)
                self.assertIn(
                    "decode-rank-1-worker-1.log logged "
                    "'Kimi-K3 mono decode declined the': "
                    f"INFO Kimi-K3 mono decode declined the {rows}-row build",
                    result.stderr,
                )

    def test_no_server_log_fails_the_cell(self):
        result = self.check(decode=MONO_GONE)
        self.assertEqual(result.returncode, 1)
        self.assertIn("no decode server log", result.stderr)

    def test_an_aggregated_prefill_list_fails_the_cell(self):
        self.write("server-rank-0.log", "Kimi-K3 mono decode armed (TP8)")
        result = self.check(aggregated="1", prefill=MONO_GONE)
        self.assertEqual(result.returncode, 1)
        self.assertIn("aggregated cell has no prefill server", result.stderr)

    def test_the_lines_are_checked_before_and_after_the_run(self):
        stubs = "\n".join(
            f"{name}() {{ echo {name}; }}"
            for name in (
                "check_expected_logs",
                "check_rejected_logs",
                "run_benchmark",
                "run_eval",
            )
        )
        for phase in ("benchmark", "eval", "combined"):
            with self.subTest(phase=phase):
                result = subprocess.run(
                    [
                        "bash",
                        "-eu",
                        "-c",
                        stubs
                        + "\n"
                        + shell_function("run_benchmark_and_eval")
                        + "run_benchmark_and_eval",
                    ],
                    env=dict(
                        os.environ,
                        ATOMESH_EXECUTION_PHASE=phase,
                        BENCHMARK_KIND="aiperf",
                        EVAL_TASK="gsm8k",
                        RUN_EVAL="false",
                    ),
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=True,
                )
                steps = result.stdout.split()
                self.assertEqual(
                    steps[:2], ["check_expected_logs", "check_rejected_logs"]
                )
                self.assertEqual(steps[-1], "check_rejected_logs")
                self.assertEqual(len(steps), 4 if phase != "combined" else 5)


class SubmitLogLinesTest(unittest.TestCase):
    """pd_submit.sh's export of a role's expect_log and reject_log, from the
    real C1 cell."""

    @classmethod
    def setUpClass(cls):
        cls.cell = next(
            cell
            for cell in kimi_k3_cells("nightly")
            if cell["name"].endswith("-c1-mono")
        )
        source = (SCRIPTS / "pd_submit.sh").read_text()
        start = source.index("python3 - <<'PY'\nimport json\nimport os\nimport shlex")
        start = source.index("\n", start) + 1
        cls.mapping = source[start : source.index("\nPY\n", start)]

    def submit(self, key=None, decode_lines=None):
        cell = copy.deepcopy(self.cell)
        if key is not None:
            cell["service"]["decode"][key] = decode_lines
        return subprocess.run(
            [sys.executable, "-c", self.mapping],
            env=dict(os.environ, CELL_JSON=json.dumps(cell)),
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

    def exported(self, result, key):
        self.assertEqual(result.returncode, 0, result.stderr)
        exports = {}
        for line in result.stdout.splitlines():
            words = shlex.split(line)
            if words[:1] == ["export"]:
                name, _, value = words[1].partition("=")
                exports[name] = value
        variable = key.upper()
        return exports[f"PREFILL_{variable}"], exports[f"DECODE_{variable}"]

    def test_c1_expects_mono_up_and_rejects_it_gone_on_its_decode_only(self):
        result = self.submit()
        for key, lines in (("expect_log", MONO_UP), ("reject_log", MONO_GONE)):
            with self.subTest(key=key):
                prefill, decode = self.exported(result, key)
                self.assertEqual(prefill, "")
                self.assertEqual(json.loads(decode), lines)
                self.assertNotIn("\n", decode)

    def test_a_single_line_is_a_list_of_one(self):
        for key in ("expect_log", "reject_log"):
            with self.subTest(key=key):
                _, decode = self.exported(
                    self.submit(key, "Kimi-K3 mono decode off"), key
                )
                self.assertEqual(json.loads(decode), ["Kimi-K3 mono decode off"])

    def test_an_empty_list_fails_the_cell(self):
        for key in ("expect_log", "reject_log"):
            with self.subTest(key=key):
                result = self.submit(key, [])
                self.assertEqual(result.returncode, 1)
                self.assertIn(f"decode {key} lists no line", result.stdout)
                self.assertEqual(result.stdout.splitlines()[-1], "exit 1")

    def test_an_unset_role_exports_nothing(self):
        for key in ("expect_log", "reject_log"):
            with self.subTest(key=key):
                _, decode = self.exported(self.submit(key, None), key)
                self.assertEqual(decode, "")

    def test_an_entry_that_is_not_a_line_fails_the_cell(self):
        for key in ("expect_log", "reject_log"):
            for entry in ("", 8):
                with self.subTest(key=key, entry=entry):
                    result = self.submit(key, ["Kimi-K3 mono decode off", entry])
                    self.assertEqual(result.returncode, 1)
                    self.assertIn(
                        f"decode {key} has an entry that is not a non-empty string",
                        result.stdout,
                    )
                    self.assertEqual(result.stdout.splitlines()[-1], "exit 1")


if __name__ == "__main__":
    unittest.main()
