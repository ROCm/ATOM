# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Keep automatic benchmark images on the default ROCm release line."""

import ast
import contextlib
import io
import os
import re
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

import yaml

ROOT = Path(__file__).resolve().parents[1]


class NightlyImageSelectionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        workflow = yaml.safe_load(
            (ROOT / ".github/workflows/atomesh-benchmark.yaml").read_text()
        )
        script = next(
            step["run"]
            for step in workflow["jobs"]["load-config"]["steps"]
            if step.get("id") == "image"
        )
        cls.python = script.split("python3 - <<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
        tree = ast.parse(cls.python)
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "latest_nightly_for"
        )
        cls.function = compile(
            ast.Module(body=[function], type_ignores=[]), "workflow-image", "exec"
        )

    def select(self, tags):
        scope = {"re": re, "docker_tags": lambda prefix: tags}
        exec(self.function, scope)  # noqa: S102 - trusted repository workflow
        return scope["latest_nightly_for"](date(2026, 10, 11))

    def test_newer_rocm10_and_custom_builds_cannot_replace_default_line(self):
        self.assertEqual(
            self.select(
                [
                    {
                        "name": "nightly_202610111450",
                        "last_updated": "2026-10-11T14:53:30Z",
                    },
                    {
                        "name": "nightly_202610111453-rocm10.0",
                        "last_updated": "2026-10-11T14:59:47Z",
                    },
                    {
                        "name": "nightly_202610111500-custom",
                        "last_updated": "2026-10-11T15:00:00Z",
                    },
                ]
            ),
            "rocm/atom-dev:nightly_202610111450",
        )

    def test_variant_only_day_falls_back_to_previous_day(self):
        self.assertIsNone(
            self.select(
                [
                    {
                        "name": "nightly_202610111453-rocm10.0",
                        "last_updated": "2026-10-11T14:59:47Z",
                    },
                ]
            )
        )

    def test_newest_default_build_wins(self):
        self.assertEqual(
            self.select(
                [
                    {
                        "name": "nightly_202610111450",
                        "last_updated": "2026-10-11T14:53:30Z",
                    },
                    {
                        "name": "nightly_202610111214",
                        "last_updated": "2026-10-11T12:17:00Z",
                    },
                ]
            ),
            "rocm/atom-dev:nightly_202610111450",
        )

    def test_explicit_image_override_never_queries_nightly_tags(self):
        for image in (
            "rocm/atom-dev:nightly_202610081455",
            "rocm/atom-dev:nightly_202610111453-rocm10.0",
        ):
            with self.subTest(image=image), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "output"
                with (
                    patch.dict(
                        os.environ,
                        {
                            "INPUT_ATOMESH_IMAGE": image,
                            "GITHUB_OUTPUT": str(output),
                            "GITHUB_STEP_SUMMARY": "",
                        },
                    ),
                    patch(
                        "urllib.request.urlopen",
                        side_effect=AssertionError("unexpected network lookup"),
                    ),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    code = compile(self.python, "workflow-image", "exec")
                    exec(code, {})  # noqa: S102 - trusted repository workflow
                self.assertIn(f"image={image}\n", output.read_text())


class ChunkedCaseEnvironmentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cfg = yaml.safe_load(
            (ROOT / ".github/benchmark/models_atomesh.yaml").read_text()
        )
        cls.cases = {
            case["name"]: case
            for cases in cfg["models"]["GLM-5.2-MXFP4"]["suites"].values()
            for case in cases
        }

    def test_existing_1p3d_decode_environments_do_not_acquire_reuse_flag(self):
        for concurrency in (12, 16, 24):
            case = self.cases[
                f"glm-52-mxfp4-1p3d-cpp4-tp4-mtp5-agentic-lmcache-1m-c{concurrency}"
            ]
            self.assertNotIn("AITER_REUSE_IDENTICAL_COMM_GROUPS", case["env"]["decode"])

    def test_chunked_case_retains_cpp4_dcp4_baseline_environment(self):
        baseline = self.cases["glm-52-mxfp4-1p1d-cpp4-dcp4-agentic-lmcache-1m-c48"]
        chunked = self.cases[
            "glm-52-mxfp4-1p1d-cpp4-dcp4-agentic-lmcache-chunked-1m-c32"
        ]
        self.assertEqual(baseline["env"]["common"], chunked["env"]["common"])
        self.assertEqual(baseline["env"]["decode"], chunked["env"]["decode"])
        p = dict(chunked["env"]["prefill"])
        p["PREFILL_KV_TRANSFER_CONFIG"] = p["PREFILL_KV_TRANSFER_CONFIG"].replace(
            ',"enable_chunked_transfer":true', ""
        )
        self.assertEqual(baseline["env"]["prefill"], p)
