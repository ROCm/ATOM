# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Execute the workflow matrix builder and V4 accuracy paths without GPUs."""

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / ".github/scripts/atomesh"
MODEL = "DeepSeek-V4-Pro-0813"
CONCURRENCIES = {1, 2, 16, 32, 128, 192, 256}


class V4WorkflowTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        workflow = yaml.safe_load(
            (ROOT / ".github/workflows/atomesh-benchmark.yaml").read_text()
        )
        cls.script = next(
            step["run"]
            for step in workflow["jobs"]["load-config"]["steps"]
            if step.get("id") == "matrix"
        )

    def matrix(self, event="schedule", schedule="0 16 * * 5", **overrides):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in (
                ".github/scripts/atomesh/pd_matrix.py",
                ".github/benchmark/models_atomesh.yaml",
            ):
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / name, target)
            env = dict(
                os.environ,
                GITHUB_EVENT_NAME=event,
                EVENT_SCHEDULE=schedule,
                GITHUB_OUTPUT=str(root / "github-output"),
                SUITE="nightly",
                RUN_ALL_MODELS="true",
                MODEL_NAMES="",
                CASE_NAMES="",
                ATOMESH_SLURM_SUBMIT_RUNNER="atomesh-cicd",
                ATOMESH_SLURM_ACCOUNT="",
                ATOMESH_SLURM_PARTITION="",
                ATOMESH_MODEL_ROOT="/models",
                ATOMESH_LOG_ROOT="/logs",
                ATOMESH_NODE_POOL="pit2-p03-g01,pit2-p03-g03,pit2-p03-g07,pit2-p03-g42",
                ATOMESH_SINGLE_NODE="auto",
                ATOMESH_1P1D_NODES="",
                ATOMESH_2P1D_NODES="",
                ATOMESH_PD_RANK_MAPPING_POLICY="none",
                ATOMESH_IMAGE="test-image",
                ATOMESH_BENCHMARK_CONCURRENCY="",
                ATOMESH_EVAL_CONCURRENCY="",
            )
            env.update(overrides)
            result = subprocess.run(
                ["bash", "-eu", "-c", self.script],
                cwd=root,
                env=env,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            cells = json.loads((root / "atomesh-matrix.json").read_text())["include"]
            self.assertIn("has_matrix=true", (root / "github-output").read_text())
            return cells

    def test_tw_schedule_includes_v4_weekly_and_existing_models(self):
        cells = self.matrix()
        v4 = [
            cell
            for cell in cells
            if cell["model"] == MODEL and "-no-offload-" in cell["name"]
        ]
        self.assertEqual(len(v4), 7)
        self.assertEqual({cell["concurrency"][0] for cell in v4}, CONCURRENCIES)
        for cell in v4:
            self.assertEqual(cell["suite"], "weekly")
            self.assertEqual(cell["runner"]["slurm_account"], "amd-frameworks")
        self.assertTrue(any(cell["model"] == "DeepSeek-V4-Pro" for cell in cells))
        self.assertTrue(any(cell["model"] == "GLM-5.2-MXFP4" for cell in cells))
        self.assertTrue(all(not cell["eval_only"] for cell in cells))

    def test_push_and_mi350_daily_exclude_v4_weekly(self):
        for event, schedule, runner in (
            ("push", "", "atomesh-cicd"),
            ("schedule", "10 16 * * *", "atomesh-cicd-mi350"),
        ):
            with self.subTest(event=event, runner=runner):
                cells = self.matrix(
                    event=event,
                    schedule=schedule,
                    ATOMESH_SLURM_SUBMIT_RUNNER=runner,
                    ATOMESH_1P1D_NODES="n1,n2,n3,n4" if "mi350" in runner else "",
                    ATOMESH_2P1D_NODES="n1,n2,n3,n4" if "mi350" in runner else "",
                )
                self.assertTrue(all(cell["model"] != MODEL for cell in cells))

    def test_manual_weekly_case_selection_preserves_256_threshold(self):
        cells = self.matrix(
            event="workflow_dispatch",
            SUITE="weekly",
            RUN_ALL_MODELS="false",
            CASE_NAMES="ds-v4-0813-1p1d-dpa-tp8-dspark3-agentic-no-offload-c256",
        )
        self.assertEqual(len(cells), 1)
        self.assertEqual(cells[0]["concurrency"], [256])
        self.assertEqual(
            cells[0]["env"]["common"]["ROUTER_BALANCE_ABS_THRESHOLD"], "40"
        )

    def test_manual_accuracy_is_eval_only_without_dpa_or_synthetic_al(self):
        cells = self.matrix(
            event="workflow_dispatch",
            SUITE="accuracy",
            RUN_ALL_MODELS="false",
            MODEL_NAMES=MODEL,
        )
        self.assertEqual(len(cells), 1)
        cell = cells[0]
        self.assertTrue(cell["eval_only"])
        self.assertTrue(cell["run_eval"])
        self.assertEqual(cell["concurrency"], [16])
        self.assertEqual(cell["server_args"]["spec_decode_acceptance_length"], "")
        for role in ("prefill", "decode"):
            self.assertEqual(cell["service"][role]["tp"], 8)
            self.assertNotIn(
                "--enable-dp-attention", cell["service"][role]["extra_args"]
            )

    def test_manual_ep_mega_mp_keeps_pd_workload_and_role_boundaries(self):
        concurrencies = [256, 192, 512]
        cases = [
            f"ds-v4-0813-1p1d-dpa-tp8-dspark3-agentic-ep8-mega-mp-c{c}"
            for c in concurrencies
        ]
        cells = self.matrix(
            event="workflow_dispatch",
            SUITE="weekly",
            RUN_ALL_MODELS="false",
            CASE_NAMES=",".join(cases),
        )
        self.assertEqual({cell["name"] for cell in cells}, set(cases))
        for cell in cells:
            with self.subTest(concurrency=cell["concurrency"]):
                self.assertEqual(cell["num_nodes"], 2)
                self.assertEqual(cell["benchmark"]["benchmark_duration"], 3600)
                self.assertEqual(cell["benchmark"]["warmup_requests_per_lane"], 1)
                self.assertEqual(
                    cell["server_args"]["spec_decode_acceptance_length"], 3.01
                )
                self.assertEqual(cell["server_args"]["max_num_seqs"], 128)
                p = cell["service"]["prefill"]["extra_args"]
                d = cell["service"]["decode"]["extra_args"]
                self.assertIn("--moe-backend mega", p)
                self.assertIn("--enable-expert-parallel", p)
                self.assertIn("--enable-eplb", p)
                self.assertNotIn("--enable-expert-parallel", d)
                self.assertNotIn("--enable-tbo", p + d)
                self.assertEqual(
                    cell["env"]["prefill"]["ATOM_KV_OFFLOAD"], "lmcache_mp"
                )
                self.assertNotIn("ATOM_KV_OFFLOAD", cell["env"]["decode"])
                for role in ("prefill", "decode"):
                    self.assertIn(
                        "mooncake",
                        cell["env"][role][f"{role.upper()}_KV_TRANSFER_CONFIG"],
                    )
                threshold = "20" if cell["concurrency"] == [192] else "40"
                self.assertEqual(
                    cell["env"]["common"]["ROUTER_BALANCE_ABS_THRESHOLD"], threshold
                )
                # Execute the same Bash expansion used by pd_server_atom.sh;
                # quotes inside the compact EPLB JSON must survive as one argv.
                argv = subprocess.check_output(
                    ["bash", "-c", "printf '%s\\n' ${ARGS}"],
                    env=dict(os.environ, ARGS=p),
                    text=True,
                ).splitlines()
                self.assertEqual(
                    json.loads(argv[argv.index("--eplb-config") + 1]),
                    {"load_window_size": 100, "rebalance_interval": 200},
                )


class V4AccuracyPhaseTest(unittest.TestCase):
    def test_eval_phase_dispatch_and_server_args(self):
        source = (SCRIPTS / "pd_server_atom.sh").read_text()
        start = source.index("run_benchmark_and_eval() {")
        dispatch = source[start : source.index("\n}\n", start) + 3]
        start = source.index("spec_decode_acceptance_for_server=")
        acceptance = source[
            start : source.index(
                'if [[ -n "${STATE_CHECKPOINT_INTERVAL_TOKENS}"', start
            )
        ]
        script = acceptance + dispatch + """
run_benchmark() { echo UNEXPECTED_BENCHMARK; }
run_eval() { echo EVAL; }
run_benchmark_and_eval
printf 'ARG:%s\\n' "${server_common[@]}"
"""
        result = subprocess.run(
            ["bash", "-eu", "-c", "server_common=()\n" + script],
            env=dict(
                os.environ,
                ATOMESH_EXECUTION_PHASE="eval",
                EVAL_TASK="gsm8k",
                SPEC_DECODE_ACCEPTANCE_LENGTH="3.01",
            ),
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        self.assertIn("\nEVAL\n", result.stdout)
        self.assertNotIn("UNEXPECTED_BENCHMARK", result.stdout)
        self.assertNotIn("--spec-decode-acceptance-length", result.stdout)
        self.assertNotIn("3.01", result.stdout)


if __name__ == "__main__":
    unittest.main()
