# SPDX-License-Identifier: MIT
"""CPU-only tests for PR sweep planning, completeness and curve provenance."""

import copy
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.parse import quote

import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / ".github/scripts/atomesh"
with patch.object(sys, "path", [str(SCRIPT_DIR), *sys.path]):
    spec = importlib.util.spec_from_file_location(
        "pr_sweep", SCRIPT_DIR / "pr_sweep.py"
    )
    sweep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sweep)


class PrSweepTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = yaml.safe_load(
            (ROOT / ".github/benchmark/models_atomesh.yaml").read_text(encoding="utf-8")
        )

    def setUp(self):
        self.config = {
            "version": 1,
            "model": "GLM-5.2-MXFP4",
            "hardware": "mi355x",
            "mode": "pd",
            "image": "rocm/atom-dev@sha256:" + "a" * 64,
            "suite": "mooncake_store",
            "cases": [
                f"glm-52-mxfp4-1p1d-cpp4-dcp4-agentic-mcstore-1m-c{conc}"
                for conc in (48, 56)
            ],
        }
        self.jobs = []

    def plan(self, config=None):
        with patch.object(sys, "path", [str(SCRIPT_DIR), *sys.path]):
            return sweep.plan_sweep(
                config or self.config,
                self.catalog,
                repository="ROCm/ATOM",
                head_sha="b" * 40,
                pr_number=123,
            )

    def test_real_catalog_expands_one_complete_recipe_sweep_at_pinned_commit(self):
        before = dict(os.environ)
        plan = self.plan()
        self.assertEqual(plan["head_sha"], "b" * 40)
        self.assertEqual([cell["concurrency"] for cell in plan["cells"]], [[48], [56]])
        self.assertEqual(len(plan["groups"]), 1)
        self.assertTrue(
            all(cell["image"] == self.config["image"] for cell in plan["cells"])
        )
        self.assertTrue(all(cell["disaggregated"] for cell in plan["cells"]))
        self.assertEqual(os.environ, before)

    def test_aggregated_recipe_has_no_pd_curve(self):
        self.config.update(
            model="Kimi-K3-MXFP4",
            hardware="mi350x",
            mode="aggregated",
            suite="nightly",
            cases=["kimi-k3-mxfp4-agg-tp8-dcp8-agentic-lmcache-1m-c72"],
            concurrency=[16, 32],
        )
        plan = self.plan()
        self.assertFalse(plan["cells"][0]["disaggregated"])
        self.assertEqual([cell["concurrency"] for cell in plan["cells"]], [[16], [32]])
        self.assertEqual(len({cell["id"] for cell in plan["cells"]}), 2)
        self.assertEqual(
            plan["cells"][0]["runner"]["slurm_submit_runner"], "atomesh-cicd-mi350"
        )

    def test_inconsistent_or_unbounded_configuration_is_rejected(self):
        for changes in (
            {"mode": "aggregated"},
            {"image": "rocm/atom-dev:latest"},
            {"cases": ["unknown-recipe"]},
            {"concurrency": [0, 16]},
            {"concurrency": [16, 16]},
            {"concurrency": [True]},
            {"concurrency": [32, 64]},  # Same recipe selected twice with an override.
            {"unexpected": 1},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.plan({**self.config, **changes})

    def add_job(self, cell, attempt=2, conclusion="success"):
        self.jobs.append(
            {
                "name": f"recipe / sweep-point:{cell['id']}",
                "run_attempt": attempt,
                "status": "completed",
                "conclusion": conclusion,
            }
        )

    def write_point(self, directory, cell, *, run_id=456, attempt=2, **overrides):
        directory = (
            directory / f"atomesh-model-benchmark-{cell['id']}-attempt-{attempt}"
        )
        directory.mkdir(parents=True)
        self.add_job(cell, attempt)
        provenance = directory / f"{cell['id']}-sweep-provenance.json"
        provenance.write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "run_attempt": attempt,
                    "head_sha": "b" * 40,
                    "cell": cell,
                }
            ),
            encoding="utf-8",
        )
        point = {
            "model": cell["model"],
            "hardware": self.config["hardware"],
            "disaggregated": cell["disaggregated"],
            "precision": "mxfp4",
            "concurrency": cell["concurrency"][0],
            "total_tput": 1000.25,
            "interactivity": 25.5,
            "interactivity_p90_itl": 80.125,
            "interactivity_method": "p90_e2e_normalized",
            "total_gpu": 8,
            "run_url": f"https://github.com/ROCm/ATOM/actions/runs/{run_id}",
            **overrides,
        }
        path = directory / f"{cell['id']}-benchmark-action.json"
        path.write_text(
            json.dumps([{"extra": "perf_point=" + quote(json.dumps(point))}]),
            encoding="utf-8",
        )
        return provenance

    def collect(self, plan, directory, result="success"):
        return sweep.collect_sweep(
            plan,
            directory,
            run_id=456,
            run_attempt=2,
            benchmark_result=result,
            jobs=self.jobs,
        )

    def test_complete_curve_keeps_raw_metrics_and_run_attempt(self):
        plan = self.plan()
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            for cell in plan["cells"]:
                self.write_point(directory, cell)
            records, manifest = self.collect(plan, directory)
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(manifest["expected_points"], 2)
        self.assertEqual(len(records), 2)
        point = records[0]
        self.assertEqual(point["request_metrics"]["latency"]["intvty"]["p90"], 80.125)
        self.assertEqual(
            point["request_metrics"]["throughput"]["total"]["tokens_per_second"],
            1000.25,
        )
        self.assertEqual(
            point["run_url"], "https://github.com/ROCm/ATOM/actions/runs/456/attempts/2"
        )
        self.assertEqual(point["pr_number"], 123)
        self.assertEqual(point["head_sha"], "b" * 40)
        self.assertEqual(
            records[0]["recipe_fingerprint"], records[1]["recipe_fingerprint"]
        )

    def test_partial_curve_does_not_fill_missing_points_from_another_cell(self):
        plan = self.plan()
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            self.write_point(directory, plan["cells"][0])
            unrelated = copy.deepcopy(plan["cells"][1])
            unrelated["id"] = "old-unrelated-case"
            self.write_point(directory, unrelated)
            records, manifest = self.collect(plan, directory, "failure")
        self.assertEqual(len(records), 1)
        self.assertEqual(manifest["status"], "partial")
        self.assertEqual(manifest["missing"][0]["concurrency"], 56)
        self.assertEqual(records[0]["sweep_status"], "partial")

    def test_wrong_run_and_model_are_rejected(self):
        plan = self.plan()
        for changes in ({"run_id": 999}, {"model": "wrong-model"}):
            with self.subTest(changes=changes), tempfile.TemporaryDirectory() as root:
                directory = Path(root)
                self.write_point(directory, plan["cells"][0], **changes)
                with self.assertRaises(ValueError):
                    self.collect(plan, directory)

    def test_missing_p90_cannot_be_marked_complete_even_when_jobs_succeed(self):
        plan = self.plan()
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            for cell in plan["cells"]:
                self.write_point(directory, cell, interactivity_p90_itl=None)
            records, manifest = self.collect(plan, directory)
        self.assertEqual(records, [])
        self.assertEqual(manifest["status"], "partial")
        self.assertEqual(len(manifest["missing"]), 2)

    def test_failed_job_retry_retains_successful_points_and_original_links(self):
        plan = self.plan()
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            self.write_point(directory, plan["cells"][0], attempt=1)
            self.add_job(plan["cells"][1], attempt=1, conclusion="failure")
            self.write_point(directory, plan["cells"][1], attempt=2)
            records, manifest = self.collect(plan, directory)
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(manifest["reused_points"], 1)
        self.assertEqual([record["run_attempt"] for record in records], [1, 2])
        self.assertTrue(all(record["sweep_run_attempt"] == 2 for record in records))
        self.assertTrue(records[0]["run_url"].endswith("/attempts/1"))
        self.assertTrue(manifest["run_url"].endswith("/attempts/2"))
        self.assertEqual(
            manifest["point_attempts"][0]["run_url"], records[0]["run_url"]
        )

    def test_retry_rejects_changed_provenance(self):
        plan = self.plan()
        for key, value in (
            ("run_id", 999),
            ("run_attempt", 2),
            ("head_sha", "c" * 40),
            ("image", "different-image"),
            ("concurrency", [999]),
        ):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as root:
                directory = Path(root)
                path = self.write_point(directory, plan["cells"][0], attempt=1)
                provenance = json.loads(path.read_text(encoding="utf-8"))
                target = (
                    provenance["cell"]
                    if key in {"image", "concurrency"}
                    else provenance
                )
                target[key] = value
                path.write_text(json.dumps(provenance), encoding="utf-8")
                with self.assertRaises(ValueError):
                    self.collect(plan, directory)

    def test_newer_failed_job_never_falls_back_even_without_an_artifact(self):
        plan = self.plan()
        for has_artifact in (False, True):
            with (
                self.subTest(has_artifact=has_artifact),
                tempfile.TemporaryDirectory() as root,
            ):
                directory = Path(root)
                self.jobs = []
                self.write_point(directory, plan["cells"][0], attempt=1)
                if has_artifact:
                    self.write_point(directory, plan["cells"][0], attempt=2)
                    self.jobs[-1]["conclusion"] = "failure"
                else:
                    self.add_job(plan["cells"][0], attempt=2, conclusion="failure")
                records, manifest = self.collect(plan, directory, "failure")
                self.assertEqual(records, [])
                self.assertEqual(manifest["status"], "partial")
                self.assertIn("failure", manifest["missing"][0]["reason"])

    def test_latest_job_missing_metrics_or_artifact_cannot_reuse_old_result(self):
        plan = self.plan()
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            self.write_point(directory, plan["cells"][0], attempt=1)
            self.write_point(directory, plan["cells"][1], attempt=1)
            self.write_point(directory, plan["cells"][0], attempt=2, interactivity=None)
            self.add_job(plan["cells"][1], attempt=2)
            records, manifest = self.collect(plan, directory)
        self.assertEqual(records, [])
        self.assertEqual(manifest["status"], "partial")

    def test_missing_metric_fails_the_point_job_so_failed_jobs_reruns_it(self):
        plan = self.plan()
        with tempfile.TemporaryDirectory() as root:
            path = self.write_point(
                Path(root), plan["cells"][0], interactivity_p90_itl=None
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_DIR / "pr_sweep.py"),
                    "validate-point",
                    str(path),
                    "--hardware",
                    "mi355x",
                ],
                env={**os.environ, "GITHUB_REPOSITORY": "ROCm/ATOM"},
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing a required P90 metric", result.stderr)

    def test_manual_run_cli_plan_validate_and_collect(self):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            config = directory / "recipe.yaml"
            config.write_text(yaml.safe_dump(self.config), encoding="utf-8")
            environment = {
                **os.environ,
                "GITHUB_REPOSITORY": "ROCm/ATOM",
                "GITHUB_RUN_ID": "456",
                "GITHUB_RUN_ATTEMPT": "2",
                "SWEEP_HEAD_SHA": "b" * 40,
                "SWEEP_PR_NUMBER": "",
                "SWEEP_BENCHMARK_RESULT": "success",
                "GITHUB_OUTPUT": str(directory / "outputs"),
                "GITHUB_STEP_SUMMARY": str(directory / "summary.md"),
            }

            def cli(*args):
                return subprocess.run(
                    [sys.executable, str(SCRIPT_DIR / "pr_sweep.py"), *args],
                    env=environment,
                    cwd=directory,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=True,
                )

            cli(
                "plan",
                str(config),
                "--catalog",
                str(ROOT / ".github/benchmark/models_atomesh.yaml"),
            )
            plan = json.loads(
                (directory / "sweep-plan.json").read_text(encoding="utf-8")
            )
            results = directory / "results"
            for cell in plan["cells"]:
                provenance = self.write_point(results, cell)
                cli("validate-point", str(provenance), "--hardware", "mi355x")
            (directory / "sweep-jobs.json").write_text(
                json.dumps(self.jobs), encoding="utf-8"
            )
            cli("collect", str(results))
            records = json.loads(
                (directory / "agg_bmk.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(records), 2)
            self.assertTrue(
                all(record["sweep_status"] == "complete" for record in records)
            )
            outputs = (directory / "outputs").read_text(encoding="utf-8")
            self.assertIn("plan_artifact=sweep-plan-attempt-2", outputs)
            self.assertIn("complete=true", outputs)
            summary = (directory / "summary.md").read_text(encoding="utf-8")
            self.assertIn("Manual", summary)
            self.assertNotIn("PR #0", summary)


if __name__ == "__main__":
    unittest.main()
