# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Exercise the GSM8K shell runner and score gate without a model or GPU."""

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / ".github/scripts/atomesh"


class GSM8KEvalTest(unittest.TestCase):
    def run_eval(self, score=0.93, threshold="0.93", write_result=True, eval_rc=0):
        source = (SCRIPTS / "pd_server_atom.sh").read_text()
        start = source.index("run_eval() {")
        function = source[start : source.index("\n}\n", start) + 3]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake_eval = root / "lm_eval"
            fake_eval.write_text(
                """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

args = sys.argv[1:]
Path(os.environ["TEST_ARGS"]).write_text(json.dumps(args))
if os.environ["TEST_WRITE_RESULT"] == "1":
    out = Path(args[args.index("--output_path") + 1]) / "model"
    out.mkdir(parents=True)
    metrics = {}
    if os.environ["TEST_SCORE"] != "missing":
        metrics["exact_match,flexible-extract"] = json.loads(os.environ["TEST_SCORE"])
    (out / "results.json").write_text(json.dumps({"results": {"gsm8k": metrics}}))
    # A newer auxiliary JSON must not hide the actual metric summary.
    (out / "samples.json").write_text(json.dumps([{"sample": "answer"}]))
raise SystemExit(int(os.environ["TEST_EVAL_RC"]))
"""
            )
            fake_eval.chmod(0o755)
            env = dict(
                os.environ,
                PATH=f"{root}:{os.environ['PATH']}",
                ATOMESH_SCRIPT_DIR=str(SCRIPTS),
                RUN_EVAL="true",
                EVAL_TASK="gsm8k",
                EVAL_LIMIT="",
                EVAL_BATCH_SIZE="40",
                EVAL_APPLY_CHAT_TEMPLATE="true",
                EVAL_FEWSHOT_AS_MULTITURN="true",
                EVAL_MAX_GEN_TOKS="16384",
                EVAL_MODEL_TYPE="local-chat-completions",
                EVAL_ENDPOINT="chat/completions",
                EVAL_CONCURRENCY="40",
                EVAL_FEWSHOT="5",
                EVAL_THRESHOLD=threshold,
                MODEL_PATH="/models/amd/MiniMax-M3-MXFP4",
                ROUTER_PORT="9000",
                TOPOLOGY="1p1d_eagle3",
                RUN_DIR=str(root / "run"),
                TEST_ARGS=str(root / "args.json"),
                TEST_SCORE="missing" if score == "missing" else json.dumps(score),
                TEST_WRITE_RESULT=str(int(write_result)),
                TEST_EVAL_RC=str(eval_rc),
            )
            result = subprocess.run(
                ["bash", "-eu", "-c", function + "\nrun_eval\n"],
                env=env,
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result, json.loads((root / "args.json").read_text())

    def test_threshold_boundary_and_client_arguments(self):
        result, args = self.run_eval()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("GSM8K threshold passed: 0.9300 >= 0.9300", result.stdout)
        self.assertIn("num_concurrent=40,", args[args.index("--model_args") + 1])
        self.assertIn("max_gen_toks=16384", args[args.index("--model_args") + 1])
        self.assertEqual(args[args.index("--batch_size") + 1], "40")
        self.assertEqual(args[args.index("--num_fewshot") + 1], "5")
        self.assertIn("--apply_chat_template", args)
        self.assertIn("--fewshot_as_multiturn", args)
        self.assertNotIn("--limit", args)

    def test_below_threshold_fails_the_shell(self):
        result, _ = self.run_eval(score=0.929)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("below threshold", result.stderr)
        self.assertNotIn("gsm8k runs done", result.stdout)

    def test_missing_or_invalid_metric_fails(self):
        for score in ("missing", None, "N/A", float("nan"), float("inf"), -1, 1.1):
            with self.subTest(score=score):
                result, _ = self.run_eval(score=score)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("GSM8K score must be a finite number", result.stderr)

    def test_missing_results_fail(self):
        result, _ = self.run_eval(write_result=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("no GSM8K result JSON", result.stderr)

    def test_lm_eval_failure_is_preserved(self):
        result, _ = self.run_eval(eval_rc=7)
        self.assertEqual(result.returncode, 7)
        self.assertNotIn("threshold passed", result.stdout)

    def test_unconfigured_threshold_still_reports_score(self):
        result, _ = self.run_eval(score=0.5, threshold="")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("exact_match,flexible-extract = 0.5", result.stdout)
        self.assertNotIn("threshold passed", result.stdout)

    def test_invalid_threshold_fails(self):
        for threshold in ("nan", "inf", "-0.1", "1.1"):
            with self.subTest(threshold=threshold):
                result, _ = self.run_eval(threshold=threshold)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("GSM8K threshold must be a finite number", result.stderr)


if __name__ == "__main__":
    unittest.main()
