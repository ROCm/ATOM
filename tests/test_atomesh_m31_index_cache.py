# SPDX-License-Identifier: MIT
"""The issue-316 pairs retain the TP4 recipe and use real accuracy acceptance."""

import copy
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def cells(suite):
    with tempfile.TemporaryDirectory() as temp:
        output = Path(temp) / "cells.json"
        subprocess.run(
            [
                "python3",
                str(ROOT / ".github/scripts/atomesh/pd_matrix.py"),
                "--suite",
                suite,
                "--model",
                "MiniMax-M3.1-Flash-Quark-MXFP4",
                "--output",
                str(output),
            ],
            env=dict(
                os.environ,
                ATOMESH_SLURM_SUBMIT_RUNNER="atomesh-cicd",
                ATOMESH_SLURM_ACCOUNT="amd-atom",
                ATOMESH_SLURM_PARTITION="amd-spur",
                ATOMESH_LOG_ROOT="/tmp/logs",
                ATOMESH_MODEL_ROOT="/models",
                ATOMESH_1P1D_NODES="pit2-p03-g12",
                ATOMESH_SINGLE_NODE="pit2-p03-g12",
                ATOMESH_PD_RANK_MAPPING_POLICY="idx2idx",
            ),
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(output.read_text())["include"]


class IndexCacheRecipeTest(unittest.TestCase):
    def test_performance_pair_only_changes_index_reuse(self):
        expanded = cells("mooncake_store")
        pair = [c for c in expanded if "index-freq" in c["name"]]
        self.assertEqual(len(pair), 2)
        baseline = next(
            c
            for c in expanded
            if c["name"]
            == "minimax-m31-flash-mxfp4-1p1d-tp4-dspark7-nvfp4-mcstore-agentx-c32"
        )
        normalized = []
        for cell in [baseline, *pair]:
            c = copy.deepcopy(cell)
            c.pop("id")
            c.pop("name")
            c["server_args"].pop("hf_overrides", None)
            normalized.append(c)
        self.assertEqual(normalized[0], normalized[1])
        self.assertEqual(normalized[0], normalized[2])
        self.assertEqual(pair[0]["benchmark"]["benchmark_duration"], 3600)
        for c in pair:
            for role in ("prefill", "decode"):
                self.assertEqual(c["service"][role]["tp"], 4)
                self.assertEqual(c["server_args"]["max_num_batched_tokens"], 32768)

    def test_both_accuracy_tasks_have_matched_full_dataset_pairs(self):
        pair = [c for c in cells("accuracy") if "index-freq" in c["name"]]
        self.assertEqual(len(pair), 4)
        for c in pair:
            self.assertTrue(c["eval_only"])
            self.assertIsNone(c["accuracy"]["limit"])
            task = c["accuracy"]["task"]
            self.assertEqual(c["accuracy"]["fewshot"], 5 if task == "gsm8k" else 0)
            for role in ("prefill", "decode"):
                self.assertNotIn("spec_decode_acceptance_length", c["server_args"])
                overrides = json.loads(c["server_args"]["hf_overrides"])
                freq = 4 if "freq4" in c["name"] else 1
                self.assertEqual(
                    overrides, {"use_index_cache": freq == 4, "index_topk_freq": freq}
                )


if __name__ == "__main__":
    unittest.main()
