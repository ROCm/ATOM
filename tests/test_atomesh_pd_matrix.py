# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU-only tests for ATOMesh Slurm node selection."""

import importlib.util
import os
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / ".github/scripts/atomesh/pd_matrix.py"
SPEC = importlib.util.spec_from_file_location("atomesh_pd_matrix", SCRIPT)
pd_matrix = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pd_matrix)


class NodeSelectionTest(unittest.TestCase):
    def build_cell(
        self,
        runner="atomesh-cicd",
        nodes="",
        layout="multi_node",
        prefill_workers=1,
        decode_workers=1,
        single_node="auto",
    ):
        with patch.dict(os.environ, {"ATOMESH_SINGLE_NODE": single_node}):
            return pd_matrix.build_cell(
                cfg={
                    "defaults": {"runner": {"slurm_submit_runner": runner}},
                    "backends": {"atom": {"image": "test-image"}},
                },
                model_name="test-model",
                model_cfg={"model_path": "/models/test"},
                suite_name="smoke",
                suite_cfg={
                    "topology": "test",
                    "pd_worker_layout": layout,
                    "nodes": nodes,
                    "prefill": {"workers": prefill_workers},
                    "decode": {"workers": decode_workers},
                    "isl": [128],
                    "osl": 128,
                    "concurrency": [1],
                },
                override_image=None,
                override_benchmark_concurrency=None,
                override_eval_concurrency=None,
            )

    def test_tw_automatic_selection_preserves_required_node_count(self):
        cases = [
            ("single_node", 1, 1, 1),
            ("multi_node", 1, 1, 2),
            ("multi_node", 2, 1, 3),
            ("prefill_single_node", 2, 1, 2),
            ("decode_single_node", 1, 2, 2),
        ]
        for layout, prefill, decode, expected in cases:
            with self.subTest(layout=layout, prefill=prefill, decode=decode):
                cell = self.build_cell(
                    layout=layout,
                    prefill_workers=prefill,
                    decode_workers=decode,
                )
                self.assertEqual(cell["nodes"], [])
                self.assertEqual(cell["num_nodes"], expected)

    def test_tw_explicit_nodes_are_preserved(self):
        nodes = ["mia1-p02-g42", "mia1-p02-g44", "mia1-p02-g47"]
        cell = self.build_cell(nodes=",".join(nodes))
        self.assertEqual(cell["nodes"], nodes)
        self.assertEqual(cell["num_nodes"], 3)
        cell = self.build_cell(
            nodes=",".join(nodes), layout="single_node", single_node="mia1-p01-g36"
        )
        self.assertEqual(cell["nodes"], ["mia1-p01-g36"])
        self.assertEqual(cell["num_nodes"], 1)

    def test_tw_insufficient_explicit_nodes_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "needs at least 2 node"):
            self.build_cell(nodes="mia1-p02-g42")

    def test_spur_requires_candidates_and_allocates_required_count(self):
        with self.assertRaisesRegex(ValueError, "non-empty Spur nodelist"):
            self.build_cell(runner="atomesh-cicd-mi350")
        cell = self.build_cell(runner="atomesh-cicd-mi350", nodes="n1,n2,n3")
        self.assertEqual(cell["nodes"], ["n1", "n2", "n3"])
        self.assertEqual(cell["num_nodes"], 2)

    def test_crusoe_uses_automatic_selection(self):
        cell = self.build_cell(runner="atomesh-cicd-mi355-crusoe", nodes="n1,n2")
        self.assertEqual(cell["nodes"], [])
        self.assertEqual(cell["num_nodes"], 2)


if __name__ == "__main__":
    unittest.main()
