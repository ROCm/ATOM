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
        node_pool="",
    ):
        with patch.dict(
            os.environ,
            {"ATOMESH_SINGLE_NODE": single_node, "ATOMESH_NODE_POOL": node_pool},
        ):
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

    def test_configured_pool_preserves_candidates_and_required_count(self):
        pool = "pit2-p03-g01,pit2-p03-g03,pit2-p03-g07,pit2-p03-g42"
        for layout, prefill, decode, expected in (
            ("single_node", 1, 1, 1),
            ("multi_node", 1, 1, 2),
            ("multi_node", 2, 1, 3),
            ("prefill_single_node", 2, 1, 2),
            ("decode_single_node", 1, 2, 2),
        ):
            with self.subTest(layout=layout, prefill=prefill, decode=decode):
                cell = self.build_cell(
                    node_pool=pool,
                    layout=layout,
                    prefill_workers=prefill,
                    decode_workers=decode,
                )
                self.assertEqual(cell["nodes"], pool.split(","))
                self.assertEqual(cell["num_nodes"], expected)

    def test_configured_pool_validates_explicit_selection(self):
        pool = "pit2-p03-g01,pit2-p03-g03,pit2-p03-g07"
        cell = self.build_cell(node_pool=pool, nodes="pit2-p03-g03,pit2-p03-g07")
        self.assertEqual(cell["nodes"], ["pit2-p03-g03", "pit2-p03-g07"])
        cell = self.build_cell(
            node_pool=pool, layout="single_node", single_node="pit2-p03-g07"
        )
        self.assertEqual(cell["nodes"], ["pit2-p03-g07"])
        for nodes in ("pit2-p03-g01,pit2-p03-g44", "pit2-p03-g01,pit2-p03-g01"):
            with self.subTest(nodes=nodes), self.assertRaises(ValueError):
                self.build_cell(node_pool=pool, nodes=nodes)

    def test_configured_pool_does_not_affect_other_runners(self):
        for runner in ("atomesh-cicd-mi350", "atomesh-cicd-mi355-crusoe"):
            with self.subTest(runner=runner):
                cell = self.build_cell(
                    runner=runner, nodes="n1,n2,n3", node_pool="tw1,tw2"
                )
                expected = ["n1", "n2", "n3"] if runner == "atomesh-cicd-mi350" else []
                self.assertEqual(cell["nodes"], expected)
                self.assertEqual(cell["num_nodes"], 2)

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


class ModelPathSelectionTest(unittest.TestCase):
    def test_runner_path_and_explicit_override(self):
        config = {
            "model_path": "${MODEL_ROOT}/native/model",
            "model_path_by_runner": {"special-runner": "/shared/native/model"},
        }
        with patch.dict(os.environ, {"MODEL_ROOT": "/models"}, clear=True):
            self.assertEqual(
                pd_matrix.resolve_model_path("test-model", config, "special-runner"),
                "/shared/native/model",
            )
            self.assertEqual(
                pd_matrix.resolve_model_path("test-model", config, "other-runner"),
                "/models/native/model",
            )
            os.environ["ATOMESH_MODEL_PATH_TEST_MODEL"] = "/explicit/model"
            self.assertEqual(
                pd_matrix.resolve_model_path("test-model", config, "special-runner"),
                "/explicit/model",
            )


if __name__ == "__main__":
    unittest.main()
