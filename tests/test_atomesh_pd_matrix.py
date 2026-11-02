# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU-only tests for ATOMesh Slurm node selection and cell expansion."""

import importlib.util
import os
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / ".github/scripts/atomesh/pd_matrix.py"
SPEC = importlib.util.spec_from_file_location("atomesh_pd_matrix", SCRIPT)
pd_matrix = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pd_matrix)


def expand_cell(
    runner="atomesh-cicd",
    nodes="",
    layout="multi_node",
    prefill_workers=1,
    decode_workers=1,
    single_node="auto",
    node_pool="",
    topology="test",
    prefill=None,
    decode=None,
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
                "topology": topology,
                "pd_worker_layout": layout,
                "nodes": nodes,
                "prefill": {"workers": prefill_workers, **(prefill or {})},
                "decode": {"workers": decode_workers, **(decode or {})},
                "isl": [128],
                "osl": 128,
                "concurrency": [1],
            },
            override_image=None,
            override_benchmark_concurrency=None,
            override_eval_concurrency=None,
        )


class NodeSelectionTest(unittest.TestCase):
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
                cell = expand_cell(
                    layout=layout,
                    prefill_workers=prefill,
                    decode_workers=decode,
                )
                self.assertEqual(cell["nodes"], [])
                self.assertEqual(cell["num_nodes"], expected)

    def test_tw_explicit_nodes_are_preserved(self):
        nodes = ["mia1-p02-g42", "mia1-p02-g44", "mia1-p02-g47"]
        cell = expand_cell(nodes=",".join(nodes))
        self.assertEqual(cell["nodes"], nodes)
        self.assertEqual(cell["num_nodes"], 3)
        cell = expand_cell(
            nodes=",".join(nodes), layout="single_node", single_node="mia1-p01-g36"
        )
        self.assertEqual(cell["nodes"], ["mia1-p01-g36"])
        self.assertEqual(cell["num_nodes"], 1)

    def test_tw_insufficient_explicit_nodes_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "needs at least 2 node"):
            expand_cell(nodes="mia1-p02-g42")

    def test_node_pool_does_not_constrain_selection(self):
        pool = "pit2-p03-g01,pit2-p03-g03,pit2-p03-g07,pit2-p03-g42"
        for layout, prefill, decode, expected in (
            ("single_node", 1, 1, 1),
            ("multi_node", 1, 1, 2),
            ("multi_node", 2, 1, 3),
            ("prefill_single_node", 2, 1, 2),
            ("decode_single_node", 1, 2, 2),
        ):
            with self.subTest(layout=layout, prefill=prefill, decode=decode):
                cell = expand_cell(
                    node_pool=pool,
                    layout=layout,
                    prefill_workers=prefill,
                    decode_workers=decode,
                )
                self.assertEqual(cell["nodes"], [])
                self.assertEqual(cell["num_nodes"], expected)
        cell = expand_cell(node_pool=pool, nodes="mia1-p02-g42,mia1-p02-g44")
        self.assertEqual(cell["nodes"], ["mia1-p02-g42", "mia1-p02-g44"])
        self.assertEqual(cell["num_nodes"], 2)

    def test_spur_requires_candidates_and_allocates_required_count(self):
        with self.assertRaisesRegex(ValueError, "non-empty Spur nodelist"):
            expand_cell(runner="atomesh-cicd-mi350")
        cell = expand_cell(runner="atomesh-cicd-mi350", nodes="n1,n2,n3")
        self.assertEqual(cell["nodes"], ["n1", "n2", "n3"])
        self.assertEqual(cell["num_nodes"], 2)

    def test_crusoe_uses_automatic_selection(self):
        cell = expand_cell(runner="atomesh-cicd-mi355-crusoe", nodes="n1,n2")
        self.assertEqual(cell["nodes"], [])
        self.assertEqual(cell["num_nodes"], 2)


DPA = {"tp": 8, "extra_args": "--enable-dp-attention --enable-expert-parallel"}


class NodesPerWorkerTest(unittest.TestCase):
    """A worker that spans nodes is one DP-attention instance cut across them."""

    def test_split_cells_count_every_node_of_every_worker(self):
        for prefill_workers, prefill_nodes, decode_nodes, expected in (
            (1, 2, 2, 4),
            (1, 2, 1, 3),
            (1, 1, 2, 3),
            (2, 2, 1, 5),
        ):
            with self.subTest(
                prefill_workers=prefill_workers,
                prefill=prefill_nodes,
                decode=decode_nodes,
            ):
                cell = expand_cell(
                    prefill_workers=prefill_workers,
                    prefill={**DPA, "nodes_per_worker": prefill_nodes},
                    decode={**DPA, "nodes_per_worker": decode_nodes},
                )
                self.assertEqual(cell["num_nodes"], expected)

    def test_aggregated_cells_count_only_the_decode_block(self):
        cell = expand_cell(topology="agg_dpa16", decode={**DPA, "nodes_per_worker": 2})
        self.assertEqual(cell["num_nodes"], 2)
        # Before, the unused prefill block added a node that only idled.
        self.assertEqual(expand_cell(topology="agg")["num_nodes"], 1)
        self.assertEqual(
            expand_cell(topology="agg", layout="single_node")["num_nodes"], 1
        )

    def test_spur_needs_a_candidate_for_every_node(self):
        spanning = {**DPA, "nodes_per_worker": 2}
        cell = expand_cell(
            runner="atomesh-cicd-mi350",
            nodes="n1,n2,n3,n4,n5",
            prefill=spanning,
            decode=spanning,
        )
        self.assertEqual(cell["nodes"], ["n1", "n2", "n3", "n4", "n5"])
        self.assertEqual(cell["num_nodes"], 4)
        with self.assertRaisesRegex(ValueError, "needs at least 4 node"):
            expand_cell(
                runner="atomesh-cicd-mi350",
                nodes="n1,n2,n3",
                prefill=spanning,
                decode=spanning,
            )

    def test_cells_a_spanning_worker_cannot_run_are_rejected(self):
        cases = (
            ("single_node", "decode", DPA, "needs pd_worker_layout multi_node"),
            ("packed_nodes", "prefill", DPA, "needs pd_worker_layout multi_node"),
            ("multi_node", "decode", {**DPA, "tp": 4}, "needs tp 8"),
            (
                "multi_node",
                "prefill",
                {"tp": 8, "extra_args": "--enable-expert-parallel"},
                "needs --enable-dp-attention in prefill.extra_args",
            ),
        )
        for layout, role, role_cfg, message in cases:
            spanning = {role: {**role_cfg, "nodes_per_worker": 2}}
            with (
                self.subTest(layout=layout, role=role, message=message),
                self.assertRaisesRegex(ValueError, f"test-model: {role}.*{message}"),
            ):
                expand_cell(layout=layout, **spanning)

    def test_a_worker_spans_at_least_one_node(self):
        with self.assertRaisesRegex(
            ValueError, "decode.nodes_per_worker must be at least 1"
        ):
            expand_cell(decode={"nodes_per_worker": 0})

    def test_an_aggregated_cell_spans_nodes_through_decode(self):
        with self.assertRaisesRegex(ValueError, "set decode.nodes_per_worker"):
            expand_cell(topology="agg", prefill={**DPA, "nodes_per_worker": 2})


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


class TopologyKindTest(unittest.TestCase):
    """The topology name is what tells a split cell from an aggregated one."""

    def test_only_agg_names_an_aggregated_topology(self):
        for topology, disaggregated in (
            ("1p1d", True),
            ("2p1d", True),
            ("1p1d_dpa", True),
            ("agg", False),
            ("AGG", False),
            ("agg-tp8-dcp8", False),
        ):
            with self.subTest(topology=topology):
                self.assertIs(pd_matrix.is_disaggregated(topology), disaggregated)

    def test_cell_carries_the_kind_downstream(self):
        for topology, disaggregated in (("1p1d", True), ("agg", False)):
            with self.subTest(topology=topology):
                cell = expand_cell(topology=topology, layout="single_node")
                self.assertIs(cell["disaggregated"], disaggregated)


if __name__ == "__main__":
    unittest.main()
