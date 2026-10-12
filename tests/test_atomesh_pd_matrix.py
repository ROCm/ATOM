# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU-only tests for ATOMesh Slurm node selection and cell expansion."""

import csv
import importlib.util
import os
import re
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


class MiniMaxAccuracyTest(unittest.TestCase):
    def test_c40_accuracy_preserves_the_reference_service(self):
        cfg = pd_matrix.load_config(
            SCRIPT.parents[2] / "benchmark/models_atomesh.yaml"
        )
        model_name = "MiniMax-M3-MXFP4"
        env = {
            "ATOMESH_MODEL_ROOT": "/models",
            "ATOMESH_SLURM_SUBMIT_RUNNER": "atomesh-cicd",
            "ATOMESH_SLURM_ACCOUNT": "",
            "ATOMESH_SLURM_PARTITION": "",
            "ATOMESH_LOG_ROOT": "/logs",
            "ATOMESH_1P1D_NODES": "",
            "ATOMESH_SINGLE_NODE": "auto",
            "ATOMESH_PD_RANK_MAPPING_POLICY": "none",
        }
        kwargs = dict(
            model_filter={model_name},
            benchmark_kind_filter=None,
            override_image=None,
            override_benchmark_concurrency=None,
            override_eval_concurrency=None,
        )
        with patch.dict(os.environ, env):
            accuracy_cells = pd_matrix.build_cells(
                cfg, suite="accuracy", case_filter=None, **kwargs
            )
            perf = pd_matrix.build_cells(
                cfg,
                suite="nightly",
                case_filter={
                    "minimax-m3-fp4-1p1d-tp4-eagle3-agentic-lmcache-1m-c40"
                },
                **kwargs,
            )[0]
        self.assertEqual(len(accuracy_cells), 1)
        accuracy = accuracy_cells[0]
        self.assertTrue(accuracy["run_eval"])
        self.assertTrue(accuracy["eval_only"])
        self.assertEqual(accuracy["num_nodes"], 1)
        self.assertTrue(accuracy["disaggregated"])
        self.assertEqual(accuracy["concurrency"], [40])
        self.assertEqual(accuracy["accuracy"]["concurrency"], [40])
        self.assertEqual(accuracy["accuracy"]["batch_size"], 40)
        self.assertEqual(accuracy["accuracy"]["threshold"], 0.93)
        self.assertIsNone(accuracy["accuracy"]["limit"])
        self.assertEqual(accuracy["accuracy"]["fewshot"], 5)
        self.assertEqual(accuracy["accuracy"]["max_gen_toks"], 16384)
        self.assertEqual(accuracy["service"], perf["service"])
        self.assertEqual(accuracy["env"], perf["env"])
        self.assertEqual(
            accuracy["server_args"],
            dict(perf["server_args"], spec_decode_acceptance_length=""),
        )
        self.assertEqual(accuracy["benchmark"]["kind"], "random")
        self.assertFalse(perf["run_eval"])
        self.assertEqual(perf["server_args"]["spec_decode_acceptance_length"], 2.78)


class MiniMaxTuningTest(unittest.TestCase):
    CONFIG_PATH = (
        Path(__file__).resolve().parents[1]
        / ".github/benchmark/aiter/minimax_m3_fp4_reduce_tuned_fmoe.csv"
    )
    CONTAINER_PATH = (
        "/workspace/ATOM/.github/benchmark/aiter/"
        "minimax_m3_fp4_reduce_tuned_fmoe.csv"
    )

    def test_tuning_override_reaches_only_the_c40_family(self):
        cfg = pd_matrix.load_config(
            SCRIPT.parents[2] / "benchmark/models_atomesh.yaml"
        )
        env = {
            "ATOMESH_MODEL_ROOT": "/models",
            "ATOMESH_SLURM_SUBMIT_RUNNER": "atomesh-cicd",
            "ATOMESH_SLURM_ACCOUNT": "",
            "ATOMESH_SLURM_PARTITION": "",
            "ATOMESH_LOG_ROOT": "/logs",
            "ATOMESH_1P1D_NODES": "",
            "ATOMESH_SINGLE_NODE": "auto",
            "ATOMESH_PD_RANK_MAPPING_POLICY": "none",
        }
        cases = [
            ("nightly", "minimax-m3-fp4-1p1d-tp4-eagle3-agentic-lmcache-1m-c40", 40),
            ("nightly", "minimax-m3-fp4-1p1d-tp4-eagle3-agentic-lmcache-1m-c48", 48),
            ("accuracy", "minimax-m3-fp4-1p1d-tp4-eagle3-lmcache-gsm8k-only-c40", 40),
            ("nightly", "minimax-m3-fp4-1p1d-tp4", None),
        ]
        self.assertTrue(self.CONFIG_PATH.is_file())
        with patch.dict(os.environ, env):
            for suite, name, concurrency in cases:
                with self.subTest(case=name):
                    cells = pd_matrix.build_cells(
                        cfg,
                        suite=suite,
                        model_filter={"MiniMax-M3-MXFP4"},
                        case_filter={name},
                        benchmark_kind_filter=None,
                        override_image=None,
                        override_benchmark_concurrency=None,
                        override_eval_concurrency=None,
                    )
                    self.assertEqual(len(cells), 1)
                    cell = cells[0]
                    if concurrency is None:
                        self.assertNotIn("AITER_CONFIG_FMOE", cell["env"]["common"])
                        continue
                    self.assertEqual(
                        cell["env"]["common"]["AITER_CONFIG_FMOE"],
                        self.CONTAINER_PATH,
                    )
                    self.assertEqual(cell["num_nodes"], 1)
                    self.assertEqual(cell["concurrency"], [concurrency])
                    self.assertEqual(
                        cell["server_args"]["max_num_batched_tokens"], 32768
                    )
                    for role in ("prefill", "decode"):
                        self.assertEqual(cell["service"][role]["workers"], 1)
                        self.assertEqual(cell["service"][role]["tp"], 4)
                        self.assertEqual(cell["service"][role]["cudagraph_mode"], "FULL")
                        self.assertEqual(
                            cell["env"][role]["AITER_CONFIG_FMOE"],
                            self.CONTAINER_PATH,
                        )
                    self.assertEqual(
                        cell["env"]["prefill"]["HIP_VISIBLE_DEVICES"], "0,1,2,3"
                    )
                    self.assertEqual(
                        cell["env"]["decode"]["HIP_VISIBLE_DEVICES"], "4,5,6,7"
                    )

    def test_tuning_table_covers_decode_and_32k_warmup_with_matching_layouts(self):
        with self.CONFIG_PATH.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        # An explicit AITER_CONFIG_FMOE replaces the default tables: preserve
        # every token bucket, including decode, instead of shipping only 32K.
        by_shape = {(int(row["inter_dim"]), int(row["token"])): row for row in rows}
        expected = {(dim, 2**power) for dim in (384, 768) for power in range(16)}
        self.assertEqual(set(by_shape), expected)
        self.assertEqual(len(rows), len(expected))
        for (dim, tokens), row in by_shape.items():
            with self.subTest(inter_dim=dim, tokens=tokens):
                self.assertEqual(
                    (row["gfx"], row["model_dim"], row["topk"]),
                    ("gfx950", "6144", "5"),
                )
                self.assertEqual(row["expert"], "129")
                # GEMM2 must agree with sorting and the paired GEMM1 tile.
                g1 = re.search(r"_(?:t)?(\d+)x\d+x\d+", row["kernelName1"])
                g2 = re.search(
                    r"_t(\d+)x\d+x\d+_(atomic|reduce)(?:_[a-z]+)*_sbm(\d+)$",
                    row["kernelName2"],
                )
                self.assertIsNotNone(g1)
                self.assertIsNotNone(g2)
                self.assertEqual(g1[1], row["block_m"])
                self.assertEqual(g2[1], row["block_m"])
                self.assertEqual(g2[3], row["block_m"])
                if tokens >= 16384:
                    self.assertEqual(g2[2], "reduce")
                    if dim == 384:
                        self.assertEqual(row["block_m"], "64")


if __name__ == "__main__":
    unittest.main()
