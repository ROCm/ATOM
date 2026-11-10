# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU-only tests for MoRI IO PD accuracy cells and their launcher support."""

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from atomesh_workflow_helpers import (
    github_expressions,
    load_workflow,
    run_matrix_step,
    workflow_step,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / ".github/scripts/atomesh"
SERVER_SCRIPT = SCRIPTS / "pd_server_atom.sh"
SPUR_NODES = ",".join(
    f"ml-ai-ubuntu-gpu-mi350x8-2304gb-fabric-{index}"
    for index in (5, 6, 7, 8, 13, 14, 15, 16)
)


def script_section(start: str, end: str) -> str:
    source = SERVER_SCRIPT.read_text()
    return source[source.index(start) : source.index(end)]


def generate_cells(
    model: str,
    suite: str = "accuracy",
    model_root: str = "/data/models2",
    extra_args: tuple[str, ...] = (),
    env: dict[str, str] | None = None,
) -> list[dict]:
    """Run the matrix over one model's suite on the Spur lane."""
    with tempfile.TemporaryDirectory() as temp:
        output = Path(temp) / "cells.json"
        subprocess.run(
            [
                "python3",
                str(SCRIPTS / "pd_matrix.py"),
                "--suite",
                suite,
                "--model",
                model,
                *extra_args,
                "--output",
                str(output),
            ],
            cwd=ROOT,
            env={
                **os.environ,
                "ATOMESH_SLURM_SUBMIT_RUNNER": "atomesh-cicd-mi350",
                "ATOMESH_SLURM_ACCOUNT": "",
                "ATOMESH_SLURM_PARTITION": "",
                "ATOMESH_LOG_ROOT": "/data/${USER}/logs/ATOMESH_LOG/",
                "ATOMESH_MODEL_ROOT": model_root,
                "ATOMESH_1P1D_NODES": SPUR_NODES,
                "ATOMESH_2P1D_NODES": SPUR_NODES,
                "ATOMESH_PD_RANK_MAPPING_POLICY": "idx2idx",
                **(env or {}),
            },
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        return json.loads(output.read_text())["include"]


class MoriIOCellTest(unittest.TestCase):
    """The catalog cell must reach the launcher as a MoRI IO gsm8k-only run."""

    @classmethod
    def setUpClass(cls):
        cells = generate_cells("Kimi-K2.5-MXFP4")
        cls.cell = next(cell for cell in cells if "moriio" in cell["id"])

    def test_cell_selects_moriio_for_both_roles(self):
        env = self.cell["env"]
        for role in ("common", "prefill", "decode"):
            self.assertEqual(env[role]["KV_CONNECTOR"], "moriio")
        self.assertEqual(env["common"]["ATOM_MORIIO_MODE"], "write")
        # The model-level Mooncake JSON must not survive into a MoRI IO cell.
        self.assertEqual(env["decode"].get("DECODE_KV_TRANSFER_CONFIG"), "")
        self.assertEqual(env["prefill"].get("PREFILL_KV_TRANSFER_CONFIG"), None)

    def test_cell_is_gsm8k_only_across_two_nodes(self):
        self.assertTrue(self.cell["run_eval"])
        self.assertTrue(self.cell["eval_only"])
        self.assertEqual(self.cell["accuracy"]["task"], "gsm8k")
        self.assertEqual(self.cell["accuracy"]["threshold"], 0.90)
        # Chat completions, so the template's thinking switch applies.
        self.assertEqual(self.cell["accuracy"]["model_type"], "local-chat-completions")
        self.assertEqual(self.cell["accuracy"]["endpoint"], "chat/completions")
        self.assertTrue(self.cell["accuracy"]["apply_chat_template"])
        # lm_eval scores content, which a model cut off mid-thought leaves empty.
        self.assertIn(
            '--default-chat-template-kwargs {"thinking":false}',
            self.cell["server_args"]["extra_args"],
        )
        # Prefill and decode own a node each, so KV crosses the fabric.
        self.assertEqual(self.cell["num_nodes"], 2)
        self.assertEqual(self.cell["service"]["prefill"]["tp"], 8)
        self.assertEqual(self.cell["service"]["decode"]["tp"], 8)


class DeepSeekV3AttributionCellsTest(unittest.TestCase):
    """The DeepSeek-V3 pair exists to attribute KV corruption to a connector.

    That only works if the two cells are otherwise identical, so this pins the
    difference down to the connector rather than trusting the YAML merge key.
    """

    @classmethod
    def setUpClass(cls):
        cells = generate_cells("DeepSeek-V3")
        cls.cells = {
            cell["env"]["common"]["KV_CONNECTOR"]: cell
            for cell in cells
            if cell["topology"].startswith("1p1d_tp8_")
        }

    def test_both_connectors_are_present(self):
        self.assertEqual(set(self.cells), {"mooncake", "moriio"})

    def test_the_pair_differs_only_by_the_connector(self):
        mooncake, moriio = self.cells["mooncake"], self.cells["moriio"]
        # Labels are expected to differ; nothing that reaches a server is.
        labels = {"id", "name", "topology", "display_topology", "log_root"}
        differing = {
            key
            for key in set(mooncake) | set(moriio)
            if mooncake.get(key) != moriio.get(key)
        }
        self.assertEqual(differing - labels, {"env"})
        for role in ("common", "prefill", "decode"):
            without = [
                {k: v for k, v in cell["env"][role].items() if k != "KV_CONNECTOR"}
                for cell in (mooncake, moriio)
            ]
            self.assertEqual(without[0], without[1])

    def test_both_cells_ask_moriio_to_push(self):
        # Shared, so the pair still differs only by the connector.
        for cell in self.cells.values():
            self.assertEqual(cell["env"]["common"]["ATOM_MORIIO_MODE"], "write")

    def test_neither_cell_pins_a_kv_transfer_config(self):
        # An explicit JSON would override the built-in config and defeat the
        # comparison, since only one of the two would be honoured.
        for cell in self.cells.values():
            self.assertIsNone(cell["env"]["prefill"].get("PREFILL_KV_TRANSFER_CONFIG"))
            self.assertIsNone(cell["env"]["decode"].get("DECODE_KV_TRANSFER_CONFIG"))

    def test_cells_are_two_node_tp8_gsm8k_only(self):
        for cell in self.cells.values():
            self.assertTrue(cell["eval_only"])
            self.assertEqual(cell["num_nodes"], 2)
            self.assertEqual(cell["service"]["prefill"]["tp"], 8)
            self.assertEqual(cell["service"]["decode"]["tp"], 8)
            self.assertEqual(cell["accuracy"]["task"], "gsm8k")
            self.assertEqual(cell["accuracy"]["endpoint"], "chat/completions")
            self.assertTrue(cell["accuracy"]["apply_chat_template"])
            # V3 does not think, so gsm8k answers fit well inside this.
            self.assertEqual(cell["accuracy"]["max_gen_toks"], 512)


# (model, topology): (nodes, nodes per worker, shared expert kept separate)
LOW_LATENCY_DECODE_CELLS = {
    ("DeepSeek-V3", "1p1d_dpa8_ep8_ll_moriio"): (2, 1, True),
    ("DeepSeek-V3", "1p1d_dpa16_ep16_ll_moriio"): (4, 2, True),
    ("Kimi-K2.5-MXFP4", "1p1d_dpa8_ep8_ll_moriio"): (2, 1, False),
    ("Kimi-K2.5-MXFP4", "1p1d_dpa16_ep16_ll_moriio"): (4, 2, False),
    ("Kimi-K3-MXFP4", "1p1d_dpa16_ep16_ll_moriio"): (4, 2, False),
}
INTER_NODE_LL_GEOMETRY = {
    "MORI_EP_BLOCK_NUM": "64",
    "MORI_EP_RDMA_BLOCK_NUM": "32",
    "MORI_EP_WARP_PER_BLOCK": "8",
}


class LowLatencyDecodeCellsTest(unittest.TestCase):
    """DP attention + MoRI EP over MoRI IO, low latency on decode only."""

    @classmethod
    def setUpClass(cls):
        models = {model for model, _ in LOW_LATENCY_DECODE_CELLS}
        cells = {model: generate_cells(model) for model in models}
        cls.cells = {
            (model, topology): next(
                cell for cell in cells[model] if cell["topology"] == topology
            )
            for model, topology in LOW_LATENCY_DECODE_CELLS
        }

    def each_cell(self):
        for key, shape in LOW_LATENCY_DECODE_CELLS.items():
            with self.subTest(cell=key):
                yield self.cells[key], shape

    def test_only_decode_runs_low_latency(self):
        for cell, _ in self.each_cell():
            prefill = cell["service"]["prefill"]["extra_args"]
            decode = cell["service"]["decode"]["extra_args"]
            for args in (prefill, decode):
                self.assertIn("--enable-dp-attention --enable-expert-parallel", args)
            self.assertNotIn("--all2all-backend", prefill)
            self.assertIn("--all2all-backend low-latency", decode)
            self.assertEqual(cell["env"]["decode"]["ATOM_MORI_V1_LL"], "1")
            self.assertIsNone(cell["env"]["prefill"].get("ATOM_MORI_V1_LL"))

    def test_cells_span_the_nodes_their_workers_need(self):
        for cell, (nodes, per_worker, _) in self.each_cell():
            self.assertEqual(cell["num_nodes"], nodes)
            for role in ("prefill", "decode"):
                service = cell["service"][role]
                self.assertEqual(service.get("nodes_per_worker", 1), per_worker)

    def test_only_cross_node_decode_sets_the_inter_node_ll_geometry(self):
        for cell, (_, per_worker, _) in self.each_cell():
            env = cell["env"]
            for key in INTER_NODE_LL_GEOMETRY:
                self.assertIsNone(env["prefill"].get(key))
            if per_worker == 1:
                for key in INTER_NODE_LL_GEOMETRY:
                    self.assertIsNone(env["decode"].get(key))
                continue
            decode = {key: env["decode"].get(key) for key in INTER_NODE_LL_GEOMETRY}
            self.assertEqual(decode, INTER_NODE_LL_GEOMETRY)
            self.assertEqual(cell["server_args"]["gpu_memory_utilization"], 0.70)
            for role in ("prefill", "decode"):
                self.assertEqual(env[role]["MORI_ENABLE_HOST_PROXY"], "1")
                self.assertEqual(env[role]["ATOM_DP_LM_HEAD_MODE"], "default")

    def test_only_fp8_moe_keeps_the_shared_expert_separate(self):
        for cell, (_, _, separate) in self.each_cell():
            for role in ("prefill", "decode"):
                fused = cell["env"][role].get("ATOM_FUSE_SHARED_EXPERT")
                self.assertEqual(fused, "0" if separate else None)

    def test_cells_are_moriio_gsm8k_runs(self):
        for cell, _ in self.each_cell():
            self.assertEqual(cell["env"]["common"]["KV_CONNECTOR"], "moriio")
            self.assertEqual(cell["env"]["common"]["MORI_SHMEM_HEAP_SIZE"], "16G")
            # MoRI sizes its EP buffers from this per-rank budget, and 32768
            # overflows the 16 GiB heap at EP16.
            self.assertLessEqual(cell["server_args"]["max_num_batched_tokens"], 16384)
            self.assertTrue(cell["eval_only"])
            self.assertEqual(cell["accuracy"]["threshold"], 0.90)


class KimiK3MoriIOCellTest(unittest.TestCase):
    """Kimi-K3's PD gsm8k cells, over MoRI IO, with nothing the connector refuses."""

    @classmethod
    def setUpClass(cls):
        cls.cells = {cell["topology"]: cell for cell in generate_cells("Kimi-K3-MXFP4")}
        cls.cell = cls.cells["1p1d_tp8_moriio"]
        cls.dp_attention = cls.cells["1p1d_dpa16_ep16_ll_moriio"]

    def each_cell(self):
        for topology, cell in self.cells.items():
            with self.subTest(topology=topology):
                yield cell

    def test_it_runs_over_moriio_with_the_built_in_config(self):
        for cell in self.each_cell():
            env = cell["env"]
            for role in ("common", "prefill", "decode"):
                self.assertEqual(env[role]["KV_CONNECTOR"], "moriio")
                self.assertEqual(env[role]["ATOM_MORIIO_MODE"], "write")
            self.assertIsNone(env["prefill"].get("PREFILL_KV_TRANSFER_CONFIG"))
            self.assertIsNone(env["decode"].get("DECODE_KV_TRANSFER_CONFIG"))

    def test_nothing_the_connector_refuses_is_on(self):
        for cell in self.each_cell():
            for role in ("common", "prefill", "decode"):
                self.assertEqual(cell["env"][role]["ATOM_ENABLE_REPLAYSSM"], "0")
            server = cell["server_args"]
            self.assertNotIn("num_speculative_tokens", server)
            self.assertIs(server["enable_prefix_caching"], False)

    def test_dp_attention_also_excludes_f_b_proj_from_fp8(self):
        def excluded(cell):
            config = json.loads(cell["server_args"]["online_quant_config"])
            return set(config["exclude_layer"])

        self.assertEqual(
            excluded(self.dp_attention) - excluded(self.cell), {"*self_attn.f_b_proj*"}
        )
        self.assertLessEqual(excluded(self.cell), excluded(self.dp_attention))

    def test_dp_attention_is_scored_like_the_tp8_cell(self):
        self.assertEqual(self.dp_attention["accuracy"], self.cell["accuracy"])
        self.assertEqual(
            self.dp_attention["server_args"]["extra_args"],
            self.cell["server_args"]["extra_args"],
        )

    def test_it_is_a_two_node_tp8_gsm8k_eval_with_thinking_off(self):
        self.assertTrue(self.cell["eval_only"])
        self.assertEqual(self.cell["num_nodes"], 2)
        self.assertEqual(self.cell["service"]["prefill"]["tp"], 8)
        self.assertEqual(self.cell["service"]["decode"]["tp"], 8)
        self.assertEqual(self.cell["accuracy"]["task"], "gsm8k")
        self.assertEqual(self.cell["accuracy"]["threshold"], 0.90)
        self.assertIn(
            '--default-chat-template-kwargs {"thinking":false}',
            self.cell["server_args"]["extra_args"],
        )


TW_WEEKLY = "0 16 * * 5"
SPUR_NIGHTLY = "10 16 * * *"
SPUR_MORIIO = "10 4 * * *"
# atomesh-benchmark.yaml's model root for the Spur lane.
SPUR_MODEL_ROOT = "/data/junyyang/models"
MORIIO_MODELS = ("DeepSeek-V3", "Kimi-K2.5-MXFP4", "Kimi-K3-MXFP4")


def is_moriio(cell: dict) -> bool:
    return cell["env"]["common"].get("KV_CONNECTOR") == "moriio"


class MoriIOScheduleTest(unittest.TestCase):
    """MoRI IO accuracy runs every night on the Spur lane, apart from the nightly."""

    @classmethod
    def setUpClass(cls):
        cls.workflow = load_workflow()
        cls.matrix_script = workflow_step(cls.workflow, "matrix")
        cls.moriio_cells = cls.matrix(SPUR_MORIIO)
        cls.nightly_cells = cls.matrix(SPUR_NIGHTLY)

    @classmethod
    def lane(cls, schedule: str) -> dict[str, str]:
        return {
            name: github_expressions(value, "schedule", schedule)
            for name, value in cls.workflow["env"].items()
        }

    @classmethod
    def matrix(cls, schedule: str) -> list[dict]:
        """Run the workflow's matrix step as that schedule would."""
        cells, _ = run_matrix_step(
            cls.matrix_script,
            dict(
                cls.lane(schedule),
                GITHUB_EVENT_NAME="schedule",
                EVENT_SCHEDULE=schedule,
                SUITE="nightly",
                RUN_ALL_MODELS="true",
                MODEL_NAMES="",
                CASE_NAMES="",
                ATOMESH_IMAGE="test-image",
                ATOMESH_BENCHMARK_CONCURRENCY="",
                ATOMESH_EVAL_CONCURRENCY="",
            ),
        )
        return cells

    def test_the_moriio_cron_runs_on_the_spur_lane_like_the_nightly(self):
        # PyYAML reads the workflow's `on:` key as True.
        self.assertIn({"cron": SPUR_MORIIO}, self.workflow[True]["schedule"])
        self.assertEqual(self.lane(SPUR_MORIIO), self.lane(SPUR_NIGHTLY))
        runner = "ATOMESH_SLURM_SUBMIT_RUNNER"
        self.assertEqual(self.lane(SPUR_MORIIO)[runner], "atomesh-cicd-mi350")
        self.assertEqual(self.lane(TW_WEEKLY)[runner], "atomesh-cicd")

    def test_the_moriio_cron_has_a_concurrency_group_of_its_own(self):
        group = self.workflow["concurrency"]["group"]
        groups = {
            schedule: github_expressions(group, "schedule", schedule)
            for schedule in (TW_WEEKLY, SPUR_NIGHTLY, SPUR_MORIIO)
        }
        self.assertEqual(len(set(groups.values())), 3)
        prefix = "Atomesh Benchmark-refs/heads/main-"
        self.assertEqual(groups[TW_WEEKLY], prefix + "schedule-atomesh-cicd")
        self.assertEqual(groups[SPUR_NIGHTLY], prefix + "schedule-atomesh-cicd-mi350")
        self.assertEqual(
            github_expressions(group, "push"), prefix + "push-atomesh-cicd"
        )

    def test_it_runs_every_moriio_accuracy_cell_and_nothing_else(self):
        expected = {
            (model, cell["name"]): cell
            for model in MORIIO_MODELS
            for cell in generate_cells(model, "accuracy", SPUR_MODEL_ROOT)
            if is_moriio(cell)
        }
        self.assertTrue(expected)
        self.assertEqual(
            sorted((cell["model"], cell["name"]) for cell in self.moriio_cells),
            sorted(expected),
        )
        for cell in self.moriio_cells:
            twin = expected[(cell["model"], cell["name"])]
            for key in ("suite", "service", "server_args", "env", "accuracy"):
                self.assertEqual(cell[key], twin[key], (cell["name"], key))

    def test_the_nightly_leaves_moriio_to_its_own_schedule(self):
        self.assertTrue(self.nightly_cells)
        self.assertFalse([cell["id"] for cell in self.nightly_cells if is_moriio(cell)])

    def test_deepseek_v3_finds_its_weights_on_the_spur_lane(self):
        dsv3 = [cell for cell in self.moriio_cells if cell["model"] == "DeepSeek-V3"]
        self.assertTrue(dsv3)
        for cell in dsv3:
            self.assertEqual(cell["model_path"], "/data/models2/DeepSeek-V3/")

    def test_an_unset_connector_counts_as_mooncake(self):
        nightly = generate_cells("Kimi-K2.5-MXFP4", "nightly", SPUR_MODEL_ROOT)
        mooncake = generate_cells(
            "Kimi-K2.5-MXFP4",
            "nightly",
            SPUR_MODEL_ROOT,
            extra_args=("--kv-connector", "mooncake"),
        )
        self.assertTrue(nightly)
        self.assertFalse(
            [cell for cell in nightly if "KV_CONNECTOR" in cell["env"]["common"]]
        )
        self.assertEqual(
            [cell["id"] for cell in mooncake], [cell["id"] for cell in nightly]
        )

    def test_a_connector_filter_drops_a_cell_before_checking_its_nodes(self):
        # Two candidate nodes: the four-node MoRI 2P2D cell fails its node check
        # unless --kv-connector mooncake drops it first.
        two_nodes = ",".join(SPUR_NODES.split(",")[:2])
        pool = {"ATOMESH_1P1D_NODES": two_nodes, "ATOMESH_2P1D_NODES": two_nodes}
        mooncake = generate_cells(
            "DeepSeek-V3",
            "accuracy",
            SPUR_MODEL_ROOT,
            extra_args=("--kv-connector", "mooncake"),
            env=pool,
        )
        names = [cell["name"] for cell in mooncake]
        self.assertIn("ds-v3-1p1d-tp8-mooncake-gsm8k", names)
        self.assertNotIn("ds-v3-2p2d-dpa16-ep16-ll-moriio-gsm8k", names)
        with self.assertRaises(subprocess.CalledProcessError) as unfiltered:
            generate_cells("DeepSeek-V3", "accuracy", SPUR_MODEL_ROOT, env=pool)
        self.assertIn("needs at least 4 node(s)", unfiltered.exception.stderr)

    def test_atomesh_kv_connector_sets_the_filter_like_the_flag(self):
        flag = generate_cells(
            "DeepSeek-V3",
            "accuracy",
            SPUR_MODEL_ROOT,
            extra_args=("--kv-connector", "moriio"),
        )
        from_env = generate_cells(
            "DeepSeek-V3",
            "accuracy",
            SPUR_MODEL_ROOT,
            env={"ATOMESH_KV_CONNECTOR": "moriio"},
        )
        everything = generate_cells("DeepSeek-V3", "accuracy", SPUR_MODEL_ROOT)
        self.assertTrue(from_env)
        self.assertEqual(from_env, flag)
        self.assertLess(len(from_env), len(everything))


class KvConnectorTest(unittest.TestCase):
    """KV_CONNECTOR picks the built-in config and guards role overrides."""

    @classmethod
    def setUpClass(cls):
        cls.functions = script_section(
            "prepare_kv_connector() {", 'host_ip="$(echo "${IPADDRS}"'
        )

    def run_shell(self, script, **env):
        base = {
            key: value
            for key, value in os.environ.items()
            if key not in ("KV_CONNECTOR", "ATOM_HOST_IP")
        }
        return subprocess.run(
            ["bash", "-c", "set -euo pipefail\n" + self.functions + script],
            env={**base, **env},
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

    def test_built_in_config_defaults_to_mooncake(self):
        result = self.run_shell('kv_transfer_config kv_producer "" 10.0.0.1 6301')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            json.loads(result.stdout),
            {
                "kv_role": "kv_producer",
                "kv_connector": "mooncake",
                "proxy_ip": "10.0.0.1",
                "handshake_port": 6301,
            },
        )

    def test_built_in_config_uses_the_selected_connector_and_port(self):
        result = self.run_shell(
            'kv_transfer_config kv_consumer "" 10.0.0.2 6309', KV_CONNECTOR="moriio"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            json.loads(result.stdout),
            {
                "kv_role": "kv_consumer",
                "kv_connector": "moriio",
                "proxy_ip": "10.0.0.2",
                "handshake_port": 6309,
                "moriio_mode": "write",
            },
        )

    def test_role_override_is_passed_through_untouched(self):
        override = '{"kv_role":"kv_consumer","kv_connector":"multi","connectors":[]}'
        result = self.run_shell(f"kv_transfer_config kv_consumer '{override}' ip 6301")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, override)

    def test_override_naming_another_connector_fails(self):
        override = '{"kv_role":"kv_consumer","kv_connector":"mooncake"}'
        result = self.run_shell(
            f"kv_transfer_config kv_consumer '{override}' ip 6301",
            KV_CONNECTOR="moriio",
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("names 'mooncake'", result.stderr)

    def test_moriio_pins_the_advertised_host_ip(self):
        # get_ip() would otherwise pick whichever address owns the default
        # route, which is not the one the peer role dials.
        result = self.run_shell(
            'host_ip=10.19.0.2; prepare_kv_connector; echo "ip=${ATOM_HOST_IP}"',
            KV_CONNECTOR="moriio",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ip=10.19.0.2", result.stdout)

    def test_explicit_host_ip_is_preserved(self):
        result = self.run_shell(
            'host_ip=10.19.0.2; prepare_kv_connector; echo "ip=${ATOM_HOST_IP}"',
            KV_CONNECTOR="moriio",
            ATOM_HOST_IP="10.9.9.9",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ip=10.9.9.9", result.stdout)

    def test_unknown_connector_fails_before_any_server_starts(self):
        result = self.run_shell("prepare_kv_connector", KV_CONNECTOR="nixl")
        self.assertEqual(result.returncode, 2)
        self.assertIn("unsupported KV_CONNECTOR=nixl", result.stderr)


class ModelPathFallbackTest(unittest.TestCase):
    """Catalog paths name the HF org; some clusters store the weights flat."""

    @classmethod
    def setUpClass(cls):
        cls.functions = script_section(
            "resolve_model_path() {", 'MODEL_PATH="$(resolve_model_path'
        )

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def resolve(self, path):
        body = "set -euo pipefail\n" + self.functions + 'resolve_model_path "$1"'
        result = subprocess.run(
            ["bash", "-c", body, "_", path],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout

    def test_org_prefixed_path_falls_back_to_the_flat_layout(self):
        (self.root / "Kimi-K2.5-MXFP4").mkdir()
        flat = self.resolve(f"{self.root}/amd/Kimi-K2.5-MXFP4/")
        self.assertEqual(flat, f"{self.root}/Kimi-K2.5-MXFP4")

    def test_existing_path_is_used_as_is(self):
        (self.root / "amd/GLM-5.2-MXFP4").mkdir(parents=True)
        path = f"{self.root}/amd/GLM-5.2-MXFP4"
        self.assertEqual(self.resolve(path), path)

    def test_unknown_and_empty_paths_are_left_for_the_engine_to_report(self):
        self.assertEqual(self.resolve(""), "")
        missing = f"{self.root}/amd/Nope"
        self.assertEqual(self.resolve(missing), missing)
        # A relative path is a Hugging Face repo id, not a local directory.
        self.assertEqual(self.resolve("amd/Kimi-K2.5-MXFP4"), "amd/Kimi-K2.5-MXFP4")


class SpurSubmitWrapperTest(unittest.TestCase):
    """Only the workflow chmods the scripts, so nothing may need the exec bit."""

    def test_wrapper_runs_the_job_script_through_bash(self):
        source = (SCRIPTS / "pd_submit.sh").read_text()
        self.assertIn("printf 'exec bash %q\\n' \"${JOB_SCRIPT}\"", source)

    def test_job_script_is_not_marked_executable(self):
        # If this ever flips, the wrapper above stops being load-bearing and
        # the 0644 case silently goes untested.
        mode = (SCRIPTS / "pd_slurm_job.sh").stat().st_mode
        self.assertFalse(mode & 0o111, "pd_slurm_job.sh is executable on disk")


class JobResultTest(unittest.TestCase):
    """A job that dies before its run dir exists still has a result to record."""

    def test_resolve_reports_the_scheduler_code_without_a_run_dir(self):
        with tempfile.TemporaryDirectory() as temp:
            run_dir = Path(temp) / "missing/slurm_job-99"
            result = subprocess.run(
                [
                    "python3",
                    str(SCRIPTS / "pd_job_result.py"),
                    "resolve",
                    "--run-dir",
                    str(run_dir),
                    "--job-id",
                    "99",
                    "--run-token",
                    "token",
                    "--num-ranks",
                    "2",
                    "--scheduler-state",
                    "FAILED",
                    "--scheduler-exit-code",
                    "126:0",
                    "--scheduler-rc",
                    "126",
                    "--spur",
                    "1",
                ],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            # A traceback here would mask the real exit code behind rc=1.
            self.assertEqual(result.returncode, 126, result.stderr)
            self.assertIn("result_state=FAILED", result.stdout)
            payload = json.loads((run_dir / "job-result.json").read_text())
        self.assertEqual(payload["scheduler"]["exit_code"], "126:0")
        self.assertEqual(payload["workload"]["state"], "UNVERIFIED")


class Gsm8kThresholdTest(unittest.TestCase):
    """A below-threshold gsm8k score must fail the job, as SWE-bench already does."""

    @classmethod
    def setUpClass(cls):
        marker = (
            'python3 - "${result_dir}" "${eval_conc}" "${EVAL_THRESHOLD}" <<\'PY\'\n'
        )
        source = SERVER_SCRIPT.read_text()
        start = source.index(marker) + len(marker)
        cls.checker = source[start : source.index("\nPY\n", start)]

    def score(self, results, threshold):
        with tempfile.TemporaryDirectory() as temp:
            result_dir = Path(temp) / "20260101_gsm8k_1p1d_c64"
            (result_dir / "model").mkdir(parents=True)
            (result_dir / "model/results_now.json").write_text(
                json.dumps({"results": results})
            )
            return subprocess.run(
                ["python3", "-", str(result_dir), "64", threshold],
                input=self.checker,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )

    def test_score_at_or_above_threshold_passes(self):
        result = self.score({"gsm8k": {"exact_match,flexible-extract": 0.9356}}, "0.90")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("[eval] PASS", result.stdout)

    def test_score_below_threshold_fails(self):
        result = self.score({"gsm8k": {"exact_match,flexible-extract": 0.0053}}, "0.90")
        self.assertEqual(result.returncode, 1)
        self.assertIn("[eval] FAIL", result.stdout + result.stderr)

    def test_missing_metric_fails_instead_of_reporting_na(self):
        result = self.score({"gsm8k": {}}, "0.90")
        self.assertEqual(result.returncode, 1)
        self.assertIn("[eval] FAIL", result.stdout + result.stderr)

    def test_no_threshold_keeps_the_run_green(self):
        result = self.score({"gsm8k": {"exact_match,flexible-extract": 0.0053}}, "")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("[eval] FAIL", result.stdout)

    def test_a_failing_concurrency_does_not_skip_the_rest(self):
        source = SERVER_SCRIPT.read_text()
        start = source.index("run_eval() {")
        run_eval = source[start : source.index("\n}\n", start) + 3]
        fake_lm_eval = r"""
lm_eval() {
  local out="${@: -1}"
  local conc="${out##*_c}" score=0.95
  if [[ "${conc}" == 8 ]]; then score=0.10; fi
  echo "${conc}" >> "${RUN_DIR}/ran"
  mkdir -p "${out}/model"
  printf '{"results": {"gsm8k": {"exact_match,flexible-extract": %s}}}\n' \
    "${score}" > "${out}/model/results.json"
}
"""
        script = "set -euo pipefail\n" + fake_lm_eval + run_eval + "run_eval\n"
        with tempfile.TemporaryDirectory() as run_dir:
            env = {
                **os.environ,
                "RUN_DIR": run_dir,
                "RUN_EVAL": "true",
                "EVAL_TASK": "gsm8k",
                "EVAL_CONCURRENCY": "8,64",
                "EVAL_THRESHOLD": "0.90",
                "EVAL_FEWSHOT": "3",
                "EVAL_LIMIT": "",
                "EVAL_BATCH_SIZE": "",
                "EVAL_MAX_GEN_TOKS": "",
                "EVAL_APPLY_CHAT_TEMPLATE": "false",
                "EVAL_FEWSHOT_AS_MULTITURN": "false",
                "EVAL_MODEL_TYPE": "local-completions",
                "EVAL_ENDPOINT": "completions",
                "MODEL_PATH": "/models/m",
                "ROUTER_PORT": "8000",
                "TOPOLOGY": "1p1d",
            }
            result = subprocess.run(
                ["bash", "-c", script],
                env=env,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            ran_file = Path(run_dir) / "ran"
            ran = ran_file.read_text().split() if ran_file.exists() else []
        self.assertEqual(ran, ["8", "64"], result.stdout + result.stderr)
        self.assertEqual(result.returncode, 1)
        self.assertIn("[eval] PASS: concurrent=64", result.stdout)
        self.assertIn("[eval] FAIL: concurrency 8 did not pass", result.stderr)


if __name__ == "__main__":
    unittest.main()
