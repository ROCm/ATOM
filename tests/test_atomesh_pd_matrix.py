# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU-only tests for ATOMesh Slurm node selection."""

import argparse
import asyncio
import importlib.util
import json
import os
import shlex
import subprocess
import sys
import tempfile
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


class SurveyConfigurationTest(unittest.TestCase):
    def test_weight_preflight_visible_missing_and_unknown(self):
        path = SCRIPT.parent / "pd_survey_preflight.py"
        spec = importlib.util.spec_from_file_location("survey_preflight", path)
        preflight = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(preflight)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "Qwen/Qwen3-0.6B"
            self.assertEqual(
                preflight.check_weights(model, root / "unmounted")["status"], "UNKNOWN"
            )
            self.assertEqual(
                preflight.check_weights(model, root)["status"], "BLOCKED_ENV"
            )
            model.mkdir(parents=True)
            (model / "config.json").write_text(
                json.dumps(
                    {
                        "architectures": ["Qwen3ForCausalLM"],
                        "text_config": {
                            "quantization_config": {"quant_method": "mxfp4"}
                        },
                    }
                )
            )
            for name in ("tokenizer.json", "tokenizer_config.json"):
                (model / name).write_text("{}")
            (model / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"a": "shard.safetensors"}})
            )
            self.assertEqual(
                preflight.check_weights(model, root)["status"], "BLOCKED_ENV"
            )
            (model / "shard.safetensors").write_bytes(b"12345678")
            report = preflight.check_weights(model, root)
            self.assertEqual(report["status"], "FILES_VISIBLE")
            self.assertEqual(report["quantization_config"]["quant_method"], "mxfp4")
            for text_config in ({}, {"quantization_config": None}):
                with self.subTest(text_config=text_config):
                    (model / "config.json").write_text(
                        json.dumps(
                            {
                                "text_config": text_config,
                                "quantization_config": {"quant_method": "mxfp8"},
                            }
                        )
                    )
                    report = preflight.check_weights(model, root)
                    self.assertEqual(report["status"], "FILES_VISIBLE")
                    self.assertEqual(
                        report["quantization_config"]["quant_method"], "mxfp8"
                    )
            (model / "tokenizer.json").unlink()
            (model / "tokenizer_config.json").write_text(
                json.dumps(
                    {
                        "auto_map": {
                            "AutoTokenizer": [
                                "tokenization_kimi.TikTokenTokenizer",
                                None,
                            ]
                        }
                    }
                )
            )
            for name in ("tiktoken.model", "tokenization_kimi.py", "encoding_k3.py"):
                self.assertEqual(
                    preflight.check_weights(model, root)["status"], "BLOCKED_ENV"
                )
                (model / name).write_bytes(b"fixture - not executed")
            self.assertEqual(
                preflight.check_weights(model, root)["status"], "FILES_VISIBLE"
            )
            (model / "tiktoken.model").write_bytes(b"")
            self.assertEqual(
                preflight.check_weights(model, root)["status"], "BLOCKED_ENV"
            )

    def test_smoke_workload_is_bounded_and_never_profiles(self):
        import httpx

        path = SCRIPT.parent / "pd_vllm_profile.py"
        spec = importlib.util.spec_from_file_location("survey_profile", path)
        profile = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(profile)
        calls = []
        fault = {"zero_external": False, "reset_false": False}
        counters = {
            host: {
                "local_compute": 0,
                "local_cache_hit": 0,
                "external_kv_transfer": 0,
                "request_success": 0,
            }
            for host in ("prefill", "decode")
        }

        def respond(request):
            calls.append(request.url.path)
            if request.url.path == "/metrics":
                c = counters[request.url.host]
                text = "\n".join(
                    f'vllm:prompt_tokens_by_source_total{{source="{k}"}} {v}'
                    for k, v in c.items()
                    if k != "request_success"
                )
                text += f'\nvllm:request_success_total{{finished_reason="length"}} {c["request_success"]}'
                return httpx.Response(200, text=text)
            if request.url.path == "/tokenize":
                return httpx.Response(200, json={"tokens": list(range(4096))})
            if request.url.path == "/reset_prefix_cache":
                return httpx.Response(200, json={"success": not fault["reset_false"]})
            self.assertEqual(request.url.path, "/v1/completions")
            body = json.loads(request.content)
            self.assertLessEqual(len(body["prompt"]), 2050)
            self.assertLessEqual(body["max_tokens"], 16)
            c = counters[request.url.host]
            c["request_success"] += 1
            if request.url.host == "decode":
                self.assertTrue(
                    body.get("kv_transfer_params", {}).get("do_remote_prefill")
                )
                c["external_kv_transfer"] += (
                    0 if fault["zero_external"] else len(body["prompt"]) - 1
                )
                c["local_compute"] += (
                    len(body["prompt"]) if fault["zero_external"] else 1
                )
            else:
                c["local_compute"] += len(body["prompt"]) - int(
                    bool(body.get("kv_transfer_params"))
                )
            result = {
                "choices": [
                    {
                        "text": "consistent output",
                        "finish_reason": "length",
                        "prompt_token_ids": body["prompt"],
                        "token_ids": [7] * body["max_tokens"],
                    }
                ]
            }
            if body.get("kv_transfer_params", {}).get("do_remote_decode"):
                result["kv_transfer_params"] = {
                    "remote_block_ids": [1],
                    "remote_host": "prefill",
                }
            return httpx.Response(200, json=result)

        client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(profile.httpx, "AsyncClient", return_value=client),
        ):
            args = argparse.Namespace(
                output=Path(tmp),
                prefill="http://prefill",
                decode="http://decode",
                model="Kimi-K3",
                phase="benchmark",
                mode="smoke",
                tp=8,
                dcp=8,
                hybrid=True,
            )
            asyncio.run(profile.run(args))
            complete = json.loads((Path(tmp) / "complete.json").read_text())
            self.assertEqual(complete["requests"], 10)
            self.assertEqual(complete["direct_pd_text_checks"], 8)
            self.assertEqual(complete["status"], "PENDING_REVIEW")
            evidence = json.loads(
                (Path(tmp) / "correctness-1025-evidence.json").read_text()
            )
            self.assertEqual(
                evidence["producer_effective_prompt_tokens_expected"], 1024
            )
            self.assertEqual(
                evidence["producer_effective_prompt_tokens_observed"], 1024
            )
            self.assertEqual(
                evidence["token_counter_deltas"]["decode"]["external_kv_transfer"], 1024
            )
            self.assertTrue(evidence["accounting_checked"])
            self.assertNotIn("/start_profile", calls)
        for key, message in (
            ("zero_external", "No external"),
            ("reset_false", "cache reset failed"),
        ):
            fault[key] = True
            client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
            with (
                tempfile.TemporaryDirectory() as tmp,
                patch.object(profile.httpx, "AsyncClient", return_value=client),
            ):
                args.output = Path(tmp)
                with self.assertRaisesRegex(AssertionError, message):
                    asyncio.run(profile.run(args))
                self.assertFalse((Path(tmp) / "complete.json").exists())
                if key == "zero_external":
                    evidence = json.loads(
                        (Path(tmp) / "correctness-127-evidence.json").read_text()
                    )
                    self.assertEqual(evidence["status"], "FAIL")
            fault[key] = False

    def test_m3_nixl_protocol_and_failed_transfer_evidence(self):
        import httpx

        spec = importlib.util.spec_from_file_location(
            "m3_smoke", SCRIPT.parent / "pd_m3_nixl_smoke.py"
        )
        smoke = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(smoke)
        for fault in (
            "none",
            "failure",
            "bytes_first",
            "missing",
            "overcount",
            "reference_late",
            "reference_missing",
            "reference_baseline_missing",
            "decode_baseline_missing",
            "partial_label_baseline_missing",
            "after_series_missing",
            "usage",
        ):
            with self.subTest(fault=fault):
                counters = {
                    role: {
                        "local_compute": 0,
                        "local_cache_hit": 0,
                        "external_kv_transfer": 0,
                        "success": 0,
                        "bytes": 0,
                        "count": 0,
                        "failed": 0,
                    }
                    for role in ("prefill", "decode")
                }
                delayed = {}
                calls = []
                handoff = {
                    "do_remote_prefill": True,
                    "do_remote_decode": False,
                    "remote_engine_id": "p-engine",
                    "remote_host": "prefill",
                    "remote_port": 15559,
                    "remote_block_ids": [[1, 2], [3, 4]],
                    "remote_request_id": "p-request",
                }

                def respond(
                    request,
                    counters=counters,
                    delayed=delayed,
                    calls=calls,
                    handoff=handoff,
                    fault=fault,
                ):
                    role = request.url.host
                    c = counters[role]
                    if request.url.path == "/metrics":
                        if role in delayed:
                            remaining, updates = delayed[role]
                            if remaining == 0:
                                for key, value in updates.items():
                                    c[key] += value
                                del delayed[role]
                            else:
                                delayed[role] = (remaining - 1, updates)
                        text = "\n".join(
                            f'vllm:prompt_tokens_by_source_total{{source="{key}"}} {c[key]}'
                            for key in smoke.SOURCES
                        )
                        text += (
                            f'\nvllm:request_success_total{{engine="0"}} {c["success"]}'
                        )
                        text += f'\nvllm:nixl_bytes_transferred_sum{{engine="0"}} {c["bytes"]}'
                        text += f'\nvllm:nixl_bytes_transferred_count{{engine="0"}} {c["count"]}'
                        for name in smoke.FAILURES:
                            if fault != "missing" or role == "prefill":
                                text += f'\n{name}{{engine="0"}} {c["failed"]}'
                        if (fault == "reference_baseline_missing" and not calls) or (
                            fault == "decode_baseline_missing"
                            and role == "decode"
                            and c["success"] == 0
                        ):
                            text = ""
                        if (
                            fault == "partial_label_baseline_missing"
                            and role == "decode"
                        ):
                            # One absent label must not disappear into another label's sum.
                            text += '\nvllm:nixl_bytes_transferred_sum{engine="1"} 0'
                            if c["success"] == 0:
                                text = "\n".join(
                                    line
                                    for line in text.splitlines()
                                    if not line.startswith(
                                        'vllm:nixl_bytes_transferred_sum{engine="0"}'
                                    )
                                )
                        if (
                            fault == "after_series_missing"
                            and role == "decode"
                            and c["success"]
                        ):
                            text = "\n".join(
                                line
                                for line in text.splitlines()
                                if not line.startswith(
                                    "vllm:nixl_bytes_transferred_count"
                                )
                            )
                        return httpx.Response(200, text=text)
                    if request.url.path == "/tokenize":
                        return httpx.Response(200, json={"tokens": list(range(1024))})
                    self.assertEqual(request.url.path, "/v1/completions")
                    body = json.loads(request.content)
                    calls.append((role, body))
                    n = len(body["prompt"])
                    self.assertTrue(body["return_token_ids"])
                    result = {
                        "choices": [
                            {
                                "text": "same",
                                "finish_reason": "length",
                                "prompt_token_ids": body["prompt"],
                                "token_ids": [7] * body["max_tokens"],
                            }
                        ],
                        "usage": {
                            "prompt_tokens": n,
                            "completion_tokens": body["max_tokens"],
                        },
                    }
                    updates = {"success": 1, "local_compute": n}
                    if role == "decode":
                        self.assertEqual(body["kv_transfer_params"], handoff)
                        c["bytes"] += 8192
                        c["count"] += 1
                        c["failed"] += int(fault == "failure")
                        updates = {
                            "success": 1,
                            "external_kv_transfer": n,
                            "local_compute": n if fault == "overcount" else 0,
                        }
                        if fault == "usage":
                            result["usage"]["completion_tokens"] = 15
                    elif "kv_transfer_params" in body:
                        self.assertNotIn("transfer_id", body["kv_transfer_params"])
                        self.assertIsNone(
                            body["kv_transfer_params"]["remote_engine_id"]
                        )
                        self.assertNotIn(
                            "prefill", delayed, "reference was not flushed"
                        )
                        result["kv_transfer_params"] = handoff
                    reference = role == "prefill" and "kv_transfer_params" not in body
                    if (fault == "bytes_first" and role == "decode") or (
                        reference and fault in ("reference_late", "reference_missing")
                    ):
                        delayed[role] = (
                            100 if fault == "reference_missing" else 2,
                            updates,
                        )
                    else:
                        for key, value in updates.items():
                            c[key] += value
                    return httpx.Response(200, json=result)

                client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
                from unittest.mock import AsyncMock

                with (
                    tempfile.TemporaryDirectory() as tmp,
                    patch.object(smoke.httpx, "AsyncClient", return_value=client),
                    patch.object(smoke.asyncio, "sleep", new=AsyncMock()),
                ):
                    args = argparse.Namespace(
                        output=Path(tmp),
                        prefill="http://prefill",
                        decode="http://decode",
                        model="M3",
                    )
                    if fault in ("failure", "overcount", "usage"):
                        with self.assertRaises(AssertionError):
                            asyncio.run(smoke.run(args))
                        evidence = json.loads(
                            (args.output / "request-127.json").read_text()
                        )
                        self.assertEqual(evidence["status"], "FAIL")
                        self.assertFalse((args.output / "complete.json").exists())
                    elif fault in (
                        "missing",
                        "reference_missing",
                        "reference_baseline_missing",
                        "decode_baseline_missing",
                        "partial_label_baseline_missing",
                        "after_series_missing",
                    ):
                        asyncio.run(smoke.run(args))
                        self.assertFalse((args.output / "complete.json").exists())
                        evidence = json.loads(
                            (args.output / "pending.json").read_text()
                        )
                        self.assertFalse(evidence["accounting_checked"])
                        reference_fault = fault in (
                            "reference_missing",
                            "reference_baseline_missing",
                        )
                        self.assertEqual(len(calls), 1 if reference_fault else 3)
                        if (
                            "baseline_missing" in fault
                            or fault == "after_series_missing"
                        ):
                            key = (
                                "reference_metric_deltas"
                                if reference_fault
                                else "labeled_metric_deltas"
                            )
                            role = "prefill" if reference_fault else "decode"
                            unknown = evidence[key + "_unknown_series"][role]
                            self.assertTrue(unknown)
                            self.assertTrue(
                                all(
                                    evidence[key][role][name] is None
                                    for name in unknown
                                )
                            )
                    else:
                        asyncio.run(smoke.run(args))
                        self.assertEqual(len(calls), 9)
                        evidence = json.loads(
                            (args.output / "request-513.json").read_text()
                        )
                        self.assertTrue(evidence["accounting_checked"])
                        self.assertEqual(evidence["nixl_bytes"], 8192)
                        self.assertEqual(evidence["external_tokens"], 513)
                        self.assertEqual(
                            evidence["remote_block_groups"], [[1, 2], [3, 4]]
                        )
                        self.assertEqual(
                            json.loads((args.output / "complete.json").read_text())[
                                "status"
                            ],
                            "PENDING_REVIEW",
                        )

    def test_transport_case_is_bounded_image_only_and_keeps_weight_gate(self):
        root = SCRIPT.parents[3]
        with patch.dict(
            os.environ,
            {
                "ATOMESH_SLURM_ACCOUNT": "amd-frameworks",
                "ATOMESH_SLURM_PARTITION": "amd-spur",
                "ATOMESH_SLURM_SUBMIT_RUNNER": "atomesh-cicd",
                "ATOMESH_MODEL_ROOT": "/mnt/models",
                "ATOMESH_LOG_ROOT": "/it-share/ATOMESH_LOG",
                "ATOMESH_PD_RANK_MAPPING_POLICY": "none",
                "ATOMESH_1P1D_NODES": "pit2-p03-g13,pit2-p03-g42",
                "ATOMESH_NODE_POOL": "pit2-p03-g13,pit2-p03-g42",
            },
        ):
            config = pd_matrix.load_config(
                root / ".github/benchmark/models_atomesh.yaml"
            )
            cells = pd_matrix.build_cells(
                config,
                suite="vllm",
                model_filter={"Transport-vLLM-Survey"},
                case_filter={"survey-transport-image-only-2node"},
                benchmark_kind_filter=None,
                override_image=None,
                override_benchmark_concurrency=None,
                override_eval_concurrency=None,
            )
        self.assertEqual(len(cells), 1)
        cell = cells[0]
        self.assertEqual(cell["num_nodes"], 2)
        self.assertEqual(cell["model_path"], "/mnt/models/Qwen/Qwen3-0.6B")
        self.assertEqual(cell["vllm"]["connector"], "nixl")
        self.assertEqual(cell["precision"], "DIAGNOSTIC_ONLY")
        self.assertEqual(cell["env"]["common"]["ATOMESH_TRANSPORT_ONLY"], "1")
        self.assertEqual(cell["env"]["common"]["ATOMESH_TRANSPORT_GPU"], "1")
        self.assertNotIn("ATOMESH_TRANSPORT_UCX_SELECTION", cell["env"]["common"])
        inputs = json.loads(
            (
                root / ".github/benchmark/rocm-pd-survey-transport-inputs.json"
            ).read_text()
        )
        self.assertEqual(inputs["publish_dashboard"], "false")
        self.assertEqual(inputs["run_all_models"], "false")
        self.assertEqual(inputs["case_names"], cell["name"])

    def test_clean_source_two_nodes_and_real_server_argv(self):
        root = SCRIPT.parents[3]
        env = {
            "ATOMESH_SLURM_ACCOUNT": "amd-frameworks",
            "ATOMESH_SLURM_PARTITION": "amd-spur",
            "ATOMESH_SLURM_SUBMIT_RUNNER": "atomesh-cicd",
            "ATOMESH_LOG_ROOT": "/it-share/ATOMESH_LOG",
            "ATOMESH_PD_RANK_MAPPING_POLICY": "none",
            "ATOMESH_MODEL_ROOT": "/mnt/models",
            "ATOMESH_1P1D_NODES": "pit2-p03-g13,pit2-p03-g42",
            "ATOMESH_NODE_POOL": "pit2-p03-g13,pit2-p03-g42",
        }
        with patch.dict(os.environ, env):
            config = pd_matrix.load_config(
                root / ".github/benchmark/models_atomesh.yaml"
            )
            cells = pd_matrix.build_cells(
                config,
                suite="vllm",
                model_filter={
                    "Kimi-K3-vLLM-Survey",
                    "Qwen3-0.6B-vLLM-Survey",
                    "MiniMax-M3-MXFP8-vLLM-Survey",
                },
                case_filter={
                    "survey-k3-main-read-1p1d-tp8-dcp8-eager",
                    "survey-k3-main-read-dspark3-1p1d-tp8-dcp8-eager",
                    "survey-k3-main-read-apc-1p1d-tp8-dcp8-eager",
                    "survey-k3-main-read-dspark3-apc-1p1d-tp8-dcp8-eager",
                    "survey-qwen3-main-read-1p1d-tp1-eager",
                    "survey-m3-mxfp8-main-nixl-1p1d-tp8-eager",
                },
                benchmark_kind_filter=None,
                override_image=None,
                override_benchmark_concurrency=None,
                override_eval_concurrency=None,
            )
        self.assertEqual(len(cells), 6)
        launcher = root / ".github/scripts/atomesh/pd_server_vllm.sh"
        # Load the actual shell function definitions but never execute installation.
        functions = launcher.read_text().split(
            '\nif [[ -n "${ATOMESH_VLLM_SOURCE_SHA:-}" ]]; then'
        )[0]
        submit = (root / ".github/scripts/atomesh/pd_submit.sh").read_text()
        export_code = submit.split("python3 - <<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
        for cell in cells:
            with self.subTest(case=cell["name"]), tempfile.TemporaryDirectory() as tmp:
                self.assertEqual(cell["num_nodes"], 2)
                self.assertEqual(cell["runner"]["gpus_per_node"], 8)
                self.assertEqual(cell["runner"]["slurm_account"], "amd-frameworks")
                self.assertEqual(
                    cell["vllm"]["source"],
                    {
                        "repo": "https://github.com/vllm-project/vllm",
                        "sha": "b22494cc0cb4bd9db4a62fb107d92429a4a3249d",
                    },
                )
                if cell["model"] == "MiniMax-M3-MXFP8-vLLM-Survey":
                    self.assertEqual(
                        cell["model_path"],
                        "/mnt/models/MiniMaxAI/MiniMax-M3-MXFP8",
                    )
                self.assertNotIn("fork", cell["vllm"])
                self.assertNotIn("lmcache", cell["vllm"])
                self.assertIn("@sha256:659b283", cell["image"])
                exported = subprocess.check_output(
                    [sys.executable, "-c", export_code],
                    text=True,
                    env={**os.environ, "CELL_JSON": json.dumps(cell)},
                )
                shell = (
                    exported
                    + "\n"
                    + "\n".join(
                        [
                            "set -euo pipefail",
                            "export PATH="
                            + shlex.quote(str(Path(sys.executable).parent))
                            + ":$PATH",
                            "host_ip=127.0.0.1; host_name=cpu-fixture; NODE0_ADDR=127.0.0.2",
                            "NODE_RANK=0; HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7",
                            "ATOMESH_SERVICE_PORT_OFFSET=0; ATOMESH_EXECUTION_PHASE=benchmark",
                            "ATOMESH_SCRIPT_DIR=" + shlex.quote(str(launcher.parent)),
                            "RUNTIME_LOG_DIR="
                            + shlex.quote(tmp)
                            + "; RUN_DIR=$RUNTIME_LOG_DIR",
                            "PREFILL_TP_SIZE=$PREFILL_TP; DECODE_TP_SIZE=$DECODE_TP",
                            'PREFILL_SERVER_ARGS="$EXTRA_SERVER_ARGS $PREFILL_EXTRA_SERVER_ARGS"',
                            'DECODE_SERVER_ARGS="$EXTRA_SERVER_ARGS $DECODE_EXTRA_SERVER_ARGS"',
                            # Stubs intercept process launch, not argv construction.
                            "apply_role_env() { :; }; build_server_cache_env() { :; }",
                            "dump_launch_info() { :; }; start_logged_process() { :; }",
                            functions,
                            "start_vllm_server prefill prefill 2584",
                            "start_vllm_server decode decode 2584",
                        ]
                    )
                )
                subprocess.run(
                    ["bash", "-c", shell], check=True, capture_output=True, text=True
                )
                for role in ("prefill", "decode"):
                    argv = json.loads((Path(tmp) / f"{role}.launch.json").read_text())[
                        "argv"
                    ]
                    dense = cell["model"] == "Qwen3-0.6B-vLLM-Survey"
                    self.assertEqual(
                        argv[argv.index("--tensor-parallel-size") + 1],
                        "1" if dense else "8",
                    )
                    nixl = cell["vllm"].get("connector") == "nixl"
                    if nixl:
                        self.assertNotIn("--decode-context-parallel-size", argv)
                        self.assertIn("--no-enable-prefix-caching", argv)
                        self.assertIn("--language-model-only", argv)
                        self.assertNotIn("hybrid", cell["vllm"])
                        self.assertEqual(
                            argv[argv.index("--max-num-batched-tokens") + 1], "512"
                        )
                    elif dense:
                        self.assertNotIn("--decode-context-parallel-size", argv)
                        self.assertNotIn("require_k3_triton", cell["vllm"])
                        self.assertEqual(
                            cell["model_path"], "/mnt/models/Qwen/Qwen3-0.6B"
                        )
                    else:
                        self.assertEqual(
                            argv[argv.index("--decode-context-parallel-size") + 1], "8"
                        )
                    self.assertIn("--enforce-eager", argv)
                    self.assertNotIn("--quantization-config", argv)
                    self.assertNotIn("--profiler-config", argv)
                    transfer = json.loads(argv[argv.index("--kv-transfer-config") + 1])
                    self.assertEqual(transfer["kv_load_failure_policy"], "fail")
                    if nixl:
                        self.assertEqual(transfer["kv_connector"], "NixlConnector")
                        self.assertEqual(
                            transfer["kv_connector_extra_config"], {"backends": ["UCX"]}
                        )
                    else:
                        self.assertEqual(transfer["kv_connector"], "MoRIIOConnector")
                        self.assertTrue(
                            transfer["kv_connector_extra_config"]["read_mode"]
                        )
                        self.assertEqual(
                            transfer["kv_connector_extra_config"]["backend"], "rdma"
                        )
                    if "dspark3" in cell["name"]:
                        spec = json.loads(argv[argv.index("--speculative-config") + 1])
                        self.assertEqual(spec["rejection_sample_method"], "standard")
                        self.assertNotIn("synthetic_acceptance_length", spec)
                    else:
                        self.assertNotIn("--speculative-config", argv)


class TransportProbeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script = SCRIPT.with_name("pd_transport_probe.py")
        spec = importlib.util.spec_from_file_location("transport_probe", cls.script)
        cls.probe = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.probe)

    def test_import_does_not_load_gpu_libraries(self):
        subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import runpy, sys; "
                    f"runpy.run_path({str(self.script)!r}); "
                    "assert 'torch' not in sys.modules; "
                    "assert 'nixl_rocm' not in sys.modules; "
                    "assert 'mori' not in sys.modules"
                ),
            ],
            check=True,
        )

    def test_registration_sizes_are_bounded_and_not_repeated(self):
        self.assertEqual(self.probe.parse_sizes("4096,2493186048"), [4096, 2493186048])
        for bad in ("0", "-1", "2493186049", "1,2,3,4", "4096,4096"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.probe.parse_sizes(bad)

    def test_selection_requires_observed_active_nonzero_roce_gid(self):
        inv = {
            "ports": [
                {
                    "device": "rdma3",
                    "port": "1",
                    "state": "4: ACTIVE",
                    "link_layer": "Ethernet",
                    "gids": [
                        {
                            "index": "1",
                            "gid": "::ffff:10.0.0.1",
                            "type": "RoCE v2",
                            "ndev": "eth3",
                        }
                    ],
                }
            ]
        }
        self.assertEqual(
            self.probe.validated_selection("rdma3:1@1", inv),
            {"UCX_NET_DEVICES": "rdma3:1", "UCX_IB_GID_INDEX": "1"},
        )
        for bad in ("rdma4:1@1", "rdma3:1@0", "rdma3:1@1;evil"):
            with self.subTest(bad=bad), self.assertRaises(self.probe.Unknown):
                self.probe.validated_selection(bad, inv)
        for field, value in (
            ("gid", "::"),
            ("gid", {"error": "unreadable"}),
            ("ndev", ""),
            ("type", {"error": "missing"}),
        ):
            with (
                self.subTest(field=field, value=value),
                patch.dict(inv["ports"][0]["gids"][0], {field: value}),
                self.assertRaises(self.probe.Unknown),
            ):
                self.probe.validated_selection("rdma3:1@1", inv)
        inv["ports"][0]["state"] = "1: DOWN"
        with self.assertRaises(self.probe.Unknown):
            self.probe.validated_selection("rdma3:1@1", inv)

    def test_timeout_signals_child_process_group_and_reaps_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "child.log"
            rc, timed_out = self.probe.supervise(
                [sys.executable, "-c", "import time; time.sleep(30)"],
                log,
                0.1,
                os.environ.copy(),
            )
            self.assertIsNone(rc)
            self.assertTrue(timed_out)

    def test_child_dependency_unknown_and_backend_failure_preserve_stage(self):
        for error, expected in (
            (ImportError("missing"), "UNKNOWN"),
            (RuntimeError("backend"), "FAIL"),
        ):
            with self.subTest(error=error), tempfile.TemporaryDirectory() as tmp:
                args = argparse.Namespace(
                    child="nixl-create", result=str(Path(tmp) / "result")
                )
                self.probe.checkpoint(args.result, "backend_create")
                with patch.object(self.probe, "nixl_probe", side_effect=error):
                    self.probe.child(args)
                result = json.loads(Path(args.result).read_text())
                self.assertEqual(result["status"], expected)
                self.assertEqual(result["stage"], "backend_create")

    def test_missing_peer_does_not_skip_local_registration_and_fail_is_not_success(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            env = {
                "NODE_RANK": "0",
                "IPADDRS": "10.0.0.1,10.0.0.2",
                "RUN_DIR": tmp,
                "SLURM_JOB_ID": "123",
                "ATOMESH_VLLM_SOURCE_SHA": self.probe.FIXED_SHA,
                "DOCKER_IMAGE": "image@" + self.probe.IMAGE_DIGEST,
                "ATOMESH_TRANSPORT_GPU": "1",
                "ATOMESH_TRANSPORT_MORI_BYTES": "4096",
            }
            calls = []

            def fake(argv, log, timeout, env):
                calls.append(argv[argv.index("--child") + 1])
                self.probe.write_json(
                    Path(argv[argv.index("--result") + 1]),
                    {"status": "FAIL", "stage": "backend_create"},
                )
                return 0, False

            with (
                patch.dict(os.environ, env, clear=True),
                patch.object(sys, "argv", [str(self.script)]),
                patch.object(self.probe, "inventory", return_value={"ports": []}),
                patch.object(self.probe, "supervise", side_effect=fake),
                patch.object(
                    self.probe,
                    "wait_json",
                    side_effect=self.probe.Unknown("peer missing"),
                ),
            ):
                self.assertEqual(self.probe.main(), 0)
            report = json.loads(
                (
                    Path(tmp) / "transport-diagnostic/benchmark/rank-0/summary.json"
                ).read_text()
            )
            self.assertEqual(calls, ["nixl-create", "mori-register"])
            self.assertEqual(report["collection"], "COMPLETED")
            self.assertEqual(report["model_pd"], "NOT_TESTED")
            self.assertEqual(report["stages"]["nixl-original-create"]["status"], "FAIL")
            self.assertEqual(
                report["stages"]["nixl-rdma-gpu-read"]["status"], "UNKNOWN"
            )


if __name__ == "__main__":
    unittest.main()
