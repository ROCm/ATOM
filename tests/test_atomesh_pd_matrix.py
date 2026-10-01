# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU-only tests for ATOMesh Slurm node selection."""

import importlib.util
import os
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / ".github/scripts/atomesh/pd_matrix.py"
SPEC = importlib.util.spec_from_file_location("atomesh_pd_matrix", SCRIPT)
pd_matrix = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pd_matrix)


class NodeSelectionTest(unittest.TestCase):
    def test_pr58968_inspection_hashes_configs_without_gpu_work(self):
        import hashlib
        import tempfile

        import yaml

        workflow = SCRIPT.parents[2] / "workflows/atomesh-benchmark.yaml"
        config = yaml.safe_load(workflow.read_text())
        step = next(
            step
            for step in config["jobs"]["inspect-run"]["steps"]
            if step.get("name") == "Hash PR58968 model configurations"
        )
        self.assertEqual(step["if"], "${{ inputs.inspect_run_id == '36845669971' }}")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            command = step["run"]
            expected = []
            for index, model in enumerate(
                ("moonshotai/Kimi-K3", "Inferact/Kimi-K3-DSpark")
            ):
                source = root / f"config-{index}.json"
                data = f'{{"model": "{model}"}}\n'.encode()
                source.write_bytes(data)
                expected.append(hashlib.sha256(data).hexdigest())
                command = command.replace(
                    f"/share_nfs/models/{model}/config.json", str(source)
                )
            result = subprocess.run(
                ["bash", "-c", command],
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            hashes = (root / "inspection/model-configs/SHA256SUMS").read_text()
            self.assertEqual(
                [line.split()[0] for line in hashes.splitlines()], expected
            )
            source.unlink()
            result = subprocess.run(
                ["bash", "-c", command],
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)

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


class PR58968HarnessTest(unittest.TestCase):
    """CPU contract: real shell argv, blocked placeholder, and evidence guards."""

    def cells(self):
        root = SCRIPT.parents[3]
        cfg = pd_matrix.load_config(root / ".github/benchmark/models_atomesh.yaml")
        name = "Kimi-K3-MXFP4-vLLM-DSpark"
        model = cfg["models"][name]
        with patch.dict(
            os.environ,
            {
                "ATOMESH_MODEL_ROOT": "/mnt/models",
                "ATOMESH_NODE_POOL": "pit2-p03-g01,pit2-p03-g03",
                "ATOMESH_1P1D_NODES": "pit2-p03-g01,pit2-p03-g03",
                "ATOMESH_SLURM_ACCOUNT": "amd-frameworks",
                "ATOMESH_SLURM_PARTITION": "amd-spur",
                "ATOMESH_SLURM_SUBMIT_RUNNER": "atomesh-cicd",
                "ATOMESH_LOG_ROOT": "/it-share/ATOMESH_LOG/",
                "ATOMESH_PD_RANK_MAPPING_POLICY": "none",
            },
        ):
            return [
                pd_matrix.build_cell(
                    cfg=cfg,
                    model_name=name,
                    model_cfg=model,
                    suite_name="vllm",
                    suite_cfg=case,
                    override_image=None,
                    override_benchmark_concurrency=None,
                    override_eval_concurrency=[64],
                )
                for case in model["suites"]["vllm"]
            ]

    def test_candidate_matrix_keeps_full_accuracy_and_two_graph_modes(self):
        import json
        import shlex

        cells = self.cells()
        self.assertEqual(len(cells), 2)
        for cell in cells:
            self.assertEqual(cell["num_nodes"], 2)
            self.assertEqual(cell["runner"]["gpus_per_node"], 8)
            self.assertNotIn("lmcache", cell["vllm"])
            self.assertNotIn("fork", cell["vllm"])
            self.assertRegex(
                cell["vllm"]["source"]["sha"],
                r"^(?:CANDIDATE_NOT_READY_REPLACE_WITH_COORDINATOR_SHA|[0-9a-f]{40})$",
            )
            for role in ("prefill", "decode"):
                self.assertEqual(cell["service"][role]["tp"], 8)
                self.assertEqual(cell["service"][role]["dcp"], 8)
            accuracy = cell["accuracy"]
            self.assertEqual(accuracy["concurrency"], [64])
            self.assertEqual(accuracy["fewshot"], 5)
            self.assertEqual(accuracy["max_gen_toks"], 4096)
            self.assertEqual(accuracy["threshold"], 0.94)
            self.assertIsNone(accuracy["limit"])
            args = shlex.split(cell["server_args"]["extra_args"])
            spec = json.loads(args[args.index("--speculative-config") + 1])
            self.assertEqual(spec["num_speculative_tokens"], 3)
            self.assertEqual(spec["rejection_sample_method"], "standard")
            self.assertNotIn("synthetic_acceptance_length", spec)

    def test_unresolved_candidate_is_blocked_before_submission(self):
        import json
        import subprocess

        cell = self.cells()[0]
        cell["vllm"]["source"]["sha"] = "CANDIDATE_NOT_READY_TEST_FIXTURE"
        proc = subprocess.run(
            [
                "bash",
                str(SCRIPT.with_name("pd_submit.sh")),
                "--cell-json",
                json.dumps(cell),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 2)
        self.assertIn("coordinator must replace", proc.stderr)

    def test_native_provenance_rejects_image_python_or_extension(self):
        from types import SimpleNamespace

        spec = importlib.util.spec_from_file_location(
            "native_provenance", SCRIPT.with_name("pd_native_provenance.py")
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for bad_name in ("vllm", "vllm._C"):
            with (
                self.subTest(module=bad_name),
                patch.object(
                    module.importlib,
                    "import_module",
                    side_effect=lambda name, bad_name=bad_name: SimpleNamespace(
                        __file__="/image/old.so" if name == bad_name else __file__
                    ),
                ),
                self.assertRaisesRegex(RuntimeError, "outside native build"),
            ):
                module.native_paths(SCRIPT.parents[3])

    def test_read_smoke_requires_external_consumption_not_just_http_success(self):
        import asyncio
        import json
        import tempfile
        from types import SimpleNamespace

        import httpx

        spec = importlib.util.spec_from_file_location(
            "read_smoke", SCRIPT.with_name("pd_vllm_profile.py")
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for consumes in (True, False):
            totals = {
                role: {
                    key: 0
                    for key in (
                        "local_compute",
                        "local_cache_hit",
                        "external_kv_transfer",
                        "request_success",
                    )
                }
                for role in ("prefill", "decode")
            }

            def handle(request, totals=totals, consumes=consumes):
                role = request.url.host
                total = totals[role]
                if request.url.path == "/metrics":
                    text = (
                        "\n".join(
                            f'vllm:prompt_tokens_by_source_total{{source="{key}"}} {value}'
                            for key, value in total.items()
                            if key != "request_success"
                        )
                        + f'\nvllm:request_success_total{{model_name="K3"}} {total["request_success"]}\n'
                    )
                    return httpx.Response(200, text=text)
                body = json.loads(request.content or b"{}")
                if request.url.path == "/tokenize":
                    return httpx.Response(200, json={"tokens": list(range(5000))})
                if request.url.path == "/reset_prefix_cache":
                    return httpx.Response(200, json={"success": True})
                transfer = body.get("kv_transfer_params", {})
                remote = transfer.get("do_remote_prefill", False) and consumes
                size = len(body["prompt"])
                total["request_success"] += 1
                total["external_kv_transfer"] += size - 1 if remote else 0
                total["local_compute"] += 1 if remote else size
                return httpx.Response(
                    200,
                    json={
                        "choices": [{"text": "same output", "finish_reason": "length"}],
                        "kv_transfer_params": {
                            "remote_block_ids": [[0]],
                            "remote_host": "prefill",
                        },
                    },
                )

            client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
            with (
                tempfile.TemporaryDirectory() as output,
                patch.object(module.httpx, "AsyncClient", return_value=client),
            ):
                args = SimpleNamespace(
                    output=Path(output),
                    prefill="http://prefill",
                    decode="http://decode",
                    model="K3",
                    tp=8,
                    dcp=8,
                    hybrid=True,
                    phase="eval",
                )
                if consumes:
                    asyncio.run(module.run(args))
                    evidence = json.loads((Path(output) / "complete.json").read_text())
                    self.assertEqual(evidence["requests"], 15)
                    self.assertFalse(evidence["full_replay_sync_read_proven"])
                else:
                    with self.assertRaisesRegex(AssertionError, "No external KV"):
                        asyncio.run(module.run(args))
                    self.assertFalse((Path(output) / "complete.json").exists())

    def test_runtime_probe_requires_same_step_rank_and_nonempty_wait(self):
        import json
        import tempfile

        spec = importlib.util.spec_from_file_location(
            "probe", SCRIPT.with_name("pd_read_replay_probe.py")
        )
        probe = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(probe)
        base = {"role": "decode", "rank": 0, "pid": 1, "thread": 2, "step": 3}
        events = [
            dict(base, event="step", sync_load=True),
            dict(base, event="wait_done", count=1, mode="FULL"),
            dict(base, event="replay_done", preexisting_graph=True, mode="FULL"),
        ]
        self.assertEqual(probe.qualifying_ranks(events), {("decode", 0)})
        for change in ({"count": 0}, {"rank": 1}, {"step": 4}, {"mode": "PIECEWISE"}):
            bad = [events[0], {**events[1], **change}, events[2]]
            self.assertEqual(probe.qualifying_ranks(bad), set())
        self.assertEqual(
            probe.qualifying_ranks(
                [*events[:2], {**events[2], "preexisting_graph": False}]
            ),
            set(),
        )
        self.assertEqual(
            probe.qualifying_ranks([*events, dict(base, event="error")]), set()
        )
        self.assertEqual(
            probe.qualifying_ranks([events[0], events[2], events[1]]), set()
        )
        with tempfile.TemporaryDirectory() as tmp:
            rec = probe.Recorder(Path(tmp), "decode", 0, limit=2)
            for _ in range(4):
                rec.begin(["req"], True)
                rec.emit("wait_done", count=1)
            rows = [
                json.loads(line)
                for path in Path(tmp).glob("*.jsonl")
                for line in path.read_text().splitlines()
            ]
            self.assertEqual(len(rows), 4)
            self.assertEqual({r["step"] for r in rows}, {1, 2})

    def test_active_dependency_closure_and_local_source_identity(self):
        import json
        from types import SimpleNamespace

        spec = importlib.util.spec_from_file_location(
            "native_provenance", SCRIPT.with_name("pd_native_provenance.py")
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        packages = {
            "vllm": ("1", ["dep[feature]>=2"]),
            "dep": ("2", ['leaf==3; extra == "feature"', 'absent; extra == "unused"']),
            "leaf": ("3", ["vllm"]),
        }

        def distribution(name):
            if name not in packages:
                raise module.metadata.PackageNotFoundError(name)
            version, requires = packages[name]
            return SimpleNamespace(version=version, requires=requires)

        with patch.object(module.metadata, "distribution", side_effect=distribution):
            self.assertEqual(module.active_dependency_errors(), [])
            roots = [module.Requirement("dep[feature]>=2")]
            self.assertEqual(module.active_dependency_errors(roots=roots), [])
            packages["leaf"] = ("4", [])
            self.assertIn("active=4", module.active_dependency_errors(roots=roots)[0])
            self.assertIn("active=4", module.active_dependency_errors()[0])
            del packages["leaf"]
            self.assertIn("not installed", module.active_dependency_errors()[0])
        for url, valid in (("file:///native/source", True), ("file:///image", False)):
            dist = SimpleNamespace(
                read_text=lambda _, url=url: json.dumps({"url": url})
            )
            if valid:
                module.source_direct_url(dist, Path("/native/source"))
            else:
                with self.assertRaisesRegex(RuntimeError, "does not match"):
                    module.source_direct_url(dist, Path("/native/source"))
        duplicates = [
            SimpleNamespace(metadata={"Name": "dep"}, version=v) for v in ("2", "1")
        ]
        with (
            patch.object(module.metadata, "distributions", return_value=duplicates),
            patch.object(module.metadata, "version", return_value="2"),
        ):
            self.assertEqual(module.active_versions(), {"dep": "2"})

    def test_correctness_evidence_survives_log_collection(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp) / "source", Path(tmp) / "output"
            paths = [
                "slurm_job-1/logs/eval/native-preflight-rank-0.json",
                "slurm_job-1/logs/eval/native-preflight-rank-0.constraints.txt",
                "slurm_job-1/logs/eval/native-build-rank-0.log",
                "slurm_job-1/logs/eval/native-build-env-rank-0.txt",
                "slurm_job-1/logs/eval/native-manifest-rank-0.json",
                "slurm_job-1/pd-smoke/eval/failure.json",
                "slurm_job-1/read-replay/decode-0.jsonl",
                "slurm_job-1/read-replay/validation.json",
            ]
            for name in paths:
                path = source / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(name)
            subprocess.run(
                [
                    "bash",
                    str(SCRIPT.with_name("pd_collect_logs.sh")),
                    str(source),
                    str(output),
                ],
                check=True,
            )
            for name in paths:
                self.assertEqual((output / name).read_text(), name)
                self.assertEqual(
                    (output / "validation-evidence" / name).read_text(), name
                )

    def test_uv_inherited_dependency_offline_regression(self):
        import shutil
        import sys
        import tempfile
        import zipfile

        if shutil.which("uv") is None:
            self.skipTest("uv is not installed")

        def run(argv):
            return subprocess.run(argv, text=True, capture_output=True, check=False)

        run(["uv", "--version"])
        with tempfile.TemporaryDirectory(prefix="uv-inheritance-") as tmp:
            root = Path(tmp)
            venv = root / "venv"
            assert (
                run(
                    [
                        "uv",
                        "venv",
                        "--python",
                        sys._base_executable,
                        "--system-site-packages",
                        str(venv),
                    ]
                ).returncode
                == 0
            )
            python = str(venv / "bin/python")
            active = run(
                [
                    python,
                    "-c",
                    (
                        'import importlib.metadata as m; d=m.distribution("packaging"); '
                        'print(d.version); print(d.locate_file(""))'
                    ),
                ]
            )
            assert active.returncode == 0
            version, location = active.stdout.splitlines()
            assert not Path(location).is_relative_to(venv)
            wheel = root / "inheritance_repro-0.0.1-py3-none-any.whl"
            with zipfile.ZipFile(wheel, "w") as archive:
                info = "inheritance_repro-0.0.1.dist-info/"
                archive.writestr(
                    info + "METADATA",
                    "Metadata-Version: 2.1\nName: inheritance-repro\nVersion: 0.0.1\n"
                    f"Requires-Dist: packaging=={version}\n",
                )
                archive.writestr(
                    info + "WHEEL",
                    "Wheel-Version: 1.0\nGenerator: repro\nRoot-Is-Purelib: true\n"
                    "Tag: py3-none-any\n",
                )
                archive.writestr(info + "RECORD", "")
            args = [
                "uv",
                "pip",
                "install",
                "--python",
                python,
                "--offline",
                "--no-index",
                "--no-cache",
                str(wheel),
            ]
            failed = run(args)
            assert failed.returncode != 0 and "packaging" in failed.stderr
            passed = run([*args, "--no-deps"])
            assert passed.returncode == 0
            validated = run(
                [
                    python,
                    "-c",
                    (
                        "from importlib import metadata as m; "
                        "from packaging.requirements import Requirement; "
                        'r=Requirement(m.requires("inheritance-repro")[0]); '
                        "assert r.specifier.contains(m.version(r.name)); "
                        'print("POST_INSTALL_ACTIVE_CLOSURE_PASS", r, m.version(r.name))'
                    ),
                ]
            )
            assert validated.returncode == 0
            print(
                "REPRO_PASS: inherited requirement visible to Python, ignored by uv "
                "resolution; no-deps install with active metadata validation passes."
            )

    def test_native_install_keeps_pre_post_gates_with_no_resolution(self):
        text = SCRIPT.with_name("pd_server_vllm.sh").read_text()
        self.assertIn("--no-deps --no-build-isolation", text)
        self.assertLess(
            text.index('"${check}" preflight'), text.index("uv pip install")
        )
        self.assertLess(text.index("uv pip install"), text.index('"${check}" manifest'))

    def test_dependency_check_rejects_old_torch_triton_and_aiter(self):
        spec = importlib.util.spec_from_file_location(
            "native_provenance", SCRIPT.with_name("pd_native_provenance.py")
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        reqs = list(
            map(
                module.Requirement,
                [
                    "torch==2.13.0",
                    "triton>=3.8,<3.9",
                    "amd-aiter>=0.1.23",
                ],
            )
        )
        self.assertEqual(
            len(
                module.dependency_errors(
                    reqs,
                    {
                        "torch": "2.12.0",
                        "triton": "3.7.1",
                        "amd-aiter": "0.1.21.post2",
                    },
                )
            ),
            3,
        )
        self.assertEqual(
            module.dependency_errors(
                reqs,
                {
                    "torch": "2.13.0+gitabc",
                    "triton": "3.8.0",
                    "amd-aiter": "0.1.23",
                },
            ),
            [],
        )


if __name__ == "__main__":
    unittest.main()
