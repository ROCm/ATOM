"""CPU contract tests for the public-profile entry, without AITER or torch.

Can also run directly with Python's unittest when the GPU environment is
remote. Actual EngineArgs/Config validation is the entry's `preflight` action.
"""

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENTRY = ROOT / "tools/dsv41/run_dashboard.py"
spec = importlib.util.spec_from_file_location("dsv41_dashboard_entry", ENTRY)
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


class DashboardProfileTests(unittest.TestCase):
    def test_matches_native_catalog_dspark_and_c256_request_protocol(self):
        profile = entry.load_profile()
        catalog = json.loads((ROOT / ".github/benchmark/models.json").read_text())
        model = next(m for m in catalog["models"] if m["path"] == profile["model"])
        variant = next(v for v in model["variants"] if v["label"] == "DSpark")
        self.assertEqual(
            profile["engine"]["tensor_parallel_size"], model["config"]["tp"]
        )
        self.assertEqual(
            profile["engine"]["kv_cache_dtype"], model["config"]["kv_cache_dtype"]
        )
        self.assertIn("--num-speculative-tokens 5", variant["extra_args"])
        self.assertIn("--use-chat-template", variant["bench_args"])
        benchmark = profile["benchmark"]
        self.assertEqual(benchmark["num_prompts"], 10 * benchmark["max_concurrency"])
        self.assertEqual(benchmark["num_warmups"], 2 * benchmark["max_concurrency"])
        self.assertEqual(benchmark["max_concurrency"], variant["conc_max"])
        self.assertIsNone(profile["acceptance"]["fixed_query_rows"])

    def test_dirty_owner_environment_cannot_enable_old_bootstrap(self):
        inherited = {
            "DSV41_CONSOLIDATED_ENTRY": "1",
            "DSV41_OWNER_SERVING": "1",
            "DSV41_STEP12_OWNER_EP4": "1",
            "ATOM_DSV41_PREFILL_TAIL_MODE": "settled",
            "ATOM_MONO_ENABLE": "0",
            "ATOM_DSV41_BENCHMARK_SYNTHETIC": "1",
            "ATOM_ENABLE_RELAXED_MTP": "1",
            "AITER_DISABLE_COMM_FUSED_MOE": "1",
            "AITER_CONFIG_FMOE_FILE": "/old/tuned.csv",
            "PYTHONPATH": "/old/acceptance:/old/model",
            "LD_PRELOAD": "/old/serialize_comgr.so",
            "ROCR_VISIBLE_DEVICES": "0,1,2,3",
            "PATH": "/opt/venv/bin:/usr/bin",
            "HOME": "/unchanged-home",
        }
        before = inherited.copy()
        env, removed = entry.clean_environment(
            inherited,
            Path("/new/atom"),
            Path("/new/aiter"),
            Path("/new/cache"),
            entry.load_profile(),
        )
        self.assertEqual(inherited, before)
        self.assertFalse(any(k.startswith(("DSV41_", "ATOM_")) for k in env))
        self.assertNotIn("AITER_DISABLE_COMM_FUSED_MOE", env)
        self.assertNotIn("AITER_CONFIG_FMOE_FILE", env)
        self.assertNotIn("LD_PRELOAD", env)
        self.assertIn("DSV41_CONSOLIDATED_ENTRY", removed)
        self.assertEqual(env["PYTHONPATH"], "/new/atom:/new/aiter")
        self.assertEqual(env["ROCR_VISIBLE_DEVICES"], "0,1,2,3")
        self.assertEqual(env["HOME"], "/unchanged-home")
        self.assertEqual(env["AITER_LOG_LEVEL"], "WARNING")
        self.assertEqual(env["TORCHINDUCTOR_CACHE_DIR"], "/new/cache/inductor")

    def test_commands_keep_speculation_and_use_native_module_entries(self):
        args = entry.parse_args(
            [
                "plan",
                "--aiter-source",
                str(ROOT.parent / "aiter"),
                "--model",
                "/models/name with spaces",
                "--output",
                "/tmp/not-created",
                "--host",
                "0.0.0.0",
            ]
        )
        server, client = entry.commands(args, entry.load_profile())
        self.assertEqual(
            server[:3], [sys.executable, "-m", "atom.entrypoints.openai_server"]
        )
        self.assertEqual(
            client[:3], [sys.executable, "-m", "atom.benchmarks.benchmark_serving"]
        )
        self.assertIn("/models/name with spaces", server)
        self.assertEqual(server[server.index("--method") + 1], "dspark")
        self.assertEqual(server[server.index("--max-num-seqs") + 1], "512")
        self.assertNotIn("--enable-expert-parallel", server)
        self.assertNotIn("--enable-dp-attention", server)
        self.assertNotIn("--spec-decode-acceptance-length", server)
        self.assertIn("http://127.0.0.1:19126", client)
        self.assertIn("--use-chat-template", client)
        self.assertEqual(client[client.index("--seed") + 1], "0")
        self.assertEqual(client[client.index("--num-warmups") + 1], "512")

    def test_plan_runs_without_gpu_imports_and_never_overwrites_evidence(self):
        with tempfile.TemporaryDirectory(prefix="dsv41-profile-test-") as temporary:
            output = Path(temporary)
            command = [
                sys.executable,
                "-S",
                str(ENTRY),
                "plan",
                "--aiter-source",
                str(ROOT.parent / "aiter"),
                "--output",
                str(output),
            ]
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
            result = subprocess.run(
                command, env=env, capture_output=True, text=True, check=False
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            plan = json.loads((output / "plan.json").read_text())
            self.assertEqual(plan["status"], "prepared_not_executed")
            self.assertFalse((output / "cache").exists())
            original = (output / "plan.json").read_bytes()
            repeated = subprocess.run(
                command, env=env, capture_output=True, text=True, check=False
            )
            self.assertNotEqual(repeated.returncode, 0)
            self.assertEqual((output / "plan.json").read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
