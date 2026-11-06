"""Optional owner execution must never leak into clients or compiler helpers."""

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


entry = load("owner_dashboard_entry", "tools/dsv41/run_dashboard.py")
scope = load("owner_scope", "tools/dsv41/owner_bootstrap/owner_bootstrap_scope.py")


class OwnerEntryTests(unittest.TestCase):
    def test_roles_exclude_compiler_client_and_resource_tracker(self):
        for argv in (
            ["python", "-m", "atom.entrypoints.openai_server"],
            [
                "python",
                "-c",
                "from multiprocessing.spawn import spawn_main; spawn_main()",
            ],
        ):
            self.assertTrue(scope.service_process(argv))
        for argv in (
            ["python", "compile.py"],
            ["python", "-m", "atom.benchmarks.benchmark_serving"],
            [
                "python",
                "-c",
                "from multiprocessing.resource_tracker import main; main(5)",
            ],
            ["python", "-c", "import atom"],
            ["python", "-m"],
        ):
            self.assertFalse(scope.service_process(argv))

    def test_opt_in_serving_uses_full_and_keeps_probe_clean(self):
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(entry, "manifest", return_value={}),
            patch.object(entry.subprocess, "run") as probe,
            patch.object(entry.os, "chdir"),
            patch.object(entry.os, "execvpe") as serve,
            patch.dict(os.environ, {"DSV41_STEP12_OWNER_EP4": "1"}),
        ):
            entry.main(
                [
                    "serve",
                    "--execution",
                    "fp8-owner",
                    "--output",
                    temporary,
                    "--aiter-source",
                    str(ROOT.parent / "aiter"),
                ]
            )
            env = serve.call_args.args[2]
            self.assertEqual(env["DSV41_FP8_OWNER_ENTRY"], "1")
            self.assertEqual(env["DSV41_OWNER_SERVING"], "1")
            self.assertNotIn("DSV41_STEP12_OWNER_EP4", env)
            self.assertEqual(
                Path(env["DSV41_OWNER_SERVING_CONTROL"]).read_text(), "full\n"
            )
            self.assertTrue(
                env["PYTHONPATH"].startswith(str(ROOT / "tools/dsv41/owner_bootstrap"))
            )
            self.assertFalse(
                any(k.startswith("DSV41_") for k in probe.call_args.kwargs["env"])
            )
            self.assertNotIn(
                "owner_bootstrap", probe.call_args.kwargs["env"]["PYTHONPATH"]
            )

    def test_client_never_installs_owner_even_when_flag_is_supplied(self):
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(entry, "manifest", return_value={}),
            patch.object(entry.os, "chdir"),
            patch.object(entry.os, "execvpe") as client,
        ):
            entry.main(
                [
                    "benchmark",
                    "--execution",
                    "fp8-owner",
                    "--output",
                    temporary,
                    "--aiter-source",
                    str(ROOT.parent / "aiter"),
                ]
            )
            env = client.call_args.args[2]
            self.assertFalse(any(k.startswith("DSV41_") for k in env))
            self.assertNotIn("owner_bootstrap", env["PYTHONPATH"])
            self.assertFalse((Path(temporary) / "owner").exists())


if __name__ == "__main__":
    unittest.main()
