"""A GPU-visible launch must isolate only its configuration probe."""

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "dashboard_gpu_preflight", ROOT / "tools/dsv41/run_dashboard.py"
)
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


class GPUPreflightTests(unittest.TestCase):
    def test_serve_preserves_devices_after_cpu_probe(self):
        selected = {
            "ROCR_VISIBLE_DEVICES": "0,1,2,3",
            "HIP_VISIBLE_DEVICES": "0,1,2,3",
            "CUDA_VISIBLE_DEVICES": "0,1,2,3",
        }
        with (
            tempfile.TemporaryDirectory() as output,
            patch.dict(os.environ, selected),
            patch.object(entry, "manifest", return_value={}),
            patch.object(entry.subprocess, "run") as probe,
            patch.object(entry.os, "chdir"),
            patch.object(entry.os, "execvpe") as serve,
        ):
            entry.main(
                [
                    "serve",
                    "--aiter-source",
                    str(ROOT.parent / "aiter"),
                    "--output",
                    output,
                ]
            )
            self.assertEqual(probe.call_count, 1)
            self.assertEqual(serve.call_count, 1)
            for key, value in selected.items():
                self.assertEqual(probe.call_args.kwargs["env"][key], "")
                self.assertEqual(serve.call_args.args[2][key], value)
                self.assertEqual(os.environ[key], value)


if __name__ == "__main__":
    unittest.main()
