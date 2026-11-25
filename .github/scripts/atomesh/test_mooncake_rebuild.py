"""Keep the two producer builds comparable and reject stale wheel delivery."""

import copy
import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from pd_mooncake_rebuild import stage_context
from pd_timeout_probe import validate_probe


class RebuildTests(unittest.TestCase):
    folder = Path(__file__).resolve().parents[2] / "benchmark/pr5090-router-ab"

    def test_both_arms_keep_sender10_and_identical_load(self):
        arms = [
            json.loads(
                (self.folder / f"single-node-pp4-dep4-te5840-{arm}.json").read_text()
            )
            for arm in ("base", "patched")
        ]
        for contract in arms:
            validate_probe(contract)
            self.assertEqual(contract["concurrency"], 80)
            self.assertEqual(contract["duration"], 900)
            self.assertTrue(contract["verify_pd"])
            self.assertNotIn("container", contract["roles"]["decode"])
        for key in (
            "image",
            "roles",
            "environment",
            "prefill_device_order",
            "router_build",
            "srtctl_sha",
        ):
            self.assertEqual(arms[0][key], arms[1][key])

    def test_staging_checks_bytes_and_never_installs_native_probe(self):
        contract = json.loads(
            (self.folder / "single-node-pp4-dep4-te5840-base.json").read_text()
        )
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            (repo / ".sa-srt-runtime").mkdir()
            for relative in (
                contract["timeout_probe"]["context"],
                contract["mooncake_rebuild"]["context"],
            ):
                source = self.folder.parents[2] / relative
                shutil.copytree(source, repo / relative)
            spec = contract["mooncake_rebuild"]
            manifest_file = repo / spec["context"] / "manifest.json"
            manifest = json.loads(manifest_file.read_text())
            spec["artifact_root"] = str(repo / "artifacts")
            wheel = Path(spec["artifact_root"]) / "base" / manifest["wheel_name"]
            wheel.parent.mkdir(parents=True)
            wheel.write_bytes(b"stale wheel")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                stage_context(repo, copy.deepcopy(contract))
            manifest["arms"]["base"]["wheel_sha256"] = hashlib.sha256(
                wheel.read_bytes()
            ).hexdigest()
            manifest_file.write_text(json.dumps(manifest))
            staged = stage_context(repo, copy.deepcopy(contract))
            context = repo / staged["timeout_probe"]["context"]
            dockerfile = (context / "Dockerfile").read_text()
            self.assertIn("--no-index --no-deps --force-reinstall", dockerfile)
            self.assertNotIn("LD_PRELOAD", dockerfile)
            self.assertFalse((context / "mooncake_timeout_probe.cpp").exists())
            self.assertEqual(staged["roles"]["decode"], contract["roles"]["decode"])
            built = json.loads((context / "manifest.json").read_text())
            for name, expected in built["files"].items():
                self.assertEqual(
                    hashlib.sha256((context / name).read_bytes()).hexdigest(), expected
                )


if __name__ == "__main__":
    unittest.main()
