"""Reject image extraction failures before starting either model server."""

import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

SCRIPT = Path(__file__).parents[1] / ".github/scripts/atomesh/pd_gpu_cleanup.py"
SPEC = importlib.util.spec_from_file_location("pd_gpu_cleanup", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
IMAGE = "registry.example:5000/example/model:nightly@sha256:" + "a" * 64
CACHED_IMAGE = "registry.example:5000/example/model@sha256:" + "a" * 64


class StoragePreflightTest(unittest.TestCase):
    def check(self, cached, free_gib, free_inodes=200000):
        with tempfile.TemporaryDirectory() as directory:
            cleanup = MODULE.Cleanup(
                SimpleNamespace(
                    job_id="123",
                    run_token="a" * 32,
                    node="node",
                    rank=0,
                    cell_id="prefix",
                    stale_cell_id=[],
                    image=IMAGE,
                    out=Path(directory) / "report.json",
                )
            )
            usage = SimpleNamespace(
                f_bavail=free_gib * 1024**3,
                f_frsize=1,
                f_favail=free_inodes,
                f_files=1000000,
            )
            with (
                patch.object(
                    MODULE,
                    "command",
                    side_effect=[
                        directory,
                        CACHED_IMAGE if cached else "",
                    ],
                ),
                patch.object(MODULE.os, "statvfs", return_value=usage),
            ):
                cleanup.check_storage()
            return cleanup.report

    def test_missing_image_rejected_before_large_extraction(self):
        with self.assertRaisesRegex(RuntimeError, "image_cached=False"):
            self.check(cached=False, free_gib=50)

    def test_cached_image_still_requires_build_space_and_inodes(self):
        for free_gib, inodes in [(1, 200000), (50, 0)]:
            with (
                self.subTest(free_gib=free_gib, inodes=inodes),
                self.assertRaisesRegex(RuntimeError, "Insufficient"),
            ):
                self.check(True, free_gib, inodes)

    def test_available_storage_is_recorded_for_both_image_states(self):
        for cached, free_gib in [(True, 50), (False, 250)]:
            with self.subTest(cached=cached):
                report = self.check(cached, free_gib)
                event = report["events"][-1]
                self.assertEqual(event["kind"], "storage")
                self.assertEqual(event["image_cached"], cached)
                self.assertEqual(event["free_bytes"], free_gib * 1024**3)


if __name__ == "__main__":
    unittest.main()
