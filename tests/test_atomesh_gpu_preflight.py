"""Reject image extraction failures before starting either model server."""

import importlib.util
import os
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

SCRIPT = Path(__file__).parents[1] / ".github/scripts/atomesh/pd_gpu_cleanup.py"
SPEC = importlib.util.spec_from_file_location("pd_gpu_cleanup", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
SELECT_SCRIPT = SCRIPT.with_name("pd_select_storage_nodes.py")
SELECT_SPEC = importlib.util.spec_from_file_location(
    "pd_select_storage_nodes", SELECT_SCRIPT
)
SELECT = importlib.util.module_from_spec(SELECT_SPEC)
SELECT_SPEC.loader.exec_module(SELECT)
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


def filesystem_metrics(free_bytes, mountpoint="/", inodes=200000):
    labels = f'device="/dev/nvme0n1p1",fstype="ext4",mountpoint="{mountpoint}"'
    values = {
        "avail_bytes": free_bytes,
        "files_free": inodes,
        "files": 1000000,
        "readonly": 0,
        "device_error": 0,
    }
    return "\n".join(
        f"node_filesystem_{key}{{{labels}}} {value}" for key, value in values.items()
    )


def host_metrics(free_bytes):
    return (
        f"node_time_seconds {time.time()}\n"
        'node_scrape_collector_success{collector="filesystem"} 1\n'
        + filesystem_metrics(free_bytes)
    )


class NodeSelectionTest(unittest.TestCase):
    def test_spur_command_and_multiline_node_records_reach_pair_selection(self):
        # Replay the layout from the 2026-10-12 Spur node-details artifact.
        # Spur accepts `show nodes <nodelist>`, without Slurm's --oneliner.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            command = root / "scontrol"
            command.write_text(
                "#!/bin/sh\n"
                'if [ "$*" != "show nodes pit2-p03-g23,pit2-p03-g42" ]; then\n'
                '  echo "Unsupported Spur show arguments" >&2; exit 1\n'
                "fi\n"
                "cat <<'NODES'\n"
                "NodeName=pit2-p03-g23\n"
                "   State=MIXED Reason=\n"
                "   Partitions=amd-spur\n"
                "   CPUTot=128 CPUAlloc=6 RealMemory=3095781 FreeMem=3034177\n\n"
                "NodeName=pit2-p03-g42\n"
                "   State=ALLOCATED Reason=\n"
                "   Partitions=amd-spur\n"
                "   CPUTot=128 CPUAlloc=128 RealMemory=3095781 FreeMem=2976596\n"
                "NODES\n"
            )
            command.chmod(0o755)
            output = root / "selection.json"
            github_env = root / "github-env"
            with (
                patch(
                    "sys.argv",
                    [
                        "select",
                        "--candidates",
                        "pit2-p03-g23,pit2-p03-g42",
                        "--output",
                        str(output),
                    ],
                ),
                patch.dict(
                    os.environ,
                    PATH=f"{root}:{os.environ['PATH']}",
                    GITHUB_ENV=str(github_env),
                ),
                patch.object(SELECT.urllib.request, "build_opener") as opener,
            ):
                response = opener.return_value.open.return_value.__enter__.return_value
                response.read.return_value = host_metrics(500 * 1024**3).encode()
                SELECT.main()
            self.assertEqual(
                github_env.read_text(),
                "ATOMESH_1P1D_NODES=pit2-p03-g23,pit2-p03-g42\n"
                "ATOMESH_NODE_POOL=pit2-p03-g23,pit2-p03-g42\n",
            )

    def test_rejects_archived_g24_capacity_before_gpu_submission(self):
        report = SELECT.storage_report(host_metrics(12153413632), time.time())
        self.assertFalse(report["passed"])
        self.assertEqual(report["filesystems"][1]["free_bytes"], 12153413632)

    def test_full_containerd_mount_is_not_hidden_by_large_root(self):
        metrics = host_metrics(500 * 1024**3)
        metrics += "\n" + filesystem_metrics(12153413632, "/var/lib/containerd")
        report = SELECT.storage_report(metrics, time.time())
        self.assertFalse(report["passed"])
        self.assertTrue(report["filesystems"][0]["passed"])
        self.assertFalse(report["filesystems"][1]["passed"])

    def test_incomplete_stale_or_failed_collection_cannot_qualify(self):
        good = host_metrics(500 * 1024**3)
        for metrics in (
            "",
            good.replace("node_time_seconds", "other_time"),
            good.replace('collector="filesystem"} 1', 'collector="filesystem"} 0'),
            good.replace("node_filesystem_files_free", "missing_files_free"),
        ):
            with self.subTest(metrics=metrics), self.assertRaises(ValueError):
                SELECT.storage_report(metrics, time.time())
        with self.assertRaisesRegex(ValueError, "stale"):
            SELECT.storage_report(good, time.time() + 3600)
        self.assertFalse(
            SELECT.storage_report(
                good + "\n" + filesystem_metrics(500 * 1024**3, "/tmp", inodes=0),
                time.time(),
            )["passed"]
        )

    def test_selects_exactly_two_healthy_nodes_and_prefers_idle(self):
        nodes = ["g24", "g42", "g12", "g07", "g25"]
        records = {node: {"State": "IDLE"} for node in nodes}
        records["g42"]["State"] = "ALLOCATED"
        records["g25"]["State"] = "IDLE+DRAIN"
        calls = []

        def fetch(node):
            calls.append(node)
            return host_metrics(12153413632 if node == "g24" else 500 * 1024**3)

        result = SELECT.select_pair(nodes, records, fetch)
        self.assertEqual(result["selected"], ["g12", "g07"])
        self.assertNotIn("g25", calls)

    def test_unreachable_metrics_or_one_healthy_node_does_not_select_pair(self):
        records = {node: {"State": "IDLE"} for node in ("g24", "g42")}

        def fetch(node):
            if node == "g24":
                raise TimeoutError("metrics unavailable")
            return host_metrics(500 * 1024**3)

        result = SELECT.select_pair(list(records), records, fetch)
        self.assertEqual(result["selected"], [])
        self.assertIn("TimeoutError", result["nodes"][0]["reason"])


if __name__ == "__main__":
    unittest.main()
