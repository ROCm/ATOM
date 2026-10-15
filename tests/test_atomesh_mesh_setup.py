# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Check Mesh setup with an unavailable build container, without Docker/GPU."""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SETUP_SCRIPT = REPO_ROOT / ".github/scripts/atomesh/setup_mesh.sh"


class MeshSetupTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.bin_dir = self.root / "bin"
        self.bin_dir.mkdir()
        self.marker = self.root / "docker-called"
        docker = self.bin_dir / "docker"
        docker.write_text(
            '#!/usr/bin/env bash\n'
            'touch "${TEST_DOCKER_MARKER}"\n'
            'exit 77\n'
        )
        docker.chmod(0o755)
        self.run_dir = self.root / "run"
        self.env = {
            "PATH": f"{self.bin_dir}:{os.environ['PATH']}",
            "TEST_DOCKER_MARKER": str(self.marker),
            "ATOMESH_BUILD_MESH": "true",
            "ATOMESH_MESH_TARGET_DIR": str(self.root / "cache"),
            # Placement alone must not determine whether Mesh is needed.
            "PD_WORKER_LAYOUT": "single_node",
        }

    def run_setup(self, topology):
        return subprocess.run(
            [
                "bash", str(SETUP_SCRIPT), str(REPO_ROOT), str(self.run_dir),
                "test-image", str(self.root / "docker.env"), "42",
            ],
            env={**self.env, "TOPOLOGY": topology},
            capture_output=True,
            text=True,
            timeout=10,
        )

    def test_agg_starts_without_accessing_the_build_container(self):
        for topology in ("agg", "AGG-TP8-DCP8"):
            with self.subTest(topology=topology):
                result = self.run_setup(topology)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("Skipping Mesh build", result.stderr)
                self.assertFalse(self.marker.exists())
                self.assertFalse(self.run_dir.exists())
                self.assertFalse((self.root / "cache").exists())

    def test_single_node_pd_still_builds_and_propagates_failure(self):
        result = self.run_setup("1p1d")
        self.assertEqual(result.returncode, 77, result.stderr)
        self.assertTrue(self.marker.exists())
        self.assertIn("Building Mesh before launching services", result.stderr)


if __name__ == "__main__":
    unittest.main()
