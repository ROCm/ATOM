# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Exercise cache directory setup and cleanup without starting GPU services."""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

SERVER_SCRIPT = (
    Path(__file__).resolve().parents[1] / ".github/scripts/atomesh/pd_server_atom.sh"
)


class LMCacheDiskTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="atomesh logs ")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        source = SERVER_SCRIPT.read_text()
        self.functions = source[
            source.index('lmcache_disk_dir=""') : source.index("cleanup_processes()")
        ]

    def run_shell(self, script, **env):
        result = subprocess.run(
            ["bash", "-c", "set -euo pipefail\n" + self.functions + script],
            env={
                **os.environ,
                "RUNTIME_LOG_DIR": str(self.root / "logs"),
                "NODE_RANK": "0",
                "LMCACHE_LOCAL_DISK": "./lmcache-job",
                **env,
            },
            cwd=self.root,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_relative_cache_is_exported_and_cleanup_preserves_logs(self):
        logs = self.root / "logs"
        logs.mkdir()
        (logs / "prefill.log").write_text("keep")
        other_cache = logs / "rank-1/lmcache-job"
        other_cache.mkdir(parents=True)
        (other_cache / "kv").write_text("keep")
        self.run_shell("""
reset_lmcache_disk
test "$LMCACHE_LOCAL_DISK" = "$RUNTIME_LOG_DIR/rank-0/lmcache-job"
bash -c 'test -d "$LMCACHE_LOCAL_DISK"'
touch "$LMCACHE_LOCAL_DISK/kv"
# Starting another worker reapplies the relative role environment.
export LMCACHE_LOCAL_DISK=./lmcache-job
reset_lmcache_disk
test -f "$LMCACHE_LOCAL_DISK/kv"
purge_lmcache_disk
test ! -e "$LMCACHE_LOCAL_DISK"
""")
        self.assertEqual((logs / "prefill.log").read_text(), "keep")
        self.assertEqual((other_cache / "kv").read_text(), "keep")
        self.assertFalse((self.root / "lmcache-job").exists())

    def test_jobs_phases_and_ranks_have_separate_caches(self):
        for job, phase, rank in (
            ("job-1", "benchmark", "0"),
            ("job-1", "eval", "0"),
            ("job-1", "benchmark", "1"),
            ("job-2", "benchmark", "0"),
        ):
            logs = self.root / job / "logs" / phase
            self.run_shell(
                'reset_lmcache_disk\ntouch "$LMCACHE_LOCAL_DISK/kv"\n',
                RUNTIME_LOG_DIR=str(logs),
                NODE_RANK=rank,
            )
        self.assertEqual(len(list(self.root.rglob("kv"))), 4)

    def test_explicit_absolute_path_is_preserved(self):
        cache = self.root / "absolute-cache"
        self.run_shell(
            'reset_lmcache_disk\ntest "$LMCACHE_LOCAL_DISK" = "$EXPECTED"\n',
            LMCACHE_LOCAL_DISK=str(cache),
            EXPECTED=str(cache),
        )
        self.assertTrue(cache.is_dir())
