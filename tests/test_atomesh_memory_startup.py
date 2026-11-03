# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Exercise nonprivileged cache advice and startup log visibility on the CPU."""

import importlib.util
import io
import os
import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / ".github/scripts/atomesh"
SPEC = importlib.util.spec_from_file_location(
    "drop_page_cache", SCRIPTS / "drop_page_cache.py"
)
CACHE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CACHE)


class MemoryStartupTest(unittest.TestCase):
    def test_advice_covers_other_snapshots_and_reports_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("current", "previous"):
                snapshot = root / name
                snapshot.mkdir()
                (snapshot / "weight").write_bytes(b"weights")
            os.mkfifo(root / "pipe")
            (root / "missing").symlink_to(root / "absent")
            output = io.StringIO()
            with (
                patch.object(CACHE.os, "posix_fadvise") as advise,
                redirect_stdout(output),
            ):
                CACHE.drop_page_cache(directory)
            self.assertEqual(advise.call_count, 2)
            self.assertIn(
                "files_advised=2 file_bytes_advised=14 errors=1",
                output.getvalue(),
            )
            self.assertIn("bytes reclaimed are not measured", output.getvalue())

    def test_failed_advice_is_visible_and_fd_is_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "weight").write_bytes(b"weights")
            output = io.StringIO()
            with (
                patch.object(
                    CACHE.os, "posix_fadvise", side_effect=OSError
                ) as advise,
                redirect_stdout(output),
            ):
                CACHE.drop_page_cache(directory)
            self.assertIn(
                "files_advised=0 file_bytes_advised=0 errors=1",
                output.getvalue(),
            )
            with self.assertRaises(OSError):
                os.fstat(advise.call_args.args[0])

    def test_small_startup_message_is_visible_before_writer_exits(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "container.log"
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(SCRIPTS / "cap_log.py"),
                    str(log),
                    "4096",
                    "unused",
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            try:
                message = b"compacting NUMA node 0\n"
                process.stdin.write(message)
                process.stdin.flush()
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    if log.exists() and log.read_bytes() == message:
                        break
                    time.sleep(0.01)
                self.assertIsNone(process.poll())
                self.assertEqual(log.read_bytes(), message)
                process.stdin.close()
                self.assertEqual(process.wait(timeout=5), 0)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=5)
                if not process.stdin.closed:
                    process.stdin.close()
                process.stderr.close()


if __name__ == "__main__":
    unittest.main()
