# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Exercise shared log forwarding without Docker, GPUs, or a scheduler."""

import importlib.util
import io
import os
import subprocess
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / ".github/scripts/atomesh"
SPEC = importlib.util.spec_from_file_location(
    "pd_stream_log", SCRIPTS / "pd_stream_log.py"
)
STREAM = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(STREAM)


class SharedLogStreamingTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.log = self.root / "container.log"

    def stream(self, offset=0):
        output = io.StringIO()
        with redirect_stderr(output):
            offset = STREAM.stream_log(self.log, offset, "[rank-0] ")
        return offset, output.getvalue()

    def test_incremental_output_without_final_newline(self):
        self.log.write_bytes(b"loading\r50%")
        offset, output = self.stream()
        self.assertIn("loading\r50%", output)
        self.assertEqual(offset, self.log.stat().st_size)
        self.assertEqual(self.stream(offset), (offset, ""))
        with self.log.open("ab") as stream:
            stream.write(b"\r100%\nready\n")
        new_offset, output = self.stream(offset)
        self.assertIn("100%\nready", output)
        self.assertNotIn("loading", output)
        self.assertEqual(new_offset, self.log.stat().st_size)

    def test_large_output_is_bounded_and_full_file_is_preserved(self):
        data = b"x" * (2 * STREAM.MAX_BYTES) + b"latest failure\n"
        self.log.write_bytes(data)
        offset, output = self.stream()
        self.assertEqual(offset, len(data))
        self.assertIn("Skipped", output)
        self.assertIn(str(self.log), output)
        self.assertIn("latest failure", output)
        self.assertLess(len(output.encode()), STREAM.MAX_BYTES + 512)
        self.assertEqual(self.log.read_bytes(), data)
        self.assertEqual(self.stream(offset), (offset, ""))

    def test_truncated_file_is_read_again(self):
        self.log.write_text("previous content\n")
        offset, _ = self.stream()
        self.log.write_text("new\n")
        offset, output = self.stream(offset)
        self.assertEqual(offset, 4)
        self.assertIn("new", output)

    def test_unavailable_file_preserves_offset_and_can_recover(self):
        offset, output = self.stream(10)
        self.assertEqual(offset, 10)
        self.assertIn("will retry", output)
        self.log.write_text("new\n")
        offset, output = self.stream(offset)
        self.assertEqual(offset, 4)
        self.assertIn("new", output)

    def test_submitter_forwards_each_rank_once_on_all_spur_runners(self):
        run_dir = self.root / "slurm_job-42"
        for name, text in (
            ("rank-0/container.log", "benchmark progress\n"),
            ("rank-1/container-prefill.log", "prefill ready\n"),
            ("rank-2/container-decode.log", "decode ready\n"),
            ("logs/router.log", "duplicate server output\n"),
        ):
            path = run_dir / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        # Execute the actual submitter hook setup without submitting a job.
        submit = (SCRIPTS / "pd_submit.sh").read_text()
        setup = submit.split("stream_spur_shared_logs_once() {", 1)[1]
        setup = (
            "stream_spur_shared_logs_once() {"
            + setup.split("install_slurm_cancel_traps", 1)[0]
        )
        for runner, spur in (
            ("atomesh-cicd", "1"),
            ("atomesh-cicd-mi350", "1"),
            ("atomesh-cicd-mi355-crusoe", "1"),
            ("native-slurm", "0"),
        ):
            with self.subTest(runner=runner):
                result = subprocess.run(
                    [
                        "bash",
                        "-c",
                        "set -euo pipefail\n"
                        "declare -A SPUR_SHARED_LOG_OFFSETS=()\n"
                        "SLURM_EXTRA_LOG_STREAMER=\n"
                        + setup
                        + '\nif [[ -n "$SLURM_EXTRA_LOG_STREAMER" ]]; then\n'
                        '  "$SLURM_EXTRA_LOG_STREAMER" 999\n'
                        '  "$SLURM_EXTRA_LOG_STREAMER" 42\n'
                        '  "$SLURM_EXTRA_LOG_STREAMER" 42\n'
                        "fi\n",
                    ],
                    env={
                        **os.environ,
                        "REPO_ROOT": str(ROOT),
                        "LOG_ROOT": str(self.root),
                        "SLURM_SUBMIT_RUNNER": runner,
                        "USES_SPUR_CONTROLLER": spur,
                    },
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                if spur == "1":
                    for message in (
                        "benchmark progress",
                        "prefill ready",
                        "decode ready",
                    ):
                        self.assertEqual(result.stderr.count(message), 1)
                    self.assertIn("[spur:rank-0/container.log]", result.stderr)
                    self.assertNotIn("duplicate server output", result.stderr)
                else:
                    self.assertEqual(result.stderr, "")


if __name__ == "__main__":
    unittest.main()
