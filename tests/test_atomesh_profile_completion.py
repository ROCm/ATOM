# SPDX-License-Identifier: MIT
"""A successful load generator alone must not pass a requested trace job."""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[1] / ".github/scripts/atomesh/pd_server_atom.sh"
)


class ProfileCompletionTest(unittest.TestCase):
    def test_trace_deliverable_controls_benchmark_success(self):
        for duration, trace_complete, expected in [
            (0, False, 0),
            (180, False, 1),
            (180, True, 0),
        ]:
            with (
                self.subTest(duration=duration, trace_complete=trace_complete),
                tempfile.TemporaryDirectory() as temp,
            ):
                self.run_case(Path(temp), duration, trace_complete, expected)

    def run_case(self, tmp_path, duration, trace_complete, expected):
        source = SCRIPT.read_text()
        start = source.index("run_aiperf_agentic_benchmark() {")
        end = source.index("\n}\n", start) + 3
        out_dir = tmp_path / "benchmark_results/aiperf-test-1p1d-c32"
        out_dir.mkdir(parents=True)
        (out_dir / "profile_export_aiperf.json").write_text("{}")
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        collector = bin_dir / "python3"
        collector.write_text("#!/bin/sh\nexit 0\n")
        collector.chmod(0o755)
        shell = source[start:end] + r"""
    ensure_aiperf() { :; }
    is_agentic_session_affinity() { return 1; }
    build_agentic_client_command() { local -n command="$1"; command=(true); }
    write_aiperf_dashboard_json() { :; }
    write_aiperf_chrome_trace() { :; }
    run_torch_profile_window() { [[ "$TRACE_COMPLETE" == 1 ]] && touch "$3"; return 0; }
    prefill_ips=(); decode_ips=()
    run_aiperf_agentic_benchmark
    """
        env = dict(
            os.environ,
            RUN_DIR=str(tmp_path),
            PATH=f"{bin_dir}:{os.environ['PATH']}",
            MODEL_NAME="test",
            MODEL_PATH="test",
            TOPOLOGY="1p1d",
            CONC_LIST="32",
            AIPERF_BENCHMARK_DURATION="3600",
            ATOMESH_TORCH_PROFILE_DURATION=str(duration),
            ATOMESH_TORCH_PROFILE_START="1800",
            TRACE_COMPLETE=str(int(trace_complete)),
        )
        result = subprocess.run(
            ["bash", "-c", shell],
            env=env,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        assert result.returncode == expected, result.stdout + result.stderr
        if duration and not trace_complete:
            assert (
                "requested P/D torch traces were not completed"
                in result.stdout + result.stderr
            )
