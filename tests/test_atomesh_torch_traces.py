# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Exercise the torch-trace completion check without starting GPU services."""

import json
import os
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

SERVER_SCRIPT = (
    Path(__file__).resolve().parents[1] / ".github/scripts/atomesh/pd_server_atom.sh"
)


class WaitForTorchTracesTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="atomesh traces ")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        source = SERVER_SCRIPT.read_text()
        self.functions = source[
            source.index("wait_for_torch_traces() {") : source.index(
                "run_aiperf_agentic_benchmark() {"
            )
        ]
        self.marker = self.root / ".torch_trace_start"
        self.marker.touch()
        past = time.time() - 100
        os.utime(self.marker, (past, past))

    def write_stop_file(self, name, trace_dirs):
        path = self.root / f".stop_profile_{name}.json"
        path.write_text(
            json.dumps(
                {
                    "status": "success",
                    "traces": [
                        {"trace_dir": str(d), "elapsed": 1.0} for d in trace_dirs
                    ],
                }
            )
        )
        return path

    def write_trace(self, rank_dir, raw=False, mtime=None):
        rank_dir.mkdir(parents=True, exist_ok=True)
        gz = rank_dir / "Kimi-K3_ts_20261026_110452_335.pt.trace.json.gz"
        gz.write_bytes(b"gz")
        if mtime is not None:
            os.utime(gz, (mtime, mtime))
        if raw:
            (rank_dir / "Kimi-K3_ts_20261026_110452_335.pt.trace.json").write_text("{}")

    def run_wait(self, *stop_files):
        return subprocess.run(
            [
                "bash",
                "-c",
                "set -uo pipefail\n" + self.functions + 'wait_for_torch_traces "$@"',
                "bash",
                str(self.marker),
                *map(str, stop_files),
            ],
            env={**os.environ, "ATOM_PROFILER_TIMEOUT": "0"},
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )

    def pp_prefill_and_tp_decode(self):
        prefill = self.root / "prefill-rank-0"
        decode = self.root / "decode-rank-1"
        prefill_ranks = [prefill / f"pp{k}_rank_0" for k in range(2)]
        decode_ranks = [decode / f"rank_{r}" for r in range(2)]
        stop_files = [
            self.write_stop_file("prefill", prefill_ranks),
            # Decode reports TP rank 0 only; its other ranks are found on disk.
            self.write_stop_file("decode", decode_ranks[:1]),
        ]
        return prefill_ranks + decode_ranks, stop_files

    def test_pp_stage_dirs_count_as_ranks(self):
        rank_dirs, stop_files = self.pp_prefill_and_tp_decode()
        for rank_dir in rank_dirs:
            self.write_trace(rank_dir)
        result = self.run_wait(*stop_files)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("all 4 rank traces on disk", result.stdout)

    def test_raw_json_keeps_a_stage_pending(self):
        rank_dirs, stop_files = self.pp_prefill_and_tp_decode()
        for rank_dir in rank_dirs:
            self.write_trace(rank_dir, raw=rank_dir.name == "pp1_rank_0")
        result = self.run_wait(*stop_files)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("1/4 rank traces still missing", result.stdout)

    def test_trace_older_than_window_is_pending(self):
        rank_dirs, stop_files = self.pp_prefill_and_tp_decode()
        for rank_dir in rank_dirs:
            stale = rank_dir.name == "rank_1"
            self.write_trace(rank_dir, mtime=time.time() - 200 if stale else None)
        result = self.run_wait(*stop_files)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("1/4 rank traces still missing", result.stdout)


if __name__ == "__main__":
    unittest.main()
