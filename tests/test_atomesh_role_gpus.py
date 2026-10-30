# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Check the GPUs the launcher exposes to each role by default."""

import os
import subprocess
import unittest
from pathlib import Path

SERVER_SCRIPT = (
    Path(__file__).resolve().parents[1] / ".github/scripts/atomesh/pd_server_atom.sh"
)


def _section(source, start, end):
    begin = source.index(start)
    return source[begin : source.index(end, begin) + len(end)]


class RoleGpusTest(unittest.TestCase):
    def setUp(self):
        source = SERVER_SCRIPT.read_text()
        self.snippet = "\n".join(
            [
                _section(
                    source,
                    "# PP size is not a top-level env var",
                    'DECODE_PP_SIZE="$(_pp_from_args "${DECODE_SERVER_ARGS}")"',
                ),
                _section(
                    source,
                    "# A role's server spans TP * PP GPUs",
                    'export HIP_VISIBLE_DEVICES="$(seq -s, 0 "$((role_gpus - 1))")"\nfi',
                ),
                'echo "${HIP_VISIBLE_DEVICES}"',
            ]
        )

    def visible_devices(self, node_rank, **overrides):
        env = {
            "PATH": os.environ["PATH"],
            "NODE_RANK": str(node_rank),
            "xP": "1",
            "AGGREGATED": "0",
            "PREFILL_SINGLE_NODE_PD": "0",
            "PREFILL_TP_SIZE": "1",
            "DECODE_TP_SIZE": "8",
            "PREFILL_SERVER_ARGS": "",
            "DECODE_SERVER_ARGS": "",
            **overrides,
        }
        result = subprocess.run(
            ["bash", "-c", self.snippet],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()

    def test_pp_prefill_gets_a_gpu_per_stage(self):
        args = "--pipeline-parallel-size 8 --enforce-eager"
        self.assertEqual(
            self.visible_devices(0, PREFILL_SERVER_ARGS=args), "0,1,2,3,4,5,6,7"
        )

    def test_decode_without_pp_gets_its_tp(self):
        self.assertEqual(
            self.visible_devices(1, PREFILL_SERVER_ARGS="--pipeline-parallel-size 8"),
            "0,1,2,3,4,5,6,7",
        )
        self.assertEqual(
            self.visible_devices(1, DECODE_TP_SIZE="4"),
            "0,1,2,3",
        )

    def test_tp_times_pp(self):
        self.assertEqual(
            self.visible_devices(
                0, PREFILL_TP_SIZE="2", PREFILL_SERVER_ARGS="--pipeline-parallel-size=2"
            ),
            "0,1,2,3",
        )

    def test_explicit_devices_are_kept(self):
        self.assertEqual(
            self.visible_devices(
                0,
                PREFILL_SERVER_ARGS="--pipeline-parallel-size 8",
                HIP_VISIBLE_DEVICES="4,5,6,7",
            ),
            "4,5,6,7",
        )


if __name__ == "__main__":
    unittest.main()
