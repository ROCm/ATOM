# SPDX-License-Identifier: MIT
"""Opt-in real-model qualification, collected by pytest with normal skip reporting."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.skipif(
    os.environ.get("ATOM_DSV41_INTEGRATION") != "1",
    reason="set ATOM_DSV41_INTEGRATION=1 and ATOM_DSV41_CHECKPOINT for model qualification",
)
def test_tp_level3_full_model():
    assert os.environ.get("ATOM_DSV41_CHECKPOINT"), "checkpoint path required"
    script = (
        Path(__file__).parent / "mono_indexer_validation/validate_tp4_cross_level3.py"
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=4",
            str(script),
        ],
        check=True,
        env=dict(
            os.environ,
            ATOM_MONO_ENABLE="1",
            ATOM_DSV41_MONO_INDEXER="1",
            ATOM_DSV41_MONO_HI="0",
        ),
    )
