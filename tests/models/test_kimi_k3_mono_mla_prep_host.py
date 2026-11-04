# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The MLA fold host check (``_kimi_k3_mono_prep_host_check.py``), run in a
subprocess because it patches ``torch.empty`` & co: the real runner over 93
layers with CPU stand-in kernels, flag off against ``ATOM_MONO_MLA_POST`` and
``ATOM_MONO_MLA_PREP`` 1 / 2 / 3, bit for bit on every drafter-visible buffer, on
"every buffer a launch is handed is still allocated", and on the step's pool
allocations (the CUDA-graph pool layout). Slow (minutes): it runs the stack
a dozen times.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

CHECK = Path(__file__).with_name("_kimi_k3_mono_prep_host_check.py")


def test_mla_fold_host_check_matrix():
    pytest.importorskip("flydsl", reason="the runner imports the FlyDSL kernels")
    pytest.importorskip("aiter", reason="the runner imports aiter.dist")
    root = Path(__file__).resolve().parents[2]
    proc = subprocess.run(
        [sys.executable, str(CHECK)],
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root), "ARCH": "gfx950"},
        capture_output=True,
        text=True,
        timeout=3600,
        check=False,
    )
    failed = [ln for ln in proc.stdout.splitlines() if ln.startswith("FAIL")]
    assert proc.returncode == 0 and not failed, "\n".join(failed) or proc.stderr[-2000:]
    assert "ALL OK" in proc.stdout
