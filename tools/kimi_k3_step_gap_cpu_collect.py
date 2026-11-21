# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The step-gap test files as a job without aiter, FlyDSL and Triton sees them:
this process hides the three packages (their import raises ImportError), then
runs pytest on the files in-process. A file that fails to collect fails the
run; the guarded files are skipped whole; ``test_kimi_k3_step_gap_host.py`` and
the layout tests must run.

    python tools/kimi_k3_step_gap_cpu_collect.py
"""

import glob
import os
import sys

HIDDEN = ("aiter", "flydsl", "triton")
for name in HIDDEN:
    sys.modules[name] = None

import pytest

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(root)
sys.path.insert(0, root)
files = sorted(glob.glob("tests/models/test_kimi_k3_step_gap_*.py"))
code = int(pytest.main(["-q", "-p", "no:cacheprovider", "-rs", *files]))
leaked = [name for name in HIDDEN if sys.modules.get(name) is not None]
print("CPU_COLLECT", "files", len(files), "exit", code, "leaked", leaked, flush=True)
sys.exit(code or bool(leaked))
