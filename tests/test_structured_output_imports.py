# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Keep optional structured-output dependencies off ordinary engine imports."""

import subprocess
import sys


def test_package_entry_points_do_not_load_backend_or_gpu_helpers():
    code = """
import sys
import atom.structured_output as so

assert callable(so.prepare_worker_grammar)
assert callable(so.StructuredOutput)
assert not any(n == "llguidance" or n.startswith("llguidance.") for n in sys.modules)
assert all(
    "atom.structured_output." + n not in sys.modules
    for n in ("backend_guidance", "mask", "transfer")
)
assert "atom.model_ops.triton_structured_output" not in sys.modules
"""
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
