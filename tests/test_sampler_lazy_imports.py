# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Regression test for atom/model_ops/sampler.py lazy imports.

The aiter Triton drop-in `aiter.ops.triton.sample.mix_sample` is not
universally present (e.g. ROCm/aiter:main does not ship it as of 2026-05).
This file's imports MUST stay lazy so ATOM is startable regardless of which
aiter branch is mounted; only when ATOM_USE_TRITON_SAMPLE=1 at call time
should the Triton import fire.

Uses a subprocess so the test is isolated from any other test-collection
side effects (the ATOM test suite has several unrelated import-time issues
when collected as a whole).
"""
import os
import subprocess
import sys
import textwrap


def test_sampler_imports_without_mix_sample():
    """Spawn a fresh interpreter that hides aiter.ops.triton.sample.mix_sample
    from the import system, then imports atom.model_ops.sampler. If anyone
    moves the Triton import back to module top, the spawned interpreter will
    exit non-zero with ModuleNotFoundError.
    """
    code = textwrap.dedent(
        """
        import sys

        class _Blocker:
            def find_spec(self, fullname, path=None, target=None):
                if fullname == 'aiter.ops.triton.sample.mix_sample':
                    raise ImportError(
                        'test forced ModuleNotFoundError for '
                        + fullname
                    )
                return None

        # Insert the blocker BEFORE any aiter or atom import.
        sys.meta_path.insert(0, _Blocker())

        # The load-bearing assertion: this import must succeed without ever
        # touching aiter.ops.triton.sample.mix_sample.
        import atom.model_ops.sampler as sampler
        assert callable(sampler.mixed_sample_outer_exponential)
        print('SAMPLER_IMPORT_OK')
        """
    )
    env = os.environ.copy()
    # The default test interpreter already has the right PYTHONPATH if the
    # test is invoked via the validate container shell; otherwise the caller
    # is expected to mount /app/ATOM + /app/aiter-test.
    proc = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, (
        f"subprocess failed (returncode={proc.returncode})\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )
    assert "SAMPLER_IMPORT_OK" in proc.stdout, (
        f"sentinel not found in stdout:\n{proc.stdout}"
    )
