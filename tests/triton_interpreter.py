# SPDX-License-Identifier: MIT
# Test helper: drive a Triton kernel on CPU tensors, without a GPU.
#
# Triton's interpreter has to be on before `triton.language` is imported (its
# `tl.max`, `tl.sum`, ... are decorated at import), so it cannot be switched on
# inside a process that has already imported Triton. A kernel test module
# therefore has two halves: with TRITON_INTERPRET=1 in the environment it holds
# the real tests; without it, it holds one test that runs the same file again in
# a child process that has the variable set:
#
#     test_under_the_interpreter = interpreter_test(__file__)
#
#     @in_the_interpreter
#     def test_a_kernel(): ...

import os
import subprocess
import sys

import pytest


def interpreting() -> bool:
    return os.environ.get("TRITON_INTERPRET") == "1"


in_the_interpreter = pytest.mark.skipif(
    not interpreting(),
    reason="runs in the child process of test_under_the_interpreter",
)


def interpreter_test(test_file: str, timeout: float = 1800.0):
    """A test that re-runs `test_file` under Triton's CPU interpreter."""

    @pytest.mark.skipif(interpreting(), reason="this is the child process")
    def test_under_the_interpreter():
        pytest.importorskip("triton")
        env = dict(os.environ, TRITON_INTERPRET="1")
        done = subprocess.run(
            [sys.executable, "-m", "pytest", test_file, "-q", "-p", "no:cacheprovider"],
            env=env,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        assert done.returncode == 0, done.stdout[-6000:] + done.stderr[-2000:]

    return test_under_the_interpreter
