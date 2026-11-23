# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The tests the step tail touches, from the tree's root, ONE PROCESS A FILE
(a file that crashes or fails to collect takes nothing else with it).

``SHIM=1`` binds ``tests`` to this tree's directory (``steptail_tests_shim``):
the container holds another ``tests`` package ahead of it on ``sys.path``, and
the DSpark tests import each other as ``tests.<module>``.

Never run here: ``test_the_step_kernel_traces_on_several_threads_at_once``
(tests/test_deepseek_v41_gfx1250.py). It calls four launchers with zero
arguments on four threads inside ``compile_only()``, whose switch is one
process-wide environment variable: the first block to leave removes it under
the others (``steptail_compile_only_race.py``), and a launcher called then
launches on the GPU with null pointers. That file is not in the default set
and the test is deselected wherever it is named.

    python tools/steptail_pytest.py
    env: SHIM (0), FILES (the default set; comma separated, globs allowed)
"""

import glob
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHIM = os.environ.get("SHIM", "0") == "1"
FILES = os.environ.get("FILES", "")
NEVER = (
    "tests/test_deepseek_v41_gfx1250.py::"
    "test_the_step_kernel_traces_on_several_threads_at_once"
)


def main() -> int:
    os.chdir(ROOT)
    tests = [
        *sorted(glob.glob("tests/test_dspark*.py")),
        "tests/test_mono_architecture.py",
        "tests/test_envs.py",
        "tests/test_kimi_k31_config.py",
        "tests/test_kimi_k3_step_tail.py",
        "tests/test_kimi_k3_step_tail_dispatch.py",
    ]
    if FILES:
        tests = [
            path
            for pattern in FILES.split(",")
            for path in (sorted(glob.glob(pattern)) or [pattern])
        ]
    env = dict(os.environ)
    tools = os.path.join(ROOT, "tools")
    env["PYTHONPATH"] = os.pathsep.join(
        [ROOT, tools, *filter(None, [env.get("PYTHONPATH")])]
    )
    args = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-rfE"]
    if SHIM:
        args += ["-p", "steptail_tests_shim"]
    args += ["--deselect", NEVER]
    worst = 0
    for path in tests:
        done = subprocess.run(
            [*args, path], env=env, capture_output=True, text=True, check=False
        )
        lines = [ln for ln in done.stdout.splitlines() if ln.strip()]
        for ln in lines:
            if ln.startswith(("FAILED", "ERROR")):
                print("  ", ln[:200])
        print(f"{path}: exit {done.returncode}: {lines[-1][:120] if lines else ''}")
        if done.returncode < 0 or "Fatal Python" in done.stderr:
            print("   CRASHED:", done.stderr.strip().splitlines()[:3])
        worst = max(worst, abs(done.returncode))
    print("PYTEST_WORST_EXIT", worst, flush=True)
    return worst


if __name__ == "__main__":
    sys.exit(main())
