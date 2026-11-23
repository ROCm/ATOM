# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The OPUS split-K LM-head kernel's sources and the tuned rows the step tail
mirrors, printed for reading.

    python tools/steptail_src2.py
"""

import glob
import os
import subprocess
import sys


def emit(path, limit=300_000):
    with open(path, "rb") as f:
        data = f.read()
    print(f"=====FILE {path} {len(data)}")
    sys.stdout.flush()
    sys.stdout.buffer.write(data[:limit])
    sys.stdout.buffer.write(b"\n=====END\n")
    sys.stdout.flush()


def main():
    import aiter

    top = os.path.dirname(os.path.dirname(aiter.__file__))
    for csv in glob.glob("/tmp/aiter_configs/bf16_tuned_gemm.csv"):
        print(f"=====FILE {csv}.rows")
        with open(csv) as f:
            for i, line in enumerate(f):
                if i == 0 or ",20480,7168," in line or ",7168,35840," in line:
                    print(line.rstrip())
        print("=====END")
    names = (
        "opus_gemm_pipeline_a16w16_clusterlaunch_tdm_splitk_ws_gfx1250.cuh",
        "opus_gemm_traits_a16w16_gfx1250.cuh",
        "splitk_reduce_launch_gfx1250.cuh",
    )
    found = []
    for root, _, files in os.walk(os.path.join(top, "csrc")):
        for name in files:
            if name in names:
                found.append(os.path.join(root, name))
    hits = subprocess.run(
        ["grep", "-rlE", "splitk_reduce_kernel_gfx1250", os.path.join(top, "csrc")],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.split()
    for path in hits:
        if path.endswith((".cuh", ".h", ".hpp", ".cu", ".cpp")) and path not in found:
            found.append(path)
    print("=====FOUND", found)
    for path in found[:8]:
        emit(path)
    # the compiled stage 2 of the Markov argmax and the argmax fold, if cached
    for path in glob.glob("/root/.triton/cache/*/*.amdgcn"):
        if "argmax_stage2" in os.path.basename(path):
            emit(path, 200_000)
            break


if __name__ == "__main__":
    main()
