# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""What production's LM head GEMM dispatch picks at every step-tail width, and
the sources / compiled code the step tail mirrors, printed for reading.

    python tools/steptail_src.py
    env: DUMP (1) print the sources, MS (the widths)
"""

import glob
import logging
import os
import subprocess
import sys

import torch

DUMP = os.environ.get("DUMP", "1") == "1"
MS = [
    int(m)
    for m in os.environ.get(
        "MS", "4,5,7,8,10,12,14,15,16,20,21,24,25,28,30,32,35,40,42,48,49,56,64"
    ).split(",")
]
HIDDEN, VR = 7168, 20480
dev = torch.device("cuda", torch.cuda.current_device())


def emit(path, limit=400_000):
    try:
        with open(path, "rb") as f:
            data = f.read()
    except OSError as err:
        print(f"=====MISSING {path}: {err}")
        return
    print(f"=====FILE {path} {len(data)}")
    sys.stdout.flush()
    sys.stdout.buffer.write(data[:limit])
    sys.stdout.buffer.write(b"\n=====END\n")
    sys.stdout.flush()


def dispatch():
    import aiter
    from aiter.tuned_gemm import tgemm
    from torch.profiler import ProfilerActivity, profile

    log = logging.getLogger("aiter")
    log.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    log.addHandler(handler)
    try:
        import aiter.ops.flydsl.gemm_a16w16_gfx1250 as fly

        original = fly._cached_launcher

        def spy(*cfg):
            print("flydsl cfg", dict(zip(fly._CFG_KEYS, cfg)))
            return original(*cfg)

        fly._cached_launcher = spy
    except Exception as err:  # noqa: BLE001 - a probe
        print("no flydsl spy:", type(err).__name__, err)
    w = (torch.randn(VR, HIDDEN, device=dev) * 0.02).to(torch.bfloat16)
    for m in MS:
        x = torch.randn(m, HIDDEN, device=dev).to(torch.bfloat16)
        print(f"== M {m}")
        tgemm.mm(x, w, None)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
            tgemm.mm(x, w, None)
            torch.cuda.synchronize()
        names = [
            e.name[:150] for e in p.events() if "cuda" in str(e.device_type).lower()
        ]
        print(f"kernels M {m}: {names}")
    return os.path.dirname(aiter.__file__)


def main():
    root = dispatch()
    if not DUMP:
        return
    for rel in (
        "tuned_gemm.py",
        "ops/flydsl/kernels/gemm_a16w16_kernel_gfx1250.py",
        "ops/flydsl/gemm_a16w16_gfx1250.py",
        "ops/flydsl/kernels/topk/topk_per_row_argmax.py",
        "ops/flydsl/kernels/kernels_common.py",
    ):
        emit(os.path.join(root, rel))
    csv = os.path.join(root, "configs/bf16_tuned_gemm.csv")
    print(f"=====FILE {csv}.rows")
    with open(csv) as f:
        for i, line in enumerate(f):
            if i == 0 or ",20480,7168," in line or ",7168,35840," in line:
                print(line.rstrip())
    print("=====END")
    top = os.path.dirname(root)
    hits = subprocess.run(
        [
            "grep",
            "-rlE",
            "clusterlaunch_tdm_splitk_ws|splitk_reduce_kernel_gfx1250",
            top,
            "--include=*.py",
            "--include=*.h",
            "--include=*.hpp",
            "--include=*.cpp",
            "--include=*.cu",
            "--include=*.inc",
        ],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.split()
    print("=====HITS", hits)
    for path in hits[:12]:
        emit(path)
    opus = os.path.join(root, "ops/opus")
    if os.path.isdir(opus):
        print("=====OPUS", sorted(os.listdir(opus)))
        emit(os.path.join(opus, "policy.py"))
    for pattern in ("/root/.triton/cache/*/*.amdgcn", "/tmp/triton*/**/*.amdgcn"):
        for path in glob.glob(pattern, recursive=True):
            if "markov_stage1" in os.path.basename(path):
                emit(path, 600_000)
                return


if __name__ == "__main__":
    main()
