# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Where ``mla_back_gfx1250``'s arithmetic comes from: prints what fixes the
bits of the launches it replaces -- AITER's sigmoid source and the gate
kernel's ISA and LLVM IR (Triton's cache, after one launch), the gluon W_UV
config a width, and the ISA of aiter's DCP merge
(``dcp_merge_ll_gfx1250<4, 1>``, extracted from the gfx1250 code object inside
``module_custom_all_reduce_gfx1250.so``; ``<8, 1>`` is saved beside it).
Read off it: the device libm's expf / logf as instructions (``_expf`` /
``_logf``), the merge's ``acc += o * f`` contracted to a fused multiply-add,
the gate's true division.

    python tools/kimi_k3_mono_mla_back_isa.py     (one gfx1250 GPU; OUT: a
    directory for the code object and the listings, /tmp/mla_back_isa)
"""

import glob
import importlib.util
import os
import struct
import subprocess
import sys
from pathlib import Path

import torch

OUT = os.environ.get("OUT", "/tmp/mla_back_isa")
os.makedirs(OUT, exist_ok=True)

print("==== activation.py")
print(
    Path(importlib.util.find_spec("aiter").origin)
    .parent.joinpath("ops/triton/_triton_kernels/common/activation.py")
    .read_text()
)

print("==== gluon config")
from aiter.ops.triton.utils.gemm_config_utils import get_gemm_config

for m in (1, 8, 16, 17, 32, 64):
    for b in (96, 12, 24):
        cfg, tuned = get_gemm_config(
            "BATCHED_GEMM-A16W8", m, 128, 512, backend="gluon", B=b
        )
        print(m, b, tuned, cfg)

print("==== gate kernel")
from aiter.ops.triton.fusions.fused_sigmoid_mul import fused_sigmoid_mul

dev = torch.device("cuda")
for rows in (8,):
    x = torch.randn(rows, 1536, device=dev).to(torch.bfloat16)
    g = torch.randn(rows, 1536, device=dev).to(torch.bfloat16)
    out = torch.empty_like(x)
    fused_sigmoid_mul(x, g, out=out)
    torch.cuda.synchronize()
cache = os.path.expanduser(os.environ.get("TRITON_CACHE_DIR", "~/.triton/cache"))
hits = sorted(
    glob.glob(cache + "/**/*fused_sigmoid_mul*.amdgcn", recursive=True),
    key=os.path.getmtime,
)
print("cache hits", hits[-3:])
if hits:
    text = Path(hits[-1]).read_text()
    Path(OUT + "/gate.amdgcn").write_text(text)
    print(text)
    llir = hits[-1].replace(".amdgcn", ".llir")
    if os.path.exists(llir):
        print("==== gate llir")
        print(Path(llir).read_text())

print("==== merge kernel")
so = "/app/aiter/aiter/jit/module_custom_all_reduce_gfx1250.so"
data = Path(so).read_bytes()
magic = b"__CLANG_OFFLOAD_BUNDLE__"
at = data.find(magic)
print("bundle at", at)
n = struct.unpack_from("<Q", data, at + len(magic))[0]
pos = at + len(magic) + 8
elf = None
for _ in range(n):
    off, size, tlen = struct.unpack_from("<QQQ", data, pos)
    triple = data[pos + 24 : pos + 24 + tlen].decode()
    pos += 24 + tlen
    print("entry", triple, off, size)
    if "gfx1250" in triple:
        elf = data[at + off : at + off + size]
if elf is None:
    sys.exit("no gfx1250 code object")
path = OUT + "/car.elf"
Path(path).write_bytes(elf)
cands = [
    p
    for p in glob.glob("/opt/rocm*/llvm/bin/llvm-objdump")
    + glob.glob("/app/llvm-project/build*/bin/llvm-objdump")
    + glob.glob("/usr/lib/llvm*/bin/llvm-objdump")
    + glob.glob(
        "/usr/local/lib/python3.12/dist-packages/**/llvm-objdump", recursive=True
    )
    if os.path.exists(p)
]
print("objdumps", cands)
for tool in cands:
    syms = subprocess.run(
        [tool, "-t", path], check=False, capture_output=True, text=True
    )
    names = [
        line.split()[-1]
        for line in syms.stdout.splitlines()
        if "dcp_merge_ll" in line and " F " in line
    ]
    print(tool, "symbols", names)
    if not names:
        print(syms.stderr[:500])
        continue
    ok = False
    for name in names:
        if "ILi4ELi1E" in name or "ILi8ELi1E" in name:
            dis = subprocess.run(
                [tool, "-d", "--mcpu=gfx1250", "--disassemble-symbols=" + name, path],
                capture_output=True,
                text=True,
                check=False,
            )
            tag = "4" if "ILi4E" in name else "8"
            Path(OUT + "/merge" + tag + ".s").write_text(dis.stdout)
            print("---- ", name, len(dis.stdout.splitlines()), dis.stderr[:300])
            if tag == "4":
                print(dis.stdout)
            ok = ok or len(dis.stdout.splitlines()) > 50
    if ok:
        break
print("==== END")
