"""Read-only: the float modes (kernel descriptors) of the aiter HIP kernels
the draft MLP path launches and of the draft mono kernels, and the mono
kernels' ISA."""

import os
import re
import struct
import subprocess
import tempfile
from pathlib import Path

from atom.models.kimi_k3.mono.kernels.stages import draft_gate_up_gfx1250 as gk
from atom.models.kimi_k3.mono.kernels.stages import draft_proj_ar_gfx1250 as dp
from atom.mono.runtime.compile import _code_object, compile_kernels

OBJDUMP = "/opt/rocm/llvm/bin/llvm-objdump"
READELF = "/opt/rocm/llvm/bin/llvm-readelf"
OUT = os.environ.get("DRAFT_ISA_DIR", os.path.join(tempfile.gettempdir(), "draft_isa"))


def modes(path, wanted=None):
    syms = subprocess.run(
        [READELF, "-sW", path], check=False, capture_output=True, text=True
    ).stdout
    secs = subprocess.run(
        [READELF, "-SW", path], check=False, capture_output=True, text=True
    ).stdout
    sec = {}
    for m in re.finditer(
        r"\[\s*(\d+)\]\s+(\S+)\s+\S+\s+([0-9a-f]+)\s+([0-9a-f]+)\s+([0-9a-f]+)", secs
    ):
        sec[int(m.group(1))] = (int(m.group(3), 16), int(m.group(4), 16))
    data = Path(path).read_bytes()
    for line in syms.splitlines():
        parts = line.split()
        if len(parts) < 8 or not parts[-1].endswith(".kd") or not parts[6].isdigit():
            continue
        if wanted is not None and not any(w in parts[-1] for w in wanted):
            continue
        value, (addr, off) = int(parts[1], 16), sec[int(parts[6])]
        kd = data[off + value - addr : off + value - addr + 64]
        rsrc1 = struct.unpack_from("<I", kd, 48)[0]
        print(
            "KD", parts[-1][:110], hex(rsrc1), "round32", (rsrc1 >> 12) & 3,
            "denorm32", (rsrc1 >> 16) & 3, "denorm16_64", (rsrc1 >> 18) & 3,
            "bit21", (rsrc1 >> 21) & 1, "bit23", (rsrc1 >> 23) & 1, flush=True,
        )  # fmt: skip


HIP = {
    "rms": ["add_rmsnorm_quant_kernelIDF16bDB8_Li1024ELi8ELb1ELb1ELb1ELi1ELb0E"],
    "act": ["silu_kernelIDF16bEEfS3_EELi8ELb0ELi3E"],
    "quant": ["quant_kernelIDF16bDB8_Li32ELi128ELb1ELi64ELb1ELb0ELi0ELb0ELb0ELb0ELb0ELi2ELi1ELb0E",
              "quant_kernelIDF16bDB8_Li32ELi128ELb1ELi64ELb1ELb0ELi0ELb1ELb0ELb0ELb0ELi2ELi1ELb0E"],
    "ar": ["ar_ll_gfx1250IDF16bLi4ELi1E", "ar_ll_gfx1250IDF16bLi8ELi1E",
           "ar_2s_ll_gfx1250IDF16bLi8ELi3E", "ar_2s_ll_gfx1250IDF16bLi4ELi3E"],
}  # fmt: skip
for tag, wanted in HIP.items():
    d = f"{OUT}/{tag}"
    for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        if "gfx1250" in f and "strict" not in f and "hipv4" in f:
            print("== HIP", tag, f, flush=True)
            modes(os.path.join(d, f), wanted)
            break

BUILDS = {
    "proj_s28_k1792": lambda: dp.build_draft_proj_ar(28, 1, 1792, poll_bound=1 << 22),
    "gate_up_s7": lambda: gk.build_draft_gate_up(7, 1792, 4),
}
for tag, build in BUILDS.items():
    kernel = build()
    (launcher,) = compile_kernels([(kernel.launcher, kernel.abi)])
    for i, artifact in enumerate(launcher._mem_cache.values()):
        path = f"{OUT}/mono_{tag}_{i}.elf"
        with open(path, "wb") as f:
            f.write(_code_object(artifact._ir_text))
        print("== MONO", tag, flush=True)
        modes(path)
        dis = subprocess.run(
            [OBJDUMP, "-d", "--mcpu=gfx1250", path],
            capture_output=True,
            text=True,
            check=False,
        ).stdout
        print(f"=====ISA {tag}")
        for line in dis.splitlines():
            print(re.sub(r"\s+//.*$", "", line).strip())
        print("=====END", flush=True)
