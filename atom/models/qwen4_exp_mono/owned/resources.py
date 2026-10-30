"""Check fresh code, spill and real 512-thread occupancy before any launch."""

import ctypes as C
import re
import subprocess
from pathlib import Path

import torch

from atom.models.qwen4_exp_mono.owned.plan import PLAN
from atom.mono.runtime.compile import _code_object, launcher_resources


def resources(kernel, out, threads=512):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows = launcher_resources(kernel)
    assert rows
    assert all(
        not r["scratch"] and not r.get("vgpr_spill") and not r.get("sgpr_spill")
        for r in rows
    ), rows
    hip = C.CDLL("libamdhip64.so")
    occ = []
    for i, artifact in enumerate(kernel._mem_cache.values()):
        data = _code_object(artifact._ir_text)
        p = out / f"{i}.co"
        p.write_bytes(data)
        isa = subprocess.check_output(
            [
                "/opt/rocm/llvm/bin/llvm-objdump",
                "--disassemble",
                "--mcpu=gfx950",
                str(p),
            ],
            text=True,
        )
        (out / f"{i}.isa").write_text(isa)
        (out / f"{i}.mlir").write_text(artifact._ir_text)
        symbol = re.search(r"^\w+ <([^>]+)>:", isa, re.MULTILINE).group(1)
        mod, fun, n = C.c_void_p(), C.c_void_p(), C.c_int()
        image = C.create_string_buffer(data)
        assert hip.hipModuleLoadData(C.byref(mod), image) == 0
        assert hip.hipModuleGetFunction(C.byref(fun), mod, symbol.encode()) == 0
        assert (
            hip.hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(
                C.byref(n), fun, C.c_int(threads), C.c_size_t(0)
            )
            == 0
        )
        capacity = n.value * torch.cuda.get_device_properties(0).multi_processor_count
        assert capacity >= PLAN.grid, (capacity, PLAN.grid)
        occ.append({"blocks_per_cu": n.value, "capacity": capacity, "threads": threads})
        assert hip.hipModuleUnload(mod) == 0
    return {"resources": rows, "occupancy": occ}
