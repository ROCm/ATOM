# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The gfx1250 instruction prefetch prologue (atom/utils/gfx1250_inst_prefetch.py).

CPU only: the block's arithmetic, where it goes in a kernel's assembly, and -- when Triton's gfx1250
backend is installed -- that a compiled kernel's prefetched range lies inside its own code (prefetching
past the code object faults the queue).
"""

import itertools
import re
import subprocess
import tempfile

import pytest

from atom.utils import gfx1250_inst_prefetch as ip


def _ranges(block, block_offset):
    """[(start, end)] bytes of the kernel each prefetch covers, the block placed at block_offset."""
    out = []
    for i, ln in enumerate(b for b in block if "s_prefetch_inst_pc_rel" in b):
        m = re.search(
            r"s_prefetch_inst_pc_rel\s+(0x[0-9a-f]+|\d+),\s*null,\s*(\d+)", ln
        )
        next_pc = block_offset + 8 + 8 * (i + 1)  # setreg, then 8 bytes a prefetch
        beg = next_pc + int(m.group(1), 0)
        out.append((beg, beg + (int(m.group(2)) + 1) * 128))
    return out


@pytest.mark.parametrize(
    "code_bytes", [200, 255, 256, 4096, 4224, 15944, 35984, 131072]
)
@pytest.mark.parametrize("block_offset", [0, 28, 96])
def test_block_stays_inside_the_code(code_bytes, block_offset):
    block = ip.prefetch_block(code_bytes)
    if code_bytes < 256:
        assert block == []
        return
    assert (
        block[0]
        .strip()
        .startswith("s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 24, 1), 1")
    )
    rng = _ranges(block, block_offset)
    # contiguous, starts right after the block, ends inside the code as compiled without the block
    assert rng[0][0] == block_offset + 16
    assert all(a[1] == b[0] for a, b in itertools.pairwise(rng))
    assert rng[-1][1] <= code_bytes
    assert (
        rng[-1][1] > code_bytes - 128 - 128 + block_offset - 96
    )  # nearly the whole kernel
    assert all(e - s <= 32 * 128 for s, e in rng)


_ASM = """\t.text
\t.globl\tk
k:                                      ; @k
; %bb.0:
\ts_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
\ts_mov_b64 s[64:65], 0
\tv_nop
\tglobal_prefetch_b8 v0, s[64:65] scope:SCOPE_SE
\ts_load_b32 s6, s[6:7], 0x0
\ts_endpgm
; codeLenInByte = 9000
"""


def test_block_goes_after_the_required_prologue():
    out = ip.add_prefetch(_ASM, "k").split("\n")
    i = next(j for j, ln in enumerate(out) if "global_prefetch_b8" in ln)
    assert "s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 24, 1)" in out[i + 1]
    assert (
        sum("s_prefetch_inst_pc_rel" in ln for ln in out) == 3
    )  # 9000 - 128 bytes in 4 KB pieces
    assert out[i + 5].strip().startswith("s_load_b32")
    # once only, and kernels without the prologue or the size are left alone
    assert ip.add_prefetch("\n".join(out), "k") == "\n".join(out)
    assert ip.add_prefetch(
        _ASM.replace("global_prefetch_b8", "s_nop 0"), "k"
    ) == _ASM.replace("global_prefetch_b8", "s_nop 0")
    assert ip.add_prefetch(
        _ASM.replace("; codeLenInByte = 9000", ""), "k"
    ) == _ASM.replace("; codeLenInByte = 9000", "")


def test_compiled_kernel_range():
    """Compile a kernel for gfx1250 (no GPU needed) through the hook and check the assembled object."""
    triton = pytest.importorskip("triton")
    tl = pytest.importorskip("triton.language")
    try:
        from triton.backends.compiler import GPUTarget
        from triton.compiler import ASTSource
    except ImportError:
        pytest.skip("no Triton compiler API")
    from triton import knobs

    @triton.jit
    def _straight_line(x_ptr, N: tl.constexpr):
        offs = tl.arange(0, 32)
        x = tl.load(x_ptr + offs)
        for i in tl.static_range(N):
            x = x * 0.999 + (i * 0.001)
        tl.store(x_ptr + offs, x)

    prev = knobs.runtime.add_stages_inspection_hook
    ip._installed.clear()
    try:
        ip.install("_straight_line")
        src = ASTSource(
            _straight_line, {"x_ptr": "*fp32", "N": "constexpr"}, {"N": 700}
        )
        try:
            ck = triton.compile(
                src, target=GPUTarget("hip", "gfx1250", 32), options={"num_warps": 1}
            )
        except Exception as e:  # noqa: BLE001
            pytest.skip(f"no gfx1250 Triton backend: {e}")
    finally:
        knobs.runtime.add_stages_inspection_hook = prev
        ip._installed.clear()
    asm = ck.asm["amdgcn"]
    assert "s_prefetch_inst_pc_rel" in asm
    objdump = ["llvm-objdump", "-d", "--mcpu=gfx1250"]
    with tempfile.NamedTemporaryFile(suffix=".hsaco") as f:
        f.write(ck.asm["hsaco"])
        f.flush()
        try:
            dis = subprocess.run(
                objdump + [f.name], capture_output=True, text=True, check=True
            ).stdout
            syms = subprocess.run(
                ["llvm-readelf", "-s", "-W", f.name],
                capture_output=True,
                text=True,
                check=True,
            ).stdout
        except (OSError, subprocess.CalledProcessError):
            pytest.skip("no llvm-objdump")
    size = next(
        int(p[2])
        for p in (ln.split() for ln in syms.splitlines())
        if len(p) >= 8 and p[-1] == ck.name
    )
    start, rng = None, []
    for ln in dis.splitlines():
        m = re.search(r"//\s*([0-9A-F]+):", ln)
        if not m:
            continue
        addr = int(m.group(1), 16)
        start = addr if start is None else start
        mm = re.search(
            r"s_prefetch_inst_pc_rel\s+(0x[0-9a-f]+|\d+),\s*null,\s*(\d+)", ln
        )
        if mm:
            beg = addr + 8 + int(mm.group(1), 0) - start
            rng.append((beg, beg + (int(mm.group(2)) + 1) * 128))
    assert rng and min(r[0] for r in rng) >= 0 and max(r[1] for r in rng) <= size
