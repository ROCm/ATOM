"""Original grouped TP sum boundary used by the owner output projection."""

import triton
import triton.language as tl


@triton.jit
def sum_original_tp(
    parts,
    out,
    OWNER: tl.constexpr,
    H: tl.constexpr,
    SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offset < SIZE
    base = (offset // H) * (4 * H) + offset % H
    acc = tl.load(parts + base + OWNER * H, mask, 0).to(tl.float32)
    for step in tl.static_range(1, 4):
        acc += tl.load(parts + base + ((OWNER + step) % 4) * H, mask, 0).to(tl.float32)
    tl.store(out + offset, acc, mask)
