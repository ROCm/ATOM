"""Byte copies between native state storage and its bounded publication packet."""

import triton
import triton.language as tl


@triton.jit
def copy_bytes(backing, packet, descriptors, PACK: tl.constexpr, BLOCK: tl.constexpr):
    tile = tl.program_id(0)
    original = tl.load(descriptors + tile * 3)
    packed = tl.load(descriptors + tile * 3 + 1)
    size = tl.load(descriptors + tile * 3 + 2)
    x = tl.arange(0, BLOCK)
    if PACK:
        value = tl.load(backing + original + x, x < size, 0)
        tl.store(packet + packed + x, value, x < size)
    else:
        value = tl.load(packet + packed + x, x < size, 0)
        tl.store(backing + original + x, value, x < size)
