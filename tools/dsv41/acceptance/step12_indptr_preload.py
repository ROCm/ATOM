"""Cover the native BF16 indptr scan's finite scalar/alignment dispatch domain.

BLOCK is min(1024, next_power_of_2(width)). Runtime widths specialize as one,
divisible by 16, or other. Only cached_until may start at an int32 view offset;
the other serving buffers have aligned bases. Keep all native specialization.
"""

import hashlib


def width_representatives():
    """One width for every reachable (BLOCK, scalar specialization) pair."""
    choices = {}
    for width in range(1, 16385):
        block = min(1024, 1 << (width - 1).bit_length())
        scalar = "one" if width == 1 else "div16" if width % 16 == 0 else "other"
        choices.setdefault((block, scalar), width)
    return tuple((width, block, scalar) for (block, scalar), width in choices.items())


def preload():
    import torch
    from triton.runtime.jit import MockTensor

    from atom.model_ops.attentions.deepseek_v41.indices import _indptr_scan_all

    class Pointer(MockTensor):
        def __init__(self, dtype, offset=0):
            super().__init__(dtype)
            self.offset = offset

        def data_ptr(self):
            return 16 + self.offset * 4

        def ptr_range(self):
            return 0

    i32, i64 = Pointer(torch.int32), Pointer(torch.int64)
    rows = []
    device = torch.cuda.current_device()
    for decode in (False, True):
        for width, block, scalar in width_representatives():
            for offset in (0, 1):
                old = set(_indptr_scan_all.device_caches[device][0])
                _indptr_scan_all.warmup(
                    i32,
                    i64,
                    i32,
                    (i32, i32, i32),
                    (i32, i32, i32),
                    width,
                    i32,
                    Pointer(torch.int32, offset),
                    DECODE=decode,
                    WINDOW=128,
                    RATIOS=(0, 1, 2),
                    TOPKS=(0, 512, 512),
                    SPLIT=decode,
                    BLOCK=block,
                    grid=(3,),
                )
                new = set(_indptr_scan_all.device_caches[device][0]) - old
                rows.append(
                    {
                        "decode": decode,
                        "split": decode,
                        "width": width,
                        "block": block,
                        "scalar": scalar,
                        "cached_offset": offset,
                        "new_key_sha256": [
                            hashlib.sha256(key.encode()).hexdigest()
                            for key in sorted(new)
                        ],
                    }
                )
    return rows
