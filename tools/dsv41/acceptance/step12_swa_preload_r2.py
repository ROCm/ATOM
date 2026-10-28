"""Preload BF16 SWA writer scalar, pool extent and cached-pointer classes.

Native D512 contiguous KV and metadata bases are aligned; the pool may exceed
2 GiB and cached_until is a view with offsets modulo four int32 elements.
Keep every native dispatch argument and pointer specialization unchanged.
"""

import hashlib
import itertools


def cases():
    # Pool rows stay int32 and >1 for this model. Triton distinguishes div16
    # from other values; ring_start also has the exact-one specialization.
    return tuple(
        itertools.product(
            (False, True), (False, True), (1, 16, 17), ("none", "aligned", "unaligned")
        )
    )


def preload(window):
    if (
        min(window.ring_slots, window.slot_rows, window.ring_stride, window.run_rows)
        <= 0
    ):
        raise ValueError("SWA layout dimensions must be positive")
    import torch
    from triton.runtime.jit import MockTensor

    from atom.model_ops.v4_kernels.state_writes import _swa_write_kernel

    class Pointer(MockTensor):
        def __init__(self, dtype, *, offset=0, large=False):
            super().__init__(dtype)
            self.offset, self.large = offset, large

        def data_ptr(self):
            return 16 + self.offset * 4

        def ptr_range(self):
            return 2**31 + 1024 if self.large else 0

    bf = Pointer(torch.bfloat16)
    i32, i64 = Pointer(torch.int32), Pointer(torch.int64)
    device = torch.cuda.current_device()
    rows = []
    for large, rows_div16, ring_start, cached in cases():
        old = set(_swa_write_kernel.device_caches[device][0])
        _swa_write_kernel.warmup(
            bf,
            i64,
            i32,
            i32,
            Pointer(torch.bfloat16, large=large),
            512,
            2_100_000 if rows_div16 else 2_100_001,
            512,
            ring_start,
            (
                i64
                if cached == "none"
                else Pointer(torch.int32, offset=int(cached == "unaligned"))
            ),
            BLOCK_D=512,
            CACHED_UNTIL=cached != "none",
            RING_SLOTS=window.ring_slots,
            SLOT_ROWS=window.slot_rows,
            RING_STRIDE=window.ring_stride,
            RUN_ROWS=window.run_rows,
            grid=(3, 8),
        )
        new = set(_swa_write_kernel.device_caches[device][0]) - old
        rows.append(
            {
                "pool_large": large,
                "pool_rows_div16": rows_div16,
                "ring_start": ring_start,
                "cached": cached,
                "new_key_sha256": [
                    hashlib.sha256(k.encode()).hexdigest() for k in sorted(new)
                ],
            }
        )
    return rows
