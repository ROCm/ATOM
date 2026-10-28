"""Preload the BF16 native index writer, retaining all pointer specialization.

Serving metadata bases are aligned and smaller than 2 GiB; cached_until may
be an int32 view. PAGE offsets/table stride/plane are multiples of 16. Ring
starts can be aligned or not, and multi-layer prefill uses the native ring
stride. Decode builds one selection plane and skips ratio zero entirely.
"""

import hashlib
import itertools


def cases():
    return tuple(
        (decode, ratio, aligned, layers, offset)
        for decode, ratio, aligned, layers, offset in itertools.product(
            (False, True), (0, 1, 2), (False, True), (1, 2), (0, 1)
        )
        if not (decode and (ratio == 0 or layers != 1))
    )


def constants(geometry, decode, ratio):
    if geometry.packed or geometry.block_size != 128 or geometry.window_size != 128:
        raise ValueError("This preload covers the actual BF16 block/window128 ABI")
    topk = geometry.batch_topk(ratio)
    if topk != (512 if ratio else 0):
        raise ValueError("Unexpected native selection width")
    window = geometry.window(0, 1)
    if window.ring_start <= 1 or any(
        geometry.main_offset(owner) % 16 for owner, _ in geometry.owners
    ):
        raise ValueError("Recheck native PAGE/STATE scalar specialization")
    return {
        "DECODE": decode,
        "ROWS_PER_PAGE": geometry.block_size // (ratio or 1),
        "PAGE_ROWS": geometry.page_bytes // geometry.row_bytes,
        "TOPK": topk,
        "BLOCK": 1 << (max(topk, geometry.window_size) - 1).bit_length(),
        "RING_SLOTS": window.ring_slots,
        "WINDOW": geometry.window_size,
        "SLOT_ROWS": window.slot_rows,
        "RING_STRIDE": window.ring_stride,
        "RUN_ROWS": window.run_rows,
        "PACKED": False,
        "MAIN_ROW_BYTES": geometry.main_row_bytes,
        "HISTORY": not decode,
    }


def preload(geometry):
    import torch
    from triton.runtime.jit import MockTensor

    from atom.model_ops.attentions.deepseek_v41.indices import _indices

    class Pointer(MockTensor):
        def __init__(self, dtype, offset=0):
            super().__init__(dtype)
            self.offset = offset

        def data_ptr(self):
            return 16 + self.offset * 4

        def ptr_range(self):
            return 0

    i32, i64 = Pointer(torch.int32), Pointer(torch.int64)
    device = torch.cuda.current_device()
    rows = []
    for decode, ratio, aligned, layers, offset in cases():
        kw = constants(geometry, decode, ratio)
        old = set(_indices.device_caches[device][0])
        _indices.warmup(
            i32,
            i32,
            i32,
            i32,
            i32,
            i64,
            i32,
            i32,
            i32,
            i32,
            16,
            0,
            16 if aligned else 17,
            0 if layers == 1 else geometry.ring_slots,
            16 * (kw["TOPK"] + 128 * kw["HISTORY"]),
            i32,
            Pointer(torch.int32, offset),
            **kw,
            grid=(16, layers),
        )
        new = set(_indices.device_caches[device][0]) - old
        rows.append(
            {
                "decode": decode,
                "ratio": ratio,
                "ring_aligned": aligned,
                "layers": layers,
                "cached_offset": offset,
                "new_key_sha256": [
                    hashlib.sha256(k.encode()).hexdigest() for k in sorted(new)
                ],
            }
        )
    return rows
