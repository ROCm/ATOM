"""Probe whether the GPU can reach every row of a production-sized SparseKV cold pool.

The decode workers pin one ``[layers, host_pages*page, kv_dim]`` host tensor per
rank (~247 GB at RATIO=14) and hand the swap kernels a ``hipHostGetDevicePointer``
translation of each layer's view. A memory access fault that only shows up once
the pool is ~99% occupied is consistent with that mapping not covering the whole
allocation: the free list pops from the top, so the lowest page indices are the
last rows a run ever touches.

This walks every row of every layer through the real gather kernel and reports
the first chunk that faults, which turns a two-hour replay into a few minutes.
Run it with the servers down — it pins the same memory they do.

  python scripts/probe_sparsekv_host_mapping.py                  # one rank
  HOST_PAGES=344106 GPU=4 python scripts/probe_sparsekv_host_mapping.py
  bash scripts/probe_sparsekv_host_mapping.sh                    # all four ranks
"""

import os
import sys
import time

import torch

from atom.sparsekv.swap_kernel import host_get_device_pointer, sparsekv_swap_in

LAYERS = int(os.environ.get("LAYERS", "78"))
HOST_PAGES = int(os.environ.get("HOST_PAGES", "344106"))
PAGE = int(os.environ.get("PAGE", "16"))
KV_DIM = int(os.environ.get("KV_DIM", "576"))
CHUNK_ROWS = int(os.environ.get("CHUNK_ROWS", "262144"))
TAG = os.environ.get("TAG", "probe")


def log(msg: str) -> None:
    print(f"[{TAG}] {msg}", flush=True)


def main() -> int:
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    rows = HOST_PAGES * PAGE
    item_size = KV_DIM  # fp8_e4m3: one byte per element
    total_gb = LAYERS * rows * item_size / 1e9
    log(
        f"pool: layers={LAYERS} rows/layer={rows} kv_dim={KV_DIM} "
        f"({total_gb:.2f} GB) chunk={CHUNK_ROWS} rows"
    )

    t0 = time.time()
    pool = torch.zeros(
        (LAYERS, rows, KV_DIM), dtype=torch.float8_e4m3fn, pin_memory=True
    )
    log(f"pinned in {time.time() - t0:.1f}s at host VA 0x{pool.data_ptr():x}")

    # Byte pattern keyed on the row index, so a gather that lands on the wrong
    # row (an aliased mapping) is visible without faulting.
    pool_u8 = pool.view(torch.uint8)
    row_ids = torch.arange(rows, dtype=torch.int64)
    pool_u8[:, :, 0] = (row_ids % 251).to(torch.uint8)
    pool_u8[:, :, 1] = ((row_ids // 251) % 251).to(torch.uint8)

    dst = torch.zeros((CHUNK_ROWS, KV_DIM), dtype=torch.float8_e4m3fn, device=device)
    dst_locs = torch.arange(CHUNK_ROWS, dtype=torch.int32, device=device)

    bad_rows = 0
    for layer in range(LAYERS):
        layer_view = pool[layer]
        dev_ptr = host_get_device_pointer(layer_view)
        if layer == 0 or layer == LAYERS - 1:
            log(f"layer {layer}: device ptr 0x{dev_ptr:x} span {rows * item_size} B")
        for start in range(0, rows, CHUNK_ROWS):
            n = min(CHUNK_ROWS, rows - start)
            src = torch.arange(start, start + n, dtype=torch.int32, device=device)
            sparsekv_swap_in(dev_ptr, dst, src, dst_locs[:n], item_size)
            torch.cuda.synchronize()
            if layer in (0, LAYERS - 1):
                got = dst[:n].view(torch.uint8)[:, :2].cpu().to(torch.int64)
                want_lo = torch.arange(start, start + n) % 251
                want_hi = (torch.arange(start, start + n) // 251) % 251
                mism = (got[:, 0] != want_lo) | (got[:, 1] != want_hi)
                if mism.any():
                    first = int(mism.nonzero()[0])
                    bad_rows += int(mism.sum())
                    log(
                        f"MISMATCH layer {layer} row {start + first}: "
                        f"got {int(got[first, 0])},{int(got[first, 1])} "
                        f"want {int(want_lo[first])},{int(want_hi[first])}"
                    )
        log(f"layer {layer}: {rows} rows readable")

    log(f"ALL LAYERS READABLE ({total_gb:.2f} GB swept), mismatched rows: {bad_rows}")
    return 1 if bad_rows else 0


if __name__ == "__main__":
    sys.exit(main())
