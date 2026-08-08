"""Measure the SparseKV planned gather against the bandwidth it could reach.

The decode profile puts this kernel at the top of the list, so the question is
whether it is slow or whether it is simply moving a lot of bytes across PCIe.
This times the real op at a realistic shape and prints the bandwidth it achieves
next to two references:

  * a contiguous pinned-host -> device copy of the same byte count, which is the
    ceiling for anything reading the host cold pool;
  * a contiguous device -> device copy, the ceiling for the GPU cold tier.

A gather that lands near its ceiling is done — the remaining lever is moving
fewer bytes (a larger GPU tier, a higher hot-buffer hit rate), not a faster
kernel.

  docker exec -e HIP_VISIBLE_DEVICES=4 -e PYTHONPATH=/it-share/yajizhan/code/ATOM \\
    atom_pp4pd_test python3 scripts/bench_sparsekv_gather.py
"""

import sys

import torch

from atom.sparsekv.swap_kernel import (
    host_get_device_pointer,
    set_pool_rows,
    sparsekv_gather_planned_dual,
)

KV_DIM = 576  # MLA compressed KV, fp8: one byte per element
HOST_ROWS = 1 << 21  # 2M rows ~= 1.2 GB, big enough to defeat any cache
GPU_ROWS = 1 << 18
WARMUP, ITERS = 5, 30


def timed(fn) -> float:
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(ITERS):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / ITERS  # ms


def main() -> int:
    if not torch.cuda.is_available():
        print("SKIP: needs a GPU")
        return 0
    dev = torch.device("cuda")
    host = torch.zeros((HOST_ROWS, KV_DIM), dtype=torch.float8_e4m3fn, pin_memory=True)
    gpu = torch.zeros((GPU_ROWS, KV_DIM), dtype=torch.float8_e4m3fn, device=dev)
    host_ptr = host_get_device_pointer(host)
    set_pool_rows(HOST_ROWS, GPU_ROWS)

    # References: contiguous copies of a comparable block.
    block = 64 << 20  # 64 MB
    src_h = torch.zeros(block, dtype=torch.uint8, pin_memory=True)
    dst_d = torch.zeros(block, dtype=torch.uint8, device=dev)
    src_d = torch.zeros(block, dtype=torch.uint8, device=dev)
    ms = timed(lambda: dst_d.copy_(src_h, non_blocking=True))
    h2d = block / (ms / 1e3) / 1e9
    ms = timed(lambda: dst_d.copy_(src_d, non_blocking=True))
    d2d = block / (ms / 1e3) / 1e9
    print(f"reference contiguous H2D : {h2d:8.1f} GB/s")
    print(f"reference contiguous D2D : {d2d:8.1f} GB/s")

    # Realistic decode shape: one block per query token, topk misses each.
    n, topk = 16, 2048
    hot = torch.zeros((n * (topk + 1), KV_DIM), dtype=torch.float8_e4m3fn, device=dev)
    req_slots = torch.arange(n, dtype=torch.int32, device=dev)
    tok = torch.arange(topk, dtype=torch.int32, device=dev).repeat(n, 1)
    slot = torch.arange(topk, dtype=torch.int32, device=dev).repeat(n, 1)
    count = torch.full((n,), topk, dtype=torch.int32, device=dev)
    stride = topk * 4  # translation-table stride; scatter the rows widely
    host_locs = torch.randint(0, HOST_ROWS, (n, stride), dtype=torch.int32, device=dev)
    gpu_locs = torch.randint(0, GPU_ROWS, (n, stride), dtype=torch.int32, device=dev)

    print(f"\nshape: n={n} queries x topk={topk} misses, {KV_DIM} B rows")
    print(f"{'host share':>11} {'ms':>8} {'GB/s':>9} {'% of its ceiling':>18}")
    for host_share in (1.0, 0.9, 0.5, 0.0):
        home = (torch.rand(n, stride, device=dev) >= host_share).to(torch.int32)
        moved = n * topk * KV_DIM

        def run(home=home):
            sparsekv_gather_planned_dual(
                host_ptr,
                gpu.data_ptr(),
                hot,
                req_slots,
                tok,
                slot,
                count,
                home,
                host_locs,
                stride,
                gpu_locs,
                stride,
                KV_DIM,
                topk + 1,
                stride,
                topk,
            )

        ms = timed(run)
        bw = moved / (ms / 1e3) / 1e9
        ceiling = host_share * h2d + (1 - host_share) * d2d
        print(f"{host_share:>11.0%} {ms:>8.3f} {bw:>9.1f} {100 * bw / ceiling:>17.0f}%")

    print(
        "\nNote: the ceilings are contiguous-copy rates; a 576 B scattered gather "
        "cannot reach them.\nWhat matters is whether the gap leaves room worth "
        "chasing in the kernel."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
