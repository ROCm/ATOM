"""Reproduce the decode-node memory fault, and prove the row bound stops it.

An out-of-range row in a SparseKV translation table is not a wrong answer — the
cold pools are sized exactly, so the swap kernels dereference host memory the
agent has no mapping for and the whole process dies with

    Memory access fault by GPU node-N on address 0x... Reason: Unknown.

which is the fault that ended the c48 round in `results/joint_sizing_m48_r14`.
This poisons a table on purpose and checks the kernels report the row unbacked
and skip it (see `sparsekv_set_pool_rows`). Before the bound existed this script
killed its own process at the first poisoned gather — which is what makes it a
regression test rather than a smoke test.

Needs one free GPU and about thirty seconds:

  docker exec -e HIP_VISIBLE_DEVICES=4 -e PYTHONPATH=/it-share/yajizhan/code/ATOM \\
    atom_pp4pd_test python3 scripts/check_sparsekv_row_bounds.py
"""

import sys

import torch

from atom.sparsekv.coordinator import SparseKVCoordinator
from atom.sparsekv.swap_kernel import sparsekv_swap_in

LAYERS, MAX_SEQS, HOT, MAX_CTX, KV_DIM, PAGE = 2, 2, 8, 256, 576, 16


def main() -> int:
    if not torch.cuda.is_available():
        print("SKIP: needs a GPU")
        return 0
    c = SparseKVCoordinator(
        num_layers=LAYERS,
        max_num_seqs=MAX_SEQS,
        hot_buffer_size=HOT,
        max_context_len=MAX_CTX,
        kv_dim=KV_DIM,
        kv_dtype=torch.float8_e4m3fn,
        device="cuda",
        index_topk=16,
        host_to_device_ratio=8,
        page_size=PAGE,
        num_gpu_cold_pages=4,
    )
    rows = c.num_host_pages * c.host_page_size
    print(f"host pool rows={rows}, gpu rows={c.num_gpu_pages * c.host_page_size}")

    slot = c.acquire(req_id=1, context_len=64)
    c.alloc_host_pages(slot, 0, 64)
    torch.cuda.synchronize()

    dev_ptr = c._ensure_cold_dev_ptr(0)
    src = c.req_to_host_pool[slot, :16].clone()
    dst = torch.arange(16, dtype=torch.int32, device="cuda")
    c.cold_pool[0, src.cpu().to(torch.long)] = torch.full((16, KV_DIM), 1.0).to(
        torch.float8_e4m3fn
    )
    out = torch.zeros((16, KV_DIM), dtype=torch.float8_e4m3fn, device="cuda")

    sparsekv_swap_in(dev_ptr, out, src, dst, KV_DIM)
    torch.cuda.synchronize()
    if not bool((out.float() == 1.0).all().item()):
        print("FAIL: in-range gather did not move data — the bound is too tight")
        return 1
    print("ok: in-range gather still moves data")

    out.zero_()
    poison = torch.full((16,), rows + 1_000_000, dtype=torch.int32, device="cuda")
    sparsekv_swap_in(dev_ptr, out, poison, dst, KV_DIM)
    torch.cuda.synchronize()
    if not bool((out.float() == 0.0).all().item()):
        print("FAIL: out-of-range gather wrote something")
        return 1
    print("ok: out-of-range gather skipped")

    # The path the production fault actually came through: a poisoned entry in
    # req_to_host_pool, resolved by cold_row_of inside the kernel.
    c.req_to_host_pool[slot, :16] = rows + 5_000_000
    c.load_initial_hot_set(slot, 64)
    torch.cuda.synchronize()
    print("ok: poisoned translation table survived load_initial_hot_set")
    print("ROW_BOUNDS_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
