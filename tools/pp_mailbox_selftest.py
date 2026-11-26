# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""Hardware check of the PP mailbox (atom/distributed/pp_mailbox.py) before a
server turns it on: each rank plays one PP stage of Kimi-K3's shapes, passes a
patterned payload down the chain for ``--steps`` microbatches of random size,
checks every element that arrives, and times the chain against the same chain
over NCCL (today's path: gloo-free here, but the same rendezvous and the whole
AttnRes bank).

One node, one rank per GPU (hops between nodes stay on NCCL in both modes)::

    torchrun --nproc_per_node 4 tools/pp_mailbox_selftest.py --transport mailbox
    torchrun --nproc_per_node 4 tools/pp_mailbox_selftest.py --transport nccl

``--check-every 0`` skips the element checks (each is a host sync) for a timing run.
``--compute-ms`` makes each stage busy for about that long per microbatch (bf16
GEMMs on the compute stream), so the copies have something to overlap with.
``--first-layers`` gives each rank's first layer (default: the first stages of
#813's 12,13,13,13,12,11,11,8 split); the bank blocks and DSpark aux taps that
cross each hop follow from it, as in the server.
"""

import argparse
import os
import statistics
import time
from types import SimpleNamespace

import torch
import torch.distributed as dist

from atom.distributed.pp_mailbox import PPMailbox, closed_bank_blocks

HIDDEN, BLOCK, LAYERS = 7168, 12, 93
TOTAL_BLOCKS = -(-LAYERS // BLOCK)
TAPS = (2, 23, 47, 71, 89)
SPLIT_813 = (12, 13, 13, 13, 12, 11, 11, 8)


def pattern(
    step: int, stage: int, rows: int, cols: int, salt: int, device
) -> torch.Tensor:
    """Exactly representable in bf16: small integers by row and column, keyed by step / stage / salt."""
    r = torch.arange(rows, device=device).unsqueeze(1)
    c = torch.arange(cols, device=device).unsqueeze(0)
    return ((r * 7 + c * 3 + step * 5 + stage * 11 + salt * 13) % 61 - 30).to(
        torch.bfloat16
    )


def payload(step, stage, rows, closed, aux_taps, device):
    hidden = pattern(step, stage, rows, HIDDEN, 0, device)
    bank = torch.empty(rows, TOTAL_BLOCKS, HIDDEN, dtype=torch.bfloat16, device=device)
    for b in range(closed):
        bank[:, b] = pattern(step, stage, rows, HIDDEN, 1 + b, device)
    aux = (
        pattern(step, stage, rows, aux_taps * HIDDEN, 99, device) if aux_taps else None
    )
    return {
        "hidden_states": hidden,
        "block_residual": bank,
        **({"pp_aux": aux} if aux is not None else {}),
    }


def check(got, step, stage, rows, closed, aux_taps, device) -> int:
    want = payload(step, stage, rows, closed, aux_taps, device)
    bad = int((got["hidden_states"] != want["hidden_states"]).sum())
    bad += int(
        (got["block_residual"][:, :closed] != want["block_residual"][:, :closed]).sum()
    )
    if aux_taps:
        bad += int((got["pp_aux"] != want["pp_aux"]).sum())
    return bad


def nccl_send(tensors, dst):
    """Today's path: every tensor, the whole bank, one NCCL send each."""
    return [dist.isend(t.contiguous(), dst) for t in tensors.values()]


def nccl_recv(rows, aux_taps, src, device):
    out = {
        "hidden_states": torch.empty(rows, HIDDEN, dtype=torch.bfloat16, device=device),
        "block_residual": torch.empty(
            rows, TOTAL_BLOCKS, HIDDEN, dtype=torch.bfloat16, device=device
        ),
    }
    if aux_taps:
        out["pp_aux"] = torch.empty(
            rows, aux_taps * HIDDEN, dtype=torch.bfloat16, device=device
        )
    for t in out.values():
        dist.recv(t, src)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transport", choices=("mailbox", "nccl"), default="mailbox")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--max-rows", type=int, default=16384)
    ap.add_argument("--compute-ms", type=float, default=150.0)
    ap.add_argument("--slots", type=int, default=2)
    ap.add_argument("--check-every", type=int, default=1)
    ap.add_argument("--first-layers", type=str, default="")
    args = ap.parse_args()

    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    dist.init_process_group("nccl", device_id=device)
    rank, world = dist.get_rank(), dist.get_world_size()
    cpu = dist.new_group(backend="gloo")
    if args.first_layers:
        firsts = [int(x) for x in args.first_layers.split(",")]
    else:
        firsts = [sum(SPLIT_813[:s]) for s in range(world)]
    assert len(firsts) == world
    first = firsts[rank]
    end = firsts[rank + 1] if rank + 1 < world else LAYERS
    closed_in = closed_bank_blocks(first, BLOCK, TOTAL_BLOCKS)
    closed_out = closed_bank_blocks(end, BLOCK, TOTAL_BLOCKS)
    taps_in = sum(t < first for t in TAPS)
    taps_out = sum(t < end for t in TAPS)
    group = SimpleNamespace(
        rank_in_group=rank, world_size=world, is_first_rank=rank == 0,
        is_last_rank=rank == world - 1, cpu_group=cpu, ranks=list(range(world)),
    )  # fmt: skip
    box = None
    if args.transport == "mailbox":
        box = PPMailbox(
            group=group, device=device, dtype=torch.bfloat16, hidden=HIDDEN,
            total_blocks=TOTAL_BLOCKS, block_size=BLOCK, first_layer=first, end_layer=end,
            aux_cols_in=taps_in * HIDDEN, aux_cols_out=0 if rank == world - 1 else taps_out * HIDDEN,
            max_rows=args.max_rows, slots=args.slots,
        )  # fmt: skip
    # the compute stand-in: a bf16 GEMM chain sized to ~compute_ms
    a = torch.randn(8192, 8192, dtype=torch.bfloat16, device=device)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(5):
        a @ a
    torch.cuda.synchronize()
    per_gemm_ms = (time.perf_counter() - t) / 5 * 1e3
    gemms = max(1, round(args.compute_ms / per_gemm_ms))

    gen = torch.Generator().manual_seed(1234)  # same row counts on every rank
    rows_of = [
        int(torch.randint(args.max_rows // 2, args.max_rows + 1, (1,), generator=gen))
        for _ in range(args.steps)
    ]
    bad, pending, step_ms = 0, [], []
    staged = (
        []
    )  # the CPU stays at most one microbatch ahead of the GPU, as the server's staging gate
    dist.barrier()
    t0 = time.perf_counter()
    for step, rows in enumerate(rows_of):
        ts = time.perf_counter()
        if len(staged) > 1:
            staged.pop(0).synchronize()
        if rank > 0:
            if box is not None and box.recv_enabled:
                got = box.recv(rows)
            else:
                got = nccl_recv(rows, taps_in, rank - 1, device)
            if args.check_every and step % args.check_every == 0:
                bad += check(got, step, rank - 1, rows, closed_in, taps_in, device)
        for _ in range(gemms):
            a @ a
        if rank < world - 1:
            out = payload(step, rank, rows, closed_out, taps_out, device)
            if box is not None and box.send_enabled:
                box.send(out, rows)
            else:
                for w in pending:
                    w.wait()
                pending = nccl_send(out, rank + 1)
        done = torch.cuda.Event()
        done.record()
        staged.append(done)
        step_ms.append((time.perf_counter() - ts) * 1e3)
    for w in pending:
        w.wait()
    torch.cuda.synchronize()
    dist.barrier()
    wall = time.perf_counter() - t0
    tot = torch.tensor([bad], device=device)
    dist.all_reduce(tot)
    if rank == 0:
        print(
            f"[pp_mailbox_selftest] transport={args.transport} stages={world} steps={args.steps} "
            f"compute~{gemms * per_gemm_ms:.0f} ms/step: {wall / args.steps * 1e3:.1f} ms per microbatch, "
            f"mismatched elements {int(tot)}"
        )
    print(
        f"  rank {rank} layers [{first}, {end}) in {closed_in} blocks + {taps_in} taps, out {closed_out} + {taps_out}"
        f" | host ms per step median {statistics.median(step_ms):.1f} p90 {sorted(step_ms)[int(0.9 * len(step_ms))]:.1f}"
        + (
            f" | recv {'mailbox' if box.recv_enabled else 'nccl'}, send {'mailbox' if box.send_enabled else 'nccl'}"
            if box
            else ""
        ),
        flush=True,
    )
    dist.destroy_process_group()
    if int(tot):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
