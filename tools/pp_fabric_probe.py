# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""First contact with a PP mailbox hop between hosts (atom/distributed/pp_fabric.py),
before any server run. Two ranks, one GPU on each of the two trays:

    # tray A                                  # tray B
    torchrun --nnodes 2 --nproc_per_node 1 --node_rank 0 \\
        --master_addr <A> --master_port 29551 tools/pp_fabric_probe.py
    torchrun --nnodes 2 --nproc_per_node 1 --node_rank 1 \\
        --master_addr <A> --master_port 29551 tools/pp_fabric_probe.py

SAFETY. Cross-tray mori symm-mem and MoRIIO fabric writes have reset e15 trays
(LLM-Booster #813 notes: 11-23 on 05+07, 11-27 on 05 with trays from different
boot batches). Run this only on idle trays booted together, with the owners'
agreement, and before any server: it touches the fabric in the smallest steps
first and stops at the first failure.

Steps, rank 0 the sender (pp3's role), rank 1 the receiver (pp4's):

1. probe    ``probe_fabric`` on both ranks: mori imports, the device exports
            fabric handles, the MORI backend allocates, the kernels build.
2. map      a small ``FabricHop`` on both ends: symm_mem rendezvous + agreement.
3. counter  rank 0 raises rank 1's flag word through the fabric, rank 1 waits on
            it with the poll kernel; then rank 1 returns a credit the same way.
4. stream   rank 1 waits on its flag with ``hipStreamWaitValue64`` while rank 0
            raises it; reports whether the command processor sees a store that
            arrives over the fabric (``ATOM_PP_MAILBOX_FABRIC_WAIT=stream``). On a
            timeout rank 1 rescues its own stream with a local write.
5. push     rank 0's push kernel writes a patterned payload (Kimi-K3 widths)
            into rank 1's ring; rank 1 waits for the flag and checks every element.
6. pull     rank 0 fills its own ring with the copy engine and raises the flag;
            rank 1 reads it across the fabric with the copy-in kernel and checks it.
7. bw       push: rank 0's kernel at each ``--ctas`` into rank 1's ring; pull:
            rank 1's copy-in from rank 0's ring. ``--rows`` rows of
            ``--row-words`` H-wide words (9 = hidden + 5 closed bank blocks + 3 DSpark
            taps, the 07->05 hop of #813's split).
"""

import argparse
import ctypes
import os
import time

import torch
import torch.distributed as dist

from atom.distributed import pp_fabric as pf
from atom.distributed import pp_mailbox as mb

H = 7168


def say(rank, msg):
    print(f"[pp_fabric_probe rank {rank}] {msg}", flush=True)


def agree(ok: bool) -> bool:
    t = torch.tensor([int(ok)], dtype=torch.int32)
    dist.all_reduce(t, op=dist.ReduceOp.MIN)
    return bool(t.item())


def pattern(rows, cols, salt, device):
    r = torch.arange(rows, device=device).unsqueeze(1)
    c = torch.arange(cols, device=device).unsqueeze(0)
    return ((r * 7 + c * 3 + salt * 13) % 61 - 30).to(torch.bfloat16)


def layout(rows, closed, taps, slots=2):
    return mb.HopLayout(rows, H, 8, closed, taps * H, 2, slots)


def hop(role, lay, device, options):
    return pf.FabricHop(
        role=role, layout=lay, group=dist.group.WORLD, device=device, options=options
    )


def payload(rows, lay, salt, device):
    hidden = pattern(rows, H, salt, device)
    bank = torch.zeros(rows, 8, H, dtype=torch.bfloat16, device=device)
    for b in range(lay.closed_blocks):
        bank[:, b] = pattern(rows, H, salt + 1 + b, device)
    aux = pattern(rows, lay.aux_cols, salt + 99, device) if lay.aux_cols else None
    return hidden, bank, aux


def push(h, stream, lay, slot, rows, hidden, bank, aux, value, ctas):
    off, base, v = lay.offsets(), h.slot_base(slot), 8  # bf16: 8 per 16 B
    h.kernels.push(
        stream,
        hidden=hidden.data_ptr(), hidden_v4=H // v, bank=bank.data_ptr(), bank_row_v4=8 * H // v,
        closed_v4=lay.closed_blocks * H // v, aux=0 if aux is None else aux.data_ptr(),
        aux_v4=lay.aux_cols // v, rows=rows, dst_hidden=base + off["hidden"], dst_bank=base + off["bank"],
        dst_aux=base + off["aux"], rows_word=h.word(pf.ROWS_WORD + slot, peer=True),
        flag_word=h.word(pf.FLAG_WORD, peer=True), flag_value=value, arrive=h.word(pf.ARRIVE_WORD), ctas=ctas,
    )  # fmt: skip


def read_back(base, lay, rows, device):
    """The slot at ``base`` (local or across the fabric) through the mailbox's copy-in."""
    w = 4  # bf16 per int64
    hidden = torch.empty(rows, H, dtype=torch.bfloat16, device=device)
    bank = torch.zeros(rows, 8, H, dtype=torch.bfloat16, device=device)
    aux = (
        torch.empty(rows, lay.aux_cols, dtype=torch.bfloat16, device=device)
        if lay.aux_cols
        else None
    )
    off = lay.offsets()
    mb.copy_in(
        base + off["hidden"], hidden.view(torch.int64), rows, H // w, H // w, H // w
    )
    cols = lay.closed_blocks * H // w
    mb.copy_in(
        base + off["bank"],
        bank.view(rows, -1).view(torch.int64),
        rows,
        cols,
        cols,
        8 * H // w,
    )
    if aux is not None:
        mb.copy_in(
            base + off["aux"],
            aux.view(torch.int64),
            rows,
            lay.aux_cols // w,
            lay.aux_cols // w,
            lay.aux_cols // w,
        )
    return hidden, bank, aux


def mismatches(got, want, lay):
    bad = int((got[0] != want[0]).sum())
    bad += int(
        (got[1][:, : lay.closed_blocks] != want[1][:, : lay.closed_blocks]).sum()
    )
    if want[2] is not None:
        bad += int((got[2] != want[2]).sum())
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--small-rows", type=int, default=256)
    ap.add_argument("--rows", type=int, default=16384)
    ap.add_argument(
        "--row-words",
        type=int,
        default=9,
        help="H-wide words per row in the bandwidth step",
    )
    ap.add_argument("--ctas", type=str, default="8,16,32,64,128")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--stream-timeout", type=float, default=10.0)
    ap.add_argument(
        "--steps", type=str, default="probe,map,counter,stream,push,pull,bw"
    )
    args = ap.parse_args()
    steps = set(args.steps.split(","))

    local = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world == 2, "two ranks: one GPU on each tray"
    role = "send" if rank == 0 else "recv"
    stream = torch.cuda.Stream(device=device)
    results = {}

    # 1. probe
    why = pf.probe_fabric(device)
    say(rank, f"probe: {'ok' if why is None else why}")
    if not agree(why is None):
        raise SystemExit("probe failed on a rank; see above")
    results["probe"] = "ok"
    if "map" not in steps:
        return

    # 2. map (small)
    small = layout(args.small_rows, 5, 3)
    opts = pf.FabricOptions(mode="push", wait="kernel", ctas=8)
    t = time.perf_counter()
    h = hop(role, small, device, opts)
    say(
        rank,
        f"map: ok={h.ok} local={h.local:#x} peer={h.peer:#x} ({(time.perf_counter() - t) * 1e3:.0f} ms)",
    )
    if not h.ok:
        raise SystemExit("the two ranks could not map each other")
    results["map"] = "ok"

    # 3. counters over the fabric, poll kernel
    if "counter" in steps:
        with torch.cuda.stream(stream):
            for k in range(1, 4):
                if rank == 0:
                    h.kernels.signal(
                        stream, None, 0, h.word(pf.FLAG_WORD, peer=True), k
                    )
                    h.wait_ge(stream, h.word(pf.CREDIT_WORD), k)
                else:
                    h.wait_ge(stream, h.word(pf.FLAG_WORD), k)
                    h.kernels.signal(
                        stream, None, 0, h.word(pf.CREDIT_WORD, peer=True), k
                    )
        stream.synchronize()
        say(rank, "counter: 3 round trips ok")
        results["counter"] = "ok"
        dist.barrier()

    # 4. hipStreamWaitValue64 on a word written over the fabric
    if "stream" in steps:
        target = 100
        verdict = "n/a"
        if rank == 1:
            hip = pf._hip_modules()
            ev = torch.cuda.Event()
            with torch.cuda.stream(stream):
                hip.hipStreamWaitValue64(
                    ctypes.c_void_p(stream.cuda_stream),
                    ctypes.c_void_p(h.word(pf.FLAG_WORD)),
                    target,
                    0,
                    (1 << 64) - 1,
                )
                ev.record(stream)
        dist.barrier()
        if rank == 0:
            time.sleep(1.0)
            with torch.cuda.stream(stream):
                h.kernels.signal(
                    stream, None, 0, h.word(pf.FLAG_WORD, peer=True), target
                )
            stream.synchronize()
        else:
            t0 = time.perf_counter()
            while not ev.query() and time.perf_counter() - t0 < args.stream_timeout:
                time.sleep(0.01)
            if ev.query():
                verdict = f"ok, released {time.perf_counter() - t0:.2f} s after the barrier (rank 0 writes at 1 s)"
            else:
                verdict = "TIMEOUT: the stream wait does not see fabric stores; use ATOM_PP_MAILBOX_FABRIC_WAIT=kernel"
                side = torch.cuda.Stream(
                    device=device
                )  # a local store unblocks the stuck stream
                h.kernels.signal(side, None, 0, h.word(pf.FLAG_WORD), target)
                side.synchronize()
                ev.synchronize()
            say(rank, f"stream wait: {verdict}")
        results["stream"] = verdict
        dist.barrier()

    # 5. push correctness (small)
    if "push" in steps:
        bad = 0
        with torch.cuda.stream(stream):
            for k in range(1, 5):
                slot, rows = (k - 1) % small.slots, args.small_rows - 7 * k
                want = payload(rows, small, k, device)
                # counters only grow: this step's flags run 101.., its credits 11..
                if rank == 0:
                    if k > small.slots:
                        h.wait_ge(stream, h.word(pf.CREDIT_WORD), 10 + k - small.slots)
                    push(h, stream, small, slot, rows, *want, value=100 + k, ctas=8)
                else:
                    h.wait_ge(stream, h.word(pf.FLAG_WORD), 100 + k)
                    got = read_back(h.slot_base(slot), small, rows, device)
                    stream.synchronize()
                    bad += mismatches(got, want, small)
                    h.kernels.signal(
                        stream, None, 0, h.word(pf.CREDIT_WORD, peer=True), 10 + k
                    )
        stream.synchronize()
        if rank == 1:
            say(rank, f"push: 4 microbatches, mismatched elements {bad}")
        results["push"] = "ok" if agree(bad == 0) else "MISMATCH"
        dist.barrier()

    # 6. pull correctness (small): a fresh hop in pull mode
    if "pull" in steps:
        hp = hop(
            role, small, device, pf.FabricOptions(mode="pull", wait="kernel", ctas=8)
        )
        bad = 0
        with torch.cuda.stream(stream):
            for k in range(1, 5):
                slot, rows = (k - 1) % small.slots, args.small_rows - 5 * k
                want = payload(rows, small, 50 + k, device)
                if rank == 0:
                    if k > small.slots:
                        hp.wait_ge(stream, hp.word(pf.CREDIT_WORD), k - small.slots)
                    box = object.__new__(mb.PPMailbox)
                    box.out_layout, box.hidden = small, H
                    box._sdma_payload(
                        ctypes.c_void_p(stream.cuda_stream),
                        hp.slot_base(slot),
                        *want,
                        rows,
                    )
                    hp.kernels.signal(
                        stream,
                        hp.word(pf.ROWS_WORD + slot, peer=True),
                        rows,
                        hp.word(pf.FLAG_WORD, peer=True),
                        k,
                    )
                else:
                    hp.wait_ge(stream, hp.word(pf.FLAG_WORD), k)
                    got = read_back(hp.slot_base(slot), small, rows, device)
                    stream.synchronize()
                    bad += mismatches(got, want, small)
                    hp.kernels.signal(
                        stream, None, 0, hp.word(pf.CREDIT_WORD, peer=True), k
                    )
        stream.synchronize()
        if rank == 1:
            say(rank, f"pull: 4 microbatches, mismatched elements {bad}")
        results["pull"] = "ok" if agree(bad == 0) else "MISMATCH"
        dist.barrier()

    # 7. bandwidth at the hop's real size
    if "bw" in steps:
        closed = max(0, min(8, args.row_words - 4))
        big = layout(args.rows, closed, args.row_words - 1 - closed, slots=1)
        hb = hop(role, big, device, opts)
        nbytes = args.rows * big.row_elems * 2
        src = payload(args.rows, big, 7, device)
        lines = []
        with torch.cuda.stream(stream):
            value = 0
            for ctas in [int(c) for c in args.ctas.split(",")]:
                dist.barrier()
                if rank == 0:
                    push(
                        hb, stream, big, 0, args.rows, *src, value=value + 1, ctas=ctas
                    )  # warm-up
                    value += 1
                    t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                        enable_timing=True
                    )
                    t0.record(stream)
                    for _ in range(args.iters):
                        value += 1
                        push(
                            hb, stream, big, 0, args.rows, *src, value=value, ctas=ctas
                        )
                    t1.record(stream)
                    t1.synchronize()
                    ms = t0.elapsed_time(t1) / args.iters
                    lines.append(
                        f"push ctas={ctas:4d}: {ms:7.2f} ms, {nbytes / ms / 1e6:6.0f} GB/s"
                    )
                else:
                    value += 1 + args.iters
                    hb.wait_ge(stream, hb.word(pf.FLAG_WORD), value)
                    stream.synchronize()
            dist.barrier()
            if (
                rank == 1
            ):  # pull: read rank 0's ring (filled above only in its own copy; content irrelevant)
                read_back(
                    hb.word(0, peer=True) + pf.RING_OFFSET, big, args.rows, device
                )
                t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                    enable_timing=True
                )
                t0.record(stream)
                for _ in range(args.iters):
                    read_back(
                        hb.word(0, peer=True) + pf.RING_OFFSET, big, args.rows, device
                    )
                t1.record(stream)
                t1.synchronize()
                ms = t0.elapsed_time(t1) / args.iters
                lines.append(
                    f"pull (copy-in across the fabric): {ms:7.2f} ms, {nbytes / ms / 1e6:6.0f} GB/s"
                )
        for line in lines:
            say(rank, f"bw {nbytes / 1e9:.2f} GB: {line}")
        results["bw"] = "; ".join(lines)
        dist.barrier()

    gathered = [None, None]
    dist.all_gather_object(gathered, results)
    if rank == 0:
        print("[pp_fabric_probe] summary:")
        for r, res in enumerate(gathered):
            for k, v in res.items():
                print(f"  rank {r} {k:8s} {v}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
