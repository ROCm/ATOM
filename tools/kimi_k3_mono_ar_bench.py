# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The gfx1250 K2 all-reduces at TP N: o_proj + AR (two-hop, or one-shot) and
the gathering AR, timed in CUDA graphs with the layers' weights and inputs
rotated, the one-shot output checked bit for bit against the two-hop's, and a
profile build's per-CTA stamps put on rank 0's clock (offsets from a
ping-pong calibration just before each stamped graph: the clocks drift apart
by microseconds a second) to split the o_proj AR into its hops. The gathering
AR likewise, today's against the one without the pre-push barrier.

    torchrun --nproc_per_node=4 tools/kimi_k3_mono_ar_bench.py
    env: ROWS (10,20,40), CHECK, BENCH, PROFILE, GATHER, GN, GREPS;
    DUMP=prefix: today's outputs to prefix_r<rank>.pt (against another tree's)
    ATOM_MONO_POLL_SLEEP_UNITS: the polls' sleep (process-wide)
A poll loop needs a side effect (the pause): a buffer load alone is hoisted.
"""

import inspect
import json
import os

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
import torch.distributed as dist
from aiter import dtypes
from aiter.ops.flydsl.kernels import buffer_ops as bo
from aiter.ops.shuffle import shuffle_weight
from flydsl.expr import range_constexpr
from flydsl.expr.typing import Int32, Int64, T

from atom.models.kimi_k3.mono.config import HIDDEN, LATENT, TOPK
from atom.models.kimi_k3.mono.kernels.execution_plan import Gfx1250KimiBuildPlan
from atom.models.kimi_k3.mono.kernels.stages import k2_allreduce_gfx1250 as ar
from atom.models.kimi_k3.mono.kernels.stages import k2_oproj_ar_gfx1250 as oar
from atom.models.kimi_k3.mono.kernels.stages.moe_tdm_gfx1250 import _realtime
from atom.mono.device.ops import load_ptr64, rsrc, spin_pause, st_dev, traced
from atom.mono.plan.architecture import GFX1250
from atom.quantization.quark.utils import quantize_weight_to_fp8_128x128_blockscale

KEEP = []
ROWS = [int(r) for r in os.environ.get("ROWS", "10,20,40").split(",")]
CHECK = os.environ.get("CHECK", "1") == "1"
BENCH = os.environ.get("BENCH", "1") == "1"
PROFILE = os.environ.get("PROFILE", "1") == "1"
GATHER = os.environ.get("GATHER", "1") == "1"
DUMP = os.environ.get("DUMP", "")
COPIES, SETS = 4, 8
ONESHOT_API = "oneshot" in inspect.signature(oar.build_oproj_allreduce).parameters
GATHER_API = "profile" in inspect.signature(ar.build_gather_allreduce).parameters
CAL_ITERS = 16
PROFILE_LAUNCHES = 16


POLL_LIMIT = 1 << 22


def poll(mine, word):
    line = bo.buffer_load(
        mine, word, vec_width=2, dtype=T.i32, cache_modifier=ar.SCOPE_SYS | ar.VOLATILE
    )
    return fx.Int32(fx.Vector(line)[0])


@traced
def wait_tag(mine, word, tag):
    got = poll(mine, word)
    tries = fx.Int32(0)
    while (got != tag) & (tries < POLL_LIMIT):
        spin_pause()
        got = poll(mine, word)
        tries = tries + 1
    return tries


def build_calibration():
    """Rank 0 and rank ``peer`` trade a tagged word ``CAL_ITERS`` times,
    stamping send and receive: (t0, t3) on rank 0, (t1, t2) on the peer."""

    @flyc.kernel(name="k3_clock_calibration", known_block_size=[64, 1, 1])
    def kernel(
        peers: Int64, rank: Int32, peer: Int32, tag0: Int32, word: Int32, stamps: Int64
    ):
        tid = fx.Int32(fx.thread_idx.x)
        mine = rsrc(load_ptr64(peers, rank))
        other = load_ptr64(peers, peer)
        if tid == 0:
            for i in range_constexpr(CAL_ITERS):
                tag = tag0 + i
                if rank == 0:
                    st_dev(fx.Int32(_realtime()), stamps, 4 * i)
                    bo.buffer_store(tag, rsrc(other), word, cache_modifier=ar.SCOPE_SYS)
                wait_tag(mine, word, tag)
                now = fx.Int32(_realtime())
                if rank == 0:
                    st_dev(now, stamps, 4 * i + 3)
                if rank != 0:
                    st_dev(now, stamps, 4 * i + 1)
                    st_dev(fx.Int32(_realtime()), stamps, 4 * i + 2)
                    bo.buffer_store(tag, rsrc(other), word, cache_modifier=ar.SCOPE_SYS)

    @flyc.jit
    def launch(
        peers: Int64,
        rank: Int32,
        peer: Int32,
        tag0: Int32,
        word: Int32,
        stamps: Int64,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        kernel(peers, rank, peer, tag0, word, stamps).launch(
            grid=(1,), block=(64,), stream=stream
        )

    return launch


def wrap(d):
    """i32 tick differences (the 100 MHz clock wraps every 21 s)."""
    return ((d + 2**31) % 2**32) - 2**31


def main():
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)
    gen = torch.Generator(device=dev).manual_seed(11 + rank)
    proj = 96 // world * 128
    import mori.allocator  # noqa: F401
    import torch.distributed._symmetric_memory as symm_mem

    symm_mem.set_backend("MORI")
    peer_bytes = max(
        oar.oproj_peer_bytes(world, max(ROWS), *([True] if ONESHOT_API else [])),
        ar.allreduce_peer_bytes(world, max(ROWS)),
    )
    buf = symm_mem.empty(peer_bytes + 4096, dtype=torch.uint8, device=dev)
    buf.zero_()
    torch.cuda.synchronize()
    ptrs = list(symm_mem.rendezvous(buf, dist.group.WORLD).buffer_ptrs)
    ptrs[rank] = buf.data_ptr()
    peers = torch.tensor(ptrs, dtype=torch.int64, device=dev)
    epochs = torch.zeros(ar.MAX_CTAS, dtype=torch.int32, device=dev)
    KEEP.extend([buf, peers, epochs])

    cal = build_calibration()
    cal_stamps = torch.zeros(4 * CAL_ITERS, dtype=torch.int32, device=dev)
    KEEP.extend([cal, cal_stamps])
    cal_tag = [1]

    def calibrate(quiet=False):
        """Rank r's clock minus rank 0's, in ticks, from the fastest trade
        (the clocks drift apart by microseconds over seconds: calibrate right
        before the launches stamped)."""
        stamps = cal_stamps
        offsets = [0] * world
        tag = cal_tag[0]
        for peer in range(1, world):
            stamps.zero_()
            torch.cuda.synchronize()
            dist.barrier()
            if rank in (0, peer):
                cal(
                    peers.data_ptr(),
                    rank,
                    peer if rank == 0 else 0,
                    tag,
                    peer_bytes // 4,
                    stamps.data_ptr(),
                )
            torch.cuda.synchronize()
            tag += CAL_ITERS + 1
            cal_tag[0] = tag
            rows = [torch.zeros(4 * CAL_ITERS, dtype=torch.int64) for _ in range(world)]
            dist.all_gather(rows, stamps.cpu().long())
            a, b = rows[0].view(-1, 4), rows[peer].view(-1, 4)
            t0, t3, t1, t2 = a[:, 0], a[:, 3], b[:, 1], b[:, 2]
            rtt = wrap(t3 - t0) - wrap(t2 - t1)
            best = int(rtt[4:].argmin()) + 4
            offsets[peer] = int((wrap(t1 - t0)[best] - wrap(t3 - t2)[best]) // 2)
            if rank == 0 and not quiet:
                print(
                    json.dumps(
                        {
                            "calibration": peer,
                            "offset_ticks": offsets[peer],
                            "rtt_us_min": float(rtt[best]) / 100,
                        }
                    ),
                    flush=True,
                )
        return offsets

    calibrate()

    def bf16(*shape, scale=1.0):
        return (torch.randn(shape, generator=gen, device=dev) * scale).to(
            torch.bfloat16
        )

    def fp8_weight(rows, cols):
        w = (torch.randn(rows, cols, generator=gen, device=dev) * 0.02).to(
            torch.bfloat16
        )
        q, s = quantize_weight_to_fp8_128x128_blockscale(
            w, dtypes.fp8, scale_dtype=dtypes.fp8_e8m0
        )
        return shuffle_weight(q, layout=(16, 16)).contiguous(), s.contiguous()

    weights = [fp8_weight(HIDDEN, proj) for _ in range(COPIES)]
    KEEP.append(weights)

    def graph_us(fn):
        n, reps = int(os.environ.get("GN", "20")), int(os.environ.get("GREPS", "7"))
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            fn(side, 0)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=side):
            for k in range(n):
                fn(torch.cuda.current_stream(), k + 1)
        KEEP.append((graph, side))
        dist.barrier()
        graph.replay()
        torch.cuda.synchronize()
        dist.barrier()
        samples = []
        for _ in range(reps):
            start, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) * 1000 / n)
            dist.barrier()
        return round(sorted(samples)[len(samples) // 2], 2)

    def profile(go, words):
        """``go`` captured ``PROFILE_LAUNCHES`` times in a graph, each launch
        its own stamps; the later half of three replays, every rank's stamps
        with the clock offsets calibrated just before."""
        stamps = torch.zeros(PROFILE_LAUNCHES, words, dtype=torch.int32, device=dev)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            go(side, 0, stamps[0].data_ptr())
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=side):
            for k in range(PROFILE_LAUNCHES):
                go(torch.cuda.current_stream(), k, stamps[k].data_ptr())
        KEEP.append((graph, side, stamps))
        samples = []
        for _ in range(3):
            offsets = calibrate(quiet=True)
            dist.barrier()
            graph.replay()
            torch.cuda.synchronize()
            gathered = [
                torch.zeros(stamps.numel(), dtype=torch.int64) for _ in range(world)
            ]
            dist.all_gather(gathered, stamps.cpu().long().flatten())
            run = torch.stack(gathered).view(world, PROFILE_LAUNCHES, words)
            samples.extend(
                (run[:, k], offsets)
                for k in range(PROFILE_LAUNCHES // 2, PROFILE_LAUNCHES)
            )
        return samples

    dumps = {}
    for rows in ROWS:
        plan = Gfx1250KimiBuildPlan(
            threads=256,
            blocks=256,
            lds_bytes=GFX1250.max_workgroup_lds_bytes,
            tokens=rows,
            nblocks=4,
        )
        if ONESHOT_API:
            kernels = {
                name: oar.build_oproj_allreduce(plan, world, proj, oneshot=os_)
                for name, os_ in (("two_hop", False), ("one_shot", True))
            }
        else:
            kernels = {"two_hop": oar.build_oproj_allreduce(plan, world, proj)}
        KEEP.append(kernels)
        cores = [bf16(rows, proj) for _ in range(SETS)]
        outs = {
            name: [
                torch.zeros(rows, HIDDEN, dtype=torch.bfloat16, device=dev)
                for _ in range(SETS)
            ]
            for name in kernels
        }
        KEEP.extend([cores, outs])

        def runner(name, kern=None, kernels=kernels, cores=cores, outs=outs):
            return oproj_launches(
                kern or kernels[name], cores, weights, outs[name], peers, epochs, rank
            )

        entry = {"rows": rows}
        if CHECK:
            stream = torch.cuda.current_stream()
            for name in kernels:
                for k in range(SETS):
                    runner(name)(stream, k)
            torch.cuda.synchronize()
            dist.barrier()
            if ONESHOT_API:
                entry["one_shot_bitwise"] = all(
                    torch.equal(outs["two_hop"][k], outs["one_shot"][k])
                    for k in range(SETS)
                )
            dumps[f"oproj{rows}"] = [o.cpu() for o in outs["two_hop"]]
        if BENCH:
            for rep in range(2):
                for name in kernels:
                    entry[f"{name}#{rep}"] = graph_us(runner(name))
        if PROFILE and ONESHOT_API:
            for name, os_ in (("two_hop", False), ("one_shot", True)):
                kern = oar.build_oproj_allreduce(
                    plan, world, proj, oneshot=os_, profile=True
                )
                KEEP.append(kern)
                go = runner(name, kern)
                samples = profile(go, oar.CTAS * oar.STAMPS)
                if rank == 0:
                    entry[f"{name}_profile_us"] = breakdown(samples, world, rows, os_)
        if GATHER:
            entry.update(
                gather_bench(
                    rows, world, rank, dev, gen, peers, epochs, graph_us, profile, dumps
                )
            )
        if rank == 0:
            print(json.dumps(entry), flush=True)
    if DUMP:
        torch.save(dumps, f"{DUMP}_r{rank}.pt")
    torch.cuda.synchronize()
    dist.barrier()


def oproj_launches(kern, cores, weights, outs, peers, epochs, rank):
    """Launch ``k`` of ``kern`` on input set ``k`` and weight copy ``k``."""

    def go(stream, k, stamps=None):
        w, s = weights[k % COPIES]
        args = (
            cores[k % SETS].data_ptr(),
            w.data_ptr(),
            s.data_ptr(),
            outs[k % SETS].data_ptr(),
            peers.data_ptr(),
            epochs.data_ptr(),
            rank,
        )
        kern.launcher(*args, *([] if stamps is None else [stamps]), stream=stream)

    return go


def breakdown(samples, world, rows, oneshot):
    """Medians (and maxima) over launches of each phase, on rank 0's clock
    (us): every rank's launch start to its partials pushed, the last push of
    a column's partials to the poll seeing them, and so on."""
    owner_ctas = oar.CTAS // world
    parts = {}

    def add(key, ticks):
        parts.setdefault(key, []).append(float(ticks) / 100)

    for s, offsets in samples:
        s = s.view(world, oar.CTAS, oar.STAMPS).clone()
        for r in range(world):
            s[r] = s[r] - offsets[r]
        rel = wrap(s - s[0, 0, oar.STAMP_START])
        start = rel[:, :, oar.STAMP_START]
        pushed = rel[:, :, oar.STAMP_PUSHED]
        end = rel[:, :, oar.STAMP_END]
        add("start_skew_ranks", start.min(1).values.max() - start.min(1).values.min())
        add("start_to_staged", (rel[:, :, oar.STAMP_STAGED] - start).float().median())
        add("staged_to_pushed", (pushed - rel[:, :, oar.STAMP_STAGED]).float().median())
        last_push_cta = pushed.max(0).values
        last_push_all = pushed.max()
        add("first_start_to_last_push", last_push_all - start.min())
        summed = rel[:, :, oar.STAMP_SUMMED]
        if oneshot:
            add("hop_last_push_to_seen", (summed - last_push_cta).float().median())
            add("hop_last_push_to_seen_max", (summed - last_push_cta).max())
            add("seen_to_end", (end - summed).float().median())
        elif rows <= oar.SPREAD_ROWS:
            ctas = torch.arange(oar.CTAS)
            owner = ctas // owner_ctas
            o_summed = summed[owner, ctas]
            o_sums_pushed = rel[:, :, oar.STAMP_SUMS_PUSHED][owner, ctas]
            add("hop1_last_push_to_seen", (o_summed - last_push_cta).float().median())
            add("sum_and_push", (o_sums_pushed - o_summed).float().median())
            landed = rel[:, :, oar.STAMP_LANDED].clone()
            landed[owner, ctas] = -(2**40)
            add(
                "hop2_sums_pushed_to_landed",
                (landed.max(0).values - o_sums_pushed).float().median(),
            )
            add("hop2_max", (landed.max(0).values - o_sums_pushed).max())
            add("landed_to_end", (end - rel[:, :, oar.STAMP_LANDED]).float().median())
        else:
            add("hop1_last_push_any_to_seen", (summed - last_push_all).float().median())
            sums_pushed = rel[:, :, oar.STAMP_SUMS_PUSHED]
            add("sum_and_push", (sums_pushed - summed).float().median())
            landed = rel[:, :, oar.STAMP_LANDED]
            add(
                "hop2_last_sums_any_to_landed",
                (landed - sums_pushed.max()).float().median(),
            )
            add("landed_to_end", (end - landed).float().median())
        add("end_skew_ranks", end.max(1).values.max() - end.max(1).values.min())
        add("launch_span", end.max() - start.min())
    out = {}
    for k, v in parts.items():
        v = sorted(v)
        out[k] = [round(v[len(v) // 2], 2), round(v[-1], 2)]
    return out


def gather_bench(rows, world, rank, dev, gen, peers, epochs, graph_us, profile, dumps):
    """Today's gathering AR and its variants (no pre-push barrier), each from
    the same inputs: outputs bit for bit, graph times, a profile breakdown."""
    variants = {"gather": {}}
    if GATHER_API:
        variants["gather_nb"] = {"prepush_barrier": False}
    kerns = {
        name: ar.build_gather_allreduce(world, rows, **kw)
        for name, kw in variants.items()
    }
    width = LATENT + HIDDEN
    sets = [
        (
            torch.randn(rows, width, generator=gen, device=dev).to(torch.bfloat16),
            torch.randn(rows * TOPK, LATENT, generator=gen, device=dev).to(
                torch.bfloat16
            ),
        )
        for _ in range(SETS)
    ]
    route_rows = (
        torch.randperm(rows * TOPK, generator=gen, device=dev)
        .to(torch.int32)
        .view(rows, TOPK)
    )
    route_weights = torch.rand(rows, TOPK, generator=gen, device=dev).float()
    copies = {name: [d.clone() for d, _ in sets] for name in kerns}
    KEEP.extend([kerns, sets, route_rows, route_weights, copies])

    def runner(kern, datas):
        def go(stream, k, stamps=None):
            args = (
                datas[k % SETS].data_ptr(),
                sets[k % SETS][1].data_ptr(),
                route_rows.data_ptr(),
                route_weights.data_ptr(),
                peers.data_ptr(),
                epochs.data_ptr(),
                rank,
            )
            kern.launcher(*args, *([] if stamps is None else [stamps]), stream=stream)

        return go

    entry = {"gather_kernel": kerns["gather"].name}
    if CHECK:
        stream = torch.cuda.current_stream()
        for name, kern in kerns.items():
            for k in range(SETS):
                runner(kern, copies[name])(stream, k)
        torch.cuda.synchronize()
        dist.barrier()
        dumps[f"gather{rows}"] = [d.cpu() for d in copies["gather"]]
        for name in kerns:
            if name != "gather":
                entry[f"{name}_bitwise"] = all(
                    torch.equal(a, b)
                    for a, b in zip(copies["gather"], copies[name], strict=True)
                )
    if BENCH:
        for rep in range(2):
            for name, kern in kerns.items():
                entry[f"{name}#{rep}"] = graph_us(runner(kern, copies[name]))
    if PROFILE and GATHER_API:
        for name, kw in variants.items():
            kern = ar.build_gather_allreduce(world, rows, profile=True, **kw)
            KEEP.append(kern)
            samples = profile(runner(kern, copies[name]), kern.ctas * ar.GATHER_STAMPS)
            if rank == 0:
                entry[f"{name}_profile_us"] = gather_breakdown(
                    samples, world, kern.ctas
                )
    return entry


def gather_breakdown(samples, world, ctas):
    """Medians over launches of the gathering AR's phases on rank 0's clock
    (us), thread 0's line of each CTA: the hop is the last rank's push of the
    line to this rank's poll seeing every copy."""
    parts = {}

    def add(key, ticks):
        parts.setdefault(key, []).append(float(ticks) / 100)

    for s, offsets in samples:
        s = s.view(world, ctas, ar.GATHER_STAMPS).clone()
        for r in range(world):
            s[r] = s[r] - offsets[r]
        rel = wrap(s - s[0, 0, ar.GATHER_START])
        start = rel[:, :, ar.GATHER_START]
        routed = rel[:, :, ar.GATHER_ROUTED]
        gathered = rel[:, :, ar.GATHER_GATHERED]
        pushed = rel[:, :, ar.GATHER_PUSHED]
        seen = rel[:, :, ar.GATHER_SEEN]
        end = rel[:, :, ar.GATHER_END]
        add("start_skew_ranks", start.min(1).values.max() - start.min(1).values.min())
        add(
            "start_skew_ctas",
            (start.max(1).values - start.min(1).values).float().median(),
        )
        add("start_to_routed", (routed - start).float().median())
        add("routed_to_gathered", (gathered - routed).float().median())
        add("gathered_to_pushed", (pushed - gathered).float().median())
        add("hop_last_push_to_seen", (seen - pushed.max(0).values).float().median())
        add("hop_max", (seen - pushed.max(0).values).max())
        drained = rel[:, :, ar.GATHER_DRAINED]
        add("seen_to_storecnt_drained", (drained - seen).float().median())
        add("seen_to_storecnt_drained_max", (drained - seen).max())
        add("own_push_to_storecnt_drained", (drained - pushed).float().median())
        add("seen_to_end", (end - seen).float().median())
        add("end_skew_ranks", end.max(1).values.max() - end.max(1).values.min())
        add("launch_span", end.max() - start.min())
    out = {}
    for k, v in parts.items():
        v = sorted(v)
        out[k] = [round(v[len(v) // 2], 2), round(v[-1], 2)]
    return out


if __name__ == "__main__":
    main()
