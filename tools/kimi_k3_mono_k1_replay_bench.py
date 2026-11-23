"""K1 alone, ReplaySSM off against on (``kimi_k3_mono_k1_replay_check``'s
setup), each a CUDA graph of REPS launches timed over RUNS ABBA rounds, then
(SUSTAIN seconds) each arm replayed back to back, off / on / on / off, at
cursor SUSTAIN_H.

Off: every token's state stored (DSpark slots). On: one checkpoint a request,
each step's records written; ``H`` committed records (non-flush: h + 2 window
<= cache_len, else a flush that folds them into the checkpoint first). The
step's K1 never moves the cursor, so every launch replays the same h.

CHUNK_AB=1 times the chunked head recurrence (ATOM_KIMI_MONO_K1_CHUNK_RECUR)
against the sequential one instead, each arm off and on at every cursor: ABBA
rounds seq / chunk / chunk / seq, then the sustained arms the same way.

SPLIT_AB=a,b times ReplaySSM on at head split a against b (the serving
recurrence both), each cursor: ABBA rounds a / b / b / a, then sustained.

Run on one gfx1250 GPU: ROWS=40 H=0,5,6,8,11 python tools/kimi_k3_mono_k1_replay_bench.py
"""

import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kimi_k3_mono_k1_replay_check as rc

from atom.models.kimi_k3.mono.kernels.attention.kda_builder_gfx1250 import (
    REPLAY_TABLE,
)
from atom.models.kimi_k3.mono.kernels.execution_plan import Gfx1250KimiBuildPlan

ROWS = int(os.environ.get("ROWS", "40"))
HS = [int(h) for h in os.environ.get("H", "0,5,6,8,11").split(",")]
REPS = int(os.environ.get("REPS", "20"))
RUNS = int(os.environ.get("RUNS", "8"))
# seconds each arm replays back to back after the bursts (0: none), at cursor
SUSTAIN = float(os.environ.get("SUSTAIN", "0"))
SUSTAIN_H = int(os.environ.get("SUSTAIN_H", "4"))
CHUNK_AB = os.environ.get("CHUNK_AB", "0") == "1"
SPLIT_AB = [s for s in os.environ.get("SPLIT_AB", "").split(",") if s]
dev, gen = rc.dev, rc.gen
QLEN, CAP, WINDOW = rc.QLEN, rc.CAP, rc.WINDOW


def graph(launch):
    launch()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(REPS):
            launch()
    return g


def time_us(g):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    g.replay()
    start.record()
    g.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000 / REPS


def main():
    req = ROWS // QLEN
    plan = Gfx1250KimiBuildPlan(
        threads=256, blocks=256, lds_bytes=320 * 1024, tokens=ROWS, nblocks=1
    )
    if SPLIT_AB:
        kernels = {}
        for split in SPLIT_AB:
            os.environ["ATOM_KIMI_MONO_K1_HEAD_SPLIT"] = split
            kernels[f"on_s{split}"] = plan.make_k1_local_runner(
                rc.spec(ROWS, True)
            ).prepare(dev)
    elif CHUNK_AB:
        kernels = {}
        for arm, chunk in (("seq", False), ("chunk", True)):
            on_k1, off_k1 = rc.builds(plan, ROWS, chunk)
            kernels[f"off_{arm}"], kernels[f"on_{arm}"] = off_k1, on_k1
    else:
        kernels = {
            "off": plan.make_k1_local_runner(rc.spec(ROWS, False)).prepare(dev),
            "on": plan.make_k1_local_runner(rc.spec(ROWS, True)).prepare(dev),
        }
    prefix = rc.bf16(ROWS, rc.HIDDEN)
    slots = req * QLEN + req
    conv = rc.bf16(slots, 3 * rc.PROJ, rc.SL, scale=0.5)
    rs = torch.randn((slots, rc.NH, rc.HD, rc.HD), generator=gen, device=dev) * 0.05
    # off: DSpark's per-token slots, every draft accepted last step
    st_off = (torch.arange(req * QLEN, device=dev, dtype=torch.int32) + req).view(
        req, QLEN
    )
    st_off[:, 0] = torch.arange(req, device=dev, dtype=torch.int32)
    acc = torch.full((req,), QLEN, device=dev, dtype=torch.int32)
    slot_idx = torch.arange(req, device=dev, dtype=torch.int32)
    st_on = torch.full((req, QLEN), -1, dtype=torch.int32, device=dev)
    st_on[:, 0] = slot_idx
    bufs = [
        rc.bf16(slots, rc.NH, CAP, rc.HD, scale=0.05),
        rc.bf16(slots, rc.NH, CAP, rc.HD, scale=0.05),
        -torch.rand((slots, rc.NH, CAP, rc.HD), generator=gen, device=dev) * 0.1,
    ]
    write_pos = torch.zeros(slots, dtype=torch.int32, device=dev)
    table = torch.tensor(
        [
            {
                "slot_idx": slot_idx,
                "write_pos": write_pos,
                "rbuf_k": bufs[0],
                "rbuf_u": bufs[1],
                "rbuf_g": bufs[2],
            }[name].data_ptr()
            for name in REPLAY_TABLE
        ],
        dtype=torch.int64,
        device=dev,
    )
    # the runtime's arguments, held for the graphs' lifetime
    held = []

    def launcher(k1, st, table_arg):
        blk1 = torch.zeros(ROWS, 1, rc.HIDDEN, dtype=torch.bfloat16, device=dev)
        core = torch.zeros((ROWS, rc.PROJ), dtype=torch.bfloat16, device=dev)
        runtime = rc.K1LocalRuntime(
            prefix=prefix,
            delta=None,
            blocks=blk1,
            blk_sm=blk1.stride(0),
            blk_sr=blk1.stride(1),
            ares_nw=rc.in_nw,
            ares_qk=rc.in_nw,
            in_nw=rc.in_nw,
            w_in=rc.w_in,
            w_in_scale=rc.w_in_scale,
            w_fb=rc.w_fb,
            conv_w=rc.conv_w,
            conv_st=conv,
            cs_seq=conv.stride(0),
            cs_dim=conv.stride(1),
            cs_tok=conv.stride(2),
            a_log=rc.a_log,
            dt_bias=rc.dt_bias,
            on_w=rc.on_w,
            rstate=rs,
            rs_seq=rs.stride(0),
            st_idx=st,
            st_stride=st.stride(0),
            num_acc=acc,
            core_out=core,
            core_q=None,
            core_scale=None,
            replay=table_arg,
        )
        held.append((blk1, core, runtime))
        return lambda: k1(runtime, stream=torch.cuda.current_stream())

    if SPLIT_AB:
        split_ab(kernels, launcher, st_on, table, write_pos)
        return
    if CHUNK_AB:
        chunk_ab(kernels, launcher, st_off, st_on, table, write_pos)
        return
    graphs = {"off": graph(launcher(kernels["off"], st_off, None))}
    on = graph(launcher(kernels["on"], st_on, table))
    times = {"off": []}
    for h in HS:
        times[f"on_h{h}"] = []

    def time_on(h):
        write_pos.fill_(h)
        torch.cuda.synchronize()
        times[f"on_h{h}"].append(time_us(on))

    for _ in range(RUNS):
        # ABBA: off, on at every cursor, on again in reverse, off
        times["off"].append(time_us(graphs["off"]))
        for h in HS:
            time_on(h)
        for h in reversed(HS):
            time_on(h)
        times["off"].append(time_us(graphs["off"]))
    sustained = {}
    if SUSTAIN > 0:
        # each arm replayed back to back for SUSTAIN seconds, off / on / on / off
        write_pos.fill_(SUSTAIN_H)
        torch.cuda.synchronize()
        for name in ("off", "on", "on", "off"):
            g = graphs["off"] if name == "off" else on
            start, end = torch.cuda.Event(True), torch.cuda.Event(True)
            launches, began = 0, time.monotonic()
            start.record()
            while time.monotonic() - began < SUSTAIN:
                for _ in range(50):
                    g.replay()
                launches += 50 * REPS
                torch.cuda.synchronize()
            end.record()
            torch.cuda.synchronize()
            sustained.setdefault(name, []).append(
                round(start.elapsed_time(end) * 1000 / launches, 2)
            )
    report = {
        "rows": ROWS,
        "names": {t: k.entrypoint.name for t, k in kernels.items()},
        "flush": {f"on_h{h}": h + 2 * WINDOW > CAP for h in HS},
        "median_us": {t: round(sorted(v)[len(v) // 2], 2) for t, v in times.items()},
        "min_us": {t: round(min(v), 2) for t, v in times.items()},
        "sustained_us": sustained,
        "sustained_h": SUSTAIN_H if SUSTAIN > 0 else None,
    }
    print(json.dumps(report), flush=True)


def sustain(g):
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    launches, began = 0, time.monotonic()
    start.record()
    while time.monotonic() - began < SUSTAIN:
        for _ in range(50):
            g.replay()
        launches += 50 * REPS
        torch.cuda.synchronize()
    end.record()
    torch.cuda.synchronize()
    return round(start.elapsed_time(end) * 1000 / launches, 2)


def chunk_ab(kernels, launcher, st_off, st_on, table, write_pos):
    graphs = {
        name: graph(
            launcher(k1, st_off, None)
            if name.startswith("off")
            else launcher(k1, st_on, table)
        )
        for name, k1 in kernels.items()
    }
    cases = [("off", None)] + [("on", h) for h in HS]
    times = {}

    def timed(kind, h, arm):
        if h is not None:
            write_pos.fill_(h)
            torch.cuda.synchronize()
        label = f"{kind}{'' if h is None else f'_h{h}'}_{arm}"
        times.setdefault(label, []).append(time_us(graphs[f"{kind}_{arm}"]))

    for _ in range(RUNS):
        for kind, h in cases:
            for arm in ("seq", "chunk", "chunk", "seq"):
                timed(kind, h, arm)
    sustained = {}
    if SUSTAIN > 0:
        for kind in ("off", "on"):
            write_pos.fill_(SUSTAIN_H)
            torch.cuda.synchronize()
            for arm in ("seq", "chunk", "chunk", "seq"):
                sustained.setdefault(f"{kind}_{arm}", []).append(
                    sustain(graphs[f"{kind}_{arm}"])
                )
    print(
        json.dumps(
            {
                "rows": ROWS,
                "names": {t: k.entrypoint.name for t, k in kernels.items()},
                "median_us": {
                    t: round(sorted(v)[len(v) // 2], 2) for t, v in times.items()
                },
                "min_us": {t: round(min(v), 2) for t, v in times.items()},
                "sustained_us": sustained,
                "sustained_h": SUSTAIN_H if SUSTAIN > 0 else None,
            }
        ),
        flush=True,
    )


def split_ab(kernels, launcher, st_on, table, write_pos):
    graphs = {name: graph(launcher(k1, st_on, table)) for name, k1 in kernels.items()}
    first, second = (f"on_s{split}" for split in SPLIT_AB)
    times = {}
    for _ in range(RUNS):
        for h in HS:
            write_pos.fill_(h)
            torch.cuda.synchronize()
            for arm in (first, second, second, first):
                times.setdefault(f"h{h}_{arm}", []).append(time_us(graphs[arm]))
    sustained = {}
    if SUSTAIN > 0:
        write_pos.fill_(SUSTAIN_H)
        torch.cuda.synchronize()
        for arm in (first, second, second, first):
            sustained.setdefault(arm, []).append(sustain(graphs[arm]))
    print(
        json.dumps(
            {
                "rows": ROWS,
                "names": {t: k.entrypoint.name for t, k in kernels.items()},
                "median_us": {
                    t: round(sorted(v)[len(v) // 2], 2) for t, v in times.items()
                },
                "min_us": {t: round(min(v), 2) for t, v in times.items()},
                "sustained_us": sustained,
                "sustained_h": SUSTAIN_H if SUSTAIN > 0 else None,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
