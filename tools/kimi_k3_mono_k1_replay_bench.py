"""K1 alone, ReplaySSM off against on (``kimi_k3_mono_k1_replay_check``'s
setup), each a CUDA graph of REPS launches timed over RUNS alternating runs.

Off: every token's state stored (DSpark slots). On: one checkpoint a request,
each step's records written; ``H`` committed records (non-flush: h + 2 window
<= cache_len, else a flush that folds them into the checkpoint first). The
step's K1 never moves the cursor, so every launch replays the same h.

Run on one gfx1250 GPU: ROWS=40 H=0,5,6,8,11 python tools/kimi_k3_mono_k1_replay_bench.py"""

import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kimi_k3_mono_k1_replay_check as rc  # noqa: E402
from atom.models.kimi_k3.mono.kernels.attention.kda_builder_gfx1250 import (
    REPLAY_TABLE,
)
from atom.models.kimi_k3.mono.kernels.execution_plan import Gfx1250KimiBuildPlan

ROWS = int(os.environ.get("ROWS", "40"))
HS = [int(h) for h in os.environ.get("H", "0,5,6,8,11").split(",")]
REPS = int(os.environ.get("REPS", "20"))
RUNS = int(os.environ.get("RUNS", "8"))
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

    graphs = {"off": graph(launcher(kernels["off"], st_off, None))}
    on = graph(launcher(kernels["on"], st_on, table))
    times = {"off": []}
    for h in HS:
        times[f"on_h{h}"] = []
    for _ in range(RUNS):
        times["off"].append(time_us(graphs["off"]))
        for h in HS:
            write_pos.fill_(h)
            torch.cuda.synchronize()
            times[f"on_h{h}"].append(time_us(on))
    report = {
        "rows": ROWS,
        "names": {t: k.entrypoint.name for t, k in kernels.items()},
        "flush": {f"on_h{h}": h + 2 * WINDOW > CAP for h in HS},
        "median_us": {
            t: round(sorted(v)[len(v) // 2], 2) for t, v in times.items()
        },
        "min_us": {t: round(min(v), 2) for t, v in times.items()},
    }
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
