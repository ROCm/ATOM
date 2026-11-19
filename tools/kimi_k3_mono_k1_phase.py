"""K1 alone (``kimi_k3_mono_k1_replay_check``'s setup, ReplaySSM off): its
CUDA-graph latency with the in-projection weights rotated over COPIES copies
(cold, as in serving), then the same graph built with ``tdm_timeline`` on and
each CTA's phases in µs from its own entry (cycle stamps, converted with the
CTA's realtime entry/end), median over the LAUNCHES launches of each phase's
median and max over the CTAs that stamped it.

Run on one gfx1250 GPU:
  ROWS=8,40 QLEN=8 WINDOW=8 python tools/kimi_k3_mono_k1_phase.py
TL=0 times only; RAW=prefix saves the stamp words to prefix_<rows>.pt;
OUT=prefix one launch's core/state/conv to prefix_<rows>.pt (bitwise checks).
AB=<switch env> times the switch off against on (ABBA, RUNS rounds).
SECS=s: sustained (s of back-to-back replays a segment, the GFX clock from
amdsmi index SMI_GPU, /opt/rocm/share/amd_smi on PYTHONPATH): the AB
segments, and the stamps taken at the end of such a run."""

import json
import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kimi_k3_mono_k1_replay_check as rc  # noqa: E402
from atom.models.kimi_k3.mono.kernels.execution_plan import (  # noqa: E402
    Gfx1250KimiBuildPlan,
)
from atom.models.kimi_k3.mono.kernels.attention.kda_builder_gfx1250 import (  # noqa: E402
    REPLAY_TABLE,
)
from atom.models.kimi_k3.mono.kernels.stages import tdm_timeline as tl  # noqa: E402
from atom.mono.runtime.compile import launcher_resources  # noqa: E402

WIDTHS = [int(r) for r in os.environ.get("ROWS", "8,40").split(",")]
COPIES = int(os.environ.get("COPIES", "8"))
LAUNCHES = int(os.environ.get("LAUNCHES", "8"))
TIMELINE = os.environ.get("TL", "1") == "1"
REPLAY = os.environ.get("REPLAY") == "1"
HREC = int(os.environ.get("H", "5"))
dev, QLEN = rc.dev, rc.QLEN
weights = [rc.w_in] + [rc.w_in.clone() for _ in range(COPIES - 1)]

PROJ_POINTS = ["gemv_start", "issued", "s0_T", "s0_W", "s0_B", "s0_C",
               "s1_C", "last_C", "project_done", "staged", "stores_issued",
               "stores_done", "recurrence"]
HEAD_POINTS = ["head_start", "head_polled", "head_conv", "head_gate_mma",
               "head_gate_sync", "head_gates", "head_recur", "head_exchanged",
               "head_done"]
SLABS = rc.HIDDEN // 512


def point_index(name):
    if name.startswith("s0_") or name.startswith("s1_") or name == "last_C":
        slab = {"s0": 0, "s1": 1, "last": SLABS - 1}[name.split("_")[0]]
        which = {"T": tl.TOKEN, "W": tl.WEIGHT, "B": tl.B, "C": tl.C}[name[-1]]
        return tl.SLAB0 + tl.SLAB_WORDS * slab + which
    return tl.POINTS[name]


def inputs(rows):
    req = rows // QLEN
    slots = req * QLEN + req
    prefix = rc.bf16(rows, rc.HIDDEN)
    conv = rc.bf16(slots, 3 * rc.PROJ, rc.SL, scale=0.5)
    rs = torch.randn((slots, rc.NH, rc.HD, rc.HD), generator=rc.gen, device=dev) * 0.05
    acc = torch.full((req,), QLEN, device=dev, dtype=torch.int32)
    if not REPLAY:
        st = (torch.arange(req * QLEN, device=dev, dtype=torch.int32) + req).view(req, QLEN)
        st[:, 0] = torch.arange(req, device=dev, dtype=torch.int32)
        return [prefix, conv, rs, st, acc, None]
    # ReplaySSM (kimi_k3_mono_k1_replay_bench's): one checkpoint a request,
    # H committed records replayed every launch
    slot_idx = torch.arange(req, device=dev, dtype=torch.int32)
    st = torch.full((req, QLEN), -1, dtype=torch.int32, device=dev)
    st[:, 0] = slot_idx
    bufs = [rc.bf16(slots, rc.NH, rc.CAP, rc.HD, scale=0.05),
            rc.bf16(slots, rc.NH, rc.CAP, rc.HD, scale=0.05),
            -torch.rand((slots, rc.NH, rc.CAP, rc.HD), generator=rc.gen, device=dev) * 0.1]
    write_pos = torch.full((slots,), HREC, dtype=torch.int32, device=dev)
    named = {"slot_idx": slot_idx, "write_pos": write_pos,
             "rbuf_k": bufs[0], "rbuf_u": bufs[1], "rbuf_g": bufs[2]}
    table = torch.tensor([named[n].data_ptr() for n in REPLAY_TABLE],
                         dtype=torch.int64, device=dev)
    return [prefix, conv, rs, st, acc, table, named]


def resources(k1):
    return [{k: r[k] for k in ("vgpr", "sgpr", "vgpr_spill", "sgpr_spill", "scratch", "lds") if k in r}
            for r in launcher_resources(k1.entrypoint.launcher)]


def graph_for(k1, held, rows):
    prefix, conv, rs, st, acc, table = held[:6]
    blk1 = torch.zeros(rows, 1, rc.HIDDEN, dtype=torch.bfloat16, device=dev)
    core = torch.zeros((rows, rc.PROJ), dtype=torch.bfloat16, device=dev)
    runtimes = [
        rc.K1LocalRuntime(
            prefix=prefix, delta=None, blocks=blk1, blk_sm=blk1.stride(0),
            blk_sr=blk1.stride(1), ares_nw=rc.in_nw, ares_qk=rc.in_nw, in_nw=rc.in_nw,
            w_in=weights[i % COPIES], w_in_scale=rc.w_in_scale, w_fb=rc.w_fb,
            conv_w=rc.conv_w, conv_st=conv, cs_seq=conv.stride(0),
            cs_dim=conv.stride(1), cs_tok=conv.stride(2), a_log=rc.a_log,
            dt_bias=rc.dt_bias, on_w=rc.on_w, rstate=rs, rs_seq=rs.stride(0),
            st_idx=st, st_stride=st.stride(0), num_acc=acc, core_out=core,
            core_q=None, core_scale=None, replay=table)
        for i in range(LAUNCHES)
    ]
    k1(runtimes[0], stream=torch.cuda.current_stream())
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for r in runtimes:
            k1(r, stream=torch.cuda.current_stream())
    held.extend([blk1, core, runtimes, g])
    return g


def time_us(g, runs=7):
    out = []
    for _ in range(runs):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        g.replay()
        a.record()
        g.replay()
        b.record()
        b.synchronize()
        out.append(a.elapsed_time(b) * 1000 / LAUNCHES)
    return round(sorted(out)[runs // 2], 2)


def phases(words):
    """[LAUNCHES, CTAS, WORDS] int64 -> per point (median, max) over CTAs,
    median over launches; plus the CTAs' entry skew and duration."""
    m32 = 1 << 32
    per = {}
    for launch in words:
        cyc0 = launch[:, tl.CYCLES_ENTRY]
        rt = (launch[:, tl.END] - launch[:, tl.ENTRY]) % m32
        cyc = (launch[:, tl.CYCLES_END] - cyc0) % m32
        live = (launch[:, tl.ENTRY] != 0) & (rt > 0)
        freq = cyc[live].float() / (rt[live].float() / 100.0)
        f = float(freq.median())
        entry = launch[live, tl.ENTRY]
        rel_entry = ((entry - entry.min()) % m32).float() / 100.0
        dur = rt[live].float() / 100.0
        rows = {"entry_skew": rel_entry, "cta_dur": dur}
        for name in PROJ_POINTS + HEAD_POINTS:
            p = point_index(name)
            stamped = live & (launch[:, p] != 0)
            if stamped.any():
                rows[name] = ((launch[stamped, p] - cyc0[stamped]) % m32).float() / f
        rows["ghz"] = torch.tensor([f / 1000.0])
        for k, v in rows.items():
            per.setdefault(k, []).append((float(v.median()), float(v.max()), int(v.numel())))
    return {k: [round(statistics.median(x[0] for x in v), 2),
                round(statistics.median(x[1] for x in v), 2),
                v[0][2]] for k, v in per.items()}


class Clock:
    """Read-only amdsmi GFX clock samples of GPU ``SMI_GPU`` (its amdsmi
    index) every 20 ms between ``start`` and ``stop``; none without amdsmi."""

    def __init__(self):
        self.samples, self._run, self._thread = [], False, None
        try:
            import amdsmi

            amdsmi.amdsmi_init()
            handle = amdsmi.amdsmi_get_processor_handles()[int(os.environ["SMI_GPU"])]
            def read():
                power = amdsmi.amdsmi_get_power_info(handle)
                watts = next((power[k] for k in ("current_socket_power", "socket_power",
                              "average_socket_power") if isinstance(power.get(k), (int, float))), None)
                return amdsmi.amdsmi_get_clock_info(handle, amdsmi.AmdSmiClkType.GFX)["clk"], watts

            self._read = read
        except Exception:
            self._read = None

    def start(self):
        import threading
        import time

        self.samples, self._run = [], True

        def loop():
            while self._run:
                if self._read is not None:
                    self.samples.append(self._read())
                time.sleep(0.02)

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._run = False
        self._thread.join()
        if not self.samples:
            return {}
        mhz = sorted(m for m, _ in self.samples)
        watts = sorted(w for _, w in self.samples if w is not None)
        out = {"mhz_med": mhz[len(mhz) // 2], "mhz_min": mhz[0], "mhz_max": mhz[-1]}
        if watts:
            out["w_med"] = watts[len(watts) // 2]
        return out


def sustained_us(g, seconds, clock=None):
    """Back-to-back replays for ``seconds``: µs a launch over the second half."""
    import time

    if clock:
        clock.start()
    begin = time.time()
    while time.time() - begin < seconds / 2:
        for _ in range(20):
            g.replay()
        torch.cuda.synchronize()
    a, b, n = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True), 0
    a.record()
    while time.time() - begin < seconds:
        for _ in range(20):
            g.replay()
        n += 20
        b.record()
        b.synchronize()
    us = round(a.elapsed_time(b) * 1000 / (n * LAUNCHES), 2)
    return (us, clock.stop()) if clock else (us, {})


def abba(switch):
    """``switch`` off against on, both built in this process, timed
    alternately (ABBA) RUNS rounds a width; SECS > 0: each a sustained
    segment of back-to-back replays, with the GFX clock."""
    rounds = int(os.environ.get("RUNS", "4"))
    seconds = float(os.environ.get("SECS", "0"))
    clock = Clock() if seconds else None
    for rows in WIDTHS:
        plan = Gfx1250KimiBuildPlan(threads=256, blocks=256, lds_bytes=320 * 1024,
                                    tokens=rows, nblocks=1)
        graphs, names, held = {}, {}, []
        for value in ("0", "1"):
            os.environ[switch] = value
            k1 = plan.make_k1_local_runner(rc.spec(rows, REPLAY)).prepare(dev)
            h = list(inputs(rows))
            graphs[value] = graph_for(k1, h, rows)
            names[value] = (k1.entrypoint.name, resources(k1))
            held.append((k1, h))
        times, clocks = {"0": [], "1": []}, {"0": [], "1": []}
        for i in range(rounds):
            for value in (("0", "1") if i % 2 == 0 else ("1", "0")):
                if seconds:
                    us, mhz = sustained_us(graphs[value], seconds, clock)
                    times[value].append(us)
                    clocks[value].append((mhz.get("mhz_med"), mhz.get("w_med")))
                else:
                    times[value].append(time_us(graphs[value]))
        print(json.dumps({"rows": rows, "qlen": QLEN, "switch": switch, "names": names,
                          "secs": seconds, "mhz_off": clocks["0"], "mhz_on": clocks["1"],
                          "off": times["0"], "on": times["1"],
                          "median_off": statistics.median(times["0"]),
                          "median_on": statistics.median(times["1"])}), flush=True)


def main():
    if os.environ.get("AB"):
        abba(os.environ["AB"])
        return
    for rows in WIDTHS:
        plan = Gfx1250KimiBuildPlan(threads=256, blocks=256, lds_bytes=320 * 1024,
                                    tokens=rows, nblocks=1)
        tl.disable()
        k1 = plan.make_k1_local_runner(rc.spec(rows, REPLAY)).prepare(dev)
        if os.environ.get("OUT"):
            prefix, conv, rs, st, acc, table, *_ = inputs(rows)
            core, _ = rc.k1_run(k1, prefix, conv, rs, st, acc, table)
            torch.cuda.synchronize()
            torch.save([t.cpu() for t in (core, rs, conv)], f"{os.environ['OUT']}_{rows}.pt")
        held = list(inputs(rows))
        us = time_us(graph_for(k1, held, rows))
        report = {"rows": rows, "qlen": QLEN, "us": us, "name": k1.entrypoint.name,
                  "resources": resources(k1)}
        if TIMELINE:
            buf = torch.zeros(tl.buffer_words(LAUNCHES), dtype=torch.int32, device=dev)
            tl.enable(buf.data_ptr(), LAUNCHES)
            k1t = plan.make_k1_local_runner(rc.spec(rows, REPLAY)).prepare(dev)
            tl.disable()
            held_t = list(inputs(rows))
            g = graph_for(k1t, held_t, rows)
            report["us_timeline"] = time_us(g)
            buf.zero_()
            seconds = float(os.environ.get("SECS", "0"))
            if seconds:
                # sustained: the stamps are the last of SECS of back-to-back
                # replays (each launch's slot is overwritten in turn)
                report["sustained_us_timeline"], report["sustained_clock"] = sustained_us(
                    g, seconds, Clock())
            else:
                g.replay()
            torch.cuda.synchronize()
            words = buf[tl.CTAS:].view(LAUNCHES, tl.CTAS, tl.WORDS).cpu().long() & 0xFFFFFFFF
            if os.environ.get("RAW"):
                torch.save(words, f"{os.environ['RAW']}_{rows}.pt")
            report["phases_med_max_n"] = phases(words)
            held.append((buf, k1t, held_t))
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
