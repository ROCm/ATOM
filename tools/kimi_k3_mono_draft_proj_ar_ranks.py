# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The cross-rank half of the gfx1250 DSpark draft projection all-reduce
(``draft_proj_ar_gfx1250``) on ONE GPU: the kernel built for a real TP
(4, 8) and that TP's shard widths, launched as one rank ``r`` of it, every
peer's bank a local buffer between guard bands. No other rank runs: the host
computes each rank's production partial (``per_1x128_e8m0_quant`` and the
dispatch's FlyDSL GEMM on that rank's shard and activation) and places the
other ranks' partials in rank ``r``'s bank as the lines their launches would
push, tagged with the launch's epoch.

Per TP, projection, width and emulated rank (CHECK), on eager launches and
on REPLAYS replays of a captured launch, each on fresh activations:
- the output bit for bit the rank-order fp32 sum of the TP partials, rounded
  to BF16 once (aiter's custom all-reduce's arithmetic, its float mode's
  denormal flush included: the host model of it is first checked against a
  production HIP kernel in that mode, ``add_rmsnorm_quant``'s residual add);
  one more launch sums crafted peer partials through denormals, infinities
  and a NaN;
- every byte of every peer's bank after the launch: rank ``r``'s lines in
  its own slice of the epoch's bank equal its production partial (spare
  tokens zero, the tag the epoch), on every peer; every other byte -- the
  other ranks' slices, the other bank, both guard bands -- unchanged;
- no poll gave up; a control launch whose peers' lines carry a stale epoch
  does give up (the polls wait for the tag).
Then (BENCH) CUDA-graph time a launch with the peers' lines already in place
(no fabric hop: only the rank's own lines make their round trip) against
production's quant + GEMM, activations and weights rotating over COPIES
sets; every launch of the graph has its own receive bank and epoch counters,
re-armed by the host before each replay.

    python tools/kimi_k3_mono_draft_proj_ar_ranks.py
With PROFILE, a profile build's per-CTA stamps in the same arrangement
(peers' lines in place): where the launch's time goes, the poll of the
ranks' lines apart from the pushes and from the GEMM before them.

    python tools/kimi_k3_mono_draft_proj_ar_ranks.py
    env: TPS (4,8), KINDS (o_proj,down), ROWS (1,7,14,21,28,35,42,49,56),
    RANKS (first,mid,last), CHECK, BENCH, PROFILE (0), ITERS (2),
    REPLAYS (20), GN (32), REPS (9), COPIES (8), BOUND (4194304), LATE
    (unset; N: the late-push build, CHECK only),
    ATOM_KIMI_MONO_DRAFT_AR_POLL_TOGETHER (the builds' polls)

Real TP4 / TP8 (ranks on other GPUs, the fabric hop, peers' timing) is not
exercised here. Every kernel, buffer and graph stays referenced in KEEP.
"""

import json
import os
import statistics

import torch
from aiter import QuantType, dtypes, gemm_a8w8_blockscale_bpreshuffle
from aiter.ops.shuffle import shuffle_weight

from atom.model_ops.linear import per_1x128_e8m0_quant
from atom.models.kimi_k3.mono.config import HIDDEN
from atom.models.kimi_k3.mono.kernels.stages import draft_mlp_layout as lay
from atom.models.kimi_k3.mono.kernels.stages import draft_proj_ar_gfx1250 as dp
from atom.models.kimi_k3.mono.kernels.stages import k2_allreduce_gfx1250 as ar
from atom.mono.runtime.compile import compile_kernels, launcher_resources
from atom.quantization.quark.utils import quant_weight_online

KEEP = []
TPS = [int(t) for t in os.environ.get("TPS", "4,8").split(",")]
KINDS = os.environ.get("KINDS", "o_proj,down").split(",")
ROWS = [int(r) for r in os.environ.get("ROWS", "1,7,14,21,28,35,42,49,56").split(",")]
RANKS = os.environ.get("RANKS", "first,mid,last").split(",")
CHECK = os.environ.get("CHECK", "1") == "1"
BENCH = os.environ.get("BENCH", "1") == "1"
PROFILE = os.environ.get("PROFILE", "0") == "1"
ITERS = int(os.environ.get("ITERS", "2"))
REPLAYS = int(os.environ.get("REPLAYS", "20"))
GN = int(os.environ.get("GN", "32"))
REPS = int(os.environ.get("REPS", "9"))
COPIES = int(os.environ.get("COPIES", "8"))
BOUND = int(os.environ.get("BOUND", str(1 << 22)))
# LATE=N (CHECK only): a late-push build -- a CTA sums its partner CTA's
# columns, the odd CTAs pushing N sleeps late: lines that land while polled
LATE = int(os.environ["LATE"]) if os.environ.get("LATE") else None
TOTAL = {"o_proj": 8192, "down": 14336}
GUARD = 65536
PATTERN = 0xA5
PAIRS = HIDDEN // 2

dev = torch.device("cuda", torch.cuda.current_device())
gen = torch.Generator(device=dev).manual_seed(17)


def bf16(*shape, scale=1.0):
    return (torch.randn(shape, generator=gen, device=dev) * scale).to(torch.bfloat16)


def fp8_linear(rows, k, scale=0.02):
    q, s = quant_weight_online(
        bf16(rows, k, scale=scale),
        QuantType.per_1x128,
        dtypes.fp8,
        scale_dtype=dtypes.fp8_e8m0,
    )
    return shuffle_weight(q, layout=(16, 16)).contiguous(), s.contiguous()


def inputs(rows, k):
    x = bf16(rows, k, scale=0.5)
    x[0, :64] *= 40.0
    x[rows - 1, 128:256] = 0.0
    return x


def production(x, shard):
    """A rank's partial as LinearBase.forward computes it."""
    x8, xs = per_1x128_e8m0_quant(x, dtypes.fp8, True)
    return gemm_a8w8_blockscale_bpreshuffle(x8, shard[0], xs, shard[1], torch.bfloat16)


TINY = 2.0**-126


def flushed(v):
    """An f32 as a FLOAT_DENORM_MODE_32 0 kernel reads or writes it: a
    denormal is a zero of its sign."""
    return torch.where(v.abs() < TINY, torch.copysign(torch.zeros_like(v), v), v)


def _nan_bits(b):
    """The BF16 bits a NaN operand leaves in a sum (itself, quieted), 0
    where the operand is no NaN."""
    return torch.where((b & 0x7FFF) > 0x7F80, b | 0x0040, torch.zeros_like(b))


def reduced(partials):
    """The all-reduce as aiter's kernels sum it: rank order, fp32, their
    float mode's denormal flush on every operand and sum, one rounding to
    BF16. On the host (IEEE, denormals kept), the flush applied by hand; a
    NaN as ``v_pk_add_f32`` leaves it (``tools/kimi_k3_mono_draft_flush_diag.py``): a
    NaN operand itself, quieted, inf - inf the negative default NaN."""
    words = [p.cpu().contiguous().view(torch.int16) for p in partials]
    acc = flushed(partials[0].cpu().float())
    nan = _nan_bits(words[0])
    for partial, word in zip(partials[1:], words[1:]):
        acc = flushed(acc + flushed(partial.cpu().float()))
        operand = _nan_bits(word)
        fresh = torch.where(
            torch.isnan(acc), torch.full_like(nan, -64), torch.zeros_like(nan)
        )
        nan = torch.where(nan != 0, nan, torch.where(operand != 0, operand, fresh))
    out = acc.to(torch.bfloat16).contiguous().view(torch.int16)
    return torch.where(nan != 0, nan, out).view(torch.bfloat16)


def special(rows, zero_row=False):
    """A BF16 ``[rows, HIDDEN]`` no GEMM would produce: random values, and
    in row 0 the bit patterns a denormal flush shows on."""
    t = bf16(rows, HIDDEN, scale=3.0)
    v = t.view(torch.int16)
    if zero_row:
        v[0] = 0
    return t


def crafted(rows, others):
    """``others`` BF16 partials whose row 0 (beside a rank whose own row 0
    is zero) sums through denormals: 1.5 x 2^-126 - 1.25 x 2^-126 (a
    denormal sum), a denormal plus 2^-126 (a denormal operand), denormals
    alone, and infinities and a NaN."""
    out = [special(rows) for _ in range(others)]
    rows0 = [t.view(torch.int16)[0] for t in out]
    for r in rows0:
        r[:1024] = 0
    rows0[0][0:256] = 0x00C0
    rows0[-1][0:256] = -32768 + 0x00A0
    rows0[0][256:512] = 0x0040
    rows0[-1][256:512] = 0x0080
    rows0[0][512:640] = 0x0003
    rows0[-1][640:768] = -32768 + 0x007F
    rows0[0][768:800] = 0x7F80
    rows0[-1][800:832] = -32768 + 0x7F80
    rows0[0][832:864] = 0x7FC0
    return out


def flush_model_check():
    """The host's flush model against a production HIP kernel in the same
    float mode doing the same two instructions (``v_pk_add_f32``, then
    ``v_cvt_pk_bf16_f32``): ``add_rmsnorm_quant``'s residual add."""
    from atom.model_ops.layernorm import _aiter_rms_quant

    a, b = crafted(4, 2)
    a.view(torch.int16)[1, :64] = 0x0003
    weight = torch.ones(HIDDEN, dtype=torch.bfloat16, device=dev)
    _, _, got = _aiter_rms_quant(
        a, weight, 1e-6, QuantType.per_1x128.value, True, b, dtypes.fp8, True
    )
    torch.cuda.synchronize()
    want = reduced([a, b])
    kept = (a.cpu().float() + b.cpu().float()).to(torch.bfloat16)
    finite = torch.isfinite(want.float())
    bad = (bits(got.cpu()) != bits(want)).nonzero()[:4].tolist()
    print(
        json.dumps({
            "flush_model_matches_hip_residual_add": torch.equal(bits(got.cpu()), bits(want)),
            "ieee_sum_differs_there": not torch.equal(
                bits(kept)[finite], bits(want)[finite]
            ),
            "first_mismatches": [
                [r, c, hex(int(bits(got.cpu())[r, c]) & 0xFFFF),
                 hex(int(bits(want)[r, c]) & 0xFFFF)]
                for r, c in bad
            ],
        }),
        flush=True,
    )  # fmt: skip


def bits(t):
    return t.contiguous().view(torch.int16)


def lines_of(partial, rows, epoch):
    """A rank's partial as the lines its launch pushes: ``[triple][pair]``,
    three tokens' BF16 pair word (spare tokens zero) and the epoch."""
    count = lay.triples(rows)
    pairs = torch.zeros(count * 3, PAIRS, dtype=torch.int32, device=dev)
    pairs[:rows] = partial.contiguous().view(torch.int32).view(rows, PAIRS)
    lines = torch.empty(count, PAIRS, 4, dtype=torch.int32, device=dev)
    lines[:, :, :3] = pairs.view(count, 3, PAIRS).permute(0, 2, 1)
    lines[:, :, 3] = epoch
    return lines.view(-1, 4)


class Banks:
    """Every rank's bank as a local buffer between two guard bands, and the
    table of their bases the kernel is handed."""

    def __init__(self, tp, rows, shared=None, mine=None):
        self.tp, self.rows = tp, rows
        self.nbytes = lay.peer_bytes(tp, rows)
        self.lines = lay.triples(rows) * PAIRS
        self.buffers = [
            self._buffer() if shared is None or p == mine else shared.buffers[p]
            for p in range(tp)
        ]
        self.peers = torch.tensor(
            [b.data_ptr() + GUARD for b in self.buffers], dtype=torch.int64, device=dev
        )
        assert len({int(p) for p in self.peers.tolist()}) == tp
        KEEP.append(self)

    def _buffer(self):
        buffer = torch.zeros(GUARD + self.nbytes + GUARD, dtype=torch.uint8, device=dev)
        buffer[:GUARD] = PATTERN
        buffer[GUARD + self.nbytes :] = PATTERN
        return buffer

    def slots(self, rank):
        """A rank's bank as ``[parity][source][line][word]`` i32 (a view)."""
        body = self.buffers[rank][GUARD : GUARD + self.nbytes]
        return body.view(torch.int32).view(2, self.tp, self.lines, 4)


def launch(kernel, x, shard, out, banks, epochs, rank, flags):
    p = torch.Tensor.data_ptr
    kernel.launcher(
        *kernel.abi.pack({
            "x": p(x), "w": p(shard[0]), "w_scale": p(shard[1]), "out": p(out),
            "peers": p(banks.peers), "epochs": p(epochs), "rank": rank,
            "flags": p(flags),
        }),
        stream=torch.cuda.current_stream(),
    )  # fmt: skip


def resources(kernel):
    (launcher,) = compile_kernels([(kernel.launcher, kernel.abi)])
    res = launcher_resources(launcher)
    hazards = [
        h
        for r in res
        for h, bad in (
            ("scratch", r["scratch"]),
            ("vgpr_spill", r["vgpr_spill"]),
            ("sgpr_spill", r["sgpr_spill"]),
        )
        if bad
    ]
    return res, hazards


def graph_of(fn, warm):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            warm()
            fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=side):
        fn()
    KEEP.append((graph, side, fn))
    return graph


def emulated_ranks(tp):
    names = {"first": 0, "mid": tp // 2, "last": tp - 1}
    return sorted({names[r] if r in names else int(r) for r in RANKS})


def check_rank(tp, rows, k, kernel, shards, rank):
    """One emulated rank of ``check``: its entry of the report."""
    banks = Banks(tp, rows)
    epochs = torch.zeros(ar.MAX_CTAS, dtype=torch.int32, device=dev)
    flags = torch.zeros(4, dtype=torch.int32, device=dev)
    out = torch.empty(rows, HIDDEN, dtype=torch.bfloat16, device=dev)
    gx = inputs(rows, k)
    KEEP.extend([epochs, flags, out, gx])
    state = {}

    def arm(stale=None, craft=False):
        """Fresh activations on every rank; the other ranks' production
        partials placed in rank ``rank``'s bank as their lines, tagged
        with the next launch's epoch (``stale``: that source's with the
        epoch before; ``craft``: this rank's row 0 zero and the others'
        partials ``crafted``). Sets what the launch must leave behind."""
        xs = [inputs(rows, k) for _ in range(tp)]
        if craft:
            xs[rank][0] = 0.0
        gx.copy_(xs[rank])
        partials = [production(xs[p], shards[p]) for p in range(tp)]
        if craft:
            for p, t in zip([p for p in range(tp) if p != rank], crafted(rows, tp - 1)):
                partials[p] = t
        epoch = int(epochs[0]) + 1
        assert bool((epochs == epoch - 1).all())
        mine = banks.slots(rank)
        for p in range(tp):
            if p != rank:
                tag = epoch - 1 if p == stale else epoch
                mine[epoch & 1, p] = lines_of(partials[p], rows, tag)
        torch.cuda.synchronize()
        expected = [b.clone() for b in banks.buffers]
        for p in range(tp):
            body = expected[p][GUARD : GUARD + banks.nbytes]
            body.view(torch.int32).view(2, tp, banks.lines, 4)[epoch & 1, rank] = (
                lines_of(partials[rank], rows, epoch)
            )
        state.update(want=reduced(partials), expected=expected, epoch=epoch)

    def verdict():
        torch.cuda.synchronize()
        got, want = bits(out.cpu()), bits(state["want"])
        if not torch.equal(got, want):
            state.setdefault(
                "bad",
                [
                    [r, c, hex(int(got[r, c]) & 0xFFFF), hex(int(want[r, c]) & 0xFFFF)]
                    for r, c in (got != want).nonzero()[:4].tolist()
                ],
            )
        return {
            "out": torch.equal(got, want),
            "banks": all(
                torch.equal(b, e) for b, e in zip(banks.buffers, state["expected"])
            ),
            "epoch": bool((epochs == state["epoch"]).all()),
            "gave_up": int(flags[0]) == 0,
        }

    eager = []
    for _ in range(ITERS):
        arm()
        launch(kernel, gx, shards[rank], out, banks, epochs, rank, flags)
        eager.append(verdict())
    # a late-push build: thread 0's polls on an even CTA (it waits for its
    # odd partner's late lines) and on an odd one
    late_polls = flags[1:3].tolist()
    late_seen = LATE is None or late_polls[0] >= LATE // 4
    graph = graph_of(
        lambda: launch(kernel, gx, shards[rank], out, banks, epochs, rank, flags),
        arm,
    )
    replays = []
    for _ in range(REPLAYS):
        arm()
        graph.replay()
        replays.append(verdict())
    # sums through denormals, infinities and a NaN
    arm(craft=True)
    launch(kernel, gx, shards[rank], out, banks, epochs, rank, flags)
    special_sum = verdict()
    # the control: a source's lines carry the epoch before this launch's
    arm(stale=(rank + 1) % tp)
    launch(kernel, gx, shards[rank], out, banks, epochs, rank, flags)
    torch.cuda.synchronize()
    stale_gave_up = (int(flags[0]) & 0x7FF00000) == lay.GAVE_UP
    rank_ok = (
        all(all(v.values()) for v in eager)
        and all(all(v.values()) for v in replays)
        and all(special_sum.values())
        and stale_gave_up
        and late_seen
    )
    bad = [v for v in eager + replays if not all(v.values())]
    return {
        "eager": len(eager), "replays": len(replays),
        "exact": sum(all(v.values()) for v in eager + replays),
        "denormal_inf_nan_sum": special_sum,
        "stale_epoch_gives_up": stale_gave_up, "ok": rank_ok,
        **({"late": LATE, "late_polls": late_polls} if LATE is not None else {}),
        **({"bad_out": state["bad"]} if "bad" in state else {}),
        **({"first_bad": bad[0]} if bad else {}),
    }  # fmt: skip


def check(tp, kind, rows, kernel, shards):
    k = TOTAL[kind] // tp
    res, hazards = resources(kernel)
    report = {
        "tp": tp, "kind": kind, "k": k, "rows": rows, "kernel": kernel.name,
        "resources": res, "hazards": hazards, "ranks": {},
    }  # fmt: skip
    ok = not hazards
    for rank in emulated_ranks(tp):
        entry = check_rank(tp, rows, k, kernel, shards, rank)
        report["ranks"][rank] = entry
        ok = ok and entry["ok"]
    report["ok"] = ok
    print(json.dumps(report), flush=True)


def timed(graph):
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    start.record()
    graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / GN


def bench(tp, kind, rows, kernel, shards):
    """GN launches of rank 0 in a graph, each with its own receive bank and
    epoch counters; the peers' lines in place before every replay, the
    rank's own lines never (they make their round trip in the launch)."""
    k = TOTAL[kind] // tp
    rank = 0
    weights = [shards[0]] + [fp8_linear(HIDDEN, k) for _ in range(COPIES - 1)]
    xs = [inputs(rows, k) for _ in range(COPIES)]
    shared = Banks(tp, rows)
    sets = [Banks(tp, rows, shared=shared, mine=rank) for _ in range(GN)]
    counters = [
        torch.zeros(ar.MAX_CTAS, dtype=torch.int32, device=dev) for _ in range(GN)
    ]
    flags = torch.zeros(4, dtype=torch.int32, device=dev)
    out = torch.empty(rows, HIDDEN, dtype=torch.bfloat16, device=dev)
    KEEP.extend([weights, xs, counters, flags, out])
    filler = lines_of(production(xs[0], weights[0]), rows, 0)
    for banks in sets:
        for p in range(1, tp):
            banks.slots(rank)[1, p] = filler
    replay = [0]

    def arm():
        replay[0] += 1
        epoch = 2 * replay[0] + 1
        for banks, epochs in zip(sets, counters):
            epochs.fill_(epoch - 1)
            mine = banks.slots(rank)
            mine[1, :, :, 3] = epoch
            mine[1, rank, :, 3] = 0
        torch.cuda.synchronize()

    def run_mono():
        for i in range(GN):
            launch(kernel, xs[i % COPIES], weights[i % COPIES], out, sets[i],
                   counters[i], rank, flags)  # fmt: skip

    def run_prod():
        for i in range(GN):
            KEEP.append(production(xs[i % COPIES], weights[i % COPIES]))

    graphs = {
        "mono_peers_in_place": graph_of(run_mono, arm),
        "production_quant_gemm": graph_of(run_prod, lambda: None),
    }
    times = {name: [] for name in graphs}
    for _ in range(REPS):
        for name, graph in graphs.items():
            if name.startswith("mono"):
                arm()
            times[name].append(timed(graph))
    torch.cuda.synchronize()
    out_report = {
        "tp": tp, "kind": kind, "k": k, "rows": rows, "gn": GN,
        "gave_up": int(flags[0]),
        "guards_intact": all(
            bool((b[:GUARD] == PATTERN).all())
            and bool((b[GUARD + banks.nbytes :] == PATTERN).all())
            for banks in sets
            for b in banks.buffers
        ),
    }  # fmt: skip
    for name, t in times.items():
        out_report[name] = {
            "median": round(statistics.median(t), 2),
            "min": round(min(t), 2),
            "max": round(max(t), 2),
        }
    print(json.dumps({"bench": out_report}), flush=True)


flush_model_check()


def profile(tp, kind, rows, shards):
    """A profile build's stamps over GN launches of rank 0 in a graph, each
    with its own receive bank and epoch counters, the peers' lines in place:
    per phase, us a CTA (its own clock), the CTAs' median, medians over the
    launches. ``poll`` is from the last push to the lines summed: the rank's
    own lines' round trip and the loads of every rank's line."""
    k = TOTAL[kind] // tp
    rank = 0
    kernel = dp.build_draft_proj_ar(rows, tp, k, poll_bound=BOUND, profile=True)
    xs = [inputs(rows, k) for _ in range(COPIES)]
    shared = Banks(tp, rows)
    sets = [Banks(tp, rows, shared=shared, mine=rank) for _ in range(GN)]
    counters = [
        torch.zeros(ar.MAX_CTAS, dtype=torch.int32, device=dev) for _ in range(GN)
    ]
    flags = torch.zeros(4, dtype=torch.int32, device=dev)
    out = torch.empty(rows, HIDDEN, dtype=torch.bfloat16, device=dev)
    stamps = torch.zeros(lay.CTAS * dp.STAMPS, dtype=torch.int32, device=dev)
    snaps = torch.zeros(GN, lay.CTAS * dp.STAMPS, dtype=torch.int32, device=dev)
    KEEP.extend([kernel, xs, counters, flags, out, stamps, snaps])
    filler = lines_of(production(xs[0], shards[0]), rows, 0)
    for banks in sets:
        for p in range(1, tp):
            banks.slots(rank)[1, p] = filler
    replay = [0]

    def arm():
        replay[0] += 1
        epoch = 2 * replay[0] + 1
        for banks, epochs in zip(sets, counters):
            epochs.fill_(epoch - 1)
            mine = banks.slots(rank)
            mine[1, :, :, 3] = epoch
            mine[1, rank, :, 3] = 0
        torch.cuda.synchronize()

    ptr = torch.Tensor.data_ptr

    def run():
        for i in range(GN):
            kernel.launcher(
                *kernel.abi.pack({
                    "x": ptr(xs[i % COPIES]), "w": ptr(shards[0][0]),
                    "w_scale": ptr(shards[0][1]), "out": ptr(out),
                    "peers": ptr(sets[i].peers), "epochs": ptr(counters[i]),
                    "rank": rank, "flags": ptr(flags), "stamps": ptr(stamps),
                }),
                stream=torch.cuda.current_stream(),
            )  # fmt: skip
            snaps[i].copy_(stamps)

    graph = graph_of(run, arm)
    for _ in range(3):
        arm()
        graph.replay()
    torch.cuda.synchronize()
    st = snaps.cpu().long().view(GN, lay.CTAS, dp.STAMPS)
    at = ((st - st[:, :, :1]) & 0xFFFFFFFF).float() / 100.0

    def median(values):
        return round(values[GN // 2 :].median(dim=1).values.median().item(), 2)

    report = {
        "tp": tp, "kind": kind, "k": k, "rows": rows, "gave_up": int(flags[0]),
        "landed": median(at[:, :, dp.STAMP_LANDED]),
        "quantized": median(at[:, :, dp.STAMP_QUANTIZED]),
        "staged": median(at[:, :, dp.STAMP_STAGED]),
        "pushed": median(at[:, :, dp.STAMP_PUSHED]),
        "summed": median(at[:, :, dp.STAMP_SUMMED]),
        "end": median(at[:, :, dp.STAMP_END]),
        "push": median(at[:, :, dp.STAMP_PUSHED] - at[:, :, dp.STAMP_STAGED]),
        "poll": median(at[:, :, dp.STAMP_SUMMED] - at[:, :, dp.STAMP_PUSHED]),
        "land": median(at[:, :, dp.STAMP_END] - at[:, :, dp.STAMP_SUMMED]),
    }  # fmt: skip
    print(json.dumps({"profile": report}), flush=True)


for tp in TPS:
    for kind in KINDS:
        k = TOTAL[kind] // tp
        shards = [fp8_linear(HIDDEN, k) for _ in range(tp)]
        KEEP.append(shards)
        for rows in ROWS:
            why = lay.refusal(rows, k, tp)
            if why is not None:
                print(json.dumps({"tp": tp, "kind": kind, "k": k, "rows": rows,
                                  "refused": why}), flush=True)  # fmt: skip
                continue
            kernel = dp.build_draft_proj_ar(
                rows, tp, k, poll_bound=BOUND, test_swap_late=LATE
            )
            KEEP.append(kernel)
            if CHECK:
                check(tp, kind, rows, kernel, shards)
            if BENCH:
                bench(tp, kind, rows, kernel, shards)
            if PROFILE:
                profile(tp, kind, rows, shards)
