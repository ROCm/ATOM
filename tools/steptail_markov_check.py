# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The step tail's draft launch (draft LM head + greedy Markov block sampling)
on one GPU, built for the real TP8 shape, against production on the same
inputs: ``tgemm.mm`` on every rank's shard (the all-gather is a
concatenation), then ``dspark_markov_argmax`` over the whole vocabulary a
block position, as ``DSparkMarkovHead.sample_block`` runs it.

The GPU is one TP rank; the host plays the other seven. Every round's line of
every peer is placed before the launch, from production's own sequence: the
op's sharded form gives each rank's ``(max, id)`` of a round given the true
x_{k-1}. Every rank's region is a local buffer between guard bands, and after
the launch every byte is compared with what the kernel may write (its own
line of every round and request, in every region).

Checked: the draft ids; the regions and guards; the counters; no poll gave
up. Inputs: random, heavy ties (64 distinct LM head rows and 32 distinct W2
rows), NaN / inf hidden rows, NaN / inf W2 rows and an inf W1 row; then
REPLAYS CUDA-graph replays; the give-up branch (a peer's line withheld).
BENCH: CUDA-graph time against the production kernels (a build that waits for
this rank's own lines only: one GPU has no peers).

    python tools/steptail_markov_check.py
    env: SHAPES (1x7,8x7,8x4: requests x drafts), RANKS (0,5), ITERS (2),
    REPLAYS (20), CHECK (1), BENCH (1), GIVEUP (1), REPS (8), ROUNDS (12),
    DIAG (0: launches to run one by one, eager then replayed),
    DESIGN ("": the served design; v1: the first design, the diagnostic alone),
    VOLATILE (0; v1: the poll's loads volatile), TICKS (0: the diagnostic
    build's poll bound in 10 ns ticks, 0 the serving 2 s), REGIONS (plain:
    torch memory; uncached, fine: what a runner's peer buffers are made of);
    v1 alone, a feature of its poll at a time: SLEEP (0: the pause's
    units), POLLROUNDS (0: bound the loop by its rounds, no clock read in
    it), EXTRA (0: threads past the last request's that poll request 0's
    lines again), POLLERS (256: the CTAs that poll), SETTLE (0: wait for
    each poll load at once), CLOCKFIRST (0: a round's clock read before
    its load)
"""

import os
import statistics
import sys
import time

import torch
from aiter.tuned_gemm import tgemm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import steptail_check as C

from atom.model_ops.dspark_markov_argmax import _launch, dspark_markov_argmax
from atom.models.kimi_k3.step_tail import head_dispatch
from atom.models.kimi_k3.step_tail import kernels as K
from atom.models.kimi_k3.step_tail import layout as L
from atom.models.kimi_k3.step_tail import markov_kernels as MK
from atom.models.kimi_k3.step_tail import markov_layout as M
from atom.models.kimi_k3.step_tail.sources import MARKOV_SOURCES
from atom.mono.runtime.compile import compile_only, launcher_resources

SHAPES = [
    tuple(int(v) for v in s.split("x"))
    for s in os.environ.get("SHAPES", "1x7,8x7,8x4").split(",")
]
RANKS = [int(r) for r in os.environ.get("RANKS", "0,5").split(",")]
ITERS = int(os.environ.get("ITERS", "2"))
REPLAYS = int(os.environ.get("REPLAYS", "20"))
CHECK = os.environ.get("CHECK", "1") == "1"
BENCH = os.environ.get("BENCH", "1") == "1"
GIVEUP = os.environ.get("GIVEUP", "1") == "1"
DIAG = int(os.environ.get("DIAG", "0"))
DESIGN = os.environ.get("DESIGN", "")
VOLATILE = os.environ.get("VOLATILE", "0") == "1"
SLEEP = int(os.environ.get("SLEEP", "0"))
POLLROUNDS = int(os.environ.get("POLLROUNDS", "0"))
EXTRA = int(os.environ.get("EXTRA", "0"))
POLLERS = int(os.environ.get("POLLERS", str(L.CTAS)))
SETTLE = os.environ.get("SETTLE", "0") == "1"
CLOCKFIRST = os.environ.get("CLOCKFIRST", "0") == "1"
TICKS = int(os.environ.get("TICKS", "0"))
REGIONS = os.environ.get("REGIONS", "plain")
if DESIGN == "v1":  # the first design (tool only): the diagnostic alone
    import hashlib

    import steptail_markov_v1 as MK

    from atom.mono.plan.build_key import source_digest

    MARKOV_SOURCES = source_digest(
        "models/kimi_k3/step_tail/kernels.py",
        "models/kimi_k3/step_tail/layout.py",
        "models/kimi_k3/step_tail/markov_layout.py",
        "models/deepseek_v41/mono/gfx1250/kernels/device.py",
    )
    with open(
        MK.__file__, "rb"
    ) as _source:  # the design is a tool's file: its bytes join the key
        MARKOV_SOURCES += hashlib.sha256(_source.read()).hexdigest()[:8]
    CHECK = BENCH = GIVEUP = False
elif DESIGN:
    raise SystemExit(f"DESIGN {DESIGN!r}: not a design")
if REGIONS not in ("plain", "uncached", "fine"):
    raise SystemExit(f"REGIONS {REGIONS!r}: plain, uncached or fine")
# a diagnostic build's records (the served design's: [round][thread], [CTA], [round][thread])
DIAG_WORDS = getattr(MK, "DG_WORDS", 2 * L.MAX_DRAFTS * L.THREADS + L.CTAS)
KNOBS = {
    "volatile": VOLATILE,
    "sleep": SLEEP,
    "rounds": POLLROUNDS,
    "extra": EXTRA,
    "pollers": POLLERS,
    "settle": SETTLE,
    "clockfirst": CLOCKFIRST,
}
V1_BUILD = KNOBS if DESIGN == "v1" else {}
REPS = int(os.environ.get("REPS", "8"))
ROUNDS = int(os.environ.get("ROUNDS", "12"))
RK = 256
dev, bf16, fail = C.dev, C.bf16, C.fail


def production(w, w1, w2, xd, anchors, requests, drafts):
    """Production's drafts [requests, drafts] i32, and each rank's (key, id)
    of each round as its kernel publishes them: [drafts, TP, requests]."""
    base = C.production_logits(w, xd).view(requests, drafts, L.VOCAB)
    block = torch.zeros(requests, drafts + 1, dtype=torch.int32, device=dev)
    block[:, 0] = anchors
    key = torch.empty(drafts, L.TP, requests, dtype=torch.int64, device=dev)
    gid = torch.empty_like(key)
    packed = torch.empty(requests, 2, dtype=torch.float32, device=dev)
    scratch = torch.empty(requests, dtype=torch.int32, device=dev)
    for k in range(drafts):
        dspark_markov_argmax(base[:, k], block[:, k], w1, w2, block[:, k + 1])
        prev = block[:, k].contiguous()
        for s in range(L.TP):
            mine = base[:, k, s * L.VR : (s + 1) * L.VR].contiguous()
            _launch(mine, prev, w1, w2, scratch, s * L.VR, packed)
            best = packed[:, 0].contiguous()
            bits = best.view(torch.int32).to(torch.int64)
            bits = torch.where(bits == -(1 << 31), torch.zeros_like(bits), bits)
            order = bits ^ ((bits >> 31) & 0x7FFFFFFF)
            # A NaN never wins, and a shard of NaN alone publishes (lowest key, its lowest id): the kernels'
            # convention. The op's packed form answers (-inf, 0) there (its tile padding), so which shards are
            # all NaN is taken from the logits themselves (fp32 in torch: only their NaN-ness is used).
            z = (
                mine.float()
                + w1[prev.long()].float() @ w2[s * L.VR : (s + 1) * L.VR].float().t()
            )
            none = torch.isnan(z).all(dim=1)
            key[k, s] = torch.where(none, torch.full_like(order, -(1 << 31)), order)
            gid[k, s] = torch.where(
                none, torch.full_like(order, s * L.VR), packed[:, 1].long()
            )
        # the ranks' bests combine to production's pick
        top = key[k].max(dim=0).values
        pick = torch.where(key[k] == top[None, :], gid[k], L.VOCAB).min(dim=0).values
        if not torch.equal(pick.to(torch.int32), block[:, k + 1]):
            fail(
                f"host model: round {k} combines to {pick.tolist()}, production {block[:, k + 1].tolist()}"
            )
    return block[:, 1:].contiguous(), key, gid, base


class Fixture:
    def __init__(
        self, requests, drafts, rank, splits, wait=True, poll_ticks=None, diag=False
    ):
        self.requests, self.drafts, self.rank = requests, drafts, rank
        self.launch = MK.build_draft_markov(
            requests,
            drafts,
            RK,
            splits,
            wait=wait,
            sources=MARKOV_SOURCES,
            poll_ticks=poll_ticks or K.POLL_TICKS,
            diag=diag,
            **V1_BUILD,
        )
        z = torch.zeros
        self.x = z(requests * drafts, L.HIDDEN, dtype=torch.bfloat16, device=dev)
        self.anchors = z(requests, dtype=torch.int32, device=dev)
        self.cnt = z(L.CNT_WORDS, dtype=torch.int32, device=dev)
        self.rec = z(L.REC_BYTES, dtype=torch.uint8, device=dev)
        self.out = torch.full((requests, drafts), -7, dtype=torch.int32, device=dev)
        self.regions = region_bytes(L.TP * C.SPAN)
        base = self.regions.data_ptr()
        self.peers = torch.tensor(
            [base + o * C.SPAN + C.GUARD for o in range(L.TP)],
            dtype=torch.int64,
            device=dev,
        )
        self.tab = z(len(L.POINTERS), dtype=torch.int64, device=dev)
        # a diagnostic build's counters: [round][thread], [CTA], then the tag last seen [round][thread]
        self.diag = z(DIAG_WORDS, dtype=torch.int32, device=dev)
        self.tab[L.POINTERS.index("tap")] = self.diag.data_ptr()
        # a round's x_k, CTA 0 to the rank's other CTAs
        self.round_ids = z(L.MAX_REQ, dtype=torch.int32, device=dev)
        self.tab[L.POINTERS.index("mbest")] = self.round_ids.data_ptr()
        for name, tensor in (
            ("rec", self.rec),
            ("cnt", self.cnt),
            ("peers", self.peers),
            ("drafts", self.out),
        ):
            self.tab[L.POINTERS.index(name)] = tensor.data_ptr()
        C.KEEP.append(self)

    def refuse_null_pointers(self):
        """Every table entry the draft launch dereferences is set, or nothing is launched."""
        table = self.tab.tolist()
        missing = [
            name
            for name in (
                "w",
                "w1",
                "w2",
                "rec",
                "cnt",
                "peers",
                "drafts",
                "mbest",
                "tap",
            )
            if not table[L.POINTERS.index(name)]
        ]
        if missing:
            raise RuntimeError(f"null table pointers {missing}: not launching")

    def region(self, buffer, o):
        return buffer[o * C.SPAN + C.GUARD : o * C.SPAN + C.GUARD + L.PEER_BYTES].view(
            torch.int32
        )

    def fire(self):
        self.launch(
            self.tab.data_ptr(),
            self.x.data_ptr(),
            self.anchors.data_ptr(),
            self.rank,
            stream=torch.cuda.current_stream(),
        )

    def stage(self, w, w1, w2, xd, anchors, key, gid, omit=None):
        self.x.copy_(xd)
        self.anchors.copy_(anchors)
        for name, tensor in (("w", C.shard(w, self.rank)), ("w1", w1), ("w2", w2)):
            self.tab[L.POINTERS.index(name)] = tensor.data_ptr()
        epoch = int(self.cnt[M.CNT_MEPOCH]) + 1
        lines = torch.stack(
            [
                (-1 - gid).to(torch.int32),
                key.to(torch.int32),
                torch.zeros_like(key, dtype=torch.int32),
                torch.full_like(key, epoch).to(torch.int32),
            ],
            dim=3,
        )  # [drafts, TP, requests, 4]

        def place(buffer, o, source):
            words = self.region(buffer, o)
            at = torch.tensor(
                [
                    [
                        L.markov_line(epoch & 1, k, source, r) // 4
                        for r in range(self.requests)
                    ]
                    for k in range(self.drafts)
                ],
                device=dev,
            )
            for word in range(4):
                words[at + word] = lines[:, source, :, word]

        for source in range(L.TP):
            if source != self.rank:
                place(self.regions, self.rank, source)
        if omit is not None:
            at = L.markov_line(epoch & 1, omit[1], omit[0], omit[2]) // 4
            stale = torch.tensor([0, 0, 0, epoch - 2], dtype=torch.int32, device=dev)
            self.region(self.regions, self.rank)[at : at + 4] = stale
        self.refuse_null_pointers()
        expected = self.regions.clone()
        for o in range(
            L.TP
        ):  # the kernel's line of every round and request, to every peer
            if o != self.rank:
                place(expected, o, self.rank)
        return epoch, expected

    def check(self, what, epoch, expected, want):
        torch.cuda.synchronize()
        tag = f"b{self.requests} t{self.drafts} rank {self.rank} {what}"
        ok = True
        if not torch.equal(self.out, want):
            fail(f"{tag}: drafts {self.out.tolist()} want {want.tolist()}")
            ok = False
        if not torch.equal(self.regions, expected):
            at = (self.regions != expected).nonzero().view(-1)
            first = int(at[0])
            fail(
                f"{tag}: {at.numel()} peer bytes differ, first in region {first // C.SPAN} at "
                f"{first % C.SPAN - C.GUARD}"
            )
            ok = False
        c = self.cnt.tolist()
        rounds = c[L.CNT_MARKOV : L.CNT_MARKOV + self.drafts]
        if (
            c[L.CNT_TIMEOUT]
            or c[M.CNT_MEPOCH] != epoch
            or c[M.CNT_MFLAG] != epoch * 8 + self.drafts
            or rounds != [epoch * L.CTAS] * self.drafts
        ):
            fail(
                f"{tag}: timeouts {c[L.CNT_TIMEOUT]} (site {c[L.CNT_SITE]}), epoch {c[M.CNT_MEPOCH]} "
                f"of {epoch}, round counts {rounds}"
            )
            ok = False
        return ok


def inputs(requests, drafts):
    xd = bf16(requests * drafts, L.HIDDEN)
    anchors = torch.randint(0, L.VOCAB, (requests,), device=dev, generator=C.gen).to(
        torch.int32
    )
    return xd, anchors


def run_case(what, f, w, w1, w2, xd, anchors):
    want, key, gid, _ = production(w, w1, w2, xd, anchors, f.requests, f.drafts)
    epoch, expected = f.stage(w, w1, w2, xd, anchors, key, gid)
    f.fire()
    return f.check(what, epoch, expected, want), want


def check(requests, drafts, rank, splits, weights):
    f = Fixture(requests, drafts, rank, splits)
    passed = cases = 0
    for kind, (w, w1, w2) in weights.items():
        for it in range(ITERS):
            ok, want = run_case(f"{kind} {it}", f, w, w1, w2, *inputs(requests, drafts))
            passed, cases = passed + ok, cases + 1
        print(
            f"  b{requests} t{drafts} rank {rank} {kind}: request 0 drafts {want[0].tolist()}",
            flush=True,
        )
    w, w1, w2 = weights["random"]
    # NaN / inf hidden rows
    xd, anchors = inputs(requests, drafts)
    xd[0] = float("nan")
    if xd.shape[0] > 2:
        xd[2, 5] = float("inf")
    if xd.shape[0] > 3:
        xd[3, 77] = float("-inf")
    ok, _ = run_case("special x", f, w, w1, w2, xd, anchors)
    passed, cases = passed + ok, cases + 1
    # NaN / inf W2 rows on three ranks, an inf W1 row under request 0's anchor
    xd, anchors = inputs(requests, drafts)
    spots = (3 * L.VR + 100, 6 * L.VR + 7, 1 * L.VR + 20000, 0)
    saved = [w2[i].clone() for i in spots], w1[int(anchors[0])].clone()
    w2[spots[0]] = float("nan")
    w2[spots[1]] = float("nan")
    w2[spots[2], :16] = float("inf")
    w2[spots[3]] = float("-inf")
    w1[int(anchors[0]), :8] = float("inf")
    ok, _ = run_case("special w", f, w, w1, w2, xd, anchors)
    passed, cases = passed + ok, cases + 1
    for i, row in zip(spots, saved[0]):
        w2[i] = row
    w1[int(anchors[0])] = saved[1]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        f.fire()
    torch.cuda.synchronize()
    C.KEEP.append(graph)
    f.cnt.zero_()
    replayed = 0
    for it in range(REPLAYS):
        w, w1, w2 = weights["ties" if it % 2 else "random"]
        xd, anchors = inputs(requests, drafts)
        want, key, gid, _ = production(w, w1, w2, xd, anchors, requests, drafts)
        epoch, expected = f.stage(w, w1, w2, xd, anchors, key, gid)
        graph.replay()
        replayed += f.check(f"replay {it}", epoch, expected, want)
    print(
        f"check b{requests} t{drafts} rank {rank} splits {splits}: {passed}/{cases} cases, "
        f"{replayed}/{REPLAYS} replays",
        flush=True,
    )
    for res in launcher_resources(f.launch):
        print(f"  resources {res}", flush=True)


def give_up(requests, drafts, rank, splits, weights):
    w, w1, w2 = weights["random"]
    f = Fixture(requests, drafts, rank, splits, poll_ticks=200000)  # 2 ms
    with compile_only():
        f.launch(0, 0, 0, 0)
    xd, anchors = inputs(requests, drafts)
    want, key, gid, _ = production(w, w1, w2, xd, anchors, requests, drafts)
    epoch, _ = f.stage(w, w1, w2, xd, anchors, key, gid, omit=((rank + 1) % L.TP, 0, 0))
    f.fire()
    torch.cuda.synchronize()
    c = f.cnt.tolist()
    flags = {
        # CTA 0 alone polls the peers' lines: its one lane gives up
        "one wait gave up at the poll": c[L.CNT_TIMEOUT] == 1
        and c[L.CNT_SITE] == K.SITE_MARKOV,
        "counters": c[M.CNT_MEPOCH] == epoch
        and c[L.CNT_MARKOV : L.CNT_MARKOV + drafts] == [epoch * L.CTAS] * drafts,
        "other requests": torch.equal(f.out[1:], want[1:]),
        "ids inside the vocabulary": bool(((f.out >= 0) & (f.out < L.VOCAB)).all()),
        "guards": bool((f.regions.view(L.TP, C.SPAN)[:, : C.GUARD] == C.PATTERN).all())
        and bool(
            (
                f.regions.view(L.TP, C.SPAN)[:, C.GUARD + L.PEER_BYTES :] == C.PATTERN
            ).all()
        ),
    }
    if not all(flags.values()):
        fail(f"give-up b{requests} t{drafts} rank {rank}: {flags} counters {c[:6]}")
    f.cnt[L.CNT_TIMEOUT] = 0
    again, _ = run_case("after a give-up", f, w, w1, w2, *inputs(requests, drafts))
    print(
        f"give-up b{requests} t{drafts} rank {rank}: one line withheld -> {c[L.CNT_TIMEOUT]} waits gave up "
        f"(site {c[L.CNT_SITE]}), launch ended: {all(flags.values())}; next launch whole: {again}",
        flush=True,
    )


def spans(values):
    values, out, i = sorted(values), [], 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[j + 1] == values[j] + 1:
            j += 1
        out.append(str(values[i]) if i == j else f"{values[i]}-{values[j]}")
        i = j + 1
    return ",".join(out)


def region_bytes(nbytes):
    """The ranks' regions with their guard bands: torch's memory, or what a
    runner's peer buffers are made of (``atom.mono.runtime.uncached``: never
    freed)."""
    if REGIONS == "plain":
        return torch.full((nbytes,), C.PATTERN, dtype=torch.uint8, device=dev)
    from atom.mono.runtime import uncached

    make = (
        uncached.uncached_bytes
        if REGIONS == "uncached"
        else uncached.fine_grained_bytes
    )
    out = make(nbytes, dev)
    if out.numel() != nbytes or not out.data_ptr():
        raise RuntimeError(f"{REGIONS} regions: {out.numel()} bytes: not launching")
    out.fill_(C.PATTERN)
    return out


def cta_keys(b, hw):
    """What could tell CTAs apart: a CTA's place in the grid, and its HW_ID1's
    fields (gfx11's layout; the unnamed bits too)."""
    return {
        "cta%8": b % 8,
        "cta%4": b % 4,
        "cta//32": b // 32,
        "se": (hw >> 26) & 7,
        "sh": (hw >> 20) & 1,
        "cu": (hw >> 16) & 15,
        "simd": (hw >> 12) & 3,
        "bits29-31": (hw >> 29) & 7,
        "bits21-25": (hw >> 21) & 31,
        "bits14-15": (hw >> 14) & 3,
        "bits10-11": (hw >> 10) & 3,
    }


def diagnose(requests, drafts, rank, splits, weights, launches):
    """Launch after launch of a diagnostic build (eager, then replays of one
    recording), each with its own records: which threads' polls gave up, in
    which round and CTAs, on what tag; for the first design also what a
    volatile re-read of the line then carried and what tells the CTAs that saw
    the lines from the others (grid place, HW_ID1)."""
    f = Fixture(requests, drafts, rank, splits, diag=True, poll_ticks=TICKS or None)
    v1 = DESIGN == "v1"
    if f.diag.numel() < DIAG_WORDS or (v1 and DIAG_WORDS != MK.DG_WORDS):
        raise RuntimeError("the diagnostic buffer is not the build's: not launching")
    w, w1, w2 = weights["random"]
    graph, bad, rounds_bad, cta0, gave_up, fresh = None, 0, 0, 0, 0, 0
    ctas_at = L.MAX_DRAFTS * L.THREADS
    tags_at = ctas_at + L.CTAS
    threads, again_tags, separates = set(), set(), {}
    started = time.perf_counter()
    for it in range(2 * launches):
        if it == launches:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                f.fire()
            torch.cuda.synchronize()
            C.KEEP.append(graph)
        if (
            it % 100 == 0
        ):  # new inputs now and then; the host's lines every launch (they carry the epoch)
            xd, anchors = inputs(requests, drafts)
            want, key, gid, _ = production(w, w1, w2, xd, anchors, requests, drafts)
        f.stage(w, w1, w2, xd, anchors, key, gid)
        f.cnt[L.CNT_TIMEOUT] = 0
        f.diag.zero_()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        if graph is None:
            f.fire()
        else:
            graph.replay()
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1e3
        c = f.cnt.tolist()
        right = torch.equal(f.out, want)
        # with fewer CTAs polling the others' ids are wrong by construction: the polls alone count
        if not c[L.CNT_TIMEOUT] and (right or POLLERS < L.CTAS):
            continue
        bad += 1
        d = f.diag.tolist()
        stale = {b for b in range(L.CTAS) if d[ctas_at + b]}
        fine = [b for b in range(L.CTAS) if b not in stale]
        cta0 += 0 in stale
        hit_rounds = []
        for k in range(drafts):
            hit = {
                t: d[k * L.THREADS + t]
                for t in range(L.THREADS)
                if d[k * L.THREADS + t]
            }
            if not hit:
                continue
            threads |= set(hit)
            tags = sorted({d[tags_at + k * L.THREADS + t] for t in hit})
            hit_rounds.append(
                (k, spans(hit), tags, min(hit.values()), max(hit.values()))
            )
            gave_up += sum(hit.values())
            if v1:
                fresh += sum(d[MK.DG_FRESH + k * L.THREADS + t] for t in hit)
                again_tags |= {d[MK.DG_AGAIN + k * L.THREADS + t] for t in hit}
        rounds_bad += len(hit_rounds)
        table = {}
        if v1 and stale and fine:
            for b in range(L.CTAS):
                for name, value in cta_keys(b, d[MK.DG_HW + b] & 0xFFFFFFFF).items():
                    table.setdefault(name, {}).setdefault(value, [0, 0])[
                        b in stale
                    ] += 1
            for name, values in table.items():
                # a key separates when none of its values has CTAs of both kinds
                if all(not (saw and missed) for saw, missed in values.values()):
                    separates[name] = separates.get(name, 0) + 1
        if bad <= 4:
            print(
                f"  diag b{requests} t{drafts} rank {rank} {'replay' if graph else 'eager'} {it}: {ms:.0f} ms, "
                f"{c[L.CNT_TIMEOUT]} waits gave up in {len(stale)} CTAs (CTA 0 {'too' if 0 in stale else 'not'}), "
                f"ids {'right' if right else 'wrong'}; (round, threads, last tag - epoch, CTAs a thread "
                f"least, most): {hit_rounds}",
                flush=True,
            )
            if table:
                print(f"    the CTAs that saw the lines: {spans(fine)}", flush=True)
                print(
                    "    a key's values, CTAs that saw the lines / that did not: "
                    + "; ".join(
                        f"{name} "
                        + ",".join(
                            f"{v}:{saw}/{missed}"
                            for v, (saw, missed) in sorted(values.items())
                        )
                        for name, values in table.items()
                    ),
                    flush=True,
                )
    own = sorted(t for t in threads if t % L.TP == rank)
    print(
        f"diag b{requests} t{drafts} rank {rank} design {DESIGN or 'served'} regions {REGIONS} knobs {V1_BUILD} "
        f"bound {(TICKS or K.POLL_TICKS) / 1e5:.0f} ms: {bad}/{2 * launches} launches with a wait that gave up or "
        f"a wrong id ({rounds_bad} rounds, {gave_up} waits; CTA 0 among them in {cta0} launches); threads that ever "
        f"gave up {spans(threads)} (reading this rank's own lines: {spans(own)}; waves "
        f"{spans({t // 32 for t in threads})}); a volatile re-read after giving up carried the epoch in {fresh} of "
        f"{gave_up} (its tags - epoch {sorted(again_tags)}); keys that separate the CTAs that saw the lines from "
        f"the others, in launches: {separates}; {time.perf_counter() - started:.0f} s",
        flush=True,
    )


def bench(requests, drafts, rank, splits, weights):
    w, w1, w2 = weights["random"]
    f = Fixture(requests, drafts, rank, splits, wait=False)
    xd, anchors = inputs(requests, drafts)
    _, key, gid, base = production(w, w1, w2, xd, anchors, requests, drafts)
    f.stage(w, w1, w2, xd, anchors, key, gid)
    mine = C.shard(w, rank)
    block = torch.zeros(requests, drafts + 1, dtype=torch.int32, device=dev)
    block[:, 0] = anchors
    C.KEEP.extend([xd, base, block])

    def markov():
        for k in range(drafts):
            dspark_markov_argmax(base[:, k], block[:, k], w1, w2, block[:, k + 1])

    def both():
        tgemm.mm(xd, mine, None)
        markov()

    graphs = {}
    for name, fn in (("production", both), ("markov", markov), ("kernel", f.fire)):
        fn()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(REPS):
                fn()
        graphs[name] = graph
    C.KEEP.append(graphs)
    times = {name: [] for name in graphs}
    for _ in range(ROUNDS):
        for name in (
            "production",
            "kernel",
            "markov",
            "markov",
            "kernel",
            "production",
        ):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            graphs[name].replay()
            torch.cuda.synchronize()
            times[name].append((time.perf_counter() - t0) / REPS * 1e6)
    med = {name: statistics.median(v) for name, v in times.items()}
    print(
        f"bench b{requests} t{drafts} rank {rank}: kernel {med['kernel']:.1f} us, production draft GEMM + "
        f"{drafts} Markov positions {med['production']:.1f} us (Markov alone {med['markov']:.1f}); no "
        f"all-gather on one GPU; timeouts {int(f.cnt[L.CNT_TIMEOUT])}",
        flush=True,
    )


def main():
    pick = torch.randint(0, 64, (L.VOCAB,), device=dev, generator=C.gen)
    pick2 = torch.randint(0, 32, (L.VOCAB,), device=dev, generator=C.gen)
    weights = {
        "random": (
            bf16(L.VOCAB, L.HIDDEN, scale=0.02),
            bf16(L.VOCAB, RK, scale=0.25),
            bf16(L.VOCAB, RK, scale=0.25),
        ),
        "ties": (
            bf16(64, L.HIDDEN, scale=0.02)[pick].contiguous(),
            bf16(L.VOCAB, RK, scale=0.25),
            bf16(32, RK, scale=0.25)[pick2].contiguous(),
        ),
    }
    C.KEEP.append(weights)
    for requests, drafts in SHAPES:
        splits = head_dispatch.head_splits(requests * drafts)
        for rank in RANKS:
            if CHECK:
                check(requests, drafts, rank, splits, weights)
        if DIAG:
            diagnose(requests, drafts, RANKS[0], splits, weights, DIAG)
        if GIVEUP and requests >= 2:
            give_up(requests, drafts, RANKS[0], splits, weights)
        if BENCH:
            bench(requests, drafts, RANKS[0], splits, weights)
    print("FAILED" if C.FAILED else "PASSED", len(C.FAILED), flush=True)


if __name__ == "__main__":
    main()
