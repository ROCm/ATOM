# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The step tail's head + verify kernel on one GPU, built for the real TP8
shape, against the production step tail on the same inputs.

The GPU is one TP rank; the host plays the other seven: every rank's peer
region is a local buffer between guard bands, the peers' argmax lines are
placed there with the launch's epoch (from production's logits of their vocab
shards), and after the launch every byte of every region is compared with what
the kernel may write. Production: ``tgemm.mm`` on each rank's shard (the
all-gather is a concatenation), ``topk_select`` and ``rejection_sample``.

Checked: the output token ids and the verdict; the pushed lines and the guard
bands; a check build's bf16 logits against production's, bit for bit; no poll
gave up. Inputs: random, heavy ties (64 distinct LM head rows: every maximum
is attained by thousands of ids on every rank and tile), NaN / inf / zero rows
and NaN / inf head rows; then REPLAYS CUDA-graph replays. BENCH: CUDA-graph
time against the production kernels it replaces (a build that does not wait
for peers: one GPU has none).

    python tools/steptail_check.py
    env: ROWS (8,64), QLEN (8), RANKS (0,5), ITERS (3), REPLAYS (20),
    CHECK (1), BENCH (1), TIMELINE (1), GIVEUP (1), REPS (16), ROUNDS (12),
    DIAG (0: launches of the serving build to run one by one, eager then
    replayed, counting those with a wait that gave up or a wrong output)
"""

import os
import statistics
import time

import torch
from aiter import topk_select
from aiter.tuned_gemm import tgemm

from atom.model_ops.rejection_sampler import rejection_sample
from atom.models.kimi_k3.step_tail import head_dispatch
from atom.models.kimi_k3.step_tail import kernels as K
from atom.models.kimi_k3.step_tail import layout as L
from atom.models.kimi_k3.step_tail.sources import SOURCES
from atom.mono.runtime.compile import compile_only, launcher_resources

ROWS = [int(r) for r in os.environ.get("ROWS", "8,64").split(",")]
QLEN = int(os.environ.get("QLEN", "8"))
RANKS = [int(r) for r in os.environ.get("RANKS", "0,5").split(",")]
ITERS = int(os.environ.get("ITERS", "3"))
REPLAYS = int(os.environ.get("REPLAYS", "20"))
CHECK = os.environ.get("CHECK", "1") == "1"
BENCH = os.environ.get("BENCH", "1") == "1"
TIMELINE = os.environ.get("TIMELINE", "1") == "1"
GIVEUP = os.environ.get("GIVEUP", "1") == "1"
DIAG = int(os.environ.get("DIAG", "0"))
REPS = int(os.environ.get("REPS", "16"))
ROUNDS = int(os.environ.get("ROUNDS", "12"))
GUARD = 4096
SPAN = L.PEER_BYTES + 2 * GUARD
PATTERN = 0xA5
KEEP = []
FAILED = []

dev = torch.device("cuda", torch.cuda.current_device())
gen = torch.Generator(device=dev).manual_seed(7)


def bf16(*shape, scale=1.0):
    return (torch.randn(shape, generator=gen, device=dev) * scale).to(torch.bfloat16)


def fail(what):
    FAILED.append(what)
    print("FAIL", what, flush=True)


def shard(w, rank):
    return w[rank * L.VR : (rank + 1) * L.VR]


def production_logits(w, x):
    """Every rank's LM head GEMM and the all-gather: [rows, VOCAB] bf16."""
    return torch.cat(
        [tgemm.mm(x, shard(w, r), None) for r in range(L.TP)], dim=1
    ).contiguous()


def step_indices(rows, qlen):
    bs, k = rows // qlen, qlen - 1
    t_idx = torch.tensor(
        [r * qlen + p for r in range(bs) for p in range(k)], device=dev
    )
    b_idx = torch.tensor([r * qlen + k for r in range(bs)], device=dev)
    cu = torch.arange(1, bs + 1, device=dev, dtype=torch.int32) * k
    return t_idx, b_idx, cu


def production_verify(logits, ids, qlen):
    """``ModelRunner.postprocess`` on a greedy verify step."""
    t_idx, b_idx, cu = step_indices(logits.shape[0], qlen)
    bonus = torch.index_select(logits, 0, b_idx)
    target = torch.index_select(logits, 0, t_idx)
    bonus_ids = topk_select(bonus, 1, tie="low")[1].view(-1)
    return rejection_sample(ids, t_idx, qlen - 1, cu, None, target, bonus_ids, b_idx)


def draft_ids(logits, qlen, salt):
    """The step's input ids: each request's drafts match production's argmax
    up to a position that varies with the request and ``salt``."""
    rows = logits.shape[0]
    k = qlen - 1
    arg = topk_select(logits, 1, tie="low")[1].view(-1).to(torch.int32)
    ids = torch.randint(0, L.VOCAB, (rows,), device=dev, generator=gen).to(torch.int32)
    for q in range(rows // qlen):
        accepted = (k, 0, k - 1, 1, k // 2, k, 2, k)[(q + salt) % 8]
        for p in range(k):
            want = arg[q * qlen + p]
            ids[q * qlen + p + 1] = want if p < accepted else (want + 1 + q) % L.VOCAB
    return ids


def host_bests(logits):
    """Each rank's (key, global id) of each row, as its kernel reduces them:
    [TP, rows] int64 each."""
    rows = logits.shape[0]
    bits = logits.float().contiguous().view(torch.int32).to(torch.int64)
    bits = torch.where(bits == -(1 << 31), torch.zeros_like(bits), bits)
    key = bits ^ ((bits >> 31) & 0x7FFFFFFF)
    nan = (bits & 0x7FFFFFFF) > 0x7F800000
    key = torch.where(nan, torch.full_like(key, 0x7FFFFFFF), key).view(rows, L.TP, L.VR)
    best = key.max(dim=2).values
    cols = torch.arange(L.VR, device=dev)
    local = torch.where(key == best[:, :, None], cols, L.VR).min(dim=2).values
    gid = local + torch.arange(L.TP, device=dev)[None, :] * L.VR
    return best.t().contiguous(), gid.t().contiguous()


class Fixture:
    """One rank's launch, its buffers and the eight peer regions."""

    def __init__(
        self,
        rows,
        qlen,
        rank,
        splits,
        tap=False,
        wait=True,
        stamps=False,
        poll_ticks=None,
    ):
        self.rows, self.qlen, self.rank, self.tapped = rows, qlen, rank, tap
        self.launch = K.build_head_verify(
            rows,
            qlen,
            splits,
            tap=tap,
            wait=wait,
            sources=SOURCES,
            stamps=stamps,
            poll_ticks=poll_ticks or K.POLL_TICKS,
        )
        bs = rows // qlen
        z = torch.zeros
        self.x = z(rows, L.HIDDEN, dtype=torch.bfloat16, device=dev)
        self.ids = z(rows, dtype=torch.int32, device=dev)
        self.cnt = z(L.CNT_WORDS, dtype=torch.int32, device=dev)
        self.rec = z(L.REC_BYTES, dtype=torch.uint8, device=dev)
        self.out = torch.full((bs, qlen), -7, dtype=torch.int32, device=dev)
        self.verdict = torch.full((3, bs), -7, dtype=torch.int32, device=dev)
        self.tap = z(rows, L.VR, dtype=torch.bfloat16, device=dev)
        self.stamps = z(L.MAX_REQ * 8, dtype=torch.int64, device=dev)
        self.regions = torch.full(
            (L.TP * SPAN,), PATTERN, dtype=torch.uint8, device=dev
        )
        base = self.regions.data_ptr()
        self.peers = torch.tensor(
            [base + o * SPAN + GUARD for o in range(L.TP)],
            dtype=torch.int64,
            device=dev,
        )
        self.tab = z(len(L.POINTERS), dtype=torch.int64, device=dev)
        for name, tensor in (
            ("rec", self.rec),
            ("cnt", self.cnt),
            ("peers", self.peers),
            ("outtok", self.out),
            ("verdict", self.verdict),
            ("tap", self.tap),
            ("stamps", self.stamps),
        ):
            self.tab[L.POINTERS.index(name)] = tensor.data_ptr()
        KEEP.append(self)

    def refuse_null_pointers(self):
        """Every table entry the launch dereferences is set, or nothing is launched."""
        table = self.tab.tolist()
        missing = [
            name
            for name in (
                "w",
                "rec",
                "cnt",
                "peers",
                "outtok",
                "verdict",
                "tap",
                "stamps",
            )
            if not table[L.POINTERS.index(name)]
        ]
        if missing:
            raise RuntimeError(f"null table pointers {missing}: not launching")

    def region(self, buffer, o):
        return buffer[o * SPAN + GUARD : o * SPAN + GUARD + L.PEER_BYTES].view(
            torch.int32
        )

    def fire(self):
        self.launch(
            self.tab.data_ptr(),
            self.x.data_ptr(),
            self.ids.data_ptr(),
            self.rank,
            stream=torch.cuda.current_stream(),
        )

    def stage(self, w, x, ids, logits, omit=None):
        """The step's inputs, and the peers' lines of the next launch; returns
        what the regions must hold after it."""
        self.x.copy_(x)
        self.ids.copy_(ids)
        self.tab[L.POINTERS.index("w")] = shard(w, self.rank).data_ptr()
        epoch = int(self.cnt[L.CNT_EPOCH]) + 1
        key, gid = host_bests(logits)
        lines = torch.stack(
            [
                (-1 - gid).to(torch.int32),
                key.to(torch.int32),
                torch.zeros_like(key, dtype=torch.int32),
                torch.full_like(key, epoch).to(torch.int32),
            ],
            dim=2,
        )  # [TP, rows, 4]

        def place(buffer, o, source):
            words = self.region(buffer, o)
            at = torch.tensor(
                [L.sample_line(epoch & 1, source, r) // 4 for r in range(self.rows)],
                device=dev,
            )
            for word in range(4):
                words[at + word] = lines[source, :, word]

        for source in range(L.TP):
            if source != self.rank:
                place(self.regions, self.rank, source)
        if omit is not None:
            # a line that never arrives: (source, row) keeps a stale tag
            at = L.sample_line(epoch & 1, omit[0], omit[1]) // 4
            stale = torch.tensor([0, 0, 0, epoch - 2], dtype=torch.int32, device=dev)
            self.region(self.regions, self.rank)[at : at + 4] = stale
        self.refuse_null_pointers()
        expected = self.regions.clone()
        for o in range(L.TP):
            if o != self.rank:
                place(expected, o, self.rank)
        return epoch, expected

    def check(self, what, epoch, expected, logits, out, verdict):
        torch.cuda.synchronize()
        tag = f"s{self.rows} q{self.qlen} rank {self.rank}{' tap' if self.tapped else ''} {what}"
        ok = True
        if not torch.equal(self.out, out):
            bad = (self.out != out).nonzero()[:4].tolist()
            fail(
                f"{tag}: ids {bad} got {self.out.flatten()[:16].tolist()} want {out.flatten()[:16].tolist()}"
            )
            ok = False
        if not torch.equal(self.verdict, verdict):
            fail(f"{tag}: verdict {self.verdict.tolist()} want {verdict.tolist()}")
            ok = False
        if not torch.equal(self.regions, expected):
            at = (self.regions != expected).nonzero().view(-1)
            first = int(at[0])
            fail(
                f"{tag}: {at.numel()} peer bytes differ, first in region {first // SPAN} at "
                f"{first % SPAN - GUARD}"
            )
            ok = False
        counters = self.cnt.tolist()
        if counters[L.CNT_TIMEOUT]:
            fail(
                f"{tag}: {counters[L.CNT_TIMEOUT]} polls gave up (site {counters[L.CNT_SITE]})"
            )
            ok = False
        if counters[L.CNT_EPOCH] != epoch or counters[L.CNT_HEAD] != epoch * L.CTAS:
            fail(f"{tag}: counters {counters[:4]} at epoch {epoch}")
            ok = False
        if self.tapped:
            want = shard_columns(logits, self.rank).view(torch.int16)
            got = self.tap.view(torch.int16)
            diff = got != want
            if diff.any():
                zeros = ((got & 0x7FFF) == 0) & ((want & 0x7FFF) == 0)
                real = diff & ~zeros
                err = (
                    self.tap.float() - shard_columns(logits, self.rank).float()
                ).abs()
                fail(
                    f"{tag}: {int(real.sum())} logits differ (+ {int((diff & zeros).sum())} "
                    f"signed zeros), max abs {float(err[real].nan_to_num(0).max()) if real.any() else 0.0}"
                )
                ok = False
        return ok


def shard_columns(logits, rank):
    return logits[:, rank * L.VR : (rank + 1) * L.VR].contiguous()


def special_x(rows):
    x = bf16(rows, L.HIDDEN)
    x[0] = float("nan")
    if rows > 1:
        x[1, 5] = float("inf")
    if rows > 2:
        x[2, 77] = float("-inf")
    if rows > 3:
        x[3] = 0.0
    if rows > 4:
        x[4, :64] = 3.0e38
    return x


def run_case(what, fixtures, w, x, salt):
    logits = production_logits(w, x)
    ids = draft_ids(logits, fixtures[0].qlen, salt)
    out, verdict = production_verify(logits, ids, fixtures[0].qlen)
    ok = True
    for f in fixtures:
        epoch, expected = f.stage(w, x, ids, logits)
        f.fire()
        ok &= f.check(what, epoch, expected, logits, out, verdict)
    return ok, verdict[0].tolist()


def check(rows, rank, splits, weights):
    fixtures = [
        Fixture(rows, QLEN, rank, splits, tap=True),
        Fixture(rows, QLEN, rank, splits),
    ]
    passed = cases = 0
    for kind, w in weights.items():
        for it in range(ITERS):
            ok, accepted = run_case(
                f"{kind} {it}", fixtures, w, bf16(rows, L.HIDDEN), it
            )
            passed, cases = passed + ok, cases + 1
        print(f"  s{rows} rank {rank} {kind}: accepted drafts {accepted}", flush=True)
    # NaN / inf / zero rows, then NaN / inf head rows on three ranks
    w = weights["random"]
    ok, _ = run_case("special x", fixtures, w, special_x(rows), 1)
    passed, cases = passed + ok, cases + 1
    spots = (3 * L.VR + 100, 6 * L.VR + 7, 1 * L.VR + 20000, 2 * L.VR + 9)
    saved = [w[i].clone() for i in spots]
    w[spots[0]] = float("nan")
    w[spots[1]] = float("nan")
    w[spots[2], :16] = float("inf")
    w[spots[3]] = float("-inf")
    ok, _ = run_case("special w", fixtures, w, bf16(rows, L.HIDDEN), 2)
    passed, cases = passed + ok, cases + 1
    for i, row in zip(spots, saved):
        w[i] = row
    # CUDA-graph replays of the serving build
    f = fixtures[1]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        f.fire()
    torch.cuda.synchronize()
    KEEP.append(graph)
    f.cnt.zero_()
    f.cnt[L.CNT_EPOCH] = 1  # the capture ran one launch on other counters' terms
    f.cnt[L.CNT_HEAD] = L.CTAS
    replayed = 0
    for it in range(REPLAYS):
        w = weights["ties" if it % 2 else "random"]
        x = bf16(rows, L.HIDDEN)
        logits = production_logits(w, x)
        ids = draft_ids(logits, QLEN, it)
        out, verdict = production_verify(logits, ids, QLEN)
        epoch, expected = f.stage(w, x, ids, logits)
        graph.replay()
        replayed += f.check(f"replay {it}", epoch, expected, logits, out, verdict)
    print(
        f"check s{rows} q{QLEN} rank {rank} splits {splits}: {passed}/{cases} cases, "
        f"{replayed}/{REPLAYS} replays",
        flush=True,
    )
    for f in fixtures:
        for res in launcher_resources(f.launch):
            print(f"  resources {res}", flush=True)


def soak(rows, rank, splits, w, launches):
    """The serving build launch after launch (eager, then replays of one
    recording): the launches with a wait that gave up or with ids or a
    verdict that are not production's."""
    f = Fixture(rows, QLEN, rank, splits)
    graph, bad, waits = None, 0, 0
    started = time.perf_counter()
    for it in range(2 * launches):
        if it % 100 == 0 or it == launches:
            x = bf16(rows, L.HIDDEN)
            logits = production_logits(w, x)
            ids = draft_ids(logits, QLEN, it // 100)
            out, verdict = production_verify(logits, ids, QLEN)
        if it == launches:
            f.stage(
                w, x, ids, logits
            )  # a capture may run the launch: its epoch's lines too
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                f.fire()
            torch.cuda.synchronize()
            KEEP.append(graph)
        f.stage(w, x, ids, logits)
        f.cnt[L.CNT_TIMEOUT] = 0
        if graph is None:
            f.fire()
        else:
            graph.replay()
        torch.cuda.synchronize()
        gave_up = int(f.cnt[L.CNT_TIMEOUT])
        if gave_up or not (torch.equal(f.out, out) and torch.equal(f.verdict, verdict)):
            bad, waits = bad + 1, waits + gave_up
    print(
        f"diag s{rows} q{QLEN} rank {rank}: {bad}/{2 * launches} launches with a wait that gave up or a wrong "
        f"output ({waits} waits); {time.perf_counter() - started:.0f} s",
        flush=True,
    )


def bench(rows, rank, splits, w):
    f = Fixture(rows, QLEN, rank, splits, wait=False)
    x = bf16(rows, L.HIDDEN)
    logits = production_logits(w, x)
    ids = draft_ids(logits, QLEN, 0)
    f.stage(w, x, ids, logits)
    t_idx, b_idx, cu = step_indices(rows, QLEN)
    mine = shard(w, rank)
    KEEP.extend([x, logits, ids, t_idx, b_idx, cu])

    def production():
        tgemm.mm(x, mine, None)
        bonus = torch.index_select(logits, 0, b_idx)
        target = torch.index_select(logits, 0, t_idx)
        bonus_ids = topk_select(bonus, 1, tie="low")[1].view(-1)
        rejection_sample(ids, t_idx, QLEN - 1, cu, None, target, bonus_ids, b_idx)

    def head_only():
        tgemm.mm(x, mine, None)

    graphs = {}
    for name, fn in (
        ("production", production),
        ("head", head_only),
        ("kernel", f.fire),
    ):
        fn()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(REPS):
                fn()
        graphs[name] = graph
    KEEP.append(graphs)
    times = {name: [] for name in graphs}
    order = ("production", "kernel", "head", "head", "kernel", "production")
    for _ in range(ROUNDS):
        for name in order:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            graphs[name].replay()
            torch.cuda.synchronize()
            times[name].append((time.perf_counter() - t0) / REPS * 1e6)
    med = {name: statistics.median(v) for name, v in times.items()}
    print(
        f"bench s{rows} q{QLEN} rank {rank}: kernel {med['kernel']:.1f} us, production GEMM "
        f"{med['head']:.1f} us, production GEMM + select + argmax + verify {med['production']:.1f} us "
        f"(no all-gather, no broadcast: one GPU); timeouts {int(f.cnt[L.CNT_TIMEOUT])}",
        flush=True,
    )


def timeline(rows, rank, splits, w):
    """Where a verify CTA's time goes, the peers' lines already in place (the
    poll is then one round: its memory round trip, no fabric)."""
    f = Fixture(rows, QLEN, rank, splits, stamps=True)
    samples = []
    for it in range(30):
        x = bf16(rows, L.HIDDEN)
        logits = production_logits(w, x)
        f.stage(w, x, draft_ids(logits, QLEN, it), logits)
        f.fire()
        torch.cuda.synchronize()
        st = f.stamps.view(L.MAX_REQ, 8)[: rows // QLEN, :6].double()
        samples.append((st[:, 1:] - st[:, :-1]) * 0.01)
    d = torch.cat(samples).median(dim=0).values.tolist()
    print(
        f"timeline s{rows} q{QLEN} rank {rank} (verify CTAs, us, 10 ns clock): head {d[0]:.2f}, "
        f"hand-off {d[1]:.2f}, reduce + push {d[2]:.2f}, poll {d[3]:.2f}, combine + verdict "
        f"{d[4]:.2f}; timeouts {int(f.cnt[L.CNT_TIMEOUT])}",
        flush=True,
    )


def give_up(rows, rank, splits, w):
    """A peer's line that never arrives: the bounded poll gives up, says so,
    and the launch ends; the next launch is whole again."""
    f = Fixture(rows, QLEN, rank, splits, poll_ticks=200000)  # 2 ms
    x = bf16(rows, L.HIDDEN)
    logits = production_logits(w, x)
    ids = draft_ids(logits, QLEN, 3)
    out, verdict = production_verify(logits, ids, QLEN)
    with compile_only():
        f.launch(0, 0, 0, 0)
    epoch, expected = f.stage(w, x, ids, logits, omit=((rank + 1) % L.TP, 0))
    t0 = time.perf_counter()
    f.fire()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1e3
    c = f.cnt.tolist()
    flags = {
        "one wait gave up at the poll": c[L.CNT_TIMEOUT] == 1
        and c[L.CNT_SITE] == K.SITE_SAMPLE,
        "counters": c[L.CNT_EPOCH] == epoch and c[L.CNT_HEAD] == epoch * L.CTAS,
        "regions and guards": torch.equal(f.regions, expected),
        # the requests whose lines all arrived are right
        "other requests": torch.equal(f.out[1:], out[1:])
        and torch.equal(f.verdict[:, 1:], verdict[:, 1:]),
    }
    ok = all(flags.values())
    if not ok:
        fail(f"give-up s{rows} rank {rank}: {flags} counters {c[:4]}")
    f.cnt[L.CNT_TIMEOUT] = 0
    x = bf16(rows, L.HIDDEN)
    logits = production_logits(w, x)
    ids = draft_ids(logits, QLEN, 4)
    out, verdict = production_verify(logits, ids, QLEN)
    epoch, expected = f.stage(w, x, ids, logits)
    f.fire()
    again = f.check("after a give-up", epoch, expected, logits, out, verdict)
    print(
        f"give-up s{rows} q{QLEN} rank {rank}: one line missing -> {c[L.CNT_TIMEOUT]} wait gave up "
        f"(site {c[L.CNT_SITE]}) in {ms:.1f} ms, launch ended, other requests right: {ok}; "
        f"next launch whole: {again}",
        flush=True,
    )


def main():
    weights = {"random": bf16(L.VOCAB, L.HIDDEN, scale=0.02)}
    patterns = bf16(64, L.HIDDEN, scale=0.02)
    pick = torch.randint(0, 64, (L.VOCAB,), device=dev, generator=gen)
    weights["ties"] = patterns[pick].contiguous()
    KEEP.append(weights)
    for rows in ROWS:
        splits = head_dispatch.head_splits(rows)
        print(f"s{rows}: production LM head splits at K steps {splits}", flush=True)
        for rank in RANKS:
            if CHECK:
                check(rows, rank, splits, weights)
        if DIAG:
            soak(rows, RANKS[0], splits, weights["random"], DIAG)
        if TIMELINE:
            timeline(rows, RANKS[0], splits, weights["random"])
        if GIVEUP and rows >= 2 * QLEN:
            give_up(rows, RANKS[0], splits, weights["random"])
        if BENCH:
            bench(rows, RANKS[0], splits, weights["random"])
    print("FAILED" if FAILED else "PASSED", len(FAILED), flush=True)


if __name__ == "__main__":
    main()
