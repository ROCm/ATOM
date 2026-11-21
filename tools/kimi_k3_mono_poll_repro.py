# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Do polled 16 B epoch-tagged lines ever stay stale? One GPU, tool only.

A resident-shaped kernel (256 CTAs x 256 threads): writer CTAs store tagged
lines with the all-reduce's own store (``k2_allreduce_gfx1250._store_line_sys``)
into a bank that alternates by epoch parity as the all-reduce's does (its
epoch counters too); polling threads of every CTA load them with the
all-reduce's own ``_load_lines`` until every tag is the launch's, bounded.
What varies (``Probe``): the loads (SCOPE_SYS | VOLATILE, a wait a load:
today's polls; SCOPE_SYS alone, a round's loads together), the pause between
polls (``spin_pause``'s side-effecting asm, or the ``s_sleep`` intrinsic) and
the bound (a count, or the 100 MHz clock asked every round), lines a thread
and rounds a launch, how many threads poll and how they are spread over the
waves, the writers (CTA 0 alone or 16 CTAs, every CTA polling their lines;
or every CTA its own lines, polled by its partner CTA alone: the
all-reduces' shape), copies a line is stored to,
stores of the polling threads' own just before a poll, a device hand-off of
every CTA ahead of the writes, launches between rewrites of an address (2:
banks by parity; 1), how late the writers store, and at run time the bank's
memory and eager launches against graph replays.

Shape v1 (SWEEP=v1, with SLOTS=1) is the sampling kernel's first design as
it failed: its region layout, a lane a line of every source (only this
rank's written in the launch, by CTA 0), the other sources' lines placed by
the host before every launch, every launch awaited.

The bank's memory (BUFFERS) is the variable a stale read is most likely a
property of: ``plain`` is torch's ordinary (cached) device memory, what the
sampling kernel's fixture and every emulated-rank tool here poll in;
``uncached`` and ``fine`` are what a runner's peer buffers are made of
(``atom.mono.runtime.uncached``); a torch.symm_mem MORI buffer, the
serving banks' own allocation, is NOT among them: nothing of symm_mem is
touched on this node until the platform is fixed (``symm`` is refused).

Per build: the poll loops as compiled (``atom.utils.poll_isa``): whether the
loads are inside the loop, back to back, and the loop's disassembly. Per
configuration: launches in which any poll hit its bound, the CTAs affected
(by ``bid % 8`` and by HW_ID1), the stale tags' age in launches, polls until
seen, payload mismatches, the writers' lines as the host reads them back.

    python tools/kimi_k3_mono_poll_repro.py
    env: SWEEP (names, comma separated: see ``sweeps``), ONLY (substring of a
    probe's key), N (10000 launches a configuration), BUFFERS
    (plain,uncached,fine), MODES (eager,graph), BOUND
    (262144 polls), TICKS
    (20000000: 0.2 s of the 100 MHz clock), MAX_S (300: a configuration's
    time cap), ISA_ONLY (1: compile and read the loops, launch nothing),
    SHOW_LOOPS (1), SLOTS (64: launches a graph replay), EAGER_BATCH (16:
    eager launches queued between awaits; 1 awaits each), HOST (unset; write:
    before every eager launch or graph replay the host stores into the
    other copies' regions and reads every region back, as the sampling
    kernel's fixture stages a launch; with SLOTS=1 a replay is one launch)

Every poll and hand-off is bounded. Every kernel, buffer and graph stays
referenced in KEEP; every pointer is checked before a launch.
"""

import hashlib
import json
import os
import time
from dataclasses import dataclass, replace

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import Int64, T, as_ir_value

from atom.models.deepseek_v41.mono.gfx1250.kernels import device as hx
from atom.models.kimi_k3.mono.kernels.stages import k2_allreduce_gfx1250 as ar
from atom.models.kimi_k3.mono.kernels.stages.moe_tdm_gfx1250 import _realtime
from atom.models.kimi_k3.mono.kernels.stages.tdm_timeline import _hw_id
from atom.mono.device.ops import rsrc, spin_pause, st_dev, traced
from atom.mono.polling import SPIN_PAUSE_ASM
from atom.mono.runtime.compile import _code_object, compile_only
from atom.utils import poll_isa

THREADS = 256
CTAS = 256
LINE_WORDS = ar.LINE_WORDS
BOUND = int(os.environ.get("BOUND", str(1 << 18)))
TICKS = int(os.environ.get("TICKS", "20000000"))
HAND_BOUND = 1 << 18
SLOTS = int(os.environ.get("SLOTS", "64"))
assert SLOTS & (SLOTS - 1) == 0
WRITERS, WRITER_STEP = 16, 17
REC_OK = 0x600D
NT_STORE = "global_store_b128 $0, $1, off th:TH_STORE_NT scope:SCOPE_SYS"
KEEP = []
# shape v1 -- the sampling kernel's first design, its region and its lanes
# (step_tail/layout.py): a region a rank, PEER_BYTES between guards; the
# Markov lines [bank][position][source][request] past the argmax lines; a
# lane a line (source = lane % 8, request = lane // 8), every source's lines
# but this rank's placed by the host before the launch
V1_TP, V1_REQ, V1_DRAFTS, V1_RANK = 8, 16, 7, 6
V1_WORD0 = 2 * V1_TP * 64 * 16 // 4
V1_GUARD = 4096
V1_BYTES = 1 << 16
V1_SPAN = V1_BYTES + 2 * V1_GUARD


def v1_word(bank, position, source, request):
    """First word, in a rank's region, of a Markov line."""
    return (
        V1_WORD0
        + (((bank * V1_DRAFTS + position) * V1_TP + source) * V1_REQ + request) * 4
    )


@dataclass(frozen=True)
class Probe:
    # vol: SCOPE_SYS | VOLATILE (a wait a load); pt: SCOPE_SYS (together)
    load: str = "vol"
    # asm: spin_pause (side-effecting asm); intr: the s_sleep intrinsic
    pause: str = "asm"
    sleep: int = 16
    # count: polls < BOUND; clock: the 100 MHz clock asked every round
    bound: str = "count"
    # every wave asks the realtime clock once at the entry
    clock_entry: bool = False
    lines: int = 1
    rounds: int = 1
    pollers: int = 16
    # poller p is thread p * spread + offset
    spread: int = 1
    offset: int = 0
    # all: every CTA polls the writers' lines (each line read by 256 CTAs, as
    # the sampling kernel's); pair: every CTA stores lines of its own and
    # polls its partner CTA's (bid ^ 1), the odd CTAs storing ``late`` sleeps
    # late (each line read by one thread, as the all-reduces' late-push builds)
    fan: str = "all"
    # fan all -- cta0, or spread: 16 CTAs (bid 0, 17, .., 255), a poller's
    # lines from one
    writers: str = "cta0"
    copies: int = 1
    own_stores: bool = False
    handoff: bool = False
    banks: int = 2
    late: int = 4
    store: str = "sys"
    # v1: the sampling kernel's first design (above); ``pollers`` its requests
    shape: str = ""

    def key(self):
        return (
            f"{self.load}_{self.pause}{self.sleep if self.pause == 'intr' else ''}"
            f"_{self.bound}{'_ce' if self.clock_entry else ''}"
            f"_l{self.lines}_r{self.rounds}_p{self.pollers}"
            f"_s{self.spread}o{self.offset}"
            f"_{'pair' if self.fan == 'pair' else self.writers}_c{self.copies}"
            f"{'_own' if self.own_stores else ''}{'_ho' if self.handoff else ''}"
            f"_k{self.banks}_d{self.late}_{self.store}"
            f"{'_v1' if self.shape == 'v1' else ''}"
        )

    @property
    def records(self):
        """Polling threads a CTA."""
        return self.pollers * V1_TP if self.shape == "v1" else self.pollers

    @property
    def guard(self):
        return V1_GUARD if self.shape == "v1" else 0

    @property
    def nl(self):
        return self.pollers * self.lines

    @property
    def cta_lines(self):
        """Lines of one CTA's block (fan pair: a block a CTA)."""
        return self.rounds * self.nl

    @property
    def bank_words(self):
        blocks = CTAS if self.fan == "pair" else 1
        return blocks * self.cta_lines * LINE_WORDS

    @property
    def region_bytes(self):
        if self.shape == "v1":
            return V1_SPAN
        return self.banks * self.bank_words * 4

    @property
    def mine(self):
        """The copy polled: as a rank's own region among its peers'."""
        return min(6, self.copies - 1)

    @property
    def pause_asm(self):
        return SPIN_PAUSE_ASM if self.pause == "asm" else f"s_sleep {self.sleep}"

    def check(self):
        assert self.load in ("vol", "pt") and self.pause in ("asm", "intr")
        assert self.bound in ("count", "clock") and self.writers in ("cta0", "spread")
        assert self.banks in (1, 2) and self.store in ("sys", "sys_nt")
        assert self.fan in ("all", "pair")
        assert self.fan == "all" or self.writers == "cta0"
        assert 1 <= self.rounds <= 8 and 1 <= self.lines <= 16
        assert self.offset < self.spread
        assert (self.pollers - 1) * self.spread + self.offset < THREADS
        assert self.shape in ("", "v1")
        if self.shape == "v1":
            assert (self.lines, self.copies, self.fan) == (1, V1_TP, "all")
            assert (self.writers, self.banks, self.spread) == ("cta0", 2, 1)
            assert self.pollers <= V1_REQ and self.rounds <= V1_DRAFTS


def rec_words(probe):
    return SLOTS * CTAS * (probe.records + 1) * 4


def build_probe(p: Probe):
    p.check()
    nl, bank_words, region_bytes = p.nl, p.bank_words, p.region_bytes
    write_rounds = -(-nl // THREADS)
    modifier = ar.POLL_TOGETHER if p.load == "pt" else ar.POLL_VOLATILE
    v1 = p.shape == "v1"
    name = f"k3_poll_repro_{p.key()}_b{BOUND}_t{TICKS}_g{SLOTS}"

    def pause():
        if p.pause == "asm":
            spin_pause()
        else:
            rocdl.s_sleep(p.sleep)

    def payload(epoch, r, j, w):
        return epoch ^ (j * 7919 + (r * 131 + w + 1))

    def store_line(address, line):
        if p.store == "sys":
            ar._store_line_sys(address, line)
        else:
            llvm.InlineAsmOp(
                None,
                [as_ir_value(fx.Int64(address)), as_ir_value(line)],
                NT_STORE,
                "v,v",
                has_side_effects=True,
            )

    @traced
    def handoff(cnt, dump, r, epoch, tid, bid, hand):
        """Every CTA here before any goes on, as the sampling kernel's device
        hand-off: device-scope stores, waited, an atomic count thread 0
        waits for."""
        if tid < 16:
            st_dev(epoch + r, dump, CTAS * THREADS * 4 + bid * 16 + tid)
        hx.wait_storecnt0()
        gpu.barrier()
        missed = fx.Int32(0)
        if tid == 0:
            at = cnt + fx.Int64(r * 4)
            hx.atomic_add(at, 1)
            tries = fx.Int32(0)
            while (
                (hx.gload(at, fx.Int32, scope="agent") - epoch * CTAS) < fx.Int32(0)
            ) & (tries < HAND_BOUND):
                hx.s_sleep(1)
                tries = tries + 1
            missed = (tries >= HAND_BOUND).select(fx.Int32(1), fx.Int32(0))
        gpu.barrier()
        return hand | missed

    @traced
    def write(bank, base_word, epoch, r, tid, bid):
        """Round ``r``'s lines of this CTA, if it is a writer (fan pair:
        every CTA, into its own block): line ``j`` on thread ``j % THREADS``,
        to every copy."""
        own = bid * p.cta_lines if p.fan == "pair" else 0
        for k in range_constexpr(write_rounds):
            j = tid + k * THREADS
            if const_expr(p.fan == "pair"):
                mine = j < nl
            elif const_expr(p.writers == "cta0"):
                mine = (j < nl) & (bid == 0)
            else:
                mine = (j < nl) & (bid == ((j // p.lines) % WRITERS) * WRITER_STEP)
            if mine:
                line = fx.Vector.from_elements(
                    [payload(epoch, r, own + j, w) for w in range(3)] + [epoch],
                    fx.Int32,
                )
                at = fx.Int64(base_word + (own + r * nl + j) * LINE_WORDS) * 4
                for c in range_constexpr(p.copies):
                    store_line(bank + fx.Int64(c * region_bytes) + at, line)

    @traced
    def write_v1(bank, epoch, r, tid, bid):
        """Shape v1: CTA 0's first lanes a line a request, this rank's, to
        every rank's region."""
        if (tid < p.pollers) & (bid == 0):
            line = fx.Vector.from_elements(
                [payload(epoch, r, V1_RANK * V1_REQ + tid, w) for w in range(3)]
                + [epoch],
                fx.Int32,
            )
            at = fx.Int64(v1_word(epoch & 1, r, V1_RANK, tid)) * 4
            for c in range_constexpr(p.copies):
                store_line(bank + fx.Int64(c * region_bytes + V1_GUARD) + at, line)

    def judge(words, epoch, r, first):
        """After a poll: (stale now, the oldest tag's age, payload wrong)."""
        late = ar._stale(words, p.lines, epoch)
        age = fx.Int32(0)
        wrong = fx.Int32(0)
        for i in range(p.lines):
            age = fx.max(age, epoch - fx.Int32(words[i * LINE_WORDS + 3]))
            for w in range(3):
                bad = fx.Int32(words[i * LINE_WORDS + w]) != payload(
                    epoch, r, first + i, w
                )
                wrong = wrong | bad.select(fx.Int32(1), fx.Int32(0))
        flag = late.select(fx.Int32(1), fx.Int32(0))
        return flag, age * flag, wrong * (1 - flag)

    @traced
    def round_count(live, mine, ats, epoch, r, first, most, gave, age, bad):
        polls = fx.Int32(0)
        late = fx.Int32(0)
        seen = fx.Int32(0)
        wrong = fx.Int32(0)
        if live:
            words = ar._load_lines(mine, ats, modifier)
            count = fx.Int32(0)
            while ar._stale(words, p.lines, epoch) & (count < BOUND):
                pause()
                words = ar._load_lines(mine, ats, modifier)
                count = count + 1
            polls = count
            verdict = judge(words, epoch, r, first)
            late = verdict[0]
            seen = verdict[1]
            wrong = verdict[2]
        return (
            fx.max(most, polls),
            gave | (late * (1 << r)),
            fx.max(age, seen),
            bad | wrong,
        )

    @traced
    def round_clock(live, mine, ats, epoch, r, first, most, gave, age, bad):
        polls = fx.Int32(0)
        late = fx.Int32(0)
        seen = fx.Int32(0)
        wrong = fx.Int32(0)
        if live:
            words = ar._load_lines(mine, ats, modifier)
            count = fx.Int32(0)
            t0 = hx.clock()
            while ar._stale(words, p.lines, epoch) & (
                hx.clock() - t0 < fx.Int64(TICKS)
            ):
                pause()
                words = ar._load_lines(mine, ats, modifier)
                count = count + 1
            polls = count
            verdict = judge(words, epoch, r, first)
            late = verdict[0]
            seen = verdict[1]
            wrong = verdict[2]
        return (
            fx.max(most, polls),
            gave | (late * (1 << r)),
            fx.max(age, seen),
            bad | wrong,
        )

    poll_round = round_clock if p.bound == "clock" else round_count

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def kernel(bank: Int64, dump: Int64, epochs: Int64, rec: Int64, cnt: Int64):
        tid = fx.Int32(fx.thread_idx.x)
        bid = fx.Int32(fx.block_idx.x)
        if const_expr(p.clock_entry):
            # every wave of every CTA: a message round trip each
            _realtime()
        epoch = (
            fx.Int32(bo.buffer_load(rsrc(epochs), bid, vec_width=1, dtype=T.i32)) + 1
        )
        # banks by epoch parity, as the all-reduce's (or one bank)
        base_word = (epoch & 1) * bank_words if p.banks == 2 else fx.Int32(0)
        mine = rsrc(bank + fx.Int64(p.mine * region_bytes + p.guard))
        live = (
            (tid < p.records)
            if v1
            else (tid % p.spread == p.offset) & (tid // p.spread < p.pollers)
        )
        poller = (
            fx.min(tid, p.records - 1) if v1 else fx.min(tid // p.spread, p.pollers - 1)
        )
        # the id a line's payload carries (v1: its source and request)
        first = (
            ((poller % V1_TP) * V1_REQ + poller // V1_TP) if v1 else poller * p.lines
        )
        # who stores ``late`` sleeps late: the writers (fan pair: the odd CTAs)
        writer = (
            (bid % 2 == 1)
            if p.fan == "pair"
            else (
                (bid == 0)
                if p.writers == "cta0"
                else ((bid % WRITER_STEP == 0) & (bid // WRITER_STEP < WRITERS))
            )
        )
        # the block polled: the writers' (fan pair: the partner CTA's)
        peer = (bid ^ 1) * p.cta_lines if p.fan == "pair" else 0
        most = fx.Int32(0)
        gave = fx.Int32(0)
        age = fx.Int32(0)
        bad = fx.Int32(0)
        hand = fx.Int32(0)
        for r in range_constexpr(p.rounds):
            if const_expr(p.handoff):
                hand = handoff(cnt, dump, r, epoch, tid, bid, hand)
            if const_expr(p.late > 0):  # noqa: SIM102
                if writer:
                    ar._sleeps(p.late)
            if const_expr(v1):
                write_v1(bank, epoch, r, tid, bid)
            else:
                write(bank, base_word, epoch, r, tid, bid)
            if const_expr(p.own_stores):  # noqa: SIM102
                if live:
                    store_line(
                        dump + fx.Int64((bid * THREADS + tid) * 16),
                        fx.Vector.from_elements(
                            [tid, bid, fx.Int32(r), epoch], fx.Int32
                        ),
                    )
            most, gave, age, bad = poll_round(
                live,
                mine,
                (
                    [v1_word(epoch & 1, r, poller % V1_TP, poller // V1_TP)]
                    if v1
                    else [
                        base_word + (peer + r * nl + first + i) * LINE_WORDS
                        for i in range(p.lines)
                    ]
                ),
                epoch,
                r,
                peer + first,
                most,
                gave,
                age,
                bad,
            )
            gpu.barrier()
        row = ((epoch & (SLOTS - 1)) * CTAS + bid) * (p.records + 1)
        out = rsrc(rec)
        if live:
            bo.buffer_store(
                fx.Vector.from_elements([most, gave, age, bad], fx.Int32),
                out,
                (row + 1 + poller) * 4,
            )
        if tid == 0:
            bo.buffer_store(
                fx.Vector.from_elements(
                    [_hw_id(), epoch, hand, fx.Int32(REC_OK)], fx.Int32
                ),
                out,
                row * 4,
            )
            bo.buffer_store(epoch, rsrc(epochs), bid)

    @flyc.jit
    def launch(
        bank: Int64,
        dump: Int64,
        epochs: Int64,
        rec: Int64,
        cnt: Int64,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        kernel(bank, dump, epochs, rec, cnt).launch(
            grid=(CTAS,), block=(THREADS,), stream=stream
        )

    return name, launch


def read_loops(probe, name, launch):
    """The build's poll loops as compiled; prints them."""
    with compile_only():
        launch(0, 0, 0, 0, 0)
    out = []
    for artifact in launch._mem_cache.values():
        text = poll_isa.disassemble(_code_object(artifact._ir_text))
        (report,) = poll_isa.analyze(text, probe.pause_asm)
        body = "\n".join(
            ln.split("//")[0].strip()
            for ln in text.splitlines()
            if ln[:1].isspace() and "//" in ln
        )
        entry = report.summary()
        entry["body_sha"] = hashlib.sha256(body.encode()).hexdigest()[:16]
        entry["loads_inside_loop"] = bool(report.loops) and all(
            lp.loads == probe.lines for lp in report.loops
        )
        # re-reading is the loads being on every way round; a hazard of
        # another kind (a message wait with a load out) is in "hazards"
        entry["rereads"] = entry["loads_inside_loop"]
        print(json.dumps({"isa": {"probe": probe.key(), **entry}}), flush=True)
        if os.environ.get("SHOW_LOOPS", "1") == "1":
            for lp in report.loops:
                print(
                    f"--- {probe.key()}: poll loop at {lp.pause:#x}, {lp.loads} loads"
                )
                for ln in lp.lines:
                    print("    " + ln)
            print(flush=True)
        out.append(entry)
    return out


def sweeps(name):
    base = Probe()
    both = [replace(base, load=load) for load in ("vol", "pt")]
    if name == "smoke":
        return [
            replace(base, load="pt", lines=4, pollers=8),
            replace(base, load="vol", lines=4, pollers=8),
        ]
    if name == "ar":
        # as the all-reduces poll: a thread's lines together, writers elsewhere
        return [
            replace(b, lines=lines, rounds=rounds, pollers=32, writers=writers)
            for b in both
            for lines in (1, 4, 8)
            for rounds in (1, 2)
            for writers in ("cta0", "spread")
        ]
    if name == "pair":
        # as the all-reduces' late-push builds: a CTA polls the lines its
        # partner CTA stores, one reader a line
        return [
            replace(b, fan="pair", lines=lines, pollers=pollers, late=late)
            for b in both
            for lines, pollers in ((1, 64), (4, 32), (8, 32))
            for late in (4, 300)
        ]
    if name == "markov":
        # as the sampling kernel polled: a lane in eight a line, the clock
        # every round, a hand-off ahead, a line stored to eight regions
        m = replace(
            base, load="pt", pause="intr", sleep=0, bound="clock", spread=8,
            offset=6, handoff=True, copies=8, rounds=4,
        )  # fmt: skip
        out = [
            replace(m, pollers=n, load=load)
            for n in (8, 9, 12, 16)
            for load in ("pt", "vol")
        ]
        at = replace(m, pollers=12)
        out += [
            replace(at, bound="count"),
            replace(at, pause="asm"),
            replace(at, pause="asm", bound="count"),
            replace(at, handoff=False),
            replace(at, copies=1),
            replace(at, spread=1, offset=0),
            replace(at, rounds=1),
            replace(at, sleep=16),
            replace(at, banks=1),
            replace(at, pollers=24),
        ]
        return out
    if name == "v1":
        # the sampling kernel's first design as it failed: its region, a
        # lane a line of every source, the host's lines every launch
        m = replace(
            base, shape="v1", load="pt", pause="intr", sleep=0, bound="clock",
            handoff=True, copies=8, rounds=4,
        )  # fmt: skip
        return [
            *[replace(m, pollers=n) for n in (8, 9, 12, 16)],
            replace(m, pollers=9, load="vol"),
            replace(m, pollers=12, load="vol"),
            replace(m, pollers=9, bound="count"),
            replace(m, pollers=9, sleep=16),
        ]
    if name == "dense":
        # the sampling kernel's poll with every lane of two to four waves
        # polling (its failing shapes had more than two waves' lanes loading
        # a line each): the clock every round, no pause, a hand-off ahead
        m = replace(
            base, load="pt", pause="intr", sleep=0, bound="clock", handoff=True,
            copies=8, rounds=4,
        )  # fmt: skip
        out = [
            replace(m, pollers=n, load=load)
            for n in (64, 72, 96, 128)
            for load in ("pt", "vol")
        ]
        at = replace(m, pollers=96)
        out += [
            replace(at, bound="count"),
            replace(at, pause="asm"),
            replace(at, sleep=16),
            replace(at, handoff=False),
            replace(at, copies=1),
            replace(at, rounds=1),
            replace(at, banks=1),
            replace(at, writers="spread"),
            replace(at, pause="asm", bound="count"),
        ]
        return out
    if name == "clock":
        return [
            replace(b, lines=lines, pollers=32, **kw)
            for b in both
            for lines in (1, 4)
            for kw in ({"clock_entry": True}, {"bound": "clock"})
        ]
    if name == "h1":
        # nothing in the loop may write memory: is the load still in it?
        return [
            replace(b, pause="intr", sleep=sleep, lines=lines, pollers=32, late=16)
            for b in both
            for sleep in (0, 16)
            for lines in (1, 4)
        ]
    if name == "own":
        return [
            replace(b, own_stores=True, lines=lines, pollers=32)
            for b in both
            for lines in (1, 8)
        ]
    if name == "banks":
        return [
            replace(b, banks=1, lines=lines, pollers=32)
            for b in both
            for lines in (1, 8)
        ]
    if name == "late":
        return [
            replace(b, late=late, lines=4, pollers=32)
            for b in both
            for late in (0, 300)
        ]
    if name == "wide":
        return [
            replace(b, lines=lines, pollers=pollers, rounds=2, writers=writers)
            for b in both
            for lines, pollers in ((1, 128), (2, 128), (16, 16), (8, 64))
            for writers in ("cta0", "spread")
        ]
    if name == "nt":
        return [replace(b, store="sys_nt", lines=4, pollers=32) for b in both]
    raise ValueError(name)


def main():
    import torch

    torch.cuda.set_device(0)
    dev = torch.device("cuda", 0)
    names = os.environ.get("SWEEP", "smoke").split(",")
    probes = []
    for name in names:
        for probe in sweeps(name):
            if probe not in probes:
                probes.append(probe)
    if os.environ.get("ONLY"):
        probes = [p for p in probes if os.environ["ONLY"] in p.key()]
    launches = int(os.environ.get("N", "10000"))
    buffers = os.environ.get("BUFFERS", "plain,uncached,fine").split(",")
    modes = os.environ.get("MODES", "eager,graph").split(",")
    max_s = float(os.environ.get("MAX_S", "300"))
    isa_only = os.environ.get("ISA_ONLY", "0") == "1"
    guard = 4096
    most_bytes = max(p.copies * p.region_bytes for p in probes)

    banks = {}
    if not isa_only:
        from atom.mono.runtime import uncached

        for kind in buffers:
            size = -(-(most_bytes + 2 * guard) // 4096) * 4096
            if kind == "plain":
                whole = torch.zeros(size, dtype=torch.uint8, device=dev)
            elif kind == "uncached":
                whole = uncached.uncached_bytes(size, dev)
            elif kind == "fine":
                whole = uncached.fine_grained_bytes(size, dev)
            elif kind == "symm":
                # a torch.symm_mem MORI bank (the serving banks' allocation):
                # not allocated here -- reads through peer pointers reset the
                # node twice on 11-21, and nothing of symm_mem is touched
                # until the platform is fixed
                raise SystemExit("BUFFERS symm: disabled (no symm_mem on this node)")
            else:
                raise SystemExit(f"BUFFERS {kind!r}: plain, uncached, fine or symm")
            torch.cuda.synchronize()
            if whole.numel() != size or not whole.data_ptr():
                raise RuntimeError(f"{kind} bank: {whole.numel()} bytes: not launching")
            banks[kind] = whole
            KEEP.append(whole)
    dump = torch.zeros(
        CTAS * THREADS * 16 + CTAS * 16 * 4, dtype=torch.uint8, device=dev
    )
    epochs = torch.zeros(CTAS, dtype=torch.int32, device=dev)
    cnt = torch.zeros(16, dtype=torch.int32, device=dev)
    KEEP.extend([dump, epochs, cnt])
    print(
        json.dumps(
            {
                "run": {
                    "probes": len(probes),
                    "launches": launches,
                    "buffers": buffers,
                    "modes": modes,
                    "bound": BOUND,
                    "ticks": TICKS,
                    "pause": SPIN_PAUSE_ASM,
                }
            }
        ),
        flush=True,
    )
    for probe in probes:
        started = time.time()
        name, launch = build_probe(probe)
        isa = read_loops(probe, name, launch)
        KEEP.append(launch)
        print(
            json.dumps({"built": probe.key(), "s": round(time.time() - started, 1)}),
            flush=True,
        )
        if isa_only:
            continue
        rec = torch.zeros(rec_words(probe), dtype=torch.int32, device=dev)
        KEEP.append(rec)
        for kind in buffers:
            for mode in modes:
                result = run(
                    torch, probe, launch, banks[kind], guard, dump, epochs, rec, cnt,
                    mode, launches, max_s,
                )  # fmt: skip
                result.update(
                    probe=probe.key(),
                    buffer=kind,
                    mode=mode,
                    rereads=all(e["rereads"] for e in isa),
                )
                print(json.dumps({"result": result}), flush=True)
    print(json.dumps({"done": len(probes)}), flush=True)


def run(
    torch, probe, launch, whole, guard, dump, epochs, rec, cnt, mode, launches, max_s
):
    """``launches`` launches of one build on one bank, eager or as graph
    replays of SLOTS launches; the records read after every launch (eager) or
    replay."""
    dev = whole.device
    whole.fill_(0xA5)
    bank = whole[guard : guard + probe.copies * probe.region_bytes]
    bank.zero_()
    for t in (dump, epochs, rec, cnt):
        t.zero_()
    torch.cuda.synchronize()
    pointers = [t.data_ptr() for t in (bank, dump, epochs, rec, cnt)]
    assert all(pointers), pointers
    view = rec.view(SLOTS, CTAS, probe.records + 1, 4)
    mine = bank[probe.mine * probe.region_bytes :][: probe.region_bytes]
    v1 = probe.shape == "v1"
    held = [None]
    if v1 and mode == "graph" and SLOTS != 1:
        return {"skipped": "shape v1 takes SLOTS=1: the host stages every launch"}
    if v1:
        words = mine[V1_GUARD : V1_GUARD + V1_BYTES].view(torch.int32)
        spots = [
            (r, s, q)
            for r in range(probe.rounds)
            for s in range(V1_TP)
            for q in range(probe.pollers)
        ]
        own_tags = torch.tensor(
            [
                [v1_word(b, r, s, q) + 3 for r, s, q in spots if s == V1_RANK]
                for b in range(2)
            ],
            device=dev,
        )
        others_at = torch.tensor(
            [
                [v1_word(b, r, s, q) for r, s, q in spots if s != V1_RANK]
                for b in range(2)
            ],
            device=dev,
        )
        code = torch.tensor(
            [(s * V1_REQ + q) * 7919 + r * 131 for r, s, q in spots if s != V1_RANK],
            dtype=torch.int32,
            device=dev,
        )
        four = torch.arange(4, device=dev)

        def stale_tags(last):
            return (words[own_tags[last & 1]] != last).sum()

        def stage():
            """The other ranks' lines of the next launch, as the fixture
            places them: tagged with its epoch, in this rank's region; then
            a read of every region."""
            torch.cuda.synchronize()
            nxt = int(epochs[0]) + 1
            lines = torch.stack(
                [(code + (w + 1)) ^ nxt for w in range(3)]
                + [torch.full_like(code, nxt)],
                dim=1,
            )
            words[others_at[nxt & 1][:, None] + four] = lines
            held[0] = whole.clone()
            torch.cuda.synchronize()

    else:
        tags = mine.view(torch.int32).view(probe.banks, probe.bank_words // 4, 4)[
            :, :, 3
        ]

        def stale_tags(last):
            return (tags[last % probe.banks] != last).sum()

    xcd = torch.arange(CTAS, device=dev) % 8
    host = os.environ.get("HOST", "")
    others = [
        bank[c * probe.region_bytes :][: probe.region_bytes]
        for c in range(probe.copies)
        if c != probe.mine
    ]

    def touch(k):
        """Shape v1: the host's lines of the launch. HOST=write: the host's
        own stores next to the polled region and a read of everything,
        between launches."""
        if v1:
            stage()
        if host != "write":
            return
        # the launches before are done before the host stores
        torch.cuda.synchronize()
        for region in others:
            region.fill_(k & 0x7F)
        held[0] = whole.clone()
        torch.cuda.synchronize()

    out = {
        "launches": 0,
        "bad_launches": 0,
        "ctas_affected": [],
        "by_xcd": [0] * 8,
        "by_hw": {},
        "ages": set(),
        "rounds": 0,
        "polls_max": 0,
        "polls_thread0": [],
        "bad_payload": 0,
        "writer_bad": 0,
        "handoff_missed": 0,
        "unwritten_records": 0,
        "first_bad": [],
    }

    def fire(stream):
        launch(*pointers, stream=stream)

    def account(slots):
        """The records of ``slots`` (each a launch), and the last launch's
        lines as the host reads them back (its epoch from the counters)."""
        rows = view[slots]
        last = epochs[0].long()
        head, body = rows[:, :, 0], rows[:, :, 1:]
        gave = body[..., 1] != 0
        per_cta = gave.any(-1)
        per_launch = per_cta.sum(-1)
        failing = per_launch > 0
        stats = torch.stack(
            [
                failing.sum(),
                (head[..., 3] != REC_OK).sum(),
                (head[..., 2] != 0).sum(),
                (body[..., 3] != 0).sum(),
                torch.where(gave, torch.zeros_like(body[..., 0]), body[..., 0]).max(),
                stale_tags(last),
            ]
        ).tolist()
        out["launches"] += len(slots)
        out["bad_launches"] += stats[0]
        out["unwritten_records"] += stats[1]
        out["handoff_missed"] += stats[2]
        out["bad_payload"] += stats[3]
        out["polls_max"] = max(out["polls_max"], stats[4])
        out["writer_bad"] += stats[5]
        if len(out["polls_thread0"]) < 4096:
            out["polls_thread0"] += body[:, ::16, 0, 0].flatten().tolist()
        if stats[0]:
            out["ctas_affected"] += per_launch[failing].tolist()
            hit = per_cta[failing]
            for g in range(8):
                out["by_xcd"][g] += int(hit[:, xcd == g].sum())
            ids = (head[failing][..., 0] >> 8)[hit].tolist()
            for value in ids:
                out["by_hw"][value] = out["by_hw"].get(value, 0) + 1
            bodies = body[failing]
            out["ages"].update(bodies[..., 2][bodies[..., 1] != 0].unique().tolist())
            out["rounds"] |= _fold_or(bodies[..., 1])
            if len(out["first_bad"]) < 3:
                k = int(torch.nonzero(failing)[0])
                out["first_bad"].append(
                    {
                        "epoch": int(head[k, 0, 1]),
                        "ctas": int(per_launch[k]),
                        "cta0": bool(per_cta[k, 0]),
                        "threads": torch.nonzero(gave[k].any(0))
                        .flatten()
                        .tolist()[:24],
                        "ok_ctas": torch.nonzero(~per_cta[k]).flatten().tolist()[:40],
                        # HW_ID1's fields of thread 0's wave, a value's CTAs
                        # that saw the lines / that did not
                        "hw": {
                            name: {
                                "saw": torch.bincount(
                                    ((head[k, :, 0] >> shift) & mask)[~per_cta[k]],
                                    minlength=mask + 1,
                                ).tolist(),
                                "stale": torch.bincount(
                                    ((head[k, :, 0] >> shift) & mask)[per_cta[k]],
                                    minlength=mask + 1,
                                ).tolist(),
                            }
                            for name, shift, mask in (
                                ("se", 26, 7),
                                ("sh", 20, 1),
                                ("cu", 16, 15),
                                ("simd", 12, 3),
                            )
                        },
                    }
                )

    def _fold_or(values):
        folded = 0
        for v in values.unique().tolist():
            folded |= v
        return folded

    started = time.time()
    stream = torch.cuda.current_stream()
    if mode == "eager":
        # a batch of launches is queued, then awaited and its records read:
        # awaiting a launch by itself costs the host about 10 ms (measured:
        # 10^4 launches in 107 s). EAGER_BATCH=1 awaits every launch.
        batch = max(1, min(SLOTS, int(os.environ.get("EAGER_BATCH", "16"))))
        if v1:
            batch = 1
        pending = []
        for i in range(launches):
            touch(i)
            fire(stream)
            pending.append((i + 1) & (SLOTS - 1))
            if len(pending) < batch and i + 1 < launches:
                continue
            torch.cuda.synchronize()
            account(pending)
            pending = []
            if out["bad_launches"] == out["launches"] >= 30:
                out["stopped"] = "every launch gave up"
                break
            if time.time() - started > max_s:
                out["stopped"] = "time"
                break
    else:
        side = torch.cuda.Stream()
        side.wait_stream(stream)
        with torch.cuda.stream(side):
            for _ in range(4):
                touch(0)
                fire(side)
                if v1:
                    torch.cuda.synchronize()
        stream.wait_stream(side)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=side):
            for _ in range(SLOTS):
                fire(torch.cuda.current_stream())
        KEEP.append((graph, side))
        torch.cuda.synchronize()
        rec.zero_()
        torch.cuda.synchronize()
        while out["launches"] < launches:
            touch(out["launches"])
            graph.replay()
            torch.cuda.synchronize()
            account(list(range(SLOTS)))
            if out["bad_launches"] == out["launches"]:
                out["stopped"] = "every launch gave up"
                break
            if time.time() - started > max_s:
                out["stopped"] = "time"
                break
    torch.cuda.synchronize()
    out["guards_ok"] = bool(
        (whole[:guard] == 0xA5).all()
        and (whole[guard + probe.copies * probe.region_bytes :] == 0xA5).all()
    )
    out["epoch_end"] = int(epochs[0])
    out["host"] = host
    out["seconds"] = round(time.time() - started, 1)
    ctas = sorted(out.pop("ctas_affected"))
    out["ctas_affected"] = [ctas[0], ctas[len(ctas) // 2], ctas[-1]] if ctas else None
    polls = sorted(out.pop("polls_thread0"))
    out["polls_thread0"] = (
        {"p50": polls[len(polls) // 2], "p99": polls[len(polls) * 99 // 100]}
        if polls
        else None
    )
    out["ages"] = sorted(out["ages"])
    out["rounds"] = [r for r in range(8) if out["rounds"] >> r & 1]
    return out


if __name__ == "__main__":
    main()
