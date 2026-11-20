# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Cross-rank latency of the gfx1250 all-reduces' tagged 16 B lines.

Rank A's CTA ``c`` pushes ``lines`` lines (three payload words and a tag) into
rank B's symm_mem bank; B's CTA ``c`` polls them until every tag is current,
then pushes as many lines back into A's bank, which A polls: one round trip
an iteration, timed on A with the 100 MHz clock. Half of it is the one-way
time of ``lines`` lines with ``ctas`` CTAs pushing at once. Tags are unique
per launch and iteration, so a line left from before is never taken; each
taken line's payload is checked.

One thing varies at a time (``Probe``): the store (scope, NT, b64, buffer),
whether the pushing threads wait out ``s_wait_storecnt`` before polling, the
poll's load (buffer or global, scope), the sleep between polls, which waves
push and which poll, and whether one wave polls every line or each thread
its own.

    torchrun --nproc_per_node=4 tools/kimi_k3_mono_line_latency_probe.py
    env: SWEEP (base | knobs | load | fan), ITERS, PAIR ("0,1")
"""

import json
import os
from dataclasses import dataclass, replace

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import Int32, Int64, T, as_ir_value

from atom.models.kimi_k3.mono.kernels.stages.moe_tdm_gfx1250 import _realtime
from atom.mono.device.ops import load_ptr64, rsrc, traced

THREADS = 256
LINE_WORDS = 4
SCOPE_SE, SCOPE_DEV, SCOPE_SYS = 1 << 3, 2 << 3, 3 << 3
VOLATILE = 1 << 31
# a poll gives up past this many retries (the launch then reports it)
POLL_LIMIT = 1 << 20
STAMP_WORDS = 8

STORES = {
    "sys": "global_store_b128 $0, $1, off scope:SCOPE_SYS",
    "dev": "global_store_b128 $0, $1, off scope:SCOPE_DEV",
    "sys_nt": "global_store_b128 $0, $1, off th:TH_STORE_NT scope:SCOPE_SYS",
    "sys_b64": "global_store_b64 $0, $1, off scope:SCOPE_SYS",
    "buf_sys": None,
}
POLLS = ("buf_sys_v", "buf_dev_v", "glb_sys", "glb_dev")


@dataclass(frozen=True)
class Probe:
    ctas: int = 1
    lines: int = 1
    store: str = "sys"
    wait_store: bool = False
    poll: str = "buf_sys_v"
    sleep: int = 16
    # threads that push (lines dealt round-robin); split: waves 0-3 push and
    # waves 4-7 poll, else the pushing threads poll their own lines
    push_threads: int = THREADS
    split: bool = False
    # "own": each line polled by the thread that pushes it; "wave": one wave
    poller: str = "own"
    # pull: each rank stores its lines into its own bank and the peer polls
    # them there, across the fabric
    pull: bool = False
    # after its stores, each pushing thread loads one local word (already
    # current) and waits for it: whether a load waits out remote stores
    load_after: str = ""

    @property
    def words(self):
        return 2 if self.store == "sys_b64" else LINE_WORDS

    def key(self):
        return (
            f"c{self.ctas}_l{self.lines}_{self.store}_w{int(self.wait_store)}_{self.poll}"
            f"_s{self.sleep}_t{self.push_threads}{'_split' if self.split else ''}_{self.poller}"
            f"{'_pull' if self.pull else ''}{'_la' + self.load_after if self.load_after else ''}"
        )


def _asm(text, operands, constraints, result=None):
    if result is None:
        llvm.InlineAsmOp(None, operands, text, constraints, has_side_effects=True)
        return None
    return llvm.inline_asm(result, operands, text, constraints, has_side_effects=True)


def _pause(sleep):
    _asm("s_nop 0" if sleep == 0 else f"s_sleep {sleep}", [], "")


def build_probe(probe: Probe):
    words = probe.words
    payload = words - 1
    pushers = probe.push_threads
    if probe.split and pushers > THREADS // 2:
        raise ValueError("split: waves 0-3 push")
    rounds = -(-probe.lines // pushers)
    poll_offset = THREADS // 2 if probe.split else 0
    wave_rounds = -(-probe.lines // 32)

    def vec_type():
        return ir.VectorType.get([words], ir.IntegerType.get_signless(32))

    def store_line(address, line, peer):
        if const_expr(probe.store == "buf_sys"):
            bo.buffer_store(line, rsrc(peer), address, cache_modifier=SCOPE_SYS)
        else:
            _asm(
                STORES[probe.store],
                [as_ir_value(fx.Int64(address)), as_ir_value(line)],
                "v,v",
            )

    def load_line(mine_base, mine, word):
        if const_expr(probe.poll.startswith("buf")):
            scope = SCOPE_SYS if probe.poll == "buf_sys_v" else SCOPE_DEV
            return fx.Vector(
                bo.buffer_load(
                    mine, word, vec_width=words, dtype=T.i32,
                    cache_modifier=scope | VOLATILE,
                )
            )  # fmt: skip
        scope = "SCOPE_SYS" if probe.poll == "glb_sys" else "SCOPE_DEV"
        width = "b128" if words == 4 else "b64"
        return fx.Vector(
            _asm(
                f"global_load_{width} $0, $1, off scope:{scope}\ns_wait_loadcnt 0x0",
                [as_ir_value(fx.Int64(mine_base + fx.Int64(word) * 4))],
                "=v,v",
                vec_type(),
            )
        )

    def payload_word(tag, cta, line, w):
        return tag ^ (cta * 7919 + line * 3 + w)

    @traced
    def push(peer, cta, tid, tag, scratch):
        if tid < pushers:
            for r in range_constexpr(rounds):
                line = tid + r * pushers
                if line < probe.lines:
                    values = [payload_word(tag, cta, line, w) for w in range(payload)]
                    store_line(
                        (cta * probe.lines + line) * words
                        if probe.store == "buf_sys"
                        else peer + fx.Int64((cta * probe.lines + line) * words * 4),
                        fx.Vector.from_elements([*values, tag], fx.Int32),
                        peer,
                    )
            if const_expr(probe.wait_store):
                _asm("s_wait_storecnt 0x0", [], "")
            if const_expr(probe.load_after != ""):
                scope = {
                    "sys": SCOPE_SYS | VOLATILE,
                    "dev": SCOPE_DEV | VOLATILE,
                    "plain": 0,
                }[probe.load_after]
                word = bo.buffer_load(
                    rsrc(scratch), tid, vec_width=1, dtype=T.i32, cache_modifier=scope
                )
                _asm("s_wait_loadcnt 0x0", [as_ir_value(fx.Int32(word))], "v")

    @traced
    def poll_one(mine_base, mine, cta, line, tag, live, polls, errors):
        word = (cta * probe.lines + line) * words
        got = load_line(mine_base, mine, word)
        tries = fx.Int32(0)
        while live & (fx.Int32(got[payload]) != tag) & (tries < POLL_LIMIT):
            _pause(probe.sleep)
            got = load_line(mine_base, mine, word)
            tries = tries + 1
        bad = fx.Int32(0)
        for w in range_constexpr(payload):
            bad = bad | (fx.Int32(got[w]) != payload_word(tag, cta, line, w)).select(
                fx.Int32(1), fx.Int32(0)
            )
        timed_out = (tries >= POLL_LIMIT).select(fx.Int32(1 << 16), fx.Int32(0))
        return polls + tries, errors | (live.select(bad, fx.Int32(0)) | timed_out)

    def poll_all(mine_base, mine, cta, tid, tag, polls, errors):
        if const_expr(probe.poller == "own"):
            local = tid - poll_offset
            live_thread = (local >= 0) & (local < pushers)
            for r in range_constexpr(rounds):
                line = fx.max(local, 0) + r * pushers
                polls, errors = poll_one(
                    mine_base, mine, cta, fx.min(line, probe.lines - 1), tag,
                    live_thread & (line < probe.lines), polls, errors,
                )  # fmt: skip
        else:
            lane = tid % 32
            live_wave = tid // 32 == poll_offset // 32
            for r in range_constexpr(wave_rounds):
                line = lane + r * 32
                polls, errors = poll_one(
                    mine_base, mine, cta, fx.min(line, probe.lines - 1), tag,
                    live_wave & (line < probe.lines), polls, errors,
                )  # fmt: skip
        return polls, errors

    name = f"k3_line_probe_{probe.key()}"

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def kernel(
        peers: Int64,
        initiator: Int32,
        responder: Int32,
        rank: Int32,
        tag0: Int32,
        iters: Int32,
        stamps: Int64,
    ):
        tid = fx.Int32(fx.thread_idx.x)
        cta = fx.Int32(fx.block_idx.x)
        mine_base = load_ptr64(peers, rank)
        mine = rsrc(mine_base)
        lead = rank == initiator
        peer = load_ptr64(peers, lead.select(responder, initiator))
        if const_expr(probe.pull):
            mine_base, peer = peer, mine_base
            mine = rsrc(mine_base)
        polls = fx.Int32(0)
        errors = fx.Int32(0)
        rtt = fx.Int64(0)
        drain = fx.Int64(0)
        i = fx.Int32(0)
        while i < iters:
            tag = tag0 + 2 * i
            start = _realtime()
            if rank == initiator:
                push(peer, cta, tid, tag, stamps)
            drained = _realtime()
            polls, errors = poll_all(mine_base, mine, cta, tid, tag, polls, errors)
            gpu.barrier()
            if rank != initiator:
                push(peer, cta, tid, tag, stamps)
                gpu.barrier()
            end = _realtime()
            rtt = rtt + (end - start)
            drain = drain + (drained - start)
            i = i + 1
        if tid == 0:
            out = rsrc(stamps)
            base = cta * STAMP_WORDS
            bo.buffer_store(fx.Int32(rtt), out, base)
            bo.buffer_store(fx.Int32(drain), out, base + 1)
            bo.buffer_store(polls, out, base + 3)
        if errors != 0:
            bo.buffer_store(errors, rsrc(stamps), cta * STAMP_WORDS + 2)

    @flyc.jit
    def launch(
        peers: Int64,
        initiator: Int32,
        responder: Int32,
        rank: Int32,
        tag0: Int32,
        iters: Int32,
        stamps: Int64,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        kernel(peers, initiator, responder, rank, tag0, iters, stamps).launch(
            grid=(probe.ctas,), block=(THREADS,), stream=stream
        )

    return launch


def bank_bytes(probe: Probe) -> int:
    return probe.ctas * probe.lines * probe.words * 4


def sweeps(name):
    base = Probe()
    if name == "base":
        return [base, replace(base, sleep=1), replace(base, sleep=0)]
    if name == "knobs":
        out = []
        for lines in (1, 64):
            b = replace(base, lines=lines)
            out += [
                b,
                replace(b, sleep=0), replace(b, sleep=1), replace(b, sleep=2),
                replace(b, store="dev"), replace(b, store="sys_nt"),
                replace(b, store="buf_sys"), replace(b, store="sys_b64"),
                replace(b, wait_store=True),
                replace(b, poll="buf_dev_v"), replace(b, poll="glb_sys"),
                replace(b, poll="glb_dev"),
                replace(b, poller="wave"),
                replace(b, push_threads=128, split=True),
                replace(b, push_threads=128, split=True, poller="wave"),
            ]  # fmt: skip
        return out
    if name == "pull":
        out = []
        for lines in (1, 64):
            b = replace(base, lines=lines, pull=True)
            out += [
                b,
                replace(b, sleep=0),
                replace(b, poll="glb_sys"),
                replace(b, poll="buf_dev_v"),
            ]
        return out
    if name == "order":
        out = []
        for lines in (1, 64):
            b = replace(base, lines=lines)
            out += [
                b,
                replace(b, load_after="sys"),
                replace(b, load_after="dev"),
                replace(b, load_after="plain"),
                replace(b, wait_store=True),
            ]
        return out
    if name == "load":
        out = []
        for sleep in (16, 1):
            for ctas in (1, 8, 32, 112, 224):
                for lines in (1, 32, 128, 256, 512):
                    out.append(replace(base, ctas=ctas, lines=lines, sleep=sleep))
        return out
    raise ValueError(name)


def main():
    import torch
    import torch.distributed as dist

    dist.init_process_group("gloo")
    rank = dist.get_rank()
    local = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)
    import mori.allocator  # noqa: F401
    import torch.distributed._symmetric_memory as symm_mem

    symm_mem.set_backend("MORI")
    probes = sweeps(os.environ.get("SWEEP", "base"))
    if os.environ.get("ONLY"):
        probes = [p for p in probes if os.environ["ONLY"] in p.key()]
    size = max(bank_bytes(p) for p in probes)
    buf = symm_mem.empty(size, dtype=torch.uint8, device=dev)
    buf.zero_()
    torch.cuda.synchronize()
    ptrs = list(symm_mem.rendezvous(buf, dist.group.WORLD).buffer_ptrs)
    ptrs[rank] = buf.data_ptr()
    peers = torch.tensor(ptrs, dtype=torch.int64, device=dev)
    pair = [int(r) for r in os.environ.get("PAIR", "0,1").split(",")]
    iters = int(os.environ.get("ITERS", "200"))
    keep = [buf, peers]
    tag = 1
    for probe in probes:
        launch = build_probe(probe)
        stamps = torch.zeros(probe.ctas * STAMP_WORDS, dtype=torch.int32, device=dev)
        keep += [launch, stamps]
        results = []
        for _ in range(3):
            stamps.zero_()
            torch.cuda.synchronize()
            dist.barrier()
            if rank in pair:
                launch(
                    peers.data_ptr(),
                    pair[0],
                    pair[1],
                    rank,
                    tag,
                    iters,
                    stamps.data_ptr(),
                )
            tag += 2 * iters + 2
            torch.cuda.synchronize()
            dist.barrier()
            s = stamps.view(probe.ctas, STAMP_WORDS).cpu().long()
            results.append(s)
        if rank == pair[0]:
            s = results[-1]
            # 100 MHz ticks: 10 ns each
            rtt = s[:, 0].float() * 10 / iters / 1000
            drain = s[:, 1].float() * 10 / iters / 1000
            errors = int((s[:, 2] != 0).sum()) + sum(
                int((r[:, 2] != 0).sum()) for r in results[:-1]
            )
            one_way = float(rtt.max()) / 2
            print(
                json.dumps(
                    {
                        "probe": probe.key(),
                        "ctas": probe.ctas,
                        "lines": probe.lines,
                        "rtt_us_med": round(float(rtt.median()), 3),
                        "rtt_us_max": round(float(rtt.max()), 3),
                        "one_way_us": round(one_way, 3),
                        "push_issue_us": round(float(drain.median()), 3),
                        "polls_a_line": round(float(s[:, 3].float().mean()) / iters, 2),
                        "GBps_one_way": round(
                            probe.ctas
                            * probe.lines
                            * probe.words
                            * 4
                            / (one_way * 1e3),
                            2,
                        ),
                        "errors": errors,
                    }
                ),
                flush=True,
            )
    dist.barrier()


if __name__ == "__main__":
    main()
