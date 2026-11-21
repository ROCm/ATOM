# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The gfx1250 MLA back half (``mla_back_gfx1250``) on ONE GPU against the
launches it replaces, with every peer emulated in local memory: no symmetric
memory, no fabric, no other process. The kernel is built for the real shape
(``TP`` ranks all-reducing, ``DCP`` ranks merging, ``HL`` heads a rank) and
run as rank ``RANK``; each emulated rank's receive memory is a local buffer
between guard bands, so every line the kernel pushes "to a peer" lands on
this GPU where the host inspects it.

* DCP=1, TP=1 (no merge): the gluon ``batched_gemm_a16w8`` W_UV on the rank's
  own heads, AITER's ``fused_sigmoid_mul`` and ``k2_oproj_ar_gfx1250`` -- the
  attention output bit for bit, the gated core too.
* DCP=N: the other ranks' merge lines are written into this rank's bank by
  the host (random BF16 rows, LSEs with -inf / NaN / +inf rows) with the
  launch's epoch; the kernel projects all N x HL heads and merges its own.
  The reference merge is a host emulation of ``dcp_merge_ll_gfx1250`` (the
  device's exp / log through torch, the fused multiply-adds in f64): it is
  not the merge kernel, whose bits are checked under torchrun
  (``kimi_k3_mono_mla_back_tp_check.py``).
* TP=N: the other ranks' o_proj partials are written into this rank's bank
  the same way; the reference sums them with production's own partial in
  rank order in fp32.

Checked every launch (CHECK): the output and the gated core bit for bit; the
lines pushed to every owner against production's W_UV and LSEs; the
all-reduce lines pushed to every rank against production's partial; the
guard bands intact; no word of any rank's memory changed outside the lines
``mla_back_layout`` says this rank writes (``stray``). ITERS eager launches,
then REPLAYS CUDA-graph replays on new inputs.

Then (BENCH, TP=DCP=1) CUDA-graph time, GN launches a replay over COPIES
weight sets, REPS alternating replays: the back half against production's
launches (the merge needs peers: timed under torchrun only). With emulated
peers (BENCH, TP or DCP > 1) only production is timed, in the group's shape:
W_UV on the group's heads, the gate and o_proj + AR among one rank, each
alone and in sequence (the back half's lines carry a launch's epoch, so a
graph of many launches cannot be fed by the host).

    python tools/kimi_k3_mono_mla_back_check.py
    env: ROWS (1,7,8,16,24,32,40,48,64), DCP (1), TP (1), RANK (0; under
    torchrun LOCAL_RANK, each process its own GPU and its own emulation), HL
    (12), CHECK (1), BENCH (1), ITERS (2), REPLAYS (20), GN (24), REPS (9),
    COPIES (24), BOUND (4194304: polls give up past it; 0 waits forever),
    LATE (unset; N: the late-push build), ATOM_KIMI_MONO_MLA_BACK_POLL_TOGETHER
    (the builds' polls),
    PROFILE (0; 1: instead, a profile build's per-CTA stamps -- each phase
    from the CTA's own start, one launch a graph replay; with emulated
    peers their lines are in place before the launch, so the phases show
    the kernel's own work without a fabric hop)
"""

import json
import os
import statistics
import time

import torch
from aiter import QuantType, dtypes
from aiter.ops.shuffle import shuffle_weight
from aiter.ops.triton.batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant import (
    batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant as fp8_bmm,
)
from aiter.ops.triton.fusions.fused_sigmoid_mul import fused_sigmoid_mul

from atom.model_ops.utils import dynamic_per_batched_tensor_quant
from atom.models.kimi_k3.mono.config import HIDDEN
from atom.models.kimi_k3.mono.kernels.execution_plan import Gfx1250KimiBuildPlan
from atom.models.kimi_k3.mono.kernels.stages import k2_allreduce_gfx1250 as ar
from atom.models.kimi_k3.mono.kernels.stages import k2_oproj_ar_gfx1250 as oar
from atom.models.kimi_k3.mono.kernels.stages import mla_back_gfx1250 as mb
from atom.models.kimi_k3.mono.kernels.stages import mla_back_layout as lay
from atom.mono.plan.architecture import GFX1250
from atom.mono.runtime.compile import compile_kernels, launcher_resources
from atom.quantization.quark.utils import quant_weight_online

ROWS = [int(r) for r in os.environ.get("ROWS", "1,7,8,16,24,32,40,48,64").split(",")]
DCP = int(os.environ.get("DCP", "1"))
TP = int(os.environ.get("TP", "1"))
WORLD = max(DCP, TP)
if DCP not in (1, WORLD) or TP not in (1, WORLD):
    raise SystemExit(f"DCP {DCP} and TP {TP}: each 1 or the same group")
LOCAL_RANK = os.environ.get("LOCAL_RANK")
RANK = int(os.environ.get("RANK", LOCAL_RANK or "0")) % WORLD
DCP_RANK = RANK if DCP > 1 else 0
AR_RANK = RANK if TP > 1 else 0
HL = int(os.environ.get("HL", "12"))
# guard words either side of an emulated rank's receive memory
GUARD = 4096
PATTERN = 0x5A5A5A5A
CHECK = os.environ.get("CHECK", "1") == "1"
BENCH = os.environ.get("BENCH", "1") == "1"
ITERS = int(os.environ.get("ITERS", "2"))
REPLAYS = int(os.environ.get("REPLAYS", "20"))
GN = int(os.environ.get("GN", "24"))
REPS = int(os.environ.get("REPS", "9"))
COPIES = int(os.environ.get("COPIES", "24"))
BOUND = int(os.environ.get("BOUND", str(1 << 22))) or None
# LATE=N (with BOUND): a late-push build -- the odd CTAs push their lines N
# sleeps late and an o_proj CTA sums its partner's columns
LATE = int(os.environ["LATE"]) if os.environ.get("LATE") else None
PROFILE = os.environ.get("PROFILE", "0") == "1"
PROFILE_LAUNCHES = 24
HEADS = DCP * HL
PROJ = HL * lay.V_DIM
KEEP = []

if LOCAL_RANK is not None:
    torch.cuda.set_device(int(LOCAL_RANK))
dev = torch.device("cuda", torch.cuda.current_device())
gen = torch.Generator(device=dev).manual_seed(23 + RANK)


def bf16(*shape, scale=1.0):
    return (torch.randn(shape, generator=gen, device=dev) * scale).to(torch.bfloat16)


def bits(t):
    return t.contiguous().view(torch.int16)


class Layer:
    """One layer's weights: W_UV for the DCP group's heads and o_proj."""

    def __init__(self):
        self.w_uv, s_uv = dynamic_per_batched_tensor_quant(
            bf16(HEADS, lay.V_DIM, lay.KV_LORA, scale=0.05), dtype=dtypes.fp8
        )
        self.s_uv = s_uv.reshape(1).contiguous()
        q, s = quant_weight_online(
            bf16(HIDDEN, PROJ, scale=0.02),
            QuantType.per_1x128,
            dtypes.fp8,
            scale_dtype=dtypes.fp8_e8m0,
        )
        self.w_o = shuffle_weight(q, layout=(16, 16)).contiguous()
        self.s_o = s.contiguous()


LAYERS = [Layer() for _ in range(COPIES if BENCH else 1)]
KEEP.append(LAYERS)


def inputs(rows):
    o = bf16(rows, HEADS, lay.KV_LORA)
    lse = torch.randn(rows, HEADS, generator=gen, device=dev) * 2.0 + 4.0
    gate = bf16(rows, PROJ, scale=2.0)
    return {"o": o, "lse": lse.float().contiguous(), "gate": gate}


def plan_of(rows):
    return Gfx1250KimiBuildPlan(
        threads=256,
        blocks=256,
        lds_bytes=GFX1250.max_workgroup_lds_bytes,
        tokens=rows,
        nblocks=1,
    )


class Production:
    """The replaced launches for one rank: W_UV, (the merge,) the gate and
    ``k2_oproj_ar_gfx1250`` among one rank."""

    def __init__(self, rows):
        self.rows = rows
        self.oproj = oar.build_oproj_allreduce(plan_of(rows), 1, PROJ)
        self.bank = torch.zeros(
            oar.oproj_peer_bytes(1, rows) + 1024, dtype=torch.uint8, device=dev
        )
        self.peers = torch.tensor([self.bank.data_ptr()], dtype=torch.int64, device=dev)
        self.epochs = torch.zeros(ar.MAX_CTAS, dtype=torch.int32, device=dev)
        KEEP.append(self)

    def w_uv(self, inp, layer):
        """``MLAAttention._v_up_proj``: [rows, heads, 128] BF16."""
        x = inp["o"].reshape(-1, HEADS, lay.KV_LORA).transpose(0, 1)
        out = torch.empty(self.rows, HEADS, lay.V_DIM, dtype=torch.bfloat16, device=dev)
        fp8_bmm(x, layer.w_uv, layer.s_uv, group_size=128, YQ=out, transpose_bm=True)
        return out

    def gate(self, merged, inp):
        """``sigmoid_mul``: out of place, AITER's fused gate."""
        attn = merged.reshape(self.rows, PROJ)
        out = torch.empty_like(attn)
        return fused_sigmoid_mul(attn, inp["gate"], out=out)

    def o_proj(self, core, layer):
        out = core.new_empty(self.rows, HIDDEN)
        self.oproj.launcher(
            core.data_ptr(),
            layer.w_o.data_ptr(),
            layer.s_o.data_ptr(),
            out.data_ptr(),
            self.peers.data_ptr(),
            self.epochs.data_ptr(),
            0,
            stream=torch.cuda.current_stream(),
        )
        KEEP.append(core)
        return out


def merge_reference(parts, lses):
    """``dcp_merge_ll_gfx1250``'s combine of ``parts`` [N, rows, HL, 128]
    BF16 by ``lses`` [N, rows, HL] f32 on the host: the device's exp / log
    (torch's kernels call the same libm), the multiply-adds fused through
    f64, rounded to BF16 once."""
    n = parts.size(0)
    neg_inf = torch.tensor(float("-inf"), device=dev)
    ls = torch.where(torch.isnan(lses) | (lses == float("inf")), neg_inf, lses)
    top = ls.max(0).values
    top = torch.where(top == float("-inf"), torch.zeros_like(top), top)
    total = torch.exp(ls[0] - top)
    for s in range(1, n):
        total = total + torch.exp(ls[s] - top)
    g = torch.log(total) + top
    acc = torch.zeros(parts.shape[1:], dtype=torch.float32, device=dev)
    for s in range(n):
        f = torch.exp(ls[s] - g)
        f = torch.where(torch.isnan(f), torch.zeros_like(f), f)[..., None]
        fused = (f.double() * parts[s].double() + acc.double()).float()
        acc = torch.where(f != 0, fused, acc)
    return acc.to(torch.bfloat16)


class Back:
    """The back half as rank ``RANK`` of the emulated groups: one local
    receive memory an emulated rank, each between guard bands."""

    def __init__(self, rows, profile=False):
        self.rows = rows
        self.kernel = mb.build_mla_back(
            plan_of(rows), TP, DCP, HL, bound=BOUND, profile=profile,
            poll_marks=BOUND is not None, test_swap_late=LATE,
        )  # fmt: skip
        self.plan = p = self.kernel.plan
        self.out = torch.empty(rows, HIDDEN, dtype=torch.bfloat16, device=dev)
        self.core = torch.empty(rows, PROJ, dtype=torch.bfloat16, device=dev)
        self.flags = torch.zeros(p.flag_words, dtype=torch.int32, device=dev)
        words = p.peer_bytes // 4
        self.memory = torch.full(
            (WORLD, GUARD + words + GUARD), PATTERN, dtype=torch.int32, device=dev
        )
        self.banks = self.memory[:, GUARD : GUARD + words]
        self.banks.zero_()

        def base(rank):
            return self.memory[rank].data_ptr() + GUARD * 4

        self.peers = torch.tensor(
            [base(r if TP > 1 else RANK) for r in range(TP)],
            dtype=torch.int64,
            device=dev,
        )
        self.merge_peers = torch.tensor(
            [base(r if DCP > 1 else RANK) for r in range(DCP)],
            dtype=torch.int64,
            device=dev,
        )
        # past the counters, a bounded build's marks (ar.POLLS_WORD,
        # ar.GAVE_UP_WORD)
        self.epochs = torch.zeros(ar.GAVE_UP_WORD + 16, dtype=torch.int32, device=dev)
        self.epoch = 0
        self.before = None
        KEEP.append(self)

    def _merge_start(self, source, epoch):
        p = self.plan
        return lay.bank_base(p, epoch) + lay.merge_line_word(p, source, 0, 0)

    def _ar_start(self, source, epoch):
        p = self.plan
        return lay.bank_base(p, epoch) + lay.ar_line_word(p, source, 0, 0)

    def _merge_region(self, bank, source, epoch):
        """``[rows, HL, 32 lines, 4 words]`` of rank ``source`` in ``bank``."""
        first = self._merge_start(source, epoch)
        count = self.plan.merge_rows * lay.ROW_LINES * 4
        return self.banks[bank][first : first + count].view(
            self.rows, HL, lay.ROW_LINES, 4
        )

    def _ar_region(self, bank, source, epoch):
        """``[triples, HIDDEN / 2 pairs, 4 words]`` of rank ``source``."""
        first = self._ar_start(source, epoch)
        count = self.plan.ar_triples * (HIDDEN // 2) * 4
        return self.banks[bank][first : first + count].view(
            self.plan.ar_triples, HIDDEN // 2, 4
        )

    def lines(self, bank, source):
        """``[rows, HL, 128]`` BF16 and ``[rows, HL, 32]`` LSE bits of rank
        ``source``'s merge lines in ``bank`` at the last launch's parity, and
        whether all carry its epoch."""
        words = self._merge_region(bank, source, self.epoch)
        values = words[..., :2].contiguous().view(torch.bfloat16)
        current = bool((words[..., 3] == self.epoch).all())
        return values.reshape(self.rows, HL, lay.V_DIM), words[..., 2], current

    def partials(self, bank, source):
        """``[triples * 3, HIDDEN]`` BF16 of rank ``source``'s all-reduce
        lines in ``bank`` (the spare tokens past ``rows`` included), and
        whether all carry the last launch's epoch."""
        words = self._ar_region(bank, source, self.epoch)
        pairs = words[..., :3].permute(0, 2, 1).contiguous()
        values = pairs.view(torch.bfloat16).reshape(-1, HIDDEN)
        return values, bool((words[..., 3] == self.epoch).all())

    def marks(self):
        """A bounded build's give-up mark and the most polls a CTA's wave
        leader took (0: none waited)."""
        torch.cuda.synchronize()
        polls = self.epochs[ar.POLLS_WORD : ar.POLLS_WORD + ar.MAX_CTAS]
        return int(self.epochs[ar.GAVE_UP_WORD]), int(polls.max())

    def inject(self, source, values, lse, tag=None):
        """Rank ``source``'s merge rows of this rank's heads, as it would
        push them for the next launch (``tag``: carrying that epoch)."""
        words = self._merge_region(RANK, source, self.epoch + 1)
        words[..., :2] = (
            values.contiguous().view(torch.int32).view(self.rows, HL, lay.ROW_LINES, 2)
        )
        words[..., 2] = lse.contiguous().view(torch.int32)[..., None]
        words[..., 3] = self.epoch + 1 if tag is None else tag

    def inject_partial(self, source, partial, tag=None):
        """Rank ``source``'s o_proj partial ``[rows, HIDDEN]`` BF16, as it
        would push it for the next launch (the spare tokens zero)."""
        p = self.plan
        padded = torch.zeros(p.ar_triples * 3, HIDDEN, dtype=torch.bfloat16, device=dev)
        padded[: self.rows] = partial
        pairs = padded.view(torch.int32).view(p.ar_triples, 3, HIDDEN // 2)
        words = self._ar_region(RANK, source, self.epoch + 1)
        words[..., :3] = pairs.permute(0, 2, 1)
        words[..., 3] = self.epoch + 1 if tag is None else tag

    def arm(self, emulated, partials, stale=False):
        """The emulated ranks' lines for the next launch, then a snapshot of
        every rank's memory. ``stale``: the first emulated rank's lines carry
        the epoch a line left in that bank would (two before the launch's):
        lines that never arrive."""
        old = self.epoch - 1 if stale else None
        for s in range(DCP):
            if s != DCP_RANK:
                self.inject(s, emulated[0][s], emulated[1][s], old)
                old = None
        for s in range(TP):
            if s != AR_RANK:
                self.inject_partial(s, partials[s], old)
                old = None
        self.before = self.banks.clone()

    def stray(self):
        """Words of any rank's memory the last launch changed outside the
        lines this rank writes (``mla_back_layout``), and whether every guard
        band is intact."""
        p = self.plan
        allowed = torch.zeros_like(self.banks, dtype=torch.bool)
        first = self._merge_start(DCP_RANK, self.epoch)
        for owner in range(DCP):
            bank = owner if DCP > 1 else RANK
            allowed[bank, first : first + p.merge_rows * lay.ROW_LINES * 4] = True
        first = self._ar_start(AR_RANK, self.epoch)
        for rank in range(TP):
            bank = rank if TP > 1 else RANK
            allowed[bank, first : first + p.ar_triples * (HIDDEN // 2) * 4] = True
        changed = self.banks != self.before
        guards = bool(
            (self.memory[:, :GUARD] == PATTERN).all()
            and (self.memory[:, -GUARD:] == PATTERN).all()
        )
        return int((changed & ~allowed).sum()), guards

    def __call__(self, inp, layer, out=None, stamps=None):
        p = torch.Tensor.data_ptr
        out = self.out if out is None else out
        named = {
            "o": p(inp["o"]), "lse": p(inp["lse"]), "w_uv": p(layer.w_uv),
            "s_uv": p(layer.s_uv), "gate": p(inp["gate"]), "w_o": p(layer.w_o),
            "w_o_scale": p(layer.s_o), "out": p(out), "core": p(self.core),
            "flags": p(self.flags), "peers": p(self.peers),
            "merge_peers": p(self.merge_peers), "epochs": p(self.epochs),
            "rank": AR_RANK, "dcp_rank": DCP_RANK,
        }  # fmt: skip
        if stamps is None:
            args = mb.MLA_BACK_ABI.pack(named)
        else:
            args = mb.MLA_BACK_PROFILE_ABI.pack({**named, "stamps": p(stamps)})
        self.kernel.launcher(*args, stream=torch.cuda.current_stream())
        self.epoch += 1
        return out


def others(rows):
    """The emulated ranks' partial rows for this rank's heads and their LSEs
    (some rows -inf, NaN or +inf: a shard holding no KV of the row)."""
    values = bf16(DCP, rows, HL, lay.V_DIM)
    lse = torch.randn(DCP, rows, HL, generator=gen, device=dev) * 2.0 + 4.0
    drop = torch.rand(DCP, rows, HL, generator=gen, device=dev)
    lse[drop < 0.08] = float("-inf")
    lse[(drop >= 0.08) & (drop < 0.10)] = float("nan")
    lse[(drop >= 0.10) & (drop < 0.12)] = float("inf")
    if rows > 1:
        lse[:, 1, 0] = float("-inf")  # no shard holds this row
    return values, lse.float()


def other_partials(rows):
    """The emulated ranks' o_proj partials (BF16, as they push them)."""
    return bf16(TP, rows, HIDDEN, scale=3.0) if TP > 1 else None


def reference(prod, inp, layer, emulated, partials):
    """Production's W_UV rows, gated core, this rank's o_proj partial (its
    one-rank all-reduce: the rounded partial itself) and the launch's output:
    the ranks' partials summed in rank order in fp32, rounded once."""
    y = prod.w_uv(inp, layer)
    own = y[:, DCP_RANK * HL : (DCP_RANK + 1) * HL]
    if DCP == 1:
        merged = own
    else:
        values, lse = emulated
        parts = values.clone()
        parts[DCP_RANK] = own
        lses = lse.clone()
        lses[DCP_RANK] = inp["lse"][:, DCP_RANK * HL : (DCP_RANK + 1) * HL]
        merged = merge_reference(parts, lses)
    core = prod.gate(merged.contiguous(), inp)
    partial = prod.o_proj(core, layer)
    out = partial
    if TP > 1:
        total = None
        for p in range(TP):
            value = (partial if p == AR_RANK else partials[p]).float()
            total = value if total is None else total + value
        out = total.to(torch.bfloat16)
    return y, core, partial, out


def compare(back, inp, y, core, partial, out):
    exact = {
        "out": torch.equal(bits(back.out), bits(out)),
        "core": torch.equal(bits(back.core), bits(core)),
    }
    pushed = True
    for owner in range(DCP):
        got, lse, current = back.lines(owner if DCP > 1 else RANK, DCP_RANK)
        want = y[:, owner * HL : (owner + 1) * HL]
        want_lse = inp["lse"][:, owner * HL : (owner + 1) * HL]
        pushed = (
            pushed
            and current
            and torch.equal(bits(got), bits(want))
            and torch.equal(lse[..., 0], want_lse.contiguous().view(torch.int32))
            and bool((lse == lse[..., :1]).all())
        )
    exact["w_uv_lines"] = pushed
    reduced = True
    for rank in range(TP):
        got, current = back.partials(rank if TP > 1 else RANK, AR_RANK)
        reduced = (
            reduced
            and current
            and torch.equal(bits(got[: back.rows]), bits(partial))
            and not bool(got[back.rows :].contiguous().view(torch.int16).any())
        )
    exact["ar_lines"] = reduced
    exact["stray"], exact["guards"] = back.stray()
    return exact


def passed(exact):
    return (
        exact["out"]
        and exact["core"]
        and exact["w_uv_lines"]
        and exact["ar_lines"]
        and exact["guards"]
        and exact["stray"] == 0
    )


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


def check(rows):
    back, prod = Back(rows), Production(rows)
    res, hazards = resources(back.kernel)
    print(json.dumps({"rows": rows, "kernel": back.kernel.name, "resources": res,
                      "hazards": hazards}), flush=True)  # fmt: skip
    layer = LAYERS[0]
    report = {"rows": rows, "tp": TP, "dcp": DCP, "rank": RANK, "eager": [],
              "ok": not hazards}  # fmt: skip
    last = None
    for _ in range(ITERS):
        inp = inputs(rows)
        emulated = others(rows) if DCP > 1 else None
        partials = other_partials(rows)
        back.arm(emulated, partials)
        back(inp, layer)
        torch.cuda.synchronize()
        y, core, partial, out = reference(prod, inp, layer, emulated, partials)
        torch.cuda.synchronize()
        exact = compare(back, inp, y, core, partial, out)
        if not exact["core"]:
            d = (back.core.float() - core.float()).abs()
            exact["core_mismatched"] = int((bits(back.core) != bits(core)).sum())
            finite = d[torch.isfinite(d)]
            exact["core_max_abs"] = float(finite.max()) if finite.numel() else 0.0
        if not exact["out"]:
            exact["out_mismatched"] = int((bits(back.out) != bits(out)).sum())
        exact["moving"] = last is None or not torch.equal(back.out, last)
        last = back.out.clone()
        report["eager"].append(exact)
        report["ok"] = report["ok"] and passed(exact)
    # one graph, REPLAYS replays on new inputs in place
    inp = inputs(rows)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    epoch = back.epoch
    with torch.cuda.graph(graph, stream=stream):
        back(inp, layer)
    back.epoch = epoch  # the capture launched nothing
    torch.cuda.current_stream().wait_stream(stream)
    KEEP.append((graph, stream, inp))
    bad = 0
    for _ in range(REPLAYS):
        new = inputs(rows)
        for key in inp:
            inp[key].copy_(new[key])
        emulated = others(rows) if DCP > 1 else None
        partials = other_partials(rows)
        back.arm(emulated, partials)
        torch.cuda.synchronize()
        graph.replay()
        back.epoch += 1
        torch.cuda.synchronize()
        y, core, partial, out = reference(prod, inp, layer, emulated, partials)
        torch.cuda.synchronize()
        bad += not passed(compare(back, inp, y, core, partial, out))
    report["replays"] = REPLAYS
    report["replays_mismatched"] = bad
    report["ok"] = report["ok"] and bad == 0
    if BOUND is not None:
        gave, polls = back.marks()
        report["gave_up"] = hex(gave)
        report["max_polls"] = polls
        report["ok"] = report["ok"] and gave == 0
        if LATE is not None:
            report["late"] = LATE
            report["late_seen"] = polls >= LATE // 4
            report["ok"] = report["ok"] and report["late_seen"]
        if WORLD > 1:
            # the control: an emulated rank's lines never arrive
            inp = inputs(rows)
            back.arm(others(rows) if DCP > 1 else None, other_partials(rows), True)
            torch.cuda.synchronize()
            start = time.time()
            back(inp, layer)
            gave, _ = back.marks()
            report["stale_gives_up"] = (gave & 0x7FF00000) == ar.GAVE_UP
            report["stale_s"] = round(time.time() - start, 3)
            report["stale_guards"] = back.stray()[1]
            report["ok"] = (
                report["ok"] and report["stale_gives_up"] and report["stale_guards"]
            )
    print(json.dumps(report), flush=True)
    return back, prod


def graph_of(fn):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    KEEP.append((g, s))
    return g


def timed(g):
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    start.record()
    g.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / GN


def bench(rows, back, prod):
    """DCP=1 only: the launches timed need no peer."""
    inps = [inputs(rows) for _ in range(COPIES)]
    outs = [
        torch.empty(rows, HIDDEN, dtype=torch.bfloat16, device=dev) for _ in range(4)
    ]
    KEEP.extend([inps, outs])

    def run_back():
        for i in range(GN):
            back(inps[i % COPIES], LAYERS[i % COPIES], outs[i % 4])

    def run_prod():
        for i in range(GN):
            inp, layer = inps[i % COPIES], LAYERS[i % COPIES]
            y = prod.w_uv(inp, layer)
            core = prod.gate(y, inp)
            KEEP.append((y, core, prod.o_proj(core, layer)))

    def run_prod_tail():
        for i in range(GN):
            inp, layer = inps[i % COPIES], LAYERS[i % COPIES]
            KEEP.append(prod.o_proj(prod.gate(inp["y"], inp), layer))

    for inp in inps:
        inp["y"] = bf16(rows, HL, lay.V_DIM)
    graphs = {
        "back": graph_of(run_back),
        "production": graph_of(run_prod),
        "production_gate_oproj": graph_of(run_prod_tail),
    }
    times = {name: [] for name in graphs}
    for _ in range(REPS):
        for name, g in graphs.items():
            times[name].append(timed(g))
    med = {name: round(statistics.median(t), 2) for name, t in times.items()}
    spread = {name: round(max(t) - min(t), 2) for name, t in times.items()}
    print(json.dumps({"rows": rows, "us_per_layer": med, "spread": spread}), flush=True)


def profile(rows):
    """A profile build's stamps, one launch a graph replay on new inputs
    (the emulated peers' lines injected before each): each CTA's phases from
    its own start (a GPU's two dies do not share a clock), over the CTAs and
    the replays past the first four, in us."""
    back = Back(rows, profile=True)
    layer = LAYERS[0]
    stamps = torch.zeros(lay.GRID * mb.STAMPS, dtype=torch.int32, device=dev)
    inp = inputs(rows)
    KEEP.extend([stamps, inp])
    back.arm(others(rows) if DCP > 1 else None, other_partials(rows))
    back(inp, layer, stamps=stamps)
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    epoch = back.epoch
    with torch.cuda.graph(graph, stream=stream):
        back(inp, layer, stamps=stamps)
    back.epoch = epoch
    torch.cuda.current_stream().wait_stream(stream)
    KEEP.append((graph, stream))
    taken = []
    for k in range(PROFILE_LAUNCHES):
        new = inputs(rows)
        for key in inp:
            inp[key].copy_(new[key])
        back.arm(others(rows) if DCP > 1 else None, other_partials(rows))
        stamps.zero_()
        torch.cuda.synchronize()
        graph.replay()
        back.epoch += 1
        torch.cuda.synchronize()
        if k >= 4:
            taken.append(stamps.cpu().long().view(lay.GRID, mb.STAMPS))
    s = torch.stack(taken)
    rel = (s - s[:, :, :1]) & 0xFFFFFFFF  # 10 ns ticks from the CTA's start
    names = ["uv_pushed", "lines_seen", "core_stored", "core_seen", "staged",
             "summed", "end"]  # fmt: skip
    report = {"rows": rows, "tp": TP, "dcp": DCP, "rank": RANK,
              "kernel": back.kernel.name}  # fmt: skip
    for i, name in enumerate(names, start=1):
        ctas = slice(0, lay.OPROJ_CTAS) if i >= mb.STAMP_CORE_SEEN else slice(None)
        v = rel[:, ctas, i].float()
        v = v[s[:, ctas, i] != 0]
        if v.numel():
            report[name] = {
                "p50": round(float(v.median()) / 100.0, 2),
                "p90": round(float(v.quantile(0.9)) / 100.0, 2),
                "max": round(float(v.max()) / 100.0, 2),
            }
    print(json.dumps({"profile": report}), flush=True)


def bench_production(rows, prod):
    """The replaced launches a rank can run alone, in the emulated group's
    shape: W_UV on the group's heads, the gate, o_proj + AR among one rank,
    and the three in sequence (the merge between them needs peers)."""
    inps = [inputs(rows) for _ in range(COPIES)]
    for inp in inps:
        inp["y"] = bf16(rows, HL, lay.V_DIM)
        inp["core"] = bf16(rows, PROJ)
    KEEP.append(inps)
    mine = slice(DCP_RANK * HL, (DCP_RANK + 1) * HL)

    def w_uv(inp, layer):
        return prod.w_uv(inp, layer)

    def gate(inp, layer):
        return prod.gate(inp["y"], inp)

    def o_proj(inp, layer):
        return prod.o_proj(inp["core"], layer)

    def sequence(inp, layer):
        y = prod.w_uv(inp, layer)
        core = prod.gate(y[:, mine].contiguous(), inp)
        return y, prod.o_proj(core, layer)

    def graph(step):
        def run():
            for i in range(GN):
                KEEP.append(step(inps[i % COPIES], LAYERS[i % COPIES]))

        return graph_of(run)

    graphs = {
        "w_uv": graph(w_uv),
        "gate": graph(gate),
        "oproj_ar": graph(o_proj),
        "w_uv+gate+oproj_ar": graph(sequence),
    }
    times = {name: [] for name in graphs}
    for _ in range(REPS):
        for name, g in graphs.items():
            times[name].append(timed(g))
    med = {name: round(statistics.median(t), 2) for name, t in times.items()}
    spread = {name: round(max(t) - min(t), 2) for name, t in times.items()}
    print(json.dumps({"rows": rows, "heads": HEADS, "proj": PROJ,
                      "production_us": med, "spread": spread}), flush=True)  # fmt: skip


for rows in ROWS:
    if PROFILE:
        profile(rows)
        continue
    if CHECK:
        back, prod = check(rows)
    else:
        back, prod = Back(rows), Production(rows)
    if BENCH and WORLD == 1:
        bench(rows, back, prod)
    elif BENCH:
        bench_production(rows, prod)
