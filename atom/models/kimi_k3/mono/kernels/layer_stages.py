# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Ported from vLLM PR #60683, vllm/models/kimi_k3/amd/mono/layer.py, whose own device layer was a
# fork of ATOM's atom/mono (MIT, Copyright (C) 2026, Advanced Micro
# Devices, Inc.). This port reads those primitives from atom.mono
# directly and takes ATOM's mailbox discipline in place of the fork's.
"""K2's o_proj and AttnRes stages, as ``layer._body`` places them on a CTA:
o_proj's tasks (``o_slot``; after the CTA's MLA post task in an ``mla_post``
build), a token group's AttnRes seam, and the routes after it."""

from flydsl.expr import range_constexpr

from atom.models.kimi_k3.mono.kernels.attention import back as k2a
from atom.models.kimi_k3.mono.kernels.attention.kda import (
    NA,
    slice_compute,
    slice_loads,
    stage_gate,
    stage_mix,
)
from atom.models.kimi_k3.mono.kernels.stages import mla_post as mp
from atom.models.kimi_k3.mono.kernels.stages import moe as k2b
from atom.mono.device import gemv
from atom.mono.device.ops import traced
from atom.mono.device.stamps import stamp
from atom.mono.plan.execution import BLOCKS
from atom.mono.plan.trace import enter_stage

O0 = NA  # the CTAs past AttnRes's: o_proj's first round, then the early GEMVs
OTASKS = k2a.OTASKS  # 448
O_ROUNDS = (OTASKS + BLOCKS - 1) // BLOCKS  # 2: every CTA takes o_proj tasks


def o_slot(bid):
    """CTA bid's first o_proj task: CTAs O0 .. take 0 .., the AttnRes CTAs the
    last 56 slots, whose second task is past OTASKS (one task each: their
    reduce waits on the slowest partial)."""
    return (bid + BLOCKS - O0) % BLOCKS


@traced
def stage_oproj(c, gi):
    """o_proj tasks gi, gi + 256 (< 448): every task's weights in flight at
    once (one memory latency, not two), then the MFMAs, a token group's core
    rows at a time. A CTA with one task loads its own slab again for the
    round it skips (cache hits), not task 447's on 64 CTAs at once."""
    # K2's bf16 GEMV weights (here, the MoE's early / shared down / up) a load
    # two slabs of 8 rows (``gemv.chunk_ops`` paired)
    ops = [
        k2a.o_loads(c, k2a.o_task_or(gi, gi + BLOCKS * j), paired=True)
        for j in range(O_ROUNDS)
    ]
    for g in range_constexpr(c["G"]):
        # ``stage_o`` ends at a barrier: the previous group's core rows are read
        cg = k2a.group_view(c, g)
        k2a.load_core(cg)
        for j in range_constexpr(O_ROUNDS):
            t = gi + BLOCKS * j
            if t < OTASKS:
                k2a.stage_o(cg, t, ops[j])


@traced
def stage_oproj_post(c, gi, task):
    """``stage_oproj`` reading the core this launch's MLA post stage writes:
    this CTA's post task (if any), the o_proj weights in flight across the
    wait for every task's ``mpdone`` pair, then a token group at a time its
    core rows at device scope and the MFMAs. (The weights issued before the
    post task stay live across it: one VGPR spilled from 4 rows.)"""
    # one stage on every CTA: a task publishes its pair before the CTA waits
    # on all of them (an exchange, as an all-reduce's push and poll)
    enter_stage("k2.mlapost")
    if task >= 0:
        mp.post_task(c, task)
    ops = [
        k2a.o_loads(c, k2a.o_task_or(gi, gi + BLOCKS * j), paired=True)
        for j in range(O_ROUNDS)
    ]
    mp.wait_core(c)
    enter_stage("k2.o")
    for g in range_constexpr(c["G"]):
        cg = k2a.group_view(c, g)
        mp.load_core_dev(cg, 0, cg["S"], k2a.OK, gemv.LDS_PAD)
        for j in range_constexpr(O_ROUNDS):
            t = gi + BLOCKS * j
            if t < OTASKS:
                k2a.stage_o(cg, t, ops[j])


def oproj(c, key, bid):
    """CTA ``bid``'s o_proj: ``stage_oproj``, or in an ``mla_post`` build
    ``stage_oproj_post`` (with the CTA's post task)."""
    if key.mla_post:
        stage_oproj_post(c, o_slot(bid), mp.task_of(bid))
    else:
        stage_oproj(c, o_slot(bid))


@traced
def stage_attnres(c, task, g, on, tls):
    """Task ``task`` (< NA) of token group g's AttnRes seam: the reduced o_proj
    -> slice -> gate (tasks 0 .. 15) -> mix -> x rows, XRDY. Group g runs on
    CTAs 56 g ..: the groups' hand-off chains side by side."""
    cg = k2a.group_view(c, g)
    enter_stage("k2.ar")
    # the slice's sources and weights in flight across the reduce's wait
    sld = slice_loads(cg, task)
    k2a.stage_reduce(cg, task)
    stamp(on, tls, c["tid"], 1)
    enter_stage("k2.slice")
    slice_compute(cg, task, sld)
    stamp(on, tls, c["tid"], 2)
    enter_stage("k2.gate")
    if task < cg["S"]:
        stage_gate(cg, task)
    stamp(on, tls, c["tid"], 8)
    enter_stage("k2.mix")
    stage_mix(cg, task)
    stamp(on, tls, c["tid"], 9)


@traced
def stage_attnres_far(c, bid, on, tls):
    """CTA bid's AttnRes task of the groups past the first (CTAs 56 .., after
    their o_proj, before their early GEMV: that waits for every group's x)."""
    for g in range_constexpr(1, c["G"]):
        if (bid >= NA * g) & (bid < NA * (g + 1)):
            stage_attnres(c, bid - NA * g, g, on, tls)


@traced
def stage_routes(c, bid):
    """CTA bid (< NA) after the AttnRes seam: route tokens bid, bid + 56, .."""
    s = c["S"]
    enter_stage("k2.route")
    for i in range_constexpr(-(-s // NA)):
        if bid + NA * i < s:
            k2b.stage_route(c, bid + NA * i)
    stamp(c["tl_on"], c["tls"], c["tid"], 10)
