"""Layer kernel builders.

:func:`build_layer` compiles one launch running a prefix-closed subset of the
layer's stages (``layout.ORDERS``). The attention half (A1 -> A2 -> B1 -> B3 ->
C1) is the subset :data:`ATTN_HALF`; without A1/A2 the host fills the A2
mailboxes (the HC's mixed row and injection weights) instead. :data:`DELTA` is
the whole DeltaNet layer. Arguments of stages that are left out are unused.

Every argument is a device address (int64) except ``layer``. ``step`` points
at the device step counter that the launch tag is derived from. ``qsa`` points
at the QSA layer's state table (``qsa.STATE``: the cache length, the caches,
the RoPE tables and the norm weights), so a cache that grows only changes the
table. The ``e_*``
arguments hold one block per expert (``moe.*_BYTES`` apart); the MoE stages
build an expert's buffer resources from the raw address (``p.a_<name>``).
"""

import functools
from types import SimpleNamespace

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import Int32, Int64, T

from atom.models.qwen4_exp.mono.kernels import buffer_ops as bo
from atom.models.qwen4_exp.mono.kernels import mailbox as mbx
from atom.models.qwen4_exp.mono.kernels.config import DEFAULT, GRID, THREADS
from atom.models.qwen4_exp.mono.kernels.device_ops import uniform
from atom.models.qwen4_exp.mono.kernels.layout import (
    MAX_ROWS,
    ORDERS,
    SC,
    SCRATCH_BYTES,
    SPLIT_GROUPS,
    WAVES,
    for_rows,
)
from atom.models.qwen4_exp.mono.kernels.layout_wide import barrier_before, for_wide
from atom.models.qwen4_exp.mono.kernels.stages import (
    delta_group,
    delta_group_wide,
    hc_wide,
    moe_wide,
    out_proj_wide,
    ple_group,
    ple_wide,
    qsa_fast,
    qsa_fast_wide,
    qsa_group,
    qsa_wide,
)
from atom.models.qwen4_exp.mono.kernels.stages.common import task_index
from atom.models.qwen4_exp.mono.kernels.stages.common_wide import fresh_ids
from atom.models.qwen4_exp.mono.kernels.stages.delta import b1, b3
from atom.models.qwen4_exp.mono.kernels.stages.hc import RECORDS as HC_RECORDS
from atom.models.qwen4_exp.mono.kernels.stages.hc import (
    a1,
    a1_preload,
    a2,
    attn_hc,
    mlp_hc,
)
from atom.models.qwen4_exp.mono.kernels.stages.moe import m3, m4, m5
from atom.models.qwen4_exp.mono.kernels.stages.out_proj import c1
from atom.models.qwen4_exp.mono.kernels.stages.ple import ple_mix, ple_proj
from atom.models.qwen4_exp.mono.kernels.stages.qsa import (
    QSA_STATE,
    q1,
    q2,
    q5,
    q6,
    q7,
    qsa_len,
    qsa_state,
)
from atom.models.qwen4_exp.mono.sources import SOURCES

LAYER_ARGS = (
    "scratch",
    "ple",
    "h_in",
    "hc_norm_w",
    "w_down",
    "s_down",
    "w_up",
    "s_up",
    "w_inj",
    "w_qkvz",
    "s_qkvz",
    "conv_w",
    "conv_in",
    "conv_out",
    "ba_w",
    "a_log",
    "dt_bias",
    "norm_w",
    "rec_in",
    "rec_out",
    "w_qsa",
    "s_qsa",
    "qsa",
    "w_out",
    "s_out",
    "m_norm_w",
    "m_w_down",
    "m_s_down",
    "m_w_up",
    "m_s_up",
    "m_w_inj",
    "r_w",
    "e_w_gu",
    "e_s_gu",
    "sh_w_gu",
    "sh_s_gu",
    "e_w_dn",
    "e_s_dn",
    "sh_w_dn",
    "sh_s_dn",
    "h_out",
)
RAW_ARGS = (
    "e_w_gu", "e_s_gu", "sh_w_gu", "sh_s_gu", "e_w_dn", "e_s_dn", "sh_w_dn", "sh_s_dn", "qsa",
    "ple",
)  # fmt: skip
QKVZ_N = DEFAULT.delta_qkv + DEFAULT.delta_value
RECORDS = {
    **HC_RECORDS,
    **{"m_" + k: v for k, v in HC_RECORDS.items()},
    "h_in": DEFAULT.hc_width * 2,  # P2's waves 4-7 read it out of range
    # B1, Q1, C1: waves 4-7 read the weights and scales out of range.
    **{
        w: DEFAULT.hidden * n // 2
        for w, n in (("w_qkvz", QKVZ_N), ("w_qsa", DEFAULT.qsa_proj))
    },
    **{
        s: DEFAULT.hidden * n // 32
        for s, n in (("s_qkvz", QKVZ_N), ("s_qsa", DEFAULT.qsa_proj))
    },
    "w_out": DEFAULT.hidden * DEFAULT.delta_value // 2,
    "s_out": DEFAULT.hidden * DEFAULT.delta_value // 32,
}
# Kernel head of the builds that start with A1: CTAs with neither an A1 nor
# an A2 task sleep ``HEAD_SLEEP`` s_sleep units (64 clocks each) first, so that
# A1's h and w' loads meet less of the other stages' weight streams.
HEAD_SLEEP = 64


def _sleep(n):
    for _ in range_constexpr(n // 127):
        rocdl.s_sleep(127)
    if const_expr(n % 127):
        rocdl.s_sleep(n % 127)


def _head_sleep(c):
    a1t = task_index(c, "a1") < c.L.tasks["a1"]
    a2t = task_index(c, "a2") < c.L.tasks["a2"]
    first = a1t.select(
        fx.Int32(0), a2t.select(fx.Int32(1), fx.Int32(2))
    )  # A1, A2, neither
    if first == 2:
        _sleep(HEAD_SLEEP)


_head_sleep = ASTRewriter.transform(_head_sleep)

ATTN_HALF = ("a1", "a2", "b1", "b3", "c1")
DELTA_HC = ATTN_HALF + ("m1", "m2")
DELTA = DELTA_HC + ("m3", "m4", "m5")
QSA = ORDERS["qsa"]
# The stages a layer of each kind launches, in order.
STAGES = {"delta": DELTA, "delta_ple": ORDERS["delta_ple"], "qsa": QSA}


def _context(kind, step, layer, scratch, lds, experts, rows, seq, group, early=False):
    tid = fx.thread_idx.x
    lds_i = lds.w.ptr
    c = SimpleNamespace(
        kind=kind,
        rows=rows,
        seq=seq,
        group=group,
        L=for_wide(rows, group) if group else for_rows(rows, seq),
        tid=tid,
        lane=tid % 64,
        wave=uniform(tid // 64),
        bid=fx.block_idx.x,
        tag=None if early else mbx.launch_tag(step, layer),
        pre=None,
        lds_i=lds_i,
        lds_f=fx.recast_iter(fx.Float32, lds_i),
        experts=experts,
        scratch=scratch,
    )
    if early:
        c.step_word = bo.buffer_load(mbx.rsrc(step), 0, vec_width=1, dtype=T.i32)
    _mailboxes(c, scratch)
    return c


def _mailboxes(c, scratch):
    for name, off in SC.items():
        setattr(c, "mb_" + name, scratch + off)


def _fresh(x):
    """A copy of the uniform ``x`` that LLVM cannot merge with another copy.
    Each stage takes a fresh ``wave``, so that index math derived from it is
    recomputed per stage rather than held in SGPRs from one stage to a later
    one running the same code (A2 and M2)."""
    asm = llvm.InlineAsmOp(
        T.i32, [fx.Int32(x).ir_value()], "; fresh $0", "=s,0", has_side_effects=True
    )
    return fx.Int32(asm.result)


def _fresh64(x):
    asm = llvm.InlineAsmOp(
        T.i64, [fx.Int64(x).ir_value()], "; fresh $0", "=s,0", has_side_effects=True
    )
    return fx.Int64(asm.result)


def _kernargs():
    """The kernarg segment pointer, as a copy LLVM cannot merge with another
    stage's: loads through it are not merged with (held from) another
    stage's loads of the same argument."""
    ptr = ir.Type.parse("!llvm.ptr<4>")
    seg = llvm.call_intrinsic(ptr, "llvm.amdgcn.kernarg.segment.ptr", [], [], [])
    addr = _fresh64(fx.Int64(llvm.PtrToIntOp(T.i64, seg).result))
    return llvm.IntToPtrOp(ptr, addr.ir_value()).result


def _kernarg(seg, name):
    """Kernel argument ``name`` (of :data:`LAYER_ARGS`, at byte ``16 + 8 k``
    after ``step`` and ``layer``), loaded through ``seg`` (:func:`_kernargs`)."""
    ptr = bo.get_element_ptr(seg, static_byte_offset=16 + 8 * LAYER_ARGS.index(name))
    return fx.Int64(llvm.LoadOp(T.i64, ptr, alignment=8, invariant=True).result)


def _stage_mailboxes(c, stage, scratch):
    """``c.mb_<name>`` (:func:`_mailboxes`) for ``stage``: a wide build's
    index score region starts past its rows."""
    _mailboxes(c, scratch)
    if c.rows > MAX_ROWS and stage in ("q3", "q4"):
        c.mb_isc = c.mb_isc + (c.rows - MAX_ROWS) * SCRATCH_BYTES


def _round_mailboxes(stage, c):
    """Mailbox addresses for a wide stage's further task round
    (``common_wide.fresh_round``), from a fresh load of ``scratch``: LLVM
    would otherwise hold (spill) the first round's through the round."""
    _stage_mailboxes(c, stage, _kernarg(_kernargs(), "scratch"))


def _run(stage, c, p, stages):
    # Wide builds (more than MAX_ROWS rows) run the ``*_wide`` stages.
    wide = c.rows > MAX_ROWS
    c.wave = _fresh(c.wave)
    if c.rows > WAVES:
        fresh_ids(c)
    seg = _kernargs() if wide else None
    # Mailbox addresses too: LLVM would otherwise compute ``scratch + off``
    # once and hold (or spill) it across the stages using the region.
    _stage_mailboxes(
        c, stage, _kernarg(seg, "scratch") if wide else _fresh64(c.scratch)
    )
    if wide:
        c.round_mailboxes = functools.partial(_round_mailboxes, stage)
        # LLVM merges the kernel arguments' loads at entry into
        # ``s_load_dwordx16``s and holds (spills) them across the stages. A
        # wide build loads each argument in the stages using it (the others'
        # loads are dropped), so its entry copies are dead.
        records = dict(RECORDS, h_in=RECORDS["h_in"] * c.rows)
        for name in LAYER_ARGS[1:]:
            addr = _kernarg(seg, name)
            if name in RAW_ARGS:
                setattr(p, "a_" + name, addr)
            else:
                setattr(
                    p,
                    name,
                    bo.create_buffer_resource_from_addr(
                        addr, num_records_bytes=records.get(name)
                    ),
                )
    if stage in QSA_STATE:
        qsa_state(p)
        if wide:
            # Its rows' ``L`` too (table values), not at kernel entry.
            qsa_group.qsa_len(p, c.rows, c.group)
    if stage == "a1":
        a1(c, attn_hc(c, p))
    elif stage == "a2":
        (hc_wide.a2 if wide else a2)(c, attn_hc(c, p))
    elif stage == "b1":
        (delta_group_wide.b1 if wide else delta_group.b1 if c.group else b1)(c, p)
    elif stage == "b3":
        (delta_group_wide.b3 if wide else delta_group.b3 if c.group else b3)(c, p)
    elif stage == "c1":
        (out_proj_wide.c1 if wide else c1)(c, p, store_out="m5" not in stages)
    elif stage == "m1":
        a1(c, mlp_hc(c, p))
    elif stage == "m2":
        (hc_wide.a2 if wide else a2)(c, mlp_hc(c, p))
    elif stage == "m3":
        (moe_wide.m3 if wide else m3)(c, p)
    elif stage == "m4":
        (moe_wide.m4 if wide else m4)(c, p)
    elif stage == "m5":
        (moe_wide.m5 if wide else m5)(c, p)
    elif stage == "q1":
        (qsa_wide.q1 if wide else q1)(c, p)
    elif stage == "q2":
        # Past WAVES rows: in passes of WAVES request rows (``qsa_wide``).
        (qsa_wide.q2 if c.rows > WAVES else q2)(c, p)
    elif stage == "q3":
        (qsa_group.q3 if c.group else qsa_fast.q3)(c, p)
    elif stage == "q4":
        (qsa_fast_wide if wide else qsa_fast).q4(c, p)
    elif stage == "q5":
        (qsa_group.q5 if c.group else q5)(c, p)
    elif stage == "q6":
        # Past WAVES rows: more tasks than CTAs, in rounds (``qsa_wide``).
        (qsa_wide.q6 if c.rows > WAVES else q6)(c, p)
    elif stage == "q7":
        (qsa_group.q7 if c.group else q7)(c, p)
    elif stage == "p1":
        (ple_wide.ple_proj if wide else ple_proj)(c, p)
    elif stage == "p2":
        (ple_group.ple_mix if c.group else ple_mix)(c, p)
    else:
        raise ValueError(stage)


@functools.cache
def build_layer(stages=DELTA, experts=DEFAULT.experts, rows=1, seq=False, group=0):
    """Launcher ``(step, layer, *LAYER_ARGS, stream)`` running ``stages``
    for a model with ``experts`` routed experts (512, or 16 in the CI fixture).

    ``rows`` is the number of requests the launch decodes, one token each
    (``layout.for_rows``); each request's results are bit-identical to its
    own ``rows=1`` launch.

    ``seq=True`` (``rows`` >= 2, an MTP verify launch): the rows are
    consecutive tokens of one request instead, row ``r`` at position
    ``L0 + r``. Row ``r``'s results are bit-identical to the ``r``-th of
    ``rows`` sequential ``rows=1`` launches. The DeltaNet and PLE state
    tables hold per row the state after that row (snapshots); only row 0's
    input state is read. The QSA table has one row, the request's.

    ``group=S`` (``S`` >= 2, ``rows`` a multiple of ``S`` with two groups at
    least; not with ``seq``): an MTP verify launch of ``rows / S`` requests,
    rows ``g S .. g S + S - 1`` the ``S`` consecutive tokens of request ``g``.
    Each group's rows are bit-identical to its own ``S``-row seq launch
    (``stages.*_group``). Every table row is its own: the DeltaNet and PLE
    rows' input state is their group's, the QSA rows' pages and caches their
    request's (``group_meta``); a group's ``L`` is its first row's.

    QSA builds index contexts up to ``qsa_fast.MAX_CONTEXT`` tokens. Every
    build takes a scratch buffer of ``layout.scratch_bytes(rows)``.
    """
    # A run of the kind's stages from A1, or from the stage after A2 with the
    # host filling A2's mailboxes; QSA tests may start anywhere after A2, PLE
    # tests at P1 or P2.
    if any(s.startswith("q") for s in stages):
        kind = "qsa"
    elif any(s in ("p1", "p2") for s in stages):
        kind = "delta_ple"
    else:
        kind = "delta"
    order = ORDERS[kind]
    first = order.index(stages[0])
    starts = {"delta": (0, 2), "qsa": (0, 2), "delta_ple": (0, 1)}[kind]
    assert first in starts or (kind == "qsa" and first > 2), stages
    assert stages == order[first : first + len(stages)], stages
    assert experts % 8 == 0 and 16 <= experts <= DEFAULT.experts, experts
    assert not seq or rows > 1, rows
    assert not group or (
        not seq and group >= 2 and rows % group == 0 and rows >= 2 * group
    )
    lds_words = (for_wide(rows, group) if group else for_rows(rows, seq)).lds_words
    records = dict(RECORDS, h_in=RECORDS["h_in"] * rows)
    # Narrow builds that start with A1 issue its h and w' loads at the entry,
    # before the step counter's load is waited on (``a1_preload``).
    early = stages[0] == "a1" and rows <= MAX_ROWS
    if early:
        records["hc_norm_w"] = DEFAULT.hc_width * 2  # a1_preload reads it out of range
    # Every build parameter and the kernels' sources, in FlyDSL's cache key
    # (``atom.mono.plan.build_key``).
    keyed = (
        ("stages", tuple(stages)),
        ("experts", experts),
        ("rows", rows),
        ("seq", seq),
        ("sources", SOURCES),
    ) + ((("group", group),) if group else ())

    @fx.struct
    class Smem:
        w: fx.Array[fx.Int32, lds_words, 16]

    # Named after its stages so that dumps and traces of the variants differ.
    @flyc.kernel(
        name="qwen4_"
        + "_".join(stages)
        + f"_e{experts}"
        + (f"_r{rows}" if rows > 1 else "")
        + ("_s" if seq else "")
        + (f"_g{group}" if group else ""),
        known_block_size=[THREADS, 1, 1],
    )
    def qwen4_layer(
        step: Int64,
        layer: Int32,
        scratch: Int64,
        ple: Int64,
        h_in: Int64,
        hc_norm_w: Int64,
        w_down: Int64,
        s_down: Int64,
        w_up: Int64,
        s_up: Int64,
        w_inj: Int64,
        w_qkvz: Int64,
        s_qkvz: Int64,
        conv_w: Int64,
        conv_in: Int64,
        conv_out: Int64,
        ba_w: Int64,
        a_log: Int64,
        dt_bias: Int64,
        norm_w: Int64,
        rec_in: Int64,
        rec_out: Int64,
        w_qsa: Int64,
        s_qsa: Int64,
        qsa: Int64,
        w_out: Int64,
        s_out: Int64,
        m_norm_w: Int64,
        m_w_down: Int64,
        m_s_down: Int64,
        m_w_up: Int64,
        m_s_up: Int64,
        m_w_inj: Int64,
        r_w: Int64,
        e_w_gu: Int64,
        e_s_gu: Int64,
        sh_w_gu: Int64,
        sh_s_gu: Int64,
        e_w_dn: Int64,
        e_s_dn: Int64,
        sh_w_dn: Int64,
        sh_s_dn: Int64,
        h_out: Int64,
    ):
        lds = fx.SharedAllocator().allocate(Smem).peek()
        c = _context(kind, step, layer, scratch, lds, experts, rows, seq, group, early)
        args = (
            ple, h_in, hc_norm_w, w_down, s_down, w_up, s_up, w_inj, w_qkvz, s_qkvz, conv_w, conv_in,
            conv_out, ba_w, a_log, dt_bias, norm_w, rec_in, rec_out, w_qsa, s_qsa, qsa, w_out,
            s_out, m_norm_w, m_w_down, m_s_down, m_w_up, m_s_up, m_w_inj, r_w, e_w_gu, e_s_gu,
            sh_w_gu, sh_s_gu, e_w_dn, e_s_dn, sh_w_dn, sh_s_dn, h_out,
        )  # fmt: skip
        p = SimpleNamespace(
            **{
                name: bo.create_buffer_resource_from_addr(
                    a, num_records_bytes=records.get(name)
                )
                for name, a in zip(LAYER_ARGS[1:], args)
                if name not in RAW_ARGS
            },
            **{
                "a_" + name: a
                for name, a in zip(LAYER_ARGS[1:], args)
                if name in RAW_ARGS
            },
        )
        if const_expr(early):
            c.pre = a1_preload(c, p)
            rocdl.sched_barrier(0)  # the loads issue before the step word is waited on
            c.tag = uniform(c.step_word) * mbx.LAYER_SLOTS + layer + 1
        if const_expr(stages[0] == "a1"):
            _head_sleep(c)
        if const_expr(rows <= MAX_ROWS and any(s in QSA_STATE for s in stages)):
            if const_expr(group):
                qsa_group.qsa_len(p, rows, group)
            else:
                qsa_len(p, rows, seq)
        for i in range_constexpr(len(stages)):
            if const_expr(c.L.split and i > 0):
                halves = {SPLIT_GROUPS.get(s) for s in stages[i - 1 : i + 1]}
                if const_expr(halves == {0, 1}):
                    gpu.barrier()  # the other half's LDS entries overlay this one's
            if const_expr(rows > MAX_ROWS and barrier_before(stages, i, c.L)):
                gpu.barrier()  # the last LDS phase's entries overlay this one's
            _run(stages[i], c, p, stages)
        _ = keyed

    @flyc.jit
    def launch(
        step: Int64,
        layer: Int32,
        scratch: Int64,
        ple: Int64,
        h_in: Int64,
        hc_norm_w: Int64,
        w_down: Int64,
        s_down: Int64,
        w_up: Int64,
        s_up: Int64,
        w_inj: Int64,
        w_qkvz: Int64,
        s_qkvz: Int64,
        conv_w: Int64,
        conv_in: Int64,
        conv_out: Int64,
        ba_w: Int64,
        a_log: Int64,
        dt_bias: Int64,
        norm_w: Int64,
        rec_in: Int64,
        rec_out: Int64,
        w_qsa: Int64,
        s_qsa: Int64,
        qsa: Int64,
        w_out: Int64,
        s_out: Int64,
        m_norm_w: Int64,
        m_w_down: Int64,
        m_s_down: Int64,
        m_w_up: Int64,
        m_s_up: Int64,
        m_w_inj: Int64,
        r_w: Int64,
        e_w_gu: Int64,
        e_s_gu: Int64,
        sh_w_gu: Int64,
        sh_s_gu: Int64,
        e_w_dn: Int64,
        e_s_dn: Int64,
        sh_w_dn: Int64,
        sh_s_dn: Int64,
        h_out: Int64,
        stream: fx.Stream = fx.Stream(None),
    ):
        _ = keyed
        qwen4_layer(
            step, layer, scratch, ple, h_in, hc_norm_w, w_down, s_down, w_up, s_up, w_inj, w_qkvz,
            s_qkvz, conv_w, conv_in, conv_out, ba_w, a_log, dt_bias, norm_w, rec_in, rec_out, w_qsa,
            s_qsa, qsa, w_out, s_out, m_norm_w, m_w_down, m_s_down, m_w_up, m_s_up, m_w_inj, r_w,
            e_w_gu, e_s_gu, sh_w_gu, sh_s_gu, e_w_dn, e_s_dn, sh_w_dn, sh_s_dn, h_out,
        ).launch(grid=(GRID,), block=(THREADS,), stream=stream)  # fmt: skip

    return launch
