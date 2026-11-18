# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx1250 BF16 GEMV that streams both operands through the Tensor Data Mover.

A CTA owns up to ``tasks`` 16-row weight tasks and every token row. K is walked
in 512-column slabs: the TDM copies each slab's token rows into one of two LDS
buffers and each task's 16 weight rows into one of ``DEPTH + 1`` buffers, so
``DEPTH`` weight slabs and the next token slab are in flight while the waves
multiply the current one. Every wave owns 64 columns of a slab (two K32 WMMA
steps); the v8 f32 accumulators keep ``gemv_gfx1250``'s lane mapping, so its
``store_accumulator``/``row_sum`` finish the cross-wave reduction.

Kernels using it launch in clusters of ``TDM_CLUSTER`` CTAs (``CLUSTER_LAUNCH``):
the token slab loads carry the cluster's multicast mask and a cluster barrier
opens every slab, so the CTAs' identical loads are in flight together and merge.

LDS rows are padded by 8 BF16 after each 512 columns: a TDM pad interval is at
most 256 dwords, and the 1040-byte row pitch spreads the 16 rows one operand
read touches over distinct banks.

``ATOM_KIMI_MONO_TDM_ROLLED=N`` (N >= 1, default 1) runs the slab loop as a
runtime loop with an LLVM unroll hint of N (1: none), so a launch fetches one
slab's code whatever K; 0 unrolls the loop at trace time.
"""

import math

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm, vector
from flydsl._mlir.dialects import rocdl as rocdl_dialect
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.rocdl import cluster
from flydsl.expr.typing import T, as_ir_value

from atom.models.kimi_k3.mono.kernels.stages import tdm_timeline
from atom.models.kimi_k3.mono.kernels.stages.gemv_gfx1250 import (
    ACCUMULATOR_ELEMENTS,
    fp8x8_to_bf16,
)
from atom.models.kimi_k3.mono.kernels.stages.gemv_layout import (
    GFX1250_GEMV,
    K_PER_MMA,
    LDS_PAD,
    PACK_ELEMENTS,
    ROWS,
    TDM_CLUSTER,
    TDM_DEPTH,
    TDM_FP8_DEPTH,
    TDM_FP8_SCALE_WORDS,
    TDM_ROW_STRIDE,
    TDM_SLAB,
    TDM_SPLIT_FP4_PITCH,
    TDM_SPLIT_FP4_ROW_WORDS,
    TDM_SPLIT_GROUP,
    TDM_SPLIT_MAX_TASKS,
    TDM_SPLIT_ROW_STRIDE,
    TDM_SPLIT_SLAB,
    TDM_SPLIT_TILE_WORDS,
    tdm_lds_words,
    tdm_split_slabs,
    tdm_split_task_words,
    tdm_task_words,
)
from atom.mono.device.ops import mma_bf16, rsrc, traced
from atom.mono.plan.architecture import GFX1250
from atom.utils import envs

SLAB = TDM_SLAB
ROW_STRIDE = TDM_ROW_STRIDE
DEPTH = TDM_DEPTH
FP8_DEPTH = TDM_FP8_DEPTH
WAVES = 8
WAVE_COLUMNS = SLAB // WAVES
WAVE_STEPS = WAVE_COLUMNS // K_PER_MMA
OPERAND_ELEMENTS = K_PER_MMA * ROWS // GFX1250.wave_size
assert WAVES * GFX1250.wave_size == TDM_FP8_SCALE_WORDS
_DESCRIPTOR_TYPE = 2 << 30
# gfx12 CPol TH_NT: each weight slab is read once per launch.
_WEIGHT_CACHE_POLICY = 1
_CLUSTER_MASK = (1 << TDM_CLUSTER) - 1 if TDM_CLUSTER > 1 else 0
_ROLLED_UNROLL = envs.ATOM_KIMI_MONO_TDM_ROLLED
_COLD = envs.ATOM_KIMI_MONO_TDM_COLD
CLUSTER_LAUNCH = {
    "value_attrs": {"rocdl.cluster_dims": f"{TDM_CLUSTER},1,1"},
    "cluster": (TDM_CLUSTER, 1, 1),
}


_COLD_DEFAULT = frozenset({"xpf"})


def cold_suffix() -> str:
    """Kernel-name suffix of the ``ATOM_KIMI_MONO_TDM_COLD`` changes built in,
    empty for the default set."""
    if _COLD == _COLD_DEFAULT:
        return ""
    return "".join(f"_c{site}" for site in sorted(_COLD)) or "_c0"


@traced
def _prefetch_on(role, asm: str):
    if role == 0:
        llvm.inline_asm(None, [], asm, "", has_side_effects=True)


def prefetch_code(wave, code_bytes: int) -> None:
    """``ATOM_KIMI_MONO_TDM_COLD=prefetch``: at the kernel's entry, wave 0
    prefetches the next ``code_bytes`` of instructions into the instruction
    cache (31 lines of 128 B per request), on top of the descriptor's
    INST_PREF_SIZE prefetch from the entry."""
    if const_expr("prefetch" not in _COLD):
        return
    step = 31 * 128
    asm = "\n".join(
        f"s_prefetch_inst_pc_rel {offset}, null, 31"
        for offset in range(0, code_bytes, step)
    )
    _prefetch_on(_uniform(wave), asm)


XCDS = 8


def prefetch_tokens(x, rows: int, row_bytes: int, bid, tid, ctas: int) -> None:
    """``ATOM_KIMI_MONO_TDM_COLD=xpf``: the grid's ``ctas`` CTAs prefetch the
    128 B lines of ``rows`` token rows of ``row_bytes`` (contiguous) into L2,
    one line a thread. Clusters go round-robin over the XCDs, so each XCD's
    CTAs split every line between them: its ``ctas // (XCDS * TDM_CLUSTER)``
    full rounds of clusters cover all shares, a partial round repeats some."""
    if const_expr("xpf" not in _COLD):
        return
    shares = ctas // (XCDS * TDM_CLUSTER) * TDM_CLUSTER
    lines = rows * row_bytes // 128
    share = (bid // (XCDS * TDM_CLUSTER) * TDM_CLUSTER + bid % TDM_CLUSTER) % shares
    per_cta = (lines + shares - 1) // shares
    for step in range_constexpr((per_cta + THREADS - 1) // THREADS):
        _prefetch_line(x, share + (tid + step * THREADS) * shares, lines)


@traced
def _prefetch_line(x, line, lines):
    if line < lines:
        rocdl_dialect.global_prefetch(
            llvm.IntToPtrOp(
                ir.Type.parse("!llvm.ptr<1>"),
                as_ir_value(x + fx.Int64(line) * 128),
            ).result
        )


def late_scratch() -> bool:
    return "preload" in _COLD


def opaque_pointer(value):
    """``value`` (an SGPR pair) through an asm copy the compiler cannot see
    through, so address arithmetic on it stays where it is written."""
    return fx.Int64(
        llvm.inline_asm(
            T.i64, [as_ir_value(fx.Int64(value))], "s_mov_b64 $0, $1", "=s,s"
        )
    )


def apply_preload(launch) -> None:
    """``ATOM_KIMI_MONO_TDM_COLD=preload``: ``launch``'s kernel arguments
    preloaded into SGPRs at dispatch (as many as fit), so the first copies
    need no kernarg load."""
    if "preload" in _COLD:
        options = {**launch.compile_hints.get("llvm_options", {})}
        options["amdgpu-kernarg-preload-count"] = 16
        launch.compile_hints = {**launch.compile_hints, "llvm_options": options}


def _i32(value: int):
    return fx.Int32(value - (1 << 32) if value >= 1 << 31 else value)


def _uniform(value):
    """A wave-uniform ``value`` in an SGPR."""
    return fx.Int32(rocdl.readfirstlane(T.i32, as_ir_value(fx.Int32(value))))


def load_rows(
    address,
    row_stride: int,
    lds,
    rows,
    rows_per_wave: int,
    mask: int = 0,
    padded: bool = True,
    cache_policy: int = 0,
    whole: bool = False,
    width: int = SLAB,
    pad_every: int = 0,
    uniform: bool = False,
    slab: int = SLAB,
):
    """Issue this wave's TDM copy of ``rows_per_wave`` of a [rows, width] BF16
    tile starting at ``address`` into ``lds``; rows past ``rows`` are skipped.
    ``mask`` is the cluster multicast mask of identical loads to merge.
    ``padded``: LDS rows ``slab + LDS_PAD`` apart, ``width`` being ``slab``,
    else packed ``width`` apart.
    ``pad_every``: with ``padded`` off, one pad dword after every ``pad_every``
    dwords the copy writes (a power of two up to 256), across rows alike.
    ``whole``: the first ``rows_per_wave`` rows in one copy, by the caller's
    one wave. Widths and strides are in BF16 units, whatever the bytes hold.
    ``uniform``: read the row count into an SGPR here, so a loop issuing the
    copy keeps its descriptor in SGPRs, not VGPRs read back every copy."""
    if padded and (width != slab or slab not in (SLAB, SLAB // 2)):
        raise ValueError("only a SLAB- or half-SLAB-wide copy pads its LDS rows")
    if pad_every and (
        padded or pad_every & (pad_every - 1) or not 2 <= pad_every <= 256
    ):
        raise ValueError(
            f"pad_every {pad_every}: a power of two in [2, 256], unpadded rows"
        )

    wave = fx.Int32(rocdl.wave_id())
    row0 = fx.Int32(0) if whole else wave * rows_per_wave
    source = address + fx.Int64(row0) * fx.Int64(row_stride * 2)
    pitch = slab + LDS_PAD if padded else width
    destination = fx.Int32(fx.ptrtoint(lds)) + row0 * (pitch * 2)
    live = fx.max(fx.min(fx.Int32(rows) - row0, rows_per_wave), 0)
    if uniform:
        live = _uniform(live)
    group0 = vector.from_elements(
        T.vec(4, T.i32),
        [
            as_ir_value(value)
            for value in (
                (live > 0).select(fx.Int32(1), fx.Int32(0)),
                destination,
                fx.Int32(source & fx.Int64(0xFFFFFFFF)),
                fx.Int32(source >> fx.Int64(32)) | _i32(_DESCRIPTOR_TYPE),
            )
        ],
    )
    pad_interval = int(math.log2(slab * 2 // 4)) - 1
    pad_amount = LDS_PAD * 2 // 4 - 1
    config = (1 << 16) | mask
    if padded:
        config |= (1 << 20) | (pad_interval << 22) | (pad_amount << 25)
    elif pad_every:
        config |= (1 << 20) | ((int(math.log2(pad_every)) - 1) << 22)
    group1 = vector.from_elements(
        T.vec(8, T.i32),
        [
            as_ir_value(value)
            for value in (
                _i32(config),
                _i32(width << 16),
                (live & 0xFFFF) << 16,
                (live >> 16) | _i32(width << 16),
                live,
                _i32(row_stride),
                _i32(0),
                _i32(0),
            )
        ],
    )
    zero4 = vector.from_elements(T.vec(4, T.i32), [as_ir_value(_i32(0))] * 4)
    zero8 = vector.from_elements(T.vec(8, T.i32), [as_ir_value(_i32(0))] * 8)
    rocdl_dialect.tensor_load_to_lds(
        group0,
        group1,
        zero4,
        zero4,
        zero8,
        cache_policy=ir.IntegerAttr.get(ir.IntegerType.get_signless(32), cache_policy),
    )


@traced
def _on_wave(wave, issuer, emit):
    """``emit()`` on wave ``issuer`` only: the others skip the instruction, so
    their tensor counters never see it."""
    if wave == issuer:
        emit()


def _operand(lds, row, wave, lane, row_stride=ROW_STRIDE, wave_columns=WAVE_COLUMNS):
    """One lane's v16 BF16 operands of row ``row`` for this wave's columns."""

    lane_group = lane // ROWS
    column = wave * wave_columns
    steps = []
    for step in range_constexpr(wave_columns // K_PER_MMA):
        words = []
        for pack in range_constexpr(GFX1250_GEMV.operand_packs_per_lane):
            element = (
                column
                + step * K_PER_MMA
                + lane_group * PACK_ELEMENTS
                + pack * GFX1250_GEMV.operand_pack_stride
            )
            raw = fx.Vector(
                fx.ptr_load(
                    lds + (row * row_stride + element) // 2,
                    result_type=fx.Vector.make_type(PACK_ELEMENTS // 2, fx.Int32),
                )
            )
            words.extend(raw[word] for word in range_constexpr(PACK_ELEMENTS // 2))
        steps.append(fx.Vector.from_elements(words, fx.Int32).bitcast(fx.BFloat16))
    return steps


def _fp8_operand(lds, wave, lane, exponent, wave_steps=WAVE_STEPS):
    """``_operand`` of a 16-row FP8 slab in AITER's gfx1250 preshuffle
    (``[K/32][2][16 rows][16 B]``), each value times the E8M0 ``exponent``'s
    power of two, so the BF16 operands are exact."""

    row = lane % ROWS
    lane_group = lane // ROWS
    steps = []
    for step in range_constexpr(wave_steps):
        chunk = wave * wave_steps + step
        elements = []
        for half in range_constexpr(2):
            byte = ((chunk * 2 + half) * ROWS + row) * 16 + lane_group * PACK_ELEMENTS
            raw = fx.Vector(
                fx.ptr_load(
                    lds + byte // 4,
                    result_type=fx.Vector.make_type(2, fx.Int32),
                )
            )
            converted = fp8x8_to_bf16(raw, exponent)
            elements.extend(converted[i] for i in range_constexpr(PACK_ELEMENTS))
        steps.append(fx.Vector.from_elements(elements, fx.BFloat16))
    return steps


def project(
    lds,
    x,
    x_stride: int,
    rows: int,
    w,
    w_rows: int,
    k: int,
    task_rows,
    lane,
    wave,
    w_scale=None,
    depth: int = DEPTH,
    prologue=None,
    x_cache_policy: int = 0,
    wide: bool = True,
    token_buffers: int = 2,
):
    """``x[rows, k] @ w[task_rows[i]:+16, k]^T`` for each task, as v8 f32 WMMA
    accumulators ``[task][16-token tile]``. ``w`` has ``w_rows`` rows of ``k``
    BF16; rows past it are not read and their outputs are meaningless.

    With ``w_scale``, ``w`` is FP8 in AITER's gfx1250 preshuffle with its rows
    padded to 128 and ``w_scale`` its ``[rows / 128, k / 128]`` E8M0 block
    scales: the per_1x128 layout ATOM's online quant leaves.

    ``prologue``: emitted once the first ``depth`` weight slabs are in flight
    and before ``x`` is first read, so ``x`` may be produced there by other
    CTAs; ``x_cache_policy`` is then the scope ``x`` is read at.

    ``wide``: each slab's tokens and each task's weight slab move in one TDM
    copy, by a wave of its own in turn, not as every wave's few rows: a wave
    keeps at most three tensor copies in flight, so small copies cap the bytes
    a CU has in flight.

    ``token_buffers``: token slabs resident in LDS (``x_words`` grows with
    it), so that many are in flight once ``x`` is first read."""

    if k % SLAB:
        raise ValueError(f"TDM GEMV needs K divisible by {SLAB}, got {k}")
    slabs = k // SLAB
    fp8 = w_scale is not None
    if fp8 and slabs > GFX1250.wave_size:
        raise ValueError(f"FP8 TDM GEMV holds one slab's scales a lane, K={k}")
    tasks = len(task_rows)
    if _ROLLED_UNROLL and wide and token_buffers == 2 and 1 + 2 * tasks <= WAVES:
        return _project_rolled(
            lds,
            x,
            x_stride,
            rows,
            w,
            w_rows,
            k,
            task_rows,
            lane,
            wave,
            w_scale,
            depth,
            prologue,
            x_cache_policy,
        )
    tiles = (rows + ROWS - 1) // ROWS
    token_words = rows * ROW_STRIDE // 2
    x_words = token_buffers * token_words
    task_words = tdm_task_words(fp8)
    stages = depth + 1
    rows_per_wave = (rows + WAVES - 1) // WAVES
    issued = []
    # FP8: wave t copies task t's scale words, one a slab (its four K blocks'
    # E8M0 bytes), to LDS past the weight slabs; the waves past the tasks and
    # the lanes past the slabs fill unread words. Read back from LDS a slab at
    # a time, the exponents stay out of SGPRs, which K1 has none to spare.
    scales = lds + x_words + stages * tasks * task_words
    if fp8:
        row = task_rows[0]
        for task in range(1, tasks):
            row = (wave == task).select(task_rows[task], row)
        word = bo.buffer_load(
            rsrc(w_scale),
            row // 128 * (k // 512) + fx.min(lane, slabs - 1),
            vec_width=1,
            dtype=T.i32,
        )
        fx.ptr_store(fx.Int32(word), scales + wave * GFX1250.wave_size + lane)

    def slab_exponent(task, slab):
        word = fx.ptr_load(scales + task * GFX1250.wave_size + slab)
        return (word >> (wave // (128 // WAVE_COLUMNS) * 8)) & 0xFF

    owned = {w: [] for w in range(WAVES)}
    turn = [0]

    def next_issuer(slab):
        issuer = turn[0] % WAVES
        turn[0] += 1
        owned[issuer].append(slab)
        return issuer

    def issue_tokens(slab):
        def emit(whole=False):
            load_rows(
                x + fx.Int64(slab * SLAB * 2),
                x_stride,
                lds + (slab % token_buffers) * token_words,
                rows,
                rows if whole else rows_per_wave,
                _CLUSTER_MASK,
                cache_policy=x_cache_policy,
                whole=whole,
            )

        if wide:
            _on_wave(wave, next_issuer(slab), lambda: emit(True))
        else:
            emit()
        issued.append(slab)

    def issue_weights(slab):
        for task in range(tasks):
            destination = lds + x_words + ((slab % stages) * tasks + task) * task_words

            def emit(task=task, destination=destination, whole=False):
                if fp8:
                    # The task's slab is one contiguous 16 x 512 B block:
                    # copied as WAVES packed 512-element rows.
                    load_rows(
                        w
                        + fx.Int64(task_rows[task]) * fx.Int64(k)
                        + fx.Int64(slab * SLAB * ROWS),
                        SLAB,
                        destination,
                        WAVES,
                        WAVES if whole else 1,
                        padded=False,
                        cache_policy=_WEIGHT_CACHE_POLICY,
                        whole=whole,
                    )
                else:
                    load_rows(
                        w
                        + fx.Int64(task_rows[task]) * fx.Int64(k * 2)
                        + fx.Int64(slab * SLAB * 2),
                        k,
                        destination,
                        fx.Int32(w_rows) - task_rows[task],
                        ROWS if whole else ROWS // WAVES,
                        cache_policy=_WEIGHT_CACHE_POLICY,
                        whole=whole,
                    )

            if wide:
                _on_wave(wave, next_issuer(slab), lambda emit=emit: emit(whole=True))
            else:
                emit()
            issued.append(slab)

    accumulators = [
        [fx.Vector.filled(ACCUMULATOR_ELEMENTS, 0.0, fx.Float32) for _ in range(tiles)]
        for _ in range(tasks)
    ]
    ahead = min(token_buffers, slabs)
    if prologue is None:
        for slab in range(max(ahead - 1, 1)):
            issue_tokens(slab)
    for slab in range(min(depth, slabs)):
        issue_weights(slab)
    if prologue is not None:
        prologue()
        for slab in range(max(ahead - 1, 1)):
            issue_tokens(slab)
    for slab in range(slabs):
        if slab:
            # The buffers slab - 1 read are the ones refilled below.
            if TDM_CLUSTER > 1:
                cluster.cluster_barrier()
            else:
                gpu.barrier()
            if slab + ahead - 1 < slabs:
                issue_tokens(slab + ahead - 1)
        elif ahead > 1:
            issue_tokens(ahead - 1)
        if slab + depth < slabs:
            issue_weights(slab + depth)
        if wide:
            for issuer, tags in owned.items():
                if slab in tags:
                    last = max(index for index, tag in enumerate(tags) if tag == slab)
                    later = len(tags) - 1 - last
                    _on_wave(wave, issuer, lambda n=later: rocdl.s_wait_tensorcnt(n))
        else:
            last = max(index for index, tag in enumerate(issued) if tag == slab)
            rocdl.s_wait_tensorcnt(len(issued) - 1 - last)
        gpu.barrier()
        weights = [
            (
                _fp8_operand(
                    lds + x_words + ((slab % stages) * tasks + task) * task_words,
                    wave,
                    lane,
                    slab_exponent(task, slab),
                )
                if fp8
                else _operand(
                    lds + x_words + ((slab % stages) * tasks + task) * task_words,
                    lane % ROWS,
                    wave,
                    lane,
                )
            )
            for task in range(tasks)
        ]
        tokens = lds + (slab % token_buffers) * token_words
        for tile in range(tiles):
            activations = _operand(
                tokens,
                tile * ROWS + fx.min(lane % ROWS, min(ROWS, rows - tile * ROWS) - 1),
                wave,
                lane,
            )
            for task in range(tasks):
                for step in range(WAVE_STEPS):
                    accumulators[task][tile] = mma_bf16(
                        weights[task][step],
                        activations[step],
                        accumulators[task][tile],
                        GFX1250,
                    )
    return accumulators


@traced
def _rolled_token_step(s, slabs, issue):
    """Wave 0's part of slab ``s``: copy the next token slab, then wait for
    this one."""
    if s + 1 < slabs:
        issue(s + 1)
    rocdl.s_wait_tensorcnt(1)
    if s + 1 >= slabs:
        rocdl.s_wait_tensorcnt(0)


@traced
def _rolled_weight_tail(s, slabs, ahead):
    """Past ``s_wait_tensorcnt(ahead)``, once fewer than ``ahead`` of this
    wave's slabs (``s + 2``, ``s + 4`` ..) follow ``s``: every count from
    ``ahead - 1`` down to that number."""
    if s + 2 * ahead >= slabs:
        for count in range_constexpr(ahead - 1, -1, -1):
            if s + 2 * (count + 1) >= slabs:
                rocdl.s_wait_tensorcnt(count)


@traced
def _rolled_weight_step(parity, s, slabs, depth, issue):
    """A weight wave's part of slab ``s``; ``parity`` is 1 on the wave whose
    slab ``s`` is, which waits with ``depth // 2`` of its later slabs in
    flight. Slab ``s + depth`` is copied by the wave of its parity."""
    if const_expr(depth % 2 == 0):
        if parity == 1:
            if s + depth < slabs:
                issue(s + depth)
            rocdl.s_wait_tensorcnt(depth // 2)
            _rolled_weight_tail(s, slabs, depth // 2)
    else:
        if parity == 1:
            rocdl.s_wait_tensorcnt(depth // 2)
            _rolled_weight_tail(s, slabs, depth // 2)
        if (parity == 0) & (s + depth < slabs):
            issue(s + depth)


@traced
def _rolled_roles(role, s, slabs, tasks, depth, issue_tokens, issue_weight):
    """Slab ``s``'s copies and waits. ``role`` is the wave in an SGPR, so every
    test is a scalar branch: under a lane-mask ``if``, ``s_wait_tensorcnt``
    loses its EXEC-empty skip and every wave runs it, draining its copies."""
    if role == 0:
        _rolled_token_step(s, slabs, issue_tokens)
    if (role > 0) & (role < 1 + 2 * tasks):
        _rolled_weight_step((role + s) & 1, s, slabs, depth, issue_weight)


def _rolled_last_tile(accs, pend, tasks, tiles):
    """The last token tile's WMMAs of the slab whose operands ``pend`` holds:
    each task's weights a step, then the tile's activations."""
    accs = list(accs)
    for tk in range(tasks):
        acc = accs[tk * tiles + tiles - 1]
        for st in range(WAVE_STEPS):
            acc = mma_bf16(
                pend[tk * WAVE_STEPS + st], pend[tasks * WAVE_STEPS + st], acc, GFX1250
            )
        accs[tk * tiles + tiles - 1] = acc
    return accs


@traced
def _project_rolled(
    lds,
    x,
    x_stride,
    rows,
    w,
    w_rows,
    k,
    task_rows,
    lane,
    wave,
    w_scale,
    depth,
    prologue,
    x_cache_policy,
):
    """``project`` (wide) with its slab loop a runtime loop. Wave 0 copies
    every token slab and waves ``1 + 2 t`` and ``2 + 2 t`` task ``t``'s
    weight slabs of even and odd slabs, so each wave's copies in flight, and
    its waits, repeat every two slabs. A slab's last token tile multiplies
    behind the next slab's operand loads, its operands held in registers, so
    the WMMAs still overlap a barrier as in the unrolled schedule."""
    slabs = k // SLAB
    fp8 = w_scale is not None
    tasks = len(task_rows)
    tiles = (rows + ROWS - 1) // ROWS
    x_words, _ = tdm_lds_words(rows, tasks, fp8)
    token_words = rows * ROW_STRIDE // 2
    task_words = tdm_task_words(fp8)
    stages = depth + 1
    scales = lds + x_words + stages * tasks * task_words
    tdm_timeline.stamp(tdm_timeline.GEMV_START)

    def stage_scales():
        row = task_rows[0]
        for t in range_constexpr(1, tasks):
            row = (wave == t).select(task_rows[t], row)
        word = bo.buffer_load(
            rsrc(w_scale),
            row // 128 * (k // 512) + fx.min(lane, slabs - 1),
            vec_width=1,
            dtype=T.i32,
        )
        fx.ptr_store(fx.Int32(word), scales + wave * GFX1250.wave_size + lane)

    # "early": the scales' load (and its wait, for the LDS store) behind the
    # first copies; they are first read past the slab-0 barrier either way.
    early = "early" in _COLD
    if const_expr(fp8 and not early):
        stage_scales()

    role = _uniform(wave)
    task = fx.min(fx.max(role - 1, 0) // 2, tasks - 1)
    task_row = task_rows[0]
    for t in range_constexpr(1, tasks):
        task_row = (task == t).select(task_rows[t], task_row)
    task_row = _uniform(task_row)

    def issue_tokens(slab):
        load_rows(
            x + fx.Int64(slab) * fx.Int64(SLAB * 2),
            x_stride,
            lds + (slab % 2) * token_words,
            rows,
            rows,
            _CLUSTER_MASK,
            cache_policy=x_cache_policy,
            whole=True,
        )

    def issue_weight(slab):
        destination = lds + x_words + ((slab % stages) * tasks + task) * task_words
        if const_expr(fp8):
            # The task's slab is one contiguous 16 x 512 B block: copied as
            # WAVES packed 512-element rows.
            load_rows(
                w
                + fx.Int64(task_row) * fx.Int64(k)
                + fx.Int64(slab) * fx.Int64(SLAB * ROWS),
                SLAB,
                destination,
                WAVES,
                WAVES,
                padded=False,
                cache_policy=_WEIGHT_CACHE_POLICY,
                whole=True,
                uniform=True,
            )
        else:
            load_rows(
                w
                + fx.Int64(task_row) * fx.Int64(k * 2)
                + fx.Int64(slab) * fx.Int64(SLAB * 2),
                k,
                destination,
                fx.Int32(w_rows) - task_row,
                ROWS,
                cache_policy=_WEIGHT_CACHE_POLICY,
                whole=True,
                uniform=True,
            )

    if const_expr(prologue is None):
        _on_wave(role, 0, lambda: issue_tokens(fx.Int32(0)))
    for first in range_constexpr(min(depth, slabs)):
        _on_wave(
            role,
            fx.Int32(1) + 2 * task + first % 2,
            lambda first=first: issue_weight(fx.Int32(first)),
        )
    if const_expr(prologue is not None):
        prologue()
        _on_wave(role, 0, lambda: issue_tokens(fx.Int32(0)))
    if const_expr(fp8 and early):
        stage_scales()
    tdm_timeline.stamp(tdm_timeline.ISSUED)
    accs = [
        fx.Vector.filled(ACCUMULATOR_ELEMENTS, 0.0, fx.Float32)
        for _ in range(tasks * tiles)
    ]
    # Zeros before slab 0: their WMMAs leave the accumulators +0.
    pend = [
        fx.Vector.filled(OPERAND_ELEMENTS, 0.0, fx.BFloat16)
        for _ in range((tasks + 1) * WAVE_STEPS)
    ]
    for s in range(fx.Int32(0), fx.Int32(slabs), unroll=_ROLLED_UNROLL):
        if s > 0:
            tdm_timeline.slab(s - 1, tdm_timeline.C)
            # The buffers slab s - 1 read are the ones refilled below.
            if const_expr(TDM_CLUSTER > 1):
                cluster.cluster_barrier()
            else:
                gpu.barrier()
        tdm_timeline.slab(s, tdm_timeline.A)
        _rolled_roles(role, s, slabs, tasks, depth, issue_tokens, issue_weight)
        tdm_timeline.slab(s, tdm_timeline.TOKEN)
        tdm_timeline.slab(s, tdm_timeline.WEIGHT, 1 + (s & 1))
        gpu.barrier()
        tdm_timeline.slab(s, tdm_timeline.B)
        slab_weights = [
            (
                _fp8_operand(
                    lds + x_words + ((s % stages) * tasks + tk) * task_words,
                    wave,
                    lane,
                    (
                        fx.ptr_load(scales + tk * GFX1250.wave_size + s)
                        >> (wave // (128 // WAVE_COLUMNS) * 8)
                    )
                    & 0xFF,
                )
                if fp8
                else _operand(
                    lds + x_words + ((s % stages) * tasks + tk) * task_words,
                    lane % ROWS,
                    wave,
                    lane,
                )
            )
            for tk in range(tasks)
        ]
        slab_tokens = lds + (s % 2) * token_words
        activations = [
            _operand(
                slab_tokens,
                tl * ROWS + fx.min(lane % ROWS, min(ROWS, rows - tl * ROWS) - 1),
                wave,
                lane,
            )
            for tl in range(tiles)
        ]
        updated = _rolled_last_tile(accs, pend, tasks, tiles)
        for tl in range_constexpr(tiles - 1):
            for tk in range_constexpr(tasks):
                acc = updated[tk * tiles + tl]
                for st in range_constexpr(WAVE_STEPS):
                    acc = mma_bf16(
                        slab_weights[tk][st], activations[tl][st], acc, GFX1250
                    )
                updated[tk * tiles + tl] = acc
        pend = [
            slab_weights[tk][st] for tk in range(tasks) for st in range(WAVE_STEPS)
        ] + list(activations[tiles - 1])
        accs = updated
    accs = _rolled_last_tile(accs, pend, tasks, tiles)
    tdm_timeline.slab(fx.Int32(slabs - 1), tdm_timeline.C)
    tdm_timeline.stamp(tdm_timeline.PROJECT_DONE)
    return [[accs[tk * tiles + tl] for tl in range(tiles)] for tk in range(tasks)]


SPLIT_SLAB = TDM_SPLIT_SLAB
SPLIT_WAVE_COLUMNS = SPLIT_SLAB // WAVES
# Waves 0-5 copy the token slabs (wave w slabs w, w + 6 ..), waves 6 and 7 the
# two task groups' weight slabs.
SPLIT_TOKEN_WAVES = WAVES - 2
# gfx12 CPol SCOPE (bits [4:3]) SE: a cluster's CTAs share their SE's caches.
SCOPE_SE = 8
THREADS = WAVES * GFX1250.wave_size
_STAGE_WORDS = WAVES * TDM_SPLIT_TILE_WORDS
_FP4_PAD_EVERY = (
    TDM_SPLIT_FP4_ROW_WORDS if TDM_SPLIT_FP4_PITCH > TDM_SPLIT_FP4_ROW_WORDS else 0
)
assert TDM_CLUSTER == 4
assert TDM_SPLIT_FP4_PITCH - TDM_SPLIT_FP4_ROW_WORDS in (0, 1)


def fp4x8_to_bf16(word, exponent, scale_sel=0):
    """Eight E2M1 values (one dword) times the E8M0 ``exponent``'s power of
    two, as BF16."""

    # gfx1250 reads the scale byte ``scale_sel`` in lanes 0-15 but byte
    # ``scale_sel + 1`` in lanes 16-31, so every byte carries the exponent.
    return fx.Vector(
        rocdl.cvt_scale_pk8_bf16_fp4(
            T.vec(PACK_ELEMENTS, T.bf16),
            as_ir_value(fx.Int32(word)),
            as_ir_value(fx.Int32(exponent) * 0x01010101),
            scale_sel,
        )
    )


def _fp4_operand(lds, wave, lane, exponent, pitch=TDM_SPLIT_FP4_PITCH):
    """``_operand`` of a 16-row MXFP4 slab (E2M1 pairs, low nibble first, rows
    ``pitch`` dwords apart) times its E8M0 ``exponent``: exact."""

    base = lds + (lane % ROWS) * pitch + wave * 4 + lane // ROWS
    elements = []
    for pack in range_constexpr(2):
        converted = fp4x8_to_bf16(fx.ptr_load(base + 2 * pack), exponent)
        elements.extend(converted[i] for i in range_constexpr(PACK_ELEMENTS))
    return fx.Vector.from_elements(elements, fx.BFloat16)


def _load8(ptr):
    return fx.Vector(
        fx.ptr_load(ptr, result_type=fx.Vector.make_type(8, fx.Int32))
    ).bitcast(fx.Float32)


def _add8(a, b):
    return fx.Vector.from_elements(
        [a[i] + b[i] for i in range_constexpr(8)], fx.Float32
    )


@traced
def _split_token_wait(role, s, slabs):
    """A token wave's wait for slab ``s``, its copies slabs ``role``,
    ``role + 6`` ..: every later one of them may stay in flight."""
    later = (slabs - 1) // SPLIT_TOKEN_WAVES
    if role == s % SPLIT_TOKEN_WAVES:
        rocdl.s_wait_tensorcnt(later)
        for count in range_constexpr(later - 1, -1, -1):
            if s + SPLIT_TOKEN_WAVES * (count + 1) >= slabs:
                rocdl.s_wait_tensorcnt(count)


@traced
def _split_weight_wait(role, s, slabs, stages):
    """A weight wave's wait for slab ``s``, its later slabs up to
    ``s + stages - 1`` in flight."""
    if role >= SPLIT_TOKEN_WAVES:
        rocdl.s_wait_tensorcnt(stages - 1)
        for count in range_constexpr(stages - 2, -1, -1):
            if s + count + 1 >= slabs:
                rocdl.s_wait_tensorcnt(count)


@traced
def _split_weight_issue(role, s, slabs, stages, groups, issue):
    if (
        (role >= SPLIT_TOKEN_WAVES)
        & (role < SPLIT_TOKEN_WAVES + groups)
        & (s + stages < slabs)
    ):
        issue(s + stages)


@traced
def _split_resident_wait(role, slabs):
    """Every wave's resident weight copies, issued before its token slabs."""
    for issuer in range_constexpr(WAVES):
        if role == issuer:
            rocdl.s_wait_tensorcnt(
                len(range(issuer, slabs, SPLIT_TOKEN_WAVES))
                if issuer < SPLIT_TOKEN_WAVES
                else 0
            )


@traced
def project_split(
    lds,
    x,
    x_stride,
    rows,
    w,
    w_rows,
    k,
    groups,
    owners,
    lane,
    wave,
    scratch,
    out,
    fmt="bf16",
    w_scale=None,
    stages=2,
    resident=False,
    prologue=None,
    x_cache_policy=0,
    cta0=0,
):
    """``project`` with K split over the TDM cluster. The cluster's tasks are
    ``len(groups)`` groups of four consecutive 16-row tasks, task ``4 g + i``
    rows ``groups[g] + 16 i`` of ``w``. CTA r (``cluster_workgroup_id_x``)
    multiplies every task over its quarter of K, its tokens copied at once
    and kept, its weights streamed through ``stages`` slab buffers by waves 6
    and 7. The f32 partials of the tasks other CTAs own go to ``scratch``
    (``tdm_split_scratch_bytes``) at SE scope and, past one cluster barrier,
    CTA r sums the tasks ``owners[j](r)`` over the cluster, in CTA order, into
    LDS at ``out + (j * tiles + tile) * TDM_SPLIT_TILE_WORDS`` as a v8 a lane
    in wave 0's accumulator layout (``gemv_gfx1250.accumulator_offset``).

    ``fmt``: ``SPLIT_FORMATS``; FP8 groups each lie in one 128-row scale block.
    ``resident``: the CTA's whole weight share is copied up front, before
    ``prologue`` (which may wait for ``x``), so only the tokens follow it.
    ``cta0``: the launch's first CTA that calls this (a multiple of the
    cluster), so ``scratch`` holds the CTAs from it on.
    Every CTA of a cluster calls this alike: it runs one cluster barrier."""

    if cta0 % TDM_CLUSTER:
        raise ValueError(f"split-K CTAs start a cluster, not at {cta0}")
    slabs = tdm_split_slabs(k)
    group_count = len(groups)
    if len(owners) != group_count or not 1 <= group_count <= 2:
        raise ValueError("split-K: one or two task groups, one owned task each")
    if fmt in ("fp8", "fp4") and w_scale is None:
        raise ValueError(f"split-K {fmt} weights need their scales")
    if fmt == "fp8" and k // 512 > GFX1250.wave_size:
        raise ValueError(f"split-K FP8 holds a group's scales a lane, K={k}")
    tasks = group_count * TDM_SPLIT_GROUP
    tiles = (rows + ROWS - 1) // ROWS
    if resident:
        stages = slabs
    token_words = rows * TDM_SPLIT_ROW_STRIDE // 2
    task_words = tdm_split_task_words(fmt)
    weights = lds + slabs * token_words
    scales = weights + stages * tasks * task_words
    rank = _uniform(rocdl.cluster_workgroup_id_x())
    column0 = rank * (slabs * SPLIT_SLAB)
    role = _uniform(wave)
    tid = wave * GFX1250.wave_size + lane
    group_rows = [_uniform(row) for row in groups]
    group = fx.min(fx.max(role - SPLIT_TOKEN_WAVES, 0), group_count - 1)
    group_row = group_rows[0]
    if const_expr(group_count == 2):
        group_row = _uniform((group == 1).select(group_rows[1], group_rows[0]))

    if const_expr(fmt == "fp8"):  # noqa: SIM102
        if wave < group_count:
            word = bo.buffer_load(
                rsrc(w_scale),
                (wave == 1).select(group_rows[-1], group_rows[0]) // 128 * (k // 512)
                + fx.min(lane, k // 512 - 1),
                vec_width=1,
                dtype=T.i32,
            )
            fx.ptr_store(fx.Int32(word), scales + wave * GFX1250.wave_size + lane)
    if const_expr(fmt == "fp4"):
        row_words = 2 * slabs
        share = tasks * ROWS * row_words
        for round_index in range_constexpr((share + THREADS - 1) // THREADS):
            item = tid + THREADS * round_index
            if item < share:
                row = item // row_words
                row_group = row // (TDM_SPLIT_GROUP * ROWS)
                source = (row_group == 1).select(
                    group_rows[-1], group_rows[0]
                ) + row % (TDM_SPLIT_GROUP * ROWS)
                word = bo.buffer_load(
                    rsrc(w_scale),
                    (source * (k // 32) + rank * (8 * slabs)) // 4 + item % row_words,
                    vec_width=1,
                    dtype=T.i32,
                )
                fx.ptr_store(fx.Int32(word), scales + item)

    def issue_tokens(slab):
        load_rows(
            x + fx.Int64(column0 + slab * SPLIT_SLAB) * fx.Int64(2),
            x_stride,
            lds + slab * token_words,
            rows,
            rows,
            cache_policy=x_cache_policy,
            whole=True,
            width=SPLIT_SLAB,
            slab=SPLIT_SLAB,
        )

    def issue_weight(slab, row, index):
        destination = (
            weights + ((slab % stages) * tasks + index * TDM_SPLIT_GROUP) * task_words
        )
        global_slab = fx.Int64(rank * slabs + slab)
        if const_expr(fmt == "bf16"):
            load_rows(
                w
                + fx.Int64(row) * fx.Int64(k * 2)
                + global_slab * fx.Int64(SPLIT_SLAB * 2),
                k,
                destination,
                fx.Int32(w_rows) - row,
                TDM_SPLIT_GROUP * ROWS,
                cache_policy=_WEIGHT_CACHE_POLICY,
                whole=True,
                uniform=True,
                width=SPLIT_SLAB,
                slab=SPLIT_SLAB,
            )
        elif const_expr(fmt == "fp8"):
            # A task's slab is one contiguous 16 x 256 B block, the group's
            # four of them 16 rows of K apart.
            load_rows(
                w
                + fx.Int64(row) * fx.Int64(k)
                + global_slab * fx.Int64(SPLIT_SLAB * ROWS),
                ROWS * k // 2,
                destination,
                TDM_SPLIT_GROUP,
                TDM_SPLIT_GROUP,
                padded=False,
                cache_policy=_WEIGHT_CACHE_POLICY,
                whole=True,
                uniform=True,
                width=SPLIT_SLAB * ROWS // 2,
            )
        else:
            load_rows(
                w
                + fx.Int64(row) * fx.Int64(k // 2)
                + global_slab * fx.Int64(SPLIT_SLAB // 2),
                k // 4,
                destination,
                fx.Int32(w_rows) - row,
                TDM_SPLIT_GROUP * ROWS,
                padded=False,
                pad_every=_FP4_PAD_EVERY,
                cache_policy=_WEIGHT_CACHE_POLICY,
                whole=True,
                uniform=True,
                width=SPLIT_SLAB // 4,
            )

    if const_expr(resident):
        for slab in range_constexpr(slabs):
            for index in range_constexpr(group_count):
                _on_wave(
                    role,
                    (slab * group_count + index) % WAVES,
                    lambda slab=slab, index=index: issue_weight(
                        fx.Int32(slab), group_rows[index], index
                    ),
                )
    else:
        for slab in range_constexpr(min(stages, slabs)):
            for index in range_constexpr(group_count):
                _on_wave(
                    role,
                    SPLIT_TOKEN_WAVES + index,
                    lambda slab=slab, index=index: issue_weight(
                        fx.Int32(slab), group_rows[index], index
                    ),
                )
    if const_expr(prologue is not None):
        prologue()
    for slab in range_constexpr(slabs):
        _on_wave(
            role,
            slab % SPLIT_TOKEN_WAVES,
            lambda slab=slab: issue_tokens(fx.Int32(slab)),
        )
    if const_expr(resident):
        _split_resident_wait(role, slabs)

    accs = [
        fx.Vector.filled(ACCUMULATOR_ELEMENTS, 0.0, fx.Float32)
        for _ in range(tasks * tiles)
    ]
    for s in range(fx.Int32(0), fx.Int32(slabs)):
        _split_token_wait(role, s, slabs)
        if const_expr(not resident):
            _split_weight_wait(role, s, slabs, stages)
        gpu.barrier()
        stage = weights + (s % stages) * (tasks * task_words)
        block = rank * (2 * slabs) + 2 * s + wave // 4
        if const_expr(fmt == "fp8"):
            exponents = [
                (
                    fx.ptr_load(scales + g * GFX1250.wave_size + block // 4)
                    >> (block % 4 * 8)
                )
                & 0xFF
                for g in range(group_count)
            ]
        slab_weights = []
        for t in range_constexpr(tasks):
            task_lds = stage + t * task_words
            if const_expr(fmt == "bf16"):
                operand = _operand(
                    task_lds,
                    lane % ROWS,
                    wave,
                    lane,
                    TDM_SPLIT_ROW_STRIDE,
                    SPLIT_WAVE_COLUMNS,
                )[0]
            elif const_expr(fmt == "fp8"):
                operand = _fp8_operand(
                    task_lds, wave, lane, exponents[t // TDM_SPLIT_GROUP], 1
                )[0]
            else:
                scale_word = fx.ptr_load(
                    scales + (t * ROWS + lane % ROWS) * (2 * slabs) + 2 * s + wave // 4
                )
                operand = _fp4_operand(
                    task_lds, wave, lane, (scale_word >> (wave % 4 * 8)) & 0xFF
                )
            slab_weights.append(operand)
        slab_tokens = lds + s * token_words
        activations = [
            _operand(
                slab_tokens,
                tl * ROWS + fx.min(lane % ROWS, min(ROWS, rows - tl * ROWS) - 1),
                wave,
                lane,
                TDM_SPLIT_ROW_STRIDE,
                SPLIT_WAVE_COLUMNS,
            )[0]
            for tl in range(tiles)
        ]
        updated = [
            mma_bf16(slab_weights[t], activations[tl], accs[t * tiles + tl], GFX1250)
            for t in range(tasks)
            for tl in range(tiles)
        ]
        accs = updated
        if const_expr(not resident):
            gpu.barrier()
            _split_weight_issue(
                role,
                s,
                slabs,
                stages,
                group_count,
                lambda slab: issue_weight(slab, group_row, group),
            )

    # Every wave's accumulators over the slab buffers, then a lane of a wave
    # sums one task's tile over the waves.
    gpu.barrier()
    for t in range_constexpr(tasks):
        for tl in range_constexpr(tiles):
            fx.ptr_store(
                accs[t * tiles + tl].bitcast(fx.Int32),
                lds
                + (t * tiles + tl) * _STAGE_WORDS
                + (wave * GFX1250.wave_size + lane) * ACCUMULATOR_ELEMENTS,
            )
    gpu.barrier()
    own = lds + tasks * tiles * _STAGE_WORDS
    owned = [owner(rank) for owner in owners]
    cta = fx.Int32(fx.block_idx.x) - cta0
    partials = rsrc(scratch)
    items = tasks * tiles * GFX1250.wave_size
    for round_index in range_constexpr((items + THREADS - 1) // THREADS):
        item = tid + THREADS * round_index
        t = item // (tiles * GFX1250.wave_size)
        tl = item // GFX1250.wave_size % tiles
        valid = (item < items) & (tl * ROWS + lane % ROWS < rows)
        mine = t == owned[0]
        slot = fx.Int32(0)
        if const_expr(group_count == 2):
            slot = (t == owned[1]).select(fx.Int32(1), fx.Int32(0))
            mine = mine | (t == owned[1])
        if valid:
            slab_sums = (
                lds + (t * tiles + tl) * _STAGE_WORDS + lane * ACCUMULATOR_ELEMENTS
            )
            total = _load8(slab_sums)
            for source_wave in range_constexpr(1, WAVES):
                total = _add8(
                    total,
                    _load8(slab_sums + source_wave * TDM_SPLIT_TILE_WORDS),
                )
            if mine:
                fx.ptr_store(
                    total.bitcast(fx.Int32),
                    own
                    + (slot * tiles + tl) * TDM_SPLIT_TILE_WORDS
                    + lane * ACCUMULATOR_ELEMENTS,
                )
            else:
                at = (
                    (cta * TDM_SPLIT_MAX_TASKS + t) * tiles + tl
                ) * TDM_SPLIT_TILE_WORDS
                for half in range_constexpr(2):
                    bo.buffer_store(
                        fx.Vector.from_elements(
                            [total[half * 4 + i] for i in range_constexpr(4)],
                            fx.Float32,
                        ),
                        partials,
                        at + lane * ACCUMULATOR_ELEMENTS + half * 4,
                        cache_modifier=SCOPE_SE,
                    )
    rocdl.s_wait_storecnt(0)
    cluster.cluster_barrier()

    items = group_count * tiles * GFX1250.wave_size
    for round_index in range_constexpr((items + THREADS - 1) // THREADS):
        item = tid + THREADS * round_index
        j = item // (tiles * GFX1250.wave_size)
        tl = item // GFX1250.wave_size % tiles
        if (item < items) & (tl * ROWS + lane % ROWS < rows):
            task_id = owned[0]
            if const_expr(group_count == 2):
                task_id = (j == 1).select(owned[1], owned[0])
            own_sum = _load8(
                own
                + (j * tiles + tl) * TDM_SPLIT_TILE_WORDS
                + lane * ACCUMULATOR_ELEMENTS
            )
            summed = None
            for source in range_constexpr(TDM_CLUSTER):
                peer_at = (
                    ((cta - rank + source) * TDM_SPLIT_MAX_TASKS + task_id) * tiles + tl
                ) * TDM_SPLIT_TILE_WORDS + lane * ACCUMULATOR_ELEMENTS
                halves = [
                    fx.Vector(
                        bo.buffer_load(
                            partials,
                            peer_at + half * 4,
                            vec_width=4,
                            dtype=T.f32,
                            cache_modifier=SCOPE_SE,
                        )
                    )
                    for half in range(2)
                ]
                term = fx.Vector.from_elements(
                    [
                        (rank == source).select(own_sum[i], halves[i // 4][i % 4])
                        for i in range(8)
                    ],
                    fx.Float32,
                )
                summed = term if summed is None else _add8(summed, term)
            fx.ptr_store(
                summed.bitcast(fx.Int32),
                out
                + (j * tiles + tl) * TDM_SPLIT_TILE_WORDS
                + lane * ACCUMULATOR_ELEMENTS,
            )
    gpu.barrier()
