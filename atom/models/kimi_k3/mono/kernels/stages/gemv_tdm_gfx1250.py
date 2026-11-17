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
from flydsl._mlir.dialects import rocdl as rocdl_dialect
from flydsl._mlir.dialects import vector
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.rocdl import cluster
from flydsl.expr.typing import T, as_ir_value

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
    tdm_lds_words,
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
CLUSTER_LAUNCH = {
    "value_attrs": {"rocdl.cluster_dims": f"{TDM_CLUSTER},1,1"},
    "cluster": (TDM_CLUSTER, 1, 1),
}


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
):
    """Issue this wave's TDM copy of ``rows_per_wave`` of a [rows, width] BF16
    tile starting at ``address`` into ``lds``; rows past ``rows`` are skipped.
    ``mask`` is the cluster multicast mask of identical loads to merge.
    ``padded``: LDS rows ``ROW_STRIDE`` apart, else packed ``width`` apart.
    ``pad_every``: with ``padded`` off, one pad dword after every ``pad_every``
    dwords the copy writes (a power of two up to 256), across rows alike.
    ``whole``: the first ``rows_per_wave`` rows in one copy, by the caller's
    one wave. Widths and strides are in BF16 units, whatever the bytes hold.
    ``uniform``: read the row count into an SGPR here, so a loop issuing the
    copy keeps its descriptor in SGPRs, not VGPRs read back every copy."""
    if padded and width != SLAB:
        raise ValueError("only a SLAB-wide copy pads its LDS rows")
    if pad_every and (
        padded or pad_every & (pad_every - 1) or not 2 <= pad_every <= 256
    ):
        raise ValueError(
            f"pad_every {pad_every}: a power of two in [2, 256], unpadded rows"
        )

    wave = fx.Int32(rocdl.wave_id())
    row0 = fx.Int32(0) if whole else wave * rows_per_wave
    source = address + fx.Int64(row0) * fx.Int64(row_stride * 2)
    pitch = ROW_STRIDE if padded else width
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
    pad_interval = int(math.log2(SLAB * 2 // 4)) - 1
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


def _operand(lds, row, wave, lane):
    """One lane's v16 BF16 operands of row ``row`` for this wave's columns."""

    lane_group = lane // ROWS
    column = wave * WAVE_COLUMNS
    steps = []
    for step in range_constexpr(WAVE_STEPS):
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
                    lds + (row * ROW_STRIDE + element) // 2,
                    result_type=fx.Vector.make_type(PACK_ELEMENTS // 2, fx.Int32),
                )
            )
            words.extend(raw[word] for word in range_constexpr(PACK_ELEMENTS // 2))
        steps.append(fx.Vector.from_elements(words, fx.Int32).bitcast(fx.BFloat16))
    return steps


def _fp8_operand(lds, wave, lane, exponent):
    """``_operand`` of a 16-row FP8 slab in AITER's gfx1250 preshuffle
    (``[K/32][2][16 rows][16 B]``), each value times the E8M0 ``exponent``'s
    power of two, so the BF16 operands are exact."""

    row = lane % ROWS
    lane_group = lane // ROWS
    steps = []
    for step in range_constexpr(WAVE_STEPS):
        chunk = wave * WAVE_STEPS + step
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
    if const_expr(fp8):
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
            # The buffers slab s - 1 read are the ones refilled below.
            if const_expr(TDM_CLUSTER > 1):
                cluster.cluster_barrier()
            else:
                gpu.barrier()
        _rolled_roles(role, s, slabs, tasks, depth, issue_tokens, issue_weight)
        gpu.barrier()
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
    return [[accs[tk * tiles + tl] for tl in range(tiles)] for tk in range(tasks)]
