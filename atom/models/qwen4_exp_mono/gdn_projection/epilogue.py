"""Channel-owned q4 convolution after PTPC projection's BF16 endpoint."""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import gpu, range_constexpr
from flydsl.expr import math as fm
from flydsl.expr.typing import T

from atom.models.qwen4_exp_mono.stages.hc import _ld_bf, _sigmoid, _st_bf
from atom.mono.device.ops import bf16_round, rsrc, traced


@traced
def complete(
    sm,
    conv_w,
    history,
    slots,
    accepted,
    cs0,
    cs1,
    cs2,
    alpha,
    beta_input,
    abs0,
    bbs0,
    a_log,
    dt_bias,
    q,
    k,
    v,
    g,
    beta,
    z,
):
    tid, task = fx.thread_idx.x, fx.block_idx.x
    # All four projected rows have passed the BF16 boundary before this call.
    gpu.barrier()
    if tid < 64:
        col = task * 64 + tid
        values = [fx.ptr_load(sm.projected.ptr + row * 64 + tid) for row in range(4)]
        if col < 10240:
            slot = fx.Int64(bo.buffer_load(rsrc(slots), 0, vec_width=1, dtype=T.i32))
            keep = fx.Int64(bo.buffer_load(rsrc(accepted), 0, vec_width=1, dtype=T.i32))
            if slot >= 0:
                old = [
                    _ld_bf(
                        history, slot * cs0 + fx.Int64(col) * cs1 + (keep - 1 + i) * cs2
                    )
                    for i in range(3)
                ]
                weights = [_ld_bf(conv_w, col * 4 + i) for i in range(4)]
                samples = old + values
                for row in range_constexpr(4):
                    # Native convolution multiplies BF16 operands, rounds each
                    # product, then sums in FP32 before SiLU and BF16 output.
                    total = bf16_round(samples[row] * weights[0])
                    for tap in range_constexpr(1, 4):
                        total = total + bf16_round(samples[row + tap] * weights[tap])
                    result = bf16_round(total * _sigmoid(total))
                    if col < 2048:
                        _st_bf(q, row * 2048 + col, result)
                    elif col < 4096:
                        _st_bf(k, row * 2048 + col - 2048, result)
                    else:
                        _st_bf(v, row * 6144 + col - 4096, result)
                # This thread owns every read and write for its channel. The
                # old values are all in registers; no cross-CTA history reader.
                _st_bf(history, slot * cs0 + fx.Int64(col) * cs1, old[1])
                _st_bf(history, slot * cs0 + fx.Int64(col) * cs1 + cs2, old[2])
                for row in range_constexpr(4):
                    _st_bf(
                        history,
                        slot * cs0 + fx.Int64(col) * cs1 + fx.Int64(row + 2) * cs2,
                        values[row],
                    )
        else:
            for row in range_constexpr(4):
                _st_bf(z, row * 6144 + col - 10240, values[row])
    if task == 0:  # noqa: SIM102 - preserve separate device predicates
        if tid < 192:
            row, head = tid // 48, tid % 48
            a = _ld_bf(alpha, fx.Int64(row) * abs0 + head)
            b = _ld_bf(beta_input, fx.Int64(row) * bbs0 + head)
            bias = _ld_bf(dt_bias, head)
            log_a = _ld_bf(a_log, head)
            x = a + bias
            softplus = (x <= fx.Float32(20)).select(
                fx.Float32(fm.log(fx.Float32(1) + fx.Float32(fm.exp(x)))), x
            )
            gate = -fx.Float32(fm.exp(log_a)) * softplus
            bo.buffer_store(gate, rsrc(g), tid)
            _st_bf(beta, tid, _sigmoid(b))
