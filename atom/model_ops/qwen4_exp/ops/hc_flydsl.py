# SPDX-License-Identifier: MIT

"""FlyDSL Qwen3.8 Flash-Next low-batch hyper-connection kernels.

The fixed checkpoint geometry is HC=4, H=2560 and lowrank=320. One CTA
computes each norm stream, down row, or injection stream. The up projection
assigns one output channel to each wave and consumes its four gates locally.
No cross-CTA barrier, persistent scratch, or epoch is needed for graph replay.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from aiter.ops.flydsl.kernels import buffer_ops as bo
from aiter.ops.flydsl.kernels.kernels_common import LOG2E
from aiter.tuned_gemm import tgemm
from flydsl.expr import gpu, range_constexpr
from flydsl.expr.typing import Int32, Int64, Stream, T

from atom.mono.device.ops import (
    dot2_bf16,
    hw_exp2,
    rsrc,
    xred,
)
from atom.mono.device.ops import (
    hw_rcp as rcp,
)
from atom.mono.device.ops import (
    hw_rsq as rsq,
)

HIDDEN = 2560
HC = 4
WIDTH = HIDDEN * HC
RANK = 320
THREADS = 256
WAVES = THREADS // 64
UP_WAVES = 4
UP_THREADS = UP_WAVES * 64
_CURRENT_STREAM = Stream(None)


def exp(value):
    return hw_exp2(fx.Float32(value) * LOG2E)


def _unpack(word):
    word = fx.Int32(word)
    return (word << 16).bitcast(fx.Float32), (word & fx.Int32(-65536)).bitcast(
        fx.Float32
    )


def _pack(lo, hi):
    return (
        fx.Vector.from_elements([fx.Float32(lo), fx.Float32(hi)], fx.Float32)
        .to(fx.BFloat16)
        .bitcast(fx.Int32)[0]
    )


def _wave_sum(value):
    for offset in (32, 16, 8, 4, 2, 1):
        value = xred(value, offset, lambda a, b: a + b)
    return value


@flyc.kernel(known_block_size=[THREADS, 1, 1])
def _gated_mean_kernel(normed: Int64, gate: Int64, mixed: Int64):
    row = gpu.block_idx.x
    hidden = gpu.block_idx.y * THREADS + gpu.thread_idx.x
    if hidden < HIDDEN:
        x_rsrc, g_rsrc, out_rsrc = rsrc(normed), rsrc(gate), rsrc(mixed)
        total = fx.Float32(0.0)
        for stream in range_constexpr(HC):
            offset = row * WIDTH + stream * HIDDEN + hidden
            x = fx.Float32(
                fx.BFloat16(bo.buffer_load(x_rsrc, offset, vec_width=1, dtype=T.bf16))
            )
            g = fx.Float32(
                fx.BFloat16(bo.buffer_load(g_rsrc, offset, vec_width=1, dtype=T.bf16))
            )
            total += rcp(fx.Float32(1.0) + exp(-g)) * x
        bo.buffer_store(
            fx.BFloat16(total * fx.Float32(1.0 / HC)),
            out_rsrc,
            row * HIDDEN + hidden,
        )


@flyc.jit
def _launch_gated_mean(
    normed: Int64,
    gate: Int64,
    mixed: Int64,
    samples: Int32,
    stream: Stream = _CURRENT_STREAM,
):
    _gated_mean_kernel(normed, gate, mixed).launch(
        grid=(samples, (HIDDEN + THREADS - 1) // THREADS, 1),
        block=(THREADS, 1, 1),
        stream=stream,
    )


@functools.cache
def build_hyperconnection(
    samples: int,
    eps: float = 1e-6,
    *,
    fused: bool = False,
    pending: bool = False,
    has_inject: bool = False,
):
    if not 1 <= samples <= 128:
        raise ValueError("Qwen3.8 HC supports 1..128 decode tokens")
    if eps <= 0:
        raise ValueError("RMSNorm epsilon must be positive")
    if pending and not fused:
        raise ValueError("deferred combine requires fused HC")
    if has_inject and not fused:
        raise ValueError("deferred injection requires fused HC")
    up_group = 2 if samples >= 4 and samples % 2 == 0 else 1
    down_group = 4 if samples == 16 else (2 if samples >= 4 and samples % 2 == 0 else 1)
    down_outputs = RANK + HC if has_inject else RANK

    @fx.struct
    class Reduction:
        partials: fx.Array[fx.Float32, WAVES * down_group, 16]

    @flyc.kernel(known_block_size=[THREADS, 1, 1])
    def norm_kernel(
        hyper_input: Int64,
        norm_weight: Int64,
        normed: Int64,
        block_output: Int64,
        raw_prev: Int64,
        hyper_output: Int64,
    ):
        bid = gpu.block_idx.x
        tid = gpu.thread_idx.x
        lane, wave = tid % 64, tid // 64
        x_rsrc, w_rsrc, out_rsrc = rsrc(hyper_input), rsrc(norm_weight), rsrc(normed)
        block_rsrc, raw_rsrc, h_out_rsrc = (
            rsrc(block_output),
            rsrc(raw_prev),
            rsrc(hyper_output),
        )
        factor = fx.Float32(0.0)
        if pending:
            raw = fx.Float32(
                fx.BFloat16(bo.buffer_load(raw_rsrc, bid, vec_width=1, dtype=T.bf16))
            )
            scaled = fx.Float32(fx.BFloat16(raw * fx.Float32(1.0 / HC)))
            factor = fx.Float32(2.0) * rcp(fx.Float32(1.0) + exp(-scaled))
        partial = fx.Float32(0.0)
        for step in range_constexpr(5):
            pair = bid * (HIDDEN // 2) + tid + step * THREADS
            word = fx.Int32(bo.buffer_load(x_rsrc, pair, vec_width=1, dtype=T.i32))
            if pending:
                bpair = (bid // HC) * (HIDDEN // 2) + tid + step * THREADS
                block_word = fx.Int32(
                    bo.buffer_load(block_rsrc, bpair, vec_width=1, dtype=T.i32)
                )
                lo, hi = _unpack(word)
                blo, bhi = _unpack(block_word)
                word = _pack(lo + blo * factor, hi + bhi * factor)
                bo.buffer_store(word, h_out_rsrc, pair)
            lo, hi = _unpack(word)
            partial = partial + lo * lo + hi * hi
        shared = fx.SharedAllocator().allocate(Reduction).peek().partials.ptr
        partial = _wave_sum(partial)
        if lane == 0:
            fx.ptr_store(partial, shared + wave)
        gpu.barrier()
        total = fx.Float32(0.0)
        for source_wave in range_constexpr(WAVES):
            total = total + fx.ptr_load(shared + source_wave)
        inverse = rsq(total * fx.Float32(1.0 / HIDDEN) + fx.Float32(eps))
        for step in range_constexpr(5):
            pair = bid * (HIDDEN // 2) + tid + step * THREADS
            word = fx.Int32(bo.buffer_load(x_rsrc, pair, vec_width=1, dtype=T.i32))
            if pending:
                bpair = (bid // HC) * (HIDDEN // 2) + tid + step * THREADS
                block_word = fx.Int32(
                    bo.buffer_load(block_rsrc, bpair, vec_width=1, dtype=T.i32)
                )
                lo, hi = _unpack(word)
                blo, bhi = _unpack(block_word)
                word = _pack(lo + blo * factor, hi + bhi * factor)
            weight_pair = pair % (WIDTH // 2)
            lo, hi = _unpack(word)
            wlo = fx.Float32(
                bo.buffer_load(w_rsrc, weight_pair * 2, vec_width=1, dtype=T.f32)
            )
            whi = fx.Float32(
                bo.buffer_load(w_rsrc, weight_pair * 2 + 1, vec_width=1, dtype=T.f32)
            )
            result = _pack(
                lo * inverse * (fx.Float32(1.0) + wlo),
                hi * inverse * (fx.Float32(1.0) + whi),
            )
            bo.buffer_store(result, out_rsrc, pair)

    @flyc.kernel(known_block_size=[THREADS, 1, 1])
    def down_kernel(normed: Int64, down_weight: Int64, low: Int64, raw_out: Int64):
        bid = gpu.block_idx.x
        row_group, output = bid // down_outputs, bid % down_outputs
        tid, lane, wave = (
            gpu.thread_idx.x,
            gpu.thread_idx.x % 64,
            gpu.thread_idx.x // 64,
        )
        x_rsrc, w_rsrc, out_rsrc, raw_rsrc = (
            rsrc(normed),
            rsrc(down_weight),
            rsrc(low),
            rsrc(raw_out),
        )
        values = []
        for local_row in range_constexpr(down_group):
            values.append(fx.Float32(0.0))
        for step in range_constexpr(20):
            pair = tid + step * THREADS
            w = fx.Int32(
                bo.buffer_load(
                    w_rsrc, output * (WIDTH // 2) + pair, vec_width=1, dtype=T.i32
                )
            )
            for local_row in range_constexpr(down_group):
                row = row_group * down_group + local_row
                x = fx.Int32(
                    bo.buffer_load(
                        x_rsrc, row * (WIDTH // 2) + pair, vec_width=1, dtype=T.i32
                    )
                )
                values[local_row] = dot2_bf16(x, w, values[local_row])
        shared = fx.SharedAllocator().allocate(Reduction).peek().partials.ptr
        for local_row in range_constexpr(down_group):
            value = _wave_sum(values[local_row])
            if lane == 0:
                fx.ptr_store(value, shared + local_row * WAVES + wave)
        gpu.barrier()
        if tid == 0:
            for local_row in range_constexpr(down_group):
                total = fx.Float32(0.0)
                for source_wave in range_constexpr(WAVES):
                    total = total + fx.ptr_load(
                        shared + local_row * WAVES + source_wave
                    )
                projected = fx.Float32(fx.BFloat16(total))
                if has_inject and output >= RANK:
                    bo.buffer_store(
                        fx.BFloat16(projected),
                        raw_rsrc,
                        (row_group * down_group + local_row) * HC + output - RANK,
                    )
                else:
                    scaled = fx.Float32(fx.BFloat16(projected * fx.Float32(1.0 / HC)))
                    activated = scaled * rcp(fx.Float32(1.0) + exp(-scaled))
                    bo.buffer_store(
                        fx.BFloat16(activated),
                        out_rsrc,
                        (row_group * down_group + local_row) * RANK + output,
                    )

    @flyc.kernel(known_block_size=[UP_THREADS, 1, 1])
    def up_kernel(normed: Int64, low: Int64, up_weight: Int64, mixed: Int64):
        bid = gpu.block_idx.x
        row_group, tile = bid // (HIDDEN // UP_WAVES), bid % (HIDDEN // UP_WAVES)
        tid = gpu.thread_idx.x
        lane, wave = tid % 64, tid // 64
        hidden = tile * UP_WAVES + wave
        x_rsrc, low_rsrc, w_rsrc, out_rsrc = (
            rsrc(normed),
            rsrc(low),
            rsrc(up_weight),
            rsrc(mixed),
        )
        low_words = []
        results = []
        for local_row in range_constexpr(up_group):
            row = row_group * up_group + local_row
            words = []
            for step in range_constexpr(3):
                pair = lane + step * 64
                word = fx.Int32(0)
                if pair < RANK // 2:
                    word = fx.Int32(
                        bo.buffer_load(
                            low_rsrc, row * (RANK // 2) + pair, vec_width=1, dtype=T.i32
                        )
                    )
                words.append(word)
            low_words.append(words)
            results.append(fx.Float32(0.0))
        for stream in range_constexpr(HC):
            output = stream * HIDDEN + hidden
            dots = []
            for local_row in range_constexpr(up_group):
                dots.append(fx.Float32(0.0))
            for step in range_constexpr(3):
                pair = lane + step * 64
                if pair < RANK // 2:
                    weight = fx.Int32(
                        bo.buffer_load(
                            w_rsrc,
                            output * (RANK // 2) + pair,
                            vec_width=1,
                            dtype=T.i32,
                        )
                    )
                    for local_row in range_constexpr(up_group):
                        dots[local_row] = dot2_bf16(
                            low_words[local_row][step], weight, dots[local_row]
                        )
            for local_row in range_constexpr(up_group):
                row = row_group * up_group + local_row
                gate = fx.Float32(fx.BFloat16(_wave_sum(dots[local_row])))
                if lane == 0:
                    value = fx.Float32(
                        fx.BFloat16(
                            bo.buffer_load(
                                x_rsrc, row * WIDTH + output, vec_width=1, dtype=T.bf16
                            )
                        )
                    )
                    results[local_row] = (
                        results[local_row] + rcp(fx.Float32(1.0) + exp(-gate)) * value
                    )
        if lane == 0:
            for local_row in range_constexpr(up_group):
                row = row_group * up_group + local_row
                bo.buffer_store(
                    fx.BFloat16(results[local_row] * fx.Float32(1.0 / HC)),
                    out_rsrc,
                    row * HIDDEN + hidden,
                )

    @flyc.kernel(known_block_size=[THREADS, 1, 1])
    def combine_kernel(
        hyper_input: Int64,
        normed: Int64,
        block_output: Int64,
        inject_weight: Int64,
        output: Int64,
    ):
        bid = gpu.block_idx.x
        row, stream = bid // HC, bid % HC
        tid, lane, wave = (
            gpu.thread_idx.x,
            gpu.thread_idx.x % 64,
            gpu.thread_idx.x // 64,
        )
        h_rsrc, x_rsrc, b_rsrc = rsrc(hyper_input), rsrc(normed), rsrc(block_output)
        w_rsrc, out_rsrc = rsrc(inject_weight), rsrc(output)
        dot = fx.Float32(0.0)
        for step in range_constexpr(20):
            pair = tid + step * THREADS
            x = fx.Int32(
                bo.buffer_load(
                    x_rsrc, row * (WIDTH // 2) + pair, vec_width=1, dtype=T.i32
                )
            )
            weight = fx.Int32(
                bo.buffer_load(
                    w_rsrc, stream * (WIDTH // 2) + pair, vec_width=1, dtype=T.i32
                )
            )
            dot = dot2_bf16(x, weight, dot)
        shared = fx.SharedAllocator().allocate(Reduction).peek().partials.ptr
        dot = _wave_sum(dot)
        if lane == 0:
            fx.ptr_store(dot, shared + wave)
        gpu.barrier()
        if tid == 0:
            total = fx.Float32(0.0)
            for source_wave in range_constexpr(WAVES):
                total = total + fx.ptr_load(shared + source_wave)
            projected = fx.Float32(fx.BFloat16(total))
            scaled = fx.Float32(fx.BFloat16(projected * fx.Float32(1.0 / HC)))
            factor = fx.Float32(2.0) * rcp(fx.Float32(1.0) + exp(-scaled))
            fx.ptr_store(factor, shared)
        gpu.barrier()
        factor = fx.ptr_load(shared)
        for step in range_constexpr(5):
            pair = tid + step * THREADS
            hword = fx.Int32(
                bo.buffer_load(
                    h_rsrc,
                    row * (WIDTH // 2) + stream * (HIDDEN // 2) + pair,
                    vec_width=1,
                    dtype=T.i32,
                )
            )
            bword = fx.Int32(
                bo.buffer_load(
                    b_rsrc, row * (HIDDEN // 2) + pair, vec_width=1, dtype=T.i32
                )
            )
            hlo, hhi = _unpack(hword)
            blo, bhi = _unpack(bword)
            bo.buffer_store(
                _pack(hlo + blo * factor, hhi + bhi * factor),
                out_rsrc,
                row * (WIDTH // 2) + stream * (HIDDEN // 2) + pair,
            )

    @flyc.jit
    def launch_norm(
        x: Int64,
        weight: Int64,
        out: Int64,
        block: Int64,
        raw: Int64,
        h_out: Int64,
        stream: Stream = _CURRENT_STREAM,
    ):
        norm_kernel(
            x,
            weight,
            out,
            block,
            raw,
            h_out,
            value_attrs={"rocdl.flat_work_group_size": f"{THREADS},{THREADS}"},
        ).launch(grid=(samples * HC, 1, 1), block=(THREADS, 1, 1), stream=stream)

    @flyc.jit
    def launch_down(
        x: Int64,
        weight: Int64,
        out: Int64,
        raw: Int64,
        stream: Stream = _CURRENT_STREAM,
    ):
        down_kernel(
            x,
            weight,
            out,
            raw,
            value_attrs={"rocdl.flat_work_group_size": f"{THREADS},{THREADS}"},
        ).launch(
            grid=((samples // down_group) * down_outputs, 1, 1),
            block=(THREADS, 1, 1),
            stream=stream,
        )

    @flyc.jit
    def launch_up(
        x: Int64,
        low: Int64,
        weight: Int64,
        out: Int64,
        stream: Stream = _CURRENT_STREAM,
    ):
        up_kernel(
            x,
            low,
            weight,
            out,
            value_attrs={"rocdl.flat_work_group_size": f"{UP_THREADS},{UP_THREADS}"},
        ).launch(
            grid=((samples // up_group) * (HIDDEN // UP_WAVES), 1, 1),
            block=(UP_THREADS, 1, 1),
            stream=stream,
        )

    @flyc.jit
    def launch_combine(
        hyper: Int64,
        normed: Int64,
        block: Int64,
        weight: Int64,
        out: Int64,
        stream: Stream = _CURRENT_STREAM,
    ):
        combine_kernel(
            hyper,
            normed,
            block,
            weight,
            out,
            value_attrs={"rocdl.flat_work_group_size": f"{THREADS},{THREADS}"},
        ).launch(grid=(samples * HC, 1, 1), block=(THREADS, 1, 1), stream=stream)

    for name, launch in (
        ("norm", launch_norm),
        ("down", launch_down),
        ("up", launch_up),
        ("combine", launch_combine),
    ):
        launch.func.__name__ = (
            f"qwen38_hc_{name}_s{samples}_e{eps}_f{int(fused)}"
            f"_p{int(pending)}_i{int(has_inject)}"
        )
    return launch_norm, launch_down, launch_up, launch_combine


def _check(
    tensor: torch.Tensor,
    shape: tuple[int, ...],
    device: torch.device,
    name: str,
    dtype: torch.dtype = torch.bfloat16,
) -> None:
    if (
        tensor.shape != shape
        or tensor.dtype != dtype
        or tensor.device != device
        or not tensor.is_contiguous()
    ):
        raise ValueError(f"{name} must be contiguous {dtype} {shape} on {device}")


def mix(hc, hyper_input: torch.Tensor):
    """Compute grouped norm and low-rank gated mean in three FlyDSL launches."""
    samples = hyper_input.shape[0]
    device = hyper_input.device
    if (
        not hyper_input.is_cuda
        or getattr(torch.cuda.get_device_properties(device), "gcnArchName", "").split(
            ":"
        )[0]
        != "gfx950"
    ):
        raise ValueError("Qwen3.8 HC FlyDSL kernels require gfx950")
    if not 1 <= samples <= 16:
        raise ValueError("Qwen3.8 HC FlyDSL kernels require 1..16 tokens")
    _check(hyper_input, (samples, WIDTH), device, "hyper_input")
    _check(hc.hc_norm.weight, (WIDTH,), device, "hc_norm.weight", torch.float32)
    _check(
        hc.input_mix_weight_down.weight,
        (RANK, WIDTH),
        device,
        "input_mix_weight_down.weight",
    )
    _check(
        hc.input_mix_weight_up.weight,
        (WIDTH, RANK),
        device,
        "input_mix_weight_up.weight",
    )
    norm, down, up, _ = build_hyperconnection(samples, hc.hc_norm.variance_epsilon)
    normed = torch.empty_like(hyper_input)
    low = torch.empty((samples, RANK), dtype=torch.bfloat16, device=device)
    mixed = torch.empty((samples, HIDDEN), dtype=torch.bfloat16, device=device)
    stream = torch.cuda.current_stream(device)
    norm(
        hyper_input.data_ptr(),
        hc.hc_norm.weight.data_ptr(),
        normed.data_ptr(),
        hyper_input.data_ptr(),
        hyper_input.data_ptr(),
        hyper_input.data_ptr(),
        stream=stream,
    )
    down(
        normed.data_ptr(),
        hc.input_mix_weight_down.weight.data_ptr(),
        low.data_ptr(),
        low.data_ptr(),
        stream=stream,
    )
    up(
        normed.data_ptr(),
        low.data_ptr(),
        hc.input_mix_weight_up.weight.data_ptr(),
        mixed.data_ptr(),
        stream=stream,
    )
    return mixed, (hyper_input, normed)


def mix_fused(hc, hyper_input: torch.Tensor, pending=None):
    """Run the served HC path: deferred combine, norm, mix and next injection.

    The deferred combine shares the norm CTA, and the next injection logits
    share the down-projection CTA. From 16 rows the up projection uses the
    existing MFMA GEMM, followed by a FlyDSL gated mean; smaller batches use
    one FlyDSL up-and-mean launch.
    """
    samples, device = hyper_input.shape[0], hyper_input.device
    if not 1 <= samples <= 32:
        raise ValueError("Qwen3.8 fused HC FlyDSL kernels require 1..32 tokens")
    if (
        not hyper_input.is_cuda
        or getattr(torch.cuda.get_device_properties(device), "gcnArchName", "").split(
            ":"
        )[0]
        != "gfx950"
    ):
        raise ValueError("Qwen3.8 fused HC FlyDSL kernels require gfx950")
    _check(hyper_input, (samples, WIDTH), device, "hyper_input")
    _check(hc.hc_norm.weight, (WIDTH,), device, "hc_norm.weight", torch.float32)
    has_inject = hc.block_inject_weight is not None
    expected_weight = (RANK + HC if has_inject else RANK, WIDTH)
    _check(hc.fused_w_cat, expected_weight, device, "fused_w_cat")
    _check(
        hc.input_mix_weight_up.weight,
        (WIDTH, RANK),
        device,
        "input_mix_weight_up.weight",
    )
    has_pending = pending is not None
    if has_pending:
        block, raw_prev = pending
        _check(block, (samples, HIDDEN), device, "pending.block_output")
        _check(raw_prev, (samples, HC), device, "pending.inject_logits")
    else:
        block = hyper_input
        raw_prev = hyper_input
    norm, down, up, _ = build_hyperconnection(
        samples,
        hc.hc_norm.variance_epsilon,
        fused=True,
        pending=has_pending,
        has_inject=has_inject,
    )
    h_new = torch.empty_like(hyper_input) if has_pending else hyper_input
    normed = torch.empty_like(hyper_input)
    low = torch.empty((samples, RANK), dtype=torch.bfloat16, device=device)
    mixed = torch.empty((samples, HIDDEN), dtype=torch.bfloat16, device=device)
    raw_out = (
        torch.empty((samples, HC), dtype=torch.bfloat16, device=device)
        if has_inject
        else None
    )
    stream = torch.cuda.current_stream(device)
    norm(
        hyper_input.data_ptr(),
        hc.hc_norm.weight.data_ptr(),
        normed.data_ptr(),
        block.data_ptr(),
        raw_prev.data_ptr(),
        h_new.data_ptr(),
        stream=stream,
    )
    down(
        normed.data_ptr(),
        hc.fused_w_cat.data_ptr(),
        low.data_ptr(),
        raw_out.data_ptr() if has_inject else low.data_ptr(),
        stream=stream,
    )
    if samples >= 16:
        projected = tgemm.mm(low, hc.input_mix_weight_up.weight, otype=torch.bfloat16)
        _launch_gated_mean(
            normed.data_ptr(),
            projected.data_ptr(),
            mixed.data_ptr(),
            samples,
            stream=stream,
        )
    else:
        up(
            normed.data_ptr(),
            low.data_ptr(),
            hc.input_mix_weight_up.weight.data_ptr(),
            mixed.data_ptr(),
            stream=stream,
        )
    return mixed, (h_new, raw_out)


def combine(hc, block_output: torch.Tensor, residuals):
    """Fuse injection projection with residual update in one FlyDSL launch."""
    hyper_input, normed = residuals
    samples, device = hyper_input.shape[0], hyper_input.device
    if (
        not hyper_input.is_cuda
        or getattr(torch.cuda.get_device_properties(device), "gcnArchName", "").split(
            ":"
        )[0]
        != "gfx950"
    ):
        raise ValueError("Qwen3.8 HC FlyDSL kernels require gfx950")
    if hc.block_inject_weight is None:
        raise ValueError("Qwen3.8 HC combine needs injection weights")
    for name, tensor, shape in (
        ("hyper_input", hyper_input, (samples, WIDTH)),
        ("normed", normed, (samples, WIDTH)),
        ("block_output", block_output, (samples, HIDDEN)),
        ("block_inject_weight.weight", hc.block_inject_weight.weight, (HC, WIDTH)),
    ):
        _check(tensor, shape, device, name)
    _, _, _, launch = build_hyperconnection(samples, hc.hc_norm.variance_epsilon)
    output = torch.empty_like(hyper_input)
    launch(
        hyper_input.data_ptr(),
        normed.data_ptr(),
        block_output.data_ptr(),
        hc.block_inject_weight.weight.data_ptr(),
        output.data_ptr(),
        stream=torch.cuda.current_stream(device),
    )
    return output
