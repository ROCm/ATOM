# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""SP unpacking all-to-all whose scales land in the layout o_proj_full's GEMM reads
(``mxfp8_gemm(..., activation_scale_preshuffled=True)``: AITER's e8m0 shuffle of
``[rows, 256]``, ``gemm._preshuffle_activation_scale``), so the GEMM's own scale
shuffle launch goes. Codes: codes[row, p d : (p + 1) d] from rank p's block (``reference.unpack_reference``).

Scale layout (32-row groups m1, 32 groups k1 of 8 scales): byte
``((((m1 * 32 + k1) * 4 + klane) * 16 + mlane) * 2 + kpack) * 2 + mpack`` for row
``m1 * 32 + mpack * 16 + mlane`` and scale ``k1 * 8 + kpack * 4 + klane``. One
task = (rows r and r + 16, source p, 16 scales): two 16-byte loads, eight dword
stores. Padding rows of the last 32-row group get scale 127 (2^0), as the host
shuffle pads. ``fence``: the release / ticket / publish (``kernel.fence_publish``)
in the same launch (64 workgroups); else the caller published
(``Collectives._publish_wait``) and ``wgs`` workgroups run.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from amd_kernels.common import buffer_ops
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import Vector as Vec

from .host import WORLD, Collectives, _run_compiled
from .kernel import FENCE_WGS, NT, PER_PEER, SYS, UNROLL, _flag_add, _flag_read, _sel4

D = 2048  # code bytes per packed row
ROW_UNITS = (D + D // 32) // 16  # 16-byte units per packed row: 128 codes + 4 scales
CODE_UNITS = D // 16
SCALE_UNITS = ROW_UNITS - CODE_UNITS
K1 = WORLD * D // 32 // 8  # 8-scale groups per gathered row (32)
WGS = 32  # pulling workgroups (8 per source rank)
PAD_SCALE = 0x7F7F7F7F


def compile_pull_unpack_q2048_shuffled(wgs: int = WGS, fence: bool = True):
    assert wgs in (16, 32, 64)
    per_src = wgs // WORLD
    name = f"pull_collectives_ranks4_{'fence_' if fence else ''}pull_unpack_q2048_shuffled_w{wgs}"
    grid = FENCE_WGS if fence else wgs

    @flyc.kernel(name=name, known_block_size=[NT, 1, 1])
    def kernel(
        arg_codes: fx.Int64,
        arg_scales: fx.Int64,
        arg_s0: fx.Int64,
        arg_s1: fx.Int64,
        arg_s2: fx.Int64,
        arg_s3: fx.Int64,
        arg_f0: fx.Int64,
        arg_f1: fx.Int64,
        arg_f2: fx.Int64,
        arg_f3: fx.Int64,
        i64_src_off: fx.Int64,
        i32_seq: fx.Int32,
        i32_me: fx.Int32,
        i32_rows: fx.Int32,
    ):
        tx = fx.thread_idx.x
        bid = fx.block_idx.x
        if const_expr(fence):
            # Each XCD releases its dirty producer lines before any peer sees seq.
            _llvm.FenceOp(_llvm.AtomicOrdering.release, syncscope="")
            own_flag = _sel4(i32_me, arg_f0, arg_f1, arg_f2, arg_f3)
            if tx == 0:
                ticket = _flag_add(own_flag + 4, 1)
                if ticket == FENCE_WGS - 1:
                    _flag_add(own_flag + 4, -FENCE_WGS)
                    fr = buffer_ops.create_buffer_resource_from_addr(own_flag, num_records_bytes=64)
                    buffer_ops.buffer_store(i32_seq, fr, 0, cache_modifier=SYS)
        if bid < wgs:
            p = bid % WORLD
            # The local block also waits (fence: for all XCD releases).
            if const_expr(fence):
                if tx == 0:
                    fa = _sel4(p, arg_f0, arg_f1, arg_f2, arg_f3)
                    f = _flag_read(fa)
                    while f < i32_seq:
                        f = _flag_read(fa)
            else:
                if (tx == 0) & (p != i32_me):
                    fa = _sel4(p, arg_f0, arg_f1, arg_f2, arg_f3)
                    f = _flag_read(fa)
                    while f < i32_seq:
                        f = _flag_read(fa)
            fx.gpu.barrier()
            _llvm.FenceOp(_llvm.AtomicOrdering.acquire, syncscope="")
            n_units = i32_rows * ROW_UNITS
            groups = (i32_rows + 31) // 32
            src_a = _sel4(p, arg_s0, arg_s1, arg_s2, arg_s3) + i64_src_off
            src = buffer_ops.create_buffer_resource_from_addr(src_a, num_records_bytes=fx.Int64(n_units) * 16)
            codes = buffer_ops.create_buffer_resource_from_addr(
                arg_codes, num_records_bytes=fx.Int64(i32_rows) * (D * WORLD)
            )
            scales = buffer_ops.create_buffer_resource_from_addr(
                arg_scales, num_records_bytes=fx.Int64(groups) * (32 * K1 * 8)
            )
            # codes: the packed rows' first 128 units, 16-byte copies
            n_code = i32_rows * CODE_UNITS
            step = NT * UNROLL
            per_block = (n_code + (PER_PEER - 1)) // PER_PEER
            per_block = (per_block + (step - 1)) // step * step
            pb = bid // WORLD
            # In the fused-fence tier, chunks beyond the last code unit contain only
            # masked loads/stores. At 1024 shard rows only 64 of the original
            # 256 chunks are nonempty; at 2048 rows only 128 are nonempty.
            n_blocks = (n_code + per_block - 1) // per_block
            while pb < (n_blocks if const_expr(fence) else PER_PEER):
                start = pb * per_block
                end = start + per_block
                u0 = start + tx
                while u0 < end:
                    vals = []
                    for k in range_constexpr(UNROLL):
                        u = u0 + k * NT
                        row = u // CODE_UNITS
                        c = u - row * CODE_UNITS
                        vals.append(
                            Vec(
                                buffer_ops.buffer_load(
                                    src, (row * ROW_UNITS + c) * 4, vec_width=4, dtype=fx.Int32,
                                    mask=u < n_code, cache_modifier=SYS,
                                )
                            )
                        )
                    for k in range_constexpr(UNROLL):
                        u = u0 + k * NT
                        row = u // CODE_UNITS
                        c = u - row * CODE_UNITS
                        off_c = (row * (D * WORLD) + p * D) // 4 + c * 4
                        buffer_ops.buffer_store(vals[k], codes, off_c, mask=u < n_code)
                    u0 = u0 + step
                pb = pb + per_src
            # scales: task = (32-row group, lane row 0..15, unit j 0..3)
            n_task = groups * (16 * SCALE_UNITS)
            t = (bid // WORLD) * NT + tx
            while t < n_task:
                m1 = t // (16 * SCALE_UNITS)
                rem = t - m1 * (16 * SCALE_UNITS)
                mlane = rem // SCALE_UNITS
                j = rem - mlane * SCALE_UNITS
                r0 = m1 * 32 + mlane
                r1 = r0 + 16
                ok0 = r0 < i32_rows
                ok1 = r1 < i32_rows
                la = Vec(buffer_ops.buffer_load(src, (r0 * ROW_UNITS + CODE_UNITS + j) * 4, vec_width=4,
                                                dtype=fx.Int32, mask=ok0, cache_modifier=SYS))
                lb = Vec(buffer_ops.buffer_load(src, (r1 * ROW_UNITS + CODE_UNITS + j) * 4, vec_width=4,
                                                dtype=fx.Int32, mask=ok1, cache_modifier=SYS))
                a = [ok0.select(la[q], fx.Int32(PAD_SCALE)) for q in range_constexpr(4)]
                b = [ok1.select(lb[q], fx.Int32(PAD_SCALE)) for q in range_constexpr(4)]
                for h in range_constexpr(2):
                    k1 = p * (D // 32 // 8) + j * 2 + h
                    for kl in range_constexpr(4):
                        sh = 8 * kl
                        w = ((a[2 * h] >> sh) & 0xFF) | (((b[2 * h] >> sh) & 0xFF) << 8)
                        w = w | (((a[2 * h + 1] >> sh) & 0xFF) << 16) | (((b[2 * h + 1] >> sh) & 0xFF) << 24)
                        off = ((m1 * K1 + k1) * 4 + kl) * 16 + mlane
                        buffer_ops.buffer_store(w, scales, off)
                t = t + per_src * NT

    @flyc.jit
    def launch_unpack_q2048_shuffled(
        arg_codes: fx.Int64,
        arg_scales: fx.Int64,
        arg_s0: fx.Int64,
        arg_s1: fx.Int64,
        arg_s2: fx.Int64,
        arg_s3: fx.Int64,
        arg_f0: fx.Int64,
        arg_f1: fx.Int64,
        arg_f2: fx.Int64,
        arg_f3: fx.Int64,
        i64_src_off: fx.Int64,
        i32_seq: fx.Int32,
        i32_me: fx.Int32,
        i32_rows: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(
            arg_codes, arg_scales, arg_s0, arg_s1, arg_s2, arg_s3, arg_f0, arg_f1, arg_f2, arg_f3,
            i64_src_off, i32_seq, i32_me, i32_rows,
        ).launch(grid=(grid, 1, 1), block=(NT, 1, 1), stream=stream)

    launch_unpack_q2048_shuffled.kernel_name = name
    return launch_unpack_q2048_shuffled


@functools.cache
def _kernel(wgs: int, fence: bool):
    return compile_pull_unpack_q2048_shuffled(wgs, fence)


# packed rows (tokens) up to which the release/publish rides in the unpack launch
# (the measured split for the row-major unpack)
FENCE_MAX_TOKENS = 8192


def all_to_all_unpack_q2048_shuffled(coll: Collectives, x: torch.Tensor, wgs: int | None = None, fence: bool | None = None):
    """The all-to-all of packed rows ``x`` ``[4 rows, 2048 + 64]`` (row block r to rank r), unpacked
    (``reference.unpack_reference``), with the scales in o_proj_full's GEMM layout:
    (codes ``[rows, 8192]`` e4m3, scales ``[ceil(rows / 32) * 32, 256]`` u8, AITER's
    shuffle), or None when ``x`` does not fit the slot (the caller falls back).
    ``fence``: force the publication form (None: ``FENCE_MAX_TOKENS``). ``wgs``
    (None): all 64 fence workgroups pull in the fenced form, 32 pull after a separate
    publication (64 in the fenced form, 4096 tokens dense
    252.7 -> 250.0 us, MoE next to MFMA 338.3 -> 333.9; 64 separate pullers lose 3-9 us
    at 16384 / 32768 tokens)."""
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("SP unpack uses eager Collectives sequence state")
    x = x.contiguous()
    rows = x.shape[0] // WORLD
    assert x.dtype == torch.uint8 and x.shape == (WORLD * rows, D + D // 32) and rows > 0
    off = coll._stage(x)
    if off is None:
        return None
    p = coll.p
    if fence is None:
        fence = x.shape[0] <= FENCE_MAX_TOKENS
    if wgs is None:
        wgs = FENCE_WGS if fence else WGS
    if fence:
        coll.seq = p.next_seq(coll.set_id)
    else:
        coll._publish_wait()
    codes = torch.empty((rows, WORLD * D), dtype=torch.uint8, device=x.device)
    scales = torch.empty(((rows + 31) // 32 * 32, WORLD * D // 32), dtype=torch.uint8, device=x.device)
    block = x.numel() // WORLD
    _run_compiled(
        _kernel(wgs, fence),
        codes.data_ptr(),
        scales.data_ptr(),
        *[ptr + off for ptr in p.pool_ptrs],
        *[f + coll.flag_off for f in p.flag_ptrs],
        p.rank * block,
        coll.seq,
        p.rank,
        rows,
        coll.stream,
    )
    return codes.view(torch.float8_e4m3fn), scales
