# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Router GEMM split-K partials of bf16 x [m, 6144] and the fp32 gate [128, 6144] as three bf16 terms (gfx950):
acc_hi = x . w0, acc_lo = per BLOCK_K x . w1 then x . w2, in the tier's MFMA (16x16x32 or 32x32x16); also the host
dispatch over the router tiers. A split's workgroups run on one XCD, which then reads its x slice once."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from amd_kernels.warm_up import launcher

H, E = 6144, 128
PRELOAD_HINTS = {
    "llvm_options": {
        "amdgpu-kernarg-preload": True,
        "amdgpu-kernarg-preload-count": 16,
    }
}


def _pk_bf16(a, b):
    return fx.Int32(
        fx.rocdl.cvt_pk_bf16_f32(
            fx.as_ir_value(fx.Float32(a)), fx.as_ir_value(fx.Float32(b))
        )
    )


def _split3(w):
    """4 fp32 (i32 bits) -> three terms of 2 bf16-pair dwords: RN(w), RN(w - w0), RN(w - w0 - w1), exact remainders."""
    parts = [[], [], []]
    for i in range_constexpr(2):
        a = fx.Int32(w[2 * i]).bitcast(fx.Float32)
        b = fx.Int32(w[2 * i + 1]).bitcast(fx.Float32)
        for p in range_constexpr(3):
            pk = _pk_bf16(a, b)
            parts[p].append(pk)
            if p < 2:
                a = a - (pk << 16).bitcast(fx.Float32)
                b = b - (pk & -65536).bitcast(fx.Float32)
    return parts


@launcher
def compile_router_gemm(
    mfma: int, split_k: int, block_k: int, waves: int, helpers: int
):
    """One m-tile per wave, ceil(tiles / waves) m-groups per (n-block, split).

    Args:
        mfma: 16 or 32, the MFMA's N / M width (k 32 or 16 per step).
        helpers: extra waves that only load and split weights.
    """
    assert mfma in (16, 32) and split_k % 8 == 0
    NW = waves
    NT = waves + helpers  # waves per workgroup
    NB = mfma  # experts per n-block = MFMA rows (src0 = weight)
    MT = mfma  # tokens per m-tile = MFMA columns
    SUBK = 32 if mfma == 16 else 16
    KC = H // split_k
    SPB = block_k // SUBK  # MFMA steps per BLOCK_K
    NSTEP = KC // block_k
    assert KC % block_k == 0 and block_k % SUBK == 0
    NACC = MT * NB // 64
    NBLK = E // NB
    W_PIECES = NB * block_k // 4 // (64 * NT)  # 16-byte weight pieces per thread
    X_PIECES = MT * block_k // 8 // 64  # 16-byte x pieces per lane
    assert NB * block_k // 4 == W_PIECES * 64 * NT and X_PIECES >= 1
    W_SLOT = 3 * SPB * 64  # 16-byte chunks: (term, step, lane)
    X_SLOT = SPB * 64
    PPR = block_k // 8  # 16-byte bf16 pieces per row and step
    assert PPR <= 16
    Q8 = SUBK // 8  # lane groups (8 k each) of an MFMA step

    def swz(chunk, kp):
        """LDS chunk of operand chunk ``chunk`` (sub, lane (row, q)), kp = sub * Q8 + q: the low 4 bits xor kp * 16 /
        PPR, so the row-contiguous writes and the lane-order reads both hit 16 distinct 16-byte bank groups."""
        return chunk ^ (kp * (16 // PPR))

    name = f"router_gemm_bf16x3_h6144_e128_mfma{mfma}_s{split_k}_bk{block_k}_w{NW}_hw{helpers}"

    @fx.struct
    class Shared:
        w: fx.Array[fx.Int32, NSTEP * W_SLOT * 4, 16]
        x: fx.Array[fx.Int32, 2 * NW * X_SLOT * 4, 16]

    def mfma_op(a, b, acc):
        av = Vec(a).bitcast(fx.BFloat16)
        bv = Vec(b).bitcast(fx.BFloat16)
        fn = (
            fx.rocdl.mfma_f32_16x16x32_bf16
            if mfma == 16
            else fx.rocdl.mfma_f32_32x32x16_bf16
        )
        return Vec(fn(Vec.make_type(NACC, fx.Float32), [av, bv, acc, 0, 0, 0]))

    @flyc.kernel(name=name, known_block_size=[64 * NT, 1, 1])
    def kernel(
        arg_x: fx.Int64,  # [m, H] bf16, row stride i32_sx
        arg_w: fx.Int64,  # [E, H] fp32
        arg_p: fx.Int64,  # partials [split_k, m, E] fp32
        i32_m: fx.Int32,
        i32_sx: fx.Int32,
    ):
        tx, bid = fx.thread_idx.x, fx.block_idx.x
        lane = tx % 64
        wave = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
        # bid = (g * NBLK + nb) * split_k + s: a split's workgroups share an XCD (split_k is a multiple of 8)
        s = bid % split_k
        nb = (bid // split_k) % NBLK
        g = bid // (split_k * NBLK)
        t0 = (g * NW + wave) * MT  # this wave's first token
        smem = fx.SharedAllocator().allocate(Shared).peek()

        def view16(arr, n):
            return fx.logical_divide(
                fx.make_view(fx.recast_iter(fx.Int32, arr.ptr), fx.make_layout(n, 1)),
                fx.make_layout(4, 1),
            )

        w16 = view16(smem.w, NSTEP * W_SLOT * 4)
        w8 = fx.logical_divide(
            fx.make_view(
                fx.recast_iter(fx.Int32, smem.w.ptr),
                fx.make_layout(NSTEP * W_SLOT * 4, 1),
            ),
            fx.make_layout(2, 1),
        )
        x16 = view16(smem.x, 2 * NW * X_SLOT * 4)
        atom16 = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Int32)
        atom8 = fx.make_copy_atom(fx.UniversalCopy64b(), fx.Int32)
        wr = buffer_ops.create_buffer_resource_from_addr(
            arg_w, num_records_bytes=E * H * 4
        )
        xr = buffer_ops.create_buffer_resource_from_addr(
            arg_x, num_records_bytes=fx.Int64(i32_m) * fx.Int64(i32_sx * 2)
        )
        pr = buffer_ops.create_buffer_resource_from_addr(
            arg_p,
            num_records_bytes=fx.Int64(i32_m) * (split_k * E * 4),
        )
        k_base = s * KC

        # every load first, row-contiguous: x piece u = (row u // (block_k / 8), k 8 (u % (block_k / 8))), w piece =
        # (expert, 4 k)
        def x_piece(it):
            u = it * 64 + lane
            return u // (block_k // 8), u % (block_k // 8)

        xl = []
        for st in range_constexpr(NSTEP):
            for it in range_constexpr(X_PIECES):
                r, kp = x_piece(it)
                row = t0 + r
                xl.append(
                    Vec(
                        buffer_ops.buffer_load(
                            xr,
                            (row * i32_sx + k_base + st * block_k + kp * 8) >> 1,
                            vec_width=4,
                            dtype=fx.Int32,
                            mask=(row < i32_m) & (wave < NW),
                        )
                    )
                )

        def w_piece(it):
            u = it * 64 * NT + tx
            return u // (block_k // 4), u % (block_k // 4)

        wl = []
        for st in range_constexpr(NSTEP):
            for it in range_constexpr(W_PIECES):
                n, kp = w_piece(it)
                wl.append(
                    Vec(
                        buffer_ops.buffer_load(
                            wr,
                            (nb * NB + n) * H + k_base + st * block_k + kp * 4,
                            vec_width=4,
                            dtype=fx.Int32,
                        )
                    )
                )

        def stage_w(st):
            """Step ``st``'s weights -> 3 bf16 terms, chunk (term, sub, lane (n, q))."""
            slot = st * W_SLOT
            for it in range_constexpr(W_PIECES):
                n, kp = w_piece(it)
                kq = kp // 2  # sub * Q8 + q of the 4 weights
                chunk = slot + swz((kq // Q8) * 64 + n + NB * (kq % Q8), kq)
                parts = _split3(
                    [fx.Int32(wl[st * W_PIECES + it][e]) for e in range_constexpr(4)]
                )
                for p in range_constexpr(3):
                    r8 = fx.make_rmem_tensor(fx.make_layout(2, 1), fx.Int32)
                    r8.store(Vec.from_elements(parts[p], fx.Int32))
                    dst = fx.slice(w8, (None, (chunk + p * SPB * 64) * 2 + kp % 2))
                    fx.copy(atom8, r8, dst)

        def stage_x(st):
            """This wave's x of step ``st`` -> its private slot (ping-pong)."""
            xslot = ((st % 2) * NW + wave) * X_SLOT
            for it in range_constexpr(X_PIECES):
                r, kp = x_piece(it)
                r16 = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                r16.store(xl[st * X_PIECES + it])
                c = swz((kp // Q8) * 64 + r + MT * (kp % Q8), kp)
                fx.copy(atom16, r16, fx.slice(x16, (None, xslot + c)))

        def operands(st):
            """(x, weight terms) of step ``st`` in the MFMA lane order."""
            slot = st * W_SLOT
            xslot = ((st % 2) * NW + wave) * X_SLOT
            xs, ws = [], []
            for j in range_constexpr(SPB):
                c = swz(j * 64 + lane, j * Q8 + lane // MT)
                r16 = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                fx.copy(atom16, fx.slice(x16, (None, xslot + c)), r16)
                xs.append(Vec(r16.load()))
            for p in range_constexpr(3):
                for j in range_constexpr(SPB):
                    c = swz(j * 64 + lane, j * Q8 + lane // MT)
                    r16 = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                    fx.copy(atom16, fx.slice(w16, (None, slot + p * SPB * 64 + c)), r16)
                    ws.append(Vec(r16.load()))
            return xs, ws

        # the whole weight slice once (one barrier); x in private slots per step
        for st in range_constexpr(NSTEP):
            stage_w(st)
        fx.gpu.barrier()
        if wave < NW:
            stage_x(0)
            acc_hi = Vec.filled(NACC, 0.0, fx.Float32)
            acc_lo = Vec.filled(NACC, 0.0, fx.Float32)
            nxt = operands(0)
            for st in range_constexpr(NSTEP):
                # the next step's operands are read before this step's MFMAs
                xs, ws = nxt
                if const_expr(st + 1 < NSTEP):
                    stage_x(st + 1)
                    nxt = operands(st + 1)
                # keep the reads ahead of the MFMAs (LLVM would sink them to their uses)
                fx.rocdl.sched_barrier(0)
                for j in range_constexpr(SPB):
                    acc_hi = mfma_op(ws[j], xs[j], acc_hi)
                for p in range_constexpr(1, 3):
                    for j in range_constexpr(SPB):
                        acc_lo = mfma_op(ws[p * SPB + j], xs[j], acc_lo)
            out = [
                fx.Float32(acc_hi[i]) + fx.Float32(acc_lo[i])
                for i in range_constexpr(NACC)
            ]
            # D[n][m]: lane (m, q) holds n = 8 j + 4 q + i (32x32) / 4 q + i (16x16)
            m = t0 + lane % MT
            q = lane // MT
            for j in range_constexpr(NACC // 4):
                n = nb * NB + j * 8 + q * 4
                buffer_ops.buffer_store(
                    Vec.from_elements(out[4 * j : 4 * j + 4], fx.Float32),
                    pr,
                    (s * i32_m + m) * E + n,
                    mask=m < i32_m,
                )

    @flyc.jit
    def launch_router_gemm_mxfp4_moe(
        arg_x: fx.Int64,
        arg_w: fx.Int64,
        arg_p: fx.Int64,
        i32_m: fx.Int32,
        i32_sx: fx.Int32,
        stream: fx.Stream,
    ):
        groups = (i32_m + (MT * NW - 1)) // (MT * NW)
        kernel(arg_x, arg_w, arg_p, i32_m, i32_sx).launch(
            grid=(fx.Int64(groups * (NBLK * split_k)), 1, 1),
            block=(64 * NT, 1, 1),
            stream=stream,
        )

    launch_router_gemm_mxfp4_moe.kernel_name = name
    # LDS allows 4 (32x32) / 8 (16x16) waves per CU: let LLVM use the registers
    launch_router_gemm_mxfp4_moe.compile_hints = {**PRELOAD_HINTS, "waves_per_eu": 1 if mfma == 32 else 2}
    return launch_router_gemm_mxfp4_moe


# ---- host: compile cache, tier dispatch
import functools  # noqa: E402

import torch  # noqa: E402


def _run_compiled(exe, *args):
    """The first call compiles and runs (``flyc.compile``); later calls dispatch the cached CompiledFunction."""
    import flydsl.compiler as flyc

    cf = getattr(exe, "_cf", None)
    if cf is None:
        exe._cf = flyc.compile(exe, *args)
    else:
        cf(*args)


# Larger calls use the fused BF16x3 router and MXFP4 input quantizer.
MAX_TOKENS = 2047
# inclusive token bound -> (MFMA width, split_k, BLOCK_K): the partials' K split and sum order (the top-k kernels and
# the router + quant launches read the split)
SPLIT_TIERS = ((64, (16, 48, 64)), (192, (16, 16, 128)), (MAX_TOKENS, (32, 16, 64)))


def split_tier(num_tokens: int) -> tuple[int, int, int] | None:
    """(MFMA width, split_k, BLOCK_K) for ``num_tokens``, None above ``MAX_TOKENS``."""
    for bound, tier in SPLIT_TIERS:
        if num_tokens <= bound:
            return tier
    return None


@functools.cache
def _get_kernel(mfma: int, split_k: int, block_k: int, waves: int, helpers: int):
    return compile_router_gemm(mfma, split_k, block_k, waves, helpers)


def _waves(mfma: int, num_tokens: int) -> tuple[int, int]:
    """(m-tile waves, weight-only helper waves) per workgroup, cold gate: 16-row tiles 8 + 0 above 8 tiles (172 rows
    4.9 vs 5.2 us with 4); 32-row tiles 2 + 2 to 8 tiles (240 rows 6.1 vs 6.9), 4 + 0 to 16."""
    # 513-2047 (64 workgroups per m-group, 256 CUs): the fewest token waves for the same rounds, the rest stage weights
    if mfma == 32 and 513 <= num_tokens <= 2047:
        rounds = (num_tokens + 1023) // 1024
        waves = (num_tokens + 128 * rounds - 1) // (128 * rounds)
        # seven waves at 769-896: no clear gain (A/B ranges overlap at 896); with two rounds seven wins
        if num_tokens <= 1024 and waves == 7:
            return 8, 0
        return waves, 8 - waves
    tiles = (num_tokens + mfma - 1) // mfma
    if mfma == 16:
        return (8, 0) if tiles > 8 else (4, 0)
    if tiles <= 8:
        return 2, 2
    return (4, 0) if tiles <= 16 else (8, 0)


def _kernel_config(num_tokens):
    tier = split_tier(num_tokens)
    if tier is None:
        return None
    mfma, split_k, block_k = tier
    if 193 <= num_tokens <= 384:
        # 16-row MFMA, 4 waves: up to three groups per CU instead of one partly idle 32-row group; bitwise equal
        return 16, split_k, block_k, 4, 0
    return mfma, split_k, block_k, *_waves(mfma, num_tokens)


# router_gemm_nblocks at 1-384 rows, two waves to 32: -0.15..-0.30 us at 8-32 rows, -0.05..-0.15 at 64-256
NBLOCKS_TWO_WAVES_MAX = 32
# router_gemm_direct (no LDS) on the line-layout gate copy to one m-tile: router + top-k chain -0.011..-0.054 us at
# 1-16 rows, +0.02 at 24-32
DIRECT_LINES_MAX = 16
# nblocks ``x_direct`` (four waves): router -0.21..-0.01 us at 33-56 rows, +0.03 at 64, +0.04..+0.08 with two waves
X_DIRECT_FROM, X_DIRECT_MAX = 33, 57


# router_gemm_nblocks_lines (two waves, line-layout gate): faster only at 17-32 rows
NBLOCKS_LINES_FROM, NBLOCKS_LINES_MAX = 17, 32

# split-48 partials token-major [m, 48, 128] to here: the top-k reads them at constant split offsets
TOKEN_MAJOR_MAX = 64


def nblocks_config(num_tokens: int) -> tuple[int, int, int, int] | None:
    """(split_k, BLOCK_K, waves, n-blocks per workgroup) of ``router_gemm_nblocks`` where it serves ``num_tokens``, else
    None (this kernel's 32-row MFMA / helper-wave tiers)."""
    config = _kernel_config(num_tokens)
    if config is None:
        return None
    mfma, split_k, block_k, waves, helpers = config
    if mfma != 16 or helpers:
        return None
    if num_tokens <= NBLOCKS_TWO_WAVES_MAX:
        waves = 2
    return split_k, block_k, waves, 1


def router_gemm_partials(
    hidden_states: torch.Tensor, router_weight: torch.Tensor,
    router_weight_lines: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """FP32 partials [m, split_k, 128] (``split_tier``'s split and order), contiguous to 64 rows and a view of
    [split_k, m, 128] above; None outside the covered shapes and row counts.

    Args:
        router_weight_lines: the gate in ``layout.router_line_layout``, used by the 1-32 row tiers when given.
    """
    m = hidden_states.shape[0]
    if (
        m == 0
        or hidden_states.dim() != 2
        or hidden_states.shape[1] != H
        or hidden_states.dtype != torch.bfloat16
        or hidden_states.stride(1) != 1
        or hidden_states.stride(0) % 2
        or router_weight.shape != (E, H)
        or router_weight.dtype != torch.float32
        or not router_weight.is_contiguous()
    ):
        return None
    config = _kernel_config(m)
    if config is None:
        return None
    _, split_k, _, _, _ = config
    token_major = m <= TOKEN_MAJOR_MAX
    partials = torch.empty(
        (split_k, m, E), dtype=torch.float32, device=hidden_states.device
    )
    if (router_weight_lines is not None and m <= DIRECT_LINES_MAX and config[1:3] == (48, 64)
            and router_weight_lines.dtype == torch.float32 and router_weight_lines.is_contiguous()
            and router_weight_lines.numel() == E * H):
        from .router_gemm_direct import _get_kernel as _get_direct_kernel
        tiles = (m + 15) // 16
        _run_compiled(
            _get_direct_kernel(48, 64, tiles, 2, True, token_major),
            hidden_states.data_ptr(),
            router_weight_lines.data_ptr(),
            partials.data_ptr(),
            m,
            hidden_states.stride(0),
            torch.cuda.current_stream(),
        )
        return partials.view(m, split_k, E) if token_major else partials.permute(1, 0, 2)
    if (router_weight_lines is not None and NBLOCKS_LINES_FROM <= m <= NBLOCKS_LINES_MAX
            and router_weight_lines.dtype == torch.float32 and router_weight_lines.is_contiguous()
            and router_weight_lines.numel() == E * H):
        from .router_gemm_nblocks_lines import _get_kernel as _get_lines_kernel
        _run_compiled(_get_lines_kernel(token_major), hidden_states.data_ptr(), router_weight_lines.data_ptr(),
                      partials.data_ptr(), m, hidden_states.stride(0), torch.cuda.current_stream())
        return partials.view(m, split_k, E) if token_major else partials.permute(1, 0, 2)
    nblocks = nblocks_config(m)
    if nblocks is not None:
        from .router_gemm_nblocks import _get_kernel as _get_nblocks_kernel
        x_direct = X_DIRECT_FROM <= m <= X_DIRECT_MAX
        _run_compiled(
            _get_nblocks_kernel(*nblocks, 0, x_direct, token_major),
            hidden_states.data_ptr(),
            router_weight.data_ptr(),
            partials.data_ptr(),
            m,
            hidden_states.stride(0),
            0, 0, 0, 0,
            torch.cuda.current_stream(),
        )
        return partials.view(m, split_k, E) if token_major else partials.permute(1, 0, 2)
    _run_compiled(
        _get_kernel(*config),
        hidden_states.data_ptr(),
        router_weight.data_ptr(),
        partials.data_ptr(),
        m,
        hidden_states.stride(0),
        torch.cuda.current_stream(),
    )
    return partials.view(m, split_k, E) if token_major else partials.permute(1, 0, 2)
