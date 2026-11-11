# SPDX-License-Identifier: MIT
"""MiniMax-M3.1 decode router for 1..128 rows, sharing a launch with the
shared expert's SwiGLU (Triton, two launches).

The work is split as SGLang splits it past 16 rows (``_router_gemv_swiglu_kernel``,
``moe_fused_gate``, the small MoE sort), with the gate and sort in one launch:

1. ``_router_partials_swiglu_kernel``: the router GEMV's split-K partials
   (bf16 x [M, 6144] times bf16 W [128, 6144], fp32) and, in the same grid,
   ``swiglu_oai_split_mxfp8`` of the shared expert's gate_up rows, byte for
   byte. Both are latency-bound and read the same layer input.
2. ``_gate_sort_kernel``: programs of four rows sum the partials in split
   order and gate them (sigmoid, correction bias, top-4, renormalize, scale)
   as ``m31_router.hip`` does; the last of them to finish writes aiter's
   block_m 16 MoE sort in both layouts (as ``router.py``). The other programs
   zero the MoE output.

Against the separate path this replaces the router hgemm, ``topk_gating``,
aiter's two sort launches and the SwiGLU launch (five launches, two). Up to
16 rows it also beats ``router.py``'s one launch plus the SwiGLU launch,
whose single reducing CTA does the whole gate and sort.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from atom.model_ops.blockscale_kernels.quantization import FP8_DTYPE
from atom.model_ops.minimax_m3.router.router import (
    BLOCK_M,
    HIDDEN,
    NUM_EXPERTS,
    TOPK,
    register_presorted_moe,
)
from atom.model_ops.swiglu_oai import (
    swiglu_mxfp8_block,
    swiglu_mxfp8_store,
    swiglu_oai_mxfp8_tile,
)

MIN_ROWS = 1
MAX_ROWS = 128
# router GEMV tile: rows, experts, K step, split partials
_GEMV_M, _GEMV_N, _GEMV_K, _SPLIT = 16, 32, 256, 12
# shared expert gate_up GEMM tile (MXFP8 x MXFP8, tl.dot_scaled): columns and
# K step; its rows and split count follow the row count (gateup_tiling)
_GU_N, _GU_K = 32, 256
_GU_MAX_M, _GU_MAX_SPLIT = 64, 8
_NUM_WARPS = 4
# rows one gate program covers
_GATE_ROWS = 4
_BUF_BLOCK = 4096


@triton.jit
def _router_gemv_tile(
    x_ptr,
    w_ptr,
    part_ptr,
    MP,
    stride_xm,
    num_m_tiles,
    pid,
    K: tl.constexpr,
    N: tl.constexpr,
    GEMV_M: tl.constexpr,
    GEMV_N: tl.constexpr,
    GEMV_K: tl.constexpr,
    SPLIT: tl.constexpr,
):
    """Program ``pid`` of the router GEMV: one (expert block, row tile, split)
    partial of [SPLIT, num_m_tiles * GEMV_M, N] fp32."""
    num_n: tl.constexpr = N // GEMV_N
    pid_n = pid % num_n
    pid_m = pid // num_n % num_m_tiles
    pid_k = pid // (num_n * num_m_tiles)
    offs_n = pid_n * GEMV_N + tl.arange(0, GEMV_N)
    offs_m = pid_m * GEMV_M + tl.arange(0, GEMV_M)
    mask_m = offs_m < MP
    acc = tl.zeros((GEMV_M, GEMV_N), tl.float32)
    for k0 in range(pid_k * GEMV_K, K, GEMV_K * SPLIT):
        offs_k = k0 + tl.arange(0, GEMV_K)
        xv = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :],
            mask=mask_m[:, None],
            other=0.0,
        )
        wv = tl.load(w_ptr + offs_n[:, None] * K + offs_k[None, :])
        acc += tl.dot(xv, tl.trans(wv), out_dtype=tl.float32)
    rows = num_m_tiles * GEMV_M
    tl.store(part_ptr + (pid_k * rows + offs_m)[:, None] * N + offs_n[None, :], acc)


@triton.jit
def _router_partials_swiglu_kernel(
    x_ptr,
    w_ptr,
    part_ptr,
    gu_ptr,
    q_ptr,
    scale_ptr,
    MP,
    stride_xm,
    stride_gu,
    num_m_tiles,
    alpha,
    beta,
    limit,
    K: tl.constexpr,
    N: tl.constexpr,
    GEMV_M: tl.constexpr,
    GEMV_N: tl.constexpr,
    GEMV_K: tl.constexpr,
    SPLIT: tl.constexpr,
    N_INTER: tl.constexpr,
    HAS_LIMIT: tl.constexpr,
    SW_BLOCK: tl.constexpr,
):
    """Programs [0, gemv): split-K partials [SPLIT, num_m_tiles * GEMV_M, N];
    then one program per (row, SW_BLOCK columns) of the SwiGLU."""
    pid = tl.program_id(0)
    num_gemv = (N // GEMV_N) * num_m_tiles * SPLIT
    if pid < num_gemv:
        _router_gemv_tile(
            x_ptr,
            w_ptr,
            part_ptr,
            MP,
            stride_xm,
            num_m_tiles,
            pid,
            K,
            N,
            GEMV_M,
            GEMV_N,
            GEMV_K,
            SPLIT,
        )
    else:
        tiles: tl.constexpr = (N_INTER + SW_BLOCK - 1) // SW_BLOCK
        t = pid - num_gemv
        swiglu_oai_mxfp8_tile(
            gu_ptr,
            q_ptr,
            scale_ptr,
            N_INTER,
            stride_gu,
            alpha,
            beta,
            limit,
            HAS_LIMIT,
            SW_BLOCK,
            t // tiles,
            t % tiles,
        )


@triton.jit
def _router_gateup_partials_kernel(
    x_ptr,
    w_ptr,
    part_ptr,
    xq_ptr,
    xs_ptr,
    wq_ptr,
    ws_ptr,
    gu_part_ptr,
    MP,
    stride_xm,
    num_m_tiles,
    gu_m_tiles,
    K: tl.constexpr,
    N: tl.constexpr,
    GEMV_M: tl.constexpr,
    GEMV_N: tl.constexpr,
    GEMV_K: tl.constexpr,
    SPLIT: tl.constexpr,
    N2: tl.constexpr,
    GU_M: tl.constexpr,
    GU_N: tl.constexpr,
    GU_K: tl.constexpr,
    GU_SPLIT: tl.constexpr,
    W_SHUFFLED: tl.constexpr,
):
    """Programs [0, gemv): the router GEMV partials, as above; then the shared
    expert's gate_up GEMM, E4M3 rows (E8M0 group32 scales) times its E4M3
    weight [N2, K] (E8M0 1x32 scales), as split-K partials
    [GU_SPLIT, gu_m_tiles * GU_M, N2] fp32 for the SwiGLU of the next launch.
    W_SHUFFLED: the weight is in aiter's (16, 16) preshuffled layout,
    [N2 / 16, K / 32, 2, 16 rows, 16 bytes]."""
    pid = tl.program_id(0)
    num_gemv = (N // GEMV_N) * num_m_tiles * SPLIT
    if pid < num_gemv:
        _router_gemv_tile(
            x_ptr,
            w_ptr,
            part_ptr,
            MP,
            stride_xm,
            num_m_tiles,
            pid,
            K,
            N,
            GEMV_M,
            GEMV_N,
            GEMV_K,
            SPLIT,
        )
    else:
        q = pid - num_gemv
        num_n2: tl.constexpr = N2 // GU_N
        pid_n = q % num_n2
        pid_m = q // num_n2 % gu_m_tiles
        pid_k = q // (num_n2 * gu_m_tiles)
        offs_m = pid_m * GU_M + tl.arange(0, GU_M)
        offs_n = pid_n * GU_N + tl.arange(0, GU_N)
        mask_m = offs_m < MP
        acc = tl.zeros((GU_M, GU_N), tl.float32)
        for k0 in range(pid_k * GU_K, K, GU_K * GU_SPLIT):
            offs_k = k0 + tl.arange(0, GU_K)
            offs_g = k0 // 32 + tl.arange(0, GU_K // 32)
            a = tl.load(
                xq_ptr + offs_m[:, None] * K + offs_k[None, :],
                mask=mask_m[:, None],
                other=0.0,
            )
            a_s = tl.load(
                xs_ptr + offs_m[:, None] * (K // 32) + offs_g[None, :],
                mask=mask_m[:, None],
                other=127,
            )
            if W_SHUFFLED:
                # the tile is GU_N / 16 contiguous runs of GU_K * 16 bytes,
                # [k / 32, k % 32 / 16, n % 16, k % 16] each: load them whole
                # and reorder to [k, n] in registers
                G: tl.constexpr = GU_N // 16
                J: tl.constexpr = GU_K // 32
                runs = ((pid_n * G + tl.arange(0, G)) * (K // 32) + k0 // 32) * 512
                raw = tl.load(wq_ptr + runs[:, None] + tl.arange(0, J * 512)[None, :])
                b = tl.reshape(
                    tl.permute(tl.reshape(raw, (G, J, 2, 16, 16)), (1, 2, 4, 0, 3)),
                    (GU_K, GU_N),
                )
            else:
                b = tl.load(wq_ptr + offs_n[None, :] * K + offs_k[:, None])
            b_s = tl.load(ws_ptr + offs_n[:, None] * (K // 32) + offs_g[None, :])
            acc = tl.dot_scaled(a, a_s, "e4m3", b, b_s, "e4m3", acc)
        rows = gu_m_tiles * GU_M
        tl.store(
            gu_part_ptr + (pid_k * rows + offs_m)[:, None] * N2 + offs_n[None, :], acc
        )


@triton.jit
def _gate_sort_kernel(
    part_ptr,
    bias_ptr,
    n_local_ptr,
    counter_ptr,
    topk_w_ptr,
    topk_ids_ptr,
    sorted_ids_ptr,
    sorted_ids_ms_ptr,
    sorted_w_ptr,
    sorted_eids_ptr,
    num_valid_ptr,
    m_indices_ptr,
    reverse_ptr,
    moe_buf_ptr,
    moe_buf_numel,
    MP,
    split_stride,
    scale,
    E: tl.constexpr,
    TOPK: tl.constexpr,
    SORT_M: tl.constexpr,
    SPLIT: tl.constexpr,
    ROWS_POW2: tl.constexpr,
    GATE_ROWS: tl.constexpr,
    HAS_NLOCAL: tl.constexpr,
    BUF_BLOCK: tl.constexpr,
    gu_part_ptr,
    q_ptr,
    q_scale_ptr,
    gu_split_stride,
    alpha,
    beta,
    limit,
    HAS_SW: tl.constexpr,
    GU_SPLIT: tl.constexpr,
    N_INTER: tl.constexpr,
    SW_BLOCK: tl.constexpr,
    HAS_LIMIT: tl.constexpr,
):
    """Programs [0, cdiv(MP, GATE_ROWS)) gate GATE_ROWS rows each; the last of
    them to finish sorts every row. With HAS_SW, one program per (row,
    SW_BLOCK columns) then sums the gate_up partials in split order, rounds
    them to bf16 (the GEMM's output dtype) and runs the SwiGLU + MXFP8 quant.
    The programs after them zero the MoE output."""
    pid = tl.program_id(0)
    num_gate = tl.cdiv(MP, GATE_ROWS)
    if pid >= num_gate:
        num_sw = 0
        if HAS_SW:
            tiles: tl.constexpr = (N_INTER + SW_BLOCK - 1) // SW_BLOCK
            num_sw = MP * tiles
            if pid < num_gate + num_sw:
                t = pid - num_gate
                row = t // tiles
                tile = t % tiles
                cols = tile * SW_BLOCK + tl.arange(0, SW_BLOCK)
                mask = cols < N_INTER
                gate = tl.zeros((SW_BLOCK,), tl.float32)
                up = tl.zeros((SW_BLOCK,), tl.float32)
                for sp in tl.static_range(GU_SPLIT):
                    src = gu_part_ptr + sp * gu_split_stride + row * (2 * N_INTER)
                    gate += tl.load(src + cols, mask=mask, other=0.0)
                    up += tl.load(src + N_INTER + cols, mask=mask, other=0.0)
                swiglu_mxfp8_store(
                    gate.to(tl.bfloat16).to(tl.float32),
                    up.to(tl.bfloat16).to(tl.float32),
                    q_ptr,
                    q_scale_ptr,
                    N_INTER,
                    alpha,
                    beta,
                    limit,
                    HAS_LIMIT,
                    SW_BLOCK,
                    row,
                    tile,
                    tl.bfloat16,
                )
                return
        # the MoE output an accumulating GEMM2 adds into; it runs later on this stream
        offs = (pid - num_gate - num_sw) * BUF_BLOCK + tl.arange(0, BUF_BLOCK)
        tl.store(
            moe_buf_ptr + offs,
            tl.zeros((BUF_BLOCK,), moe_buf_ptr.dtype.element_ty),
            mask=offs < moe_buf_numel,
        )
        return

    # real rows: the step's token count when the caller passes one, else every row
    M = MP
    if HAS_NLOCAL:
        M = tl.minimum(tl.maximum(tl.load(n_local_ptr), 0), MP)
    offs_m = pid * GATE_ROWS + tl.arange(0, GATE_ROWS)
    offs_n = tl.arange(0, E)
    offs_k = tl.arange(0, TOPK)
    live = offs_m < M

    # ---- logits[m][n] = ((0 + p0) + p1) + ..., padding rows never read
    lg = tl.zeros((GATE_ROWS, E), tl.float32)
    for s in tl.static_range(SPLIT):
        lg += tl.load(
            part_ptr + s * split_stride + offs_m[:, None] * E + offs_n[None, :],
            mask=live[:, None],
            other=0.0,
        )

    # ---- gate, as m31_router.hip: act = 1 / (exp2(-x log2 e) + 1), rank by
    # act + bias (NaN -> -1e30), top-4 with the lowest expert id among equals
    act = 1.0 / (tl.exp2(lg * -1.44269502) + 1.0)
    biased = act + tl.load(bias_ptr + offs_n)[None, :]
    cur = tl.where(biased == biased, biased, -1e30)  # noqa: PLR0124 (NaN test)
    sel_v = tl.zeros((GATE_ROWS, TOPK), tl.float32)
    sel_i = tl.zeros((GATE_ROWS, TOPK), tl.int32)
    for k in tl.static_range(TOPK):
        best = tl.max(cur, axis=1)
        win = tl.min(tl.where(cur == best[:, None], offs_n[None, :], E + 1), axis=1)
        is_win = offs_n[None, :] == win[:, None]
        win_act = tl.sum(tl.where(is_win, act, 0.0), axis=1)
        sel_v = tl.where(offs_k[None, :] == k, win_act[:, None], sel_v)
        sel_i = tl.where(offs_k[None, :] == k, win[:, None], sel_i)
        cur = tl.where(is_win, -float("inf"), cur)
    # renormalize in the HIP kernel's order: (s0 + s2) + (s1 + s3)
    v0 = tl.sum(tl.where(offs_k[None, :] == 0, sel_v, 0.0), axis=1)
    v1 = tl.sum(tl.where(offs_k[None, :] == 1, sel_v, 0.0), axis=1)
    v2 = tl.sum(tl.where(offs_k[None, :] == 2, sel_v, 0.0), axis=1)
    v3 = tl.sum(tl.where(offs_k[None, :] == 3, sel_v, 0.0), axis=1)
    total = (v0 + v2) + (v1 + v3)
    norm = tl.where(total > 0.0, total, 1.0)
    wk = (sel_v / norm[:, None]) * scale
    # padding rows still get finite ids / weights: callers may read every row
    wk = tl.where(live[:, None], wk, 0.0)
    out_ids = tl.where(live[:, None], sel_i, offs_k[None, :])
    rows_mask = (offs_m < MP)[:, None]
    tl.store(topk_w_ptr + offs_m[:, None] * TOPK + offs_k[None, :], wk, mask=rows_mask)
    tl.store(
        topk_ids_ptr + offs_m[:, None] * TOPK + offs_k[None, :], out_ids, mask=rows_mask
    )

    # the last gate program to finish sorts: its acquire sees every program's
    # routes; it resets the ticket for the next launch. The barrier completes
    # every wave's stores ahead of the release.
    tl.debug_barrier()
    ticket = tl.atomic_add(counter_ptr, 1, sem="acq_rel", scope="gpu")
    if ticket != num_gate - 1:
        return
    tl.atomic_xchg(counter_ptr, 0, sem="relaxed", scope="gpu")

    # ---- sort (aiter's block_m SORT_M layout): experts ascending, a pair's
    # (token, slot) order kept within an expert; routes of padding rows go to
    # the sentinel bin E_POW2 - 1 and sort last.
    P_POW2: tl.constexpr = ROWS_POW2 * TOPK
    E_POW2: tl.constexpr = 2 * E
    P = M * TOPK
    offs_p = tl.arange(0, P_POW2)
    offs_e = tl.arange(0, E_POW2)
    e_flat = tl.load(topk_ids_ptr + offs_p, mask=offs_p < P, other=E_POW2 - 1)
    cnt = tl.where(offs_e < E, tl.histogram(e_flat, E_POW2), 0)
    blocks = (cnt + SORT_M - 1) // SORT_M
    first_row = (tl.cumsum(blocks, 0) - blocks) * SORT_M
    # a pair's place in the unpadded order -> its padded row: add the tail
    # padding of the smaller experts
    shift = first_row - (tl.cumsum(cnt, 0) - cnt)
    num_valid = tl.sum(blocks, 0) * SORT_M

    key = tl.sort(e_flat * P_POW2 + offs_p)
    es = key // P_POW2
    ps = key % P_POW2
    valid = offs_p < P
    dest = offs_p + tl.gather(shift, tl.where(valid, es, 0), 0)
    tok = ps // TOPK
    ids = ((ps % TOPK) << 24) | tok
    tl.store(sorted_ids_ptr + dest, ids, mask=valid)
    tl.store(sorted_ids_ms_ptr + dest, ids, mask=valid)
    tl.store(sorted_w_ptr + dest, tl.load(topk_w_ptr + ps, mask=valid), mask=valid)
    tl.store(m_indices_ptr + dest, tok, mask=valid)
    # every block holds a real pair; its pairs all write the same expert id
    tl.store(sorted_eids_ptr + dest // SORT_M, es, mask=valid)
    tl.store(reverse_ptr + ps, dest, mask=valid)
    tl.store(reverse_ptr + offs_p, -1, mask=(offs_p >= P) & (offs_p < MP * TOPK))

    # padding rows of each expert's last block: rows the scatter above leaves,
    # so the two write sets are disjoint and need no ordering
    offs_j = tl.arange(0, SORT_M)
    pad = (first_row + cnt)[:, None] + offs_j[None, :]
    pad_mask = offs_j[None, :] < (blocks * SORT_M - cnt)[:, None]
    tl.store(sorted_ids_ptr + pad, tl.zeros_like(pad) + MP, mask=pad_mask)
    tl.store(
        sorted_ids_ms_ptr + pad, tl.zeros_like(pad) + ((TOPK << 24) | MP), mask=pad_mask
    )
    tl.store(sorted_w_ptr + pad, tl.zeros(pad.shape, tl.float32), mask=pad_mask)
    tl.store(m_indices_ptr + pad, tl.zeros_like(pad) + MP, mask=pad_mask)
    tl.store(
        num_valid_ptr + tl.arange(0, 2),
        tl.where(tl.arange(0, 2) == 0, num_valid, M),
    )


_workspaces: dict[torch.device, tuple[torch.Tensor, torch.Tensor]] = {}


def _workspace(device: torch.device) -> tuple[torch.Tensor, torch.Tensor] | None:
    """The split partials (always fully overwritten) and the self-resetting
    ticket counter, one pair per device shared by every layer: the launches of
    one stream run in order."""
    ws = _workspaces.get(device)
    if ws is None:
        if torch.cuda.is_current_stream_capturing():
            return None
        ws = _workspaces[device] = (
            torch.empty(
                _SPLIT * MAX_ROWS * NUM_EXPERTS, dtype=torch.float32, device=device
            ),
            torch.zeros(1, dtype=torch.int32, device=device),
        )
    return ws


_gu_workspaces: dict[tuple[torch.device, int], torch.Tensor] = {}


def gateup_tiling(m: int) -> tuple[int, int, int]:
    """(row tile, row tiles, split-K partials) of the gate_up GEMM at ``m``
    rows: one tile of up to 64 rows reads the weight once, two past 64; the
    splits keep about 8 x 48 programs."""
    tile = min(max(16, triton.next_power_of_2(m)), _GU_MAX_M)
    tiles = triton.cdiv(m, tile)
    return tile, tiles, max(1, _GU_MAX_SPLIT // tiles)


def _gu_workspace(device: torch.device, n2: int) -> torch.Tensor | None:
    ws = _gu_workspaces.get((device, n2))
    if ws is None:
        if torch.cuda.is_current_stream_capturing():
            return None
        rows = max(
            tile * tiles * split
            for tile, tiles, split in map(gateup_tiling, range(1, MAX_ROWS + 1))
        )
        ws = _gu_workspaces[(device, n2)] = torch.empty(
            rows * n2, dtype=torch.float32, device=device
        )
    return ws


def router_gateup_supported(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    bias: torch.Tensor | None,
    xq: torch.Tensor,
    xs: torch.Tensor,
    wq: torch.Tensor,
    ws: torch.Tensor,
) -> bool:
    """Whether ``router_gateup_partials`` serves these operands: the router's
    as for ``router_swiglu_supported``, and the shared expert's gate_up as
    E4M3 rows ``xq`` [M, 6144] with E8M0 group32 scales ``xs`` [M, 192] and an
    E4M3 weight ``wq`` [N2, 6144], row-major or in aiter's (16, 16)
    preshuffle (``is_shuffled``), with row-major E8M0 1x32 scales ``ws``."""
    m, n2 = x.shape[0], wq.shape[0]
    e8m0 = (torch.float8_e8m0fnu, torch.uint8)
    return (
        router_swiglu_supported(
            x, gate_weight, bias, torch.empty((m, n2), device="meta")
        )
        and xq.dtype == FP8_DTYPE
        and xq.shape == (m, HIDDEN)
        and xq.is_contiguous()
        and xs.dtype in e8m0
        and xs.shape == (m, HIDDEN // 32)
        and xs.is_contiguous()
        and wq.dtype == FP8_DTYPE
        and wq.shape == (n2, HIDDEN)
        and wq.is_contiguous()
        # row-major, or aiter's default (16, 16) preshuffle, unpadded
        and getattr(wq, "aiter_padded_k", None) is None
        and ws.dtype in e8m0
        and ws.shape == (n2, HIDDEN // 32)
        and ws.is_contiguous()
        and n2 % _GU_N == 0
        and n2 % 16 == 0
    )


def router_gateup_partials(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    xq: torch.Tensor,
    xs: torch.Tensor,
    wq: torch.Tensor,
    ws: torch.Tensor,
) -> bool:
    """Launch 1 when the shared expert's gate_up GEMM also joins it: the router
    GEMV partials and the gate_up partials into the device workspaces, for the
    next ``gate_sort(..., swiglu=...)`` on this stream, which finishes the
    gate_up and runs the SwiGLU. False while a workspace is missing (a first
    call under graph capture). Operands as ``router_gateup_supported``."""
    rws = _workspace(x.device)
    gws = _gu_workspace(x.device, wq.shape[0])
    if rws is None or gws is None:
        return False
    m = x.shape[0]
    m_tiles = triton.cdiv(m, _GEMV_M)
    gu_m, gu_m_tiles, gu_split = gateup_tiling(m)
    num_gemv = (NUM_EXPERTS // _GEMV_N) * m_tiles * _SPLIT
    num_gu = (wq.shape[0] // _GU_N) * gu_m_tiles * gu_split
    _router_gateup_partials_kernel[(num_gemv + num_gu,)](
        x,
        gate_weight,
        rws[0],
        xq,
        xs.view(torch.uint8),
        wq,
        ws.view(torch.uint8),
        gws,
        m,
        x.stride(0),
        m_tiles,
        gu_m_tiles,
        K=HIDDEN,
        N=NUM_EXPERTS,
        GEMV_M=_GEMV_M,
        GEMV_N=_GEMV_N,
        GEMV_K=_GEMV_K,
        SPLIT=_SPLIT,
        N2=wq.shape[0],
        GU_M=gu_m,
        GU_N=_GU_N,
        GU_K=_GU_K,
        GU_SPLIT=gu_split,
        W_SHUFFLED=bool(getattr(wq, "is_shuffled", False)),
        num_warps=_NUM_WARPS,
    )
    return True


def router_swiglu_supported(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    bias: torch.Tensor | None,
    gate_up: torch.Tensor,
) -> bool:
    """Whether ``m31_route_swiglu`` serves these operands."""
    m = x.shape[0]
    return (
        x.dim() == 2
        and MIN_ROWS <= m <= MAX_ROWS
        and x.shape[1] == HIDDEN
        and x.dtype == torch.bfloat16
        and x.stride(1) == 1
        and gate_weight.dtype == torch.bfloat16
        and gate_weight.shape == (NUM_EXPERTS, HIDDEN)
        and gate_weight.is_contiguous()
        and bias is not None
        and bias.dtype == torch.float32
        and bias.shape == (NUM_EXPERTS,)
        and bias.is_contiguous()
        and gate_up.dim() == 2
        and gate_up.shape[0] == m
        and gate_up.shape[1] % 64 == 0
        and gate_up.stride(1) == 1
        and register_presorted_moe is not None
        and torch.cuda.get_device_properties(x.device).gcnArchName.split(":")[0]
        == "gfx950"
    )


def router_partials_swiglu(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    gate_up: torch.Tensor,
    alpha: float,
    beta: float,
    limit: float | None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Launch 1: the router GEMV's split-K partials into the device workspace,
    for the next ``gate_sort`` on this stream, and ``swiglu_oai_split_mxfp8(
    gate_up, alpha, beta, limit)``, returned as ``(values, scales)``. None while
    the workspace is missing (a first call under graph capture)."""
    ws = _workspace(x.device)
    if ws is None:
        return None
    part, _ = ws
    m = x.shape[0]
    dev = x.device
    n_inter = gate_up.shape[1] // 2
    q = torch.empty((m, n_inter), dtype=FP8_DTYPE, device=dev)
    q_scale = torch.empty((m, n_inter // 32), dtype=torch.float8_e8m0fnu, device=dev)
    sw_block = swiglu_mxfp8_block(n_inter)
    m_tiles = triton.cdiv(m, _GEMV_M)
    num_gemv = (NUM_EXPERTS // _GEMV_N) * m_tiles * _SPLIT
    _router_partials_swiglu_kernel[(num_gemv + m * triton.cdiv(n_inter, sw_block),)](
        x,
        gate_weight,
        part,
        gate_up,
        q,
        q_scale.view(torch.uint8),
        m,
        x.stride(0),
        gate_up.stride(0),
        m_tiles,
        float(alpha),
        float(beta),
        0.0 if limit is None else float(limit),
        K=HIDDEN,
        N=NUM_EXPERTS,
        GEMV_M=_GEMV_M,
        GEMV_N=_GEMV_N,
        GEMV_K=_GEMV_K,
        SPLIT=_SPLIT,
        N_INTER=n_inter,
        HAS_LIMIT=limit is not None,
        SW_BLOCK=sw_block,
        num_warps=_NUM_WARPS,
    )
    return q, q_scale


def gate_sort(
    m: int,
    bias: torch.Tensor,
    routed_scaling_factor: float = 1.0,
    n_local: torch.Tensor | None = None,
    moe_buf: torch.Tensor | None = None,
    zero_moe_buf: bool = True,
    swiglu: tuple[int, float, float, float | None] | None = None,
):
    """Launch 2, after launch 1 (``router_partials_swiglu`` or, with
    ``swiglu``, ``router_gateup_partials``) of the same ``m`` rows on this
    stream: ``(topk_weights [m, 4] fp32, topk_ids [m, 4] int32, presort)``,
    plus ``(values, scales)`` of the SwiGLU with ``swiglu = (n_inter, alpha,
    beta, limit)``.

    ``presort`` is ``router.m31_route``'s registration (both sort layouts and
    the MoE output GEMM2 accumulates into). Weights are multiplied by
    ``routed_scaling_factor``. ``n_local`` (int32 device scalar): rows from it on
    are padding -- no routes, zero weights. ``moe_buf``: a caller's [m, 6144]
    bf16 output, already holding what GEMM2 should add to (e.g. the shared
    expert's output); without it the launch zeroes a fresh one, or, with
    ``zero_moe_buf=False``, leaves the presort without one for the caller to
    add with ``set_presort_output`` before the MoE runs.
    """
    dev = bias.device
    part, counter = _workspaces[dev]
    topk_w = torch.empty(m, TOPK, dtype=torch.float32, device=dev)
    topk_ids = torch.empty(m, TOPK, dtype=torch.int32, device=dev)
    # aiter's BM=16 sort capacity, a whole number of blocks (as router.py)
    max_sorted = -(-(m * TOPK + NUM_EXPERTS * BLOCK_M - TOPK) // BLOCK_M) * BLOCK_M
    sorted_ids = torch.empty(max_sorted, dtype=torch.int32, device=dev)
    sorted_w = torch.empty(max_sorted, dtype=torch.float32, device=dev)
    sorted_eids = torch.empty(max_sorted // BLOCK_M, dtype=torch.int32, device=dev)
    num_valid = torch.empty(2, dtype=torch.int32, device=dev)
    m_indices = torch.empty(max_sorted, dtype=torch.int32, device=dev)
    reverse_sorted = torch.empty(m * TOPK, dtype=torch.int32, device=dev)
    sorted_ids_ms = torch.empty(max_sorted, dtype=torch.int32, device=dev)
    num_zero = 0
    if moe_buf is None and zero_moe_buf:
        moe_buf = torch.empty(m, HIDDEN, dtype=torch.bfloat16, device=dev)
        num_zero = triton.cdiv(moe_buf.numel(), _BUF_BLOCK)
    num_sw, out = 0, None
    if swiglu is not None:
        n_inter, alpha, beta, limit = swiglu
        gu_m, gu_m_tiles, gu_split = gateup_tiling(m)
        sw_block = swiglu_mxfp8_block(n_inter)
        q = torch.empty((m, n_inter), dtype=FP8_DTYPE, device=dev)
        q_scale = torch.empty(
            (m, n_inter // 32), dtype=torch.float8_e8m0fnu, device=dev
        )
        out = (q, q_scale)
        num_sw = m * triton.cdiv(n_inter, sw_block)
        sw_args = {
            "gu_part_ptr": _gu_workspaces[(dev, 2 * n_inter)],
            "q_ptr": q,
            "q_scale_ptr": q_scale.view(torch.uint8),
            "gu_split_stride": gu_m * gu_m_tiles * 2 * n_inter,
            "alpha": float(alpha),
            "beta": float(beta),
            "limit": 0.0 if limit is None else float(limit),
            "HAS_SW": True,
            "GU_SPLIT": gu_split,
            "N_INTER": n_inter,
            "SW_BLOCK": sw_block,
            "HAS_LIMIT": limit is not None,
        }
    else:
        sw_args = {
            "gu_part_ptr": part,
            "q_ptr": part,
            "q_scale_ptr": part,
            "gu_split_stride": 0,
            "alpha": 0.0,
            "beta": 0.0,
            "limit": 0.0,
            "HAS_SW": False,
            "GU_SPLIT": 1,
            "N_INTER": 32,
            "SW_BLOCK": 32,
            "HAS_LIMIT": False,
        }
    m_tiles = triton.cdiv(m, _GEMV_M)
    num_gate = triton.cdiv(m, _GATE_ROWS)
    _gate_sort_kernel[(num_gate + num_sw + num_zero,)](
        part,
        bias,
        n_local if n_local is not None else num_valid,
        counter,
        topk_w,
        topk_ids,
        sorted_ids,
        sorted_ids_ms,
        sorted_w,
        sorted_eids,
        num_valid,
        m_indices,
        reverse_sorted,
        moe_buf if moe_buf is not None else part,
        moe_buf.numel() if moe_buf is not None else 0,
        m,
        m_tiles * _GEMV_M * NUM_EXPERTS,
        float(routed_scaling_factor),
        E=NUM_EXPERTS,
        TOPK=TOPK,
        SORT_M=BLOCK_M,
        SPLIT=_SPLIT,
        ROWS_POW2=triton.next_power_of_2(m),
        GATE_ROWS=_GATE_ROWS,
        HAS_NLOCAL=n_local is not None,
        BUF_BLOCK=_BUF_BLOCK,
        num_warps=4,
        **sw_args,
    )
    presort = {
        "sorted_token_ids": sorted_ids,
        "sorted_weights": sorted_w,
        "sorted_expert_ids": sorted_eids,
        "num_valid_ids": num_valid,
        "m_indices": m_indices,
        "reverse_sorted": reverse_sorted,
        "sorted_token_ids_moe_sorting": sorted_ids_ms,
        "topk_ids_ptr": topk_ids.data_ptr(),
        "topk_weights_ptr": topk_w.data_ptr(),
        "M": m,
        "block_m": BLOCK_M,
        "num_experts": NUM_EXPERTS,
        "topk": TOPK,
    }
    if moe_buf is not None:
        set_presort_output(presort, moe_buf)
    if out is not None:
        return topk_w, topk_ids, presort, out
    return topk_w, topk_ids, presort


def set_presort_output(presort: dict, moe_buf: torch.Tensor) -> None:
    """GEMM2's destination, ready for it to accumulate into (aiter's word for
    that is "zeroed"): zeros, or what GEMM2 should add to."""
    presort["moe_buf"] = moe_buf
    presort["moe_buf_zeroed"] = True
    presort["moe_buf_ptr"] = moe_buf.data_ptr()


def m31_route_swiglu(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    bias: torch.Tensor,
    gate_up: torch.Tensor,
    alpha: float,
    beta: float,
    limit: float | None,
    routed_scaling_factor: float = 1.0,
    n_local: torch.Tensor | None = None,
):
    """``router_partials_swiglu`` then ``gate_sort`` with a zeroed MoE output:
    ``(topk_weights, topk_ids, presort, (q, scale))``, or None while the
    workspace is missing (a first call under graph capture)."""
    swiglu = router_partials_swiglu(x, gate_weight, gate_up, alpha, beta, limit)
    if swiglu is None:
        return None
    routes = gate_sort(x.shape[0], bias, routed_scaling_factor, n_local)
    return (*routes, swiglu)
