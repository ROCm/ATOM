# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The W4A4 GEMM1 and GEMM2 in one launch (128- or 256-row tiles): workgroups [0, N_CTAS) run GEMM1's persistent tile
loop (one per CU, per-XCD claims), the rest GEMM2's (m-tile, n-chunk) work items, each computing what it computes in
the BM256 host's two launches (bit-equal slots). GEMM2 tiles start on the CUs GEMM1 leaves while its last tiles run: no
partial last GEMM1 round, no GEMM2 ramp, no launch gap between the W13 and the W2 stream.

A GEMM1 wave stores its rows write-through and counts the tile in its m-block's ready counter at the next tile's tail
wait, which retires the workgroup's stores (24 units per block, 48 with split-column tasks); a GEMM2 work item waits
for its block, and the block's last consumer resets the counters (``a16w4_utils.wait_ready``). Counters only order
work: every output element has one writer. With six fixed chunks GEMM2 work item w runs on XCD w % 8 and follows
GEMM1's claim order, earliest first. Deadlock-free: every GEMM1 workgroup has a lower id than every GEMM2 one and never
waits, and each XCD dispatches its workgroups in id order.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr

from amd_kernels.warm_up import launcher


@launcher
def compile_moe_gemm12_mxfp4_h6144_i768(
    *, E: int, topk: int, n_split: int = 6, bm: int = 128, n_tokens: int = 2049,
    adaptive: bool = False, epilogue_groups: bool = False,
):
    """The fused launch on ``bm``-row tiles: GEMM1 skips the upper half's math on tiles with at most ``bm`` / 2 valid
    rows (``gemm1_a4w4_bm128_half_rows`` / ``gemm1_a4w4_bm256_half_rows``); at 256 GEMM2 skips it too.

    Args:
        n_tokens: picks GEMM2's row-probe form (``bm`` 128).
        adaptive: GEMM2's adaptive chunks (``bm`` 128); the device-selected split also gives the number of
            counter-reset consumers.
        epilogue_groups: GEMM2 also skips each wave's wholly padded 16-row epilogue groups (``bm`` 256,
            ``gemm2_a4w4_bm256_epilogue_groups``).
    """
    if bm == 128:
        from .gemm1_a4w4_bm128_half_rows import compile_moe_gemm1_mxfp4_h6144_i768 as _compile_g1
        from .gemm2_a4w4_bm128 import compile_moe_gemm2_mxfp4_h6144_i768 as _compile_g2
        k1 = _compile_g1(E=E, BLOCK_M=128, signal=True)
        g2_kw = dict(n_tokens=n_tokens, n_split=n_split)
        if adaptive:
            from .gemm2_a4w4_bm128_adaptive import compile_moe_gemm2_mxfp4_h6144_i768 as _compile_g2
            g2_kw = dict(n_tokens=n_tokens, pad_skip=True)
        elif n_tokens <= 1536:
            # the fixed-chunk host takes this GEMM2 at 705-1536 rows
            from .gemm2_a4w4_bm128_epilogue_groups import compile_moe_gemm2_mxfp4_h6144_i768 as _compile_g2
        stem = "kernel_gemm2_mxfp4_h6144_i768_bm128"
    else:
        assert bm == 256
        from .gemm1_a4w4_bm256_half_rows import compile_moe_gemm1_mxfp4_h6144_i768 as _compile_g1
        from .gemm2_a4w4_bm256_half_rows import compile_moe_gemm2_mxfp4_h6144_i768 as _compile_g2
        if epilogue_groups:
            from .gemm2_a4w4_bm256_epilogue_groups import compile_moe_gemm2_mxfp4_h6144_i768 as _compile_g2
        k1 = _compile_g1(E=E, BLOCK_M=256, signal=True)
        g2_kw = dict(sort_block_m=256, n_split=n_split)
        stem = "kernel_gemm2_mxfp4_h6144_i768"
    n_ready = getattr(k1, "ready_per_block", k1.n_tiles_n * 4)
    k2 = _compile_g2(E=E, topk=topk, wait_for=n_ready, **g2_kw)
    N_CTAS = k1.n_ctas
    t1, s1 = k1.lds_bytes
    t2, s2, stg, n_sids = k2.lds_bytes
    name = f"kernel_gemm12_mxfp4_h6144_i768_bm{bm}" + k2.kernel_name[len(stem):] + f"_ns{n_split}_g1_half_rows"

    if bm == 256:
        name = (f"kernel_gemm12_mxfp4_h6144_i768_bm256_half_rows"
                + ("_epilogue_groups" if epilogue_groups else "") + f"_ns{n_split}")

    # GEMM1 = tiles + scale slots + claim words, GEMM2 = tiles + scale slots + staging + ids. If both side by side
    # exceed the CU's 160 KB (BM256: 161 KB), GEMM1's scale slots share GEMM2's staging bytes, not GEMM2's scale slots.
    flat = max(t1, t2) + max(s1, s2) + stg + 4 * n_sids <= 160 * 1024
    a_bytes = max(s1, s2) if flat else max(s1, stg)
    b_bytes = stg if flat else s2

    @fx.struct
    class SharedStorage:
        all_lds: fx.Array[fx.Int8, max(t1, t2), 16]
        scale_lds: fx.Array[fx.Int8, a_bytes, 16]  # scales (GEMM1; flat: GEMM2 too)
        stage_lds: fx.Array[fx.Int8, b_bytes, 16]  # flat: GEMM2's staging, else its scales
        sids: fx.Array[fx.Int32, max(n_sids, 4), 16]  # GEMM2's ids / GEMM1's claim words

    @flyc.kernel(name=name)
    def kernel_gemm12(
        A: fx.Tensor,
        W13: fx.Tensor,
        HQ: fx.Tensor,
        A_scale: fx.Tensor,
        W13_scale: fx.Tensor,
        HS: fx.Tensor,
        sorted_ids: fx.Tensor,
        sorted_expert_ids: fx.Tensor,
        n_tokens: fx.Int32,
        num_m_blocks: fx.Int32,
        a_scale_bytes: fx.Int32,
        tile_map_t: fx.Tensor,
        grid1: fx.Int32,
        W2: fx.Tensor,
        OUT: fx.Tensor,
        W2_scale: fx.Tensor,
        sorted_weights: fx.Tensor,
        num_valid_ids: fx.Tensor,
        ready: fx.Int64,
    ):
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        bx = fx.block_idx.x
        if bx < N_CTAS:
            k1.tile(
                lds.all_lds.ptr, lds.scale_lds.ptr, lds.sids.ptr, bx,
                A, W13, HQ, A_scale, W13_scale, HS, sorted_ids, sorted_expert_ids,
                n_tokens, num_m_blocks, a_scale_bytes, tile_map_t, grid1, ready,
            )  # fmt: skip
        else:
            if const_expr(flat):
                k2.tile(
                    lds.all_lds.ptr, lds.scale_lds.ptr, lds.stage_lds.ptr, lds.sids.ptr,
                    bx - N_CTAS, HQ, W2, OUT, HS, W2_scale, sorted_ids, sorted_expert_ids,
                    sorted_weights, num_valid_ids, n_tokens, num_m_blocks, ready,
                )  # fmt: skip
            else:
                k2.tile(
                    lds.all_lds.ptr, lds.stage_lds.ptr, lds.scale_lds.ptr, lds.sids.ptr,
                    bx - N_CTAS, HQ, W2, OUT, HS, W2_scale, sorted_ids, sorted_expert_ids,
                    sorted_weights, num_valid_ids, n_tokens, num_m_blocks, ready,
                )  # fmt: skip

    @flyc.jit
    def launch_gemm12_mxfp4_h6144_i768(
        A: fx.Tensor,
        W13: fx.Tensor,
        HQ: fx.Tensor,
        A_scale: fx.Tensor,
        W13_scale: fx.Tensor,
        HS: fx.Tensor,
        sorted_ids: fx.Tensor,
        sorted_expert_ids: fx.Tensor,
        n_tokens: fx.Int32,
        num_m_blocks: fx.Int32,
        a_scale_bytes: fx.Int32,
        tile_map_t: fx.Tensor,
        grid1: fx.Int32,
        W2: fx.Tensor,
        OUT: fx.Tensor,
        W2_scale: fx.Tensor,
        sorted_weights: fx.Tensor,
        num_valid_ids: fx.Tensor,
        ready: fx.Int64,
        grid2: fx.Int32,
        stream: fx.Stream,
    ):
        kernel_gemm12(
            A, W13, HQ, A_scale, W13_scale, HS, sorted_ids, sorted_expert_ids,
            n_tokens, num_m_blocks, a_scale_bytes, tile_map_t, grid1,
            W2, OUT, W2_scale, sorted_weights, num_valid_ids, ready,
            value_attrs={
                "rocdl.waves_per_eu": 1,
                "rocdl.flat_work_group_size": "256,256",
            },
        ).launch(grid=(grid2 + N_CTAS, 1, 1), block=(256, 1, 1), stream=stream)  # fmt: skip

    launch_gemm12_mxfp4_h6144_i768.kernel_name = name
    return launch_gemm12_mxfp4_h6144_i768
