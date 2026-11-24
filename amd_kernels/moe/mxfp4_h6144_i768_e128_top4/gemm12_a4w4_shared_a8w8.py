# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Routed MXFP4 and shared MXFP8 GEMM1 / GEMM2 in one persistent launch (BF16-entry rows 449-3583).

Both formats run their own tile bodies and quantization; producers precede consumers and never wait on them, the
workgroup classes alias whole LDS arenas, and no output uses an atomic add.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr

from amd_kernels.warm_up import launcher


@launcher
def compile_moe_gemm12_mxfp4_shared_mxfp8_h6144_i768(*,n_tokens,adaptive,shared_nt):
    """Four workgroup classes in launch order: shared GEMM1, routed GEMM1, shared GEMM2, routed GEMM2.

    The producers finish without waiting; a shared GEMM1 workgroup retires every wave's stores before its leader
    bumps the ready counter, and the last shared consumer restores the two ready words.
    """
    E,topk,bm,n_split=128,4,128,6
    half_rows=True
    from .gemm1_a4w4_bm128_half_rows import compile_moe_gemm1_mxfp4_h6144_i768 as _compile_g1
    if adaptive:
        from .gemm2_a4w4_bm128_adaptive import compile_moe_gemm2_mxfp4_h6144_i768 as _compile_g2
        g2_kw=dict(n_tokens=n_tokens,pad_skip=True)
    elif n_tokens<=1536:
        from .gemm2_a4w4_bm128_epilogue_groups import compile_moe_gemm2_mxfp4_h6144_i768 as _compile_g2
        g2_kw=dict(n_tokens=n_tokens,n_split=6)
    else:
        from .gemm2_a4w4_bm128 import compile_moe_gemm2_mxfp4_h6144_i768 as _compile_g2
        g2_kw=dict(n_tokens=n_tokens,n_split=6)
    k1=_compile_g1(E=128,BLOCK_M=128,signal=True)
    stem="kernel_gemm2_mxfp4_h6144_i768_bm128"
    n_ready = getattr(k1, "ready_per_block", k1.n_tiles_n * 4)
    k2 = _compile_g2(E=E, topk=topk, wait_for=n_ready, **g2_kw)
    N_CTAS = k1.n_ctas
    t1, s1 = k1.lds_bytes
    t2, s2, stg, n_sids = k2.lds_bytes
    from .shared_gemm1_a8w8_write_through import compile_shared_gemm1_a8w8_write_through
    from .shared_dense import _gemm2
    from .a16w4_utils import global_atomic_add_agent,wait_ready
    # 128-row shared tiles through 1536 rows, 256-row tiles above
    shared_tiles=(2,4) if n_tokens<=1536 else (4,4)
    sh1=compile_shared_gemm1_a8w8_write_through(tiles=shared_tiles)
    sh2=_gemm2((4,4),nt=shared_nt)
    SHARED_SUB_SPLIT=sh1.sub_split
    name=(f"kernel_gemm12_a4w4_shared_a8w8_h6144_i768"+k2.kernel_name[len(stem):]
          +f"_st{shared_tiles[0]}x{shared_tiles[1]}"+("_nt" if shared_nt else ""))

    # GEMM1 = tiles + scale slots + claim words, GEMM2 = tiles + scale slots + staging + ids. If both side by side
    # exceed the CU's 160 KB (BM256: 161 KB), GEMM1's scale slots share GEMM2's staging bytes instead of its slots.
    flat = max(t1, t2) + max(s1, s2) + stg + 4 * n_sids <= 160 * 1024
    a_bytes = max(s1, s2) if flat else max(s1, stg)
    b_bytes = stg if flat else s2

    raw_bytes=max(max(t1,t2)+a_bytes+b_bytes+4*max(n_sids,4),sum(sh2.lds_bytes))
    assert raw_bytes<=160*1024,(raw_bytes,bm,t1,s1,t2,s2,stg,n_sids)
    @fx.struct
    class SharedStorage:
        all_lds: fx.Array[fx.Int8, raw_bytes, 16]

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
        A8: fx.Tensor, AS8: fx.Tensor, W138: fx.Tensor, WS138: fx.Tensor,
        Q8: fx.Tensor, S8: fx.Tensor, W28: fx.Tensor, WS28: fx.Tensor, OUT8: fx.Tensor,
        shared_ready: fx.Int64, nsh1: fx.Int32, nsh2: fx.Int32,
    ):
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        bx = fx.block_idx.x
        p1=((n_tokens+255)//256)*6
        if const_expr(SHARED_SUB_SPLIT>1):
            p1=((p1+7)//8*8)*SHARED_SUB_SPLIT
        needed=p1
        if bx < nsh1:
            if bx<p1:
                sh1.tile(lds.all_lds.ptr,lds.all_lds.ptr+131072,lds.all_lds.ptr+139264,fx.Int32(bx),
                         A8,W138,Q8,S8,AS8,WS138,n_tokens,fx.Int32(1536),fx.Int32(0),p1)
                fx.rocdl.s_waitcnt(vmcnt=0)
                fx.gpu.barrier()
                if fx.thread_idx.x==0:
                    global_atomic_add_agent(shared_ready,1)
        elif bx < nsh1+N_CTAS:
            bx=bx-nsh1
            k1.tile(
                lds.all_lds.ptr, (lds.all_lds.ptr+max(t1,t2)), fx.recast_iter(fx.Int32,(lds.all_lds.ptr+max(t1,t2))+a_bytes+b_bytes), bx,
                A, W13, HQ, A_scale, W13_scale, HS, sorted_ids, sorted_expert_ids,
                n_tokens, num_m_blocks, a_scale_bytes, tile_map_t, grid1, ready,
            )  # fmt: skip
        elif bx < nsh1+N_CTAS+nsh2:
            wait_ready(shared_ready,needed,nsh2)
            sh2.tile(lds.all_lds.ptr,lds.all_lds.ptr+131072,lds.all_lds.ptr+139264,
                     fx.Int32(bx-nsh1-N_CTAS),Q8,W28,OUT8,S8,WS28,n_tokens,fx.Int32(6144),
                     fx.Int32(0),((n_tokens+255)//256)*24)
        else:
            bx=bx-nsh1-nsh2
            if const_expr(flat):
                k2.tile(
                    lds.all_lds.ptr, (lds.all_lds.ptr+max(t1,t2)), (lds.all_lds.ptr+max(t1,t2))+a_bytes, fx.recast_iter(fx.Int32,(lds.all_lds.ptr+max(t1,t2))+a_bytes+b_bytes),
                    bx - N_CTAS, HQ, W2, OUT, HS, W2_scale, sorted_ids, sorted_expert_ids,
                    sorted_weights, num_valid_ids, n_tokens, num_m_blocks, ready,
                )  # fmt: skip
            else:
                k2.tile(
                    lds.all_lds.ptr, (lds.all_lds.ptr+max(t1,t2))+a_bytes, (lds.all_lds.ptr+max(t1,t2)), fx.recast_iter(fx.Int32,(lds.all_lds.ptr+max(t1,t2))+a_bytes+b_bytes),
                    bx - N_CTAS, HQ, W2, OUT, HS, W2_scale, sorted_ids, sorted_expert_ids,
                    sorted_weights, num_valid_ids, n_tokens, num_m_blocks, ready,
                )  # fmt: skip

    @flyc.jit
    def launch_gemm12_mxfp4_shared_mxfp8_h6144_i768(
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
        A8: fx.Tensor, AS8: fx.Tensor, W138: fx.Tensor, WS138: fx.Tensor,
        Q8: fx.Tensor, S8: fx.Tensor, W28: fx.Tensor, WS28: fx.Tensor, OUT8: fx.Tensor,
        shared_ready: fx.Int64, nsh1: fx.Int32, nsh2: fx.Int32,
        grid2: fx.Int32,
        stream: fx.Stream,
    ):
        kernel_gemm12(
            A, W13, HQ, A_scale, W13_scale, HS, sorted_ids, sorted_expert_ids,
            n_tokens, num_m_blocks, a_scale_bytes, tile_map_t, grid1,
            W2, OUT, W2_scale, sorted_weights, num_valid_ids, ready,
            A8,AS8,W138,WS138,Q8,S8,W28,WS28,OUT8,shared_ready,nsh1,nsh2,
            value_attrs={
                "rocdl.waves_per_eu": 1,
                "rocdl.flat_work_group_size": "256,256",
            },
        ).launch(grid=(grid2 + N_CTAS+nsh1+nsh2, 1, 1), block=(256, 1, 1), stream=stream)  # fmt: skip

    launch_gemm12_mxfp4_shared_mxfp8_h6144_i768.kernel_name = name
    return launch_gemm12_mxfp4_shared_mxfp8_h6144_i768
