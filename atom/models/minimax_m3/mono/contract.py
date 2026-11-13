# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The fused layer kernel's hand-off contract, checked on a build.

Tracing a build (compile-only, no launch, no GPU needed with ``ARCH`` set) records
its mailbox accesses (``atom.mono.plan.trace``); ``check_traced`` holds them to
``layout.mailbox_regions``. A build served from FlyDSL's disk cache is not traced,
so the caller disables the cache (``FLYDSL_RUNTIME_ENABLE_CACHE=0``).

Usage: python -m atom.models.minimax_m3.mono.contract [S ...]
"""

import sys

from atom.models.minimax_m3.mono.config import TP, IndexHeads
from atom.models.minimax_m3.mono.kernels.dense_post import (
    DENSE_POST_ABI,
    build_dense_post_kernel,
)
from atom.models.minimax_m3.mono.kernels.post_attn import K4_ABI, build_post_attn_kernel
from atom.models.minimax_m3.mono.layout import (
    WIDE_FROM,
    dense_mailbox_regions,
    mailbox_regions,
)
from atom.mono.runtime.compile import check_traced


def build_layer(
    tokens: int,
    index_heads: int,
    fuse_k1: bool = True,
    index_topk: bool = True,
    xn_flag: int = 0,
    shared_mxfp8: bool = False,
    mid_flag: int = 0,
    kv_nvfp4: bool = False,
    index_nvfp4: bool = False,
    sh_first: int = 0,
    attn_mxfp8: bool = False,
    sp_f32: bool = True,
    moe_a16w4: bool = True,
    moe_a4w4: bool = False,
    moe_back: bool = False,
):
    """The layer kernel for ``tokens`` and ``index_heads`` (1 or TP), K1 fused in
    or not (``fuse_k1``), selecting or reusing a selection (``index_topk``), its
    xn behind flags (``xn_flag``), the shared expert an MXFP8 MLP of its own
    (``shared_mxfp8``), NVFP4 K/V / index pages (``kv_nvfp4`` / ``index_nvfp4``), native MXFP8 qkv / o_proj
    (``attn_mxfp8``)."""
    heads = IndexHeads(index_heads, 0)
    return build_post_attn_kernel(
        TP, 0.088, 1e-6, 1.0, 1.0, 7.0, 1, 1, tokens, fuse_k1=fuse_k1, heads=heads,
        index_topk=index_topk, xn_flag=xn_flag, shared_mxfp8=shared_mxfp8, mid_flag=mid_flag,
        kv_nvfp4=kv_nvfp4, index_nvfp4=index_nvfp4, sh_first=sh_first,
        attn_mxfp8=attn_mxfp8, sp_f32=sp_f32, moe_a16w4=moe_a16w4, moe_a4w4=moe_a4w4,
        moe_back=moe_back,
    )  # fmt: skip


def build_front(tokens: int, n_split: int = 8, pipe: bool = False, nvfp4: bool = True):
    """The front-half kernel (more than MAX_TOKENS rows: K1 .. the post-attention norm, ``front``)
    for ``tokens`` rows, one index head, native MXFP8 attention + NVFP4 K/V / index pages
    (``nvfp4``; else fp8 caches and ptpc attention) with ``n_split`` split tasks a token.
    """
    return build_post_attn_kernel(
        TP, 0.088, 1e-6, 1.0, 1.0, 7.0, 1, 1, tokens, fuse_k1=True,
        heads=IndexHeads(1, 0), shared_mxfp8=True, kv_nvfp4=nvfp4,
        index_nvfp4=nvfp4, attn_mxfp8=nvfp4, front=True, n_split=n_split,
        pipe=pipe and nvfp4,
    )  # fmt: skip


def check_front(tokens: int, n_split: int = 8, pipe: bool = False, nvfp4: bool = True):
    """Trace ``build_front`` and check its hand-offs; raises ``ContractError``."""
    check_traced(
        build_front(tokens, n_split, pipe, nvfp4),
        K4_ABI,
        mailbox_regions(tokens, 1, True, True, 0, 0, front=True),
    )


def build_dense(tokens: int):
    """The dense layer kernel (dense_post) for ``tokens``."""
    return build_dense_post_kernel(TP, 1e-6, 1.702, 1.0, 7.0, tokens)


def check_build(
    tokens: int,
    index_heads: int,
    fuse_k1: bool = True,
    index_topk: bool = True,
    xn_flag: int = 0,
    shared_mxfp8: bool = False,
    mid_flag: int = 0,
    kv_nvfp4: bool = False,
    index_nvfp4: bool = False,
    sh_first: int = 0,
    attn_mxfp8: bool = False,
    sp_f32: bool = True,
    moe_a16w4: bool = True,
    moe_a4w4: bool = False,
    moe_back: bool = False,
) -> None:
    """Trace ``build_layer`` and check its hand-offs; raises ``ContractError``."""
    check_traced(
        build_layer(
            tokens,
            index_heads,
            fuse_k1,
            index_topk,
            xn_flag,
            shared_mxfp8,
            mid_flag,
            kv_nvfp4,
            index_nvfp4,
            sh_first,
            attn_mxfp8,
            sp_f32,
            moe_a16w4,
            moe_a4w4,
            moe_back,
        ),
        K4_ABI,
        mailbox_regions(
            tokens,
            index_heads,
            fuse_k1,
            index_topk,
            xn_flag,
            mid_flag,
            sp_f32,
            moe_a16w4 and shared_mxfp8 and not moe_a4w4,
            moe_a4w4,
            False,
            moe_back,
        ),
    )


def check_dense_build(tokens: int) -> None:
    """Trace ``build_dense`` and check its hand-offs; raises ``ContractError``."""
    check_traced(build_dense(tokens), DENSE_POST_ABI, dense_mailbox_regions(tokens))


if __name__ == "__main__":
    sizes = [int(a) for a in sys.argv[1:]] or list(range(1, 17))
    for s in sizes:
        if s > 16:
            for ns, pipe in ((8, False), (8, True), (4, False), (4, True)):
                check_front(s, ns, pipe)
                print(f"S={s} front n_split={ns} pipe={pipe}: contract holds")
            check_front(s, 8, False, False)
            print(f"S={s} front fp8 / ptpc: contract holds")
            if (
                s % 16 == 0 and s <= 64
            ):  # the 32 / 48 / 64-row whole-layer builds (front half + back half in one launch)
                check_build(
                    s, 1, shared_mxfp8=True, attn_mxfp8=True, kv_nvfp4=True, index_nvfp4=True, moe_a4w4=True,
                    sp_f32=False, moe_back=True,
                )  # fmt: skip
                print(
                    f"S={s} moe_back (front + router + a4w4 MoE + shared + FFN reduce): contract holds"
                )
            continue
        for ih in (1, TP):
            # a16w4 (the default) and the previous a8w4 experts (only a build with the
            # unfused MXFP8 shared expert, ``shared_mxfp8``, takes a16w4)
            for a16 in (True, False):
                tag = "a16w4" if a16 else "a8w4"
                for it in (True, False):
                    for xf in (0, 1, 2):
                        check_build(s, ih, index_topk=it, xn_flag=xf, moe_a16w4=a16)
                        print(
                            f"S={s} index heads={ih} index_topk={it} xn_flag={xf} {tag}:"
                            " contract holds"
                        )
                if s < WIDE_FROM and not a16:
                    check_build(s, ih, xn_flag=1, mid_flag=1, moe_a16w4=False)
                    print(
                        f"S={s} index heads={ih} xn_flag=1 mid_flag=1: contract holds"
                    )
                check_build(s, ih, shared_mxfp8=True, moe_a16w4=a16)
                print(f"S={s} index heads={ih} shared_mxfp8 {tag}: contract holds")
                if s >= WIDE_FROM:
                    for k in (1, 2, 4):
                        check_build(s, ih, shared_mxfp8=True, sh_first=k, moe_a16w4=a16)
                        print(
                            f"S={s} index heads={ih} sh_first={k} {tag}: contract holds"
                        )
                check_build(s, ih, kv_nvfp4=True, index_nvfp4=True, moe_a16w4=a16)
                print(f"S={s} index heads={ih} nvfp4 K/V + index {tag}: contract holds")
                for nv in (False, True):
                    check_build(
                        s,
                        ih,
                        shared_mxfp8=True,
                        attn_mxfp8=True,
                        kv_nvfp4=nv,
                        index_nvfp4=nv,
                        moe_a16w4=a16,
                    )
                    print(
                        f"S={s} index heads={ih} attn_mxfp8 nvfp4={nv} {tag}: contract holds"
                    )
            # a4w4 routed experts (needs the unfused MXFP8 shared expert)
            for spf in (True, False):
                check_build(s, ih, shared_mxfp8=True, moe_a4w4=True, sp_f32=spf)
                print(f"S={s} index heads={ih} a4w4 sp_f32={spf}: contract holds")
            check_build(
                s, ih, shared_mxfp8=True, attn_mxfp8=True, kv_nvfp4=True,
                index_nvfp4=True, moe_a4w4=True, sp_f32=False,
            )  # fmt: skip
            print(f"S={s} index heads={ih} a4w4 attn_mxfp8 nvfp4: contract holds")
