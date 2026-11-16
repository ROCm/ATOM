# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""A split-K GEMM that leaves its reduction to the consumer
(`LinearBase.splitk_consumer_max_m`) must leave every output bit as the GEMM's
own reduce kernel would: the MiniMax-M3 dense layer's fused q/k norm + RoPE +
cache kernel and its SwiGLU-OAI reading the fp32 partials, against the same
kernels reading the reduced bf16 output."""

from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is None,
    reason="gfx1250 split-K PTPC GEMM",
)

HEAD_DIM = 128
K = 6144
# MiniMax-M3 TP4 dense layer: 16 q heads, 1 kv head.
NUM_HEADS, NUM_KV_HEADS = 16, 1
N = (NUM_HEADS + 2 * NUM_KV_HEADS) * HEAD_DIM


def _aiter_or_skip():
    aiter = pytest.importorskip("aiter")
    from aiter.jit.utils.chip_info import get_gfx

    if get_gfx() != "gfx1250" or not hasattr(
        aiter, "gemm_a8w8_bpreshuffle_splitk_deferred"
    ):
        pytest.skip("needs gfx1250 and aiter's consumer-side split-K reduction")
    return aiter


def _bitwise(a, b):
    return torch.equal(
        a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
    )


def _layer_state(rows, kv_cache_dtype, fp8_dtype):
    torch.manual_seed(100 + rows)
    rotary_dim, max_pos, block_size = 64, 4096, 128
    inv = 1.0 / (10000 ** (torch.arange(0, rotary_dim, 2).float() / rotary_dim))
    freqs = torch.outer(torch.arange(max_pos).float(), inv).cuda()
    cache_dtype = fp8_dtype if kv_cache_dtype == "fp8" else torch.bfloat16
    x = 16 // torch.empty(0, dtype=cache_dtype).element_size()
    blocks = rows // block_size + 2

    def caches():
        return {
            "k_cache": torch.zeros(
                blocks,
                NUM_KV_HEADS,
                HEAD_DIM // x,
                block_size,
                x,
                dtype=cache_dtype,
                device="cuda",
            ),
            "v_cache": torch.zeros(
                blocks,
                NUM_KV_HEADS,
                block_size // x,
                HEAD_DIM,
                x,
                dtype=cache_dtype,
                device="cuda",
            ),
            "k_scale": torch.zeros(blocks, NUM_KV_HEADS, block_size, device="cuda"),
            "v_scale": torch.zeros(blocks, NUM_KV_HEADS, block_size, device="cuda"),
        }

    norm = lambda: SimpleNamespace(
        weight=torch.randn(HEAD_DIM, device="cuda", dtype=torch.bfloat16) * 0.1,
        variance_epsilon=1e-6,
    )
    common = {
        "positions": torch.randint(0, max_pos, (rows,), device="cuda"),
        "q_norm": norm(),
        "k_norm": norm(),
        "rotary_emb": SimpleNamespace(
            rotary_dim=rotary_dim,
            cos_cache=freqs.cos().to(torch.bfloat16),
            sin_cache=freqs.sin().to(torch.bfloat16),
        ),
        "num_heads": NUM_HEADS,
        "num_kv_heads": NUM_KV_HEADS,
        "head_dim": HEAD_DIM,
        "slot_mapping": torch.randperm(blocks * block_size, device="cuda")[:rows],
        "kv_cache_dtype": kv_cache_dtype,
    }
    return common, caches


def _consume(qkv, common, cache, qkv_partials=None):
    from atom.model_ops.triton_fused_qkv_norm_rope_cache import (
        triton_fused_norm_rope_cache,
    )

    q_size = NUM_HEADS * HEAD_DIM
    kv_size = NUM_KV_HEADS * HEAD_DIM
    q, k, v = torch.split(qkv, [q_size, kv_size, kv_size], dim=-1)
    scaled = common["kv_cache_dtype"] == "fp8"
    return triton_fused_norm_rope_cache(
        q,
        k,
        v,
        k_scale=cache["k_scale"] if scaled else None,
        v_scale=cache["v_scale"] if scaled else None,
        k_cache=cache["k_cache"],
        v_cache=cache["v_cache"],
        qkv_partials=qkv_partials,
        **common,
    )


@pytest.mark.parametrize("split_k", [2, 4, 8])
def test_triton_consumers_match_the_reduce_kernel_on_edge_values(split_k):
    aiter = _aiter_or_skip()
    from aiter.ops.flydsl.bpreshuffle_gemm_gfx1250 import (
        reduce_splitk_partials_gfx1250,
    )

    from atom.model_ops.swiglu_oai import swiglu_oai_split

    rows = 16
    tiny = torch.finfo(torch.float32).tiny
    # Sums that are subnormal (from normal and from subnormal partials), overflow,
    # NaN and a bf16 rounding tie, between random columns.
    cases = [
        (1.5 * tiny, -tiny),
        (3e-39, 1e-39),
        (3.3e38, 3.3e38),
        (float("inf"), float("-inf")),
        (1.0, 2.0**-9),
    ]

    def partials(n):
        torch.manual_seed(split_k)
        p = torch.randn(split_k, rows, n, device="cuda") * 4
        for c, (a, b) in enumerate(cases):
            p[:, :, c :: len(cases) + 2] = 0.0
            p[0, :, c :: len(cases) + 2] = a
            p[1, :, c :: len(cases) + 2] = b
        ref = torch.empty(rows, n, device="cuda", dtype=torch.bfloat16)
        reduce_splitk_partials_gfx1250(p, ref)
        return p, ref

    p, ref = partials(N)
    qkv = torch.empty(rows, N, device="cuda", dtype=torch.bfloat16)
    common, caches = _layer_state(rows, "bf16", aiter.dtypes.fp8)
    _consume(qkv, common, caches(), p)
    assert _bitwise(qkv, ref)

    p, ref = partials(6144)
    gate_up = torch.empty(rows, 6144, device="cuda", dtype=torch.bfloat16)
    swiglu_oai_split(gate_up, 1.702, 1.0, 7.0, gate_up_partials=p)
    assert _bitwise(gate_up, ref)
