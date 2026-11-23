# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The gfx1250 MLA absorb's builds (CPU) and, on a gfx1250 GPU, its outputs
bit for bit against the kernels the decode branch runs: q_b_proj's
``gemm_a8w8_blockscale_bpreshuffle``, the absorb's gluon ``batched_gemm_a16w8``
and ``fused_qk_rope_concat_and_cache_mla`` -- for the DCP8 group's 96 QREP
heads (every token's q_out) and one rank's 12 heads (q_out padded to 16 zero
heads, a token without a KV slot zero)."""

import pytest
import torch

flydsl = pytest.importorskip("flydsl", reason="the MLA absorb builds with FlyDSL")

from atom.models.kimi_k3.mono.kernels.stages import mla_absorb_gfx1250 as ma
from atom.models.kimi_k3.mono.kernels.stages import mla_absorb_layout as lay


@pytest.mark.parametrize("rows", (1, 16, 17, 64))
def test_the_absorb_is_one_launch(rows):
    kernel = ma.build_mla_absorb(rows, 96)
    assert kernel.grid == (256,) and kernel.block == (256,)
    assert kernel.launch_count == 1 and kernel.abi is ma.MLA_ABSORB_ABI
    assert kernel.name == f"k3_mla_absorb_gfx1250_s{rows}_h96"
    local = ma.build_mla_absorb(rows, 12, 16, compute_all=False)
    assert local.grid == (24 + 64,)
    assert local.name == f"k3_mla_absorb_gfx1250_s{rows}_h12_o16_slot"


def test_the_abi_takes_the_front_outputs_first_and_the_tap_last():
    names = ma.MLA_ABSORB_ABI.names
    assert names[:2] == ("q_fp8", "q_scale") and names[-1] == "ql_nope"
    assert names.index("kv_stride") == names.index("kv_cache") + 1


def _gfx1250():
    if not torch.cuda.is_available():
        return False
    from aiter.jit.utils.chip_info import get_gfx

    return get_gfx() == "gfx1250"


gpu = pytest.mark.skipif(not _gfx1250(), reason="needs a gfx1250 GPU")
SLOTS = 1024


def _setup(heads, rows, gen, dev):
    from aiter import QuantType, dtypes
    from aiter.ops.shuffle import shuffle_weight

    pytest.importorskip("aiter", reason="the kernel under test needs aiter")
    from atom.model_ops.linear import per_1x128_e8m0_quant
    from atom.model_ops.utils import dynamic_per_batched_tensor_quant

    pytest.importorskip("triton", reason="the kernel under test needs triton")
    from atom.quantization.quark.utils import quant_weight_online

    def bf16(*shape, scale=1.0):
        return (torch.randn(shape, generator=gen, device=dev) * scale).to(
            torch.bfloat16
        )

    q_w, s_qb = quant_weight_online(
        bf16(heads * lay.HEAD, lay.Q_LORA, scale=0.03),
        QuantType.per_1x128,
        dtypes.fp8,
        scale_dtype=dtypes.fp8_e8m0,
    )
    w_uk, s_uk = dynamic_per_batched_tensor_quant(
        bf16(heads, lay.KV_LORA, lay.NOPE, scale=0.05), dtype=dtypes.fp8
    )
    q_fp8, q_scale = per_1x128_e8m0_quant(bf16(rows, lay.Q_LORA), dtypes.fp8, True)
    kv, k_rope = bf16(rows, lay.KV_LORA), bf16(rows, lay.ROPE, scale=200.0)
    kv[0, :4] = 1000.0
    slots = torch.randperm(SLOTS, generator=gen, device=dev)[:rows].long()
    slots[1::3] = -1
    return {
        "w_qb": shuffle_weight(q_w, layout=(16, 16)).contiguous(),
        "s_qb": s_qb.contiguous(),
        "w_uk": w_uk,
        "s_uk": s_uk.reshape(1).contiguous(),
        "q_fp8": q_fp8.contiguous(),
        "q_scale": q_scale,
        "kv": kv,
        "k_rope": k_rope,
        "slots": slots,
        "positions": torch.zeros(rows, dtype=torch.long, device=dev),
    }


def _production(t, cache, heads, out_heads, compute_all, q=None):
    """The decode branch's three kernels; ``q`` replaces q_b_proj's output."""
    import aiter
    from aiter import dtypes, gemm_a8w8_blockscale_bpreshuffle
    from aiter.ops.triton.batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant import (
        batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant as bmm,
    )

    pytest.importorskip("aiter", reason="the kernel under test needs aiter")
    from atom.model_ops.rotary_embedding import NoPositionalRotaryEmbedding

    rows = t["q_fp8"].size(0)
    if q is None:
        q = gemm_a8w8_blockscale_bpreshuffle(
            t["q_fp8"], t["w_qb"], t["q_scale"], t["s_qb"], torch.bfloat16
        ).view(rows, heads, lay.HEAD)
    ql_nope = bmm(
        q[..., : lay.NOPE].transpose(0, 1),
        t["w_uk"],
        t["s_uk"],
        group_size=128,
        transpose_bm=True,
    )
    default = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    rope = NoPositionalRotaryEmbedding(lay.ROPE, lay.ROPE, 64, 10000.0).to(q.device)
    torch.set_default_dtype(default)
    q_out = torch.zeros(rows, out_heads, lay.ENTRY, dtype=dtypes.fp8, device=q.device)
    one = torch.ones(1, dtype=torch.float32, device=q.device)
    aiter.fused_qk_rope_concat_and_cache_mla(
        ql_nope,
        q[..., lay.NOPE :],
        t["kv"],
        t["k_rope"],
        cache.view(cache.shape[0], -1, lay.ENTRY),
        q_out[:, :heads] if out_heads != heads else q_out,
        t["slots"],
        one,
        one,
        t["positions"],
        rope.cos_cache,
        rope.sin_cache,
        is_neox=rope.is_neox_style,
        is_nope_first=True,
        compute_all_q_rope=compute_all,
    )
    return q, ql_nope, q_out


def _bits(x):
    return x.contiguous().view(torch.uint8)


@gpu
@pytest.mark.parametrize(
    ("heads", "out_heads", "compute_all"), [(96, 96, True), (12, 16, False)]
)
@pytest.mark.parametrize("rows", (1, 17, 64))
def test_outputs_are_the_decode_branchs_bits(heads, out_heads, compute_all, rows):
    from aiter import dtypes

    dev = torch.device("cuda", torch.cuda.current_device())
    gen = torch.Generator(device=dev).manual_seed(rows + heads)
    t = _setup(heads, rows, gen, dev)
    kernel = ma.build_mla_absorb(rows, heads, out_heads, compute_all)
    cache = torch.randint(
        0, 255, (SLOTS, 1, lay.ENTRY), generator=gen, device=dev, dtype=torch.uint8
    ).view(dtypes.fp8)
    cache_ref = cache.clone()
    q = torch.empty(rows, heads, lay.HEAD, dtype=torch.bfloat16, device=dev)
    ql_nope = torch.empty(rows, heads, lay.KV_LORA, dtype=torch.bfloat16, device=dev)
    q_out = torch.full((rows, out_heads, lay.ENTRY), 7, dtype=torch.uint8, device=dev)
    p = torch.Tensor.data_ptr
    args = ma.MLA_ABSORB_ABI.pack({
        **{k: p(t[k]) for k in ("q_fp8", "q_scale", "w_qb", "s_qb", "w_uk", "s_uk",
                                "kv", "k_rope", "slots")},
        "kv_cache": p(cache), "kv_stride": cache.stride(0), "q_out": p(q_out),
        "q": p(q), "ql_nope": p(ql_nope),
    })  # fmt: skip
    kernel.launcher(*args, stream=torch.cuda.current_stream())
    q_prod, _, q_out_prod = _production(t, cache_ref, heads, out_heads, compute_all)
    _, uk_own, q_out_own = _production(
        t, cache_ref.clone(), heads, out_heads, compute_all, q=q
    )
    torch.cuda.synchronize()
    assert torch.equal(_bits(q), _bits(q_prod))
    assert torch.equal(_bits(ql_nope), _bits(uk_own))
    assert torch.equal(_bits(q_out), _bits(q_out_own))
    assert torch.equal(_bits(q_out), _bits(q_out_prod))
    assert torch.equal(_bits(cache), _bits(cache_ref))
