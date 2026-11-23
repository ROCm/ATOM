# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The gfx1250 DSpark draft MLA kernels' builds (CPU) and, on a gfx1250 GPU,
their outputs bit for bit against the kernels the draft's decode path runs:
``mla_draft_front_gfx1250`` against fused_qkv_a_proj's
``gemm_a8w8_blockscale_bpreshuffle`` (the FlyDSL mxfp8_128 GEMM at the
width's split-K, and its reduce) and ``fused_qk_rmsnorm_group_quant``;
``mla_absorb_draft_gfx1250`` against q_b_proj's (fp32 activation scales: the
gluon blockscale GEMM at the width's split-K), the W_UK batched GEMM and
``fused_qk_rope_concat_and_cache_mla`` (GPT-J, the draft's YaRN cache, DCP's
every-token q_out, slots of -1)."""

import pytest
import torch

flydsl = pytest.importorskip("flydsl", reason="the draft kernels build with FlyDSL")

from atom.models.kimi_k3.mono.kernels.stages import mla_absorb_draft_gfx1250 as ma
from atom.models.kimi_k3.mono.kernels.stages import mla_absorb_layout as alay
from atom.models.kimi_k3.mono.kernels.stages import mla_draft_front_gfx1250 as mf
from atom.models.kimi_k3.mono.kernels.stages import mla_draft_front_layout as flay


@pytest.mark.parametrize(
    ("rows", "ksplit"), [(1, 4), (7, 4), (28, 4), (33, 1), (64, 1)]
)
def test_the_front_is_one_launch(rows, ksplit):
    kernel = mf.build_mla_draft_front(rows, ksplit)
    assert kernel.grid == (flay.grid(rows),) and kernel.block == (256,)
    assert kernel.launch_count == 1 and kernel.abi is mf.MLA_DRAFT_FRONT_ABI
    assert kernel.name == f"k3_mla_draft_front_gfx1250_s{rows}_k{ksplit}"


@pytest.mark.parametrize("rows", (1, 7, 64))
def test_the_absorb_is_one_launch(rows):
    kernel = ma.build_mla_absorb_draft(rows, 64, 96, True, 2, 4096)
    assert kernel.grid == (alay.grid(64),) and kernel.qb_split_groups == 2
    assert kernel.name == f"k3_mla_absorb_draft_gfx1250_s{rows}_h64_o96_k2"
    with pytest.raises(ValueError):
        ma.build_mla_absorb_draft(rows, 64, 64, True, 0, 4096)


def test_the_absorb_abi_is_the_absorbs_with_the_rope_inputs():
    names = ma.MLA_ABSORB_DRAFT_ABI.names
    assert names[:2] == ("q_fp8", "q_scale") and names[-1] == "ql_nope"
    assert names.index("k_rope_stride") == names.index("k_rope") + 1
    assert {"positions", "cos", "sin"} <= set(names)


def _gfx1250():
    if not torch.cuda.is_available():
        return False
    from aiter.jit.utils.chip_info import get_gfx

    return get_gfx() == "gfx1250"


gpu = pytest.mark.skipif(not _gfx1250(), reason="needs a gfx1250 GPU")
SLOTS = 1024
EPS = 1e-5


def _bf16(gen, dev, *shape, scale=1.0, shift=0.0):
    return (torch.randn(shape, generator=gen, device=dev) * scale + shift).to(
        torch.bfloat16
    )


def _fp8_linear(gen, dev, rows, cols, scale):
    from aiter import QuantType, dtypes
    from aiter.ops.shuffle import shuffle_weight

    pytest.importorskip("aiter", reason="the kernel under test needs aiter")
    pytest.importorskip("triton", reason="the kernel under test needs triton")
    from atom.quantization.quark.utils import quant_weight_online

    q, s = quant_weight_online(
        _bf16(gen, dev, rows, cols, scale=scale),
        QuantType.per_1x128,
        dtypes.fp8,
        scale_dtype=dtypes.fp8_e8m0,
    )
    pad = -rows % 128
    if pad:
        q = torch.nn.functional.pad(q.view(torch.uint8), (0, 0, 0, pad)).view(q.dtype)
    return shuffle_weight(q, layout=(16, 16)).contiguous(), s.contiguous()


def _bits(t):
    return t.contiguous().view(torch.uint8)


def _raw(t):
    return torch.empty(0, dtype=t.dtype, device=t.device).set_(
        t.untyped_storage(), t.storage_offset(), (t.numel(),)
    )


def _group_quant(q_c, q_w, kv_c, kv_w):
    from aiter import QuantType, dtypes

    pytest.importorskip("aiter", reason="the kernel under test needs aiter")
    from atom.models.deepseek_v2 import _fused_rms_fp8_quant

    q8, qs, _, kv, _ = _fused_rms_fp8_quant(
        q_c, q_w, EPS, kv_c, kv_w, EPS, None, dtype_quant=dtypes.fp8, group_size=128,
        quant_type=QuantType.per_1x128.value, output_unquantized_inp1=False,
        transpose_scale=True,
    )  # fmt: skip
    return q8, qs, kv


@gpu
@pytest.mark.parametrize("rows", (7, 21, 35))
def test_the_front_is_productions_bits(rows):
    from aiter import dtypes, gemm_a8w8_blockscale_bpreshuffle

    pytest.importorskip("aiter", reason="the kernel under test needs aiter")
    from atom.model_ops.linear import per_1x128_e8m0_quant
    from atom.models.kimi_k3.mono import draft_dispatch

    dev = torch.device("cuda", torch.cuda.current_device())
    gen = torch.Generator(device=dev).manual_seed(rows)
    w_a, s_a = _fp8_linear(gen, dev, flay.QKV_A_ROWS, 7168, 0.02)
    q_w = _bf16(gen, dev, flay.Q_LORA, scale=0.1, shift=1.0)
    kv_w = _bf16(gen, dev, flay.KV_LORA, scale=0.1, shift=1.0)
    x, x_scale = per_1x128_e8m0_quant(
        _bf16(gen, dev, rows, 7168, scale=0.5), dtypes.fp8, True
    )
    x = x.contiguous()
    kernel = mf.build_mla_draft_front(
        rows, draft_dispatch.qkv_a_split_k(rows), EPS, EPS
    )
    e = torch.empty
    out = {
        "q_fp8": e(rows, flay.Q_LORA, dtype=dtypes.fp8, device=dev),
        "q_scale": e(rows, flay.Q_GROUPS, dtype=torch.float32, device=dev),
        "kv": e(rows, flay.KV_LORA, dtype=torch.bfloat16, device=dev),
        "k_rope": e(rows, flay.ROPE, dtype=torch.bfloat16, device=dev),
    }
    qkv_c = torch.full(
        (flay.qkv_words(),), flay.UNWRITTEN - (1 << 32), dtype=torch.int32, device=dev
    )
    partials = torch.empty(flay.partial_bytes(), dtype=torch.uint8, device=dev)
    p = torch.Tensor.data_ptr
    for _ in range(2):  # the second launch reads the hand-off the first restored
        args = mf.MLA_DRAFT_FRONT_ABI.pack({
            "x": p(x), "x_scale": p(x_scale), "w_a": p(w_a), "s_a": p(s_a),
            "q_norm": p(q_w), "kv_norm": p(kv_w), "qkv_c": p(qkv_c),
            "partials": p(partials), "qkv_seen": 0,
            **{name: p(t) for name, t in out.items()},
        })  # fmt: skip
        kernel.launcher(*args, stream=torch.cuda.current_stream())
    y = gemm_a8w8_blockscale_bpreshuffle(x, w_a, x_scale, s_a, torch.bfloat16)
    q_c, kv_c, k_pe = torch.split(
        y[:, : flay.QKV_A_ROWS], [flay.Q_LORA, flay.KV_LORA, flay.ROPE], dim=-1
    )
    q8, qs, kv = _group_quant(q_c, q_w, kv_c, kv_w)
    torch.cuda.synchronize()
    assert torch.equal(_bits(out["k_rope"]), _bits(k_pe))
    assert torch.equal(_bits(out["kv"]), _bits(kv))
    assert torch.equal(_bits(out["q_fp8"]), _bits(q8))
    assert torch.equal(
        _raw(out["q_scale"]).view(torch.uint8), _raw(qs).view(torch.uint8)
    )
    assert bool((qkv_c == flay.UNWRITTEN - (1 << 32)).all())


@gpu
@pytest.mark.parametrize(
    ("heads", "out_heads", "rows"), [(64, 64, 7), (64, 96, 21), (96, 96, 14)]
)
def test_the_absorb_is_productions_bits(heads, out_heads, rows):
    import aiter
    from aiter import dtypes, gemm_a8w8_blockscale_bpreshuffle
    from aiter.ops.triton.batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant import (
        batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant as bmm,
    )
    from aiter.rotary_embedding import get_rope

    pytest.importorskip("aiter", reason="the kernel under test needs aiter")
    from atom.model_ops.utils import dynamic_per_batched_tensor_quant
    from atom.models.kimi_k3.mono import draft_dispatch

    dev = torch.device("cuda", torch.cuda.current_device())
    gen = torch.Generator(device=dev).manual_seed(heads + rows)
    default = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    rope = get_rope(
        64, rotary_dim=64, max_position=4096, base=50000.0, is_neox_style=False,
        rope_scaling={"rope_type": "deepseek_yarn", "factor": 4.0,
                      "original_max_position_embeddings": 1024},
    ).to(dev)  # fmt: skip
    torch.set_default_dtype(default)
    w_qb, s_qb = _fp8_linear(gen, dev, heads * alay.HEAD, alay.Q_LORA, 0.03)
    w_uk, s_uk = dynamic_per_batched_tensor_quant(
        _bf16(gen, dev, heads, alay.KV_LORA, alay.NOPE, scale=0.05), dtype=dtypes.fp8
    )
    s_uk = s_uk.reshape(1).contiguous()
    q8, qs, kv = _group_quant(
        _bf16(gen, dev, rows, alay.Q_LORA, scale=2.0),
        _bf16(gen, dev, alay.Q_LORA, scale=0.1, shift=1.0),
        _bf16(gen, dev, rows, alay.KV_LORA),
        _bf16(gen, dev, alay.KV_LORA, scale=0.1, shift=1.0),
    )
    k_rope = _bf16(gen, dev, rows, alay.ROPE, scale=200.0)
    slots = torch.randperm(SLOTS, generator=gen, device=dev)[:rows].long()
    slots[1::3] = -1
    positions = torch.randint(0, 6000, (rows,), generator=gen, device=dev).long()
    cache = torch.randint(0, 255, (SLOTS, 1, alay.ENTRY), generator=gen, device=dev,
                          dtype=torch.uint8).view(dtypes.fp8)  # fmt: skip
    cache_ref = cache.clone()
    max_pos = rope.cos_cache.size(0)
    split = draft_dispatch.q_b_split_groups(rows, heads * alay.HEAD)
    kernel = ma.build_mla_absorb_draft(rows, heads, out_heads, True, split, max_pos)
    q = torch.empty(rows, heads, alay.HEAD, dtype=torch.bfloat16, device=dev)
    q_out = torch.empty(rows, out_heads, alay.ENTRY, dtype=dtypes.fp8, device=dev)
    p = torch.Tensor.data_ptr
    args = ma.MLA_ABSORB_DRAFT_ABI.pack({
        "q_fp8": p(q8), "q_scale": p(qs), "w_qb": p(w_qb), "s_qb": p(s_qb),
        "w_uk": p(w_uk), "s_uk": p(s_uk), "kv": p(kv), "k_rope": p(k_rope),
        "k_rope_stride": k_rope.stride(0), "slots": p(slots),
        "positions": p(positions), "cos": p(rope.cos_cache), "sin": p(rope.sin_cache),
        "kv_cache": p(cache), "kv_stride": cache.stride(0), "q_out": p(q_out),
        "q": p(q), "ql_nope": 0,
    })  # fmt: skip
    kernel.launcher(*args, stream=torch.cuda.current_stream())
    q_ref = gemm_a8w8_blockscale_bpreshuffle(q8, w_qb, qs, s_qb, torch.bfloat16).view(
        rows, heads, alay.HEAD
    )
    ql_nope = bmm(q_ref[..., : alay.NOPE].transpose(0, 1), w_uk, s_uk, group_size=128,
                  transpose_bm=True)  # fmt: skip
    ref = torch.empty(rows, heads, alay.ENTRY, dtype=dtypes.fp8, device=dev)
    one = torch.ones(1, dtype=torch.float32, device=dev)
    aiter.fused_qk_rope_concat_and_cache_mla(
        ql_nope, q_ref[..., alay.NOPE :], kv, k_rope,
        cache_ref.view(SLOTS, -1, alay.ENTRY), ref, slots, one, one, positions,
        rope.cos_cache, rope.sin_cache, is_neox=False, is_nope_first=True,
        compute_all_q_rope=True,
    )  # fmt: skip
    ref = torch.nn.functional.pad(ref, (0, 0, 0, out_heads - heads))
    torch.cuda.synchronize()
    assert torch.equal(_bits(q), _bits(q_ref))
    assert torch.equal(_bits(q_out), _bits(ref))
    assert torch.equal(_bits(cache), _bits(cache_ref))
