# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2024-2026 ATOM Project Contributors

"""The fused DSpark draft-KV tail against the op chain it replaces.

`DeepseekV41DSpark.write_context_kv` used to spend ten kernels turning one
fused-GEMM output into the three stages' quantized target keys: a per-stage
RMSNorm, a cat, a RoPE and a quantize, each re-reading the 512-wide row from
HBM to hand it to the next. `fused_draft_kv_tail` is those ten in one launch.

The fused kernel rounds where the chain does (the norm's bf16 store, RoPE's
bf16 store, then the QAT), so the two store the same codes. Not every one:
the fp32 sum-of-squares reduction tree and `rsqrt`'s approximation differ from
the HIP kernel's, and an element whose bf16 rounding sits on that last-ulp
difference lands one code over -- swept at about 1 in 1M.

The fp64 reference (the layout test's) reuses the rope's own `cos_cache` /
`sin_cache` rather than rebuilding the frequencies, because reusing them is
exactly what the kernel contract promises -- YaRN scaling and cache dtype have
to come along.
"""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip(
        "exercises a Triton kernel; needs a real GPU",
        allow_module_level=True,
    )

from atom.model_ops.blockscale import quantize_fp8
from atom.model_ops.deepseek_v41.dspark import fused_draft_kv_tail
from atom.model_ops.deepseek_v41.rotary import RotaryEmbedding
from atom.model_ops.layernorm import rmsnorm2d_fwd_

DEV = "cuda"
DIM = 512  # config.head_dim
PE_DIM = 64  # config.qk_rope_head_dim
STAGES = 3  # config.num_nextn_predict_layers
EPS = 1e-20  # config.rms_norm_eps
MAX_POSITION = 4096


def _inputs(width, seed=0):
    """A drafting step's worth of fused-GEMM output, weights and positions."""
    gen = torch.Generator(device=DEV).manual_seed(seed)
    kv = torch.randn(
        1, width, STAGES, DIM, generator=gen, device=DEV, dtype=torch.bfloat16
    )
    # Distinct per stage, so a kernel that read one stage's weight for another
    # cannot pass; spread around 1 the way a trained norm weight sits.
    norm_weight = (
        0.5 + torch.rand(STAGES, DIM, generator=gen, device=DEV, dtype=torch.float32)
    ).to(torch.bfloat16)
    positions = torch.randint(
        0, MAX_POSITION, (width,), generator=gen, device=DEV, dtype=torch.int64
    )
    rope = RotaryEmbedding(PE_DIM, MAX_POSITION, base=10000.0).to(DEV)
    return kv, norm_weight, positions, rope


def _reference(kv, norm_weight, positions, rope):
    """The chain in fp64, stage-major, stopping before the quantizer."""
    x = kv.reshape(-1, STAGES, DIM).double().transpose(0, 1)  # [L, W, D]
    w = norm_weight.double()[:, None, :]
    y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * w
    pairs = y[..., -PE_DIM:].unflatten(-1, (PE_DIM // 2, 2))
    cos = rope.cos_cache.double()[positions]  # [W, PE_DIM // 2]
    sin = rope.sin_cache.double()[positions]
    even, odd = pairs[..., 0], pairs[..., 1]
    rotated = torch.stack(
        (even * cos - odd * sin, even * sin + odd * cos), dim=-1
    ).flatten(-2)
    return torch.cat((y[..., :-PE_DIM], rotated), dim=-1)


def _op_chain(kv, norm_weight, positions, rope, *, packed):
    """What `write_context_kv` ran before the fusion, spelled the same way."""
    normed = torch.cat(
        [rmsnorm2d_fwd_(kv[..., i, :], norm_weight[i], EPS, DIM) for i in range(STAGES)]
    )
    return quantize_fp8(rope(normed, positions), dequantize=not packed)


def _stored(keys, *, packed):
    """The value the cache ends up holding, in fp64, for either layout."""
    if not packed:
        return keys.double()
    values, scales = keys
    # The E8M0 cast is exact; `torch.ldexp` on gfx950 (torch 2.10 / ROCm
    # 7.2.4) comes out an ulp low for some exponents.
    scale = scales.view(torch.float8_e8m0fnu).double()
    return values.double() * scale.repeat_interleave(32, dim=-1)


@pytest.mark.parametrize("packed", [True, False])
@pytest.mark.parametrize("width", [1, 37, 384])
def test_ctx_kv_tail_stores_the_op_chains_codes(packed, width):
    kv, norm_weight, positions, rope = _inputs(width, seed=width)

    fused = _stored(
        fused_draft_kv_tail(
            kv,
            norm_weight,
            positions,
            rope.cos_cache,
            rope.sin_cache,
            EPS,
            packed=packed,
        ),
        packed=packed,
    )
    # The op chain rotates in place, so it gets its own copy of the input.
    chain = _stored(
        _op_chain(kv.clone(), norm_weight, positions, rope, packed=packed),
        packed=packed,
    )

    assert fused.shape == (STAGES, width, DIM)

    # Swept over widths 1..1024 x 3 seeds (6.7M elements): 5 differ, each by
    # one E4M3 code (relative 2 ** -3 at most). A missed bf16 rounding point
    # moves about 2 % of codes, a wrong weight or rotation nearly all of them.
    differs = fused != chain
    assert differs.sum() <= max(1, differs.numel() // 100_000)
    step = (fused - chain).abs()[differs] / chain.abs()[differs]
    assert (step <= 2.0**-3).all()


@pytest.mark.parametrize("packed", [True, False])
def test_ctx_kv_tail_is_stage_major_and_per_stage(packed):
    """Guards the transpose and the per-stage weight indexing.

    Both window writers address rows by width, so a stage written strided --
    or a (stage, token) pair read in the wrong order -- would be written into
    the ring at the wrong offset rather than failing loudly.
    """
    width = 29
    kv, norm_weight, positions, rope = _inputs(width, seed=7)
    reference = _reference(kv, norm_weight, positions, rope)
    keys = fused_draft_kv_tail(
        kv,
        norm_weight,
        positions,
        rope.cos_cache,
        rope.sin_cache,
        EPS,
        packed=packed,
    )
    stored = _stored(keys, packed=packed)

    # Quantization noise puts ~2.6% between a stage and its own reference;
    # reading the wrong stage's weight, or the wrong token's position, puts
    # ~110% there. Asserting the gap rather than an absolute tolerance keeps
    # this test about layout and lets the numerics test own the numerics.
    for i in range(STAGES):
        row = keys[0][i : i + 1] if packed else keys[i : i + 1]
        # Both window writers address rows by width, so a stage that came out
        # strided would be written into the ring at the wrong offset rather
        # than failing loudly.
        assert row.is_contiguous()
        own = (stored[i] - reference[i]).abs().mean()
        for j in range(STAGES):
            if j != i:
                assert own * 10 < (stored[i] - reference[j]).abs().mean()
        assert own * 10 < (stored[i] - reference[i].roll(1, 0)).abs().mean()


def test_ctx_kv_tail_empty_step():
    """A zero-width step allocates the right shapes and launches nothing."""
    kv, norm_weight, positions, rope = _inputs(0)
    values, scales = fused_draft_kv_tail(
        kv, norm_weight, positions, rope.cos_cache, rope.sin_cache, EPS, packed=True
    )
    assert values.shape == (STAGES, 0, DIM)
    assert scales.shape == (STAGES, 0, DIM // 32)
