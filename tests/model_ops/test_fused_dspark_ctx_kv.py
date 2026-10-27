# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The fused DSpark draft-KV tail against the op chain it replaces.

`DeepseekV41DSpark.write_context_kv` used to spend ten kernels turning one
fused-GEMM output into the three stages' quantized target keys: a per-stage
RMSNorm, a cat, a RoPE and a quantize, each re-reading the 512-wide row from
HBM to hand it to the next. `fused_draft_kv_tail` is those ten in one launch.

Bit-exactness is not claimed and is not achievable from Triton -- the fp32
sum-of-squares reduction tree and `rsqrt`'s approximation differ from the HIP
kernel's -- and the fused kernel additionally rounds *less*: the op chain lands
in bf16 three times on the way to the FP8 store (aiter's `rmsnorm2d_fwd`
rounds `x * rstd` before applying `w`, then again on its store, then RoPE
stores for `quantize_fp8` to read back) where the fused one carries fp32
throughout. So the assertion here is the weaker, true one the MLA context
kernel's test makes: build an fp64 reference of the same chain, and where the
two GPU paths disagree, require the fused value to be the one nearer fp64.

The fp64 reference reuses the rope's own `cos_cache` / `sin_cache` rather than
rebuilding the frequencies, because reusing them is exactly what the kernel
contract promises -- YaRN scaling and cache dtype have to come along.
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
    exponent = scales.view(torch.uint8).to(torch.int32) - 127
    scale = torch.ldexp(torch.ones_like(exponent, dtype=torch.float64), exponent)
    return values.double() * scale.repeat_interleave(32, dim=-1)


@pytest.mark.parametrize("packed", [True, False])
@pytest.mark.parametrize("width", [1, 37, 384])
def test_ctx_kv_tail_beats_op_chain_against_fp64(packed, width):
    kv, norm_weight, positions, rope = _inputs(width, seed=width)
    reference = _reference(kv, norm_weight, positions, rope)

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

    # Both paths run the same quantizer, so a disagreement is its *input*
    # differing by the roundings the fused path does not do. About 3.4% of
    # elements disagree -- E4M3's three mantissa bits flip far more readily
    # than the bf16 store the MLA precedent measured at 1 in 5M -- and the
    # claim is that the fused side of each is the one nearer fp64.
    #
    # Not quite every one, and the exception is worth naming rather than
    # tolerancing away. Swept over widths 1..1024 x 3 seeds (7.3M elements),
    # 7 elements -- 1 in 1M -- come out nearer under the op chain. All 7 are
    # *straddles*: the two paths pick the two adjacent E4M3 codes on either
    # side of the reference, and there the chain's happens to be the closer
    # one. Which side of a code boundary a value lands on is decided by the
    # ~1e-7 the fp32 reduction tree and rsqrt move it, which is exactly what
    # this kernel does not claim to reproduce. So: never worse by more than
    # one code, and -- the statement with no exceptions -- never worse per
    # quantization group, which is the unit a scale is actually chosen over.
    differs = fused != chain
    fused_error = (fused - reference).abs()
    chain_error = (chain - reference).abs()
    worse = differs & (fused_error > chain_error)
    straddles = (torch.minimum(fused, chain) <= reference) & (
        reference <= torch.maximum(fused, chain)
    )
    assert (worse & ~straddles).sum() == 0, (
        f"{(worse & ~straddles).sum().item()} of {differs.sum().item()} "
        "disagreements move AWAY from the fp64 reference by more than the "
        "code boundary they sit on"
    )
    grouped = (STAGES, width, DIM // 32, 32)
    fused_group = fused_error.reshape(grouped).pow(2).mean(-1).sqrt()
    chain_group = chain_error.reshape(grouped).pow(2).mean(-1).sqrt()
    # The margin absorbs fp64 ties, which do occur: the sibling kernel's sweep
    # turned up 3 groups in 724k that differ by 0.000% of the chain's RMS.
    regressed = fused_group > chain_group * (1 + 1e-6)
    assert regressed.sum() == 0, (
        f"{regressed.sum().item()} quantization groups are measurably further "
        "from the fp64 reference under the fusion"
    )

    def _relative_rms(stored):
        return (stored - reference).pow(2).mean().sqrt() / reference.pow(
            2
        ).mean().sqrt()

    assert _relative_rms(fused) <= _relative_rms(chain)
    # E4M3 rounds to 3 mantissa bits, so a correct store sits inside a half ULP
    # (2 ** -4) of the exact value; anything systematically wrong -- a missed
    # rotation, the wrong stage's norm weight -- is an order of magnitude out.
    assert _relative_rms(fused) < 2.0**-4


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
