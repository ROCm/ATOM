# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The DSpark context projection's fused norm + quant against the op chain.

With the stages' `wkv` GEMMs fused, `write_context_kv` used to normalize the
projected context with `main_norm` to bf16 and let `native_quant_linear`
quantize that with `quantize_fp8`. `project_context_quant` emits the GEMM's
`(x, x_scale)` from the norm instead. Both take the target's taps, as
`write_context_kv` does, and project them through one shared helper; joined
into one block first, the taps give the same bytes.

That is not bit-exact: the fused kernel quantizes the fp32 normalized value,
the chain the bf16-rounded one. So, as in the draft-KV tail's test, the claim
is that where the two disagree the fused value is the one nearer an fp64
reference. The second test covers the scale layout: a scale read in the wrong
view does not fail, it writes a wrong KV row, so the GEMM output is checked
against fp64 and a misread scale is shown to fail that check.
"""

from types import SimpleNamespace

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip(
        "exercises AITER GPU kernels; needs a real GPU", allow_module_level=True
    )

pytest.importorskip("aiter", reason="RMSNorm and its fused quant are AITER kernels")

from aiter import QuantType
from aiter.ops.shuffle import shuffle_weight

from atom.model_ops.blockscale import native_quant_linear, quantize_fp8
from atom.model_ops.layernorm import RMSNorm
from atom.model_ops.linear import weight_is_stored_preshuffled
from atom.models.deepseek_v41.dspark import DeepseekV41DSpark

DEV = "cuda"
HIDDEN = 5120  # config.hidden_size
KV_WIDTH = 3 * 512  # num_nextn_predict_layers x head_dim, the fused wkv rows
EPS = 1e-20  # config.rms_norm_eps
TAPS = 3  # dspark_target_layer_ids


class _Draft:
    """The part of `DeepseekV41DSpark` the context projection reads."""

    project_context = DeepseekV41DSpark.project_context
    project_context_quant = DeepseekV41DSpark.project_context_quant
    _project_taps = DeepseekV41DSpark._project_taps
    _main_norm_quant = DeepseekV41DSpark._main_norm_quant

    def __init__(self, norm):
        self.mtp = [SimpleNamespace(main_proj=_main_proj, main_norm=norm)]
        DeepseekV41DSpark._build_main_norm_quant(self)


def _main_proj(x, x_scale):
    """Stands in for `main_proj`: takes the fp8 input and its scales, as the
    real one is handed them, and sums the taps' dequantized slices, so the
    context keeps the per-group gains every tap carries."""
    values = _dequant(x, x_scale).unflatten(-1, (TAPS, HIDDEN))
    return values.sum(-2).to(torch.bfloat16)


def _draft(seed):
    gen = torch.Generator(device=DEV).manual_seed(seed)
    norm = RMSNorm(HIDDEN, EPS).to(device=DEV, dtype=torch.bfloat16)
    # Spread around 1, the way a trained norm weight sits.
    norm.weight.data.copy_(
        0.5 + torch.rand(HIDDEN, generator=gen, device=DEV, dtype=torch.float32)
    )
    return _Draft(norm), gen


def _taps(width, gen):
    # A power of two per 32-column group, so groups pick different scales and
    # a scale read for the wrong group is far off rather than coincidentally
    # equal.
    group_gain = 2.0 ** torch.randint(
        -4, 5, (HIDDEN // 32,), generator=gen, device=DEV
    ).repeat_interleave(32)
    return [
        (torch.randn(width, HIDDEN, generator=gen, device=DEV) * group_gain).to(
            torch.bfloat16
        )
        for _ in range(TAPS)
    ]


def _reference(ctx, weight):
    x = ctx.double()
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * weight.double()


def _dequant(values, scales, rows=1):
    exponent = scales.view(torch.uint8).to(torch.int32) - 127
    scale = torch.ldexp(torch.ones_like(exponent, dtype=torch.float64), exponent)
    scale = scale.repeat_interleave(32, dim=-1)
    if rows > 1:
        scale = scale.repeat_interleave(rows, dim=-2)
    return values.double() * scale


def _quantize_32x32(weight):
    """E4M3 with one UE8M0 scale a 32x32 block: V4.1's native `wkv` layout."""
    n, k = weight.shape
    blocks = weight.float().reshape(n // 32, 32, k // 32, 32)
    amax = blocks.abs().amax(dim=(1, 3)).clamp(min=1e-10)
    code = torch.ceil(torch.log2(amax / 448.0)).to(torch.int32) + 127
    scale = torch.ldexp(torch.ones_like(amax), code - 127)
    values = (blocks / scale[:, None, :, None]).clamp(-448.0, 448.0)
    return (
        values.reshape(n, k).to(torch.float8_e4m3fn),
        code.to(torch.uint8).view(torch.float8_e8m0fnu),
    )


@pytest.mark.parametrize("width", [1, 37, 384])
def test_context_norm_quant_beats_op_chain_against_fp64(single_rank, width):
    draft, gen = _draft(seed=width)
    taps = _taps(width, gen)

    with torch.inference_mode():
        reference = _reference(draft._project_taps(taps), draft.mtp[0].main_norm.weight)
        values, scales = draft.project_context_quant(taps)
        chain = _dequant(*quantize_fp8(draft.project_context(taps)))
    assert values.shape == (1, width, HIDDEN)
    assert values.dtype == torch.float8_e4m3fn
    assert scales.shape == (1, width, HIDDEN // 32)
    assert scales.dtype == torch.float8_e8m0fnu
    fused = _dequant(values, scales)

    # Disagreements come from the bf16 rounding the fused path skips, so the
    # fused side of each should be nearer fp64 -- except a straddle, where the
    # two take the adjacent E4M3 codes either side of the reference and which
    # one is nearer is decided by the fp32 reduction order. Per quantization
    # group, the unit a scale is chosen over, the fused path is never worse.
    differs = fused != chain
    fused_error = (fused - reference).abs()
    chain_error = (chain - reference).abs()
    worse = differs & (fused_error > chain_error)
    straddles = (torch.minimum(fused, chain) <= reference) & (
        reference <= torch.maximum(fused, chain)
    )
    assert (worse & ~straddles).sum() == 0
    grouped = (width, HIDDEN // 32, 32)
    fused_group = fused_error.reshape(grouped).pow(2).mean(-1).sqrt()
    chain_group = chain_error.reshape(grouped).pow(2).mean(-1).sqrt()
    assert (fused_group > chain_group * (1 + 1e-6)).sum() == 0

    def _relative_rms(stored):
        return (stored - reference).pow(2).mean().sqrt() / reference.pow(
            2
        ).mean().sqrt()

    assert _relative_rms(fused) <= _relative_rms(chain)
    assert _relative_rms(fused) < 2.0**-4


@pytest.mark.parametrize("width", [1, 37, 384])
def test_context_norm_quant_scale_layout_feeds_the_gemm(single_rank, width):
    draft, gen = _draft(seed=100 + width)
    taps = _taps(width, gen)
    with torch.inference_mode():
        hidden = draft._project_taps(taps)
    weight = (
        torch.randn(KV_WIDTH, HIDDEN, generator=gen, device=DEV, dtype=torch.float32)
        * 0.02
    ).to(torch.bfloat16)
    weight_q, weight_scale = _quantize_32x32(weight)
    expected = _reference(hidden, draft.mtp[0].main_norm.weight) @ (
        _dequant(weight_q, weight_scale, rows=32).T
    )
    # Stored the way the loader stores `wkv` on this GPU.
    preshuffled = weight_is_stored_preshuffled(
        QuantType.per_1x32, torch.float8_e4m3fn, native_group_rows=32
    )
    if preshuffled:
        weight_q = shuffle_weight(weight_q, layout=(16, 16))

    def _relative_rms(kv):
        return (kv.double() - expected).pow(2).mean().sqrt() / expected.pow(
            2
        ).mean().sqrt()

    with torch.inference_mode():
        values, scales = draft.project_context_quant(taps)

        def _gemm(x, x_scale):
            return native_quant_linear(
                x,
                weight_q,
                weight_scale,
                weight_group_rows=32,
                weight_preshuffled=preshuffled,
                x_scale=x_scale,
            )

        fused = _gemm(values, scales)
        chain = _gemm(draft.project_context(taps), None)
        # The same bytes read as if the groups were laid out the other way
        # round: what a transposed or column-major scale view would do.
        flipped = scales.flip(-1)
        misread = _gemm(values, flipped)

    assert fused.shape == (1, width, KV_WIDTH)
    # Both carry an E4M3 activation into the GEMM, so both sit near 2 ** -5
    # of fp64; the fused one no further out than the chain plus noise.
    assert _relative_rms(fused) < 2.0**-4
    assert _relative_rms(fused) <= _relative_rms(chain) * 1.05
    if not torch.equal(flipped.view(torch.uint8), scales.view(torch.uint8)):
        assert _relative_rms(misread) > 4 * _relative_rms(fused)


def test_main_norm_quant_shares_the_main_norm_weight(single_rank):
    draft, gen = _draft(seed=3)
    fused_norm = draft._main_norm_quant()
    assert fused_norm.weight is draft.mtp[0].main_norm.weight
    assert draft._main_norm_quant() is fused_norm
    # In place, the way a weight load or sync writes it.
    draft.mtp[0].main_norm.weight.data.mul_(2)
    taps = _taps(5, gen)
    with torch.inference_mode():
        values, scales = draft.project_context_quant(taps)
        reference = _reference(draft._project_taps(taps), draft.mtp[0].main_norm.weight)
    assert (_dequant(values, scales) - reference).abs().max() < (
        reference.abs().max() * 2.0**-3
    )


@pytest.mark.parametrize("width", [1, 37, 384])
def test_context_projection_takes_taps_as_the_joined_block(single_rank, width):
    """The taps, quantized where they lie, against the same rows joined into
    one block (a PP relay, which `quantize_fp8_taps` quantizes as it is):
    both projections come out byte for byte the same."""
    draft, gen = _draft(seed=200 + width)
    taps = _taps(width, gen)
    joined = [torch.cat(taps, dim=-1)]
    with torch.inference_mode():
        values, scales = draft.project_context_quant(taps)
        joined_values, joined_scales = draft.project_context_quant(joined)
        hidden = draft.project_context(taps)
        joined_hidden = draft.project_context(joined)
    assert torch.equal(values.view(torch.uint8), joined_values.view(torch.uint8))
    assert torch.equal(scales.view(torch.uint8), joined_scales.view(torch.uint8))
    assert torch.equal(hidden, joined_hidden)
