# SPDX-License-Identifier: MIT
"""Small PyTorch oracles for the pinned DeepSeek-V4.1 TileLang kernel contracts.

These are test code, not serving fallbacks. Source provenance is recorded in
fixtures/reference_manifest.json. GEMMs dequantize for clarity; sparse attention
retains the reference's 64-row online softmax and BF16 probability rounding.
"""

import torch
import torch.nn.functional as F


def _pow2_scale(amax, maximum, minimum):
    return torch.exp2(torch.ceil(torch.log2(amax.clamp_min(minimum) / maximum)))


def act_quant(
    x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False
):
    grouped = x.float().unflatten(-1, (-1, block_size))
    amax = grouped.abs().amax(-1)
    scale = (
        _pow2_scale(amax, 448.0, 1e-4)
        if scale_fmt is not None
        else amax.clamp_min(1e-4) / 448.0
    )
    quant = (grouped / scale.unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn)
    if inplace:
        x.copy_((quant.float() * scale.unsqueeze(-1)).flatten(-2).to(x.dtype))
        return x
    return quant.flatten(-2), scale.to(scale_dtype)


def unpack_fp4(packed):
    """E2M1, low nibble first, including both signed zeros."""
    bits = packed.view(torch.uint8)
    codes = torch.stack((bits & 15, bits >> 4), dim=-1).flatten(-2).long()
    values = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=packed.device)
    return torch.where(codes >= 8, -values[codes & 7], values[codes & 7])


def fp4_act_quant(x, block_size=32, inplace=False, scale_dtype=torch.float8_e8m0fnu):
    grouped = x.float().unflatten(-1, (-1, block_size))
    amax = grouped.abs().amax(-1)
    if scale_dtype == torch.float8_e4m3fn:
        scale = (amax.clamp_min(6 * 2**-9) / 6).to(scale_dtype).float()
    else:
        scale = _pow2_scale(amax, 6.0, 6 * 2**-126)
    scaled = (grouped / scale.unsqueeze(-1)).clamp(-6, 6)
    levels = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=x.device)
    distance = (scaled.abs().unsqueeze(-1) - levels).abs()
    code = distance.argmin(-1)
    # A tie must select the even encoding, not always the smaller magnitude.
    upper = (code + 1).clamp_max(7)
    ties = distance.gather(-1, code.unsqueeze(-1)) == distance.gather(
        -1, upper.unsqueeze(-1)
    )
    code = torch.where(ties.squeeze(-1) & (code % 2 == 1), upper, code)
    code = (code | (torch.signbit(scaled).long() << 3)).flatten(-2).to(torch.uint8)
    packed = (code[..., ::2] | (code[..., 1::2] << 4)).contiguous()
    if inplace:
        x.copy_(
            (unpack_fp4(packed).unflatten(-1, (-1, block_size)) * scale.unsqueeze(-1))
            .flatten(-2)
            .to(x.dtype)
        )
        return x
    return packed.view(torch.float4_e2m1fn_x2), scale.to(scale_dtype)


def _dequant_rows(x, scales, block_size):
    return (
        x.float().unflatten(-1, (-1, block_size)) * scales.float().unsqueeze(-1)
    ).flatten(-2)


def fp8_gemm(a, a_s, b, b_s, scale_dtype=torch.float32, block_size=128):
    activation = _dequant_rows(a, a_s, block_size)
    weight_scale = (
        b_s.float().repeat_interleave(block_size, 0).repeat_interleave(block_size, 1)
    )
    weight = b.float() * weight_scale[: b.shape[0], : b.shape[1]]
    return F.linear(activation, weight).to(torch.get_default_dtype())


def fp4_gemm(a, a_s, b, b_s, scale_dtype=torch.float32, act_block_size=128):
    activation = _dequant_rows(a, a_s, act_block_size)
    weight = _dequant_rows(unpack_fp4(b), b_s, 32)
    return F.linear(activation, weight).to(torch.get_default_dtype())


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    pre, post, comb = mixes.float().split((hc_mult, hc_mult, hc_mult**2), dim=-1)
    pre = (pre * hc_scale[0] + hc_base[:hc_mult]).sigmoid() + eps
    post = 2 * (post * hc_scale[1] + hc_base[hc_mult : 2 * hc_mult]).sigmoid()
    comb = (comb * hc_scale[2] + hc_base[2 * hc_mult :]).unflatten(
        -1, (hc_mult, hc_mult)
    )
    comb = comb.softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb


def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
    batch, length, heads, dim = q.shape
    maximum = q.new_full((batch, length, heads), -1e30, dtype=torch.float32)
    denominator = torch.zeros_like(maximum)
    numerator = q.new_zeros((batch, length, heads, dim), dtype=torch.float32)
    for start in range(0, topk_idxs.shape[-1], 64):
        indices = topk_idxs[..., start : start + 64].long()
        valid = indices >= 0
        if kv.shape[1]:
            values = kv[
                torch.arange(batch, device=q.device)[:, None, None],
                indices.clamp_min(0),
            ]
        else:
            values = kv.new_zeros((*indices.shape, dim))
        scores = (
            torch.einsum("bshd,bskd->bshk", q.float(), values.float()) * softmax_scale
        )
        scores = scores.masked_fill(~valid.unsqueeze(-2), -torch.inf)
        next_max = torch.maximum(maximum, scores.amax(-1))
        correction = (maximum - next_max).exp()
        probability = (scores - next_max.unsqueeze(-1)).exp()
        numerator = numerator * correction.unsqueeze(-1) + torch.einsum(
            "bshk,bskd->bshd", probability.to(torch.bfloat16).float(), values.float()
        )
        denominator = denominator * correction + probability.sum(-1)
        maximum = next_max
    denominator = denominator + (attn_sink.float() - maximum).exp()
    return (numerator / denominator.unsqueeze(-1)).to(q.dtype)
