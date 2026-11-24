# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Torch references of this entry's kernels: the contract the tests check (never called in serving).

MXFP4: e2m1 codes packed two per byte along K (low nibble first), one e8m0 scale per 32 K. The activation quant
(``quant.mxfp4_quant``, GEMM1's intermediate quant, ``mxfp4_from_mxfp8``) uses Quark's even rule: e8m0 =
exponent(amax + half an e2m1 mantissa ulp) - 2, at least 1; values round to the e2m1 grid {0, 0.5, 1, 1.5, 2, 3, 4, 6}
to nearest, ties to the even code, |v| > 6 saturating to 6. ``mxfp4_quant_ref`` is bitwise what the kernels write for
an f32 / bf16 input.

``w4a4_moe_ref`` is one MoE call on the dequantized operands in f32: per (token, top-k slot) gate/up = x @ W13[e]^T,
h = SwiGLU-OAI(gate, up) (alpha 1.702, limit 7, up + 1) quantized by the same rule, out = sum_k w * (h @ W2[e]^T). The
kernels accumulate in f32 in another order and round each slot to bf16 before the top-k sum, so a few codes of h differ
near rounding ties: the tests bound the relative L2 error (< 4e-3) and check bitwise repeatability separately.
"""

import torch

ALPHA, LIMIT = 1.702, 7.0
_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def dequant_mxfp4(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Row-major MXFP4 ([.., K/2] u8, [.., K/32] e8m0 u8) -> f32 [.., K]."""
    lut = torch.cat([_GRID, -_GRID]).to(codes.device)
    v = torch.stack(
        [lut[(codes & 0xF).long()], lut[(codes >> 4).long()]], dim=-1
    ).flatten(-2)
    return v * torch.exp2(scales.float() - 127.0).repeat_interleave(32, dim=-1)


def mxfp4_quant_ref(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """f32 / bf16 [M, K] -> (codes [M, K/2], e8m0 [M, K/32]), Quark even."""
    m, k = x.shape
    v = x.float().view(m, k // 32, 32)
    amax = v.abs().amax(dim=-1)
    e = ((amax.view(torch.int32) - 0xE00000) >> 23).clamp(min=1)
    a = v.abs() / torch.exp2(e.float() - 127.0).unsqueeze(-1)
    # the code is the number of midpoints below a; a tie goes to the even code: the lower neighbour at
    # 0.25 / 1.25 / 2.5 / 5, the upper one at 0.75 / 1.75 / 3.5
    code = torch.zeros_like(a, dtype=torch.uint8)
    for i, mid in enumerate((0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)):
        code += (a >= mid if i % 2 else a > mid).to(torch.uint8)
    code = (code | torch.signbit(v).to(torch.uint8) << 3).view(m, k)
    packed = code[:, 0::2] | (code[:, 1::2] << 4)
    return packed.contiguous(), e.to(torch.uint8).contiguous()


def swiglu_oai(g: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    g = g.clamp(max=LIMIT)
    u = u.clamp(-LIMIT, LIMIT)
    return g * torch.sigmoid(ALPHA * g) * (u + 1.0)


def w4a4_moe_ref(
    x_quant: tuple[torch.Tensor, torch.Tensor],
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    """f32 [M, H]; weights row-major as stored: ``w13`` [E, 2I, H/2] (gate rows, then up rows), ``w13_scale``
    [E, 2I, H/32], ``w2`` [E, H, I/2], ``w2_scale`` [E, H, I/32]."""
    x = dequant_mxfp4(*x_quant)
    inter = w13.shape[1] // 2
    out = torch.zeros((x.shape[0], w2.shape[1]), device=x.device)
    for e in topk_ids.unique().tolist():
        t, s = torch.nonzero(topk_ids == e, as_tuple=True)
        gu = x[t] @ dequant_mxfp4(w13[e], w13_scale[e]).t()
        h = dequant_mxfp4(*mxfp4_quant_ref(swiglu_oai(gu[:, :inter], gu[:, inter:])))
        y = h @ dequant_mxfp4(w2[e], w2_scale[e]).t()
        out.index_add_(0, t, y * topk_weights[t, s].float().unsqueeze(1))
    return out


def a16w4_moe_ref(x, w13, w13_scale, w2, w2_scale, topk_weights, topk_ids, *, return_partials=False):
    """BF16 activations x MXFP4 weights (checkpoint layouts): FP32 GEMMs, SwiGLU-OAI rounded to BF16.

    Each slot is weighted (routed scale included) in FP32 and rounded once to BF16; the reduced output sums slots 0..3
    in FP32, then rounds to BF16. Tests: cosine >= 0.9999 overall and per token (another FP32 MFMA order).
    """
    m, topk = topk_ids.shape
    inter = w13.shape[1] // 2
    parts = torch.empty((m, topk, w2.shape[1]), device=x.device, dtype=torch.bfloat16)
    for e in topk_ids.unique().tolist():
        t, slot = torch.nonzero(topk_ids == e, as_tuple=True)
        gu = x[t].float() @ dequant_mxfp4(w13[e], w13_scale[e]).t()
        h = swiglu_oai(gu[:, :inter], gu[:, inter:]).bfloat16().float()
        y = h @ dequant_mxfp4(w2[e], w2_scale[e]).t()
        parts[t, slot] = (y * topk_weights[t, slot, None]).bfloat16()
    if return_partials:
        return parts
    out = torch.zeros((m, w2.shape[1]), device=x.device)
    for slot in range(topk):
        out = out + parts[:, slot].float()
    return out.bfloat16()


def router_gemm_partials_ref(x, weight, split_k=16):
    """FP64 split-K router partials of the FP32 weight and their absolute-dot bounds, [split, M, 128].

    The BF16x3 kernel keeps high and low terms in separate FP32 chains: tests bound each partial's absolute error by
    2**-18 times its bound (MFMA rounding is not reproduced) and check bitwise determinism separately.
    """
    x = x.double().reshape(x.shape[0], split_k, -1)
    w = weight.double().reshape(weight.shape[0], split_k, -1)
    exact = torch.stack([x[:, k] @ w[:, k].t() for k in range(split_k)])
    bound = torch.stack([x[:, k].abs() @ w[:, k].abs().t() for k in range(split_k)])
    return exact, bound


TIE_EPS = 1e-6
WEIGHT_RTOL = 2.0**-20  # measured <= 2.6e-7 (2^-21.9)


def splitk_logits_reference(partials: torch.Tensor) -> torch.Tensor:
    """fp32 logits ``[M, E]`` from partials ``[M, S, E]``, summed from 0.0 in split order (the top-k kernels' order)."""
    p = [partials[:, k].float() for k in range(partials.shape[1])]
    out = torch.zeros_like(p[0])
    for k in range(len(p)):
        out = out + p[k]
    return out


def biased_topk_reference(
    logits: torch.Tensor, bias: torch.Tensor, topk: int, scale: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(weights ``[M, topk]`` fp64, ids ``[M, topk]`` int64, sorted choice values ``[M, topk + 1]`` fp64) of the biased
    top-k on fp32 ``logits``: ``s = sigmoid(logit)``, ``choice = s + bias``, the ``topk`` largest choices in descending
    order, weight ``s * scale / sum(s over the chosen)``. The kernels evaluate the sigmoid with the hardware exp2 and
    reciprocal, so ids match where the deciding choices differ by more than ``TIE_EPS`` (``decided``) and weights
    within ``WEIGHT_RTOL`` relative; the ``topk + 1`` largest choices let a test tell a decided id from a near tie."""
    s = torch.sigmoid(logits.double())
    choice = s + bias.double()
    vals, ids = torch.topk(choice, topk + 1, dim=-1, sorted=True)
    chosen = torch.gather(s, 1, ids[:, :topk])
    weights = chosen * scale / chosen.sum(-1, keepdim=True)
    return weights, ids[:, :topk], vals


def decided(vals: torch.Tensor) -> torch.Tensor:
    """``[M, topk]`` bool: the reference id at position j is decided (its choice is more than ``TIE_EPS`` from the
    neighbours j - 1 and j + 1)."""
    gaps = vals[:, :-1] - vals[:, 1:]  # [M, topk]: gap below position j
    above = torch.cat([torch.full_like(gaps[:, :1], float("inf")), gaps[:, :-1]], 1)
    return (gaps > TIE_EPS) & (above > TIE_EPS)


def reduce_slots_bf16_ref(partials):
    """BF16 [M, 4, H] -> BF16 [M, H]: +0, then slots 0..3 in FP32, rounded once.

    Finite results match bit for bit, signed zero and subnormals included; NaN payloads are unspecified.
    """
    acc = torch.zeros_like(partials[:, 0], dtype=torch.float32)
    for k in range(partials.shape[1]):
        acc = acc + partials[:, k].float()
    return acc.to(partials.dtype)


def shared_mxfp8_quant_ref(x):
    """Row-major E4M3 codes and e8m0 ``ceil_pow2(amax / 448)`` (exponent >= 1) per 32, no BF16 rounding first.

    The rule of the shared expert's input and of its FP32 SwiGLU output, at every M.
    """
    v=x.float().reshape(*x.shape[:-1],x.shape[-1]//32,32)
    u=(v.abs().amax(-1)/448).contiguous().view(torch.int32)
    e=(u>>23)&255
    e=(e+(((u&0x7fffff)!=0)&(e<255)).int()).clamp(min=1)
    q=(v/torch.exp2(e.float()-127).unsqueeze(-1)).to(torch.float8_e4m3fn)
    return q.reshape(x.shape).view(torch.uint8),e.to(torch.uint8)


def dequant_shared_mxfp8(q,s):
    return q.view(torch.float8_e4m3fn).float()*torch.exp2(s.view(torch.uint8).float()-127).repeat_interleave(32,-1)


def shared_mxfp8_ref(x,w13,s13,w2,s2,*,quantized=True):
    """The shared expert in FP32 sums (the kernels round the output to BF16); tests: cosine >= .999 per token (FP8
    rounding ties can differ).

    ``quantized=False``: BF16 input and an unquantized FP32 intermediate, for the high-precision comparison.
    """
    x=dequant_shared_mxfp8(*shared_mxfp8_quant_ref(x)) if quantized else x.float()
    gu=x @ dequant_shared_mxfp8(w13,s13).t()
    h=swiglu_oai(gu[:,:768],gu[:,768:])
    if quantized:h=dequant_shared_mxfp8(*shared_mxfp8_quant_ref(h))
    return h @ dequant_shared_mxfp8(w2,s2).t()


def shared_mxfp8_operand_ref(x_quant, w13, s13, w2, s2):
    """``shared_mxfp8_ref`` on an already quantized row-major FP8 / e8m0 input (a gathered SP payload)."""
    x = dequant_shared_mxfp8(*x_quant)
    gu = x @ dequant_shared_mxfp8(w13, s13).t()
    h = swiglu_oai(gu[:, :768], gu[:, 768:])
    h = dequant_shared_mxfp8(*shared_mxfp8_quant_ref(h))
    return h @ dequant_shared_mxfp8(w2, s2).t()


def _mxfp4_via_shared_codes_ref(s4,q8,s8):
    """The FP4 codes of the FP8 values at the FP4 scales (computed in FP8 units, no FTZ), and their magnitudes."""
    ratio=torch.exp2((s4.int()-s8.int()).float()).repeat_interleave(32,-1)
    values=q8.view(torch.float8_e4m3fn).float()
    magnitude=values.abs()/ratio
    codes=torch.zeros_like(q8)
    for j,mid in enumerate((.25,.75,1.25,1.75,2.5,3.5,5.)):
        codes+=(magnitude>=mid if j%2 else magnitude>mid).to(torch.uint8)
    return codes|((q8>>7)<<3),magnitude


def mxfp4_sidecar_pack_ref(q4,s4,q8,s8):
    """The mask, row-major [M, 768] (bit i % 8 is value i, low bit first): set where the direct FP4 code differs from
    the FP4 rounding of the shared FP8 code.

    Both inputs come from the same finite BF16 rows through this entry's quantizers, each with its own scales.
    """
    candidate,_=_mxfp4_via_shared_codes_ref(s4,q8,s8)
    direct=torch.stack((q4&15,q4>>4),-1).flatten(-2)
    bits=(candidate!=direct).reshape(q8.shape[0],-1,8).long()
    return (bits*(1<<torch.arange(8,device=q4.device))).sum(-1).to(torch.uint8)


def mxfp4_sidecar_unpack_ref(mask,s4,q8,s8):
    """The direct FP4 bytes from the mask, byte for byte (signed zero included), with no new scale.

    The FP8 double rounding differs only at FP4 midpoints, where RNE takes the upper code at .75 / 1.75 / 3.5 and the
    lower one elsewhere; a set bit takes the other neighbour. Unrelated FP4 / FP8 pairs are outside the contract.
    """
    candidate,magnitude=_mxfp4_via_shared_codes_ref(s4,q8,s8)
    bits=(mask.unsqueeze(-1).int()>>torch.arange(8,device=mask.device))&1
    down=(magnitude==.75)|(magnitude==1.75)|(magnitude==3.5)
    other=(candidate&8)|(((candidate&7).int()+torch.where(down,-1,1))&7).to(torch.uint8)
    codes=torch.where(bits.flatten(-2).bool(),other,candidate)
    return (codes[:,::2]|(codes[:,1::2]<<4)).contiguous()
