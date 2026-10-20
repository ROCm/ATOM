# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""``lm_head`` from MXFP4 weights on decode steps.

The converter rounds ``lm_head`` to MXFP4 values and keeps it in bf16
(``tools/qwen38_flash_next/convert_mxfp4.py``), so the same weights packed
as MXFP4 are exact. ``Mxfp4LMHead`` packs them once, when the mono is installed,
and checks the round trip bit for bit; a decode step's head then reads 0.34 GB
instead of 1.27 GB. The activations stay bf16 and the products accumulate in
fp32 (``tl.dot_scaled``, bf16 x e2m1), so the logits differ from the bf16
GEMM's only in summation order. The target's verify head and the drafter's
head share the module (the MTP checkpoint has no ``lm_head`` of its own), so
both take it. Prefill, and a head of more than ``MAX_HEAD_ROWS`` rows, keep
the bf16 GEMM. A greedy draft's token (``compute_argmax_token``) is folded
from the logits as the head computes them (``mxfp4_argmax``): two launches
against the head and ``topk_select``'s three, the logits never stored.
"""

import logging

import torch
import triton
import triton.language as tl
from torch import nn

from atom.utils.forward_context import get_forward_context

logger = logging.getLogger("atom")

# the most rows a head runs from MXFP4, in one block: the mono's widest head
# is 16 (4 requests, 3 drafts). A second block reads the weight again, and
# at 64 rows the bf16 GEMM is faster (347 us against 290)
MAX_HEAD_ROWS = 16
BK = 256
# the widest BN; ``_tile`` picks the narrow one for a head of few rows
BN = 128
# E2M1 code of |value| / scale: thresholds between the magnitudes
# 0, 0.5, 1, 1.5, 2, 3, 4, 6 (the converter's ``FP4_THRESHOLDS``)
_THRESHOLDS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
_FP4_VALUES = (0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6)


@triton.jit
def _mxfp4_head(
    x_ptr,
    w_ptr,
    s_ptr,
    y_ptr,
    M,
    sxm,
    sym,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    ARGMAX: tl.constexpr,
):
    """y[M, N] = x[M, K] (bf16) @ w[N, K]^T for M <= BM, w packed E2M1 (two
    codes a byte, the even k low) with one E8M0 scale per 32 k; fp32
    accumulation.

    With ``ARGMAX`` it stores no logits: a program folds its columns into
    one key a row, the bf16 logit's bits ordered as an integer above the
    column's distance from the last, and stores it to ``y`` (int64
    ``[M, programs]``) for ``_argmax_fold``. No atomics: a gpu-scope one
    writes back and invalidates the L2 around it, in every program (304 us
    a head, against 72)."""
    rm = tl.arange(0, BM)
    rn = tl.program_id(0) * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in tl.range(0, K, BK):
        rk = k0 + tl.arange(0, BK)
        a = tl.load(
            x_ptr + rm[:, None] * sxm + rk[None, :], mask=rm[:, None] < M, other=0.0
        )
        rkw = k0 // 2 + tl.arange(0, BK // 2)
        # read once a step: streamed (non-temporal loads)
        w = tl.load(w_ptr + rn[:, None] * (K // 2) + rkw[None, :], cache_modifier=".cg")
        rks = k0 // 32 + tl.arange(0, BK // 32)
        s = tl.load(
            s_ptr + rn[:, None] * (K // 32) + rks[None, :], cache_modifier=".cg"
        )
        # fast_math skips the NaN scale (0xFF): ``_pack`` writes 1..254
        acc = tl.dot_scaled(
            a, None, "bf16", tl.trans(w), s, "e2m1", acc, fast_math=True
        )
    if not ARGMAX:
        tl.store(
            y_ptr + rm[:, None] * sym + rn[None, :],
            acc.to(tl.bfloat16),
            mask=rm[:, None] < M,
        )
    else:
        last = tl.num_programs(0) * BN - 1
        # the bf16 logits, -0 as +0: a float's order as its bits' signed order
        v = acc.to(tl.bfloat16).to(tl.float32)
        bits = tl.where(v == 0, 0, v.to(tl.int32, bitcast=True))
        key = bits ^ ((bits >> 31) & 0x7FFFFFFF)
        top = tl.max(key, 1)
        col = tl.min(tl.where(key == top[:, None], rn[None, :], last), 1)
        packed = (top.to(tl.int64) << 32) | (last - col).to(tl.int64)
        at = rm * tl.num_programs(0) + tl.program_id(0)
        tl.store(y_ptr + at, packed, mask=rm < M)


@triton.jit
def _argmax_fold(keys_ptr, out_ptr, P, last, BLOCK: tl.constexpr):
    """``out[row]`` (int32): the column of the largest of ``row``'s keys from
    ``P`` programs of ``_mxfp4_head``'s ``ARGMAX``; the lowest on a tie (its
    distance from ``last`` the largest)."""
    row = tl.program_id(0)
    best = tl.full((BLOCK,), -(2**63), tl.int64)
    for p0 in range(0, P, BLOCK):
        p = p0 + tl.arange(0, BLOCK)
        k = tl.load(keys_ptr + row * P + p, mask=p < P, other=-(2**63))
        best = tl.maximum(best, k)
    won = tl.max(best, 0) & 0xFFFFFFFF
    tl.store(out_ptr + row, (last - won).to(tl.int32))


def _pack(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """bf16 ``[N, K]`` -> (E2M1 uint8 ``[N, K/2]``, E8M0 uint8 ``[N, K/32]``),
    the converter's ``quantize``."""
    n, k = weight.shape
    x = weight.float().view(n, k // 32, 32)
    bits = (x.abs().amax(-1) / 6).clamp_min(1e-12).view(torch.int32)
    exponent = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    exponent = exponent.clamp(1, 254)
    scale = (exponent << 23).view(torch.float32)
    magnitude = (x / scale.unsqueeze(-1)).abs()
    codes = torch.zeros_like(magnitude, dtype=torch.uint8)
    for threshold in _THRESHOLDS:
        codes += (magnitude > threshold).to(torch.uint8)
    negative = (x.view(torch.int32) < 0) & (codes > 0)
    codes = (codes | (negative.to(torch.uint8) << 3)).view(n, k)
    return codes[:, 0::2] | (codes[:, 1::2] << 4), exponent.to(torch.uint8)


def _unpack(packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """The converter's ``dequantize``, to bf16."""
    values = torch.tensor(_FP4_VALUES, dtype=torch.float32, device=packed.device)
    codes = torch.stack([packed & 15, packed >> 4], dim=-1).view(packed.shape[0], -1)
    scale = (scales.to(torch.int32) << 23).view(torch.float32)
    out = values[codes.long()].view(packed.shape[0], -1, 32) * scale.unsqueeze(-1)
    return out.view(packed.shape[0], -1).to(torch.bfloat16)


def pack_exact(
    weight: torch.Tensor, rows: int = 8192
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """``weight`` as MXFP4, or None when it is not exactly MXFP4-valued.
    Packed ``rows`` at a time: the fp32 temporaries of the whole head are
    several GB."""
    n, k = weight.shape
    packed = torch.empty(n, k // 2, dtype=torch.uint8, device=weight.device)
    scales = torch.empty(n, k // 32, dtype=torch.uint8, device=weight.device)
    for r in range(0, n, rows):
        p, s = _pack(weight[r : r + rows])
        if not torch.equal(
            _unpack(p, s).view(torch.int16), weight[r : r + rows].view(torch.int16)
        ):
            return None
        packed[r : r + rows] = p
        scales[r : r + rows] = s
    return packed, scales


def head_refusal(head: nn.Module) -> str | None:
    """Why ``head`` cannot run from MXFP4, or None."""
    weight = getattr(head, "weight", None)
    if weight is None or weight.dim() != 2 or weight.dtype != torch.bfloat16:
        return "lm_head is not a bf16 [vocab, hidden] weight"
    if getattr(head, "bias", None) is not None:
        return "lm_head has a bias"
    if getattr(head, "tp_size", 1) != 1:
        return "lm_head is vocab-parallel"
    n, k = weight.shape
    if k % BK or n % BN:
        return f"lm_head [{n}, {k}] is not a multiple of the tile [{BN}, {BK}]"
    return None


def _tile(rows: int) -> tuple[int, int]:
    """(BN, warps) of a head of ``rows`` rows. HBM-cold on MI355X, the
    [248320, 2560] head takes 71.9 / 73.6 us at 1 / 4 rows with (64, 4) and
    80.6 us at 16 with (128, 8), against 195.8 / 198.4 / 213.1 us bf16
    (``tgemm.mm``)."""
    return (64, 4) if rows <= 4 else (BN, 8)


def mxfp4_head(
    x: torch.Tensor, weight: torch.Tensor, scales: torch.Tensor
) -> torch.Tensor:
    """bf16 logits ``[M, N]`` of ``x [M, K]`` and the packed head."""
    m, k = x.shape
    n = weight.shape[0]
    assert m <= MAX_HEAD_ROWS
    if x.stride(-1) != 1:
        x = x.contiguous()
    y = torch.empty(m, n, dtype=torch.bfloat16, device=x.device)
    _launch(x, weight, scales, y, False)
    return y


def _launch(x, weight, scales, y, argmax) -> int:
    """``_mxfp4_head`` into ``y``; its number of programs."""
    m, k = x.shape
    n = weight.shape[0]
    bn, warps = _tile(m)
    _mxfp4_head[(n // bn,)](
        x,
        weight,
        scales,
        y,
        m,
        x.stride(0),
        y.stride(0),
        K=k,
        BM=MAX_HEAD_ROWS,
        BN=bn,
        BK=BK,
        ARGMAX=argmax,
        num_warps=warps,
        num_stages=2,
    )
    return n // bn


def argmax_keys(weight: torch.Tensor) -> torch.Tensor:
    """``mxfp4_argmax``'s scratch for the packed head ``weight``."""
    programs = weight.shape[0] // min(_tile(r)[0] for r in (1, MAX_HEAD_ROWS))
    return torch.empty(
        MAX_HEAD_ROWS * programs, dtype=torch.int64, device=weight.device
    )


def mxfp4_argmax(x, weight, scales, keys, out) -> torch.Tensor:
    """``out`` (int32 [M]): each row of ``x [M, K]``'s lowest index of its
    largest bf16 logit, as ``topk_select(mxfp4_head(...), 1, tie="low")``, in
    two launches and the logits never stored; ``keys`` the scratch
    (``argmax_keys``)."""
    m = x.shape[0]
    assert m <= MAX_HEAD_ROWS and out.dtype == torch.int32 and out.is_contiguous()
    if x.stride(-1) != 1:
        x = x.contiguous()
    programs = _launch(x, weight, scales, keys, True)
    _argmax_fold[(m,)](keys, out, programs, weight.shape[0] - 1, BLOCK=1024)
    return out


class Mxfp4LMHead(nn.Module):
    """``head`` (a TP1 ``ParallelLMHead`` without bias) with its decode rows
    run from MXFP4 (``pack_exact``); every other call passes to it."""

    def __init__(self, head: nn.Module, packed: torch.Tensor, scales: torch.Tensor):
        super().__init__()
        self.head = head
        self.register_buffer("fp4_weight", packed, persistent=False)
        self.register_buffer("fp4_scales", scales, persistent=False)
        self.register_buffer("argmax_keys", argmax_keys(packed), persistent=False)

    def _fp4_rows(self, x: torch.Tensor) -> bool:
        return x.dim() == 2 and x.shape[0] <= MAX_HEAD_ROWS

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        context = get_forward_context().context
        # a target prefill's head picks each sequence's last row first
        if (context.is_prefill and not context.is_draft) or not self._fp4_rows(x):
            return self.head(x)
        return mxfp4_head(x, self.fp4_weight, self.fp4_scales)

    def compute_argmax_token(
        self, x: torch.Tensor, *, out: torch.Tensor
    ) -> torch.Tensor:
        """``ParallelLMHead.compute_argmax_token`` on the MXFP4 logits: the
        lowest index of the largest, the logits never stored."""
        if not self._fp4_rows(x):
            return self.head.compute_argmax_token(x, out=out)
        assert out.shape == x.shape[:-1] and out.dtype == torch.int32
        return mxfp4_argmax(x, self.fp4_weight, self.fp4_scales, self.argmax_keys, out)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.head, name)


def install_fp4_head(model: nn.Module, drafter) -> bool:
    """Swap ``model.lm_head`` for its ``Mxfp4LMHead``, and the drafter's model's
    when it shares the module; False (and the reason logged) when the head
    cannot run from MXFP4. Call it before the drafter's model is wrapped."""
    head = getattr(model, "lm_head", None)
    why = "the model has no lm_head" if head is None else head_refusal(head)
    if why is None:
        with torch.no_grad():
            packed = pack_exact(head.weight)
        if packed is None:
            why = "lm_head is not exactly MXFP4-valued"
    if why is not None:
        logger.info("Qwen3.8 mono MXFP4 lm_head off: %s", why)
        return False
    fp4 = Mxfp4LMHead(head, *packed)
    model.lm_head = fp4
    draft = getattr(drafter, "model", None)
    shared = draft is not None and getattr(draft, "lm_head", None) is head
    if shared:
        draft.lm_head = fp4
    logger.info(
        "Qwen3.8 mono MXFP4 lm_head on (%s), %.0f MB",
        "target and drafter" if shared else "target",
        (packed[0].numel() + packed[1].numel()) / 1e6,
    )
    return True
