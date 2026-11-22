# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""ATOM_MTP_DRAFT_LM_HEAD_FP8: the MTP draft passes' LM head in 128x128-block FP8.

A draft pass reads the whole bf16 LM head (248320 x 2048, 1 GB) to pick one
argmax a sequence. The draft's pick only decides what the verifier is offered:
the verify rows keep the bf16 head and sample the target distribution, so the
output distribution does not depend on it -- only how often a draft is accepted.
This keeps a second copy of the head as e4m3 with e8m0 128 x 128 block scales
(`quant_weight_online`, the FP8 projections' scheme), preshuffled for aiter's
blockscale GEMM, and computes draft logits from it (activations quantized per
1 x 128 with e8m0 scales). Half the bytes a pass."""

import torch
from aiter import QuantType, dtypes
from aiter.ops.shuffle import shuffle_weight


class DraftLMHeadFP8:
    def __init__(self, weight: torch.Tensor):
        from atom.quantization.quark.utils import quant_weight_online

        assert (
            weight.dim() == 2
            and weight.shape[0] % 128 == 0
            and weight.shape[1] % 128 == 0
        )
        wq, ws = quant_weight_online(
            weight, QuantType.per_1x128, dtypes.fp8, e8m0_scale=True
        )
        self.weight = shuffle_weight(wq, layout=(16, 16))
        self.weight_scale = ws

    def mm_stats(self, x: torch.Tensor, temperatures, presence, rng_state):
        """ATOM_MTP_DRAFT_LMHEAD_STATS: ``mm(x)`` and, from the GEMM's
        epilogue, the draft sampler's records (int32 [rows, vocab / 256, 8]:
        group keys, softmax partials; aiter gemm_a8w8_lmhead_stats), the draw
        counter advanced by rows. None when the stats GEMM does not take x."""
        from aiter.ops.flydsl.gemm_a8w8_lmhead_stats import (
            gemm_a8w8_lmhead_stats,
            lmhead_stats_supported,
        )

        rows, k = x.shape
        n = self.weight.shape[0]
        if not lmhead_stats_supported(rows, n, k):
            return None
        mask = presence["mask"] if presence is not None else None
        if mask is not None and (
            mask.dtype != torch.uint8
            or mask.stride(1) != 1
            or mask.stride(0) % 16
            or mask.data_ptr() % 16
            or presence.get("anchor", False)
        ):
            return None
        from atom.model_ops.linear import per_1x128_e8m0_quant

        from atom.model_ops import draft_tail_quant

        pre = draft_tail_quant.take_prepared(x) if draft_tail_quant.ENABLED else None
        if pre is not None:
            xq, xs = pre
        else:
            xq, xs = per_1x128_e8m0_quant(
                x.contiguous(), dtypes.fp8, transpose_scale=True
            )
        logits = torch.empty((rows, n), dtype=torch.bfloat16, device=x.device)
        rec = torch.empty((rows, n // 256, 8), dtype=torch.int32, device=x.device)
        gemm_a8w8_lmhead_stats(
            xq,
            self.weight,
            xs,
            self.weight_scale,
            logits,
            temperatures,
            rec,
            presence,
            rng_state,
        )
        return logits, rec

    def mm(self, x: torch.Tensor, prepared: bool = True) -> torch.Tensor:
        """bf16 `[rows, vocab]` logits of `x` `[rows, hidden]` bf16. ``prepared=False``
        (the verifier's head, ATOM_VERIFY_LM_HEAD_FP8): never takes the draft tail
        norm's prepared quant."""
        from atom.model_ops.linear import (
            gemm_a8w8_blockscale_preshuffle_impl,
            per_1x128_e8m0_quant,
        )

        from atom.model_ops import draft_tail_quant

        # ATOM_MTP_TAIL_LMHEAD_QUANT: the quant the final norm's launch wrote, if any.
        pre = (
            draft_tail_quant.take_prepared(x)
            if prepared and draft_tail_quant.ENABLED
            else None
        )
        if pre is not None:
            xq, xs = pre
        else:
            xq, xs = per_1x128_e8m0_quant(
                x.contiguous(), dtypes.fp8, transpose_scale=True
            )
        return gemm_a8w8_blockscale_preshuffle_impl(
            xq, self.weight, xs, self.weight_scale, torch.bfloat16
        )
