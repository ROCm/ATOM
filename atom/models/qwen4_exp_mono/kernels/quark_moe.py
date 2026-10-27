# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Tiny-row Quark routing and MXFP4 preparation for native FlyDSL experts.

Each selected (token, expert) owns one M32 tile. Expert ids may repeat across
tiles, just as they do when an expert spans several tiles in native sorting.
The packed token id, route slot, BF16 output zero and sorted E8M0 scales follow
AITER's two-stage ABI. No expert weights are repacked here.

Within one launch, each token uses one CTA for top-k/route metadata and five
CTAs for disjoint 512-element quantization/scale/output-clear tiles. These
writes are independent; the subsequent G1 launch consumes all of them.
"""

from dataclasses import dataclass, replace
from functools import partial

import torch
import triton
import triton.language as tl


def validate_native_config(rows):
    """Reject a different consumer ABI before any recurrent state is updated."""
    from aiter import ActivationType, QuantType, dtypes
    from aiter.fused_moe import GateMode, get_2stage_cfgs, get_padded_M

    from atom.mono.runtime.consensus import MonoUnsupported

    cfg = get_2stage_cfgs(
        get_padded_M(rows),
        2560,
        640,
        513,
        11,
        torch.bfloat16,
        dtypes.fp4x2,
        dtypes.fp4x2,
        QuantType.per_1x32,
        True,
        ActivationType.Silu,
        False,
        0,
        0,
        True,
        GateMode.SEPARATED,
        opus_weights_shuffled=True,
        input_dtype=torch.bfloat16,
    )
    if (
        cfg.block_m != 32
        or cfg.run_1stage
        or cfg.output_aux
        or cfg.ksplit > 1
        or cfg.full_impl is not None
        or not cfg.prequant
        or cfg.stage1.func.__name__ != "_flydsl_stage1_wrapper"
    ):
        raise MonoUnsupported(f"Quark route/quant consumer ABI changed: {cfg}")


def expert_config(cfg):
    """Use N64 G1 with the native BF16 output and separate MXFP4 quantization."""
    return replace(
        cfg,
        stage1=partial(
            cfg.stage1.func,
            **{
                **cfg.stage1.keywords,
                "kernelName": "flydsl_moe1_afp4_wfp4_bf16_t32x64x256_w2",
            },
        ),
        fuse_quant="",
    )


@triton.jit
def _prepare(
    logits,
    x,
    ids,
    weights,
    sorted_ids,
    sorted_weights,
    experts,
    valid,
    quant,
    scales,
    out,
    ROWS: tl.constexpr,
):
    row = tl.program_id(0)
    slot = tl.arange(0, 16)
    lane = tl.arange(0, 32)
    if tl.program_id(1) == 0:
        e = tl.arange(0, 512)
        z = tl.load(logits + row * 513 + e).to(tl.float32)
        # Float-order keys, with lower expert id winning an equal-logit tie.
        bits = z.to(tl.bfloat16).to(tl.int16, bitcast=True).to(tl.int32)
        ordered = bits ^ ((bits >> 31) & 0x7FFF)
        ordered = tl.where(z == 0, 0, ordered)
        key = (ordered << 9) | (511 - e)
        # Only the first ten routes are consumed; top-16 retains their order.
        ranked = tl.topk(key, 16, descending=True)
        selected = ranked
        eid = 511 - (selected & 511)
        value = tl.load(logits + row * 513 + eid, slot < 10, 0).to(tl.float32)
        largest = tl.max(tl.where(slot < 10, value, float("-inf")), 0)
        prob = tl.exp(value - largest)
        prob = tl.where(slot < 10, prob, 0.0)
        prob = prob / tl.sum(prob, 0)
        shared = tl.load(logits + row * 513 + 512).to(tl.float32)
        eid = tl.where(slot == 10, 512, eid)
        prob = tl.where(slot == 10, 1.0 / (1.0 + tl.exp(-shared)), prob)
        tl.store(ids + row * 11 + slot, eid, slot < 11)
        tl.store(weights + row * 11 + slot, prob, slot < 11)
        tl.store(experts + row * 11 + slot, eid, slot < 11)
        offset = (row * 11 + slot[:, None]) * 32 + lane[None, :]
        packed = (slot[:, None] << 24) | row
        tl.store(
            sorted_ids + offset,
            tl.where(lane[None, :] == 0, packed, (11 << 24) | ROWS),
            slot[:, None] < 11,
        )
        tl.store(
            sorted_weights + offset,
            tl.where(lane[None, :] == 0, prob[:, None], 0.0),
            slot[:, None] < 11,
        )
        if row == 0:
            tl.store(valid, ROWS * 11 * 32)
            tl.store(valid + 1, ROWS)

    else:
        group = (tl.program_id(1) - 1) * 16 + tl.arange(0, 16)
        col = group[:, None] * 32 + lane[None, :]
        values = tl.load(x + row * 2560 + col, col < 2560, 0).to(tl.float32)
        amax = tl.maximum(tl.max(tl.abs(values), 1), 1.0e-10)
        # Native HIP quant uses RoundUp: ceil_pow2(amax / 6).
        divided = amax / 6.0
        rounded = divided.to(tl.int32, bitcast=True) + 0x7FFFFF
        exponent = tl.minimum(tl.maximum(rounded >> 23, 0), 254)
        inv = tl.exp2((127 - exponent).to(tl.float32))
        scaled = values * inv[:, None]
        # E2M1 round-to-nearest-even, including the sign of zero.
        a = tl.abs(scaled)
        code = (
            (a > 0.25).to(tl.int32)
            + (a >= 0.75).to(tl.int32)
            + (a > 1.25).to(tl.int32)
            + (a >= 1.75).to(tl.int32)
            + (a > 2.5).to(tl.int32)
            + (a >= 3.5).to(tl.int32)
            + (a > 5.0).to(tl.int32)
        )
        code |= (scaled.to(tl.int32, bitcast=True) >> 28) & 8
        even, odd = tl.split(code.reshape(16, 16, 2))
        qcol = group[:, None] * 16 + tl.arange(0, 16)[None, :]
        tl.store(quant + row * 1280 + qcol, even | (odd << 4), qcol < 1280)
        # AITER E8M0 tile: [M/32, K/256, K%4, M%16, K%8/4, M%32/16].
        soff = (row * 11 + slot[:, None]) * 2560 + (group[None, :] // 8) * 256
        soff += (group[None, :] % 4) * 64 + (group[None, :] % 8 // 4) * 2
        tl.store(
            scales + soff,
            exponent[None, :],
            (slot[:, None] < 11) & (group[None, :] < 80),
        )
        c = group[:, None] * 32 + lane[None, :]
        tl.store(out + row * 2560 + c, 0.0, c < 2560)


@dataclass
class Workspace:
    ids: torch.Tensor
    weights: torch.Tensor
    sorted_ids: torch.Tensor
    sorted_weights: torch.Tensor
    experts: torch.Tensor
    valid: torch.Tensor
    quant: torch.Tensor
    scales: torch.Tensor
    out: torch.Tensor

    @classmethod
    def allocate(cls, rows, device):
        assert rows in (1, 2, 4)

        def buf(shape, dtype):
            return torch.empty(shape, dtype=dtype, device=device)

        return cls(
            buf((rows, 11), torch.int32),
            buf((rows, 11), torch.float32),
            buf((rows * 11 * 32,), torch.int32),
            buf((rows * 11 * 32,), torch.float32),
            buf((rows * 11,), torch.int32),
            buf((2,), torch.int32),
            buf((rows, 1280), torch.uint8),
            buf((rows * 11 * 32, 80), torch.uint8),
            buf((rows, 2560), torch.bfloat16),
        )

    def prepare(self, logits, x):
        _prepare[(x.shape[0], 6)](
            logits,
            x,
            self.ids,
            self.weights,
            self.sorted_ids,
            self.sorted_weights,
            self.experts,
            self.valid,
            self.quant,
            self.scales,
            self.out,
            x.shape[0],
            num_warps=4,
        )

    def experts_forward(self, x, w1, w2, s1, s2):
        from aiter import ActivationType, QuantType, dtypes
        from aiter.fused_moe import fused_moe_2stages

        return fused_moe_2stages(
            x,
            w1,
            w2,
            11,
            self.sorted_ids,
            self.sorted_weights,
            self.experts,
            self.valid,
            self.out,
            True,
            32,
            activation=ActivationType.Silu,
            quant_type=QuantType.per_1x32,
            q_dtype_a=dtypes.fp4x2,
            q_dtype_w=dtypes.fp4x2,
            w1_scale=s1,
            w2_scale=s2,
            topk_ids=self.ids,
            topk_weights=self.weights,
            routing_num_experts=513,
            prequantized_a1=self.quant.view(dtypes.fp4x2),
            prequantized_a1_scale=self.scales.view(dtypes.fp8_e8m0),
            _metadata_transform=expert_config,
        )
