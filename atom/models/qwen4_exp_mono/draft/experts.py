"""Compact q1 descriptors, retaining the selected Native BF16 CK GEMMs."""

import torch
import triton
import triton.language as tl
from aiter import ActivationType, QuantType
from aiter.fused_moe import get_2stage_cfgs, get_padded_M


@triton.jit
def publish_q1(
    IDS,
    WEIGHTS,
    SORTED_IDS,
    SORTED_WEIGHTS,
    EXPERTS,
    VALID,
    OUT,
    TOPK: tl.constexpr,
    BM: tl.constexpr,
    H: tl.constexpr,
    B: tl.constexpr,
):
    i = tl.arange(0, B)
    slot = i // BM
    active = i < TOPK * BM
    first = i % BM == 0
    expert = tl.load(IDS + slot, active, 0)
    routed = expert >= 0
    # CK decodes token from low 24 bits, routing slot from the high byte.
    # Native routing marks padding with expert -1. Keep an empty CK tile for
    # that slot: token 1 is out of range for q1, and its weight base must be valid.
    tl.store(SORTED_IDS + i, (slot << 24) | tl.where(first & routed, 0, 1), active)
    weight = tl.load(WEIGHTS + slot, active & first & routed, 0.0)
    tl.store(SORTED_WEIGHTS + i, weight, active)
    tl.store(EXPERTS + slot, tl.maximum(expert, 0), active & first)
    tl.store(VALID, TOPK * BM)
    tl.store(VALID + 1, 1)
    for base in tl.static_range(tl.cdiv(H, B)):
        col = base * B + i
        tl.store(OUT + col, 0.0, col < H)


class NativeExperts:
    def __init__(self, w13, w2, topk):
        self.w13, self.w2, self.topk = w13, w2, topk
        assert w13.is_shuffled and w2.is_shuffled
        assert w13.shape == (512, 1280, 2560) and w2.shape == (512, 2560, 640)
        assert w13.dtype == w2.dtype == torch.bfloat16
        self.meta = get_2stage_cfgs(
            get_padded_M(1),
            2560,
            640,
            512,
            topk,
            torch.bfloat16,
            torch.bfloat16,
            torch.bfloat16,
            QuantType.No,
            True,
            ActivationType.Silu,
            False,
            0,
            0,
            True,
            opus_weights_shuffled=True,
            input_dtype=torch.bfloat16,
        )
        m = self.meta
        # Native metadata sets prequant even for QuantType.No; its quantizer
        # is the identity (x, None), both before G1 and between G1/G2.
        assert not m.run_1stage and not m.fuse_quant
        assert getattr(m.stage1, "func", m.stage1).__name__ == "ck_moe_stage1"
        assert getattr(m.stage2, "func", m.stage2).__name__ == "ck_moe_stage2_fwd"
        self.bm = int(m.block_m)
        dev = w13.device
        self.ids = torch.empty(topk * self.bm, device=dev, dtype=torch.int32)
        self.weights = torch.empty(topk * self.bm, device=dev, dtype=torch.float32)
        self.experts = torch.empty(topk, device=dev, dtype=torch.int32)
        self.valid = torch.empty(2, device=dev, dtype=torch.int32)
        self.mid = torch.empty((1, topk, 640), device=dev, dtype=torch.bfloat16)
        self.out = torch.empty((1, 2560), device=dev, dtype=torch.bfloat16)

    def __call__(self, x, ids, weights):
        assert x.shape == (1, 2560)
        publish_q1[(1,)](
            ids,
            weights,
            self.ids,
            self.weights,
            self.experts,
            self.valid,
            self.out,
            self.topk,
            self.bm,
            2560,
            triton.next_power_of_2(self.topk * self.bm),
            num_warps=4,
        )
        self.meta.stage1(
            x,
            self.w13,
            self.w2,
            self.ids,
            self.experts,
            self.valid,
            self.mid,
            self.topk,
            block_m=self.bm,
            a1_scale=None,
            w1_scale=None,
            sorted_weights=None,
        )
        self.meta.stage2(
            self.mid,
            self.w13,
            self.w2,
            self.ids,
            self.experts,
            self.valid,
            self.out,
            self.topk,
            block_m=self.bm,
            a2_scale=None,
            w2_scale=None,
            sorted_weights=self.weights,
        )
        return self.out
