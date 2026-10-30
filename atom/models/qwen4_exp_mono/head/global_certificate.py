"""Same Group64 packed weights and complete-head consumers, global certificate."""

import torch
import triton
from aiter import topk_select

import atom.models.qwen4_exp_mono.head.packed as frozen
from atom.models.qwen4_exp_mono.head.global_kernels import (
    aggregate_input_certificate,
    screen_global_certificate,
)


def aggregate_weight_certificate(group_certificate):
    # Input FP32 certificates are immutable upper bounds. FP64 squares are
    # exact, and 2^-40 dominates gamma_128 for the sum/sqrt/product. The final
    # nextafter covers conversion to FP32 and its rounded midpoint.
    value = group_certificate.double().square().sum(dim=1).sqrt() * (1.0 + 2.0**-40)
    value = torch.where(
        torch.isfinite(value) & (value >= 0),
        value,
        torch.full_like(value, float("inf")),
    ).float()
    return torch.nextafter(value, torch.full_like(value, float("inf")))


class HeadScreenGlobalCertificate:
    def __init__(self, prepared_w4):
        self.owner = prepared_w4
        for name in ("weight", "n", "k", "packed", "ws", "we", "wqnorm", "wnorm"):
            setattr(self, name, getattr(prepared_w4, name))
        assert (self.n, self.k) == (248320, 2560)
        self.we_global = aggregate_weight_certificate(self.we)
        self.wq_global = aggregate_weight_certificate(self.wqnorm)
        device = self.weight.device

        def alloc(*shape, dtype=torch.float32):
            return torch.empty(shape, device=device, dtype=dtype)

        groups = self.k // frozen.GROUP
        blocks = triton.cdiv(self.n, frozen.BLOCK_N)
        self.qeven = alloc(self.k // 8, dtype=torch.uint32)
        self.qodd = alloc(self.k // 8, dtype=torch.uint32)
        self.xscale, self.xstats = alloc(groups), alloc(groups, 3)
        self.xglobal = alloc(3)
        self.xsum = alloc(groups, dtype=torch.int32)
        self.approx = alloc(1, self.n, dtype=torch.bfloat16)
        self.error, self.refined = alloc(1, self.n), alloc(1, self.n)
        self.lower, self.best_y = alloc(blocks), alloc(blocks)
        self.best_id = alloc(blocks, dtype=torch.int32)
        self.threshold, self.anchor_value, self.anchor_bound = (
            alloc(1),
            alloc(1),
            alloc(1),
        )
        self.anchor = alloc(1, dtype=torch.int32)
        self.counts = alloc(1, triton.cdiv(self.n, 16), dtype=torch.int32)
        self.row_lens = torch.full((1,), self.n, device=device, dtype=torch.int32)
        self.ids = alloc(1, dtype=torch.int32)
        self.last_stats = self.last_screen = self.last_anchor = self.last_refine = None

    def prepare(self, x):
        assert x.shape == (1, self.k) and x.dtype == torch.bfloat16
        assert x.is_contiguous() and x.device == self.weight.device
        frozen.quant_groups[(self.k // frozen.GROUP,)](
            x,
            self.qeven,
            self.qodd,
            self.xscale,
            self.xstats,
            self.xsum,
            self.k,
            frozen.GROUP,
            num_warps=4,
        )
        self.last_stats = aggregate_input_certificate[(1,)](
            x,
            self.xscale,
            self.xstats,
            self.xglobal,
            self.k,
            self.k // frozen.GROUP,
            triton.next_power_of_2(self.k),
            triton.next_power_of_2(self.k // frozen.GROUP),
            num_warps=4,
        )
        blocks = triton.cdiv(self.n, frozen.BLOCK_N)
        self.last_screen = screen_global_certificate[(blocks,)](
            self.qeven,
            self.qodd,
            self.xscale,
            self.xglobal,
            self.xsum,
            self.packed,
            self.ws,
            self.we_global,
            self.wq_global,
            self.wnorm,
            self.approx,
            self.error,
            self.lower,
            self.best_y,
            self.best_id,
            self.n,
            self.k,
            frozen.GROUP,
            frozen.BLOCK_N,
            frozen.BLOCK_K,
            num_warps=4,
        )

    def run(self, x, *, out=None):
        answer = self.ids if out is None else out
        assert answer.shape == (1,) and answer.dtype == torch.int32
        assert answer.is_contiguous() and answer.device == x.device
        self.prepare(x)
        blocks = triton.cdiv(self.n, frozen.BLOCK_N)
        self.last_anchor = frozen.anchor_lower[(1,)](
            x,
            self.weight,
            self.wnorm,
            self.lower,
            self.best_y,
            self.best_id,
            self.threshold,
            self.anchor,
            self.anchor_value,
            self.anchor_bound,
            self.n,
            self.k,
            frozen.GROUP,
            blocks,
            triton.next_power_of_2(blocks),
            triton.next_power_of_2(self.k),
            num_warps=8,
        )
        self.last_refine = frozen.refine[(triton.cdiv(self.n, 16), 1)](
            x,
            self.weight,
            self.approx,
            self.error,
            self.threshold,
            self.refined,
            self.counts,
            self.n,
            self.k,
            16,
            512,
            num_warps=4,
        )
        topk_select(
            self.refined, 1, tie="low", end=self.row_lens, output_idx=answer.view(-1, 1)
        )
        return answer
