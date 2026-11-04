# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The draft heads (``MonoDraftModel.compute_draft_ids_from_head``): one row on ``HeadScreenGlobalCertificate``, two
or four rows on ``BatchHeadScreenW4`` (the same packed weights and certificates, one pass for every row). Each returns
the original BF16 head's argmax ids (``kernels.head``)."""

import json
import logging

import torch
import triton
from aiter import topk_select

from atom.models.qwen38.mono.config import HEAD_ROWS
from atom.models.qwen38.mono.kernels import head as K

logger = logging.getLogger("atom")


class HeadScreenGlobalCertificate:
    def __init__(self, weight):
        self.weight = weight
        self.n, self.k = weight.shape
        assert (self.n, self.k) == (248320, 2560)
        self.packed, self.ws, we, wqnorm, self.wnorm = K.prepare_weight(weight)
        self.we_global = K.aggregate_weight_certificate(we)
        self.wq_global = K.aggregate_weight_certificate(wqnorm)
        device = self.weight.device

        def alloc(*shape, dtype=torch.float32):
            return torch.empty(shape, device=device, dtype=dtype)

        groups = self.k // K.GROUP
        blocks = triton.cdiv(self.n, K.BLOCK_N)
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
        K.quant_groups[(self.k // K.GROUP,)](
            x,
            self.qeven,
            self.qodd,
            self.xscale,
            self.xstats,
            self.xsum,
            self.k,
            K.GROUP,
            num_warps=4,
        )
        self.last_stats = K.aggregate_input_certificate[(1,)](
            x,
            self.xscale,
            self.xstats,
            self.xglobal,
            self.k,
            self.k // K.GROUP,
            triton.next_power_of_2(self.k),
            triton.next_power_of_2(self.k // K.GROUP),
            num_warps=4,
        )
        blocks = triton.cdiv(self.n, K.BLOCK_N)
        self.last_screen = K.screen_global_certificate[(blocks,)](
            self.qeven, self.qodd, self.xscale, self.xglobal, self.xsum, self.packed, self.ws, self.we_global,
            self.wq_global, self.wnorm, self.approx, self.error, self.lower, self.best_y, self.best_id,
            self.n, self.k, K.GROUP, K.BLOCK_N, K.BLOCK_K, num_warps=4,
        )  # fmt: skip

    def run(self, x, *, out=None):
        answer = self.ids if out is None else out
        assert answer.shape == (1,) and answer.dtype == torch.int32
        assert answer.is_contiguous() and answer.device == x.device
        self.prepare(x)
        blocks = triton.cdiv(self.n, K.BLOCK_N)
        self.last_anchor = K.anchor_lower[(1,)](
            x, self.weight, self.wnorm, self.lower, self.best_y, self.best_id, self.threshold, self.anchor,
            self.anchor_value, self.anchor_bound, self.n, self.k, K.GROUP, blocks, triton.next_power_of_2(blocks),
            triton.next_power_of_2(self.k), num_warps=8,
        )  # fmt: skip
        self.last_refine = K.refine[(triton.cdiv(self.n, 16), 1)](
            x, self.weight, self.approx, self.error, self.threshold, self.refined, self.counts, self.n, self.k, 16, 512,
            num_warps=4,
        )  # fmt: skip
        topk_select(
            self.refined, 1, tie="low", end=self.row_lens, output_idx=answer.view(-1, 1)
        )
        return answer


class BatchHeadScreenW4:
    """``rows`` draft rows on the screen of ``selected`` (a ``HeadScreenGlobalCertificate``: its packed weights, scales,
    norms and global weight certificates are shared, nothing is copied): every packed weight tile is loaded once and
    dotted with each row's INT8 groups."""

    def __init__(self, selected, rows):
        assert rows in (2, 4), "a power-of-two row axis (the screen's tl.arange)"
        self.rows = rows
        for name in (
            "weight",
            "n",
            "k",
            "packed",
            "ws",
            "wnorm",
            "we_global",
            "wq_global",
        ):
            setattr(self, name, getattr(selected, name))
        dev = self.weight.device
        groups, blocks = self.k // K.GROUP, triton.cdiv(self.n, K.BLOCK_N)

        def alloc(*shape, dtype=torch.float32):
            return torch.empty(shape, device=dev, dtype=dtype)

        self.qeven = alloc(rows, self.k // 8, dtype=torch.uint32)
        self.qodd = alloc(rows, self.k // 8, dtype=torch.uint32)
        self.xscale, self.xstats = alloc(rows, groups), alloc(rows, groups, 3)
        self.xglobal = alloc(rows, 3)
        self.xsum = alloc(rows, groups, dtype=torch.int32)
        self.approx = alloc(rows, self.n, dtype=torch.bfloat16)
        self.error, self.refined = alloc(rows, self.n), alloc(rows, self.n)
        self.lower, self.best_y = alloc(rows, blocks), alloc(rows, blocks)
        self.best_id = alloc(rows, blocks, dtype=torch.int32)
        self.threshold, self.anchor_value, self.anchor_bound = (
            alloc(rows),
            alloc(rows),
            alloc(rows),
        )
        self.anchor = alloc(rows, dtype=torch.int32)
        self.counts = alloc(rows, triton.cdiv(self.n, 16), dtype=torch.int32)
        self.row_lens = torch.full((rows,), self.n, device=dev, dtype=torch.int32)

    def run(self, x, *, out):
        r = self.rows
        assert (
            x.shape == (r, self.k) and x.is_contiguous() and x.dtype == torch.bfloat16
        )
        assert out.shape == (r,) and out.dtype == torch.int32 and out.is_contiguous()
        g = K.GROUP
        groups, blocks = self.k // g, triton.cdiv(self.n, K.BLOCK_N)
        K.quant_groups_rows[(groups, r)](x, self.qeven, self.qodd, self.xscale, self.xstats, self.xsum, self.k, g,
                                         num_warps=4)  # fmt: skip
        K.aggregate_rows[(r,)](x, self.xscale, self.xstats, self.xglobal, self.k, groups,
                               triton.next_power_of_2(self.k), triton.next_power_of_2(groups), num_warps=4)  # fmt: skip
        K.screen_rows[(blocks,)](
            self.qeven, self.qodd, self.xscale, self.xglobal, self.xsum, self.packed, self.ws, self.we_global,
            self.wq_global, self.wnorm, self.approx, self.error, self.lower, self.best_y, self.best_id,
            self.n, self.k, g, K.BLOCK_N, K.BLOCK_K, r, num_warps=4,
        )  # fmt: skip
        K.anchor_rows[(r,)](
            x, self.weight, self.wnorm, self.lower, self.best_y, self.best_id, self.threshold, self.anchor,
            self.anchor_value, self.anchor_bound, self.n, self.k, blocks, triton.next_power_of_2(blocks),
            triton.next_power_of_2(self.k), num_warps=8,
        )  # fmt: skip
        K.refine[(triton.cdiv(self.n, 16), r)](
            x, self.weight, self.approx, self.error, self.threshold, self.refined, self.counts, self.n, self.k, 16,
            512, num_warps=4,
        )  # fmt: skip
        topk_select(
            self.refined, 1, tie="low", end=self.row_lens, output_idx=out.view(-1, 1)
        )
        return out


def prepare(weight):
    """The draft heads of the BF16 head ``weight``, bound once after the draft weights load and warmed (compiled)
    before any graph capture: (the one-row head, {rows: the batched head})."""
    assert weight.shape == (248320, 2560) and weight.dtype == torch.bfloat16
    assert weight.is_contiguous() and not torch.cuda.is_current_stream_capturing()
    one = HeadScreenGlobalCertificate(weight)
    assert bool(
        (one.we_global <= 2.0**40).all()
        & (one.wq_global <= 2.0**40).all()
        & (one.wnorm <= 2.0**40).all()
    )
    example = (
        torch.linspace(-1.0, 1.0, 2560, device=weight.device)
        .to(torch.bfloat16)
        .view(1, -1)
    )
    out = torch.empty((1,), dtype=torch.int32, device=weight.device)
    assert one.run(example, out=out).data_ptr() == out.data_ptr()
    torch.cuda.synchronize()
    resources = {}
    for name, binary in [
        ("input_certificate", one.last_stats),
        ("screen", one.last_screen),
        ("anchor", one.last_anchor),
        ("refine", one.last_refine),
    ]:
        assert binary.n_spills == 0 and binary.metadata.shared <= 65536
        resources[name] = {
            "vgpr": binary.n_regs,
            "spills": binary.n_spills,
            "lds": binary.metadata.shared,
        }
    batch = {}
    for rows in HEAD_ROWS:
        head = BatchHeadScreenW4(one, rows)
        x = torch.linspace(-1.0, 1.0, rows * 2560, device=weight.device)
        out = torch.empty((rows,), dtype=torch.int32, device=weight.device)
        head.run(x.to(torch.bfloat16).view(rows, 2560), out=out)
        batch[rows] = head
    torch.cuda.synchronize()
    logger.info("Qwen mono draft heads bound %s", json.dumps({"rows": [1, *HEAD_ROWS], "group": K.GROUP,
                                                               "one_row_resources": resources}))  # fmt: skip
    return one, batch
