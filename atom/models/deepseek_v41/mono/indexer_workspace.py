# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Index buffers shared by mutually exclusive layer and native decode paths.

Allocate at model installation, before warmup and KV sizing, so both PyTorch
and uncached allocations are included in the measured non-KV footprint.
The layer runner uses the first 48 rows; the native hook can use all 64.
"""

import torch

from atom.mono.dsa_indexer.paged import build
from atom.mono.dsa_indexer.paged import plan as ip
from atom.mono.dsa_indexer.paged import tp_plan as tp
from atom.mono.runtime.uncached import uncached_bytes


class IndexerWorkspace:
    def __init__(self, model, max_model_len, tp_size):
        rows = build.MAX_ROWS
        width = ip.cdiv(max_model_len, ip.CHUNK) * ip.CHUNK
        world = tp.kernel_world(tp_size, rows, width)
        key = build.IndexerBuild(rows, width, world=world)
        selecting = [b.attn.indexer for b in model.layers if b.attn.indexer is not None]
        dev = torch.device("cuda", torch.cuda.current_device())
        self.scratch = uncached_bytes(build.scratch_bytes(key), dev)
        self.local_scratch = (
            uncached_bytes(build.scratch_bytes(build.reindex_build(key)), dev)
            if any(i.spec.candidate_source is not None for i in selecting)
            else None
        )
        ratio = min(
            (
                i.spec.ratio
                for i in selecting
                if i.spec.candidate_source is None and not i.spec.produces_candidates
            ),
            default=None,
        )
        self.full_scratch = (
            uncached_bytes(build.scratch_bytes(build.full_build(key, ratio)), dev)
            if ratio is not None
            else None
        )
        self.ilog = torch.empty(rows, width, device=dev, dtype=torch.float32)
        self.ibmax = torch.empty(
            rows, ip.blocks(width), device=dev, dtype=torch.float32
        )
        self.selected = {
            i.spec.layer_id: torch.empty(rows, ip.TOPK, device=dev, dtype=torch.int32)
            for i in selecting
        }
        self.iq = uncached_bytes(rows * ip.HEADS * ip.DIM // 2, dev).view(
            rows, ip.HEADS, ip.DIM // 2
        )
        self.iqs = uncached_bytes(rows * ip.QS_BYTES, dev).view(rows, ip.QS_BYTES)
        self.iw = (
            uncached_bytes(rows * ip.HEADS * 4, dev)
            .view(torch.float32)
            .view(rows, ip.HEADS)
        )
