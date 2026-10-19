# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The FP4 MQA scorer's ragged rows and their launch. DeepSeek-V4, V4.1 and
V3.2's decode (its DCP shard included) score through it."""

from typing import NamedTuple

import torch
from aiter.ops.flydsl import make_fp4_mqa_plan


class Fp4MqaRaggedMetadata(NamedTuple):
    """Query rows grouped into the scorer's sequences: sequence b's rows are
    `query_start_loc[b] .. query_start_loc[b + 1] - 1`, at most `max_qlen`,
    and share each key load off its `block_tables` row, an entry naming
    `pages_per_block` consecutive pages."""

    query_start_loc: torch.Tensor
    max_qlen: int
    block_tables: torch.Tensor
    pages_per_block: int = 1

    def band(self, start, count, rows):
        """These sequences over rows [start, start + count) of `rows`: each
        sequence's rows inside, none for one outside."""
        if count == rows:
            return self
        return self._replace(
            query_start_loc=self.query_start_loc.clamp(start, start + count) - start
        )

    def kernel_args(self, row_ends, *, heads, page_size, max_seq_len):
        """`flydsl_pa_mqa_logits_fp4`'s keyword arguments for these rows, each
        bounded by `row_ends` [rows], scored by `heads` query heads against
        `page_size`-row pages into a `max_seq_len`-wide logits plane. The plan
        (`make_fp4_mqa_plan`) depends on shapes alone, so a captured graph
        keeps its launch."""
        plan = make_fp4_mqa_plan(
            num_seqs=self.query_start_loc.shape[0] - 1,
            max_qlen=self.max_qlen,
            num_rows=row_ends.shape[0],
            heads=heads,
            page_size=page_size,
            max_seq_len=max_seq_len,
            pages_per_block=self.pages_per_block,
        )
        return {
            "block_tables": self.block_tables,
            "context_lens": None,
            "row_ends": row_ends,
            "query_start_loc": self.query_start_loc,
            "plan": plan,
        }
