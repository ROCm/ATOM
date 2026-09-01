# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch

from atom.model_ops.attention_mla import _narrow_prefill_cu_seqlens


def test_plain_dense_prefill_keeps_full_query_cumulative_lengths() -> None:
    cu_seqlens_q = torch.tensor([0, 4, 9, 9], dtype=torch.int32)

    narrowed = _narrow_prefill_cu_seqlens(cu_seqlens_q, None)

    assert narrowed is cu_seqlens_q


def test_paged_prefill_drops_query_cumulative_length_padding() -> None:
    cu_seqlens_q = torch.tensor([0, 4, 9, 9], dtype=torch.int32)
    kv_last_page_lens = torch.tensor([4, 5], dtype=torch.int32)

    narrowed = _narrow_prefill_cu_seqlens(
        cu_seqlens_q,
        kv_last_page_lens,
    )

    assert torch.equal(narrowed, torch.tensor([0, 4, 9], dtype=torch.int32))
