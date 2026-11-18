# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU contracts for MiniMax-M3 cached-prefill sparse-PA planning."""

import torch

from atom.model_ops import base_attention
from atom.model_ops.minimax_m3 import sparse_attn


def test_cached_prefill_lets_device_lengths_choose_plan_capacity(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        base_attention, "run_pa_decode", lambda **kwargs: captured.update(kwargs)
    )
    monkeypatch.setattr(
        "aiter.ops.triton.gluon.pa_decode_gluon.get_recommended_splits",
        lambda *_: 1,
    )
    monkeypatch.setattr(sparse_attn, "_sparse_pa_ran_on_asm", lambda *_: False)
    monkeypatch.setattr(sparse_attn, "_default_plan_task_budget", lambda *_: 512)
    rows = 128
    sparse_attn.sparse_pa_per_row(
        torch.randn(rows, 16, 128, dtype=torch.bfloat16),
        torch.empty((16, 1, 8, 16, 16), dtype=torch.bfloat16),
        torch.empty((16, 1, 1, 128, 16), dtype=torch.bfloat16),
        torch.zeros((rows, 32), dtype=torch.int32),
        torch.full((rows,), 512, dtype=torch.int32),
        1,
        128**-0.5,
        torch.empty((rows, 16, 128), dtype=torch.bfloat16),
        None,
        None,
        dynamic_partition_plan=True,
    )

    assert captured["max_context_partition_num"] == 1
    assert captured["plan_partition_cap"] is None


def test_prefill_uses_dynamic_plan_only_after_every_prefix_fills_window(monkeypatch):
    dynamic = []
    monkeypatch.setattr(
        sparse_attn,
        "sparse_pa_per_row",
        lambda *_args, **kwargs: dynamic.append(kwargs["dynamic_partition_plan"]),
    )
    rows, pages = 4, 8
    common = dict(
        q=torch.empty((rows, 16, 128), dtype=torch.bfloat16),
        k_cache=torch.empty((1, 1, 8, 16, 16), dtype=torch.bfloat16),
        v_cache=torch.empty((1, 1, 1, 128, 16), dtype=torch.bfloat16),
        topk_idx=torch.empty((1, rows, 1), dtype=torch.int32),
        block_table=torch.empty((1, pages), dtype=torch.int32),
        query_req_id=None,
        query_abs_pos=None,
        qo_indptr=torch.arange(rows + 1, dtype=torch.int32),
        num_kv_heads=1,
        sm_scale=128**-0.5,
        output=torch.empty((rows, 16, 128), dtype=torch.bfloat16),
        sparse_bt=torch.empty((rows, pages), dtype=torch.int32),
        sparse_ctx=torch.full((rows,), pages * 16, dtype=torch.int32),
    )

    sparse_attn.minimax_m3_sparse_attn_prefill_asm(
        **common, min_prefix_len=pages * 16 - 1
    )
    sparse_attn.minimax_m3_sparse_attn_prefill_asm(
        **common, min_prefix_len=pages * 16
    )

    assert dynamic == [False, True]
